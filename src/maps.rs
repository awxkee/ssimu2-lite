/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! SIMD map evaluation and f64 pooling, shared by full and incremental scores.
use crate::metric::{edge_ratio, ssim_value};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(crate) enum MapKind {
    Ssim,
    Artifact,
    Detail,
}

// SSIM: mu1, mu2, s11, s22, s12. Edges: src, rec, mu1, mu2, unused.
pub(crate) type MapInputs<'a> = [&'a [f32]; 5];
pub(crate) const SUM_LANES: usize = 8;

/// Keep the same eight independent f64 lanes on every backend, including tails.
/// Fourth powers must be formed in f64: finite HDR f32 maps can overflow in f32.
#[derive(Default)]
pub(crate) struct MapSums {
    pub(crate) first: [f64; SUM_LANES],
    pub(crate) fourth: [f64; SUM_LANES],
}

impl MapSums {
    fn add<const REPLACE: bool, const FOURTH: bool>(&mut self, i: usize, new: f32, old: f32) {
        let lane = i % SUM_LANES;
        let v = new as f64;
        let old = old as f64;
        self.first[lane] += if REPLACE { v - old } else { v };
        if FOURTH {
            let square = v * v;
            let mut fourth = square * square;
            if REPLACE {
                let square = old * old;
                fourth -= square * square;
            }
            self.fourth[lane] += fourth;
        }
    }

    fn total(self) -> (f64, f64) {
        (self.first.iter().sum(), self.fourth.iter().sum())
    }
}

type MapFn = unsafe fn(MapInputs<'_>, &mut [f32], &mut MapSums) -> usize;
type EdgesFn = unsafe fn(MapInputs<'_>, [&mut [f32]; 2], &mut [MapSums; 2]) -> usize;
type SumsFn = unsafe fn(&[f32], &mut MapSums) -> usize;

#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    compute: [MapFn; 3],
    replace: [[MapFn; 2]; 3],
    sums: SumsFn,
    edges: EdgesFn,
    replace_edges: [EdgesFn; 4],
}

macro_rules! kernels {
    ($module:ident, $map:ident, $sums:ident, $edges:ident) => {
        Kernels {
            compute: [
                crate::$module::$map::<0, false, false>,
                crate::$module::$map::<1, false, false>,
                crate::$module::$map::<2, false, false>,
            ],
            replace: [
                [
                    crate::$module::$map::<0, true, false>,
                    crate::$module::$map::<0, true, true>,
                ],
                [
                    crate::$module::$map::<1, true, false>,
                    crate::$module::$map::<1, true, true>,
                ],
                [
                    crate::$module::$map::<2, true, false>,
                    crate::$module::$map::<2, true, true>,
                ],
            ],
            sums: crate::$module::$sums,
            edges: crate::$module::$edges::<false, false, false>,
            replace_edges: [
                crate::$module::$edges::<true, false, false>,
                crate::$module::$edges::<true, true, false>,
                crate::$module::$edges::<true, false, true>,
                crate::$module::$edges::<true, true, true>,
            ],
        }
    };
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
const SCALAR: Kernels = kernels!(maps, map_scalar, sums_scalar, edges_scalar);

pub(crate) fn selected_kernels() -> &'static Kernels {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    KERNELS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            return kernels!(avx, ss2_map_avx2, ss2_sums_avx2, ss2_edges_avx2);
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::is_x86_feature_detected!("sse4.1") {
            return kernels!(sse, ss2_map_sse41, ss2_sums_sse41, ss2_edges_sse41);
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            kernels!(neon, ss2_map_neon, ss2_sums_neon, ss2_edges_neon)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        SCALAR
    })
}

#[inline(always)]
fn value(kind: usize, [a, b, c, d, e]: [f32; 5]) -> f32 {
    match kind {
        0 => ssim_value(a, b, c, d, e),
        1 => edge_ratio(a, b, c, d).max(0.0),
        2 => (-edge_ratio(a, b, c, d)).max(0.0),
        _ => unreachable!(),
    }
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
fn map_scalar<const KIND: usize, const REPLACE: bool, const FOURTH: bool>(
    inputs: MapInputs<'_>,
    map: &mut [f32],
    sums: &mut MapSums,
) -> usize {
    map_tail::<REPLACE, FOURTH>(KIND, inputs, map, sums, 0);
    map.len()
}

fn map_tail<const REPLACE: bool, const FOURTH: bool>(
    kind: usize,
    inputs: MapInputs<'_>,
    map: &mut [f32],
    sums: &mut MapSums,
    start: usize,
) {
    match kind {
        0 => map_tail_kind::<0, REPLACE, FOURTH>(inputs, map, sums, start),
        1 => map_tail_kind::<1, REPLACE, FOURTH>(inputs, map, sums, start),
        2 => map_tail_kind::<2, REPLACE, FOURTH>(inputs, map, sums, start),
        _ => unreachable!(),
    }
}

fn map_tail_kind<const KIND: usize, const REPLACE: bool, const FOURTH: bool>(
    inputs: MapInputs<'_>,
    map: &mut [f32],
    sums: &mut MapSums,
    start: usize,
) {
    for i in start..map.len() {
        let new = value(KIND, inputs.map(|p| p[i]));
        if REPLACE {
            sums.add::<true, FOURTH>(i, new, map[i]);
        }
        map[i] = new;
    }
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
fn sums_scalar(map: &[f32], sums: &mut MapSums) -> usize {
    for (i, &v) in map.iter().enumerate() {
        sums.add::<false, true>(i, v, 0.0);
    }
    map.len()
}

impl Kernels {
    pub(crate) fn map_sums(&self, map: &[f32]) -> (f64, f64) {
        let mut sums = MapSums::default();
        // SAFETY: dispatch checked CPU features; kernels read complete vectors.
        let done = unsafe { (self.sums)(map, &mut sums) };
        for (i, &v) in map.iter().enumerate().skip(done) {
            sums.add::<false, true>(i, v, 0.0);
        }
        sums.total()
    }

    pub(crate) fn compute_map(&self, kind: MapKind, inputs: MapInputs<'_>, map: &mut [f32]) {
        assert!(inputs.iter().all(|p| p.len() == map.len()));
        let mut unused = MapSums::default();
        // SAFETY: dispatch checked CPU features and every input matches the output.
        let done = unsafe { (self.compute[kind as usize])(inputs, map, &mut unused) };
        map_tail::<false, false>(kind as usize, inputs, map, &mut unused, done);
    }

    pub(crate) fn replace_map(
        &self,
        kind: MapKind,
        inputs: MapInputs<'_>,
        map: &mut [f32],
        sums: &mut (f64, f64),
        fourth: bool,
    ) {
        assert!(inputs.iter().all(|p| p.len() == map.len()));
        let mut delta = MapSums::default();
        // SAFETY: dispatch checked CPU features and every input matches the output.
        let done =
            unsafe { self.replace[kind as usize][usize::from(fourth)](inputs, map, &mut delta) };
        if fourth {
            map_tail::<true, true>(kind as usize, inputs, map, &mut delta, done);
        } else {
            map_tail::<true, false>(kind as usize, inputs, map, &mut delta, done);
        }
        let delta = delta.total();
        sums.0 += delta.0;
        if fourth {
            sums.1 += delta.1;
        }
    }
}

fn edges_tail<const REPLACE: bool, const ARTIFACT_FOURTH: bool, const DETAIL_FOURTH: bool>(
    inputs: MapInputs<'_>,
    [artifact, detail]: [&mut [f32]; 2],
    sums: &mut [MapSums; 2],
    start: usize,
) {
    for i in start..artifact.len() {
        let ratio = edge_ratio(inputs[0][i], inputs[1][i], inputs[2][i], inputs[3][i]);
        let a = ratio.max(0.0);
        let d = (-ratio).max(0.0);
        if REPLACE {
            sums[0].add::<true, ARTIFACT_FOURTH>(i, a, artifact[i]);
            sums[1].add::<true, DETAIL_FOURTH>(i, d, detail[i]);
        }
        artifact[i] = a;
        detail[i] = d;
    }
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
fn edges_scalar<const REPLACE: bool, const ARTIFACT_FOURTH: bool, const DETAIL_FOURTH: bool>(
    inputs: MapInputs<'_>,
    maps: [&mut [f32]; 2],
    sums: &mut [MapSums; 2],
) -> usize {
    let n = maps[0].len();
    edges_tail::<REPLACE, ARTIFACT_FOURTH, DETAIL_FOURTH>(inputs, maps, sums, 0);
    n
}

impl Kernels {
    pub(crate) fn compute_edge_maps(
        &self,
        inputs: MapInputs<'_>,
        [artifact, detail]: [&mut [f32]; 2],
    ) {
        let n = artifact.len();
        assert_eq!(detail.len(), n);
        assert!(inputs.iter().all(|p| p.len() == n));
        let mut unused = std::array::from_fn(|_| MapSums::default());
        // SAFETY: dispatch checked features; every input and output has n elements.
        let done = unsafe { (self.edges)(inputs, [artifact, detail], &mut unused) };
        edges_tail::<false, false, false>(inputs, [artifact, detail], &mut unused, done);
    }

    pub(crate) fn replace_edge_maps(
        &self,
        inputs: MapInputs<'_>,
        [artifact, detail]: [&mut [f32]; 2],
        sums: [&mut (f64, f64); 2],
        fourth: [bool; 2],
    ) {
        let n = artifact.len();
        assert_eq!(detail.len(), n);
        assert!(inputs.iter().all(|p| p.len() == n));
        let which = usize::from(fourth[0]) + 2 * usize::from(fourth[1]);
        let mut delta = std::array::from_fn(|_| MapSums::default());
        // SAFETY: dispatch checked features; every input and output has n elements.
        let done = unsafe { self.replace_edges[which](inputs, [artifact, detail], &mut delta) };
        match fourth {
            [false, false] => {
                edges_tail::<true, false, false>(inputs, [artifact, detail], &mut delta, done)
            }
            [true, false] => {
                edges_tail::<true, true, false>(inputs, [artifact, detail], &mut delta, done)
            }
            [false, true] => {
                edges_tail::<true, false, true>(inputs, [artifact, detail], &mut delta, done)
            }
            [true, true] => {
                edges_tail::<true, true, true>(inputs, [artifact, detail], &mut delta, done)
            }
        }
        for ((sum, delta), fourth) in sums.into_iter().zip(delta).zip(fourth) {
            let delta = delta.total();
            sum.0 += delta.0;
            if fourth {
                sum.1 += delta.1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available_kernels() -> Vec<(&'static str, Kernels)> {
        #[allow(unused_mut)]
        let mut all = vec![("scalar", SCALAR), ("dispatch", *selected_kernels())];
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            all.push((
                "neon",
                kernels!(neon, ss2_map_neon, ss2_sums_neon, ss2_edges_neon),
            ));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            all.push((
                "avx2",
                kernels!(avx, ss2_map_avx2, ss2_sums_avx2, ss2_edges_avx2),
            ));
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::is_x86_feature_detected!("sse4.1") {
            all.push((
                "sse4.1",
                kernels!(sse, ss2_map_sse41, ss2_sums_sse41, ss2_edges_sse41),
            ));
        }
        all
    }

    #[test]
    fn map_kernels_and_replacements_match_scalar_bits_and_guard_pixels() {
        const GUARD: f32 = -12345.0;
        let values = [-0.0, 0.0, 1e-23, -1e-20, -0.7, 0.125, 1.0, 1e15, 1e37];
        for (name, kernels) in available_kernels() {
            for n in (0..=65).chain([127, 129, 257]) {
                for offset in 0..4 {
                    let input: [Vec<f32>; 5] = std::array::from_fn(|c| {
                        (0..n + 8)
                            .map(|i| values[(i * (c + 1) + c) % values.len()])
                            .collect()
                    });
                    let before = input.clone();
                    let inputs = input.each_ref().map(|p| &p[offset..offset + n]);
                    for kind in [MapKind::Ssim, MapKind::Artifact, MapKind::Detail] {
                        let expected: Vec<_> = (0..n)
                            .map(|i| value(kind as usize, inputs.map(|p| p[i])))
                            .collect();
                        let mut actual = vec![GUARD; n + 8];
                        let map = &mut actual[offset..offset + n];
                        let mut unused = MapSums::default();
                        // SAFETY: available_kernels checked features; all slices have n pixels.
                        let done =
                            unsafe { (kernels.compute[kind as usize])(inputs, map, &mut unused) };
                        assert!(done <= n);
                        map_tail::<false, false>(kind as usize, inputs, map, &mut unused, done);
                        for (i, (&a, &b)) in map.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "{name}, {kind:?}, n={n}, offset={offset}, i={i}"
                            );
                        }
                        assert!(actual[..offset].iter().all(|&v| v == GUARD));
                        assert!(actual[offset + n..].iter().all(|&v| v == GUARD));
                        for fourth in [false, true] {
                            let old: Vec<_> =
                                (0..n).map(|i| values[i % values.len()].abs()).collect();
                            actual[offset..offset + n].copy_from_slice(&old);
                            let mut expected_delta = MapSums::default();
                            for i in 0..n {
                                if fourth {
                                    expected_delta.add::<true, true>(i, expected[i], old[i]);
                                } else {
                                    expected_delta.add::<true, false>(i, expected[i], old[i]);
                                }
                            }
                            let mut delta = MapSums::default();
                            let map = &mut actual[offset..offset + n];
                            // SAFETY: same detected features and matching slices as above.
                            let done = unsafe {
                                (kernels.replace[kind as usize][usize::from(fourth)])(
                                    inputs, map, &mut delta,
                                )
                            };
                            assert!(done <= n);
                            if fourth {
                                map_tail::<true, true>(
                                    kind as usize,
                                    inputs,
                                    map,
                                    &mut delta,
                                    done,
                                );
                            } else {
                                map_tail::<true, false>(
                                    kind as usize,
                                    inputs,
                                    map,
                                    &mut delta,
                                    done,
                                );
                            }
                            assert_eq!(map, expected, "{name}, {kind:?}, fourth={fourth}, n={n}");
                            assert_eq!(
                                delta.first, expected_delta.first,
                                "first: {name}, {kind:?}, n={n}"
                            );
                            assert_eq!(
                                delta.fourth, expected_delta.fourth,
                                "fourth: {name}, {kind:?}, n={n}"
                            );
                            assert!(actual[..offset].iter().all(|&v| v == GUARD));
                            assert!(actual[offset + n..].iter().all(|&v| v == GUARD));
                        }
                    }
                    assert_eq!(input, before);
                }
            }
        }
    }

    #[test]
    fn fused_edges_match_separate_maps_and_each_norm_combination() {
        const GUARD: f32 = -12345.0;
        for (name, kernels) in available_kernels() {
            for n in (0..=65).chain([127, 129, 257]) {
                let input: [Vec<f32>; 5] = std::array::from_fn(|c| {
                    (0..n)
                        .map(|i| ((i * (c + 3) + c * 17) % 101) as f32 / 50.0 - 1.0)
                        .collect()
                });
                let inputs = input.each_ref().map(|p| &p[..]);
                let mut expected: [Vec<f32>; 2] = std::array::from_fn(|_| vec![0.0; n]);
                let [a, d] = &mut expected;
                kernels.compute_map(MapKind::Artifact, inputs, a);
                kernels.compute_map(MapKind::Detail, inputs, d);
                let mut output = [vec![GUARD; n + 2], vec![GUARD; n + 2]];
                let [a, d] = &mut output;
                let mut unused = std::array::from_fn(|_| MapSums::default());
                // SAFETY: features are available and all active slices hold n pixels.
                let done = unsafe {
                    (kernels.edges)(inputs, [&mut a[1..n + 1], &mut d[1..n + 1]], &mut unused)
                };
                assert!(done <= n);
                edges_tail::<false, false, false>(
                    inputs,
                    [&mut a[1..n + 1], &mut d[1..n + 1]],
                    &mut unused,
                    done,
                );
                assert_eq!(a[1..n + 1], expected[0], "artifact: {name}, n={n}");
                assert_eq!(d[1..n + 1], expected[1], "detail: {name}, n={n}");
                for which in 0..4 {
                    let fourth = [which & 1 != 0, which & 2 != 0];
                    let mut old: [Vec<f32>; 2] = std::array::from_fn(|c| {
                        (0..n)
                            .map(|i| ((i * 7 + c * 11) % 17) as f32 / 16.0)
                            .collect()
                    });
                    let mut expected_delta: [MapSums; 2] =
                        std::array::from_fn(|_| MapSums::default());
                    for c in 0..2 {
                        for i in 0..n {
                            if fourth[c] {
                                expected_delta[c].add::<true, true>(i, expected[c][i], old[c][i]);
                            } else {
                                expected_delta[c].add::<true, false>(i, expected[c][i], old[c][i]);
                            }
                        }
                    }
                    let [a, d] = &mut old;
                    let mut delta = std::array::from_fn(|_| MapSums::default());
                    // SAFETY: same feature and length contracts as above.
                    let done =
                        unsafe { (kernels.replace_edges[which])(inputs, [a, d], &mut delta) };
                    assert!(done <= n);
                    match fourth {
                        [false, false] => {
                            edges_tail::<true, false, false>(inputs, [a, d], &mut delta, done)
                        }
                        [true, false] => {
                            edges_tail::<true, true, false>(inputs, [a, d], &mut delta, done)
                        }
                        [false, true] => {
                            edges_tail::<true, false, true>(inputs, [a, d], &mut delta, done)
                        }
                        [true, true] => {
                            edges_tail::<true, true, true>(inputs, [a, d], &mut delta, done)
                        }
                    }
                    assert_eq!(old, expected, "{name}, n={n}, fourth={fourth:?}");
                    for c in 0..2 {
                        assert_eq!(
                            delta[c].first, expected_delta[c].first,
                            "{name}, n={n}, c={c}"
                        );
                        assert_eq!(
                            delta[c].fourth, expected_delta[c].fourth,
                            "{name}, n={n}, c={c}"
                        );
                    }
                }
                for out in output {
                    assert_eq!(out[0], GUARD);
                    assert_eq!(out[n + 1], GUARD);
                }
            }
        }
    }

    #[test]
    fn f64_pooling_matches_scalar_lanes_for_tails_and_hdr() {
        for (name, kernels) in available_kernels() {
            for n in (0..=65).chain([127, 129, 4099, 307_200]) {
                for magnitude in [0.0, 1e-35, 1e-7, 1.0, 1e20, 1e37] {
                    let map: Vec<_> = (0..n)
                        .map(|i| magnitude * ((i % 17 + 1) as f32 / 17.0))
                        .collect();
                    let mut expected = MapSums::default();
                    sums_scalar(&map, &mut expected);
                    let mut actual = MapSums::default();
                    // SAFETY: available_kernels checked features; kernels read complete vectors.
                    let done = unsafe { (kernels.sums)(&map, &mut actual) };
                    assert!(done <= n);
                    for (i, &v) in map.iter().enumerate().skip(done) {
                        actual.add::<false, true>(i, v, 0.0);
                    }
                    assert_eq!(
                        actual.first, expected.first,
                        "first: {name}, n={n}, magnitude={magnitude}"
                    );
                    assert_eq!(
                        actual.fourth, expected.fourth,
                        "fourth: {name}, n={n}, magnitude={magnitude}"
                    );
                    assert_eq!(kernels.map_sums(&map), expected.total());
                }
            }
        }
    }
}
