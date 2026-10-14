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

//! The metric: six scales, X/Y/B planes in "positive XYB", the SSIM map
//! without the luminance denominator, artifact / detail-loss edge maps, 1- and
//! 4-norm pooling, the 108-weight sum and the final nonlinearity
//! (libjxl `tools/ssimulacra2.cc`). Scales are box-downsampled in linear RGB.
//! The reference's recursive Gaussian kernel is applied as an 11-tap FIR.
//!
//! The pooled sums are additive over pixels before the final nonlinearity, so
//! a per-pixel map of the linearized loss has the deficit `100 - score` as its mean.

use crate::blur::GaussianBlur;
use crate::downsample::downsample;
use crate::xyb::positive_xyb_of_linear;
use crate::{Error, RgbImage};
use std::sync::{Arc, OnceLock};

pub(crate) const C2: f32 = 0.0009;
pub(crate) const NUM_SCALES: usize = 6;
pub(crate) const MIN_SIZE: usize = 8;

// The weights and score mapping are from libjxl tools/ssimulacra2.cc,
// Copyright (c) the JPEG XL Project Authors, BSD-3-Clause.

/// Reference weights, indexed `weight_base(plane, scale, scales) + norm * 3 + map`
/// with map 0 = SSIM, 1 = artifact, 2 = detail loss.
#[rustfmt::skip]
pub(crate) static WEIGHTS: [f64; 108] = [
    0.0, 0.0007376606707406586, 0.0, 0.0, 0.0007793481682867309, 0.0, 0.0, 0.0004371155730107379, 0.0,
    1.1041726426657346, 0.00066284834129271, 0.00015231632783718752, 0.0, 0.0016406437456599754, 0.0,
    1.8422455520539298, 11.441172603757666, 0.0, 0.0007989109436015163, 0.000176816438078653, 0.0,
    1.8787594979546387, 10.94906990605142, 0.0, 0.0007289346991508072, 0.9677937080626833, 0.0,
    0.00014003424285435884, 0.9981766977854967, 0.00031949755934435053, 0.0004550992113792063, 0.0, 0.0,
    0.0013648766163243398, 0.0, 0.0, 0.0, 0.0, 0.0, 7.466890328078848, 0.0, 17.445833984131262,
    0.0006235601634041466, 0.0, 0.0, 6.683678146179332, 0.00037724407979611296, 1.027889937768264,
    225.20515300849274, 0.0, 0.0, 19.213238186143016, 0.0011401524586618361, 0.001237755635509985,
    176.39317598450694, 0.0, 0.0, 24.43300999870476, 0.28520802612117757, 0.0004485436923833408, 0.0, 0.0,
    0.0, 34.77906344483772, 44.835625328877896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0008680556573291698, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005313191874358747, 0.0, 0.00016533814161379112,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0004179171803251336, 0.0017290828234722833, 0.0, 0.0020827005846636437,
    0.0, 0.0, 8.826982764996862, 23.19243343998926, 0.0, 95.1080498811086, 0.9863978034400682,
    0.9834382792465353, 0.0012286405048278493, 171.2667255897307, 0.9807858872435379, 0.0, 0.0, 0.0,
    0.0005130064588990679, 0.0, 0.00010854057858411537,
];

/// Offset of a plane's and scale's six weights. Like the reference, which
/// walks the table with a running index, this packs the evaluated scales:
/// images with fewer than six scales use other planes' and scales' weights.
pub(crate) fn weight_base(plane: usize, scale: usize, scales: usize) -> usize {
    (plane * scales + scale) * 6
}

/// A map is needed if either of its pooling norms has a nonzero weight.
pub(crate) fn active_maps(plane: usize, scale: usize, scales: usize) -> [bool; 3] {
    let base = weight_base(plane, scale, scales);
    std::array::from_fn(|m| WEIGHTS[base + m] != 0.0 || WEIGHTS[base + 3 + m] != 0.0)
}

/// Score from the weighted pooled sum.
pub(crate) fn score_from_sum(s: f64) -> f64 {
    let s = s * 0.9562382616834844;
    let s =
        2.326765642916932 * s - 0.020884521182843837 * s * s + 6.248496625763138e-05 * s * s * s;
    if s > 0.0 {
        100.0 - 10.0 * s.powf(0.6276336467831387)
    } else {
        100.0
    }
}

/// d(100 - score)/d(sum) at `s`: how many score points one unit of the
/// weighted sum costs.
pub(crate) fn deficit_slope(s: f64) -> f64 {
    let u = s * 0.9562382616834844;
    let g =
        2.326765642916932 * u - 0.020884521182843837 * u * u + 6.248496625763138e-05 * u * u * u;
    if g <= 0.0 {
        return 0.0;
    }
    let dg = (2.326765642916932 - 2.0 * 0.020884521182843837 * u
        + 3.0 * 6.248496625763138e-05 * u * u)
        * 0.9562382616834844;
    10.0 * 0.6276336467831387 * g.powf(0.6276336467831387 - 1.0) * dg
}

/// Reference blur: libjxl's recursive Gaussian (Charalampidis 2016) is a
/// truncated sum of three cosines with radius `round(3.2795σ + 0.2546)` = 5 at
/// σ = 1.5 and zero padding outside the image; this is the same kernel applied
/// as a separable FIR.
pub(crate) const RADIUS: usize = crate::blur::RADIUS;
const TAPS: usize = crate::blur::WINDOW;

pub(crate) fn kernel() -> &'static [f32; TAPS] {
    static K: OnceLock<[f32; TAPS]> = OnceLock::new();
    K.get_or_init(|| {
        let sigma = 1.5f64;
        let radius = RADIUS as f64;
        let pi_div_2r = std::f64::consts::PI / (2.0 * radius);
        let omega = [pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r];
        let p = [
            1.0 / (0.5 * omega[0]).tan(),
            -1.0 / (0.5 * omega[1]).tan(),
            1.0 / (0.5 * omega[2]).tan(),
        ];
        let r = [
            p[0] * p[0] / omega[0].sin(),
            -p[1] * p[1] / omega[1].sin(),
            p[2] * p[2] / omega[2].sin(),
        ];
        let rho: Vec<f64> = omega
            .iter()
            .map(|w| (-0.5 * sigma * sigma * w * w).exp() / radius)
            .collect();
        let d13 = p[0] * r[1] - r[0] * p[1];
        let d35 = p[1] * r[2] - r[1] * p[2];
        let d51 = p[2] * r[0] - r[2] * p[0];
        let zeta15 = d35 / d13;
        let zeta35 = d51 / d13;
        // solve A·beta = gamma with A = [[p], [r], [zeta15, zeta35, 1]]
        let a = [
            [p[0], p[1], p[2]],
            [r[0], r[1], r[2]],
            [zeta15, zeta35, 1.0],
        ];
        let g = [
            1.0,
            radius * radius - sigma * sigma,
            zeta15 * rho[0] + zeta35 * rho[1] + rho[2],
        ];
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        let solve = |col: usize| {
            let mut m = a;
            for row in 0..3 {
                m[row][col] = g[row];
            }
            (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
                / det
        };
        let beta = [solve(0), solve(1), solve(2)];
        let mut k = [0f32; TAPS];
        for (i, slot) in k.iter_mut().enumerate() {
            let x = i as f64 - radius;
            *slot = (0..3).map(|j| beta[j] * (omega[j] * x).cos()).sum::<f64>() as f32;
        }
        k
    })
}

/// Pooled 1-norm and 4-norm of one map; with `acc`, also its per-pixel linearized
/// contribution to the weighted sum (`w1·x/N + w4·x⁴/(4·N·M4^(3/4))`).
fn pool(map: &[f32], w1: f64, w4: f64, acc: Option<&mut [f32]>) -> f64 {
    if w1 == 0.0 && w4 == 0.0 {
        return 0.0;
    }
    let n = map.len() as f64;
    let (s1, s4) = map_sums(map);
    let m1 = s1 / n;
    let m4 = s4 / n;
    let n4 = m4.sqrt().sqrt();
    if let Some(acc) = acc {
        add_map_contribution(map, w1, w4, n4, acc);
    }
    w1 * m1 + w4 * n4
}

pub(crate) fn map_sums(map: &[f32]) -> (f64, f64) {
    let mut s1 = [0.0f64; SUM_LANES];
    let mut s4 = [0.0f64; SUM_LANES];
    let (chunks, tail) = map.as_chunks::<SUM_LANES>();
    for chunk in chunks {
        for lane in 0..SUM_LANES {
            let v = chunk[lane] as f64;
            s1[lane] += v;
            let v_sqr = v * v;
            s4[lane] += v_sqr * v_sqr;
        }
    }
    for (lane, &v) in tail.iter().enumerate() {
        let v = v as f64;
        s1[lane] += v;
        let v_sqr = v * v;
        s4[lane] += v_sqr * v_sqr;
    }
    (s1.iter().sum(), s4.iter().sum())
}

/// Compute the pixel contribution in f32. Normalizing by the fourth norm
/// keeps the fourth powers bounded by the pixel count, even for HDR maps.
/// The pooled totals and final score stay f64, including incremental deltas.
pub(crate) fn add_map_contribution(map: &[f32], w1: f64, w4: f64, n4: f64, acc: &mut [f32]) {
    let n = map.len() as f64;
    let g1 = w1 / n;
    if w4 == 0.0 || n4 == 0.0 {
        contribute(
            map,
            MapContribution {
                linear: g1 as f32,
                inv_norm: 0.0,
                norm_weight: 0.0,
            },
            acc,
        );
        return;
    }
    let inv_n4 = (1.0 / n4) as f32;
    let norm_weight = (w4 * n4 / (4.0 * n)) as f32;
    if inv_n4.is_normal() && norm_weight.is_normal() {
        contribute(
            map,
            MapContribution {
                linear: g1 as f32,
                inv_norm: inv_n4,
                norm_weight,
            },
            acc,
        );
        return;
    }
    // Tiny or large norms may have unrepresentable f32 factors.
    let g4 = w4 / (4.0 * n * n4 * n4 * n4);
    for (a, &v) in acc.iter_mut().zip(map) {
        let v = v as f64;
        let v_sqr = v * v;
        *a += (g1 * v + g4 * v_sqr * v_sqr) as f32;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MapContribution {
    pub(crate) linear: f32,
    pub(crate) inv_norm: f32,
    pub(crate) norm_weight: f32,
}

type ContributionFn = unsafe fn(&[f32], MapContribution, &mut [f32]) -> usize;
type ExpandRowFn = unsafe fn(&[f32], &mut [f32], f32) -> usize;

struct MapKernels {
    contribution: ContributionFn,
    expand: [ExpandRowFn; NUM_SCALES],
}

fn map_kernels() -> &'static MapKernels {
    static KERNELS: OnceLock<MapKernels> = OnceLock::new();
    KERNELS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return MapKernels {
                contribution: crate::avx::ss2_contribution_avx2,
                expand: [
                    crate::avx::ss2_expand_row_avx2::<1>,
                    crate::avx::ss2_expand_row_avx2::<2>,
                    crate::avx::ss2_expand_row_avx2::<4>,
                    crate::avx::ss2_expand_row_avx2::<8>,
                    crate::avx::ss2_expand_row_avx2::<16>,
                    crate::avx::ss2_expand_row_avx2::<32>,
                ],
            };
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        return MapKernels {
            contribution: crate::neon::ss2_contribution_neon,
            expand: [
                crate::neon::ss2_expand_row_neon::<1>,
                crate::neon::ss2_expand_row_neon::<2>,
                crate::neon::ss2_expand_row_neon::<4>,
                crate::neon::ss2_expand_row_neon::<8>,
                crate::neon::ss2_expand_row_neon::<16>,
                crate::neon::ss2_expand_row_neon::<32>,
            ],
        };
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        MapKernels {
            contribution: contribution_scalar,
            expand: [
                expand_row_scalar::<1>,
                expand_row_scalar::<2>,
                expand_row_scalar::<4>,
                expand_row_scalar::<8>,
                expand_row_scalar::<16>,
                expand_row_scalar::<32>,
            ],
        }
    })
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
pub(crate) fn contribution_scalar(map: &[f32], p: MapContribution, acc: &mut [f32]) -> usize {
    if p.norm_weight == 0.0 {
        for (a, &v) in acc.iter_mut().zip(map) {
            *a += p.linear * v;
        }
    } else {
        for (a, &v) in acc.iter_mut().zip(map) {
            let normalized = v * p.inv_norm;
            let square = normalized * normalized;
            *a += p.linear * v + p.norm_weight * square * square;
        }
    }
    acc.len()
}

fn contribute(map: &[f32], p: MapContribution, acc: &mut [f32]) {
    assert_eq!(map.len(), acc.len());
    // SAFETY: dispatch checked the CPU features and both slices have equal
    // lengths. The kernel reads/writes complete vectors and returns its prefix.
    let done = unsafe { (map_kernels().contribution)(map, p, acc) };
    // SIMD kernels use FMA; keep their short tails fused as well. The portable
    // kernel processes the whole slice without requiring hardware FMA.
    for (a, &v) in acc[done..].iter_mut().zip(&map[done..]) {
        if p.norm_weight == 0.0 {
            *a = p.linear.mul_add(v, *a);
        } else {
            let normalized = v * p.inv_norm;
            let square = normalized * normalized;
            *a += (p.norm_weight * square).mul_add(square, p.linear * v);
        }
    }
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
fn expand_row_scalar<const STEP: usize>(src: &[f32], dst: &mut [f32], share: f32) -> usize {
    for (pixels, &v) in dst.chunks_mut(STEP).zip(src) {
        let contribution = v * share;
        for out in pixels {
            *out += contribution;
        }
    }
    dst.len()
}

/// Spread a scale's contribution in contiguous runs. Each source pixel covers
/// STEP×STEP pixels, with the last run cropped at odd image edges.
pub(crate) fn accumulate_scale_map(
    acc: &[f32],
    w0: usize,
    h0: usize,
    scale: usize,
    map: &mut [f32],
) {
    let step = 1usize << scale;
    let w = w0.div_ceil(step);
    let h = h0.div_ceil(step);
    assert_eq!(acc.len(), w * h);
    assert_eq!(map.len(), w0 * h0);
    if map.is_empty() {
        return;
    }
    let share = 1.0 / (step * step) as f32;
    let expand = map_kernels().expand[scale];
    for (src, rows) in acc.chunks_exact(w).zip(map.chunks_mut(w0 * step)) {
        for row in rows.chunks_exact_mut(w0) {
            // SAFETY: dispatch checked CPU features. `src` has ceil(row/STEP)
            // pixels and the specialized kernel handles only complete runs.
            let done = unsafe { expand(src, row, share) };
            for (pixels, &v) in row[done..].chunks_mut(step).zip(&src[done / step..]) {
                for out in pixels {
                    *out = v.mul_add(share, *out);
                }
            }
        }
    }
}

/// Independent lanes break the pooled sums' per-pixel dependency chain and
/// let LLVM vectorize the map calculation together with the f64 reductions.
const SUM_LANES: usize = 8;

pub(crate) fn replace_map<const INPUTS: usize>(
    map: &mut [f32],
    inputs: [&[f32]; INPUTS],
    sums: &mut (f64, f64),
    need_fourth: bool,
    value: impl Fn([f32; INPUTS]) -> f32,
) {
    // Select once per row, keeping the map math and independent f64 lanes
    // vectorizable without a norm-weight branch inside the pixel loop.
    if need_fourth {
        replace_map_with_norm::<INPUTS, true>(map, inputs, sums, value);
    } else {
        replace_map_with_norm::<INPUTS, false>(map, inputs, sums, value);
    }
}

fn replace_map_with_norm<const INPUTS: usize, const FOURTH: bool>(
    map: &mut [f32],
    inputs: [&[f32]; INPUTS],
    sums: &mut (f64, f64),
    value: impl Fn([f32; INPUTS]) -> f32,
) {
    let n = map.len();
    assert!(inputs.iter().all(|input| input.len() == n));
    let mut delta1 = [0.0f64; SUM_LANES];
    let mut delta4 = [0.0f64; SUM_LANES];
    let (chunks, tail) = map.as_chunks_mut::<SUM_LANES>();
    let input_chunks = inputs.map(|input| input.as_chunks::<SUM_LANES>().0);
    for (i, chunk) in chunks.iter_mut().enumerate() {
        let input = input_chunks.map(|chunks| &chunks[i]);
        for lane in 0..SUM_LANES {
            let new = value(input.map(|input| input[lane]));
            let old = chunk[lane] as f64;
            let v = new as f64;
            delta1[lane] += v - old;
            if FOURTH {
                let v_sqr = v * v;
                let old_sqr = old * old;
                delta4[lane] += v_sqr * v_sqr - old_sqr * old_sqr;
            }
            chunk[lane] = new;
        }
    }
    let offset = n - tail.len();
    for (i, out) in tail.iter_mut().enumerate() {
        let new = value(inputs.map(|input| input[offset + i]));
        let old = *out as f64;
        let v = new as f64;
        delta1[i] += v - old;
        if FOURTH {
            let v_sqr = v * v;
            let old_sqr = old * old;
            delta4[i] += v_sqr * v_sqr - old_sqr * old_sqr;
        }
        *out = new;
    }
    sums.0 += delta1.iter().sum::<f64>();
    if FOURTH {
        sums.1 += delta4.iter().sum::<f64>();
    }
}

/// Scale sizes: as in the reference, the next scale exists while this one is
/// at least 8×8, so the coarsest scale may be smaller.
pub(crate) fn scale_dims(mut w: usize, mut h: usize) -> Vec<(usize, usize)> {
    let mut dims = vec![(w, h)];
    while dims.len() < NUM_SCALES && w >= MIN_SIZE && h >= MIN_SIZE {
        (w, h) = (w.div_ceil(2), h.div_ceil(2));
        dims.push((w, h));
    }
    dims
}

/// The SSIM map value from the two means, variances and covariance.
#[inline(always)]
pub(crate) fn ssim_value(m1: f32, m2: f32, v11: f32, v22: f32, v12: f32) -> f32 {
    let num_m = 1.0 - (m1 - m2) * (m1 - m2);
    let num_s = 2.0 * (v12 - m1 * m2) + C2;
    let denom_s = (v11 - m1 * m1) + (v22 - m2 * m2) + C2;
    (1.0 - num_m * num_s / denom_s).max(0.0)
}

/// Signed edge ratio: positive is an artifact, negative is detail loss.
#[inline(always)]
pub(crate) fn edge_ratio(a: f32, b: f32, m1: f32, m2: f32) -> f32 {
    (1.0 + (b - m2).abs()) / (1.0 + (a - m1).abs()) - 1.0
}

/// One scale of an analysed reference.
pub(crate) struct ReferenceScale {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Positive XYB planes.
    pub(crate) xyb: [Vec<f32>; 3],
    /// Blurred planes; empty for planes without weighted maps.
    pub(crate) mu: [Vec<f32>; 3],
    /// Blurred squares; empty unless the plane's SSIM map is weighted.
    pub(crate) sigma: [Vec<f32>; 3],
}

pub(crate) struct ReferenceData {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) scales: Vec<ReferenceScale>,
}

/// A reference image analysed once, to be compared with many test images.
///
/// Holds the reference's XYB pyramid and its blurred moments, so a comparison
/// only processes the test image. Cloning is cheap (shared data).
#[derive(Clone)]
pub struct PreparedReference {
    pub(crate) data: Arc<ReferenceData>,
}

impl PreparedReference {
    /// Analyses `image`. Both dimensions must be at least 8 pixels.
    pub fn new(image: RgbImage<'_>) -> Result<Self, Error> {
        let (w0, h0) = (image.width(), image.height());
        if w0 < MIN_SIZE || h0 < MIN_SIZE {
            return Err(Error::ImageTooSmall);
        }
        let mut lin = image.to_linear()?;
        let dims = scale_dims(w0, h0);
        let mut blur = GaussianBlur::with_kernel(w0, *kernel(), true);
        let mut scales = Vec::with_capacity(dims.len());
        for (scale, &(w, h)) in dims.iter().enumerate() {
            if scale > 0 {
                let (pw, ph) = dims[scale - 1];
                for plane in &mut lin {
                    *plane = downsample(plane, pw, ph).0;
                }
            }
            let xyb = positive_xyb_of_linear(&lin);
            blur.set_width(w);
            let mut mu: [Vec<f32>; 3] = Default::default();
            let mut sigma: [Vec<f32>; 3] = Default::default();
            for c in 0..3 {
                let active = active_maps(c, scale, dims.len());
                if active.contains(&true) {
                    mu[c] = vec![0.0; w * h];
                    blur.blur(&xyb[c], h, &mut mu[c]);
                }
                if active[0] {
                    sigma[c] = vec![0.0; w * h];
                    blur.blur_product([&xyb[c], &xyb[c]], h, &mut sigma[c]);
                }
            }
            scales.push(ReferenceScale {
                width: w,
                height: h,
                xyb,
                mu,
                sigma,
            });
        }
        Ok(Self {
            data: Arc::new(ReferenceData {
                width: w0,
                height: h0,
                scales,
            }),
        })
    }

    /// Image width, in pixels.
    pub fn width(&self) -> usize {
        self.data.width
    }

    /// Image height, in pixels.
    pub fn height(&self) -> usize {
        self.data.height
    }

    /// SSIMULACRA 2 score of `test`: 100 is identical, lower is worse
    /// (about 90 is visually lossless, 70 high quality, 50 medium).
    pub fn compare(&self, test: RgbImage<'_>) -> Result<f64, Error> {
        self.evaluate(test, false).map(|(score, _)| score)
    }

    /// The score and a row-major per-pixel error map: each pixel's share of
    /// `100 - score`, linearized at the current score. The map's mean is
    /// approximately `100 - score`, so sums over regions are comparable.
    pub fn compare_with_map(&self, test: RgbImage<'_>) -> Result<(f64, Vec<f32>), Error> {
        self.evaluate(test, true)
    }

    fn evaluate(&self, test: RgbImage<'_>, want_map: bool) -> Result<(f64, Vec<f32>), Error> {
        let d = &*self.data;
        if (test.width(), test.height()) != (d.width, d.height) {
            return Err(Error::DimensionMismatch);
        }
        let (w0, h0) = (d.width, d.height);
        let mut lin = test.to_linear()?;
        let mut blur = GaussianBlur::with_kernel(w0, *kernel(), true);
        let mut moments: [Vec<f32>; 3] = Default::default();
        let mut maps: [Vec<f32>; 3] = Default::default();
        let mut acc = Vec::new();
        let mut map = if want_map {
            vec![0f32; w0 * h0]
        } else {
            Vec::new()
        };
        let mut sum = 0f64;
        for (s, reference) in d.scales.iter().enumerate() {
            let (w, h) = (reference.width, reference.height);
            if s > 0 {
                let previous = &d.scales[s - 1];
                for plane in &mut lin {
                    *plane = downsample(plane, previous.width, previous.height).0;
                }
            }
            let xyb = positive_xyb_of_linear(&lin);
            blur.set_width(w);
            for buffer in moments.iter_mut().chain(&mut maps) {
                buffer.resize(w * h, 0.0);
            }
            if want_map {
                acc.clear();
                acc.resize(w * h, 0.0);
            }
            for c in 0..3 {
                let active = active_maps(c, s, d.scales.len());
                if !active.contains(&true) {
                    continue;
                }
                let (src, rec) = (&reference.xyb[c], &xyb[c]);
                let (mu1, s11) = (&reference.mu[c], &reference.sigma[c]);
                let [mu2, s22, s12] = &mut moments;
                let [ssim, artifact, detail] = &mut maps;
                blur.blur(rec, h, mu2);
                // The finest X/B scales only weight edge maps; the variance
                // and covariance blurs cannot affect the score there.
                if active[0] {
                    blur.blur_product([rec, rec], h, s22);
                    blur.blur_product([src, rec], h, s12);
                    for (i, out) in ssim.iter_mut().enumerate() {
                        *out = ssim_value(mu1[i], mu2[i], s11[i], s22[i], s12[i]);
                    }
                }
                for (i, (artifact, detail)) in
                    artifact.iter_mut().zip(detail.iter_mut()).enumerate()
                {
                    let d1 = edge_ratio(src[i], rec[i], mu1[i], mu2[i]);
                    *artifact = d1.max(0.0);
                    *detail = (-d1).max(0.0);
                }
                let base = weight_base(c, s, d.scales.len());
                for (m, values) in [&*ssim, &*artifact, &*detail].into_iter().enumerate() {
                    let acc = want_map.then_some(&mut acc[..]);
                    sum += pool(values, WEIGHTS[base + m], WEIGHTS[base + 3 + m], acc);
                }
            }
            if want_map {
                accumulate_scale_map(&acc, w0, h0, s, &mut map);
            }
        }
        if want_map {
            let factor = deficit_slope(sum) as f32 * (w0 * h0) as f32;
            for v in &mut map {
                *v *= factor;
            }
        }
        Ok((score_from_sum(sum), map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_kernels() -> [(&'static str, MapKernels, bool); 2] {
        let selected = map_kernels();
        let fused = cfg!(all(target_arch = "aarch64", feature = "neon")) || {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
            }
            #[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
            {
                false
            }
        };
        [
            (
                "portable",
                MapKernels {
                    contribution: contribution_scalar,
                    expand: [
                        expand_row_scalar::<1>,
                        expand_row_scalar::<2>,
                        expand_row_scalar::<4>,
                        expand_row_scalar::<8>,
                        expand_row_scalar::<16>,
                        expand_row_scalar::<32>,
                    ],
                },
                false,
            ),
            (
                "dispatch",
                MapKernels {
                    contribution: selected.contribution,
                    expand: selected.expand,
                },
                fused,
            ),
        ]
    }

    #[test]
    fn contribution_kernels_cover_unaligned_vectors_and_tails() {
        const GUARD: f32 = -12345.0;
        for (name, kernels, fused) in test_kernels() {
            for n in (0..=65).chain([127, 128, 129, 255, 257]) {
                for offset in 0..4 {
                    let mut input = vec![GUARD; n + 8];
                    let map = &mut input[offset..offset + n];
                    for (i, v) in map.iter_mut().enumerate() {
                        *v = (i % 31 + 1) as f32 * 0.173829;
                    }
                    let input_before = input.clone();
                    for norm_weight in [0.0, 0.0009876543] {
                        let p = MapContribution {
                            linear: 0.0001234567,
                            inv_norm: 0.7123456,
                            norm_weight,
                        };
                        let mut output = vec![GUARD; n + 8];
                        let initial: Vec<_> =
                            (0..n).map(|i| (i % 17) as f32 * 0.00001876543).collect();
                        output[offset..offset + n].copy_from_slice(&initial);
                        // SAFETY: the selected kernel's CPU requirements were
                        // detected; the offset slices have matching lengths.
                        let done = unsafe {
                            (kernels.contribution)(
                                &input[offset..offset + n],
                                p,
                                &mut output[offset..offset + n],
                            )
                        };
                        assert!(done <= n);
                        for i in 0..n {
                            let v = input[offset + i];
                            let expected = if i >= done {
                                initial[i]
                            } else if norm_weight == 0.0 {
                                if fused {
                                    p.linear.mul_add(v, initial[i])
                                } else {
                                    initial[i] + p.linear * v
                                }
                            } else {
                                let normalized = v * p.inv_norm;
                                let square = normalized * normalized;
                                initial[i]
                                    + if fused {
                                        (norm_weight * square).mul_add(square, p.linear * v)
                                    } else {
                                        p.linear * v + norm_weight * square * square
                                    }
                            };
                            assert_eq!(
                                output[offset + i].to_bits(),
                                expected.to_bits(),
                                "{name}, n={n}, offset={offset}, pixel={i}"
                            );
                        }
                        assert!(output[..offset].iter().all(|&v| v == GUARD));
                        assert!(output[offset + n..].iter().all(|&v| v == GUARD));
                        assert_eq!(input, input_before);

                        if name == "dispatch" {
                            let mut actual = initial.clone();
                            contribute(&input[offset..offset + n], p, &mut actual);
                            for i in done..n {
                                let v = input[offset + i];
                                let normalized = v * p.inv_norm;
                                let square = normalized * normalized;
                                let expected = if norm_weight == 0.0 {
                                    p.linear.mul_add(v, initial[i])
                                } else {
                                    initial[i]
                                        + (norm_weight * square).mul_add(square, p.linear * v)
                                };
                                assert_eq!(actual[i].to_bits(), expected.to_bits());
                            }
                            assert_eq!(actual[..done], output[offset..offset + done]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scale_expansion_covers_all_scales_and_cropped_edges() {
        const GUARD: f32 = -12345.0;
        for (name, kernels, _) in test_kernels() {
            for scale in 0..NUM_SCALES {
                let step = 1 << scale;
                let share = 1.0 / (step * step) as f32;
                for width in (0usize..=65).chain([127, 129, 255, 257]) {
                    let n = width.div_ceil(step);
                    for offset in 0..4 {
                        let mut input = vec![GUARD; n + 8];
                        for (i, v) in input[offset..offset + n].iter_mut().enumerate() {
                            *v = ((i * 19) % 127) as f32 * 0.125;
                        }
                        let before = input.clone();
                        let mut output = vec![GUARD; width + 8];
                        let initial: Vec<_> = (0..width)
                            .map(|i| ((i * 11) % 23) as f32 * 0.0625)
                            .collect();
                        output[offset..offset + width].copy_from_slice(&initial);
                        // SAFETY: detected CPU features and ceil(width/STEP)
                        // source pixels cover every complete destination run.
                        let done = unsafe {
                            (kernels.expand[scale])(
                                &input[offset..offset + n],
                                &mut output[offset..offset + width],
                                share,
                            )
                        };
                        assert!(done <= width && (done == width || done.is_multiple_of(step)));
                        for i in 0..width {
                            let expected = initial[i]
                                + if i < done {
                                    input[offset + i / step] * share
                                } else {
                                    0.0
                                };
                            assert_eq!(
                                output[offset + i].to_bits(),
                                expected.to_bits(),
                                "{name}, scale={scale}, width={width}, offset={offset}, pixel={i}"
                            );
                        }
                        assert!(output[..offset].iter().all(|&v| v == GUARD));
                        assert!(output[offset + width..].iter().all(|&v| v == GUARD));
                        assert_eq!(input, before);
                    }
                    for height in [0usize, 1, 3, 17, 33] {
                        let h = height.div_ceil(step);
                        let acc: Vec<_> = (0..n * h)
                            .map(|i| ((i * 19) % 127) as f32 * 0.125)
                            .collect();
                        let mut actual: Vec<_> = (0..width * height)
                            .map(|i| ((i * 11) % 23) as f32 * 0.0625)
                            .collect();
                        let mut expected = actual.clone();
                        for y in 0..height {
                            for x in 0..width {
                                expected[y * width + x] += acc[(y / step) * n + x / step] * share;
                            }
                        }
                        accumulate_scale_map(&acc, width, height, scale, &mut actual);
                        assert_eq!(actual, expected, "scale={scale}, {width}x{height}");
                    }
                }
            }
        }
    }

    #[test]
    fn replacement_tails_update_fourth_powers_and_revert() {
        let totals = |values: &[f32]| {
            values.iter().fold((0.0, 0.0), |(s1, s4), &v| {
                let v = v as f64;
                (s1 + v, s4 + v.powi(4))
            })
        };
        for n in (0..=65).chain([127, 129, 257]) {
            let old: Vec<_> = (0..n).map(|i| (i % 23) as f32 / 16.0).collect();
            let new: Vec<_> = (0..n).map(|i| (i % 29) as f32 / 32.0).collect();
            for fourth in [false, true] {
                let expected = |values: &[f32]| {
                    let (s1, s4) = totals(values);
                    (s1, if fourth { s4 } else { 0.0 })
                };
                let mut map = old.clone();
                let mut sums = expected(&old);
                replace_map(&mut map, [&new], &mut sums, fourth, |[v]| v);
                assert_eq!(map, new);
                assert_eq!(sums, expected(&new), "n={n}, fourth={fourth}");
                replace_map(&mut map, [&old], &mut sums, fourth, |[v]| v);
                assert_eq!(map, old);
                assert_eq!(sums, expected(&old), "n={n}, fourth={fourth}");
            }
        }
    }

    #[test]
    fn first_norm_reduction_matches_full_reduction_across_tails_and_hdr_values() {
        for n in (0..=65).chain([127, 129, 4099, 307_200]) {
            for magnitude in [0.0, 1e-35, 1e-7, 1.0, 1e20, 1e37] {
                let map: Vec<_> = (0..n)
                    .map(|i| magnitude * ((i % 17 + 1) as f32 / 17.0))
                    .collect();
                let mut actual = vec![0.0; n];
                let mut first = (0.0, 0.0);
                replace_map(&mut actual, [&map], &mut first, false, |[v]| v);
                let both = map_sums(&map);
                assert_eq!(actual, map);
                assert_eq!(first.0.to_bits(), both.0.to_bits());
                assert_eq!(first.1, 0.0);
            }
        }
    }

    // Original f64 per-pixel pooling, kept as an independent precision oracle.
    fn reference_pool(map: &[f32], w1: f64, w4: f64, acc: &mut [f32]) -> f64 {
        if w1 == 0.0 && w4 == 0.0 {
            return 0.0;
        }
        let n = map.len() as f64;
        let (mut s1, mut s4) = (0.0f64, 0.0f64);
        for &v in map {
            let v = v as f64;
            s1 += v;
            s4 += v * v * v * v;
        }
        let n4 = (s4 / n).sqrt().sqrt();
        let g1 = w1 / n;
        let g4 = if n4 > 0.0 {
            w4 / (4.0 * n * n4 * n4 * n4)
        } else {
            0.0
        };
        for (a, &v) in acc.iter_mut().zip(map) {
            let v = v as f64;
            let v_sqr = v * v;
            *a += (g1 * v + g4 * v_sqr * v_sqr) as f32;
        }
        w1 * s1 / n + w4 * n4
    }

    #[test]
    fn pooling_matches_f64_for_sparse_maps_tails_and_extreme_norms() {
        let mut max_relative_error = 0.0f32;
        for n in (1..=17).chain([31, 257, 4099, 307_200]) {
            for magnitude in [0.0, 1e-44, 1e-32, 1e-20, 1e-7, 1.0, 1e5, 1e20, 1e37] {
                for sparse in [false, true] {
                    let map: Vec<_> = (0..n)
                        .map(|i| {
                            if sparse && i != n / 2 {
                                0.0
                            } else {
                                magnitude * ((i % 17 + 1) as f32 / 17.0)
                            }
                        })
                        .collect();
                    let mut actual = vec![0.0; n];
                    let mut expected = vec![0.0; n];
                    let (weight_sets, _) = WEIGHTS.as_chunks::<6>();
                    for weights in weight_sets.iter().step_by(if n > 4099 { 6 } else { 1 }) {
                        for m in 0..3 {
                            actual.fill(0.0);
                            expected.fill(0.0);
                            let sum = pool(&map, weights[m], weights[m + 3], Some(&mut actual));
                            let reference =
                                reference_pool(&map, weights[m], weights[m + 3], &mut expected);
                            assert!(
                                (sum - reference).abs() <= 2e-12 * reference.abs(),
                                "{n}, {magnitude}, {sparse}: {sum} vs {reference}"
                            );
                            for (&a, &b) in actual.iter().zip(&expected) {
                                if b.is_infinite() {
                                    assert_eq!(a, b);
                                    continue;
                                }
                                let error = (a - b).abs() / b.abs().max(f32::MIN_POSITIVE);
                                max_relative_error = max_relative_error.max(error);
                                assert!(
                                    error <= 2e-6,
                                    "{n}, {magnitude}, {sparse}: {a} vs {b}, error {error}"
                                );
                            }
                        }
                    }
                }
            }
        }
        eprintln!("f32 contribution max relative error: {max_relative_error}");
    }

    #[test]
    fn kernel_is_a_normalized_gaussian_like_window() {
        let k = kernel();
        let sum: f32 = k.iter().sum();
        eprintln!("kernel {k:?} sum {sum}");
        assert!((sum - 1.0).abs() < 1e-3, "{sum}");
        assert!(k[RADIUS] > k[RADIUS - 1] && k[RADIUS - 1] > k[RADIUS - 2]);
        // interior of a constant plane stays constant
        let (w, h) = (32, 20);
        let src = vec![0.7f32; w * h];
        let mut out = vec![0f32; w * h];
        GaussianBlur::with_kernel(w, *kernel(), true).blur(&src, h, &mut out);
        assert!(
            (out[10 * w + 16] - 0.7).abs() < 1e-4,
            "{}",
            out[10 * w + 16]
        );
        assert!(out[0] < 0.7 * 0.9, "zero padding at the corner: {}", out[0]);
    }

    #[test]
    fn identical_images_score_100_with_zero_map() {
        let img = crate::testing::Planes::textured(3, 64, 48);
        let (d, map) = crate::testing::full(&img, &img);
        assert!(d.abs() < 1e-4, "{d}");
        assert!(map.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn map_mean_tracks_the_deficit_and_degradation_lowers_the_score() {
        let img = crate::testing::Planes::textured(5, 96, 80);
        let mut rec = img.clone();
        let mut rng = 12345u32;
        for v in &mut rec.p[1] {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            *v += 0.03 * ((rng >> 24) as f32 / 255.0 - 0.5);
        }
        let (d, map) = crate::testing::full(&img, &rec);
        assert!(d > 0.5 && d < 60.0, "{d}");
        let s = map.iter().sum::<f32>() as f64 / map.len() as f64;
        // the map is the local derivative of a concave mapping: positive and of the deficit's order
        assert!(s > 0.1 * d && s < 2.0 * d, "map mean {s} vs deficit {d}");
        let mut worse = rec.clone();
        for v in &mut worse.p[1] {
            *v = *v * 1.5 - 0.05;
        }
        let (d2, _) = crate::testing::full(&img, &worse);
        assert!(d2 > d, "{d2} <= {d}");
    }
}
