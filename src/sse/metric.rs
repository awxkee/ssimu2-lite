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
use crate::maps::{MapInputs, MapSums};
use crate::metric::C2;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::metric::MapContribution;

/// # Safety
/// SSE4.1 must be available and both slices must have equal lengths.
#[target_feature(enable = "sse4.1")]
pub(crate) fn ss2_contribution_sse41(map: &[f32], p: MapContribution, acc: &mut [f32]) -> usize {
    if p.norm_weight == 0.0 {
        contribution::<false>(map, p, acc)
    } else {
        contribution::<true>(map, p, acc)
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn contribution<const FOURTH: bool>(map: &[f32], p: MapContribution, acc: &mut [f32]) -> usize {
    let (map_vectors, _) = map.as_chunks::<4>();
    let (acc_vectors, _) = acc.as_chunks_mut::<4>();
    let done = acc_vectors.len() * 4;
    let linear = _mm_set1_ps(p.linear);
    let inv_norm = _mm_set1_ps(p.inv_norm);
    let norm_weight = _mm_set1_ps(p.norm_weight);
    for (map, acc) in map_vectors.iter().zip(acc_vectors) {
        // The entry point's equal-length contract covers complete vectors.
        unsafe {
            let v = _mm_loadu_ps(map.as_ptr());
            let old = _mm_loadu_ps(acc.as_ptr());
            let mut value = _mm_mul_ps(v, linear);
            if FOURTH {
                let normalized = _mm_mul_ps(v, inv_norm);
                let square = _mm_mul_ps(normalized, normalized);
                value = _mm_add_ps(value, _mm_mul_ps(_mm_mul_ps(norm_weight, square), square));
            }
            _mm_storeu_ps(acc.as_mut_ptr(), _mm_add_ps(old, value));
        }
    }
    done
}

/// # Safety
/// SSE4.1 must be available, STEP must be 1/2/4/8/16/32, and src must contain
/// ceil(dst.len()/STEP) pixels. Only complete vector-sized runs are processed.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ss2_expand_row_sse41<const STEP: usize>(
    src: &[f32],
    dst: &mut [f32],
    share: f32,
) -> usize {
    let factor = _mm_set1_ps(share);
    // Fixed-size runs leave cropped tails for the scalar caller.
    unsafe {
        if STEP == 1 {
            let (runs, _) = dst.as_chunks_mut::<4>();
            let done = runs.len() * 4;
            for (dst, src) in runs.iter_mut().zip(src.as_chunks::<4>().0) {
                let value = _mm_loadu_ps(src.as_ptr());
                let old = _mm_loadu_ps(dst.as_ptr());
                _mm_storeu_ps(dst.as_mut_ptr(), _mm_add_ps(old, _mm_mul_ps(value, factor)));
            }
            done
        } else if STEP == 2 {
            let (runs, _) = dst.as_chunks_mut::<8>();
            let done = runs.len() * 8;
            for (dst, src) in runs.iter_mut().zip(src.as_chunks::<4>().0) {
                let v = _mm_loadu_ps(src.as_ptr());
                let a = _mm_unpacklo_ps(v, v);
                let b = _mm_unpackhi_ps(v, v);
                for (dst, value) in dst.as_chunks_mut::<4>().0.iter_mut().zip([a, b]) {
                    let old = _mm_loadu_ps(dst.as_ptr());
                    _mm_storeu_ps(dst.as_mut_ptr(), _mm_add_ps(old, _mm_mul_ps(value, factor)));
                }
            }
            done
        } else {
            let (runs, _) = dst.as_chunks_mut::<STEP>();
            let done = runs.len() * STEP;
            for (run, &value) in runs.iter_mut().zip(src) {
                let value = _mm_mul_ps(_mm_set1_ps(value), factor);
                for dst in run.as_chunks_mut::<4>().0 {
                    let old = _mm_loadu_ps(dst.as_ptr());
                    _mm_storeu_ps(dst.as_mut_ptr(), _mm_add_ps(old, value));
                }
            }
            done
        }
    }
}

/// # Safety
/// SSE4.1 must be available; every input must match the output length.
/// REPLACE accumulates deltas into the eight f64 lanes; FOURTH also updates x⁴.
#[target_feature(enable = "sse4.1")]
pub(crate) fn ss2_map_sse41<const KIND: usize, const REPLACE: bool, const FOURTH: bool>(
    inputs: MapInputs<'_>,
    map: &mut [f32],
    sums: &mut MapSums,
) -> usize {
    let (blocks, tail) = map.as_chunks_mut::<8>();
    let (tail, _) = tail.as_chunks_mut::<4>();
    let done = blocks.len() * 8 + tail.len() * 4;
    // Eight-pixel blocks preserve the reduction's lane order.
    unsafe {
        let first_chunks = sums.first.as_chunks::<2>().0;
        let mut first: [__m128d; 4] =
            std::array::from_fn(|i| _mm_loadu_pd(first_chunks[i].as_ptr()));
        let fourth_chunks = sums.fourth.as_chunks::<2>().0;
        let mut fourth: [__m128d; 4] =
            std::array::from_fn(|i| _mm_loadu_pd(fourth_chunks[i].as_ptr()));
        for (block, dst) in blocks.iter_mut().enumerate() {
            for (vector, dst) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let i = block * 8 + vector * 4;
                let new = map_value::<KIND>(inputs.map(|p| p.as_ptr().add(i)));
                if REPLACE {
                    let old = _mm_loadu_ps(dst.as_ptr());
                    reduce::<true, FOURTH>(new, old, vector * 2, &mut first, &mut fourth);
                }
                _mm_storeu_ps(dst.as_mut_ptr(), new);
            }
        }
        for dst in tail {
            let i = done - 4;
            let new = map_value::<KIND>(inputs.map(|p| p.as_ptr().add(i)));
            if REPLACE {
                let old = _mm_loadu_ps(dst.as_ptr());
                reduce::<true, FOURTH>(new, old, 0, &mut first, &mut fourth);
            }
            _mm_storeu_ps(dst.as_mut_ptr(), new);
        }
        if REPLACE {
            for (dst, value) in sums.first.as_chunks_mut::<2>().0.iter_mut().zip(first) {
                _mm_storeu_pd(dst.as_mut_ptr(), value);
            }
            if FOURTH {
                for (dst, value) in sums.fourth.as_chunks_mut::<2>().0.iter_mut().zip(fourth) {
                    _mm_storeu_pd(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn map_value<const KIND: usize>(input: [*const f32; 5]) -> __m128 {
    // The caller provides a complete vector at each pointer.
    unsafe {
        let [a, b, c, d] = [input[0], input[1], input[2], input[3]].map(|p| _mm_loadu_ps(p));
        let one = _mm_set1_ps(1.0);
        let value = if KIND == 0 {
            let e = _mm_loadu_ps(input[4]);
            let diff = _mm_sub_ps(a, b);
            let num_m = _mm_sub_ps(one, _mm_mul_ps(diff, diff));
            let num_s = _mm_add_ps(
                _mm_mul_ps(_mm_set1_ps(2.0), _mm_sub_ps(e, _mm_mul_ps(a, b))),
                _mm_set1_ps(C2),
            );
            let denom = _mm_add_ps(
                _mm_add_ps(
                    _mm_sub_ps(c, _mm_mul_ps(a, a)),
                    _mm_sub_ps(d, _mm_mul_ps(b, b)),
                ),
                _mm_set1_ps(C2),
            );
            _mm_sub_ps(one, _mm_div_ps(_mm_mul_ps(num_m, num_s), denom))
        } else {
            let num = _mm_add_ps(one, _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(b, d)));
            let denom = _mm_add_ps(one, _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(a, c)));
            let ratio = _mm_sub_ps(_mm_div_ps(num, denom), one);
            if KIND == 1 {
                ratio
            } else {
                _mm_sub_ps(_mm_setzero_ps(), ratio)
            }
        };
        _mm_max_ps(value, _mm_setzero_ps())
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn reduce<const REPLACE: bool, const FOURTH: bool>(
    new: __m128,
    old: __m128,
    lane: usize,
    first: &mut [__m128d; 4],
    fourth: &mut [__m128d; 4],
) {
    let new = [_mm_cvtps_pd(new), _mm_cvtps_pd(_mm_movehl_ps(new, new))];
    let old = [_mm_cvtps_pd(old), _mm_cvtps_pd(_mm_movehl_ps(old, old))];
    for (half, (v, old)) in new.into_iter().zip(old).enumerate() {
        let delta = if REPLACE { _mm_sub_pd(v, old) } else { v };
        first[lane + half] = _mm_add_pd(first[lane + half], delta);
        if FOURTH {
            let square = _mm_mul_pd(v, v);
            let mut delta = _mm_mul_pd(square, square);
            if REPLACE {
                let square = _mm_mul_pd(old, old);
                delta = _mm_sub_pd(delta, _mm_mul_pd(square, square));
            }
            fourth[lane + half] = _mm_add_pd(fourth[lane + half], delta);
        }
    }
}

/// # Safety
/// SSE4.1 must be available. Only complete eight-pixel blocks are read.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ss2_sums_sse41(map: &[f32], sums: &mut MapSums) -> usize {
    let (blocks, _) = map.as_chunks::<8>();
    // Complete blocks fit the source, and the sums hold exactly eight f64 lanes.
    unsafe {
        let first_chunks = sums.first.as_chunks::<2>().0;
        let mut first: [__m128d; 4] =
            std::array::from_fn(|i| _mm_loadu_pd(first_chunks[i].as_ptr()));
        let fourth_chunks = sums.fourth.as_chunks::<2>().0;
        let mut fourth: [__m128d; 4] =
            std::array::from_fn(|i| _mm_loadu_pd(fourth_chunks[i].as_ptr()));
        for block in blocks {
            for (vector, src) in block.as_chunks::<4>().0.iter().enumerate() {
                let value = _mm_loadu_ps(src.as_ptr());
                reduce::<false, true>(value, _mm_setzero_ps(), vector * 2, &mut first, &mut fourth);
            }
        }
        for (dst, value) in sums.first.as_chunks_mut::<2>().0.iter_mut().zip(first) {
            _mm_storeu_pd(dst.as_mut_ptr(), value);
        }
        for (dst, value) in sums.fourth.as_chunks_mut::<2>().0.iter_mut().zip(fourth) {
            _mm_storeu_pd(dst.as_mut_ptr(), value);
        }
    }
    blocks.len() * 8
}

/// # Safety
/// SSE4.1 must be available and all input/output slices must have equal lengths.
/// The signed edge ratio is computed once for both maps and their f64 deltas.
#[target_feature(enable = "sse4.1")]
pub(crate) fn ss2_edges_sse41<
    const REPLACE: bool,
    const ARTIFACT_FOURTH: bool,
    const DETAIL_FOURTH: bool,
>(
    inputs: MapInputs<'_>,
    [artifact, detail]: [&mut [f32]; 2],
    sums: &mut [MapSums; 2],
) -> usize {
    let (artifact_blocks, artifact_tail) = artifact.as_chunks_mut::<8>();
    let (detail_blocks, detail_tail) = detail.as_chunks_mut::<8>();
    let (artifact_tail, _) = artifact_tail.as_chunks_mut::<4>();
    let (detail_tail, _) = detail_tail.as_chunks_mut::<4>();
    let done = artifact_blocks.len() * 8 + artifact_tail.len() * 4;
    // Keep all eight reduction lanes grouped, including four-wide backends.
    unsafe {
        let af_chunks = sums[0].first.as_chunks::<2>().0;
        let mut af: [__m128d; 4] = std::array::from_fn(|i| _mm_loadu_pd(af_chunks[i].as_ptr()));
        let a4_chunks = sums[0].fourth.as_chunks::<2>().0;
        let mut a4: [__m128d; 4] = std::array::from_fn(|i| _mm_loadu_pd(a4_chunks[i].as_ptr()));
        let df_chunks = sums[1].first.as_chunks::<2>().0;
        let mut df: [__m128d; 4] = std::array::from_fn(|i| _mm_loadu_pd(df_chunks[i].as_ptr()));
        let d4_chunks = sums[1].fourth.as_chunks::<2>().0;
        let mut d4: [__m128d; 4] = std::array::from_fn(|i| _mm_loadu_pd(d4_chunks[i].as_ptr()));
        let one = _mm_set1_ps(1.0);
        for (block, (artifact, detail)) in artifact_blocks.iter_mut().zip(detail_blocks).enumerate()
        {
            let artifact = artifact.as_chunks_mut::<4>().0;
            let detail = detail.as_chunks_mut::<4>().0;
            for (vector, (artifact, detail)) in artifact.iter_mut().zip(detail).enumerate() {
                let i = block * 8 + vector * 4;
                let [a, b, c, d] = [inputs[0], inputs[1], inputs[2], inputs[3]]
                    .map(|p| _mm_loadu_ps(p.as_ptr().add(i)));
                let ratio = _mm_sub_ps(
                    _mm_div_ps(
                        _mm_add_ps(one, _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(b, d))),
                        _mm_add_ps(one, _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(a, c))),
                    ),
                    one,
                );
                let a = _mm_max_ps(ratio, _mm_setzero_ps());
                let d = _mm_max_ps(_mm_xor_ps(ratio, _mm_set1_ps(-0.0)), _mm_setzero_ps());
                if REPLACE {
                    let lane = vector * 2;
                    reduce::<true, ARTIFACT_FOURTH>(
                        a,
                        _mm_loadu_ps(artifact.as_ptr()),
                        lane,
                        &mut af,
                        &mut a4,
                    );
                    reduce::<true, DETAIL_FOURTH>(
                        d,
                        _mm_loadu_ps(detail.as_ptr()),
                        lane,
                        &mut df,
                        &mut d4,
                    );
                }
                _mm_storeu_ps(artifact.as_mut_ptr(), a);
                _mm_storeu_ps(detail.as_mut_ptr(), d);
            }
        }
        for (artifact, detail) in artifact_tail.iter_mut().zip(detail_tail) {
            let i = done - 4;
            let [a, b, c, d] = [inputs[0], inputs[1], inputs[2], inputs[3]]
                .map(|p| _mm_loadu_ps(p.as_ptr().add(i)));
            let ratio = _mm_sub_ps(
                _mm_div_ps(
                    _mm_add_ps(one, _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(b, d))),
                    _mm_add_ps(one, _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(a, c))),
                ),
                one,
            );
            let a = _mm_max_ps(ratio, _mm_setzero_ps());
            let d = _mm_max_ps(_mm_xor_ps(ratio, _mm_set1_ps(-0.0)), _mm_setzero_ps());
            if REPLACE {
                let lane = 0;
                reduce::<true, ARTIFACT_FOURTH>(
                    a,
                    _mm_loadu_ps(artifact.as_ptr()),
                    lane,
                    &mut af,
                    &mut a4,
                );
                reduce::<true, DETAIL_FOURTH>(
                    d,
                    _mm_loadu_ps(detail.as_ptr()),
                    lane,
                    &mut df,
                    &mut d4,
                );
            }
            _mm_storeu_ps(artifact.as_mut_ptr(), a);
            _mm_storeu_ps(detail.as_mut_ptr(), d);
        }
        if REPLACE {
            for (dst, value) in sums[0].first.as_chunks_mut::<2>().0.iter_mut().zip(af) {
                _mm_storeu_pd(dst.as_mut_ptr(), value);
            }
            for (dst, value) in sums[1].first.as_chunks_mut::<2>().0.iter_mut().zip(df) {
                _mm_storeu_pd(dst.as_mut_ptr(), value);
            }
            if ARTIFACT_FOURTH {
                for (dst, value) in sums[0].fourth.as_chunks_mut::<2>().0.iter_mut().zip(a4) {
                    _mm_storeu_pd(dst.as_mut_ptr(), value);
                }
            }
            if DETAIL_FOURTH {
                for (dst, value) in sums[1].fourth.as_chunks_mut::<2>().0.iter_mut().zip(d4) {
                    _mm_storeu_pd(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}
