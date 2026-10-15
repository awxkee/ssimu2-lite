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
use crate::metric::MapContribution;
use std::arch::x86_64::*;

/// # Safety
/// AVX2/FMA must be available and the input/output slices must have equal lengths.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn ss2_contribution_avx2(map: &[f32], p: MapContribution, acc: &mut [f32]) -> usize {
    debug_assert_eq!(map.len(), acc.len());
    if p.norm_weight == 0.0 {
        let done = contribution::<4, false>(map, p, acc);
        done + contribution::<1, false>(&map[done..], p, &mut acc[done..])
    } else {
        let done = contribution::<4, true>(map, p, acc);
        done + contribution::<1, true>(&map[done..], p, &mut acc[done..])
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn contribution<const N: usize, const FOURTH: bool>(
    map: &[f32],
    p: MapContribution,
    acc: &mut [f32],
) -> usize {
    assert_eq!(map.len(), acc.len());
    let (map_vectors, _) = map.as_chunks::<8>();
    let (acc_vectors, _) = acc.as_chunks_mut::<8>();
    let (map_blocks, _) = map_vectors.as_chunks::<N>();
    let (acc_blocks, _) = acc_vectors.as_chunks_mut::<N>();
    let done = acc_blocks.len() * N * 8;
    let linear = _mm256_set1_ps(p.linear);
    let inv_norm = _mm256_set1_ps(p.inv_norm);
    let norm_weight = _mm256_set1_ps(p.norm_weight);
    for (map, acc) in map_blocks.iter().zip(acc_blocks) {
        for (map, acc) in map.iter().zip(acc) {
            // The entry point's equal-length contract covers complete vectors.
            unsafe {
                let v = _mm256_loadu_ps(map.as_ptr());
                let old = _mm256_loadu_ps(acc.as_ptr());
                let out = if FOURTH {
                    let normalized = _mm256_mul_ps(v, inv_norm);
                    let square = _mm256_mul_ps(normalized, normalized);
                    let weighted = _mm256_mul_ps(square, norm_weight);
                    let first = _mm256_mul_ps(v, linear);
                    _mm256_add_ps(old, _mm256_fmadd_ps(weighted, square, first))
                } else {
                    _mm256_fmadd_ps(v, linear, old)
                };
                _mm256_storeu_ps(acc.as_mut_ptr(), out);
            }
        }
    }
    done
}

/// # Safety
/// AVX2/FMA must be available, STEP must be 1/2/4/8/16/32, and src must contain
/// ceil(dst.len()/STEP) pixels. Only complete vector-sized runs are processed.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn ss2_expand_row_avx2<const STEP: usize>(
    src: &[f32],
    dst: &mut [f32],
    share: f32,
) -> usize {
    let factor = _mm256_set1_ps(share);
    // Fixed-size runs leave cropped tails for the scalar caller.
    unsafe {
        if STEP == 1 {
            let (runs, _) = dst.as_chunks_mut::<8>();
            let done = runs.len() * 8;
            for (dst, src) in runs.iter_mut().zip(src.as_chunks::<8>().0) {
                let value = _mm256_loadu_ps(src.as_ptr());
                let old = _mm256_loadu_ps(dst.as_ptr());
                _mm256_storeu_ps(dst.as_mut_ptr(), _mm256_fmadd_ps(value, factor, old));
            }
            done
        } else if STEP == 2 {
            let (runs, _) = dst.as_chunks_mut::<16>();
            let done = runs.len() * 16;
            for (dst, src) in runs.iter_mut().zip(src.as_chunks::<8>().0) {
                let v = _mm256_loadu_ps(src.as_ptr());
                let lo = _mm256_unpacklo_ps(v, v);
                let hi = _mm256_unpackhi_ps(v, v);
                let a = _mm256_permute2f128_ps::<0x20>(lo, hi);
                let b = _mm256_permute2f128_ps::<0x31>(lo, hi);
                for (dst, value) in dst.as_chunks_mut::<8>().0.iter_mut().zip([a, b]) {
                    let old = _mm256_loadu_ps(dst.as_ptr());
                    _mm256_storeu_ps(dst.as_mut_ptr(), _mm256_fmadd_ps(value, factor, old));
                }
            }
            done
        } else if STEP == 4 {
            let (runs, _) = dst.as_chunks_mut::<8>();
            let done = runs.len() * 8;
            for (dst, &[a, b]) in runs.iter_mut().zip(src.as_chunks::<2>().0) {
                let value = _mm256_set_m128(_mm_set1_ps(b), _mm_set1_ps(a));
                let old = _mm256_loadu_ps(dst.as_ptr());
                _mm256_storeu_ps(dst.as_mut_ptr(), _mm256_fmadd_ps(value, factor, old));
            }
            done
        } else {
            let (runs, _) = dst.as_chunks_mut::<STEP>();
            let done = runs.len() * STEP;
            for (run, &value) in runs.iter_mut().zip(src) {
                let value = _mm256_set1_ps(value);
                for dst in run.as_chunks_mut::<8>().0 {
                    let old = _mm256_loadu_ps(dst.as_ptr());
                    _mm256_storeu_ps(dst.as_mut_ptr(), _mm256_fmadd_ps(value, factor, old));
                }
            }
            done
        }
    }
}

/// # Safety
/// AVX2 must be available; every input must match the output length.
/// REPLACE accumulates deltas into the eight f64 lanes; FOURTH also updates x⁴.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ss2_map_avx2<const KIND: usize, const REPLACE: bool, const FOURTH: bool>(
    inputs: MapInputs<'_>,
    map: &mut [f32],
    sums: &mut MapSums,
) -> usize {
    let (blocks, _) = map.as_chunks_mut::<8>();
    let done = blocks.len() * 8;
    // Eight-pixel blocks preserve the reduction's lane order.
    unsafe {
        let first_chunks = sums.first.as_chunks::<4>().0;
        let mut first: [__m256d; 2] =
            std::array::from_fn(|i| _mm256_loadu_pd(first_chunks[i].as_ptr()));
        let fourth_chunks = sums.fourth.as_chunks::<4>().0;
        let mut fourth: [__m256d; 2] =
            std::array::from_fn(|i| _mm256_loadu_pd(fourth_chunks[i].as_ptr()));
        for (block, dst) in blocks.iter_mut().enumerate() {
            let i = block * 8;
            let new = map_value::<KIND>(inputs.map(|p| p.as_ptr().add(i)));
            if REPLACE {
                let old = _mm256_loadu_ps(dst.as_ptr());
                reduce::<true, FOURTH>(new, old, 0, &mut first, &mut fourth);
            }
            _mm256_storeu_ps(dst.as_mut_ptr(), new);
        }
        if REPLACE {
            for (dst, value) in sums.first.as_chunks_mut::<4>().0.iter_mut().zip(first) {
                _mm256_storeu_pd(dst.as_mut_ptr(), value);
            }
            if FOURTH {
                for (dst, value) in sums.fourth.as_chunks_mut::<4>().0.iter_mut().zip(fourth) {
                    _mm256_storeu_pd(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}

#[inline]
#[target_feature(enable = "avx2")]
fn map_value<const KIND: usize>(input: [*const f32; 5]) -> __m256 {
    // The caller provides a complete vector at each pointer.
    unsafe {
        let [a, b, c, d] = [input[0], input[1], input[2], input[3]].map(|p| _mm256_loadu_ps(p));
        let one = _mm256_set1_ps(1.0);
        let value = if KIND == 0 {
            let e = _mm256_loadu_ps(input[4]);
            let diff = _mm256_sub_ps(a, b);
            let num_m = _mm256_sub_ps(one, _mm256_mul_ps(diff, diff));
            let num_s = _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(2.0), _mm256_sub_ps(e, _mm256_mul_ps(a, b))),
                _mm256_set1_ps(C2),
            );
            let denom = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_sub_ps(c, _mm256_mul_ps(a, a)),
                    _mm256_sub_ps(d, _mm256_mul_ps(b, b)),
                ),
                _mm256_set1_ps(C2),
            );
            _mm256_sub_ps(one, _mm256_div_ps(_mm256_mul_ps(num_m, num_s), denom))
        } else {
            let num = _mm256_add_ps(
                one,
                _mm256_andnot_ps(_mm256_set1_ps(-0.0), _mm256_sub_ps(b, d)),
            );
            let denom = _mm256_add_ps(
                one,
                _mm256_andnot_ps(_mm256_set1_ps(-0.0), _mm256_sub_ps(a, c)),
            );
            let ratio = _mm256_sub_ps(_mm256_div_ps(num, denom), one);
            if KIND == 1 {
                ratio
            } else {
                _mm256_sub_ps(_mm256_setzero_ps(), ratio)
            }
        };
        _mm256_max_ps(value, _mm256_setzero_ps())
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn reduce<const REPLACE: bool, const FOURTH: bool>(
    new: __m256,
    old: __m256,
    lane: usize,
    first: &mut [__m256d; 2],
    fourth: &mut [__m256d; 2],
) {
    let new = [
        _mm256_cvtps_pd(_mm256_castps256_ps128(new)),
        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(new)),
    ];
    let old = [
        _mm256_cvtps_pd(_mm256_castps256_ps128(old)),
        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(old)),
    ];
    for (half, (v, old)) in new.into_iter().zip(old).enumerate() {
        let delta = if REPLACE { _mm256_sub_pd(v, old) } else { v };
        first[lane + half] = _mm256_add_pd(first[lane + half], delta);
        if FOURTH {
            let square = _mm256_mul_pd(v, v);
            let mut delta = _mm256_mul_pd(square, square);
            if REPLACE {
                let square = _mm256_mul_pd(old, old);
                delta = _mm256_sub_pd(delta, _mm256_mul_pd(square, square));
            }
            fourth[lane + half] = _mm256_add_pd(fourth[lane + half], delta);
        }
    }
}

/// # Safety
/// AVX2 must be available. Only complete eight-pixel blocks are read.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ss2_sums_avx2(map: &[f32], sums: &mut MapSums) -> usize {
    let (blocks, _) = map.as_chunks::<8>();
    // Complete blocks fit the source, and the sums hold exactly eight f64 lanes.
    unsafe {
        let first_chunks = sums.first.as_chunks::<4>().0;
        let mut first: [__m256d; 2] =
            std::array::from_fn(|i| _mm256_loadu_pd(first_chunks[i].as_ptr()));
        let fourth_chunks = sums.fourth.as_chunks::<4>().0;
        let mut fourth: [__m256d; 2] =
            std::array::from_fn(|i| _mm256_loadu_pd(fourth_chunks[i].as_ptr()));
        for block in blocks {
            let value = _mm256_loadu_ps(block.as_ptr());
            reduce::<false, true>(value, _mm256_setzero_ps(), 0, &mut first, &mut fourth);
        }
        for (dst, value) in sums.first.as_chunks_mut::<4>().0.iter_mut().zip(first) {
            _mm256_storeu_pd(dst.as_mut_ptr(), value);
        }
        for (dst, value) in sums.fourth.as_chunks_mut::<4>().0.iter_mut().zip(fourth) {
            _mm256_storeu_pd(dst.as_mut_ptr(), value);
        }
    }
    blocks.len() * 8
}

/// # Safety
/// AVX2 must be available and all input/output slices must have equal lengths.
/// The signed edge ratio is computed once for both maps and their f64 deltas.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ss2_edges_avx2<
    const REPLACE: bool,
    const ARTIFACT_FOURTH: bool,
    const DETAIL_FOURTH: bool,
>(
    inputs: MapInputs<'_>,
    [artifact, detail]: [&mut [f32]; 2],
    sums: &mut [MapSums; 2],
) -> usize {
    let (artifact_blocks, _) = artifact.as_chunks_mut::<8>();
    let (detail_blocks, _) = detail.as_chunks_mut::<8>();
    let done = artifact_blocks.len() * 8;
    // Keep all eight reduction lanes grouped, including four-wide backends.
    unsafe {
        let af_chunks = sums[0].first.as_chunks::<4>().0;
        let mut af: [__m256d; 2] = std::array::from_fn(|i| _mm256_loadu_pd(af_chunks[i].as_ptr()));
        let a4_chunks = sums[0].fourth.as_chunks::<4>().0;
        let mut a4: [__m256d; 2] = std::array::from_fn(|i| _mm256_loadu_pd(a4_chunks[i].as_ptr()));
        let df_chunks = sums[1].first.as_chunks::<4>().0;
        let mut df: [__m256d; 2] = std::array::from_fn(|i| _mm256_loadu_pd(df_chunks[i].as_ptr()));
        let d4_chunks = sums[1].fourth.as_chunks::<4>().0;
        let mut d4: [__m256d; 2] = std::array::from_fn(|i| _mm256_loadu_pd(d4_chunks[i].as_ptr()));
        let one = _mm256_set1_ps(1.0);
        for (block, (artifact, detail)) in artifact_blocks.iter_mut().zip(detail_blocks).enumerate()
        {
            let i = block * 8;
            let [a, b, c, d] = [inputs[0], inputs[1], inputs[2], inputs[3]]
                .map(|p| _mm256_loadu_ps(p.as_ptr().add(i)));
            let ratio = _mm256_sub_ps(
                _mm256_div_ps(
                    _mm256_add_ps(
                        one,
                        _mm256_andnot_ps(_mm256_set1_ps(-0.0), _mm256_sub_ps(b, d)),
                    ),
                    _mm256_add_ps(
                        one,
                        _mm256_andnot_ps(_mm256_set1_ps(-0.0), _mm256_sub_ps(a, c)),
                    ),
                ),
                one,
            );
            let a = _mm256_max_ps(ratio, _mm256_setzero_ps());
            let d = _mm256_max_ps(
                _mm256_xor_ps(ratio, _mm256_set1_ps(-0.0)),
                _mm256_setzero_ps(),
            );
            if REPLACE {
                let lane = 0;
                reduce::<true, ARTIFACT_FOURTH>(
                    a,
                    _mm256_loadu_ps(artifact.as_ptr()),
                    lane,
                    &mut af,
                    &mut a4,
                );
                reduce::<true, DETAIL_FOURTH>(
                    d,
                    _mm256_loadu_ps(detail.as_ptr()),
                    lane,
                    &mut df,
                    &mut d4,
                );
            }
            _mm256_storeu_ps(artifact.as_mut_ptr(), a);
            _mm256_storeu_ps(detail.as_mut_ptr(), d);
        }
        if REPLACE {
            for (dst, value) in sums[0].first.as_chunks_mut::<4>().0.iter_mut().zip(af) {
                _mm256_storeu_pd(dst.as_mut_ptr(), value);
            }
            for (dst, value) in sums[1].first.as_chunks_mut::<4>().0.iter_mut().zip(df) {
                _mm256_storeu_pd(dst.as_mut_ptr(), value);
            }
            if ARTIFACT_FOURTH {
                for (dst, value) in sums[0].fourth.as_chunks_mut::<4>().0.iter_mut().zip(a4) {
                    _mm256_storeu_pd(dst.as_mut_ptr(), value);
                }
            }
            if DETAIL_FOURTH {
                for (dst, value) in sums[1].fourth.as_chunks_mut::<4>().0.iter_mut().zip(d4) {
                    _mm256_storeu_pd(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}
