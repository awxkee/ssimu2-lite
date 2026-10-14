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
use crate::metric::MapContribution;
use std::arch::x86_64::*;

/// # Safety
/// AVX2/FMA must be available and the input/output slices must have equal lengths.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn ss2_contribution_avx2(
    map: &[f32],
    p: MapContribution,
    acc: &mut [f32],
) -> usize {
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
    let done = acc.len() / (8 * N) * (8 * N);
    let linear = _mm256_set1_ps(p.linear);
    let inv_norm = _mm256_set1_ps(p.inv_norm);
    let norm_weight = _mm256_set1_ps(p.norm_weight);
    for start in (0..done).step_by(8 * N) {
        for lane in 0..N {
            let i = start + 8 * lane;
            // The entry point's equal-length contract covers complete vectors.
            unsafe {
                let v = _mm256_loadu_ps(map.as_ptr().add(i));
                let old = _mm256_loadu_ps(acc.as_ptr().add(i));
                let out = if FOURTH {
                    let normalized = _mm256_mul_ps(v, inv_norm);
                    let square = _mm256_mul_ps(normalized, normalized);
                    let weighted = _mm256_mul_ps(square, norm_weight);
                    let first = _mm256_mul_ps(v, linear);
                    _mm256_add_ps(old, _mm256_fmadd_ps(weighted, square, first))
                } else {
                    _mm256_fmadd_ps(v, linear, old)
                };
                _mm256_storeu_ps(acc.as_mut_ptr().add(i), out);
            }
        }
    }
    done
}

/// # Safety
/// AVX2/FMA must be available, STEP must be 1/2/4/8/16/32, and src must contain
/// ceil(dst.len()/STEP) pixels. Only complete vector-sized runs are processed.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn ss2_expand_row_avx2<const STEP: usize>(
    src: &[f32],
    dst: &mut [f32],
    share: f32,
) -> usize {
    let span = if STEP == 2 { 16 } else { STEP.max(8) };
    let done = dst.len() / span * span;
    let factor = _mm256_set1_ps(share);
    for i in (0..done).step_by(span) {
        // Complete runs guarantee every input pixel and output vector exists.
        unsafe {
            if STEP == 1 {
                let v = _mm256_loadu_ps(src.as_ptr().add(i));
                let old = _mm256_loadu_ps(dst.as_ptr().add(i));
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(v, factor, old));
            } else if STEP == 2 {
                let v = _mm256_loadu_ps(src.as_ptr().add(i / 2));
                let lo = _mm256_unpacklo_ps(v, v);
                let hi = _mm256_unpackhi_ps(v, v);
                let a = _mm256_permute2f128_ps::<0x20>(lo, hi);
                let b = _mm256_permute2f128_ps::<0x31>(lo, hi);
                let old_a = _mm256_loadu_ps(dst.as_ptr().add(i));
                let old_b = _mm256_loadu_ps(dst.as_ptr().add(i + 8));
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(a, factor, old_a));
                _mm256_storeu_ps(
                    dst.as_mut_ptr().add(i + 8),
                    _mm256_fmadd_ps(b, factor, old_b),
                );
            } else if STEP == 4 {
                let a = _mm_set1_ps(*src.get_unchecked(i / 4));
                let b = _mm_set1_ps(*src.get_unchecked(i / 4 + 1));
                let v = _mm256_set_m128(b, a);
                let old = _mm256_loadu_ps(dst.as_ptr().add(i));
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(v, factor, old));
            } else {
                let v = _mm256_set1_ps(*src.get_unchecked(i / STEP));
                for lane in 0..STEP / 8 {
                    let offset = i + 8 * lane;
                    let old = _mm256_loadu_ps(dst.as_ptr().add(offset));
                    _mm256_storeu_ps(
                        dst.as_mut_ptr().add(offset),
                        _mm256_fmadd_ps(v, factor, old),
                    );
                }
            }
        }
    }
    done
}
