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
use std::arch::aarch64::*;

/// # Safety
/// NEON must be available and the input/output slices must have equal lengths.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ss2_contribution_neon(
    map: &[f32],
    p: MapContribution,
    acc: &mut [f32],
) -> usize {
    debug_assert_eq!(map.len(), acc.len());
    if p.norm_weight == 0.0 {
        let done = unsafe { contribution::<4, false>(map, p, acc) };
        done + unsafe { contribution::<1, false>(&map[done..], p, &mut acc[done..]) }
    } else {
        let done = unsafe { contribution::<4, true>(map, p, acc) };
        done + unsafe { contribution::<1, true>(&map[done..], p, &mut acc[done..]) }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn contribution<const N: usize, const FOURTH: bool>(
    map: &[f32],
    p: MapContribution,
    acc: &mut [f32],
) -> usize {
    let done = acc.len() / (4 * N) * (4 * N);
    for start in (0..done).step_by(4 * N) {
        for lane in 0..N {
            let i = start + 4 * lane;
            // The entry point's equal-length contract covers complete vectors.
            unsafe {
                let v = vld1q_f32(map.as_ptr().add(i));
                let old = vld1q_f32(acc.as_ptr().add(i));
                let out = if FOURTH {
                    let normalized = vmulq_n_f32(v, p.inv_norm);
                    let square = vmulq_f32(normalized, normalized);
                    let weighted = vmulq_n_f32(square, p.norm_weight);
                    let first = vmulq_n_f32(v, p.linear);
                    vaddq_f32(old, vfmaq_f32(first, weighted, square))
                } else {
                    vfmaq_n_f32(old, v, p.linear)
                };
                vst1q_f32(acc.as_mut_ptr().add(i), out);
            }
        }
    }
    done
}

/// # Safety
/// NEON must be available, STEP must be 1/2/4/8/16/32, and src must contain
/// ceil(dst.len()/STEP) pixels. Only complete vector-sized runs are processed.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ss2_expand_row_neon<const STEP: usize>(
    src: &[f32],
    dst: &mut [f32],
    share: f32,
) -> usize {
    let span = if STEP == 1 {
        4
    } else if STEP == 2 {
        8
    } else {
        STEP
    };
    let done = dst.len() / span * span;
    for i in (0..done).step_by(span) {
        // Complete runs guarantee every input pixel and output vector exists.
        unsafe {
            if STEP == 1 {
                let value = vld1q_f32(src.as_ptr().add(i));
                let old = vld1q_f32(dst.as_ptr().add(i));
                vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_n_f32(old, value, share));
            } else if STEP == 2 {
                let v = vld1q_f32(src.as_ptr().add(i / 2));
                let a = vzip1q_f32(v, v);
                let b = vzip2q_f32(v, v);
                let old_a = vld1q_f32(dst.as_ptr().add(i));
                let old_b = vld1q_f32(dst.as_ptr().add(i + 4));
                vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_n_f32(old_a, a, share));
                vst1q_f32(dst.as_mut_ptr().add(i + 4), vfmaq_n_f32(old_b, b, share));
            } else {
                let value = vdupq_n_f32(*src.get_unchecked(i / STEP));
                for lane in 0..STEP / 4 {
                    let offset = i + 4 * lane;
                    let old = vld1q_f32(dst.as_ptr().add(offset));
                    vst1q_f32(dst.as_mut_ptr().add(offset), vfmaq_n_f32(old, value, share));
                }
            }
        }
    }
    done
}
