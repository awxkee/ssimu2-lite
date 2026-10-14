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

use crate::xyb::{NEG_BIAS_CBRT, OPSIN, OPSIN_BIAS};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn halley(x: float32x4_t, a: float32x4_t) -> float32x4_t {
    let tx = vmulq_f32(vmulq_f32(x, x), x);
    let num = vfmaq_n_f32(tx, a, 2.0);
    let den = vfmaq_n_f32(a, tx, 2.0);
    vmulq_f32(x, vdivq_f32(num, den))
}

/// Nonnegative cube root: `hx·341 >> 10` seed (≈ /3), two Halley steps.
#[inline]
#[target_feature(enable = "neon")]
fn cbrt_positive(a: float32x4_t) -> float32x4_t {
    let hx = vreinterpretq_u32_f32(a);
    let scale = vdupq_n_u32(341);
    let hi = vshrq_n_u64::<10>(vmull_high_u32(hx, scale));
    let lo = vshrq_n_u64::<10>(vmull_u32(vget_low_u32(hx), vget_low_u32(scale)));
    let third = vcombine_u32(vmovn_u64(lo), vmovn_u64(hi));
    let seed = vreinterpretq_f32_u32(vaddq_u32(third, vdupq_n_u32(709_958_130)));
    let x = halley(halley(seed, a), a);
    let zero = vdupq_n_f32(0.0);
    vbslq_f32(vceqq_f32(a, zero), zero, x)
}

#[inline]
#[target_feature(enable = "neon")]
fn mix(row: usize, r: float32x4_t, g: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut v = vfmaq_n_f32(vdupq_n_f32(OPSIN_BIAS), b, OPSIN[row * 3 + 2]);
    v = vfmaq_n_f32(v, g, OPSIN[row * 3 + 1]);
    v = vfmaq_n_f32(v, r, OPSIN[row * 3]);
    let v = cbrt_positive(vmaxq_f32(v, vdupq_n_f32(0.0)));
    vaddq_f32(v, vdupq_n_f32(NEG_BIAS_CBRT))
}

/// One vector of plain XYB from `lanes` pixels at the given pointers.
///
/// # Safety
/// Each pointer must be valid for 4 reads/writes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn convert(input: [*const f32; 3], output: [*mut f32; 3]) {
    // The caller guarantees complete vectors behind every pointer.
    unsafe {
        let [r, g, b] = input.map(|p| vld1q_f32(p));
        let l = mix(0, r, g, b);
        let m = mix(1, r, g, b);
        let s = mix(2, r, g, b);
        vst1q_f32(output[0], vmulq_n_f32(vsubq_f32(l, m), 0.5));
        vst1q_f32(output[1], vmulq_n_f32(vaddq_f32(l, m), 0.5));
        vst1q_f32(output[2], s);
    }
}

/// Converts every pixel, the tail through a padded vector, so that any
/// pixel's result is independent of its position in the row.
///
/// # Safety
/// The CPU features must be available and all six slices must have equal lengths.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xyb_rows_neon(input: [&[f32]; 3], [x, y, out_b]: [&mut [f32]; 3]) -> usize {
    let n = x.len();
    let done = n / 4 * 4;
    for i in (0..done).step_by(4) {
        // Every slice holds `n` elements, so each complete vector is in bounds.
        unsafe {
            convert(
                input.map(|p| p.as_ptr().add(i)),
                [
                    x.as_mut_ptr().add(i),
                    y.as_mut_ptr().add(i),
                    out_b.as_mut_ptr().add(i),
                ],
            );
        }
    }
    if done < n {
        let tail = n - done;
        let src: [[f32; 4]; 3] = std::array::from_fn(|c| {
            let mut v = [0f32; 4];
            v[..tail].copy_from_slice(&input[c][done..]);
            v
        });
        let mut dst = [[0f32; 4]; 3];
        let [dx, dy, db] = &mut dst;
        // Stack buffers hold one complete vector each.
        unsafe {
            convert(
                src.each_ref().map(|v| v.as_ptr()),
                [dx.as_mut_ptr(), dy.as_mut_ptr(), db.as_mut_ptr()],
            );
        }
        x[done..].copy_from_slice(&dst[0][..tail]);
        y[done..].copy_from_slice(&dst[1][..tail]);
        out_b[done..].copy_from_slice(&dst[2][..tail]);
    }
    n
}
