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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn halley(x: __m128, a: __m128) -> __m128 {
    let tx = _mm_mul_ps(_mm_mul_ps(x, x), x);
    let two = _mm_set1_ps(2.0);
    let num = _mm_add_ps(tx, _mm_mul_ps(a, two));
    let den = _mm_add_ps(a, _mm_mul_ps(tx, two));
    _mm_mul_ps(x, _mm_div_ps(num, den))
}

/// Nonnegative cube root: `hx·341 >> 10` seed (≈ /3), two Halley steps.
#[inline]
#[target_feature(enable = "sse4.1")]
fn cbrt_positive(a: __m128) -> __m128 {
    let hx = _mm_and_si128(_mm_castps_si128(a), _mm_set1_epi32(0x7fff_ffff));
    let scale = _mm_set1_epi32(341);
    let even = _mm_srli_epi64::<10>(_mm_mul_epu32(hx, scale));
    let odd = _mm_slli_epi64::<32>(_mm_srli_epi64::<10>(_mm_mul_epu32(
        _mm_srli_epi64::<32>(hx),
        scale,
    )));
    let seed = _mm_add_epi32(_mm_or_si128(even, odd), _mm_set1_epi32(709_958_130));
    let x = halley(halley(_mm_castsi128_ps(seed), a), a);
    let zero = _mm_setzero_ps();
    _mm_blendv_ps(x, zero, _mm_cmpeq_ps(a, zero))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn mix(row: usize, r: __m128, g: __m128, b: __m128) -> __m128 {
    let mut v = _mm_add_ps(
        _mm_set1_ps(OPSIN_BIAS),
        _mm_mul_ps(b, _mm_set1_ps(OPSIN[row * 3 + 2])),
    );
    v = _mm_add_ps(v, _mm_mul_ps(g, _mm_set1_ps(OPSIN[row * 3 + 1])));
    v = _mm_add_ps(v, _mm_mul_ps(r, _mm_set1_ps(OPSIN[row * 3])));
    let v = cbrt_positive(_mm_max_ps(v, _mm_setzero_ps()));
    _mm_add_ps(v, _mm_set1_ps(NEG_BIAS_CBRT))
}

/// One vector of positive XYB from `lanes` pixels at the given pointers.
///
/// # Safety
/// Each pointer must be valid for 4 reads/writes.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn convert(input: [*const f32; 3], output: [*mut f32; 3]) {
    let half = _mm_set1_ps(0.5);
    // The caller guarantees complete vectors behind every pointer.
    unsafe {
        let [r, g, b] = input.map(|p| _mm_loadu_ps(p));
        let l = mix(0, r, g, b);
        let m = mix(1, r, g, b);
        let s = mix(2, r, g, b);
        let x = _mm_mul_ps(_mm_sub_ps(l, m), half);
        let y = _mm_mul_ps(_mm_add_ps(l, m), half);
        _mm_storeu_ps(
            output[0],
            _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(14.0)), _mm_set1_ps(0.42)),
        );
        _mm_storeu_ps(output[1], _mm_add_ps(y, _mm_set1_ps(0.01)));
        _mm_storeu_ps(output[2], _mm_add_ps(_mm_sub_ps(s, y), _mm_set1_ps(0.55)));
    }
}

/// Converts every pixel with the same vector arithmetic. The tail uses an
/// overlapping vector when it fits or a padded vector for very short rows.
///
/// # Safety
/// The CPU features must be available and all six slices must have equal lengths.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn xyb_rows_sse41(input: [&[f32]; 3], [x, y, out_b]: [&mut [f32]; 3]) -> usize {
    let n = x.len();
    if n >= 4 {
        let inputs = input.map(|p| p.as_ptr());
        let outputs = [x.as_mut_ptr(), y.as_mut_ptr(), out_b.as_mut_ptr()];
        let mut i = 0;
        // Each iteration covers two complete vectors. The last one or two
        // calls also stay within the row, rewriting a few lanes if needed.
        unsafe {
            while n - i >= 8 {
                convert(inputs.map(|p| p.add(i)), outputs.map(|p| p.add(i)));
                convert(inputs.map(|p| p.add(i + 4)), outputs.map(|p| p.add(i + 4)));
                i += 8;
            }
            if n - i >= 4 {
                convert(inputs.map(|p| p.add(i)), outputs.map(|p| p.add(i)));
                i += 4;
            }
            if i < n {
                let last = n - 4;
                convert(inputs.map(|p| p.add(last)), outputs.map(|p| p.add(last)));
            }
        }
    } else if n > 0 {
        let src: [[f32; 4]; 3] = std::array::from_fn(|c| {
            let mut v = [0f32; 4];
            v[..n].copy_from_slice(input[c]);
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
        x.copy_from_slice(&dst[0][..n]);
        y.copy_from_slice(&dst[1][..n]);
        out_b.copy_from_slice(&dst[2][..n]);
    }
    n
}
