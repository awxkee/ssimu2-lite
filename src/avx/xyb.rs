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
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn halley(x: __m256, a: __m256) -> __m256 {
    let tx = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    let two = _mm256_set1_ps(2.0);
    let num = _mm256_fmadd_ps(a, two, tx);
    let den = _mm256_fmadd_ps(tx, two, a);
    _mm256_mul_ps(x, _mm256_div_ps(num, den))
}

/// Nonnegative cube root: `hx·341 >> 10` seed (≈ /3), two Halley steps.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn cbrt_positive(a: __m256) -> __m256 {
    let hx = _mm256_and_si256(_mm256_castps_si256(a), _mm256_set1_epi32(0x7fff_ffff));
    let scale = _mm256_set1_epi32(341);
    let even = _mm256_srli_epi64::<10>(_mm256_mul_epu32(hx, scale));
    let odd = _mm256_slli_epi64::<32>(_mm256_srli_epi64::<10>(_mm256_mul_epu32(
        _mm256_srli_epi64::<32>(hx),
        scale,
    )));
    let seed = _mm256_add_epi32(_mm256_or_si256(even, odd), _mm256_set1_epi32(709_958_130));
    let x = halley(halley(_mm256_castsi256_ps(seed), a), a);
    let zero = _mm256_setzero_ps();
    _mm256_blendv_ps(x, zero, _mm256_cmp_ps::<_CMP_EQ_OQ>(a, zero))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn mix(row: usize, r: __m256, g: __m256, b: __m256) -> __m256 {
    let mut v = _mm256_fmadd_ps(
        b,
        _mm256_set1_ps(OPSIN[row * 3 + 2]),
        _mm256_set1_ps(OPSIN_BIAS),
    );
    v = _mm256_fmadd_ps(g, _mm256_set1_ps(OPSIN[row * 3 + 1]), v);
    v = _mm256_fmadd_ps(r, _mm256_set1_ps(OPSIN[row * 3]), v);
    let v = cbrt_positive(_mm256_max_ps(v, _mm256_setzero_ps()));
    _mm256_add_ps(v, _mm256_set1_ps(NEG_BIAS_CBRT))
}

/// One vector of positive XYB from `lanes` pixels at the given pointers.
///
/// # Safety
/// Each pointer must be valid for 8 reads/writes.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn convert(input: [*const f32; 3], output: [*mut f32; 3]) {
    let half = _mm256_set1_ps(0.5);
    // The caller guarantees complete vectors behind every pointer.
    unsafe {
        let [r, g, b] = input.map(|p| _mm256_loadu_ps(p));
        let l = mix(0, r, g, b);
        let m = mix(1, r, g, b);
        let s = mix(2, r, g, b);
        let x = _mm256_mul_ps(_mm256_sub_ps(l, m), half);
        let y = _mm256_mul_ps(_mm256_add_ps(l, m), half);
        _mm256_storeu_ps(
            output[0],
            _mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(14.0)), _mm256_set1_ps(0.42)),
        );
        _mm256_storeu_ps(output[1], _mm256_add_ps(y, _mm256_set1_ps(0.01)));
        _mm256_storeu_ps(
            output[2],
            _mm256_add_ps(_mm256_sub_ps(s, y), _mm256_set1_ps(0.55)),
        );
    }
}

/// Converts two vectors per iteration. A final overlapping vector handles
/// partial tails, with a padded vector for inputs shorter than one vector.
///
/// # Safety
/// The CPU features must be available and all six slices must have equal lengths.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn xyb_rows_avx2(input: [&[f32]; 3], mut output: [&mut [f32]; 3]) -> usize {
    const LANES: usize = 8;
    const BLOCK: usize = 2 * LANES;
    let n = output[0].len();
    if n >= LANES {
        let inputs = input.map(|p| p.as_ptr());
        let outputs = output.each_mut().map(|p| p.as_mut_ptr());
        let mut i = 0;
        // SAFETY: all six slices have length n. The last call may rewrite
        // earlier lanes, but every read and write stays inside its slice.
        unsafe {
            while n - i >= BLOCK {
                convert(inputs.map(|p| p.add(i)), outputs.map(|p| p.add(i)));
                convert(
                    inputs.map(|p| p.add(i + LANES)),
                    outputs.map(|p| p.add(i + LANES)),
                );
                i += BLOCK;
            }
            if n - i >= LANES {
                convert(inputs.map(|p| p.add(i)), outputs.map(|p| p.add(i)));
                i += LANES;
            }
            if i < n {
                let last = n - LANES;
                convert(inputs.map(|p| p.add(last)), outputs.map(|p| p.add(last)));
            }
        }
    } else if n != 0 {
        let src: [[f32; LANES]; 3] = input.map(|p| {
            let mut v = [0.0; LANES];
            v[..n].copy_from_slice(p);
            v
        });
        let mut dst = [[0.0; LANES]; 3];
        // SAFETY: each stack buffer holds a complete vector.
        unsafe {
            convert(
                src.each_ref().map(|v| v.as_ptr()),
                dst.each_mut().map(|v| v.as_mut_ptr()),
            );
        }
        for (plane, tail) in output.into_iter().zip(&dst) {
            plane.copy_from_slice(&tail[..n]);
        }
    }
    n
}
