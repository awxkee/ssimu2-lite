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

//! Linear sRGB → "positive XYB", as the reference converts every scale.

use std::sync::OnceLock;

/// libjxl's opsin absorbance matrix (rows: L, M, S mixes of linear R, G, B).
pub(crate) const OPSIN: [f32; 9] = [
    0.30,
    0.622,
    0.078,
    0.23,
    0.692,
    0.078,
    0.243_422_69,
    0.204_767_45,
    0.551_809_86,
];
pub(crate) const OPSIN_BIAS: f32 = 0.003_793_073_4;
pub(crate) const NEG_BIAS_CBRT: f32 = -0.155_954_2;

#[inline(always)]
fn halley_refine(x: f32, a: f32) -> f32 {
    let tx = x * x * x;
    x * ((2.0 * a + tx) / (2.0 * tx + a))
}

/// Cube root of a nonnegative value: bit-trick seed plus two Halley steps.
#[inline]
fn cbrt_positive(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let seed = f32::from_bits((x.to_bits() & 0x7fff_ffff) / 3 + 709_958_130);
    halley_refine(halley_refine(seed, x), x)
}

#[inline]
fn opsin_pixel(r: f32, g: f32, b: f32) -> [f32; 3] {
    let m = &OPSIN;
    let mix = |row: usize| {
        let v = m[row * 3] * r + (m[row * 3 + 1] * g + (m[row * 3 + 2] * b + OPSIN_BIAS));
        cbrt_positive(v.max(0.0)) + NEG_BIAS_CBRT
    };
    let (l, m_, s) = (mix(0), mix(1), mix(2));
    [0.5 * (l - m_), 0.5 * (l + m_), s]
}

/// Row kernels convert to positive XYB and return how many pixels they handled.
pub(crate) type XybRowsFn = unsafe fn([&[f32]; 3], [&mut [f32]; 3]) -> usize;

#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    rows: XybRowsFn,
}

impl Kernels {
    pub(crate) fn new() -> Self {
        Self {
            rows: selected_xyb_rows(),
        }
    }
}

fn xyb_rows_scalar([r, g, b]: [&[f32]; 3], [x, y, out_b]: [&mut [f32]; 3]) -> usize {
    for i in 0..x.len() {
        let [px, py, pb] = opsin_pixel(r[i], g[i], b[i]);
        x[i] = px * 14.0 + 0.42;
        y[i] = py + 0.01;
        out_b[i] = (pb - py) + 0.55;
    }
    x.len()
}

fn selected_xyb_rows() -> XybRowsFn {
    static ROWS: OnceLock<XybRowsFn> = OnceLock::new();
    *ROWS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return crate::avx::xyb_rows_avx2;
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return crate::sse::xyb_rows_sse41;
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            crate::neon::xyb_rows_neon
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        xyb_rows_scalar
    })
}

impl Kernels {
    /// Linear RGB → positive XYB (`X·14 + 0.42`, `Y + 0.01`, `B − Y + 0.55`).
    pub(crate) fn positive_xyb_into(self, lin: [&[f32]; 3], [x, y, b]: [&mut [f32]; 3]) {
        let n = x.len();
        assert!(lin.iter().all(|p| p.len() == n));
        assert_eq!(y.len(), n);
        assert_eq!(b.len(), n);
        // SAFETY: dispatch checked the CPU features and all six slices have `n`
        // elements; the kernel touches complete vectors only.
        let done = unsafe { (self.rows)(lin, [&mut *x, &mut *y, &mut *b]) };
        xyb_rows_scalar(
            lin.map(|p| &p[done..]),
            [&mut x[done..], &mut y[done..], &mut b[done..]],
        );
    }

    pub(crate) fn positive_xyb_of_linear(self, lin: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
        let n = lin[0].len();
        let mut out = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
        let [x, y, b] = &mut out;
        self.positive_xyb_into([&lin[0], &lin[1], &lin[2]], [x, y, b]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_rows_match_scalar_rows() {
        let kernels = Kernels::new();
        for n in (0..=40).chain([127, 255, 1000]) {
            let lin: [Vec<f32>; 3] = std::array::from_fn(|c| {
                (0..n)
                    .map(|i| ((i * (7 + c) + 3 * c) % 101) as f32 / 97.0 - 0.02)
                    .collect()
            });
            let input = [&lin[0][..], &lin[1][..], &lin[2][..]];
            let mut simd: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; n]);
            let mut scalar = simd.clone();
            let [a, b, c] = &mut simd;
            kernels.positive_xyb_into(input, [a, b, c]);
            let [a, b, c] = &mut scalar;
            xyb_rows_scalar(input, [&mut a[..], &mut b[..], &mut c[..]]);
            for (s, v) in simd.iter().zip(&scalar) {
                for (p, q) in s.iter().zip(v) {
                    assert!((p - q).abs() <= 1e-5, "{n}: {p} vs {q}");
                }
            }
        }
    }

    #[test]
    fn cube_root_is_accurate() {
        for i in 1..2000 {
            let x = i as f32 * 0.0007;
            let c = cbrt_positive(x);
            let error = (c - x.cbrt()).abs() / x.cbrt();
            assert!(error <= 1e-6, "{x}: {c} vs {}, {error}", x.cbrt());
        }
    }
}
