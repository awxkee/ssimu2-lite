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

//! Shared fixtures for unit tests.

use crate::{IncrementalState, PreparedReference, RgbImage};

/// Linear RGB planes with an image-like API.
#[derive(Clone)]
pub(crate) struct Planes {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) p: [Vec<f32>; 3],
}

impl Planes {
    /// A checkerboard with noise, green-dominant.
    pub(crate) fn textured(seed: u32, w: usize, h: usize) -> Self {
        let mut p: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; w * h]);
        let mut rng = seed;
        for (c, plane) in p.iter_mut().enumerate() {
            for y in 0..h {
                for x in 0..w {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    let noise = (rng >> 24) as f32 / 255.0;
                    let gain = if c == 1 { 1.0 } else { 0.4 };
                    plane[y * w + x] =
                        0.1 + 0.3 * ((x / 8 + y / 8) % 2) as f32 * gain + 0.05 * noise;
                }
            }
        }
        Self { w, h, p }
    }

    pub(crate) fn image(&self) -> RgbImage<'_> {
        RgbImage::linear_rgb([&self.p[0], &self.p[1], &self.p[2]], self.w, self.h, self.w).unwrap()
    }

    pub(crate) fn plane_row(&self, c: usize, y: usize) -> &[f32] {
        &self.p[c][y * self.w..(y + 1) * self.w]
    }

    pub(crate) fn plane_row_mut(&mut self, c: usize, y: usize) -> &mut [f32] {
        &mut self.p[c][y * self.w..(y + 1) * self.w]
    }
}

/// `100 - score` and the error map of a full evaluation.
pub(crate) fn full(src: &Planes, rec: &Planes) -> (f64, Vec<f32>) {
    let (score, map) = PreparedReference::new(src.image())
        .unwrap()
        .compare_with_map(rec.image())
        .unwrap();
    (100.0 - score, map)
}

pub(crate) fn state_of(src: &Planes, rec: &Planes) -> IncrementalState {
    PreparedReference::new(src.image())
        .unwrap()
        .incremental(rec.image())
        .unwrap()
}

/// SIMD CI jobs must exercise their backend instead of silently falling back
/// to scalar when the runner lacks a CPU feature or a Cargo feature is missing.
#[test]
fn required_simd_backend_is_available() {
    let Some(required) = std::env::var_os("SSIMU2_REQUIRE_SIMD") else {
        return;
    };
    let required = required
        .to_str()
        .expect("SSIMU2_REQUIRE_SIMD must be UTF-8");
    let available = match required {
        "avx2" => {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
            }
            #[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
            {
                false
            }
        }
        "sse4.1" => {
            #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
            {
                std::is_x86_feature_detected!("sse4.1")
            }
            #[cfg(not(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse")))]
            {
                false
            }
        }
        "neon" => cfg!(all(target_arch = "aarch64", feature = "neon")),
        _ => panic!("unknown SIMD backend requested: {required}"),
    };
    assert!(
        available,
        "required backend {required} is unavailable on {} (avx={}, sse={}, neon={}); \
         AVX2 color/contribution kernels also require FMA",
        std::env::consts::ARCH,
        cfg!(feature = "avx"),
        cfg!(feature = "sse"),
        cfg!(feature = "neon"),
    );
}
