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

//! 2×2 box downsampling of linear planes, with the reference's ceil sizes.

use std::sync::OnceLock;

type DownsampleRowsFn = fn(&[f32], &[f32], &mut [f32]) -> usize;

fn selected_downsample_fn() -> DownsampleRowsFn {
    static ROWS: OnceLock<DownsampleRowsFn> = OnceLock::new();
    *ROWS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            return |top, bottom, dst| unsafe {
                crate::avx::downsample_rows_avx2(top, bottom, dst)
            };
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return |top, bottom, dst| unsafe {
                crate::sse::downsample_rows_sse41(top, bottom, dst)
            };
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            |top, bottom, dst| unsafe { crate::neon::downsample_rows_neon(top, bottom, dst) }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        downsample_rows_scalar
    })
}

fn downsample_rows_scalar(top: &[f32], bottom: &[f32], dst: &mut [f32]) -> usize {
    for (x, dst) in dst.iter_mut().enumerate() {
        let i = 2 * x;
        *dst = 0.25 * (top[i] + top[i + 1] + bottom[i] + bottom[i + 1]);
    }
    dst.len()
}

/// One output row from two input rows. A final unpaired column repeats the
/// right edge when the destination requests ceil rather than floor width.
pub(crate) fn downsample_rows(top: &[f32], bottom: &[f32], dst: &mut [f32]) {
    assert_eq!(top.len(), bottom.len());
    assert!(dst.len() <= top.len().div_ceil(2));
    let pairs = dst.len().min(top.len() / 2);
    let done = selected_downsample_fn()(top, bottom, &mut dst[..pairs]);
    downsample_rows_scalar(&top[2 * done..], &bottom[2 * done..], &mut dst[done..pairs]);
    if pairs < dst.len() {
        let i = 2 * pairs;
        dst[pairs] = 0.25 * (top[i] + top[i] + bottom[i] + bottom[i]);
    }
}

/// Whole-plane 2×2 box downsample with edge replication (ceil dimensions).
pub(crate) fn downsample(a: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0f32; nw * nh];
    for y in 0..nh {
        let (y0, y1) = ((2 * y).min(h - 1), (2 * y + 1).min(h - 1));
        downsample_rows(
            &a[y0 * w..(y0 + 1) * w],
            &a[y1 * w..(y1 + 1) * w],
            &mut out[y * nw..(y + 1) * nw],
        );
    }
    (out, nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scalar_for_odd_sizes() {
        for w in (1..=40).chain([63, 64, 65, 129]) {
            for h in [1, 2, 3, 9] {
                let a: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 101) as f32 / 7.0).collect();
                let (out, nw, nh) = downsample(&a, w, h);
                for y in 0..nh {
                    for x in 0..nw {
                        let at = |xx: usize, yy: usize| a[yy.min(h - 1) * w + xx.min(w - 1)];
                        let expected = 0.25
                            * (at(2 * x, 2 * y)
                                + at(2 * x + 1, 2 * y)
                                + at(2 * x, 2 * y + 1)
                                + at(2 * x + 1, 2 * y + 1));
                        assert_eq!(out[y * nw + x].to_bits(), expected.to_bits(), "{w}x{h}");
                    }
                }
            }
        }
    }
}
