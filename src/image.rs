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

use crate::{Error, Rect};
use std::sync::OnceLock;

#[derive(Clone, Copy)]
enum Pixels<'a> {
    Srgb8(&'a [u8]),
    Srgb16(&'a [u16]),
    SrgbF32(&'a [f32]),
    Linear([&'a [f32]; 3]),
}

/// A borrowed RGB image without alpha. Composite alpha and convert ICC color
/// profiles to sRGB before constructing this view. Strides are in samples.
#[derive(Clone, Copy)]
pub struct RgbImage<'a> {
    pixels: Pixels<'a>,
    width: usize,
    height: usize,
    stride: usize,
}

impl<'a> RgbImage<'a> {
    /// Interleaved, gamma-encoded sRGB bytes. Row padding is allowed.
    pub fn srgb8(
        pixels: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 3, &[pixels.len()])?;
        Ok(Self::new(Pixels::Srgb8(pixels), width, height, stride))
    }

    /// Interleaved, gamma-encoded 16-bit sRGB samples.
    pub fn srgb16(
        pixels: &'a [u16],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 3, &[pixels.len()])?;
        Ok(Self::new(Pixels::Srgb16(pixels), width, height, stride))
    }

    /// Interleaved, gamma-encoded sRGB floats, nominally in `[0,1]`. Values
    /// outside that range are extended, not clipped; non-finite ones are errors.
    pub fn srgb_f32(
        pixels: &'a [f32],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 3, &[pixels.len()])?;
        Ok(Self::new(Pixels::SrgbF32(pixels), width, height, stride))
    }

    /// Three linear sRGB planes sharing one stride, 1.0 being display white.
    /// Values are used as given, without clipping.
    pub fn linear_rgb(
        planes: [&'a [f32]; 3],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 1, &planes.map(|p| p.len()))?;
        Ok(Self::new(Pixels::Linear(planes), width, height, stride))
    }

    fn new(pixels: Pixels<'a>, width: usize, height: usize, stride: usize) -> Self {
        Self {
            pixels,
            width,
            height,
            stride,
        }
    }

    /// Image width, in pixels.
    pub fn width(self) -> usize {
        self.width
    }

    /// Image height, in pixels.
    pub fn height(self) -> usize {
        self.height
    }

    /// Linear planes of the whole image.
    pub(crate) fn to_linear(self) -> Result<[Vec<f32>; 3], Error> {
        let n = self.width * self.height;
        let mut out = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
        let [r, g, b] = &mut out;
        self.linear_into(
            Rect::new(0, 0, self.width, self.height),
            [r, g, b],
            self.width,
        )?;
        Ok(out)
    }

    /// Converts `rect` (inside the image) into planes with `out_stride`,
    /// at the same coordinates. Nothing is written if a sample is non-finite.
    pub(crate) fn linear_into(
        self,
        rect: Rect,
        [r, g, b]: [&mut [f32]; 3],
        out_stride: usize,
    ) -> Result<(), Error> {
        debug_assert!(rect.x1 <= self.width && rect.y1 <= self.height);
        let rows = rect.y0..rect.y1;
        let span = |y: usize, step: usize| {
            y * self.stride + rect.x0 * step..y * self.stride + rect.x1 * step
        };
        let non_finite = match self.pixels {
            Pixels::SrgbF32(p) => rows.clone().any(|y| has_non_finite(&p[span(y, 3)])),
            Pixels::Linear(p) => rows
                .clone()
                .any(|y| p.iter().any(|p| has_non_finite(&p[span(y, 1)]))),
            _ => false,
        };
        if non_finite {
            return Err(Error::NonFiniteSample);
        }
        for y in rows {
            let o = y * out_stride + rect.x0..y * out_stride + rect.x1;
            let (r, g, b) = (&mut r[o.clone()], &mut g[o.clone()], &mut b[o]);
            match self.pixels {
                Pixels::Srgb8(p) => {
                    let lut = srgb8_table();
                    interleaved(&p[span(y, 3)], [r, g, b], |v| lut[v as usize]);
                }
                Pixels::Srgb16(p) => {
                    let lut = srgb16_table();
                    interleaved(&p[span(y, 3)], [r, g, b], |v| lut[v as usize]);
                }
                Pixels::SrgbF32(p) => interleaved(&p[span(y, 3)], [r, g, b], srgb_to_linear),
                Pixels::Linear(p) => {
                    for (out, src) in [r, g, b].into_iter().zip(p) {
                        out.copy_from_slice(&src[span(y, 1)]);
                    }
                }
            }
        }
        Ok(())
    }
}

fn has_non_finite(pixels: &[f32]) -> bool {
    type CheckFn = unsafe fn(&[f32]) -> bool;
    static CHECK: OnceLock<CheckFn> = OnceLock::new();
    let check = CHECK.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            return crate::avx::has_non_finite_avx2;
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::is_x86_feature_detected!("sse4.1") {
            return crate::sse::has_non_finite_sse41;
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            crate::neon::has_non_finite_neon
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        |p| p.iter().any(|v| !v.is_finite())
    });
    // SAFETY: dispatch checked the CPU requirements; kernels respect the slice bounds.
    unsafe { check(pixels) }
}

fn interleaved<T: Copy>(src: &[T], [r, g, b]: [&mut [f32]; 3], f: impl Fn(T) -> f32) {
    for (i, px) in src.as_chunks::<3>().0.iter().enumerate() {
        r[i] = f(px[0]);
        g[i] = f(px[1]);
        b[i] = f(px[2]);
    }
}

/// The sRGB transfer function, extended antisymmetrically outside `[0,1]`.
fn srgb_to_linear(v: f32) -> f32 {
    let a = (v as f64).abs();
    let l = if a <= 0.04045 {
        a / 12.92
    } else {
        ((a + 0.055) / 1.055).powf(2.4)
    };
    (l as f32).copysign(v)
}

fn srgb8_table() -> &'static [f32; 256] {
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
}

fn srgb16_table() -> &'static [f32] {
    static T: OnceLock<Vec<f32>> = OnceLock::new();
    T.get_or_init(|| {
        (0..=u16::MAX)
            .map(|i| srgb_to_linear(i as f32 / 65535.0))
            .collect()
    })
}

fn validate(
    width: usize,
    height: usize,
    stride: usize,
    channels: usize,
    lengths: &[usize],
) -> Result<(), Error> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions);
    }
    let row = width
        .checked_mul(channels)
        .ok_or(Error::InvalidDimensions)?;
    if stride < row {
        return Err(Error::InvalidLayout);
    }
    let needed = (height - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(row))
        .ok_or(Error::InvalidDimensions)?;
    if lengths.iter().any(|&len| len < needed) {
        return Err(Error::InvalidLayout);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_finite_checks_cover_every_position_and_tail() {
        #[allow(unused_mut)]
        let mut checks: Vec<unsafe fn(&[f32]) -> bool> = vec![has_non_finite];
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            checks.push(crate::neon::has_non_finite_neon);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            checks.push(crate::avx::has_non_finite_avx2);
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::is_x86_feature_detected!("sse4.1") {
            checks.push(crate::sse::has_non_finite_sse41);
        }
        for check in checks {
            for n in 0..=65 {
                let mut data: Vec<_> = (0..n + 2)
                    .map(|i| [0.0, -0.0, 1e-40, f32::MAX, f32::MIN][i % 5])
                    .collect();
                // SAFETY: features were detected; the offset slices exercise unaligned loads.
                assert!(!unsafe { check(&data[1..n + 1]) });
                for i in 1..=n {
                    let saved = data[i];
                    for invalid in [
                        f32::NAN,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                        f32::from_bits(0x7f80_0001),
                    ] {
                        data[i] = invalid;
                        assert!(
                            unsafe { check(&data[1..n + 1]) },
                            "n={n}, i={i}, invalid={invalid}"
                        );
                    }
                    data[i] = saved;
                }
            }
        }
    }

    #[test]
    fn transfer_endpoints_and_tables_agree() {
        assert_eq!(srgb8_table()[0], 0.0);
        assert!((srgb8_table()[255] - 1.0).abs() < 1e-7);
        assert!((srgb16_table()[65535] - 1.0).abs() < 1e-7);
        assert!((srgb8_table()[128] - srgb16_table()[128 * 257]).abs() < 1e-7);
        assert_eq!(srgb_to_linear(-0.5), -srgb_to_linear(0.5));
    }

    #[test]
    fn layout_is_validated() {
        let px = [0u8; 30];
        assert!(RgbImage::srgb8(&px, 3, 3, 9).is_ok());
        assert!(RgbImage::srgb8(&px, 3, 3, 8).is_err());
        assert!(RgbImage::srgb8(&px, 4, 3, 12).is_err());
        assert!(RgbImage::srgb8(&px, 0, 3, 9).is_err());
    }

    #[test]
    fn non_finite_floats_are_rejected_before_writing() {
        let mut px = vec![0.5f32; 4 * 4 * 3];
        px[20] = f32::NAN;
        let img = RgbImage::srgb_f32(&px, 4, 4, 12).unwrap();
        assert_eq!(img.to_linear().unwrap_err(), Error::NonFiniteSample);
    }
}
