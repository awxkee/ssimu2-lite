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

//! Scores against libjxl 0.12 `ssimulacra2` on deterministic sRGB images.
//! `tools/fixtures.py` regenerates the same images and reference scores.

use ssimu2_lite::{Error, PreparedReference, Rect, RgbImage};

/// (pattern, width, height, distortion, amount, libjxl score)
const CASES: &[(&str, usize, usize, &str, i32, f64)] = &[
    ("texture", 97, 83, "noise", 1, 82.815764),
    ("texture", 97, 83, "noise", 4, 36.021482),
    ("texture", 97, 83, "shift", 3, 95.687277),
    ("texture", 97, 83, "posterize", 16, 73.067730),
    ("texture", 97, 83, "blur", 0, 33.433139),
    ("smooth", 256, 200, "noise", 1, 89.071006),
    ("smooth", 256, 200, "noise", 4, 57.881781),
    ("smooth", 256, 200, "shift", 3, 94.050787),
    ("smooth", 256, 200, "posterize", 16, 35.090614),
    ("smooth", 256, 200, "blur", 0, 50.441160),
    ("smooth", 31, 17, "noise", 4, 77.847300),
    ("texture", 9, 8, "blur", 0, 96.016600),
];

fn pattern(kind: &str, w: usize, h: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let p = if kind == "texture" {
                [
                    (x * 3 + y * 5 + (x * y) % 17) % 200 + 20,
                    (x * 7 + y * 11) % 200 + 20,
                    (x * 13 + y * 2) % 200 + 20,
                ]
            } else {
                [
                    x * 255 / (w - 1),
                    y * 255 / (h - 1),
                    ((x / 8 + y / 8) % 2) * 120 + 60,
                ]
            };
            px.extend(p.map(|v| v as u8));
        }
    }
    px
}

fn distort(px: &[u8], w: usize, h: usize, kind: &str, amount: i32) -> Vec<u8> {
    let at = |x: usize, y: usize, c: usize| px[(y * w + x) * 3 + c] as i32;
    let mut out = Vec::with_capacity(px.len());
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = at(x, y, c);
                let v = match kind {
                    "noise" => v + (((x * 19 + y * 23 + c * 7) % 11) as i32 - 5) * amount,
                    "shift" if c == 1 => v + amount,
                    "shift" => v,
                    "posterize" => v / amount * amount,
                    "blur" => {
                        let mut sum = 0;
                        for dy in [-1isize, 0, 1] {
                            for dx in [-1isize, 0, 1] {
                                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                                sum += at(sx, sy, c);
                            }
                        }
                        sum / 9
                    }
                    _ => unreachable!(),
                };
                out.push(v.clamp(0, 255) as u8);
            }
        }
    }
    out
}

#[test]
fn scores_match_libjxl() {
    for &(pattern_kind, w, h, kind, amount, expected) in CASES {
        let reference = pattern(pattern_kind, w, h);
        let test = distort(&reference, w, h, kind, amount);
        let score = ssimu2_lite::compare(
            RgbImage::srgb8(&reference, w, h, w * 3).unwrap(),
            RgbImage::srgb8(&test, w, h, w * 3).unwrap(),
        )
        .unwrap();
        // libjxl's float recursive blur adds a small noise floor that the
        // FIR blur here does not have; it shows near-lossless.
        let tolerance = if expected > 85.0 { 0.3 } else { 0.05 };
        assert!(
            (score - expected).abs() <= tolerance,
            "{pattern_kind} {w}x{h} {kind} {amount}: {score} vs libjxl {expected}"
        );
    }
}

#[test]
fn identical_images_score_100() {
    let px = pattern("texture", 64, 40);
    let image = RgbImage::srgb8(&px, 64, 40, 64 * 3).unwrap();
    assert_eq!(ssimu2_lite::compare(image, image).unwrap(), 100.0);
}

#[test]
fn all_input_formats_agree() {
    let (w, h) = (97, 83);
    let reference = pattern("texture", w, h);
    let test = distort(&reference, w, h, "noise", 2);
    let prepared =
        PreparedReference::new(RgbImage::srgb8(&reference, w, h, w * 3).unwrap()).unwrap();
    let expected = prepared
        .compare(RgbImage::srgb8(&test, w, h, w * 3).unwrap())
        .unwrap();

    let sixteen: Vec<u16> = test.iter().map(|&v| v as u16 * 257).collect();
    let floats: Vec<f32> = test.iter().map(|&v| v as f32 / 255.0).collect();
    // Padded rows exercise the stride.
    let stride = w + 5;
    let mut planes = [
        vec![0f32; stride * h],
        vec![0f32; stride * h],
        vec![0f32; stride * h],
    ];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = test[(y * w + x) * 3 + c] as f64 / 255.0;
                let linear = if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                };
                planes[c][y * stride + x] = linear as f32;
            }
        }
    }
    let images = [
        RgbImage::srgb16(&sixteen, w, h, w * 3).unwrap(),
        RgbImage::srgb_f32(&floats, w, h, w * 3).unwrap(),
        RgbImage::linear_rgb([&planes[0], &planes[1], &planes[2]], w, h, stride).unwrap(),
    ];
    for image in images {
        let score = prepared.compare(image).unwrap();
        // Each format rounds to f32 linear light at a different point.
        assert!((score - expected).abs() < 2e-3, "{score} vs {expected}");
    }
}

#[test]
fn incremental_updates_track_full_scores() {
    let (w, h) = (256, 200);
    let reference = pattern("smooth", w, h);
    let mut test = distort(&reference, w, h, "noise", 2);
    let prepared =
        PreparedReference::new(RgbImage::srgb8(&reference, w, h, w * 3).unwrap()).unwrap();
    let mut state = prepared
        .incremental(RgbImage::srgb8(&test, w, h, w * 3).unwrap())
        .unwrap();
    for (i, rect) in [
        Rect::new(0, 0, 16, 16),
        Rect::new(100, 60, 164, 124),
        Rect::new(240, 190, 256, 200),
    ]
    .into_iter()
    .enumerate()
    {
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                test[(y * w + x) * 3 + i] = reference[(y * w + x) * 3 + i] / 2;
            }
        }
        let image = RgbImage::srgb8(&test, w, h, w * 3).unwrap();
        state.update(image, rect).unwrap();
        let full = prepared.compare(image).unwrap();
        assert!(
            (state.score() - full).abs() < 1e-3,
            "{} vs {full}",
            state.score()
        );
        let (score, map) = prepared.compare_with_map(image).unwrap();
        let state_map = state.error_map();
        let mean = map.iter().map(|&v| v as f64).sum::<f64>() / map.len() as f64;
        assert!(mean > 0.1 * (100.0 - score) && mean < 2.0 * (100.0 - score));
        let diff = state_map
            .iter()
            .zip(&map)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / map.len() as f64;
        assert!(diff < 1e-3 * mean.max(1e-3), "map {diff}");
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let px = vec![0u8; 16 * 16 * 3];
    let small = RgbImage::srgb8(&px, 7, 16, 7 * 3).unwrap();
    assert_eq!(
        PreparedReference::new(small).err(),
        Some(Error::ImageTooSmall)
    );
    let a = RgbImage::srgb8(&px, 16, 16, 48).unwrap();
    let b = RgbImage::srgb8(&px, 16, 15, 48).unwrap();
    assert_eq!(ssimu2_lite::compare(a, b), Err(Error::DimensionMismatch));
    let prepared = PreparedReference::new(a).unwrap();
    assert_eq!(prepared.compare(b), Err(Error::DimensionMismatch));
    assert!(matches!(
        prepared.incremental(b),
        Err(Error::DimensionMismatch)
    ));
    assert_eq!(
        RgbImage::srgb8(&px, 16, 16, 47).err(),
        Some(Error::InvalidLayout)
    );
}
