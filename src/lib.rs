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

//! SSIMULACRA 2 perceptual image quality metric, compatible with libjxl's
//! `ssimulacra2` tool, with a prepared-reference and a region-incremental
//! API for encoder search loops.
//!
//! Scores: 100 is identical, lower is worse. Roughly 90 is visually lossless,
//! 70 high quality, 50 medium, 30 low. Very poor images score below 0.
//!
//! ```
//! use ssimu2_lite::RgbImage;
//! let (w, h) = (64, 48);
//! let reference: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
//! let mut test = reference.clone();
//! test[100] ^= 0x10;
//!
//! let score = ssimu2_lite::compare(
//!     RgbImage::srgb8(&reference, w, h, w * 3)?,
//!     RgbImage::srgb8(&test, w, h, w * 3)?,
//! )?;
//! assert!(score < 100.0);
//! # Ok::<(), ssimu2_lite::Error>(())
//! ```
//!
//! Reuse a [`PreparedReference`] when one reference is compared with many
//! candidates, and an [`IncrementalState`] when candidates differ locally:
//!
//! ```
//! use ssimu2_lite::{Checkpoint, PreparedReference, Rect, RgbImage};
//! # let (w, h) = (64, 48);
//! # let reference: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
//! # let mut test = reference.clone();
//! let prepared = PreparedReference::new(RgbImage::srgb8(&reference, w, h, w * 3)?)?;
//! let mut state = prepared.incremental(RgbImage::srgb8(&test, w, h, w * 3)?)?;
//!
//! // Try a local change, keep it only if it scores better.
//! test[(10 * w + 10) * 3] = 0;
//! let mut checkpoint = Checkpoint::default();
//! let before = state.score();
//! state.update_checkpointed(
//!     RgbImage::srgb8(&test, w, h, w * 3)?,
//!     Rect::new(10, 10, 11, 11),
//!     &mut checkpoint,
//! )?;
//! if state.score() < before {
//!     state.rollback(&checkpoint);
//! }
//! # Ok::<(), ssimu2_lite::Error>(())
//! ```
//!
//! Single-threaded and dependency-free. Blur, downsampling, color conversion
//! and pooling have AVX2, SSE4.1 and NEON kernels selected at runtime, with a
//! portable fallback.

#![forbid(unsafe_op_in_unsafe_fn)]
#![allow(clippy::needless_range_loop)]

#[cfg(all(feature = "avx", target_arch = "x86_64"))]
mod avx;
mod blur;
mod downsample;
mod image;
mod incremental;
mod metric;
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon;
#[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
mod sse;
#[cfg(test)]
mod testing;
mod xyb;

pub use image::RgbImage;
pub use incremental::{Checkpoint, IncrementalState, Rect};
pub use metric::PreparedReference;

/// SSIMULACRA 2 score of `test` against `reference` (100 is identical).
pub fn compare(reference: RgbImage<'_>, test: RgbImage<'_>) -> Result<f64, Error> {
    if (reference.width(), reference.height()) != (test.width(), test.height()) {
        return Err(Error::DimensionMismatch);
    }
    PreparedReference::new(reference)?.compare(test)
}

impl PreparedReference {
    /// Starts a region-incremental evaluation of `test`.
    pub fn incremental(&self, test: RgbImage<'_>) -> Result<IncrementalState, Error> {
        IncrementalState::new(self, test)
    }
}

/// Invalid image layout, size or samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Width and height must be nonzero and fit addressable memory.
    InvalidDimensions,
    /// A stride is too short, or a buffer does not cover the last pixel.
    InvalidLayout,
    /// The metric needs at least 8×8 pixels.
    ImageTooSmall,
    /// Reference and test dimensions differ.
    DimensionMismatch,
    /// A floating-point sample is NaN or infinite.
    NonFiniteSample,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidDimensions => "image dimensions must be nonzero and fit in memory",
            Self::InvalidLayout => "image stride or buffer length is invalid",
            Self::ImageTooSmall => "image must be at least 8x8 pixels",
            Self::DimensionMismatch => "reference and test dimensions differ",
            Self::NonFiniteSample => "image contains a non-finite sample",
        })
    }
}

impl std::error::Error for Error {}
