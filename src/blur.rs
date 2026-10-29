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
//! Separable 11-tap symmetric filter with a row ring, for the SSIM moments.

use std::{ops::Range, sync::OnceLock};

pub(crate) const WINDOW: usize = 11;
pub(crate) const RADIUS: usize = WINDOW / 2;
pub(crate) type BlurRows<'a> = [&'a [f32]; WINDOW];
/// Each row contains mean, square and cross-product planes consecutively.
pub(crate) type MomentRows<'a> = [&'a [f32]; WINDOW];

// SIMD kernels return the processed prefix; the driver handles scalar tails.
type HorizontalFn = fn(&[f32], &[f32; WINDOW], &mut [f32]) -> usize;
type MomentRowFn = fn([&[f32]; 2], [&mut [f32]; 3]) -> usize;
type MomentHorizontalFn = fn([&[f32]; 3], &[f32; WINDOW], [&mut [f32]; 3]) -> usize;
type MomentVerticalFn = fn(&MomentRows<'_>, usize, &[f32; WINDOW], [&mut [f32]; 3]) -> usize;
type ProductFn = fn([&[f32]; 2], &mut [f32]) -> usize;
type VerticalFn = fn(&BlurRows<'_>, &[f32; WINDOW], &mut [f32]) -> usize;

#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    horizontal: HorizontalFn,
    product: ProductFn,
    vertical: VerticalFn,
    moment_row: MomentRowFn,
    moment_horizontal: MomentHorizontalFn,
    moment_vertical: MomentVerticalFn,
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
const SCALAR: Kernels = Kernels {
    horizontal: horizontal_scalar,
    product: product_scalar,
    vertical: vertical_scalar,
    moment_row: moment_row_scalar,
    moment_horizontal: moment_horizontal_scalar,
    moment_vertical: moment_vertical_scalar,
};

pub(crate) fn selected_kernels() -> Kernels {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    *KERNELS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            return Kernels {
                horizontal: |src, k, dst| unsafe { crate::avx::blur_horizontal_avx2(src, k, dst) },
                product: |src, dst| unsafe { crate::avx::blur_product_row_avx2(src, dst) },
                vertical: |rows, k, dst| unsafe { crate::avx::blur_vertical_avx2(rows, k, dst) },
                moment_row: |s, d| unsafe { crate::avx::blur_moment_row_avx2(s, d) },
                moment_horizontal: |s, k, d| unsafe {
                    crate::avx::blur_moment_horizontal_avx2(s, k, d)
                },
                moment_vertical: |s, w, k, d| unsafe {
                    crate::avx::blur_moment_vertical_avx2(s, w, k, d)
                },
            };
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Kernels {
                horizontal: |src, k, dst| unsafe { crate::sse::blur_horizontal_sse41(src, k, dst) },
                product: |src, dst| unsafe { crate::sse::blur_product_row_sse41(src, dst) },
                vertical: |rows, k, dst| unsafe { crate::sse::blur_vertical_sse41(rows, k, dst) },
                moment_row: |s, d| unsafe { crate::sse::blur_moment_row_sse41(s, d) },
                moment_horizontal: |s, k, d| unsafe {
                    crate::sse::blur_moment_horizontal_sse41(s, k, d)
                },
                moment_vertical: |s, w, k, d| unsafe {
                    crate::sse::blur_moment_vertical_sse41(s, w, k, d)
                },
            };
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            Kernels {
                horizontal: |src, k, dst| unsafe { crate::neon::blur_horizontal_neon(src, k, dst) },
                product: |src, dst| unsafe { crate::neon::blur_product_row_neon(src, dst) },
                vertical: |rows, k, dst| unsafe { crate::neon::blur_vertical_neon(rows, k, dst) },
                moment_row: |s, d| unsafe { crate::neon::blur_moment_row_neon(s, d) },
                moment_horizontal: |s, k, d| unsafe {
                    crate::neon::blur_moment_horizontal_neon(s, k, d)
                },
                moment_vertical: |s, w, k, d| unsafe {
                    crate::neon::blur_moment_vertical_neon(s, w, k, d)
                },
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        SCALAR
    })
}

#[cfg(test)]
fn kernel() -> &'static [f32; WINDOW] {
    static KERNEL: OnceLock<[f32; WINDOW]> = OnceLock::new();
    KERNEL.get_or_init(|| {
        const SIGMA: f32 = 1.5;
        let mut k = [0.0; WINDOW];
        let mut sum = 0.0;
        for (i, v) in k.iter_mut().enumerate() {
            let d = i as f32 - RADIUS as f32;
            *v = (-0.5 * d * d / (SIGMA * SIGMA)).exp();
            sum += *v;
        }
        for v in &mut k {
            *v /= sum;
        }
        k
    })
}

/// Reused for the five moment planes at one scale. Only WINDOW rows are kept,
/// so the horizontal intermediate stays in cache instead of filling a frame.
pub(crate) struct GaussianBlur {
    width: usize,
    ring: Vec<f32>,
    padded: Vec<f32>,
    /// Three moment planes per ring row, allocated only when SSIM needs them.
    moment_ring: Vec<f32>,
    moment_padded: [Vec<f32>; 2],
    /// All-zero row for zero padding in the vertical pass.
    zero_row: Vec<f32>,
    kernels: Kernels,
    kernel: [f32; WINDOW],
    zero_pad: bool,
}

impl GaussianBlur {
    /// σ = 1.5 Gaussian with edge replication.
    #[cfg(test)]
    pub(crate) fn new(width: usize) -> Self {
        Self::with_kernel(width, *kernel(), false)
    }

    /// Any symmetric `WINDOW`-tap kernel; `zero_pad` treats pixels outside the
    /// plane as zero instead of replicating the edge.
    #[cfg(test)]
    pub(crate) fn with_kernel(width: usize, kernel: [f32; WINDOW], zero_pad: bool) -> Self {
        Self::with_kernels(width, kernel, zero_pad, selected_kernels())
    }

    pub(crate) fn with_kernels(
        width: usize,
        kernel: [f32; WINDOW],
        zero_pad: bool,
        kernels: Kernels,
    ) -> Self {
        Self {
            width,
            ring: vec![0.0; width * WINDOW],
            padded: vec![0.0; width + 2 * RADIUS],
            moment_ring: Default::default(),
            moment_padded: Default::default(),
            zero_row: vec![0.0; 3 * width],
            kernels,
            kernel,
            zero_pad,
        }
    }

    /// Reconfigure the row scratch without discarding its allocated capacity.
    pub(crate) fn set_width(&mut self, width: usize) {
        self.width = width;
        self.ring.resize(width * WINDOW, 0.0);
        self.padded.resize(width + 2 * RADIUS, 0.0);
        if self.zero_row.len() < 3 * width {
            self.zero_row.resize(3 * width, 0.0);
        }
    }

    /// Separable blur; borders replicate the edge or read zeros (`zero_pad`).
    pub(crate) fn blur(&mut self, src: &[f32], height: usize, out: &mut [f32]) {
        let w = self.width;
        assert_eq!(src.len(), w * height);
        assert_eq!(out.len(), src.len());
        self.blur_region(src, w, height, [0..w, 0..height], out, w);
    }

    /// Blur pointwise products directly from their source planes. Products are
    /// computed in the padded row scratch, without materializing a full plane.
    pub(crate) fn blur_product(&mut self, src: [&[f32]; 2], height: usize, out: &mut [f32]) {
        let w = self.width;
        assert_eq!(out.len(), w * height);
        self.blur_product_region(src, w, height, [0..w, 0..height], out, w);
    }

    /// Blur only the requested columns and rows, reading their halo from the
    /// source plane. The first output pixel is at `out[0]`; `out_stride` lets
    /// callers write directly into a larger plane. Scratch width equals the
    /// number of output columns, independently of the source width.
    pub(crate) fn blur_region(
        &mut self,
        src: &[f32],
        src_width: usize,
        height: usize,
        [columns, rows]: [Range<usize>; 2],
        out: &mut [f32],
        out_stride: usize,
    ) {
        self.blur_region_impl::<false>(
            [src, src],
            src_width,
            height,
            [columns, rows],
            out,
            out_stride,
        );
    }

    /// Product version of `blur_region`, reading both planes' halo directly.
    pub(crate) fn blur_product_region(
        &mut self,
        src: [&[f32]; 2],
        src_width: usize,
        height: usize,
        region: [Range<usize>; 2],
        out: &mut [f32],
        out_stride: usize,
    ) {
        self.blur_region_impl::<true>(src, src_width, height, region, out, out_stride);
    }

    /// Blur the test mean, test square and reference/test cross-product together.
    /// Inputs are `[reference, test]`; outputs are `[mu2, s22, s12]`.
    pub(crate) fn blur_moments(&mut self, src: [&[f32]; 2], height: usize, out: [&mut [f32]; 3]) {
        let w = self.width;
        assert!(out.iter().all(|p| p.len() == w * height));
        self.blur_moments_region(src, w, height, [0..w, 0..height], out, w);
    }

    /// Region version of `blur_moments`, with the same halo and output layout
    /// as `blur_region`. Products round to f32 before horizontal filtering.
    pub(crate) fn blur_moments_region(
        &mut self,
        [reference, test]: [&[f32]; 2],
        src_width: usize,
        height: usize,
        [columns, rows]: [Range<usize>; 2],
        mut out: [&mut [f32]; 3],
        out_stride: usize,
    ) {
        let w = self.width;
        assert_eq!(test.len(), src_width * height);
        assert_eq!(reference.len(), test.len());
        assert!(columns.start <= columns.end && columns.end <= src_width);
        assert!(rows.start <= rows.end && rows.end <= height);
        assert_eq!(columns.len(), w);
        assert!(out_stride >= w);
        if w == 0 || rows.is_empty() {
            return;
        }
        assert!(
            out.iter()
                .all(|p| p.len() >= (rows.len() - 1) * out_stride + w)
        );
        // Retain initialized scratch across narrower scales/regions. Every live
        // ring row is overwritten before use, so shrinking and zeroing it again
        // on the next wider update would only add memory traffic.
        if self.moment_ring.len() < 3 * w * WINDOW {
            self.moment_ring.resize(3 * w * WINDOW, 0.0);
        }
        for padded in &mut self.moment_padded {
            if padded.len() < w + 2 * RADIUS {
                padded.resize(w + 2 * RADIUS, 0.0);
            }
        }
        let [square_padded, cross_padded] = &mut self.moment_padded;
        let mut padded = [
            &mut self.padded[..],
            &mut square_padded[..w + 2 * RADIUS],
            &mut cross_padded[..w + 2 * RADIUS],
        ];
        let k = &self.kernel;
        let first = columns.start.saturating_sub(RADIUS);
        let last = (columns.end + RADIUS).min(src_width);
        let left_pad = RADIUS.saturating_sub(columns.start);
        let copied_end = left_pad + last - first;
        let mut next_row = rows.start.saturating_sub(RADIUS);
        for y in rows.clone() {
            let last_row = (y + RADIUS).min(height - 1);
            while next_row <= last_row {
                let start = next_row * src_width;
                let rec = &test[start..start + src_width];
                let src = &reference[start..start + src_width];
                let edge = |x: usize| {
                    if self.zero_pad {
                        [0.0; 3]
                    } else {
                        [rec[x], rec[x] * rec[x], src[x] * rec[x]]
                    }
                };
                let left = edge(0);
                let right = edge(src_width - 1);
                for c in 0..3 {
                    padded[c][..left_pad].fill(left[c]);
                    padded[c][copied_end..].fill(right[c]);
                }
                let inputs = [&src[first..last], &rec[first..last]];
                let done = (self.kernels.moment_row)(
                    inputs,
                    padded.each_mut().map(|p| &mut p[left_pad..copied_end]),
                );
                if done < last - first {
                    moment_row_scalar(
                        inputs.map(|p| &p[done..]),
                        padded
                            .each_mut()
                            .map(|p| &mut p[left_pad + done..copied_end]),
                    );
                }
                let slot = next_row % WINDOW;
                let start = slot * 3 * w;
                let (mean, products) = self.moment_ring[start..start + 3 * w].split_at_mut(w);
                let (square, cross) = products.split_at_mut(w);
                let mut dst = [mean, square, cross];
                let done = (self.kernels.moment_horizontal)(
                    padded.each_ref().map(|p| &p[..]),
                    k,
                    dst.each_mut().map(|p| &mut p[..]),
                );
                if done < w {
                    moment_horizontal_scalar(
                        padded.each_ref().map(|p| &p[done..]),
                        k,
                        dst.each_mut().map(|p| &mut p[done..]),
                    );
                }
                next_row += 1;
            }
            // Compute each source-row/slot index once for all three moments.
            let window: MomentRows<'_> = std::array::from_fn(|i| {
                let sy = (y + i) as isize - RADIUS as isize;
                if self.zero_pad && (sy < 0 || sy >= height as isize) {
                    return &self.zero_row[..3 * w];
                }
                let sy = sy.max(0).min(height as isize - 1) as usize;
                let start = sy % WINDOW * 3 * w;
                &self.moment_ring[start..start + 3 * w]
            });
            let start = (y - rows.start) * out_stride;
            let done = (self.kernels.moment_vertical)(
                &window,
                w,
                k,
                out.each_mut().map(|p| &mut p[start..start + w]),
            );
            if done < w {
                moment_vertical_tail(
                    &window,
                    w,
                    k,
                    out.each_mut().map(|p| &mut p[start..start + w]),
                    done,
                );
            }
        }
    }

    fn blur_region_impl<const PRODUCT: bool>(
        &mut self,
        [src, second]: [&[f32]; 2],
        src_width: usize,
        height: usize,
        [columns, rows]: [Range<usize>; 2],
        out: &mut [f32],
        out_stride: usize,
    ) {
        let w = self.width;
        assert_eq!(src.len(), src_width * height);
        if PRODUCT {
            assert_eq!(second.len(), src.len());
        }
        assert!(columns.start <= columns.end && columns.end <= src_width);
        assert!(rows.start <= rows.end && rows.end <= height);
        assert_eq!(columns.len(), w);
        assert!(out_stride >= w);
        if w == 0 || rows.is_empty() {
            return;
        }
        assert!(out.len() >= (rows.len() - 1) * out_stride + w);
        let k = &self.kernel;
        let first = columns.start.saturating_sub(RADIUS);
        let last = (columns.end + RADIUS).min(src_width);
        let left_pad = RADIUS.saturating_sub(columns.start);
        let copied_end = left_pad + last - first;
        let mut next_row = rows.start.saturating_sub(RADIUS);
        for y in rows.clone() {
            let last_row = (y + RADIUS).min(height - 1);
            while next_row <= last_row {
                let row = &src[next_row * src_width..(next_row + 1) * src_width];
                let (left, right) = if self.zero_pad {
                    (0.0, 0.0)
                } else if PRODUCT {
                    let other = &second[next_row * src_width..(next_row + 1) * src_width];
                    (row[0] * other[0], row[src_width - 1] * other[src_width - 1])
                } else {
                    (row[0], row[src_width - 1])
                };
                self.padded[..left_pad].fill(left);
                let dst = &mut self.padded[left_pad..copied_end];
                if PRODUCT {
                    let other = &second[next_row * src_width..(next_row + 1) * src_width];
                    let inputs = [&row[first..last], &other[first..last]];
                    let done = (self.kernels.product)(inputs, dst);
                    product_scalar(inputs.map(|p| &p[done..]), &mut dst[done..]);
                } else {
                    dst.copy_from_slice(&row[first..last]);
                }
                self.padded[copied_end..].fill(right);
                let slot = next_row % WINDOW;
                let dst = &mut self.ring[slot * w..(slot + 1) * w];
                let done = (self.kernels.horizontal)(&self.padded, k, dst);
                horizontal_scalar(&self.padded[done..], k, &mut dst[done..]);
                next_row += 1;
            }
            // At most WINDOW consecutive source rows are live. Repeated edge
            // rows point to the same slot without copying or re-filtering them;
            // with zero padding, rows outside the plane read the zero row.
            let window_rows: BlurRows<'_> = std::array::from_fn(|i| {
                let sy = (y + i) as isize - RADIUS as isize;
                if self.zero_pad && (sy < 0 || sy >= height as isize) {
                    return &self.zero_row[..w];
                }
                let sy = sy.max(0).min(height as isize - 1) as usize;
                let slot = sy % WINDOW;
                &self.ring[slot * w..(slot + 1) * w]
            });
            let start = (y - rows.start) * out_stride;
            let dst = &mut out[start..start + w];
            let done = (self.kernels.vertical)(&window_rows, k, dst);
            if done < w {
                let tail = window_rows.map(|row| &row[done..]);
                vertical_scalar(&tail, k, &mut dst[done..]);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn scratch_layout(&self) -> [(usize, usize); 6] {
        [
            &self.ring,
            &self.padded,
            &self.zero_row,
            &self.moment_ring,
            &self.moment_padded[0],
            &self.moment_padded[1],
        ]
        .map(|buffer| (buffer.as_ptr() as usize, buffer.capacity()))
    }
}

fn product_scalar([a, b]: [&[f32]; 2], dst: &mut [f32]) -> usize {
    for ((out, &a), &b) in dst.iter_mut().zip(a).zip(b) {
        *out = a * b;
    }
    dst.len()
}

fn moment_row_scalar(
    [reference, test]: [&[f32]; 2],
    [mean, square, cross]: [&mut [f32]; 3],
) -> usize {
    for i in 0..mean.len() {
        let rec = test[i];
        mean[i] = rec;
        square[i] = rec * rec;
        cross[i] = reference[i] * rec;
    }
    mean.len()
}

fn moment_horizontal_scalar(src: [&[f32]; 3], k: &[f32; WINDOW], dst: [&mut [f32]; 3]) -> usize {
    for x in 0..dst[0].len() {
        let mut acc = src.map(|p| p[x + RADIUS] * k[RADIUS]);
        for i in 0..RADIUS {
            for c in 0..3 {
                acc[c] += (src[c][x + i] + src[c][x + WINDOW - 1 - i]) * k[i];
            }
        }
        for c in 0..3 {
            dst[c][x] = acc[c];
        }
    }
    dst[0].len()
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
fn moment_vertical_scalar(
    rows: &MomentRows<'_>,
    width: usize,
    k: &[f32; WINDOW],
    dst: [&mut [f32]; 3],
) -> usize {
    let n = dst[0].len();
    moment_vertical_tail(rows, width, k, dst, 0);
    n
}

fn moment_vertical_tail(
    rows: &MomentRows<'_>,
    width: usize,
    k: &[f32; WINDOW],
    dst: [&mut [f32]; 3],
    start: usize,
) {
    for x in start..dst[0].len() {
        let mut acc: [f32; 3] = std::array::from_fn(|c| rows[RADIUS][c * width + x] * k[RADIUS]);
        for i in 0..RADIUS {
            for c in 0..3 {
                let at = c * width + x;
                acc[c] += (rows[i][at] + rows[WINDOW - 1 - i][at]) * k[i];
            }
        }
        for c in 0..3 {
            dst[c][x] = acc[c];
        }
    }
}

// Center plus symmetric pairs, as in libblur. Separate multiplies/adds keep
// SIMD and scalar tails consistent without requiring FMA on x86.
fn horizontal_scalar(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    for (x, dst) in dst.iter_mut().enumerate() {
        let mut acc = src[x + RADIUS] * k[RADIUS];
        for i in 0..RADIUS {
            acc += (src[x + i] + src[x + WINDOW - 1 - i]) * k[i];
        }
        *dst = acc;
    }
    dst.len()
}

fn vertical_scalar(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    for (x, dst) in dst.iter_mut().enumerate() {
        let mut acc = rows[RADIUS][x] * k[RADIUS];
        for i in 0..RADIUS {
            acc += (rows[i][x] + rows[WINDOW - 1 - i][x]) * k[i];
        }
        *dst = acc;
    }
    dst.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available_kernels() -> Vec<(&'static str, Kernels)> {
        #[allow(unused_mut)]
        let mut kernels = vec![("dispatch", selected_kernels()), ("scalar", SCALAR)];
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            kernels.push((
                "neon",
                Kernels {
                    horizontal: |s, k, d| unsafe { crate::neon::blur_horizontal_neon(s, k, d) },
                    product: |s, d| unsafe { crate::neon::blur_product_row_neon(s, d) },
                    vertical: |s, k, d| unsafe { crate::neon::blur_vertical_neon(s, k, d) },
                    moment_row: |s, d| unsafe { crate::neon::blur_moment_row_neon(s, d) },
                    moment_horizontal: |s, k, d| unsafe {
                        crate::neon::blur_moment_horizontal_neon(s, k, d)
                    },
                    moment_vertical: |s, w, k, d| unsafe {
                        crate::neon::blur_moment_vertical_neon(s, w, k, d)
                    },
                },
            ));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            kernels.push((
                "avx2",
                Kernels {
                    horizontal: |s, k, d| unsafe { crate::avx::blur_horizontal_avx2(s, k, d) },
                    product: |s, d| unsafe { crate::avx::blur_product_row_avx2(s, d) },
                    vertical: |s, k, d| unsafe { crate::avx::blur_vertical_avx2(s, k, d) },
                    moment_row: |s, d| unsafe { crate::avx::blur_moment_row_avx2(s, d) },
                    moment_horizontal: |s, k, d| unsafe {
                        crate::avx::blur_moment_horizontal_avx2(s, k, d)
                    },
                    moment_vertical: |s, w, k, d| unsafe {
                        crate::avx::blur_moment_vertical_avx2(s, w, k, d)
                    },
                },
            ));
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            kernels.push((
                "sse4.1",
                Kernels {
                    horizontal: |s, k, d| unsafe { crate::sse::blur_horizontal_sse41(s, k, d) },
                    product: |s, d| unsafe { crate::sse::blur_product_row_sse41(s, d) },
                    vertical: |s, k, d| unsafe { crate::sse::blur_vertical_sse41(s, k, d) },
                    moment_row: |s, d| unsafe { crate::sse::blur_moment_row_sse41(s, d) },
                    moment_horizontal: |s, k, d| unsafe {
                        crate::sse::blur_moment_horizontal_sse41(s, k, d)
                    },
                    moment_vertical: |s, w, k, d| unsafe {
                        crate::sse::blur_moment_vertical_sse41(s, w, k, d)
                    },
                },
            ));
        }
        kernels
    }

    // Independent full-frame, eleven-tap implementation from the old metric.
    fn reference(src: &[f32], w: usize, h: usize) -> Vec<f32> {
        let k = kernel();
        let mut tmp = vec![0.0; src.len()];
        let mut out = vec![0.0; src.len()];
        for y in 0..h {
            for x in 0..w {
                for (i, &weight) in k.iter().enumerate() {
                    let sx = (x + i).saturating_sub(RADIUS).min(w - 1);
                    tmp[y * w + x] += weight * src[y * w + sx];
                }
            }
        }
        for y in 0..h {
            for (i, &weight) in k.iter().enumerate() {
                let sy = (y + i).saturating_sub(RADIUS).min(h - 1);
                for x in 0..w {
                    out[y * w + x] += weight * tmp[sy * w + x];
                }
            }
        }
        out
    }

    fn check(src: &[f32], w: usize, h: usize) {
        let expected = reference(src, w, h);
        let mut scalar = vec![f32::NAN; src.len()];
        let mut blur = GaussianBlur::new(w);
        blur.kernels = SCALAR;
        blur.blur(src, h, &mut scalar);
        for (name, kernels) in available_kernels() {
            blur.kernels = kernels;
            let mut out = vec![f32::NAN; src.len()];
            blur.blur(src, h, &mut out);
            for (i, ((actual, expected), scalar)) in
                out.iter().zip(&expected).zip(&scalar).enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    scalar.to_bits(),
                    "{name}: {w}x{h}, index={i}"
                );
                assert!(
                    (actual - expected).abs() <= 5e-7,
                    "{name}: {w}x{h}, index={i}, {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn matches_reference_for_edges_tails_and_ring_wraps() {
        let widths = (1..=40).chain([47, 63, 64, 65, 127, 128, 129]);
        for w in widths {
            for h in (1..=13).chain([23, 37]) {
                let src: Vec<f32> = (0..w * h)
                    .map(|i| ((i * 71 + i / w * 113) % 101) as f32 / 50.0 - 1.0)
                    .collect();
                check(&src, w, h);
            }
        }
    }

    #[test]
    fn preserves_constants_and_clamps_corner_impulses() {
        for (w, h) in [(1, 1), (3, 2), (17, 29), (65, 23)] {
            check(&vec![0.4; w * h], w, h);
            for i in [0, w - 1, (h - 1) * w, w * h - 1, (h / 2) * w + w / 2] {
                let mut src = vec![0.0; w * h];
                src[i] = 1.0;
                check(&src, w, h);
            }
        }
    }

    #[test]
    fn scratch_can_be_reused_for_different_heights() {
        let w = 19;
        let mut blur = GaussianBlur::new(w);
        for h in [25, 1, 3, 37, 12] {
            let src: Vec<f32> = (0..w * h).map(|i| (i % 47) as f32 / 47.0).collect();
            let mut out = vec![f32::NAN; src.len()];
            blur.blur(&src, h, &mut out);
            for (a, b) in out.iter().zip(reference(&src, w, h)) {
                assert!((a - b).abs() <= 5e-7);
            }
        }
        assert_eq!(blur.ring.len(), w * WINDOW);
    }

    #[test]
    fn region_matches_full_blur_and_preserves_surrounding_pixels() {
        const GUARD: f32 = -12345.0;
        for zero_pad in [false, true] {
            for (name, kernels) in available_kernels() {
                for (w, h) in [(1, 1), (3, 7), (17, 13), (65, 37)] {
                    let src: Vec<f32> = (0..w * h)
                        .map(|i| ((i * 71 + i / w * 113) % 101) as f32 / 50.0 - 1.0)
                        .collect();
                    let mut blur = GaussianBlur::with_kernel(w, *kernel(), zero_pad);
                    blur.kernels = kernels;
                    let ring = blur.ring.as_ptr();
                    let padded = blur.padded.as_ptr();
                    let zero = blur.zero_row.as_ptr();
                    let mut full = vec![f32::NAN; src.len()];
                    blur.blur(&src, h, &mut full);
                    for [columns, rows] in [
                        [0..w, 0..h],
                        [0..1, 0..1],
                        [w - 1..w, h - 1..h],
                        [w / 3..w.div_ceil(2), h / 3..h.div_ceil(2)],
                        [0..w.div_ceil(2), h / 2..h],
                        [w / 2..w, 0..h.div_ceil(2)],
                    ] {
                        blur.set_width(columns.len());
                        let mut out = vec![GUARD; src.len()];
                        let offset = rows.start * w + columns.start;
                        blur.blur_region(
                            &src,
                            w,
                            h,
                            [columns.clone(), rows.clone()],
                            &mut out[offset..],
                            w,
                        );
                        for y in 0..h {
                            for x in 0..w {
                                let expected = if columns.contains(&x) && rows.contains(&y) {
                                    full[y * w + x]
                                } else {
                                    GUARD
                                };
                                assert_eq!(
                                    out[y * w + x].to_bits(),
                                    expected.to_bits(),
                                    "{name}, zero_pad={zero_pad}, {w}x{h}, {columns:?}x{rows:?}, ({x},{y})"
                                );
                            }
                        }
                    }
                    assert_eq!(blur.ring.as_ptr(), ring);
                    assert_eq!(blur.padded.as_ptr(), padded);
                    assert_eq!(blur.zero_row.as_ptr(), zero);
                }
            }
        }
    }

    #[test]
    fn product_blurs_match_materialized_planes_and_regions_bit_exactly() {
        const GUARD: f32 = -12345.0;
        for zero_pad in [false, true] {
            for (name, kernels) in available_kernels() {
                for w in (1..=40).chain([47, 63, 64, 65, 127, 129]) {
                    for h in [1, 3, 17, 37] {
                        let values = [-0.0, 1e-23, -1e-20, -0.7, 0.125, 1.0, 1e15];
                        let a: Vec<_> =
                            (0..w * h).map(|i| values[(i * 3) % values.len()]).collect();
                        let b: Vec<_> = (0..w * h)
                            .map(|i| values[(i * 5 + 1) % values.len()])
                            .collect();
                        for sources in [[&a[..], &a[..]], [&a[..], &b[..]]] {
                            let product: Vec<_> = sources[0]
                                .iter()
                                .zip(sources[1])
                                .map(|(&a, &b)| a * b)
                                .collect();
                            let mut blur = GaussianBlur::with_kernel(w, *kernel(), zero_pad);
                            blur.kernels = kernels;
                            let mut expected = vec![f32::NAN; w * h];
                            blur.blur(&product, h, &mut expected);
                            let mut actual = vec![f32::NAN; w * h];
                            blur.blur_product(sources, h, &mut actual);
                            assert!(
                                actual
                                    .iter()
                                    .zip(&expected)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "{name}, zero_pad={zero_pad}, {w}x{h}"
                            );
                            let scratch = blur.scratch_layout();
                            for [columns, rows] in [
                                [0..w, 0..h],
                                [0..1, 0..1],
                                [w - 1..w, h - 1..h],
                                [w / 3..w.div_ceil(2), h / 3..h.div_ceil(2)],
                                [0..w.div_ceil(2), h / 2..h],
                                [w / 2..w, 0..h.div_ceil(2)],
                                [0..0, 0..h],
                                [0..w, 0..0],
                            ] {
                                blur.set_width(columns.len());
                                actual.fill(GUARD);
                                let offset = rows.start * w + columns.start;
                                blur.blur_product_region(
                                    sources,
                                    w,
                                    h,
                                    [columns.clone(), rows.clone()],
                                    &mut actual[offset..],
                                    w,
                                );
                                for y in 0..h {
                                    for x in 0..w {
                                        let reference = if columns.contains(&x) && rows.contains(&y)
                                        {
                                            expected[y * w + x]
                                        } else {
                                            GUARD
                                        };
                                        assert_eq!(
                                            actual[y * w + x].to_bits(),
                                            reference.to_bits(),
                                            "{name}, zero_pad={zero_pad}, {w}x{h}, {columns:?}x{rows:?}, ({x},{y})"
                                        );
                                    }
                                }
                            }
                            assert_eq!(blur.scratch_layout(), scratch);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fused_moments_match_three_passes_bit_exactly_and_reuse_scratch() {
        const GUARD: f32 = -12345.0;
        let values = [-0.0, 1e-23, -1e-20, -0.7, 0.125, 1.0, 1e15];
        for zero_pad in [false, true] {
            for (name, kernels) in available_kernels() {
                for w in (1..=40).chain([47, 63, 64, 65, 127, 128, 129]) {
                    for h in [1, 3, 17, 37] {
                        let a: Vec<_> =
                            (0..w * h).map(|i| values[(i * 3) % values.len()]).collect();
                        let b: Vec<_> = (0..w * h)
                            .map(|i| values[(i * 5 + 1) % values.len()])
                            .collect();
                        for [src, rec] in [[&a[..], &a[..]], [&a[..], &b[..]]] {
                            let mut separate = GaussianBlur::with_kernel(w, *kernel(), zero_pad);
                            separate.kernels = kernels;
                            let mut expected: [Vec<f32>; 3] =
                                std::array::from_fn(|_| vec![f32::NAN; w * h]);
                            separate.blur(rec, h, &mut expected[0]);
                            separate.blur_product([rec, rec], h, &mut expected[1]);
                            separate.blur_product([src, rec], h, &mut expected[2]);
                            let mut fused = GaussianBlur::with_kernel(w, *kernel(), zero_pad);
                            fused.kernels = kernels;
                            // Extra buffers are lazy; mean/product blurs need no fused scratch.
                            assert!(
                                fused.moment_ring.capacity() == 0
                                    && fused.moment_padded.iter().all(|p| p.capacity() == 0)
                            );
                            let mut actual: [Vec<f32>; 3] =
                                std::array::from_fn(|_| vec![GUARD; w * h + 2]);
                            fused.blur_moments(
                                [src, rec],
                                h,
                                actual.each_mut().map(|p| &mut p[1..w * h + 1]),
                            );
                            for c in 0..3 {
                                for i in 0..w * h {
                                    assert_eq!(
                                        actual[c][i + 1].to_bits(),
                                        expected[c][i].to_bits(),
                                        "{name}, zero_pad={zero_pad}, {w}x{h}, c={c}, i={i}"
                                    );
                                }
                                assert_eq!(actual[c][0], GUARD);
                                assert_eq!(actual[c][w * h + 1], GUARD);
                            }
                            let scratch = fused.scratch_layout();
                            for [columns, rows] in [
                                [0..w, 0..h],
                                [0..1, 0..1],
                                [w - 1..w, h - 1..h],
                                [w / 3..w.div_ceil(2), h / 3..h.div_ceil(2)],
                                [0..w.div_ceil(2), h / 2..h],
                                [w / 2..w, 0..h.div_ceil(2)],
                                [0..0, 0..h],
                                [0..w, 0..0],
                            ] {
                                fused.set_width(columns.len());
                                for plane in &mut actual {
                                    plane.fill(GUARD);
                                }
                                let offset = 1 + rows.start * w + columns.start;
                                fused.blur_moments_region(
                                    [src, rec],
                                    w,
                                    h,
                                    [columns.clone(), rows.clone()],
                                    actual.each_mut().map(|p| &mut p[offset..]),
                                    w,
                                );
                                for c in 0..3 {
                                    for y in 0..h {
                                        for x in 0..w {
                                            let wanted =
                                                if columns.contains(&x) && rows.contains(&y) {
                                                    expected[c][y * w + x]
                                                } else {
                                                    GUARD
                                                };
                                            assert_eq!(
                                                actual[c][1 + y * w + x].to_bits(),
                                                wanted.to_bits(),
                                                "{name}, zero_pad={zero_pad}, {w}x{h}, {columns:?}x{rows:?}, c={c}, ({x},{y})"
                                            );
                                        }
                                    }
                                    assert_eq!(actual[c][0], GUARD);
                                    assert_eq!(actual[c][w * h + 1], GUARD);
                                }
                                assert_eq!(
                                    fused.scratch_layout(),
                                    scratch,
                                    "{name}, {w}x{h}, {columns:?}x{rows:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn empty_images_are_supported() {
        GaussianBlur::new(0).blur(&[], 4, &mut []);
        GaussianBlur::new(7).blur(&[], 0, &mut []);
        GaussianBlur::new(0).blur_product([&[], &[]], 4, &mut []);
        GaussianBlur::new(7).blur_product([&[], &[]], 0, &mut []);
        GaussianBlur::new(0).blur_moments([&[], &[]], 4, [&mut [], &mut [], &mut []]);
        GaussianBlur::new(7).blur_moments([&[], &[]], 0, [&mut [], &mut [], &mut []]);
    }
}
