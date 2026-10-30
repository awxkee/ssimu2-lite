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

//! Region-incremental scoring for encoder loops that change one area of a
//! test image at a time. The state keeps the test image's linear and XYB
//! pyramids, every weighted map and the pooled sums. Blurred test moments use
//! reusable region scratch. An update recomputes only the changed area
//! (plus the blur halo) at every scale and adjusts the sums; the score
//! then costs a few hundred flops.

use crate::blur::{GaussianBlur, VERTICAL_BATCH};
use crate::dispatch::Kernels;
use crate::maps::MapKind;
use crate::metric::{
    PooledContribution, PreparedReference, RADIUS, ReferenceData, WEIGHTS, active_maps,
    deficit_slope, kernel, score_from_sum, weight_base,
};
use crate::{Error, RgbImage};
use std::sync::Arc;

/// A half-open pixel rectangle `[x0, x1) × [y0, y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Rect {
    pub fn new(x0: usize, y0: usize, x1: usize, y1: usize) -> Self {
        Self { x0, y0, x1, y1 }
    }

    fn expand(self, r: usize, w: usize, h: usize) -> Self {
        Self::new(
            self.x0.saturating_sub(r),
            self.y0.saturating_sub(r),
            (self.x1 + r).min(w),
            (self.y1 + r).min(h),
        )
    }

    fn half(self, w: usize, h: usize) -> Self {
        Self::new(
            self.x0 / 2,
            self.y0 / 2,
            self.x1.div_ceil(2).min(w),
            self.y1.div_ceil(2).min(h),
        )
    }

    fn clamp(self, w: usize, h: usize) -> Self {
        Self::new(self.x0, self.y0, self.x1.min(w), self.y1.min(h))
    }

    pub(crate) fn is_empty(self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
}

/// Saved image planes, maps and sums for [`IncrementalState::update_checkpointed`].
/// Reuse one checkpoint across candidates to keep its allocation.
#[derive(Default)]
pub struct Checkpoint {
    rect: Option<Rect>,
    data: Vec<f32>,
    sums: Vec<[[(f64, f64); 3]; 3]>,
}

/// Copy a short row without a library call: checkpoints move thousands of
/// rows of a few dozen samples per update, where `memcpy` call overhead
/// outweighs the copy itself.
#[inline(always)]
fn copy_row(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    let (dst_vectors, dst_tail) = dst.as_chunks_mut::<8>();
    let (src_vectors, src_tail) = src.as_chunks::<8>();
    for (d, s) in dst_vectors.iter_mut().zip(src_vectors) {
        *d = *s;
    }
    for (d, &s) in dst_tail.iter_mut().zip(src_tail) {
        *d = s;
    }
}

fn save_rows(buf: &[f32], w: usize, r: Rect, out: &mut [f32], mut at: usize) -> usize {
    if buf.is_empty() {
        return at;
    }
    let n = r.x1 - r.x0;
    for y in r.y0..r.y1 {
        copy_row(&mut out[at..at + n], &buf[y * w + r.x0..y * w + r.x1]);
        at += n;
    }
    at
}

fn load_rows(buf: &mut [f32], w: usize, r: Rect, data: &[f32], mut at: usize) -> usize {
    if buf.is_empty() {
        return at;
    }
    let n = r.x1 - r.x0;
    for y in r.y0..r.y1 {
        copy_row(&mut buf[y * w + r.x0..y * w + r.x1], &data[at..at + n]);
        at += n;
    }
    at
}

/// Filtered moment planes a plane's row cache holds.
fn moment_planes(active: [bool; 3]) -> usize {
    if active[0] {
        3
    } else if active.contains(&true) {
        1
    } else {
        0
    }
}

/// Slots of a row cache; one vertical batch needs `VERTICAL_BATCH + 2·RADIUS`.
const ROW_SLOTS: usize = 64;
const _: () = assert!(VERTICAL_BATCH + 2 * RADIUS <= ROW_SLOTS);

/// Horizontally filtered test moments of one plane at one scale, for a band
/// of recent rows: slot `y % ROW_SLOTS` holds row `y`, valid over one column
/// interval. Missing columns are filtered on demand, so a scan in raster
/// order filters each halo row about once instead of once per update.
struct FilteredRows {
    /// Filtered planes per row (see [`moment_planes`]).
    planes: usize,
    width: usize,
    /// `ROW_SLOTS` rows of `planes` planes of `width` samples.
    data: Vec<f32>,
    /// Source row held by each slot (`usize::MAX` for none).
    row: [usize; ROW_SLOTS],
    /// Valid columns of each slot.
    valid: [(usize, usize); ROW_SLOTS],
}

impl FilteredRows {
    fn new(planes: usize, width: usize) -> Self {
        Self {
            planes,
            width,
            data: vec![0.0; ROW_SLOTS * planes * width],
            row: [usize::MAX; ROW_SLOTS],
            valid: [(0, 0); ROW_SLOTS],
        }
    }

    /// Forget rows whose test samples changed.
    fn invalidate(&mut self, rows: std::ops::Range<usize>) {
        for y in rows {
            let slot = y % ROW_SLOTS;
            if self.row[slot] == y {
                self.row[slot] = usize::MAX;
            }
        }
    }

    /// Filter whatever part of `columns` of `rows` the cache lacks.
    fn fill(
        &mut self,
        blur: &mut GaussianBlur,
        inputs: [&[f32]; 2],
        rows: std::ops::Range<usize>,
        columns: std::ops::Range<usize>,
    ) {
        let len = self.planes * self.width;
        for y in rows {
            let slot = y % ROW_SLOTS;
            if self.row[slot] != y {
                self.row[slot] = y;
                self.valid[slot] = (0, 0);
            }
            let (a, b) = self.valid[slot];
            let dst = &mut self.data[slot * len..(slot + 1) * len];
            if a < b && columns.start <= b && a <= columns.end {
                // Overlapping or adjacent: extend the valid interval.
                if columns.start < a {
                    blur.horizontal_row(inputs, self.width, y, columns.start..a, self.planes, dst);
                }
                if b < columns.end {
                    blur.horizontal_row(inputs, self.width, y, b..columns.end, self.planes, dst);
                }
                self.valid[slot] = (a.min(columns.start), b.max(columns.end));
            } else {
                blur.horizontal_row(inputs, self.width, y, columns.clone(), self.planes, dst);
                self.valid[slot] = (columns.start, columns.end);
            }
        }
    }

    fn row(&self, y: usize) -> &[f32] {
        let len = self.planes * self.width;
        let slot = y % ROW_SLOTS;
        debug_assert_eq!(self.row[slot], y);
        &self.data[slot * len..(slot + 1) * len]
    }
}

/// SSIMULACRA 2 of a fixed reference against a test image that changes
/// locally: after [`update`](Self::update) of a rectangle only that rectangle
/// and its blur halo are recomputed at each scale. Scores match a full
/// [`PreparedReference::compare`] up to float summation order.
pub struct IncrementalState {
    reference: Arc<ReferenceData>,
    /// Dispatch inherited from the prepared reference, reused by every update.
    kernels: Kernels,
    /// Test image's linear RGB planes per scale: `[scale][channel]`.
    lin: Vec<[Vec<f32>; 3]>,
    /// Test image's positive XYB planes per scale.
    xyb: Vec<[Vec<f32>; 3]>,
    /// Reused region-sized mean, square and cross-product scratch.
    moments: [Vec<f32>; 3],
    /// Horizontally filtered test moments per scale and plane for a band of
    /// rows: the mean, square and cross-product (only the mean when the plane
    /// has no SSIM map). Updates refilter only rows whose samples changed.
    filtered: Vec<[FilteredRows; 3]>,
    /// Zero rows standing in for rows outside the plane in the vertical pass.
    zero: Vec<f32>,
    /// Maps per scale, plane: ssim, artifact, detail. Unweighted maps are empty.
    maps: Vec<[[Vec<f32>; 3]; 3]>,
    /// Pooled sums per scale, plane, map: (Σx, Σx⁴). Unweighted fourth sums
    /// stay zero; first sums also detect exact-zero maps after local reverts.
    sums: Vec<[[(f64, f64); 3]; 3]>,
    /// Persistent row scratch, also used to form products for moment blurs.
    blur: GaussianBlur,
}

impl IncrementalState {
    /// Scores `test` in full and keeps everything needed for local updates.
    pub fn new(reference: &PreparedReference, test: RgbImage<'_>) -> Result<Self, Error> {
        let data = Arc::clone(&reference.data);
        if (test.width(), test.height()) != (data.width, data.height) {
            return Err(Error::DimensionMismatch);
        }
        let kernels = data.kernels;
        let mut lin = test.to_linear(kernels.image)?;
        let scales = data.scales.len();
        let mut st = IncrementalState {
            blur: GaussianBlur::with_kernels(data.width, *kernel(), true, kernels.blur),
            kernels,
            reference: data,
            lin: Vec::with_capacity(scales),
            xyb: Vec::with_capacity(scales),
            moments: Default::default(),
            filtered: Vec::with_capacity(scales),
            zero: vec![0.0; 3 * test.width()],
            maps: Vec::with_capacity(scales),
            sums: Vec::with_capacity(scales),
        };
        for s in 0..scales {
            let (w, h) = st.dims(s);
            if s > 0 {
                let (pw, ph) = st.dims(s - 1);
                let previous = &st.lin[s - 1];
                lin =
                    std::array::from_fn(|c| kernels.downsample.downsample(&previous[c], pw, ph).0);
            }
            let n = w * h;
            let mut xyb = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
            {
                let [x, y, b] = &mut xyb;
                kernels
                    .xyb
                    .positive_xyb_into([&lin[0], &lin[1], &lin[2]], [x, y, b]);
            }
            st.lin.push(std::mem::take(&mut lin));
            st.xyb.push(xyb);
            let active: [[bool; 3]; 3] = std::array::from_fn(|c| active_maps(c, s, scales));
            let buffer = |needed| if needed { vec![0.0; n] } else { Vec::new() };
            st.maps.push(std::array::from_fn(|c| {
                std::array::from_fn(|m| buffer(active[c][m]))
            }));
            st.sums.push([[(0.0, 0.0); 3]; 3]);
            st.filtered.push(std::array::from_fn(|c| {
                FilteredRows::new(moment_planes(active[c]), w)
            }));
            let whole = Rect::new(0, 0, w, h);
            st.refresh_moments_and_maps(s, whole, whole);
        }
        // Initialization covers whole scales; local updates should retain only
        // the scratch they actually need, rather than these full-frame buffers.
        st.moments = Default::default();
        Ok(st)
    }

    fn scales(&self) -> usize {
        self.reference.scales.len()
    }

    fn dims(&self, s: usize) -> (usize, usize) {
        let scale = &self.reference.scales[s];
        (scale.width, scale.height)
    }

    /// Image width, in pixels.
    pub fn width(&self) -> usize {
        self.reference.width
    }

    /// Image height, in pixels.
    pub fn height(&self) -> usize {
        self.reference.height
    }

    /// Recompute the weighted maps and their test moments at scale `s` over
    /// `r`, after the test samples changed within `changed` (`r` is
    /// `changed` expanded by the blur radius).
    fn refresh_moments_and_maps(&mut self, s: usize, changed: Rect, r: Rect) {
        let (w, h) = self.dims(s);
        if r.is_empty() {
            return;
        }
        let rw = r.x1 - r.x0;
        let n = rw * (r.y1 - r.y0);
        self.blur.set_width(rw);
        for c in 0..3 {
            let active = active_maps(c, s, self.scales());
            if !active.contains(&true) {
                continue;
            }
            let planes = moment_planes(active);
            let rec = &self.xyb[s][c];
            let src = &self.reference.scales[s].xyb[c];
            let cache = &mut self.filtered[s][c];
            // Rows outside `changed` kept their samples, so their filtered
            // columns stay valid; only the changed rows are refiltered.
            cache.invalidate(changed.y0..changed.y1);
            let [mu2, s22, s12] = &mut self.moments;
            for buffer in [&mut *mu2, &mut *s22, &mut *s12].into_iter().take(planes) {
                if buffer.len() < n {
                    buffer.resize(n, 0.0);
                }
            }
            let mut y = r.y0;
            while y < r.y1 {
                let count = VERTICAL_BATCH.min(r.y1 - y);
                let first = y.saturating_sub(RADIUS);
                let last = (y + count + RADIUS).min(h);
                cache.fill(&mut self.blur, [src, rec], first..last, r.x0..r.x1);
                // Window rows `y - RADIUS ..`; rows outside the plane read zeros.
                let zero = &self.zero[r.x0..];
                let mut rows = [zero; VERTICAL_BATCH + 2 * RADIUS];
                let top = RADIUS - (y - first);
                for (row, sy) in rows[top..].iter_mut().zip(first..last) {
                    *row = &cache.row(sy)[r.x0..];
                }
                let at = (y - r.y0) * rw;
                let out = [
                    &mut mu2[at..],
                    if planes == 3 { &mut s22[at..] } else { &mut [] },
                    if planes == 3 { &mut s12[at..] } else { &mut [] },
                ];
                self.blur
                    .vertical_rows(&rows[..count + 2 * RADIUS], w, planes, rw, out, rw);
                y += count;
            }
            self.refresh_maps(s, c, r);
        }
    }

    /// Recompute the weighted maps over `r` from their current moments.
    fn refresh_maps(&mut self, s: usize, c: usize, r: Rect) {
        let (w, _) = self.dims(s);
        let reference = &self.reference.scales[s];
        let active = active_maps(c, s, self.scales());
        let base = weight_base(c, s, self.scales());
        let need_fourth: [bool; 3] = std::array::from_fn(|m| WEIGHTS[base + 3 + m] != 0.0);
        let rec = &self.xyb[s][c];
        let src = &reference.xyb[c];
        let mu1 = &reference.mu[c];
        let s11 = &reference.sigma[c];
        let [mu2, s22, s12] = &self.moments;
        let [ssim, artifact, detail] = &mut self.maps[s][c];
        for y in r.y0..r.y1 {
            let span = y * w + r.x0..y * w + r.x1;
            let start = (y - r.y0) * (r.x1 - r.x0);
            let scratch = start..start + r.x1 - r.x0;
            if active[0] {
                self.kernels.maps.replace_map(
                    MapKind::Ssim,
                    [
                        &mu1[span.clone()],
                        &mu2[scratch.clone()],
                        &s11[span.clone()],
                        &s22[scratch.clone()],
                        &s12[scratch.clone()],
                    ],
                    &mut ssim[span.clone()],
                    &mut self.sums[s][c][0],
                    need_fourth[0],
                );
            }
            let inputs = [
                &src[span.clone()],
                &rec[span.clone()],
                &mu1[span.clone()],
                &mu2[scratch.clone()],
                &src[span.clone()],
            ];
            if active[1] && active[2] {
                let [_, artifact_sum, detail_sum] = &mut self.sums[s][c];
                self.kernels.maps.replace_edge_maps(
                    inputs,
                    [&mut artifact[span.clone()], &mut detail[span]],
                    [artifact_sum, detail_sum],
                    [need_fourth[1], need_fourth[2]],
                );
            } else if active[1] {
                self.kernels.maps.replace_map(
                    MapKind::Artifact,
                    inputs,
                    &mut artifact[span],
                    &mut self.sums[s][c][1],
                    need_fourth[1],
                );
            } else if active[2] {
                self.kernels.maps.replace_map(
                    MapKind::Detail,
                    inputs,
                    &mut detail[span],
                    &mut self.sums[s][c][2],
                    need_fourth[2],
                );
            }
        }
        // Maps are nonnegative: an exact zero first sum implies a zero
        // fourth sum. Clear residual round-off after a candidate revert;
        // the fourth root would otherwise amplify it near a perfect match.
        for sums in &mut self.sums[s][c] {
            if sums.0 == 0.0 {
                sums.1 = 0.0;
            }
        }
    }

    /// Weighted pooled sum over all scales, planes and maps.
    fn weighted_sum(&self) -> f64 {
        let mut sum = 0f64;
        for s in 0..self.sums.len() {
            let (w, h) = self.dims(s);
            let n = (w * h) as f64;
            for c in 0..3 {
                let base = weight_base(c, s, self.scales());
                for m in 0..3 {
                    let (s1, s4) = self.sums[s][c][m];
                    let m1 = (s1 / n).max(0.0);
                    let n4 = (s4 / n).max(0.0).sqrt().sqrt();
                    sum += WEIGHTS[base + m] * m1 + WEIGHTS[base + 3 + m] * n4;
                }
            }
        }
        sum
    }

    /// The current SSIMULACRA 2 score (100 is identical).
    pub fn score(&self) -> f64 {
        score_from_sum(self.weighted_sum())
    }

    /// `test` differs from the previously scored image only inside `rect`:
    /// refresh the state. `test` is the whole image; only `rect` is read.
    /// The rectangle is clipped to the image. On error the state is unchanged.
    pub fn update(&mut self, test: RgbImage<'_>, rect: Rect) -> Result<(), Error> {
        if (test.width(), test.height()) != (self.width(), self.height()) {
            return Err(Error::DimensionMismatch);
        }
        let (w0, h0) = (self.width(), self.height());
        let r0 = rect.clamp(w0, h0);
        if r0.is_empty() {
            return Ok(());
        }
        {
            let [r, g, b] = &mut self.lin[0];
            test.linear_into(r0, [r, g, b], w0, self.kernels.image)?;
        }
        let mut rs = r0;
        for s in 0..self.lin.len() {
            let (w, h) = self.dims(s);
            if s > 0 {
                // downsample the changed region from the previous scale
                let (pw, ph) = self.dims(s - 1);
                rs = rs.half(w, h);
                let (previous, current) = self.lin.split_at_mut(s);
                let current = &mut current[0];
                for y in rs.y0..rs.y1 {
                    let (y0, y1) = ((2 * y).min(ph - 1), (2 * y + 1).min(ph - 1));
                    let (x0, x1) = (2 * rs.x0, (2 * rs.x1).min(pw));
                    for c in 0..3 {
                        let p = &previous[s - 1][c];
                        self.kernels.downsample.downsample_rows(
                            &p[y0 * pw + x0..y0 * pw + x1],
                            &p[y1 * pw + x0..y1 * pw + x1],
                            &mut current[c][y * w + rs.x0..y * w + rs.x1],
                        );
                    }
                }
            }
            let lin = &self.lin[s];
            let [x, y, b] = &mut self.xyb[s];
            for row in rs.y0..rs.y1 {
                let span = row * w + rs.x0..row * w + rs.x1;
                self.kernels.xyb.positive_xyb_into(
                    [
                        &lin[0][span.clone()],
                        &lin[1][span.clone()],
                        &lin[2][span.clone()],
                    ],
                    [&mut x[span.clone()], &mut y[span.clone()], &mut b[span]],
                );
            }
            // moments and maps depend on the test image within the blur radius
            self.refresh_moments_and_maps(s, rs, rs.expand(RADIUS, w, h));
        }
        Ok(())
    }

    /// Per scale, the rect `update` rewrites in the image planes and the
    /// (blur-expanded) rect it rewrites in moments and maps.
    fn update_rects(&self, rect: Rect) -> impl Iterator<Item = (Rect, Rect)> + use<> {
        let (mut w, mut h) = (self.width(), self.height());
        let mut rs = rect.clamp(w, h);
        let scales = if rs.is_empty() { 0 } else { self.scales() };
        (0..scales).map(move |s| {
            if s > 0 {
                (w, h) = (w.div_ceil(2), h.div_ceil(2));
                rs = rs.half(w, h);
            }
            (rs, rs.expand(RADIUS, w, h))
        })
    }

    /// [`update`](Self::update), first saving everything it overwrites into
    /// `checkpoint`, so that [`rollback`](Self::rollback) restores the
    /// previous state exactly (by copying).
    pub fn update_checkpointed(
        &mut self,
        test: RgbImage<'_>,
        rect: Rect,
        checkpoint: &mut Checkpoint,
    ) -> Result<(), Error> {
        if (test.width(), test.height()) != (self.width(), self.height()) {
            return Err(Error::DimensionMismatch);
        }
        let cp = checkpoint;
        cp.rect = None;
        let rect = rect.clamp(self.width(), self.height());
        if rect.is_empty() {
            return Ok(());
        }
        cp.rect = Some(rect);
        cp.sums.clone_from(&self.sums);
        let needed = self
            .update_rects(rect)
            .enumerate()
            .map(|(s, (rs, region))| {
                let pixels = (rs.x1 - rs.x0) * (rs.y1 - rs.y0);
                let halo = (region.x1 - region.x0) * (region.y1 - region.y0);
                let maps = self.maps[s]
                    .iter()
                    .flatten()
                    .filter(|p| !p.is_empty())
                    .count();
                6 * pixels + maps * halo
            })
            .sum();
        // Grow without zeroing on reuse: every saved sample is overwritten.
        if cp.data.len() < needed {
            cp.data.resize(needed, 0.0);
        }
        cp.data.truncate(needed);
        let mut at = 0;
        for (s, (rs, region)) in self.update_rects(rect).enumerate() {
            let w = self.dims(s).0;
            for c in 0..3 {
                at = save_rows(&self.lin[s][c], w, rs, &mut cp.data, at);
                at = save_rows(&self.xyb[s][c], w, rs, &mut cp.data, at);
                for map in &self.maps[s][c] {
                    at = save_rows(map, w, region, &mut cp.data, at);
                }
            }
        }
        debug_assert_eq!(at, needed);
        let result = self.update(test, rect);
        if result.is_err() {
            cp.rect = None;
            cp.data.clear();
        }
        result
    }

    /// Undoes the `update_checkpointed` that filled `checkpoint`.
    pub fn rollback(&mut self, checkpoint: &Checkpoint) {
        let cp = checkpoint;
        let Some(rect) = cp.rect else {
            return;
        };
        let mut at = 0usize;
        for (s, (rs, region)) in self.update_rects(rect).enumerate() {
            let w = self.dims(s).0;
            for c in 0..3 {
                at = load_rows(&mut self.lin[s][c], w, rs, &cp.data, at);
                at = load_rows(&mut self.xyb[s][c], w, rs, &cp.data, at);
                for m in 0..3 {
                    at = load_rows(&mut self.maps[s][c][m], w, region, &cp.data, at);
                }
                // The restored rows' filtered moments are stale; refilter on use.
                self.filtered[s][c].invalidate(rs.y0..rs.y1);
            }
        }
        debug_assert_eq!(at, cp.data.len());
        self.sums.clone_from(&cp.sums);
    }

    /// The per-pixel error map of the current image, as in
    /// [`PreparedReference::compare_with_map`].
    pub fn error_map(&self) -> Vec<f32> {
        let (w0, h0) = (self.width(), self.height());
        let mut map = vec![0f32; w0 * h0];
        let sum = self.weighted_sum();
        // A small row block keeps SIMD loops long without a full-frame accumulator.
        const ROWS: usize = 16;
        let mut acc = vec![0.0; w0 * ROWS.min(h0)];
        for s in 0..self.sums.len() {
            let (w, h) = self.dims(s);
            let pixels = w * h;
            let coefficients: [[Option<PooledContribution>; 3]; 3] = std::array::from_fn(|c| {
                let base = weight_base(c, s, self.scales());
                std::array::from_fn(|m| {
                    let (w1, w4) = (WEIGHTS[base + m], WEIGHTS[base + 3 + m]);
                    if w1 == 0.0 && w4 == 0.0 {
                        None
                    } else {
                        let n4 = (self.sums[s][c][m].1 / pixels as f64)
                            .max(0.0)
                            .sqrt()
                            .sqrt();
                        Some(PooledContribution::new(pixels, w1, w4, n4))
                    }
                })
            });
            let output_rows = (ROWS * (1 << s)).min(h0);
            for (block, rows) in map.chunks_mut(w0 * output_rows).enumerate() {
                let start = block * ROWS * w;
                let end = start + ROWS.min(h - block * ROWS) * w;
                let acc = &mut acc[..end - start];
                acc.fill(0.0);
                for (maps, coefficients) in self.maps[s].iter().zip(coefficients) {
                    for (values, coefficient) in maps.iter().zip(coefficients) {
                        if let Some(coefficient) = coefficient {
                            coefficient.add(self.kernels.error_map, &values[start..end], acc);
                        }
                    }
                }
                self.kernels
                    .error_map
                    .accumulate_scale_map(acc, w0, rows.len() / w0, s, rows);
            }
        }
        let factor = deficit_slope(sum) as f32 * (w0 * h0) as f32;
        for v in &mut map {
            *v *= factor;
        }
        map
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;
    use crate::Error;
    use crate::metric::{MapContribution, contribution_scalar};
    use crate::testing::{Planes, full, state_of};

    fn textured(seed: u32, w: usize, h: usize) -> Planes {
        Planes::textured(seed, w, h)
    }

    impl IncrementalState {
        fn deficit(&self) -> f64 {
            100.0 - self.score()
        }

        fn all_dims(&self) -> Vec<(usize, usize)> {
            (0..self.lin.len()).map(|s| self.dims(s)).collect()
        }
    }

    #[test]
    fn state_retains_only_weighted_fourth_norms_after_updates_and_reverts() {
        let (w, h) = (257, 259);
        let src = textured(7, w, h);
        let mut rec = textured(8, w, h);
        let mut state = state_of(&src, &rec);
        assert_eq!(state.lin.len(), crate::metric::NUM_SCALES);
        let check = |state: &IncrementalState| {
            for s in 0..state.lin.len() {
                for c in 0..3 {
                    let base = weight_base(c, s, state.scales());
                    for m in 0..3 {
                        let (s1, s4) = state.sums[s][c][m];
                        let expected = state.kernels.maps.map_sums(&state.maps[s][c][m]);
                        // The first sum also detects exact-zero maps when only
                        // their fourth norm is weighted.
                        assert!((s1 - expected.0).abs() <= 2e-12 * expected.0.abs().max(1.0));
                        if WEIGHTS[base + 3 + m] == 0.0 {
                            assert_eq!(s4, 0.0, "scale={s}, plane={c}, map={m}");
                        } else {
                            assert!((s4 - expected.1).abs() <= 2e-12 * expected.1.abs().max(1.0));
                        }
                    }
                }
            }
        };
        check(&state);
        let r = Rect {
            x0: w - 13,
            y0: h - 9,
            x1: w,
            y1: h,
        };
        for row in r.y0..r.y1 {
            for v in &mut rec.plane_row_mut(1, row)[r.x0..r.x1] {
                *v += 0.001;
            }
        }
        state.update(rec.image(), r).unwrap();
        check(&state);
        state.update(textured(8, w, h).image(), r).unwrap();
        check(&state);
    }

    #[test]
    fn fused_map_stays_close_to_unfused_contributions_and_expansion() {
        let mut max_relative_error = 0.0f32;
        for (w0, h0) in [(17, 19), (129, 67), (257, 259), (768, 513)] {
            let src = textured(7, w0, h0);
            for seed in [7, 8] {
                let rec = textured(seed, w0, h0);
                let state = state_of(&src, &rec);
                let mut expected = vec![0.0; w0 * h0];
                for (s, &(w, h)) in state.all_dims().iter().enumerate() {
                    let n = (w * h) as f64;
                    let mut acc = vec![0.0; w * h];
                    for c in 0..3 {
                        let base = weight_base(c, s, state.scales());
                        for m in 0..3 {
                            let (w1, w4) = (WEIGHTS[base + m], WEIGHTS[base + 3 + m]);
                            if w1 == 0.0 && w4 == 0.0 {
                                continue;
                            }
                            let n4 = (state.sums[s][c][m].1 / n).max(0.0).sqrt().sqrt();
                            let p = if w4 == 0.0 || n4 == 0.0 {
                                MapContribution {
                                    linear: (w1 / n) as f32,
                                    inv_norm: 0.0,
                                    norm_weight: 0.0,
                                }
                            } else {
                                let p = MapContribution {
                                    linear: (w1 / n) as f32,
                                    inv_norm: (1.0 / n4) as f32,
                                    norm_weight: (w4 * n4 / (4.0 * n)) as f32,
                                };
                                assert!(p.inv_norm.is_normal() && p.norm_weight.is_normal());
                                p
                            };
                            contribution_scalar(&state.maps[s][c][m], p, &mut acc);
                        }
                    }
                    // Preserve the previous per-output-pixel scale expansion
                    // as an independent oracle for indexing and arithmetic.
                    let step = 1 << s;
                    let share = 1.0 / (step * step) as f32;
                    for y in 0..h0 {
                        for x in 0..w0 {
                            expected[y * w0 + x] +=
                                acc[(y / step).min(h - 1) * w + (x / step).min(w - 1)] * share;
                        }
                    }
                }
                let factor = deficit_slope(state.weighted_sum()) as f32 * (w0 * h0) as f32;
                for v in &mut expected {
                    *v *= factor;
                }
                for (&actual, &reference) in state.error_map().iter().zip(&expected) {
                    let error = (actual - reference).abs() / reference.abs().max(f32::MIN_POSITIVE);
                    max_relative_error = max_relative_error.max(error);
                    assert!(error <= 1e-6, "{w0}x{h0}, seed={seed}: {error}");
                }
            }
        }
        eprintln!("FMA map max relative error vs unfused: {max_relative_error}");
    }

    #[test]
    fn incremental_update_matches_full_recompute() {
        let src = textured(7, 160, 96);
        let mut rec = textured(7, 160, 96);
        for y in 20..60 {
            for x in 40..90 {
                rec.plane_row_mut(1, y)[x] += 0.02;
            }
        }
        let mut st = state_of(&src, &rec);
        let (d_full, map_full) = full(&src, &rec);
        assert!(
            (st.deficit() - d_full).abs() < 1e-3,
            "{} vs {d_full}",
            st.deficit()
        );
        let map_st = st.error_map();
        let diff: f32 = map_st
            .iter()
            .zip(&map_full)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / map_full.len() as f32;
        // Both paths use the same SIMD blur and pooling convention.
        assert!(diff < 1e-2, "{diff}");
        // change a region and update incrementally
        for y in 48..80 {
            for x in 96..128 {
                rec.plane_row_mut(0, y)[x] += 0.01;
                rec.plane_row_mut(1, y)[x] -= 0.03;
            }
        }
        st.update(
            rec.image(),
            Rect {
                x0: 96,
                y0: 48,
                x1: 128,
                y1: 80,
            },
        )
        .unwrap();
        let (d2, _) = full(&src, &rec);
        assert!(
            (st.deficit() - d2).abs() < 1e-3,
            "incremental {} vs full {d2}",
            st.deficit()
        );
        // and back
        for y in 48..80 {
            for x in 96..128 {
                rec.plane_row_mut(0, y)[x] -= 0.01;
                rec.plane_row_mut(1, y)[x] += 0.03;
            }
        }
        st.update(
            rec.image(),
            Rect {
                x0: 96,
                y0: 48,
                x1: 128,
                y1: 80,
            },
        )
        .unwrap();
        assert!(
            (st.deficit() - d_full).abs() < 1e-3,
            "revert {} vs {d_full}",
            st.deficit()
        );
    }

    #[test]
    fn repeated_updates_match_rebuild_at_odd_edges_and_reuse_blur_scratch() {
        for (w, h) in [(9, 8), (31, 17), (257, 259)] {
            let src = textured(7, w, h);
            let mut rec = textured(8, w, h);
            let mut state = state_of(&src, &rec);
            // Warm the largest possible scratch before checking smaller updates.
            state
                .update(
                    rec.image(),
                    Rect {
                        x0: 0,
                        y0: 0,
                        x1: w,
                        y1: h,
                    },
                )
                .unwrap();
            let scratch = state.blur.scratch_layout();
            let regions = [
                Rect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                Rect {
                    x0: w - 1,
                    y0: h - 1,
                    x1: w,
                    y1: h,
                },
                Rect {
                    x0: w / 3,
                    y0: h / 3,
                    x1: w.div_ceil(2),
                    y1: h.div_ceil(2),
                },
                Rect {
                    x0: 0,
                    y0: h - 3,
                    x1: w,
                    y1: h,
                },
                Rect {
                    x0: w - 3,
                    y0: 0,
                    x1: w,
                    y1: h,
                },
            ];
            for r in regions.into_iter().cycle().take(15) {
                let saved = rec.clone();
                for c in 0..3 {
                    for y in r.y0..r.y1 {
                        for v in &mut rec.plane_row_mut(c, y)[r.x0..r.x1] {
                            *v += if c == 1 { -0.003 } else { 0.001 };
                        }
                    }
                }
                for image in [&rec, &saved] {
                    state.update(image.image(), r).unwrap();
                    let fresh = state_of(&src, image);
                    assert!(
                        (state.deficit() - fresh.deficit()).abs() < 1e-3,
                        "{w}x{h}, {r:?}: {} vs {}",
                        state.deficit(),
                        fresh.deficit()
                    );
                    for (actual, expected) in
                        state.lin.iter().flatten().zip(fresh.lin.iter().flatten())
                    {
                        assert_eq!(actual, expected, "linear pyramid: {w}x{h}, {r:?}");
                    }
                    for (actual, expected) in state
                        .maps
                        .iter()
                        .flatten()
                        .flatten()
                        .zip(fresh.maps.iter().flatten().flatten())
                    {
                        for (a, b) in actual.iter().zip(expected) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "maps: {w}x{h}, {r:?}: {a} vs {b}"
                            );
                        }
                    }
                    let map = state.error_map();
                    let fresh_map = fresh.error_map();
                    let diff = map
                        .iter()
                        .zip(&fresh_map)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f32>()
                        / (w * h) as f32;
                    assert!(diff < 1e-2, "map: {w}x{h}, {r:?}: {diff}");
                    assert_eq!(state.blur.scratch_layout(), scratch);
                }
                rec = saved;
            }
        }
    }

    #[test]
    fn weighted_state_matches_full_maps_across_all_six_scales() {
        let (w, h) = (257, 259);
        let src = textured(7, w, h);
        let mut rec = textured(8, w, h);
        let mut state = state_of(&src, &rec);
        assert_eq!(state.lin.len(), crate::metric::NUM_SCALES);
        let regions = [
            Rect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
            },
            Rect {
                x0: w - 7,
                y0: h - 5,
                x1: w,
                y1: h,
            },
            Rect {
                x0: 73,
                y0: 51,
                x1: 131,
                y1: 107,
            },
        ];
        for r in regions {
            let saved = rec.clone();
            for c in 0..3 {
                for y in r.y0..r.y1 {
                    for v in &mut rec.plane_row_mut(c, y)[r.x0..r.x1] {
                        *v += if c == 1 { -0.02 } else { 0.007 };
                    }
                }
            }
            for image in [&rec, &saved] {
                state.update(image.image(), r).unwrap();
                let (deficit, map) = full(&src, image);
                assert!(
                    (state.deficit() - deficit).abs() < 1e-4,
                    "{r:?}: {} vs {deficit}",
                    state.deficit()
                );
                let actual = state.error_map();
                let mean_error: f64 = actual
                    .iter()
                    .zip(&map)
                    .map(|(&a, &b)| (a as f64 - b as f64).abs())
                    .sum::<f64>()
                    / (w * h) as f64;
                assert!(mean_error < 1e-4, "{r:?}: map mean error {mean_error}");
            }
            rec = saved;
        }
    }

    #[test]
    fn small_images_are_rejected_and_outside_updates_are_empty() {
        for (w, h) in [(7, 9), (9, 7)] {
            let src = textured(7, w, h);
            assert_eq!(
                PreparedReference::new(src.image()).err(),
                Some(Error::ImageTooSmall)
            );
        }
        for (w, h) in [(8, 8), (9, 9), (15, 16)] {
            let src = textured(7, w, h);
            let mut state = state_of(&src, &src);
            state.update(src.image(), Rect::new(0, 0, w, h)).unwrap();
            state
                .update(src.image(), Rect::new(w, h, w + 1, h + 1))
                .unwrap();
            assert_eq!(state.deficit(), 0.0);
            assert!(state.error_map().iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn local_scratch_and_checkpoint_buffers_are_reused() {
        let (w, h) = (257, 259);
        let src = textured(7, w, h);
        let rec = textured(8, w, h);
        let mut state = state_of(&src, &rec);
        assert!(state.moments.iter().all(|p| p.capacity() == 0));
        let before = state.score().to_bits();
        let map_before = state.error_map();
        let mut cp = Checkpoint::default();
        let largest = Rect::new(103, 77, 135, 109);
        state
            .update_checkpointed(src.image(), largest, &mut cp)
            .unwrap();
        state.rollback(&cp);
        let checkpoint_layout = (cp.data.as_ptr(), cp.data.capacity(), cp.sums.as_ptr());
        let moments = state.moments.each_ref().map(|p| (p.as_ptr(), p.capacity()));
        assert!(
            state
                .moments
                .iter()
                .all(|p| p.len() <= (32 + 2 * RADIUS).pow(2))
        );
        assert!(cp.data.capacity() < w * h);
        for r in [
            largest,
            Rect::new(w - 7, h - 5, w + 9, h + 7),
            Rect::new(0, 0, 1, 1),
            Rect::new(w, h, w + 1, h + 1),
            Rect::new(10, 12, 5, 7),
        ] {
            state.update_checkpointed(src.image(), r, &mut cp).unwrap();
            state.rollback(&cp);
            assert_eq!(state.score().to_bits(), before);
            assert_eq!(state.error_map(), map_before);
            assert_eq!(
                (cp.data.as_ptr(), cp.data.capacity(), cp.sums.as_ptr()),
                checkpoint_layout
            );
            assert_eq!(
                state.moments.each_ref().map(|p| (p.as_ptr(), p.capacity())),
                moments
            );
        }
    }

    #[test]
    fn rollback_restores_the_exact_state() {
        let (w, h) = (131, 77);
        let src = textured(7, w, h);
        let rec = textured(8, w, h);
        let mut state = state_of(&src, &rec);
        let before = state.score();
        let map_before = state.error_map();
        let mut cand = rec.clone();
        let r = Rect::new(40, 20, 72, 52);
        for y in r.y0..r.y1 {
            for v in &mut cand.plane_row_mut(1, y)[r.x0..r.x1] {
                *v *= 0.8;
            }
        }
        let mut cp = Checkpoint::default();
        state.update_checkpointed(cand.image(), r, &mut cp).unwrap();
        assert_ne!(state.score(), before);
        state.rollback(&cp);
        assert_eq!(state.score().to_bits(), before.to_bits());
        assert_eq!(state.error_map(), map_before);
        // A rejected update leaves the state and the checkpoint inert.
        let mut bad = cand.clone();
        bad.plane_row_mut(0, 30)[50] = f32::NAN;
        assert_eq!(
            state.update_checkpointed(bad.image(), r, &mut cp),
            Err(Error::NonFiniteSample)
        );
        state.rollback(&cp);
        assert_eq!(state.score().to_bits(), before.to_bits());
    }

    #[test]
    fn repeated_local_reverts_preserve_near_identical_scores() {
        let (w, h) = (97, 81);
        let src = textured(7, w, h);
        let mut rec = src.clone();
        let mut state = state_of(&src, &rec);
        let initial = state.deficit();
        for iteration in 0..256 {
            let x = iteration * 13 % (w - 7);
            let y = iteration * 7 % (h - 5);
            let r = Rect {
                x0: x,
                y0: y,
                x1: x + 7,
                y1: y + 5,
            };
            for row in r.y0..r.y1 {
                for v in &mut rec.plane_row_mut(1, row)[r.x0..r.x1] {
                    *v += 0.001;
                }
            }
            state.update(rec.image(), r).unwrap();
            if iteration % 32 == 0 {
                let (full, _) = full(&src, &rec);
                assert!(
                    (state.deficit() - full).abs() < 1e-4,
                    "{iteration}: {} vs {full}",
                    state.deficit()
                );
            }
            state.update(src.image(), r).unwrap();
            for row in r.y0..r.y1 {
                rec.plane_row_mut(1, row)[r.x0..r.x1]
                    .copy_from_slice(&src.plane_row(1, row)[r.x0..r.x1]);
            }
            assert!(
                (state.deficit() - initial).abs() < 1e-4,
                "revert {iteration}: {} vs {initial}",
                state.deficit()
            );
        }
    }
}
