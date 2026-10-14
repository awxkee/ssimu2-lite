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
//! pyramids, its blurred moments, every weighted map and the pooled sums. An
//! update recomputes only the changed area (plus the blur halo) at every
//! scale and adjusts the sums; the score then costs a few hundred flops.

use crate::blur::GaussianBlur;
use crate::downsample::downsample_rows;
use crate::metric::{
    PreparedReference, RADIUS, ReferenceData, WEIGHTS, accumulate_scale_map, active_maps,
    add_map_contribution, deficit_slope, edge_ratio, kernel, replace_map, score_from_sum,
    ssim_value, weight_base,
};
use crate::xyb::positive_xyb_into;
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

/// Everything one [`IncrementalState::update_checkpointed`] overwrote.
/// Reuse one checkpoint across candidates to keep its allocation.
#[derive(Default)]
pub struct Checkpoint {
    rects: Vec<(Rect, Rect)>,
    data: Vec<f32>,
    sums: Vec<[[(f64, f64); 3]; 3]>,
}

fn save_rows(buf: &[f32], w: usize, r: Rect, out: &mut Vec<f32>) {
    if buf.is_empty() {
        return;
    }
    for y in r.y0..r.y1 {
        out.extend_from_slice(&buf[y * w + r.x0..y * w + r.x1]);
    }
}

fn load_rows(buf: &mut [f32], w: usize, r: Rect, data: &[f32], mut at: usize) -> usize {
    if buf.is_empty() {
        return at;
    }
    let n = r.x1 - r.x0;
    for y in r.y0..r.y1 {
        buf[y * w + r.x0..y * w + r.x1].copy_from_slice(&data[at..at + n]);
        at += n;
    }
    at
}

/// SSIMULACRA 2 of a fixed reference against a test image that changes
/// locally: after [`update`](Self::update) of a rectangle only that rectangle
/// and its blur halo are recomputed at each scale. Scores match a full
/// [`PreparedReference::compare`] up to float summation order.
pub struct IncrementalState {
    reference: Arc<ReferenceData>,
    /// Test image's linear RGB planes per scale: `[scale][channel]`.
    lin: Vec<[Vec<f32>; 3]>,
    /// Test image's positive XYB planes per scale.
    xyb: Vec<[Vec<f32>; 3]>,
    /// Blurred test moments per scale and plane. Unweighted planes are empty;
    /// squared/cross moments are empty unless the SSIM map is weighted.
    mu2: Vec<[Vec<f32>; 3]>,
    s22: Vec<[Vec<f32>; 3]>,
    s12: Vec<[Vec<f32>; 3]>,
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
        let mut lin = test.to_linear()?;
        let mut st = IncrementalState {
            blur: GaussianBlur::with_kernel(data.width, *kernel(), true),
            reference: data,
            lin: Vec::new(),
            xyb: Vec::new(),
            mu2: Vec::new(),
            s22: Vec::new(),
            s12: Vec::new(),
            maps: Vec::new(),
            sums: Vec::new(),
        };
        let scales = st.reference.scales.len();
        for s in 0..scales {
            let (w, h) = st.dims(s);
            if s > 0 {
                let (pw, ph) = st.dims(s - 1);
                let previous = &st.lin[s - 1];
                lin =
                    std::array::from_fn(|c| crate::downsample::downsample(&previous[c], pw, ph).0);
            }
            let n = w * h;
            let mut xyb = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
            {
                let [x, y, b] = &mut xyb;
                positive_xyb_into([&lin[0], &lin[1], &lin[2]], [x, y, b]);
            }
            st.lin.push(std::mem::take(&mut lin));
            st.xyb.push(xyb);
            let active: [[bool; 3]; 3] = std::array::from_fn(|c| active_maps(c, s, scales));
            let buffer = |needed| if needed { vec![0.0; n] } else { Vec::new() };
            st.mu2
                .push(std::array::from_fn(|c| buffer(active[c].contains(&true))));
            st.s22.push(std::array::from_fn(|c| buffer(active[c][0])));
            st.s12.push(std::array::from_fn(|c| buffer(active[c][0])));
            st.maps.push(std::array::from_fn(|c| {
                std::array::from_fn(|m| buffer(active[c][m]))
            }));
            st.sums.push([[(0.0, 0.0); 3]; 3]);
            st.blur.set_width(w);
            for c in 0..3 {
                if !active[c].contains(&true) {
                    continue;
                }
                let rec = &st.xyb[s][c];
                st.blur.blur(rec, h, &mut st.mu2[s][c]);
                if active[c][0] {
                    let src = &st.reference.scales[s].xyb[c];
                    st.blur.blur_product([rec, rec], h, &mut st.s22[s][c]);
                    st.blur.blur_product([src, rec], h, &mut st.s12[s][c]);
                }
            }
            st.refresh_maps(s, Rect::new(0, 0, w, h));
        }
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
    /// `r` (whose inputs changed within `r` shrunk by the blur radius).
    fn refresh_moments_and_maps(&mut self, s: usize, r: Rect) {
        let (w, h) = self.dims(s);
        if r.is_empty() {
            return;
        }
        self.blur.set_width(r.x1 - r.x0);
        let offset = r.y0 * w + r.x0;
        let region = [r.x0..r.x1, r.y0..r.y1];
        for c in 0..3 {
            let active = active_maps(c, s, self.scales());
            if !active.contains(&true) {
                continue;
            }
            let rec = &self.xyb[s][c];
            let src = &self.reference.scales[s].xyb[c];
            self.blur
                .blur_region(rec, w, h, region.clone(), &mut self.mu2[s][c][offset..], w);
            if !active[0] {
                continue;
            }
            self.blur.blur_product_region(
                [rec, rec],
                w,
                h,
                region.clone(),
                &mut self.s22[s][c][offset..],
                w,
            );
            self.blur.blur_product_region(
                [src, rec],
                w,
                h,
                region.clone(),
                &mut self.s12[s][c][offset..],
                w,
            );
        }
        self.refresh_maps(s, r);
    }

    /// Recompute the weighted maps over `r` from their current moments.
    fn refresh_maps(&mut self, s: usize, r: Rect) {
        let (w, _) = self.dims(s);
        let reference = &self.reference.scales[s];
        for c in 0..3 {
            let active = active_maps(c, s, self.scales());
            if !active.contains(&true) {
                continue;
            }
            let base = weight_base(c, s, self.scales());
            let need_fourth: [bool; 3] = std::array::from_fn(|m| WEIGHTS[base + 3 + m] != 0.0);
            let rec = &self.xyb[s][c];
            let src = &reference.xyb[c];
            let mu1 = &reference.mu[c];
            let s11 = &reference.sigma[c];
            let mu2 = &self.mu2[s][c];
            let s22 = &self.s22[s][c];
            let s12 = &self.s12[s][c];
            let [ssim, artifact, detail] = &mut self.maps[s][c];
            for y in r.y0..r.y1 {
                let span = y * w + r.x0..y * w + r.x1;
                if active[0] {
                    replace_map(
                        &mut ssim[span.clone()],
                        [
                            &mu1[span.clone()],
                            &mu2[span.clone()],
                            &s11[span.clone()],
                            &s22[span.clone()],
                            &s12[span.clone()],
                        ],
                        &mut self.sums[s][c][0],
                        need_fourth[0],
                        |[m1, m2, v11, v22, v12]| ssim_value(m1, m2, v11, v22, v12),
                    );
                }
                let inputs = [
                    &src[span.clone()],
                    &rec[span.clone()],
                    &mu1[span.clone()],
                    &mu2[span.clone()],
                ];
                if active[1] {
                    replace_map(
                        &mut artifact[span.clone()],
                        inputs,
                        &mut self.sums[s][c][1],
                        need_fourth[1],
                        |[a, b, m1, m2]| edge_ratio(a, b, m1, m2).max(0.0),
                    );
                }
                if active[2] {
                    replace_map(
                        &mut detail[span],
                        inputs,
                        &mut self.sums[s][c][2],
                        need_fourth[2],
                        |[a, b, m1, m2]| (-edge_ratio(a, b, m1, m2)).max(0.0),
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
            test.linear_into(r0, [r, g, b], w0)?;
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
                        downsample_rows(
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
                positive_xyb_into(
                    [
                        &lin[0][span.clone()],
                        &lin[1][span.clone()],
                        &lin[2][span.clone()],
                    ],
                    [&mut x[span.clone()], &mut y[span.clone()], &mut b[span]],
                );
            }
            // moments and maps depend on the test image within the blur radius
            self.refresh_moments_and_maps(s, rs.expand(RADIUS, w, h));
        }
        Ok(())
    }

    /// Per scale, the rect `update` rewrites in the image planes and the
    /// (blur-expanded) rect it rewrites in moments and maps.
    fn update_rects(&self, rect: Rect) -> Vec<(Rect, Rect)> {
        let r0 = rect.clamp(self.width(), self.height());
        let mut out = Vec::with_capacity(self.lin.len());
        if r0.is_empty() {
            return out;
        }
        let mut rs = r0;
        for s in 0..self.lin.len() {
            let (w, h) = self.dims(s);
            if s > 0 {
                rs = rs.half(w, h);
            }
            out.push((rs, rs.expand(RADIUS, w, h)));
        }
        out
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
        cp.data.clear();
        cp.rects = self.update_rects(rect);
        cp.sums.clone_from(&self.sums);
        for (s, &(rs, region)) in cp.rects.iter().enumerate() {
            let w = self.dims(s).0;
            for c in 0..3 {
                save_rows(&self.lin[s][c], w, rs, &mut cp.data);
                save_rows(&self.xyb[s][c], w, rs, &mut cp.data);
                for buf in [&self.mu2[s][c], &self.s22[s][c], &self.s12[s][c]] {
                    save_rows(buf, w, region, &mut cp.data);
                }
                for map in &self.maps[s][c] {
                    save_rows(map, w, region, &mut cp.data);
                }
            }
        }
        let result = self.update(test, rect);
        if result.is_err() {
            cp.rects.clear();
            cp.data.clear();
        }
        result
    }

    /// Undoes the `update_checkpointed` that filled `checkpoint`.
    pub fn rollback(&mut self, checkpoint: &Checkpoint) {
        let cp = checkpoint;
        if cp.rects.is_empty() {
            return;
        }
        let mut at = 0usize;
        for (s, &(rs, region)) in cp.rects.iter().enumerate() {
            let w = self.dims(s).0;
            for c in 0..3 {
                at = load_rows(&mut self.lin[s][c], w, rs, &cp.data, at);
                at = load_rows(&mut self.xyb[s][c], w, rs, &cp.data, at);
                at = load_rows(&mut self.mu2[s][c], w, region, &cp.data, at);
                at = load_rows(&mut self.s22[s][c], w, region, &cp.data, at);
                at = load_rows(&mut self.s12[s][c], w, region, &cp.data, at);
                for m in 0..3 {
                    at = load_rows(&mut self.maps[s][c][m], w, region, &cp.data, at);
                }
            }
        }
        self.sums.clone_from(&cp.sums);
    }

    /// The per-pixel error map of the current image, as in
    /// [`PreparedReference::compare_with_map`].
    pub fn error_map(&self) -> Vec<f32> {
        let (w0, h0) = (self.width(), self.height());
        let mut map = vec![0f32; w0 * h0];
        let sum = self.weighted_sum();
        let mut acc = Vec::new();
        for s in 0..self.sums.len() {
            let (w, h) = self.dims(s);
            let n = (w * h) as f64;
            acc.clear();
            acc.resize(w * h, 0.0);
            for c in 0..3 {
                let base = weight_base(c, s, self.scales());
                for m in 0..3 {
                    if WEIGHTS[base + m] == 0.0 && WEIGHTS[base + 3 + m] == 0.0 {
                        continue;
                    }
                    let n4 = (self.sums[s][c][m].1 / n).max(0.0).sqrt().sqrt();
                    add_map_contribution(
                        &self.maps[s][c][m],
                        WEIGHTS[base + m],
                        WEIGHTS[base + 3 + m],
                        n4,
                        &mut acc,
                    );
                }
            }
            accumulate_scale_map(&acc, w0, h0, s, &mut map);
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
    use crate::metric::{MapContribution, contribution_scalar, map_sums};
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
                        let expected = map_sums(&state.maps[s][c][m]);
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
                    for (actual, expected) in [&state.mu2, &state.s22, &state.s12]
                        .into_iter()
                        .zip([&fresh.mu2, &fresh.s22, &fresh.s12])
                    {
                        for (a, b) in actual
                            .iter()
                            .flatten()
                            .flatten()
                            .zip(expected.iter().flatten().flatten())
                        {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "moments: {w}x{h}, {r:?}: {a} vs {b}"
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
