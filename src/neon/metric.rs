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
use crate::maps::{MapInputs, MapSums};
use crate::metric::C2;
use crate::metric::MapContribution;
use std::arch::aarch64::*;

/// # Safety
/// NEON must be available and the input/output slices must have equal lengths.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ss2_contribution_neon(
    map: &[f32],
    p: MapContribution,
    acc: &mut [f32],
) -> usize {
    debug_assert_eq!(map.len(), acc.len());
    if p.norm_weight == 0.0 {
        let done = contribution::<4, false>(map, p, acc);
        done + contribution::<1, false>(&map[done..], p, &mut acc[done..])
    } else {
        let done = contribution::<4, true>(map, p, acc);
        done + contribution::<1, true>(&map[done..], p, &mut acc[done..])
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn contribution<const N: usize, const FOURTH: bool>(
    map: &[f32],
    p: MapContribution,
    acc: &mut [f32],
) -> usize {
    let (map_vectors, _) = map.as_chunks::<4>();
    let (acc_vectors, _) = acc.as_chunks_mut::<4>();
    let (map_blocks, _) = map_vectors.as_chunks::<N>();
    let (acc_blocks, _) = acc_vectors.as_chunks_mut::<N>();
    let done = acc_blocks.len() * N * 4;
    for (map, acc) in map_blocks.iter().zip(acc_blocks) {
        for (map, acc) in map.iter().zip(acc) {
            // The entry point's equal-length contract covers complete vectors.
            unsafe {
                let v = vld1q_f32(map.as_ptr());
                let old = vld1q_f32(acc.as_ptr());
                let out = if FOURTH {
                    let normalized = vmulq_n_f32(v, p.inv_norm);
                    let square = vmulq_f32(normalized, normalized);
                    let weighted = vmulq_n_f32(square, p.norm_weight);
                    let first = vmulq_n_f32(v, p.linear);
                    vaddq_f32(old, vfmaq_f32(first, weighted, square))
                } else {
                    vfmaq_n_f32(old, v, p.linear)
                };
                vst1q_f32(acc.as_mut_ptr(), out);
            }
        }
    }
    done
}

/// # Safety
/// NEON must be available, STEP must be 1/2/4/8/16/32, and src must contain
/// ceil(dst.len()/STEP) pixels. Only complete vector-sized runs are processed.
#[target_feature(enable = "neon")]
pub(crate) fn ss2_expand_row_neon<const STEP: usize>(
    src: &[f32],
    dst: &mut [f32],
    share: f32,
) -> usize {
    // Fixed-size runs leave cropped tails for the scalar caller.
    unsafe {
        if STEP == 1 {
            let (runs, _) = dst.as_chunks_mut::<4>();
            let done = runs.len() * 4;
            for (dst, src) in runs.iter_mut().zip(src.as_chunks::<4>().0) {
                let value = vld1q_f32(src.as_ptr());
                let old = vld1q_f32(dst.as_ptr());
                vst1q_f32(dst.as_mut_ptr(), vfmaq_n_f32(old, value, share));
            }
            done
        } else if STEP == 2 {
            let (runs, _) = dst.as_chunks_mut::<8>();
            let done = runs.len() * 8;
            for (dst, src) in runs.iter_mut().zip(src.as_chunks::<4>().0) {
                let v = vld1q_f32(src.as_ptr());
                let a = vzip1q_f32(v, v);
                let b = vzip2q_f32(v, v);
                for (dst, value) in dst.as_chunks_mut::<4>().0.iter_mut().zip([a, b]) {
                    let old = vld1q_f32(dst.as_ptr());
                    vst1q_f32(dst.as_mut_ptr(), vfmaq_n_f32(old, value, share));
                }
            }
            done
        } else {
            let (runs, _) = dst.as_chunks_mut::<STEP>();
            let done = runs.len() * STEP;
            for (run, &value) in runs.iter_mut().zip(src) {
                let value = vdupq_n_f32(value);
                for dst in run.as_chunks_mut::<4>().0 {
                    let old = vld1q_f32(dst.as_ptr());
                    vst1q_f32(dst.as_mut_ptr(), vfmaq_n_f32(old, value, share));
                }
            }
            done
        }
    }
}

/// # Safety
/// NEON must be available; every input must match the output length.
/// REPLACE accumulates deltas into the eight f64 lanes; FOURTH also updates x⁴.
#[target_feature(enable = "neon")]
pub(crate) fn ss2_map_neon<const KIND: usize, const REPLACE: bool, const FOURTH: bool>(
    inputs: MapInputs<'_>,
    map: &mut [f32],
    sums: &mut MapSums,
) -> usize {
    let (blocks, tail) = map.as_chunks_mut::<8>();
    let (tail, _) = tail.as_chunks_mut::<4>();
    let done = blocks.len() * 8 + tail.len() * 4;
    // Eight-pixel blocks preserve the reduction's lane order.
    unsafe {
        let first_chunks = sums.first.as_chunks::<2>().0;
        let mut first: [float64x2_t; 4] =
            std::array::from_fn(|i| vld1q_f64(first_chunks[i].as_ptr()));
        let fourth_chunks = sums.fourth.as_chunks::<2>().0;
        let mut fourth: [float64x2_t; 4] =
            std::array::from_fn(|i| vld1q_f64(fourth_chunks[i].as_ptr()));
        for (block, dst) in blocks.iter_mut().enumerate() {
            for (vector, dst) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let i = block * 8 + vector * 4;
                let new = map_value::<KIND>(inputs.map(|p| p.as_ptr().add(i)));
                if REPLACE {
                    let old = vld1q_f32(dst.as_ptr());
                    reduce::<true, FOURTH>(new, old, vector * 2, &mut first, &mut fourth);
                }
                vst1q_f32(dst.as_mut_ptr(), new);
            }
        }
        for dst in tail {
            let i = done - 4;
            let new = map_value::<KIND>(inputs.map(|p| p.as_ptr().add(i)));
            if REPLACE {
                let old = vld1q_f32(dst.as_ptr());
                reduce::<true, FOURTH>(new, old, 0, &mut first, &mut fourth);
            }
            vst1q_f32(dst.as_mut_ptr(), new);
        }
        if REPLACE {
            for (dst, value) in sums.first.as_chunks_mut::<2>().0.iter_mut().zip(first) {
                vst1q_f64(dst.as_mut_ptr(), value);
            }
            if FOURTH {
                for (dst, value) in sums.fourth.as_chunks_mut::<2>().0.iter_mut().zip(fourth) {
                    vst1q_f64(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}

#[inline]
#[target_feature(enable = "neon")]
fn map_value<const KIND: usize>(input: [*const f32; 5]) -> float32x4_t {
    // The caller provides a complete vector at each pointer.
    unsafe {
        let [a, b, c, d] = [input[0], input[1], input[2], input[3]].map(|p| vld1q_f32(p));
        let one = vdupq_n_f32(1.0);
        let value = if KIND == 0 {
            let e = vld1q_f32(input[4]);
            let diff = vsubq_f32(a, b);
            let num_m = vsubq_f32(one, vmulq_f32(diff, diff));
            let num_s = vaddq_f32(
                vmulq_f32(vdupq_n_f32(2.0), vsubq_f32(e, vmulq_f32(a, b))),
                vdupq_n_f32(C2),
            );
            let denom = vaddq_f32(
                vaddq_f32(vsubq_f32(c, vmulq_f32(a, a)), vsubq_f32(d, vmulq_f32(b, b))),
                vdupq_n_f32(C2),
            );
            vsubq_f32(one, vdivq_f32(vmulq_f32(num_m, num_s), denom))
        } else {
            let num = vaddq_f32(one, vabsq_f32(vsubq_f32(b, d)));
            let denom = vaddq_f32(one, vabsq_f32(vsubq_f32(a, c)));
            let ratio = vsubq_f32(vdivq_f32(num, denom), one);
            if KIND == 1 {
                ratio
            } else {
                vsubq_f32(vdupq_n_f32(0.0), ratio)
            }
        };
        vmaxnmq_f32(value, vdupq_n_f32(0.0))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn reduce<const REPLACE: bool, const FOURTH: bool>(
    new: float32x4_t,
    old: float32x4_t,
    lane: usize,
    first: &mut [float64x2_t; 4],
    fourth: &mut [float64x2_t; 4],
) {
    let new = [vcvt_f64_f32(vget_low_f32(new)), vcvt_high_f64_f32(new)];
    let old = [vcvt_f64_f32(vget_low_f32(old)), vcvt_high_f64_f32(old)];
    for (half, (v, old)) in new.into_iter().zip(old).enumerate() {
        let delta = if REPLACE { vsubq_f64(v, old) } else { v };
        first[lane + half] = vaddq_f64(first[lane + half], delta);
        if FOURTH {
            let square = vmulq_f64(v, v);
            let mut delta = vmulq_f64(square, square);
            if REPLACE {
                let square = vmulq_f64(old, old);
                delta = vsubq_f64(delta, vmulq_f64(square, square));
            }
            fourth[lane + half] = vaddq_f64(fourth[lane + half], delta);
        }
    }
}

/// # Safety
/// NEON must be available. Only complete eight-pixel blocks are read.
#[target_feature(enable = "neon")]
pub(crate) fn ss2_sums_neon(map: &[f32], sums: &mut MapSums) -> usize {
    let (blocks, _) = map.as_chunks::<8>();
    // Complete blocks fit the source, and the sums hold exactly eight f64 lanes.
    unsafe {
        let first_chunks = sums.first.as_chunks::<2>().0;
        let mut first: [float64x2_t; 4] =
            std::array::from_fn(|i| vld1q_f64(first_chunks[i].as_ptr()));
        let fourth_chunks = sums.fourth.as_chunks::<2>().0;
        let mut fourth: [float64x2_t; 4] =
            std::array::from_fn(|i| vld1q_f64(fourth_chunks[i].as_ptr()));
        for block in blocks {
            for (vector, src) in block.as_chunks::<4>().0.iter().enumerate() {
                let value = vld1q_f32(src.as_ptr());
                reduce::<false, true>(value, vdupq_n_f32(0.0), vector * 2, &mut first, &mut fourth);
            }
        }
        for (dst, value) in sums.first.as_chunks_mut::<2>().0.iter_mut().zip(first) {
            vst1q_f64(dst.as_mut_ptr(), value);
        }
        for (dst, value) in sums.fourth.as_chunks_mut::<2>().0.iter_mut().zip(fourth) {
            vst1q_f64(dst.as_mut_ptr(), value);
        }
    }
    blocks.len() * 8
}

/// # Safety
/// NEON must be available and all input/output slices must have equal lengths.
/// The signed edge ratio is computed once for both maps and their f64 deltas.
#[target_feature(enable = "neon")]
pub(crate) fn ss2_edges_neon<
    const REPLACE: bool,
    const ARTIFACT_FOURTH: bool,
    const DETAIL_FOURTH: bool,
>(
    inputs: MapInputs<'_>,
    [artifact, detail]: [&mut [f32]; 2],
    sums: &mut [MapSums; 2],
) -> usize {
    let (artifact_blocks, artifact_tail) = artifact.as_chunks_mut::<8>();
    let (detail_blocks, detail_tail) = detail.as_chunks_mut::<8>();
    let (artifact_tail, _) = artifact_tail.as_chunks_mut::<4>();
    let (detail_tail, _) = detail_tail.as_chunks_mut::<4>();
    let done = artifact_blocks.len() * 8 + artifact_tail.len() * 4;
    // Keep all eight reduction lanes grouped, including four-wide backends.
    unsafe {
        let af_chunks = sums[0].first.as_chunks::<2>().0;
        let mut af: [float64x2_t; 4] = std::array::from_fn(|i| vld1q_f64(af_chunks[i].as_ptr()));
        let a4_chunks = sums[0].fourth.as_chunks::<2>().0;
        let mut a4: [float64x2_t; 4] = std::array::from_fn(|i| vld1q_f64(a4_chunks[i].as_ptr()));
        let df_chunks = sums[1].first.as_chunks::<2>().0;
        let mut df: [float64x2_t; 4] = std::array::from_fn(|i| vld1q_f64(df_chunks[i].as_ptr()));
        let d4_chunks = sums[1].fourth.as_chunks::<2>().0;
        let mut d4: [float64x2_t; 4] = std::array::from_fn(|i| vld1q_f64(d4_chunks[i].as_ptr()));
        let one = vdupq_n_f32(1.0);
        for (block, (artifact, detail)) in artifact_blocks.iter_mut().zip(detail_blocks).enumerate()
        {
            let artifact = artifact.as_chunks_mut::<4>().0;
            let detail = detail.as_chunks_mut::<4>().0;
            for (vector, (artifact, detail)) in artifact.iter_mut().zip(detail).enumerate() {
                let i = block * 8 + vector * 4;
                let [a, b, c, d] = [inputs[0], inputs[1], inputs[2], inputs[3]]
                    .map(|p| vld1q_f32(p.as_ptr().add(i)));
                let ratio = vsubq_f32(
                    vdivq_f32(
                        vaddq_f32(one, vabsq_f32(vsubq_f32(b, d))),
                        vaddq_f32(one, vabsq_f32(vsubq_f32(a, c))),
                    ),
                    one,
                );
                let a = vmaxnmq_f32(ratio, vdupq_n_f32(0.0));
                let d = vmaxnmq_f32(vnegq_f32(ratio), vdupq_n_f32(0.0));
                if REPLACE {
                    let lane = vector * 2;
                    reduce::<true, ARTIFACT_FOURTH>(
                        a,
                        vld1q_f32(artifact.as_ptr()),
                        lane,
                        &mut af,
                        &mut a4,
                    );
                    reduce::<true, DETAIL_FOURTH>(
                        d,
                        vld1q_f32(detail.as_ptr()),
                        lane,
                        &mut df,
                        &mut d4,
                    );
                }
                vst1q_f32(artifact.as_mut_ptr(), a);
                vst1q_f32(detail.as_mut_ptr(), d);
            }
        }
        for (artifact, detail) in artifact_tail.iter_mut().zip(detail_tail) {
            let i = done - 4;
            let [a, b, c, d] =
                [inputs[0], inputs[1], inputs[2], inputs[3]].map(|p| vld1q_f32(p.as_ptr().add(i)));
            let ratio = vsubq_f32(
                vdivq_f32(
                    vaddq_f32(one, vabsq_f32(vsubq_f32(b, d))),
                    vaddq_f32(one, vabsq_f32(vsubq_f32(a, c))),
                ),
                one,
            );
            let a = vmaxnmq_f32(ratio, vdupq_n_f32(0.0));
            let d = vmaxnmq_f32(vnegq_f32(ratio), vdupq_n_f32(0.0));
            if REPLACE {
                let lane = 0;
                reduce::<true, ARTIFACT_FOURTH>(
                    a,
                    vld1q_f32(artifact.as_ptr()),
                    lane,
                    &mut af,
                    &mut a4,
                );
                reduce::<true, DETAIL_FOURTH>(
                    d,
                    vld1q_f32(detail.as_ptr()),
                    lane,
                    &mut df,
                    &mut d4,
                );
            }
            vst1q_f32(artifact.as_mut_ptr(), a);
            vst1q_f32(detail.as_mut_ptr(), d);
        }
        if REPLACE {
            for (dst, value) in sums[0].first.as_chunks_mut::<2>().0.iter_mut().zip(af) {
                vst1q_f64(dst.as_mut_ptr(), value);
            }
            for (dst, value) in sums[1].first.as_chunks_mut::<2>().0.iter_mut().zip(df) {
                vst1q_f64(dst.as_mut_ptr(), value);
            }
            if ARTIFACT_FOURTH {
                for (dst, value) in sums[0].fourth.as_chunks_mut::<2>().0.iter_mut().zip(a4) {
                    vst1q_f64(dst.as_mut_ptr(), value);
                }
            }
            if DETAIL_FOURTH {
                for (dst, value) in sums[1].fourth.as_chunks_mut::<2>().0.iter_mut().zip(d4) {
                    vst1q_f64(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}
