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
// Adapted from libblur's symmetric f32 filter1d kernels.
use crate::blur::{BlurRows, MomentRows, RADIUS, WINDOW};
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn blur_horizontal_neon(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    assert!(src.len() >= dst.len() + 2 * RADIUS);
    let done = horizontal::<4>(src, k, dst);
    let done = done + horizontal::<1>(&src[done..], k, &mut dst[done..]);
    if done < dst.len() && done >= 4 {
        // Recompute the last full vector to cover the short tail in bounds.
        let start = dst.len() - 4;
        horizontal::<1>(&src[start..], k, &mut dst[start..]);
        dst.len()
    } else {
        done
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn horizontal<const N: usize>(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let center = vdupq_n_f32(k[RADIUS]);
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    let done = blocks.len() * N * LANES;
    for (block, dst) in blocks.iter_mut().enumerate() {
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [float32x4_t; N] = std::array::from_fn(|lane| {
                vmulq_f32(
                    vld1q_f32(src.as_ptr().add(x + lane * LANES + RADIUS)),
                    center,
                )
            });
            for (i, &weight) in k[..RADIUS].iter().enumerate() {
                let coeff = vdupq_n_f32(weight);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = vld1q_f32(src.as_ptr().add(x + lane * LANES + i));
                    let b = vld1q_f32(src.as_ptr().add(x + lane * LANES + WINDOW - 1 - i));
                    *acc = vaddq_f32(*acc, vmulq_f32(vaddq_f32(a, b), coeff));
                }
            }
            for (dst, acc) in dst.iter_mut().zip(acc) {
                vst1q_f32(dst.as_mut_ptr(), acc);
            }
        }
    }
    done
}

#[target_feature(enable = "neon")]
pub(crate) fn blur_vertical_neon(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|row| row.len() >= dst.len()));
    let done = vertical::<4>(rows, k, dst);
    let done = done + vertical::<1>(&rows.map(|row| &row[done..]), k, &mut dst[done..]);
    if done < dst.len() && done >= 4 {
        let start = dst.len() - 4;
        vertical::<1>(&rows.map(|row| &row[start..]), k, &mut dst[start..]);
        dst.len()
    } else {
        done
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn vertical<const N: usize>(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let center = vdupq_n_f32(k[RADIUS]);
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    let done = blocks.len() * N * LANES;
    for (block, dst) in blocks.iter_mut().enumerate() {
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [float32x4_t; N] = std::array::from_fn(|lane| {
                vmulq_f32(
                    vld1q_f32(rows[RADIUS].as_ptr().add(x + lane * LANES)),
                    center,
                )
            });
            for (i, &weight) in k[..RADIUS].iter().enumerate() {
                let coeff = vdupq_n_f32(weight);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = vld1q_f32(rows[i].as_ptr().add(x + lane * LANES));
                    let b = vld1q_f32(rows[WINDOW - 1 - i].as_ptr().add(x + lane * LANES));
                    *acc = vaddq_f32(*acc, vmulq_f32(vaddq_f32(a, b), coeff));
                }
            }
            for (dst, acc) in dst.iter_mut().zip(acc) {
                vst1q_f32(dst.as_mut_ptr(), acc);
            }
        }
    }
    done
}

/// # Safety
/// NEON must be available and all three slices must have equal lengths.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn blur_product_row_neon([a, b]: [&[f32]; 2], dst: &mut [f32]) -> usize {
    let (vectors, _) = dst.as_chunks_mut::<4>();
    let done = vectors.len() * 4;
    for ((dst, a), b) in vectors
        .iter_mut()
        .zip(a.as_chunks::<4>().0)
        .zip(b.as_chunks::<4>().0)
    {
        // Equal lengths ensure that every complete vector is in bounds.
        unsafe {
            vst1q_f32(
                dst.as_mut_ptr(),
                vmulq_f32(vld1q_f32(a.as_ptr()), vld1q_f32(b.as_ptr())),
            );
        }
    }
    done
}

/// # Safety
/// NEON must be available, inputs must have equal lengths, and every
/// output must match that length. Products round before filtering.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn blur_moment_row_neon(
    [reference, test]: [&[f32]; 2],
    mut dst: [&mut [f32]; 3],
) -> usize {
    let n = dst[0].len();
    let [mean, square, cross] = dst.each_mut().map(|p| p.as_chunks_mut::<4>().0);
    let done = mean.len() * 4;
    let reference_vectors = reference.as_chunks::<4>().0;
    let test_vectors = test.as_chunks::<4>().0;
    let inputs = reference_vectors.iter().zip(test_vectors);
    let outputs = mean.iter_mut().zip(square).zip(cross);
    for ((reference, test), ((mean, square), cross)) in inputs.zip(outputs) {
        // Fixed-size chunks cover both loads and all three stores.
        unsafe {
            moment_row_vector(
                reference.as_ptr(),
                test.as_ptr(),
                [mean.as_mut_ptr(), square.as_mut_ptr(), cross.as_mut_ptr()],
            );
        }
    }
    if done < n && done >= 4 {
        let start = n - 4;
        // An overlapping last vector covers the tail without reading past either input.
        unsafe {
            moment_row_vector(
                reference.as_ptr().add(start),
                test.as_ptr().add(start),
                dst.each_mut().map(|p| p.as_mut_ptr().add(start)),
            );
        }
        n
    } else {
        done
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn moment_row_vector(reference: *const f32, test: *const f32, dst: [*mut f32; 3]) {
    // The caller provides one full vector behind every pointer.
    unsafe {
        let src = vld1q_f32(reference);
        let rec = vld1q_f32(test);
        vst1q_f32(dst[0], rec);
        vst1q_f32(dst[1], vmulq_f32(rec, rec));
        vst1q_f32(dst[2], vmulq_f32(src, rec));
    }
}

/// # Safety
/// NEON must be available, all outputs must have the same length,
/// and each input must hold output.len() + 2 * RADIUS samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn blur_moment_horizontal_neon(
    src: [&[f32]; 3],
    k: &[f32; WINDOW],
    mut dst: [&mut [f32]; 3],
) -> usize {
    let n = dst[0].len();
    // Length contracts cover complete vectors, including the overlapping tail.
    unsafe {
        let done = moment_horizontal::<2>(src, k, dst.each_mut().map(|p| &mut p[..]));
        let done = if n - done >= 4 {
            done + moment_horizontal::<1>(
                src.map(|p| &p[done..]),
                k,
                dst.each_mut().map(|p| &mut p[done..]),
            )
        } else {
            done
        };
        if done < n && done >= 4 {
            let start = n - 4;
            moment_horizontal::<1>(
                src.map(|p| &p[start..]),
                k,
                dst.each_mut().map(|p| &mut p[start..]),
            );
            n
        } else {
            done
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn moment_horizontal<const N: usize>(
    src: [&[f32]; 3],
    k: &[f32; WINDOW],
    dst: [&mut [f32]; 3],
) -> usize {
    let [mean, square, cross] = dst.map(|p| p.as_chunks_mut::<4>().0.as_chunks_mut::<N>().0);
    let done = mean.len() * N * 4;
    let center = vdupq_n_f32(k[RADIUS]);
    for (block, ((mean, square), cross)) in mean.iter_mut().zip(square).zip(cross).enumerate() {
        let x = block * N * 4;
        // Every input includes the halo, and every output contains this block.
        unsafe {
            let mut acc: [[float32x4_t; N]; 3] = std::array::from_fn(|c| {
                std::array::from_fn(|lane| {
                    vmulq_f32(
                        vld1q_f32(src[c].as_ptr().add(x + lane * 4 + RADIUS)),
                        center,
                    )
                })
            });
            for (i, &weight) in k[..RADIUS].iter().enumerate() {
                let coeff = vdupq_n_f32(weight);
                for (c, plane) in acc.iter_mut().enumerate() {
                    for (lane, sum) in plane.iter_mut().enumerate() {
                        let offset = x + lane * 4;
                        let a = vld1q_f32(src[c].as_ptr().add(offset + i));
                        let b = vld1q_f32(src[c].as_ptr().add(offset + WINDOW - 1 - i));
                        *sum = vaddq_f32(*sum, vmulq_f32(vaddq_f32(a, b), coeff));
                    }
                }
            }
            for (dst, plane) in [mean, square, cross].into_iter().zip(acc) {
                for (dst, value) in dst.iter_mut().zip(plane) {
                    vst1q_f32(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}

/// # Safety
/// NEON must be available. Each row contains three planes of `width`
/// samples; all outputs must have equal lengths no greater than width.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn blur_moment_vertical_neon(
    rows: &MomentRows<'_>,
    width: usize,
    k: &[f32; WINDOW],
    mut dst: [&mut [f32]; 3],
) -> usize {
    let n = dst[0].len();
    // Offsets keep the same eleven row pointers for every plane and tail.
    unsafe {
        let done = moment_vertical::<2>(rows, width, k, dst.each_mut().map(|p| &mut p[..]), 0, n);
        let done = if n - done >= 4 {
            moment_vertical::<1>(rows, width, k, dst.each_mut().map(|p| &mut p[..]), done, n)
        } else {
            done
        };
        if done < n && done >= 4 {
            moment_vertical::<1>(rows, width, k, dst.each_mut().map(|p| &mut p[..]), n - 4, n);
            n
        } else {
            done
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn moment_vertical<const N: usize>(
    rows: &MomentRows<'_>,
    width: usize,
    k: &[f32; WINDOW],
    dst: [&mut [f32]; 3],
    start: usize,
    end: usize,
) -> usize {
    let [mean, square, cross] = dst.map(|p| {
        // The caller derives start..end from the shared output length.
        let span = unsafe { p.get_unchecked_mut(start..end) };
        span.as_chunks_mut::<4>().0.as_chunks_mut::<N>().0
    });
    let done = start + mean.len() * N * 4;
    let center = vdupq_n_f32(k[RADIUS]);
    for (block, ((mean, square), cross)) in mean.iter_mut().zip(square).zip(cross).enumerate() {
        let x = start + block * N * 4;
        // Rows contain all three planes; every load/store lies in this block.
        unsafe {
            let mut acc: [[float32x4_t; N]; 3] = std::array::from_fn(|c| {
                std::array::from_fn(|lane| {
                    vmulq_f32(
                        vld1q_f32(rows[RADIUS].as_ptr().add(c * width + x + lane * 4)),
                        center,
                    )
                })
            });
            for (i, &weight) in k[..RADIUS].iter().enumerate() {
                let coeff = vdupq_n_f32(weight);
                for (c, plane) in acc.iter_mut().enumerate() {
                    for (lane, sum) in plane.iter_mut().enumerate() {
                        let offset = c * width + x + lane * 4;
                        let a = vld1q_f32(rows[i].as_ptr().add(offset));
                        let b = vld1q_f32(rows[WINDOW - 1 - i].as_ptr().add(offset));
                        *sum = vaddq_f32(*sum, vmulq_f32(vaddq_f32(a, b), coeff));
                    }
                }
            }
            for (dst, plane) in [mean, square, cross].into_iter().zip(acc) {
                for (dst, value) in dst.iter_mut().zip(plane) {
                    vst1q_f32(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}

/// # Safety
/// NEON must be available. `rows[j]` points at the first output column of
/// source row `j` of the window (zeros for rows outside the plane), and each
/// is valid for `(planes - 1) · plane_stride + width` samples; plane `c`
/// starts `c · plane_stride` samples after it. `out[c]` must be valid for
/// `(rows.len() - 2·RADIUS - 1) · out_stride + width` samples, `width >= 4`.
///
/// Every output is the per-row kernels' expression (`center·k` plus the
/// symmetric pairs in tap order, multiplies and adds unfused), so results
/// match them bit for bit. Four output rows share fourteen row loads.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn blur_vertical_region_neon(
    rows: &[*const f32],
    plane_stride: usize,
    planes: usize,
    width: usize,
    k: &[f32; WINDOW],
    out: [*mut f32; 3],
    out_stride: usize,
) {
    debug_assert!(width >= 4 && rows.len() > 2 * RADIUS);
    let outputs = rows.len() - 2 * RADIUS;
    let coeff: [float32x4_t; RADIUS + 1] = std::array::from_fn(|i| vdupq_n_f32(k[i]));
    #[inline]
    #[target_feature(enable = "neon")]
    fn tap(window: &[float32x4_t], coeff: &[float32x4_t; RADIUS + 1]) -> float32x4_t {
        let mut acc = vmulq_f32(window[RADIUS], coeff[RADIUS]);
        for i in 0..RADIUS {
            acc = vaddq_f32(
                acc,
                vmulq_f32(vaddq_f32(window[i], window[WINDOW - 1 - i]), coeff[i]),
            );
        }
        acc
    }
    for (c, &dst) in out.iter().enumerate().take(planes) {
        let offset = c * plane_stride;
        // Complete vectors, then one overlapping vector for a short tail.
        let last = width - 4;
        let mut x = 0;
        loop {
            let column = x.min(last);
            let mut y = 0;
            unsafe {
                while y + 4 <= outputs {
                    let window: [float32x4_t; WINDOW + 3] =
                        std::array::from_fn(|j| vld1q_f32(rows[y + j].add(offset + column)));
                    for o in 0..4 {
                        let value = tap(&window[o..o + WINDOW], &coeff);
                        vst1q_f32(dst.add((y + o) * out_stride + column), value);
                    }
                    y += 4;
                }
                while y < outputs {
                    let window: [float32x4_t; WINDOW] =
                        std::array::from_fn(|j| vld1q_f32(rows[y + j].add(offset + column)));
                    vst1q_f32(dst.add(y * out_stride + column), tap(&window, &coeff));
                    y += 1;
                }
            }
            if column == last {
                break;
            }
            x += 4;
        }
    }
}
