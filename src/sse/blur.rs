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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub(crate) fn blur_horizontal_sse41(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
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
#[target_feature(enable = "sse4.1")]
fn horizontal<const N: usize>(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let center = _mm_set1_ps(k[RADIUS]);
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    let done = blocks.len() * N * LANES;
    for (block, dst) in blocks.iter_mut().enumerate() {
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [__m128; N] = std::array::from_fn(|lane| {
                _mm_mul_ps(
                    _mm_loadu_ps(src.as_ptr().add(x + lane * LANES + RADIUS)),
                    center,
                )
            });
            for (i, &weight) in k[..RADIUS].iter().enumerate() {
                let coeff = _mm_set1_ps(weight);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = _mm_loadu_ps(src.as_ptr().add(x + lane * LANES + i));
                    let b = _mm_loadu_ps(src.as_ptr().add(x + lane * LANES + WINDOW - 1 - i));
                    *acc = _mm_add_ps(*acc, _mm_mul_ps(_mm_add_ps(a, b), coeff));
                }
            }
            for (dst, acc) in dst.iter_mut().zip(acc) {
                _mm_storeu_ps(dst.as_mut_ptr(), acc);
            }
        }
    }
    done
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn blur_vertical_sse41(
    rows: &BlurRows<'_>,
    k: &[f32; WINDOW],
    dst: &mut [f32],
) -> usize {
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
#[target_feature(enable = "sse4.1")]
fn vertical<const N: usize>(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let center = _mm_set1_ps(k[RADIUS]);
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    let done = blocks.len() * N * LANES;
    for (block, dst) in blocks.iter_mut().enumerate() {
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [__m128; N] = std::array::from_fn(|lane| {
                _mm_mul_ps(
                    _mm_loadu_ps(rows[RADIUS].as_ptr().add(x + lane * LANES)),
                    center,
                )
            });
            for (i, &weight) in k[..RADIUS].iter().enumerate() {
                let coeff = _mm_set1_ps(weight);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = _mm_loadu_ps(rows[i].as_ptr().add(x + lane * LANES));
                    let b = _mm_loadu_ps(rows[WINDOW - 1 - i].as_ptr().add(x + lane * LANES));
                    *acc = _mm_add_ps(*acc, _mm_mul_ps(_mm_add_ps(a, b), coeff));
                }
            }
            for (dst, acc) in dst.iter_mut().zip(acc) {
                _mm_storeu_ps(dst.as_mut_ptr(), acc);
            }
        }
    }
    done
}

/// # Safety
/// SSE4.1 must be available and all three slices must have equal lengths.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn blur_product_row_sse41([a, b]: [&[f32]; 2], dst: &mut [f32]) -> usize {
    let (vectors, _) = dst.as_chunks_mut::<4>();
    let done = vectors.len() * 4;
    for ((dst, a), b) in vectors
        .iter_mut()
        .zip(a.as_chunks::<4>().0)
        .zip(b.as_chunks::<4>().0)
    {
        // Equal lengths ensure that every complete vector is in bounds.
        unsafe {
            _mm_storeu_ps(
                dst.as_mut_ptr(),
                _mm_mul_ps(_mm_loadu_ps(a.as_ptr()), _mm_loadu_ps(b.as_ptr())),
            );
        }
    }
    done
}

/// # Safety
/// SSE4.1 must be available, inputs must have equal lengths, and every
/// output must match that length. Products round before filtering.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn blur_moment_row_sse41(
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
#[target_feature(enable = "sse4.1")]
unsafe fn moment_row_vector(reference: *const f32, test: *const f32, dst: [*mut f32; 3]) {
    // The caller provides one full vector behind every pointer.
    unsafe {
        let src = _mm_loadu_ps(reference);
        let rec = _mm_loadu_ps(test);
        _mm_storeu_ps(dst[0], rec);
        _mm_storeu_ps(dst[1], _mm_mul_ps(rec, rec));
        _mm_storeu_ps(dst[2], _mm_mul_ps(src, rec));
    }
}

/// # Safety
/// SSE4.1 must be available, all outputs must have the same length,
/// and each input must hold output.len() + 2 * RADIUS samples.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn blur_moment_horizontal_sse41(
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
#[target_feature(enable = "sse4.1")]
unsafe fn moment_horizontal<const N: usize>(
    src: [&[f32]; 3],
    k: &[f32; WINDOW],
    dst: [&mut [f32]; 3],
) -> usize {
    let [mean, square, cross] = dst.map(|p| p.as_chunks_mut::<4>().0.as_chunks_mut::<N>().0);
    let done = mean.len() * N * 4;
    let center = _mm_set1_ps(k[RADIUS]);
    for (block, ((mean, square), cross)) in mean.iter_mut().zip(square).zip(cross).enumerate() {
        let x = block * N * 4;
        // Every input includes the halo, and every output contains this block.
        unsafe {
            let mut acc: [[__m128; N]; 3] = std::array::from_fn(|c| {
                std::array::from_fn(|lane| {
                    _mm_mul_ps(
                        _mm_loadu_ps(src[c].as_ptr().add(x + lane * 4 + RADIUS)),
                        center,
                    )
                })
            });
            for (i, &weight) in k[..RADIUS].iter().enumerate() {
                let coeff = _mm_set1_ps(weight);
                for (c, plane) in acc.iter_mut().enumerate() {
                    for (lane, sum) in plane.iter_mut().enumerate() {
                        let offset = x + lane * 4;
                        let a = _mm_loadu_ps(src[c].as_ptr().add(offset + i));
                        let b = _mm_loadu_ps(src[c].as_ptr().add(offset + WINDOW - 1 - i));
                        *sum = _mm_add_ps(*sum, _mm_mul_ps(_mm_add_ps(a, b), coeff));
                    }
                }
            }
            for (dst, plane) in [mean, square, cross].into_iter().zip(acc) {
                for (dst, value) in dst.iter_mut().zip(plane) {
                    _mm_storeu_ps(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}

/// # Safety
/// SSE4.1 must be available. Each row contains three planes of `width`
/// samples; all outputs must have equal lengths no greater than width.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn blur_moment_vertical_sse41(
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
#[target_feature(enable = "sse4.1")]
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
    let center = _mm_set1_ps(k[RADIUS]);
    for (block, ((mean, square), cross)) in mean.iter_mut().zip(square).zip(cross).enumerate() {
        let x = start + block * N * 4;
        // Rows contain all three planes; every load/store lies in this block.
        unsafe {
            let mut acc: [[__m128; N]; 3] = std::array::from_fn(|c| {
                std::array::from_fn(|lane| {
                    _mm_mul_ps(
                        _mm_loadu_ps(rows[RADIUS].as_ptr().add(c * width + x + lane * 4)),
                        center,
                    )
                })
            });
            for (i, &weight) in k[..RADIUS].iter().enumerate() {
                let coeff = _mm_set1_ps(weight);
                for (c, plane) in acc.iter_mut().enumerate() {
                    for (lane, sum) in plane.iter_mut().enumerate() {
                        let offset = c * width + x + lane * 4;
                        let a = _mm_loadu_ps(rows[i].as_ptr().add(offset));
                        let b = _mm_loadu_ps(rows[WINDOW - 1 - i].as_ptr().add(offset));
                        *sum = _mm_add_ps(*sum, _mm_mul_ps(_mm_add_ps(a, b), coeff));
                    }
                }
            }
            for (dst, plane) in [mean, square, cross].into_iter().zip(acc) {
                for (dst, value) in dst.iter_mut().zip(plane) {
                    _mm_storeu_ps(dst.as_mut_ptr(), value);
                }
            }
        }
    }
    done
}
