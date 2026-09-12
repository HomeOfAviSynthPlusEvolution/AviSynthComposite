// SPDX-License-Identifier: GPL-2.0-or-later
// With the inherited AviSynth linking exception; see LICENSE.
#ifndef COMPOSITE_YUV_H
#define COMPOSITE_YUV_H
#include "plane.h"
#ifdef __cplusplus
extern "C" {
#endif
enum cp_yuv_operation {
  CP_YUV_ADD = 0,
  CP_YUV_SUBTRACT = 1,
  CP_YUV_SOFT_LIGHT = 2,
  CP_YUV_HARD_LIGHT = 3,
  CP_YUV_DIFFERENCE = 4,
  CP_YUV_EXCLUSION = 5,
  CP_YUV_MULTIPLY = 6
};
typedef struct cp_const_yuv {
  cp_const_plane y, u, v;
} cp_const_yuv;
typedef struct cp_yuv {
  cp_plane y, u, v;
} cp_yuv;
typedef struct cp_yuv_config {
  cp_format format;
  int operation;
  double opacity;
} cp_yuv_config;

// Full-resolution YUV (444). Independent strides/steps; YUV samples use code
// units, with neutral=2^(bits-1), or float Y=0..1 and UV neutral=0.
// Mask channels all use nonnegative opacity units, INCLUDING float U/V masks.
// Null mask means full opacity mask. All channels read before any are written;
// exact base/source in-place allowed. Alpha is deliberately independent: use
// cp_process_plane/cp_copy/cp_fill according to the caller's filter semantics.
// MULTIPLY shares source.y as guide for all channels: target = neutral +
// (base-neutral)*source.y/max, followed by continuous opacity*mask blending.
// Y neutral is zero; UV neutral is 2^(bits-1) for integer, zero for float.
// Source U/V descriptors are validated but their samples are unused in MULTIPLY.
// This is the fused equivalent of three CP_GUIDED_MULTIPLY/CONTINUOUS calls.
// Integer SIMD MULTIPLY with non-dyadic opacity (not k/256) may differ
// from scalar by at most 1 LSB for canonical codes. Zero/full opacity and
// F32 retain their existing behavior. Use cp_process_yuv for reference arithmetic.
// F32 supports Add/Subtract/Multiply. Other artistic modes reject float. Integer artistic
// modes (excluding MULTIPLY) desaturate UV on Y overshoot across 32*2^(bits-8) codes and saturates outputs.
// Float Add/Subtract clamps only overflowing Y in the operation's direction,
// proportionally desaturates UV over 32/255, and otherwise preserves excursions.
int cp_process_yuv(const cp_yuv_config* config, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* mask,
                   cp_yuv destination, cp_rows rows);
#ifdef __cplusplus
}
#endif
#endif
