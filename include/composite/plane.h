// SPDX-License-Identifier: GPL-2.0-or-later
// With the inherited AviSynth linking exception; see LICENSE.
#ifndef COMPOSITE_PLANE_H
#define COMPOSITE_PLANE_H
#include "types.h"
#ifdef __cplusplus
extern "C" {
#endif
enum cp_operation {
  CP_MIX = 0,             // interpolate toward source (Layer Add).
  CP_ADD = 1,             // base + weight*source (Overlay RGB Add).
  CP_SUBTRACT = 2,        // base - weight*source (Overlay RGB Subtract).
  CP_PRODUCT = 3,         // interpolate toward base*source/maximum.
  CP_INVERT_MIX = 4,      // interpolate toward inversion_sum-source.
  CP_GUIDED_MULTIPLY = 5, // neutral+(base-neutral)*(1+w*(source_guide/max-1)).
  CP_SELECT_LIGHTER = 6,  // interpolate only if source_guide > base_guide+threshold.
  CP_SELECT_DARKER = 7,   // interpolate only if source_guide < base_guide-threshold.
  CP_DIFFERENCE = 8       // interpolate toward base-source+bias (explicit direction).
};
enum cp_weight_rule {
  CP_WEIGHT_CONTINUOUS = 0, // Reference w=opacity*mask/max; SIMD tolerance documented below.
  CP_WEIGHT_CODE = 1        // opacity and combined mask rounded to integer code scale.
};
typedef struct cp_plane_config {
  cp_format format;
  int operation;
  double opacity;       // finite [0,1]; not gamma or color-space conversion.
  double neutral;       // guided multiply center, in sample units.
  double inversion_sum; // max for RGB/Y/A; 2*neutral for integer chroma; 0 for float chroma.
  double bias;          // difference equality code, in sample units.
  double threshold;     // nonnegative, guide units; float threshold sum rounds to F32 before comparison.
  int inclusive;        // 0 strict Layer comparison; 1 inclusive Overlay comparison.
  int weight_rule;
} cp_plane_config;

// Optional mask has the same format and geometry. Null means full mask.
// Guided modes require source_guide; selection additionally requires base_guide.
// Unused guides are ignored. Destination can exactly alias base/source; guides
// and masks may alias them only if every call reads each weight before writing
// that same sample and there are no cross-call hazards. Use separate immutable
// guides when applying a luma decision to multiple color channels.
// Integers saturate and round nearest (ties toward +infinity); PRODUCT first
// floors the product, matching Layer's target construction. Float is unclipped.
// Opacity/mask zero returns base exactly; MIX and selected branches at weight 1
// copy source exactly (including float signed zero/NaN payload).
// SIMD dispatch may quantize unmasked integer CP_MIX continuous opacity to
// round(opacity*32768)/32768. Output differs from this scalar reference by at
// most 1 LSB per call, for U8 and U16 (9..16 bits). Exact opacity 0/1 copies
// remain exact. Unmasked integer continuous CP_INVERT_MIX may similarly use
// Q16 for integral inversion_sum in [0,65535], with at most 1 LSB error.
// Masked integer continuous MIX and INVERT_MIX (inversion_sum == maximum)
// may round the combined effective weight to Q16, also within 1 LSB.
// Integer continuous GUIDED_MULTIPLY also permits 1 LSB for interior opacity
// and neutral in [0, maximum].
// Integer continuous DIFFERENCE with integral bias in [0,65535] also permits 1 LSB.
// Integer continuous PRODUCT, ADD and SUBTRACT may quantize weights to Q16
// within 1 LSB, preserving product floor before blending.
// Mask zero and exact full-weight endpoints remain exact.
// Integer input samples and masks must fit the declared depth (see types.h).
// Continuous F32 masked MIX/PRODUCT may use binary32 for all opacities and finite
// colors (including negative/HDR). Error <= 16*FLT_EPSILON*max(1,S), where
// w=opacity*mask; MIX S=abs(a)*(1-w)+abs(b)*w;
// PRODUCT S=abs(a)*((1-w)+abs(b)*w). Zero mask and full-weight endpoints are exact.
// Nonfinite/near-overflow or severely ill-conditioned near-full blends use the
// reference calculation. Both F32 weight rules are covered; integer CODE is unchanged.
// See README for fallback details; cp_process_plane retains reference arithmetic.
// The integer 1 LSB tolerance does not apply to integer CODE or other integer operations.
// Repeated calls can accumulate error; use CP_TARGET_C / this function
// for reference arithmetic, or raise the working bit depth before processing
// to reduce the normalized size of 1 LSB.
int cp_process_plane(const cp_plane_config* config, cp_const_plane base, cp_const_plane source,
                     const cp_const_plane* mask, const cp_const_plane* base_guide, const cp_const_plane* source_guide,
                     cp_plane destination, cp_rows rows);

// Historical Minus blend_compat scalar arithmetic, opacity 0..256. Integer only.
// This is distinct from CP_WEIGHT_CODE. Exact historical endpoint/rounding
// contract is specified in tests/scalar_tests.cpp; mask null uses 8-bit opacity.
int cp_blend_compat(cp_format format, cp_const_plane base, cp_const_plane source, const cp_const_plane* mask,
                    cp_plane destination, cp_rows rows, int opacity);
#ifdef __cplusplus
}
#endif
#endif
