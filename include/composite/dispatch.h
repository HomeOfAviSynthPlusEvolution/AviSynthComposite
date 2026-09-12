// SPDX-License-Identifier: GPL-2.0-or-later
// With the inherited AviSynth linking exception; see LICENSE.
#ifndef COMPOSITE_DISPATCH_H
#define COMPOSITE_DISPATCH_H
#include "plane.h"
#include "utilities.h"
#include "yuv.h"
#include "mask.h"
#ifdef __cplusplus
extern "C" {
#endif
#define CP_TARGET_C INT64_C(0)
#define CP_TARGET_NATIVE INT64_C(-1)
// Highway target bits, excluding SCALAR/EMU128. C is always available, even when
// these masks are zero. choose_target intersects caller policy with CPU support;
// it never changes Highway process-global target selection.
int64_t cp_compiled_targets(void);
int64_t cp_supported_targets(void);
int64_t cp_choose_target(int64_t allowed_targets);
// Immutable, process-lifetime tables; no creation/destruction, allocation, or
// per-call CPU detection. Cache per instance after applying the host CPU policy.
// Unsupported single target or a multi-bit mask returns null. Native selects
// the best supported compiled target. Scalar-only builds return the C table.
// Pixel operations are covered by the table. Targets without FP64 fall back to
// C for double-precision arithmetic; integer code-weight MIX/PRODUCT, copy/fill,
// sampling, luma and integer compat
// still use SIMD. Partial vectors and stepped channels use bounded staging when
// needed. This preserves access boundaries; it does not promise a speedup for
// every layout. Rectangle intersection is scalar O(1) and stays directly callable.
typedef struct cp_kernels {
  int (*process_plane)(const cp_plane_config*, cp_const_plane, cp_const_plane, const cp_const_plane*,
                       const cp_const_plane*, const cp_const_plane*, cp_plane, cp_rows);
  int (*blend_compat)(cp_format, cp_const_plane, cp_const_plane, const cp_const_plane*, cp_plane, cp_rows, int);
  int (*copy)(cp_format, cp_const_plane, cp_plane, cp_rows);
  int (*fill)(cp_format, cp_plane, cp_rows, double);
  int (*process_yuv)(const cp_yuv_config*, cp_const_yuv, cp_const_yuv, const cp_const_yuv*, cp_yuv, cp_rows);
  int (*resample_mask)(cp_format, cp_const_plane, cp_plane, const cp_sampling*, cp_rows);
  int (*affine)(cp_format, cp_const_plane, cp_plane, cp_rows, double, double);
  int (*clamp)(cp_format, cp_const_plane, cp_plane, cp_rows, double, double);
  int (*rgb_luma)(cp_format, cp_const_rgb, cp_plane, cp_rows, int);
  int (*color_key)(cp_format, cp_const_rgb, cp_const_plane, cp_plane, cp_rows, const double*, const double*);
} cp_kernels;
const cp_kernels* cp_get_kernels(int64_t target);
#ifdef __cplusplus
}
#endif
#endif
