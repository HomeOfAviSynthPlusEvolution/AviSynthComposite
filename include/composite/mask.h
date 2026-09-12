// SPDX-License-Identifier: GPL-2.0-or-later
// With the inherited AviSynth linking exception; see LICENSE.
#ifndef COMPOSITE_MASK_H
#define COMPOSITE_MASK_H
#include "types.h"
#ifdef __cplusplus
extern "C" {
#endif
enum cp_placement { CP_CENTER = 0, CP_MPEG2 = 1, CP_TOP_LEFT = 2 };
typedef struct cp_sampling {
  int source_width, source_height;
  int subsample_x, subsample_y; // x=1,2,4; y=1,2. 4x is 411 (y=1).
  int placement;
  int origin_x, origin_y; // signed source phase of output (0,0), supports clipped overlaps.
} cp_sampling;
// Source positions outside the described image replicate its nearest edge.
// 411 always box4; CENTER box2/box2x2; MPEG2 horizontal [1,2,1]/4,
// vertical box2; TOP_LEFT point. 444 is identity. Integers round once after
// spatial filtering, BEFORE any caller's opacity combination or comparison.
// Float follows upstream binary32 tap addition order to preserve guide decisions.
// Source and destination must be disjoint except exact identity sampling.
int cp_resample_mask(cp_format format, cp_const_plane source, cp_plane destination, const cp_sampling* sampling,
                     cp_rows output_rows);
#ifdef __cplusplus
}
#endif
#endif
