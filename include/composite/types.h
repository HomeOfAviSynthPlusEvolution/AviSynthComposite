// SPDX-License-Identifier: GPL-2.0-or-later
// With the inherited AviSynth linking exception; see LICENSE.
#ifndef COMPOSITE_TYPES_H
#define COMPOSITE_TYPES_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum cp_status { CP_OK = 0, CP_INVALID_ARGUMENT = 1, CP_UNSUPPORTED = 2 };
enum cp_storage { CP_U8 = 1, CP_U16 = 2, CP_F32 = 3 };
typedef struct cp_format {
  int storage, bits;
} cp_format;

// Native-endian, naturally aligned storage: U8/8, U16/9..16, F32/32.
// Integer samples must fit the declared depth. F32 color may be out of range;
// masks must be finite and in [0,1]. Integer masks use [0,2^bits-1].
// Sample content and allocation lengths are the caller's responsibility.
// step is positive BYTES between adjacent samples: sizeof(T) for planar,
// e.g. 4*sizeof(T) for one channel of packed RGBA. stride is signed BYTES.
// Data always points to logical row zero. No padding or other channels touched.
typedef struct cp_const_plane {
  const void* data;
  ptrdiff_t stride, step;
} cp_const_plane;
typedef struct cp_plane {
  void* data;
  ptrdiff_t stride, step;
} cp_plane;
typedef struct cp_rows {
  int width, height, first, count;
} cp_rows;

// All functions validate geometry/descriptors before writing. Positive width and
// height required; [first,first+count) must be within height. Empty bands succeed
// without buffer access or descriptor validation. stride magnitude covers a row.
// All storage must cover the full described image. No allocation, no exceptions,
// no mutable global state. Concurrent calls require disjoint written samples and
// no read/write hazards (including guides/masks). Headers and library must match.
// Output may exactly alias a matching input descriptor where documented; partial
// overlap is never supported. Disjoint interleaved channels are permitted.
// Overlap/allocation size cannot in general be validated by this API.
int cp_sample_bytes(cp_format format);

#ifdef __cplusplus
}
#endif
#endif
