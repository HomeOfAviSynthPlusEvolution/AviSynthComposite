// SPDX-License-Identifier: GPL-2.0-or-later
// With the inherited AviSynth linking exception; see LICENSE.
#ifndef COMPOSITE_UTILITIES_H
#define COMPOSITE_UTILITIES_H
#include "types.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct cp_const_rgb {
  cp_const_plane r, g, b;
} cp_const_rgb;
// Channel gather/scatter uses step, permitting packed<->planar without scratch.
int cp_copy(cp_format format, cp_const_plane source, cp_plane destination, cp_rows rows);
int cp_fill(cp_format format, cp_plane destination, cp_rows rows, double value);
// Integer saturates to code range. Float is unclipped. Finite scale/offset.
// Invert: scale=-1, offset=max for RGB/Y/A, 2*neutral for integer UV, 0 for float UV.
int cp_affine(cp_format format, cp_const_plane source, cp_plane destination, cp_rows rows, double scale, double offset);
// Explicit clamp for standalone Subtract / float range handling. Finite low<=high;
// NaN maps to low. Float MIX and artistic operations never implicitly invoke it.
int cp_clamp(cp_format format, cp_const_plane source, cp_plane destination, cp_rows rows, double low, double high);
// Layer luma uses floor; Mask uses nearest. Float ignores rounding (but validates
// it): 0.114B+0.587G+0.299R evaluated in binary32 in that order (no FMA).
// Integer coefficients are 9798R+19234G+3736B,
// divided by 32768. This is not a configurable color matrix.
enum cp_luma_rounding { CP_LUMA_FLOOR = 0, CP_LUMA_NEAREST = 1 };
int cp_rgb_luma(cp_format format, cp_const_rgb source, cp_plane destination, cp_rows rows, int rounding);
// Per-channel inclusive abs(sample-key)<=tolerance: alpha becomes zero if all
// match, otherwise it is copied exactly from alpha. Key/tolerances in sample units.
// Supports packed or planar RGBA via stepped views. Exact alpha in-place allowed.
int cp_color_key(cp_format format, cp_const_rgb source, cp_const_plane alpha, cp_plane destination, cp_rows rows,
                 const double key[3], const double tolerance[3]);
typedef struct cp_overlap {
  int base_x, base_y, source_x, source_y, width, height;
} cp_overlap;
// Intersect source translated by x,y with base; uses wide arithmetic for INT_MIN.
// Empty intersection returns all zeros. No frame/filter objects are involved.
int cp_intersect(int base_width, int base_height, int source_width, int source_height, int x, int y,
                 cp_overlap* output);
#ifdef __cplusplus
}
#endif
#endif
