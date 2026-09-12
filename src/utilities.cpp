// SPDX-License-Identifier: GPL-2.0-or-later
// AviSynth/AviSynth+ luma/masking semantics, independently reimplemented.
// AviSynth (c) 2002 Ben Rudiak-Gould et al.; inherited exception in LICENSE.
#include "common.h"

int cp_copy(cp_format f, cp_const_plane s, cp_plane d, cp_rows r) {
  const int b = cp::bytes(f);
  if (!b || !cp::rows_ok(r))
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!cp::plane_ok(s, r, b) || !cp::plane_ok(d, r, b))
    return CP_INVALID_ARGUMENT;
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x)
      cp::copy_sample(s, d, x, y, b);
  return CP_OK;
}
int cp_fill(cp_format f, cp_plane d, cp_rows r, double value) {
  const int b = cp::bytes(f);
  if (!b || !cp::rows_ok(r) || !std::isfinite(value))
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!cp::plane_ok(d, r, b))
    return CP_INVALID_ARGUMENT;
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x)
      cp::put(d, f, x, y, value);
  return CP_OK;
}
int cp_affine(cp_format f, cp_const_plane s, cp_plane d, cp_rows r, double scale, double offset) {
  const int b = cp::bytes(f);
  if (!b || !cp::rows_ok(r) || !std::isfinite(scale) || !std::isfinite(offset))
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!cp::plane_ok(s, r, b) || !cp::plane_ok(d, r, b))
    return CP_INVALID_ARGUMENT;
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x)
      cp::put(d, f, x, y, cp::get(s, f, x, y) * scale + offset);
  return CP_OK;
}
int cp_clamp(cp_format f, cp_const_plane s, cp_plane d, cp_rows r, double low, double high) {
  const int b = cp::bytes(f);
  if (!b || !cp::rows_ok(r) || !std::isfinite(low) || !std::isfinite(high) || low > high)
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!cp::plane_ok(s, r, b) || !cp::plane_ok(d, r, b))
    return CP_INVALID_ARGUMENT;
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x) {
      double v = cp::get(s, f, x, y);
      cp::put(d, f, x, y, std::isnan(v) ? low : std::clamp(v, low, high));
    }
  return CP_OK;
}
int cp_rgb_luma(cp_format f, cp_const_rgb rgb, cp_plane d, cp_rows r, int rounding) {
  const int b = cp::bytes(f);
  if (!b || !cp::rows_ok(r) || (rounding != CP_LUMA_FLOOR && rounding != CP_LUMA_NEAREST))
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!cp::plane_ok(rgb.r, r, b) || !cp::plane_ok(rgb.g, r, b) || !cp::plane_ok(rgb.b, r, b) || !cp::plane_ok(d, r, b))
    return CP_INVALID_ARGUMENT;
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x) {
      double rr = cp::get(rgb.r, f, x, y), gg = cp::get(rgb.g, f, x, y), bb = cp::get(rgb.b, f, x, y);
      // Float luma is also a selection guide: preserve upstream B+G+R binary32
      // arithmetic, not a double sum that could flip a threshold decision.
      double v =
          f.storage == CP_F32
              ? static_cast<float>(0.114f * static_cast<float>(bb) + 0.587f * static_cast<float>(gg) +
                                   0.299f * static_cast<float>(rr))
              : std::floor((9798 * rr + 19234 * gg + 3736 * bb + (rounding == CP_LUMA_NEAREST ? 16384 : 0)) / 32768);
      cp::put(d, f, x, y, v);
    }
  return CP_OK;
}
int cp_color_key(cp_format f, cp_const_rgb rgb, cp_const_plane alpha, cp_plane d, cp_rows r, const double key[3],
                 const double tolerance[3]) {
  const int b = cp::bytes(f);
  if (!b || !cp::rows_ok(r) || !key || !tolerance)
    return CP_INVALID_ARGUMENT;
  for (int p = 0; p < 3; ++p)
    if (!std::isfinite(key[p]) || !std::isfinite(tolerance[p]) || tolerance[p] < 0)
      return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!cp::plane_ok(rgb.r, r, b) || !cp::plane_ok(rgb.g, r, b) || !cp::plane_ok(rgb.b, r, b) ||
      !cp::plane_ok(alpha, r, b) || !cp::plane_ok(d, r, b))
    return CP_INVALID_ARGUMENT;
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x) {
      const bool hit = std::abs(cp::get(rgb.r, f, x, y) - key[0]) <= tolerance[0] &&
                       std::abs(cp::get(rgb.g, f, x, y) - key[1]) <= tolerance[1] &&
                       std::abs(cp::get(rgb.b, f, x, y) - key[2]) <= tolerance[2];
      if (hit)
        cp::put(d, f, x, y, 0);
      else
        cp::copy_sample(alpha, d, x, y, b);
    }
  return CP_OK;
}
int cp_intersect(int bw, int bh, int sw, int sh, int x, int y, cp_overlap* out) {
  if (!out)
    return CP_INVALID_ARGUMENT;
  *out = {};
  if (bw <= 0 || bh <= 0 || sw <= 0 || sh <= 0)
    return CP_INVALID_ARGUMENT;
  const int64_t left = std::max<int64_t>(0, x), top = std::max<int64_t>(0, y);
  const int64_t right = std::min<int64_t>(bw, static_cast<int64_t>(x) + sw),
                bottom = std::min<int64_t>(bh, static_cast<int64_t>(y) + sh);
  if (right <= left || bottom <= top)
    return CP_OK;
  *out = {static_cast<int>(left),    static_cast<int>(top),          static_cast<int>(left - x),
          static_cast<int>(top - y), static_cast<int>(right - left), static_cast<int>(bottom - top)};
  return CP_OK;
}
