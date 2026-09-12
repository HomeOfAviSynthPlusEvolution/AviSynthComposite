// SPDX-License-Identifier: GPL-2.0-or-later
// With the inherited AviSynth linking exception; see LICENSE.
#ifndef CP_COMMON_H
#define CP_COMMON_H
#include "composite/composite.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace cp {
// Caller supplies 4096 table words, 16 exceptional keys, and two total counts.
// Each plane retains its first eight keys (guide*256 + base). Level is 1..255.
void u8_multiply_corrections(unsigned level, uint32_t* corrections, uint32_t* keys, unsigned* counts);
int check_plane(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
                const cp_const_plane* ga, const cp_const_plane* gb, cp_plane d, cp_rows r);
inline int bytes(cp_format f) {
  if (f.storage == CP_U8 && f.bits == 8)
    return 1;
  if (f.storage == CP_U16 && f.bits >= 9 && f.bits <= 16)
    return 2;
  if (f.storage == CP_F32 && f.bits == 32)
    return 4;
  return 0;
}
inline double maximum(cp_format f) {
  return f.storage == CP_F32 ? 1.0 : (1u << f.bits) - 1;
}
inline bool rows_ok(cp_rows r) {
  return r.width > 0 && r.height > 0 && r.first >= 0 && r.count >= 0 && r.first <= r.height &&
         r.count <= r.height - r.first;
}
inline cp_const_plane view(cp_plane p) {
  return {p.data, p.stride, p.step};
}
inline bool plane_ok(cp_const_plane p, cp_rows r, int b) {
  if (!p.data || p.step < b || p.step % b || p.stride % b || reinterpret_cast<uintptr_t>(p.data) % b)
    return false;
  const auto limit = std::numeric_limits<ptrdiff_t>::max();
  if (p.step > (limit - b) / r.width)
    return false;
  const ptrdiff_t span = (r.width - 1) * p.step + b;
  if (p.stride == std::numeric_limits<ptrdiff_t>::min())
    return false;
  const ptrdiff_t stride = p.stride < 0 ? -p.stride : p.stride;
  if (stride < span || (r.height > 1 && stride > (limit - span) / (r.height - 1)))
    return false;
  return true;
}
inline bool plane_ok(cp_plane p, cp_rows r, int b) {
  return plane_ok(view(p), r, b);
}
inline const unsigned char* address(cp_const_plane p, int x, int y) {
  return static_cast<const unsigned char*>(p.data) + p.stride * y + p.step * x;
}
inline unsigned char* address(cp_plane p, int x, int y) {
  return static_cast<unsigned char*>(p.data) + p.stride * y + p.step * x;
}
inline double get(cp_const_plane p, cp_format f, int x, int y) {
  const auto* a = address(p, x, y);
  if (f.storage == CP_U8)
    return *a;
  if (f.storage == CP_U16) {
    uint16_t v;
    std::memcpy(&v, a, 2);
    return v;
  }
  float v;
  std::memcpy(&v, a, 4);
  return v;
}
inline void put(cp_plane p, cp_format f, int x, int y, double v) {
  auto* a = address(p, x, y);
  if (f.storage == CP_F32) {
    const float out = static_cast<float>(v);
    std::memcpy(a, &out, 4);
    return;
  }
  const double m = maximum(f);
  v = std::isnan(v) ? 0 : std::clamp(v, 0.0, m);
  const uint16_t out = static_cast<uint16_t>(std::floor(v + 0.5));
  if (f.storage == CP_U8)
    *a = static_cast<unsigned char>(out);
  else
    std::memcpy(a, &out, 2);
}
inline void copy_sample(cp_const_plane s, cp_plane d, int x, int y, int b) {
  // Local staging also supports exact in-place without memcpy overlap questions.
  unsigned char v[4];
  std::memcpy(v, address(s, x, y), b);
  std::memcpy(address(d, x, y), v, b);
}
inline bool opacity_ok(double w) {
  return std::isfinite(w) && w >= 0 && w <= 1;
}
inline double effective(cp_format f, double opacity, const cp_const_plane* mask, int x, int y, int rule) {
  const double max = maximum(f), m = mask ? get(*mask, f, x, y) : max;
  if (rule == CP_WEIGHT_CODE && f.storage != CP_F32)
    return std::floor(m * std::floor(opacity * max + 0.5) / max + 0.5) / max;
  return opacity * (m / max);
}
inline double mix(double a, double b, double w) {
  return w == 0 ? a : w == 1 ? b : a + (b - a) * w;
}
} // namespace cp
#endif
