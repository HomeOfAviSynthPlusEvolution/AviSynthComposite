// SPDX-License-Identifier: GPL-2.0-or-later
// AviSynth (c) 2002 Ben Rudiak-Gould et al.; Overlay (c) 2003, 2004 Klaus Post.
// Reimplemented arithmetic contracts from AviSynth/AviSynth+; see LICENSE for
// the inherited linking exception. No AviSynth implementation is compiled here.
#include "common.h"

int cp_sample_bytes(cp_format f) {
  return cp::bytes(f);
}

int cp::check_plane(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
                    const cp_const_plane* ga, const cp_const_plane* gb, cp_plane d, cp_rows r) {
  if (!c || !cp::bytes(c->format) || !cp::rows_ok(r) || !cp::opacity_ok(c->opacity) || c->operation < CP_MIX ||
      c->operation > CP_DIFFERENCE || c->weight_rule < CP_WEIGHT_CONTINUOUS || c->weight_rule > CP_WEIGHT_CODE ||
      (c->inclusive != 0 && c->inclusive != 1) || !std::isfinite(c->threshold) || c->threshold < 0 ||
      !std::isfinite(c->neutral) || !std::isfinite(c->inversion_sum) || !std::isfinite(c->bias))
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  const int bytes = cp::bytes(c->format);
  const bool select = c->operation == CP_SELECT_LIGHTER || c->operation == CP_SELECT_DARKER;
  if (!cp::plane_ok(a, r, bytes) || !cp::plane_ok(b, r, bytes) || !cp::plane_ok(d, r, bytes) ||
      (mask && !cp::plane_ok(*mask, r, bytes)) ||
      ((select || c->operation == CP_GUIDED_MULTIPLY) && (!gb || !cp::plane_ok(*gb, r, bytes))) ||
      (select && (!ga || !cp::plane_ok(*ga, r, bytes))))
    return CP_INVALID_ARGUMENT;
  return CP_OK;
}

int cp_process_plane(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
                     const cp_const_plane* ga, const cp_const_plane* gb, cp_plane d, cp_rows r) {
  const int status = cp::check_plane(c, a, b, mask, ga, gb, d, r);
  if (status != CP_OK || !r.count)
    return status;
  const int bytes = cp::bytes(c->format);
  const bool select = c->operation == CP_SELECT_LIGHTER || c->operation == CP_SELECT_DARKER;
  const cp_format f = c->format;
  const double max = cp::maximum(f);
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x) {
      double w = cp::effective(f, c->opacity, mask, x, y, c->weight_rule);
      if (select) {
        const bool lighter = c->operation == CP_SELECT_LIGHTER;
        const double base_value = cp::get(*ga, f, x, y), source_value = cp::get(*gb, f, x, y);
        // Preserve binary32 threshold addition before comparison. A higher precision
        // reassociation can select entirely different UV, not merely change an LSB.
        double boundary = base_value + (lighter ? c->threshold : -c->threshold);
        if (f.storage == CP_F32) {
          const float t = static_cast<float>(c->threshold);
          boundary = static_cast<float>(base_value) + (lighter ? t : -t);
        }
        const bool accepted = lighter ? (c->inclusive ? source_value >= boundary : source_value > boundary)
                                      : (c->inclusive ? source_value <= boundary : source_value < boundary);
        if (!accepted)
          w = 0;
      }
      if (w == 0) {
        cp::copy_sample(a, d, x, y, bytes);
        continue;
      }
      if (w == 1 && (c->operation == CP_MIX || select)) {
        cp::copy_sample(b, d, x, y, bytes);
        continue;
      }
      const double av = cp::get(a, f, x, y), bv = cp::get(b, f, x, y);
      double target = bv;
      switch (c->operation) {
        case CP_ADD:
          target = av + bv;
          break;
        case CP_SUBTRACT:
          target = av - bv;
          break;
        case CP_PRODUCT:
          target = av * bv / max;
          if (f.storage != CP_F32)
            target = std::floor(target);
          break;
        case CP_INVERT_MIX:
          target = c->inversion_sum - bv;
          break;
        case CP_GUIDED_MULTIPLY:
          if (c->weight_rule == CP_WEIGHT_CODE && f.storage != CP_F32) {
            const double darken = std::floor(w * (max - cp::get(*gb, f, x, y)) + 0.5) / max;
            cp::put(d, f, x, y, cp::mix(av, c->neutral, darken));
            continue;
          }
          target = c->neutral + (av - c->neutral) * cp::get(*gb, f, x, y) / max;
          break;
        case CP_DIFFERENCE:
          target = av - bv + c->bias;
          break;
        default:
          break;
      }
      cp::put(d, f, x, y, cp::mix(av, target, w));
    }
  return CP_OK;
}

int cp_blend_compat(cp_format f, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask, cp_plane d, cp_rows r,
                    int opacity) {
  if (!cp::bytes(f) || !cp::rows_ok(r) || opacity < 0 || opacity > 256)
    return CP_INVALID_ARGUMENT;
  if (f.storage == CP_F32)
    return CP_UNSUPPORTED;
  if (!r.count)
    return CP_OK;
  const int bytes = cp::bytes(f);
  if (!cp::plane_ok(a, r, bytes) || !cp::plane_ok(b, r, bytes) || !cp::plane_ok(d, r, bytes) ||
      (mask && !cp::plane_ok(*mask, r, bytes)))
    return CP_INVALID_ARGUMENT;
  const int64_t scale = INT64_C(1) << f.bits;
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x) {
      const int64_t av = static_cast<int64_t>(cp::get(a, f, x, y)), bv = static_cast<int64_t>(cp::get(b, f, x, y));
      const int64_t m = mask ? static_cast<int64_t>(cp::get(*mask, f, x, y)) : 0;
      int64_t out;
      if (!mask)
        out = (av * (256 - opacity) + bv * opacity + 128) / 256;
      else if (opacity == 256 && m == scale - 1)
        out = bv;
      else {
        const int64_t weight = opacity == 256 ? m : m * opacity / 256;
        out = (av * (scale - weight) + bv * weight + scale / 2) / scale;
      }
      cp::put(d, f, x, y, static_cast<double>(out));
    }
  return CP_OK;
}
