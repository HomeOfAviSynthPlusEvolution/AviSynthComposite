// SPDX-License-Identifier: GPL-2.0-or-later
// AviSynth/AviSynth+ chroma-mask placement semantics, reimplemented with explicit
// bounds and phase. See LICENSE for inherited copyright/linking exception.
#include "common.h"

int cp_resample_mask(cp_format f, cp_const_plane source, cp_plane destination, const cp_sampling* s, cp_rows r) {
  if (!s || !cp::bytes(f) || !cp::rows_ok(r) || s->source_width <= 0 || s->source_height <= 0 ||
      (s->subsample_x != 1 && s->subsample_x != 2 && s->subsample_x != 4) ||
      (s->subsample_y != 1 && s->subsample_y != 2) || (s->subsample_y == 2 && s->subsample_x != 2) ||
      s->placement < CP_CENTER || s->placement > CP_TOP_LEFT)
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  const cp_rows input = {s->source_width, s->source_height, 0, s->source_height};
  if (!cp::plane_ok(source, input, cp::bytes(f)) || !cp::plane_ok(destination, r, cp::bytes(f)))
    return CP_INVALID_ARGUMENT;
  auto sample = [&](int64_t x, int64_t y) {
    return cp::get(source, f, static_cast<int>(std::clamp<int64_t>(x, 0, s->source_width - 1)),
                   static_cast<int>(std::clamp<int64_t>(y, 0, s->source_height - 1)));
  };
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x) {
      const int64_t sx = static_cast<int64_t>(x) * s->subsample_x + s->origin_x;
      const int64_t sy = static_cast<int64_t>(y) * s->subsample_y + s->origin_y;
      if (f.storage == CP_F32) {
        auto at = [&](int dx, int dy) {
          return static_cast<float>(sample(sx + dx, sy + dy));
        };
        float value;
        if (s->subsample_x == 1 || (s->placement == CP_TOP_LEFT && s->subsample_x != 4))
          value = at(0, 0);
        else if (s->subsample_x == 4)
          value = (at(0, 0) + at(1, 0) + at(2, 0) + at(3, 0)) * 0.25f;
        else if (s->placement == CP_CENTER) {
          value =
              s->subsample_y == 2 ? (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25f : (at(0, 0) + at(1, 0)) * 0.5f;
        } else if (s->subsample_y == 2) {
          const float left = at(-1, 0) + at(-1, 1), mid = at(0, 0) + at(0, 1), right = at(1, 0) + at(1, 1);
          value = (left + 2.0f * mid + right) * 0.125f;
        } else
          value = (at(-1, 0) + 2.0f * at(0, 0) + at(1, 0)) * 0.25f;
        cp::put(destination, f, x, y, value);
        continue;
      }
      double sum = 0, divisor = 1;
      if (s->subsample_x == 1 || (s->placement == CP_TOP_LEFT && s->subsample_x != 4))
        sum = sample(sx, sy);
      else {
        const int vertical = s->subsample_y;
        if (s->subsample_x == 4 || s->placement == CP_CENTER) {
          divisor = s->subsample_x * vertical;
          for (int j = 0; j < vertical; ++j)
            for (int i = 0; i < s->subsample_x; ++i)
              sum += sample(sx + i, sy + j);
        } else {
          divisor = 4 * vertical;
          for (int j = 0; j < vertical; ++j)
            sum += sample(sx - 1, sy + j) + 2 * sample(sx, sy + j) + sample(sx + 1, sy + j);
        }
      }
      cp::put(destination, f, x, y, sum / divisor);
    }
  return CP_OK;
}
