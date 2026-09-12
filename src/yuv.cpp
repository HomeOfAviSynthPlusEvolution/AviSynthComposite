// SPDX-License-Identifier: GPL-2.0-or-later
// AviSynth (c) 2002 Ben Rudiak-Gould et al.; Overlay (c) 2003, 2004 Klaus Post.
// Reimplementation of AviSynth+ artistic YUV semantics; see LICENSE (including
// inherited linking exception). These names do not denote Photoshop RGB modes.
#include "common.h"

// Enumerate only exact half-integer outputs for opacity level/256. Solving
// factor*a + neutral*(D-factor) == D/2 (mod D) skips all non-ties. Their
// distance from a half is >=1/D, well above binary64 evaluation error.
void cp::u8_multiply_corrections(unsigned level, uint32_t* corrections, uint32_t* keys, unsigned* counts) {
  constexpr int denominator = 255 * 256;
  std::memset(corrections, 0, 4096 * sizeof(*corrections));
  counts[0] = counts[1] = 0;
  for (int guide = 0; guide < 256; ++guide) {
    const int factor = 255 * (256 - int(level)) + guide * int(level);
    int a = factor, b = denominator;
    while (b) {
      const int rest = a % b;
      a = b;
      b = rest;
    }
    const int divisor = a, modulus = denominator / divisor;
    int r0 = modulus, r1 = factor / divisor;
    int64_t t0 = 0, t1 = 1;
    while (r1) {
      const int q = r0 / r1, r2 = r0 - q * r1;
      const int64_t t2 = t0 - q * t1;
      r0 = r1;
      r1 = r2;
      t0 = t1;
      t1 = t2;
    }
    const int64_t inverse = (t0 % modulus + modulus) % modulus;
    for (int p = 0; p < 2; ++p) {
      const int neutral = p ? 128 : 0;
      int rhs = (denominator / 2 - neutral * (denominator - factor)) % denominator;
      if (rhs < 0)
        rhs += denominator;
      if (rhs % divisor)
        continue;
      const int first = int((rhs / divisor * inverse) % modulus);
      for (int value = first; value < 256; value += modulus) {
        const int exact = (value * factor + neutral * (denominator - factor) + denominator / 2) / denominator;
        const double target = neutral + (double(value) - neutral) * guide / 255;
        const int original = int(std::floor(cp::mix(value, target, level / 256.0) + .5));
        if (original < exact) {
          corrections[p * 2048 + guide * 8 + value / 32] |= uint32_t(1) << (value % 32);
          if (counts[p] < 8)
            keys[p * 8 + counts[p]] = unsigned(guide * 256 + value);
          ++counts[p];
        }
      }
    }
  }
}

int cp_process_yuv(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* mask,
                   cp_yuv destination, cp_rows r) {
  if (!c || !cp::bytes(c->format) || !cp::rows_ok(r) || !cp::opacity_ok(c->opacity) || c->operation < CP_YUV_ADD ||
      c->operation > CP_YUV_MULTIPLY)
    return CP_INVALID_ARGUMENT;
  const cp_format f = c->format;
  const bool floating = f.storage == CP_F32;
  if (floating && c->operation > CP_YUV_SUBTRACT && c->operation != CP_YUV_MULTIPLY)
    return CP_UNSUPPORTED;
  if (!r.count)
    return CP_OK;
  const int bytes = cp::bytes(f);
  cp_const_plane a[3] = {base.y, base.u, base.v}, b[3] = {source.y, source.u, source.v};
  cp_plane d[3] = {destination.y, destination.u, destination.v};
  cp_const_plane m[3] = {};
  if (mask) {
    m[0] = mask->y;
    m[1] = mask->u;
    m[2] = mask->v;
  }
  for (int p = 0; p < 3; ++p)
    if (!cp::plane_ok(a[p], r, bytes) || !cp::plane_ok(b[p], r, bytes) || !cp::plane_ok(d[p], r, bytes) ||
        (mask && !cp::plane_ok(m[p], r, bytes)))
      return CP_INVALID_ARGUMENT;
  if (c->operation == CP_YUV_MULTIPLY) {
    const double maximum = cp::maximum(f), half = floating ? 0 : (maximum + 1) / 2;
    for (int y = r.first; y < r.first + r.count; ++y)
      for (int x = 0; x < r.width; ++x) {
        unsigned char original[3][4];
        double result[3], weight[3];
        const double guide = cp::get(b[0], f, x, y);
        for (int p = 0; p < 3; ++p) {
          std::memcpy(original[p], cp::address(a[p], x, y), bytes);
          const double av = cp::get(a[p], f, x, y), neutral = p == 0 ? 0 : half;
          weight[p] = cp::effective(f, c->opacity, mask ? &m[p] : nullptr, x, y, CP_WEIGHT_CONTINUOUS);
          result[p] = cp::mix(av, neutral + (av - neutral) * guide / maximum, weight[p]);
        }
        for (int p = 0; p < 3; ++p) {
          if (weight[p] == 0)
            std::memcpy(cp::address(d[p], x, y), original[p], bytes);
          else
            cp::put(d[p], f, x, y, result[p]);
        }
      }
    return CP_OK;
  }
  const double max = cp::maximum(f), half = floating ? 0 : (max + 1) / 2;
  const double over = floating ? 32.0 / 255 : (max + 1) / 8;
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width; ++x) {
      double av[3], bv[3], w[3], v[3];
      for (int p = 0; p < 3; ++p) {
        av[p] = cp::get(a[p], f, x, y);
        bv[p] = cp::get(b[p], f, x, y);
        w[p] = cp::effective(f, c->opacity, mask ? &m[p] : nullptr, x, y, CP_WEIGHT_CODE);
      }
      if (w[0] == 0 && w[1] == 0 && w[2] == 0) {
        for (int p = 0; p < 3; ++p)
          cp::copy_sample(a[p], d[p], x, y, bytes);
        continue;
      }
      for (int p = 0; p < 3; ++p) {
        const double center = p == 0 ? 0 : half;
        double target;
        switch (c->operation) {
          case CP_YUV_ADD:
          case CP_YUV_SUBTRACT: {
            double delta = (bv[p] - center) * w[p];
            if (!floating)
              delta = std::copysign(std::floor(std::abs(delta) + 0.5), delta);
            v[p] = av[p] + (c->operation == CP_YUV_ADD ? delta : -delta);
            continue;
          }
          case CP_YUV_SOFT_LIGHT:
            target = av[p] + bv[p] - half;
            break;
          case CP_YUV_HARD_LIGHT:
            target = av[p] + (p == 0 ? 2 : 1) * (bv[p] - half);
            break;
          case CP_YUV_DIFFERENCE:
            target = std::abs(av[p] - bv[p]) + half;
            break;
          default:
            target = std::floor(((max - av[p]) * bv[0] + (max - bv[0]) * av[p]) / max);
            break;
        }
        v[p] = cp::mix(av[p], target, w[p]);
        if (!floating)
          v[p] = std::floor(v[p] + 0.5);
      }
      double keep = 1;
      // Integer upper transition starts at max+1; preserve this one-code boundary.
      if (v[0] > max && (!floating || c->operation == CP_YUV_ADD)) {
        keep = std::max(0.0, (max + (floating ? 0 : 1) + over - v[0]) / over);
        v[0] = max;
      } else if (v[0] < 0 && (!floating || c->operation == CP_YUV_SUBTRACT)) {
        keep = std::max(0.0, 1 + v[0] / over);
        v[0] = 0;
      }
      if (keep != 1)
        for (int p = 1; p < 3; ++p) {
          v[p] = half + (v[p] - half) * keep;
          if (!floating)
            v[p] = std::floor(v[p]);
        }
      for (int p = 0; p < 3; ++p)
        cp::put(d[p], f, x, y, v[p]);
    }
  return CP_OK;
}
