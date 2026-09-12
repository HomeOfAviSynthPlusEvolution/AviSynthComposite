// SPDX-License-Identifier: GPL-2.0-or-later
// Independent mathematical and contract tests, no imported upstream source.
#include <composite/composite.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(expr)                                                                                                    \
  do {                                                                                                                 \
    if (!(expr))                                                                                                       \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #expr);                   \
  } while (false)
static uint64_t checks = 0;
static void equal(double a, double b, double tolerance = 0) {
  ++checks;
  if (!(std::abs(a - b) <= tolerance)) {
    std::ostringstream message;
    message << std::hexfloat << "actual=" << a << " expected=" << b << " tolerance=" << tolerance;
    throw std::runtime_error(message.str());
  }
}
static cp_rows rows(int w, int h = 1) {
  return {w, h, 0, h};
}
template <class T>
static cp_const_plane in(const T* p, int w, int step = 1) {
  return {p, ptrdiff_t(w * sizeof(T)), ptrdiff_t(step * sizeof(T))};
}
template <class T>
static cp_plane out(T* p, int w, int step = 1) {
  return {p, ptrdiff_t(w * sizeof(T)), ptrdiff_t(step * sizeof(T))};
}
static cp_format fmt(int bits) {
  return {bits == 8 ? CP_U8 : bits == 32 ? CP_F32 : CP_U16, bits};
}
static cp_plane_config config(int bits, int op = CP_MIX, double opacity = 1) {
  cp_plane_config c = {};
  c.format = fmt(bits);
  c.operation = op;
  c.opacity = opacity;
  return c;
}
static double rounded(double x, int max) {
  return std::floor(std::clamp(x, 0.0, double(max)) + 0.5);
}

template <class T>
static void arithmetic_depth(int bits) {
  const int max = (1 << bits) - 1;
  std::mt19937 random(182 + bits);
  const int n = 1003;
  std::vector<T> a(n), b(n), m(n), d(n), g(n);
  for (int i = 0; i < n; ++i) {
    a[i] = random() % (max + 1);
    b[i] = random() % (max + 1);
    m[i] = random() % (max + 1);
    g[i] = random() % (max + 1);
  }
  a[0] = 0;
  b[0] = max;
  m[0] = max;
  a[1] = max;
  b[1] = max;
  m[1] = 0;
  auto mask = in(m.data(), n), guide = in(g.data(), n), baseguide = in(a.data(), n);
  for (int op = CP_MIX; op <= CP_DIFFERENCE; ++op)
    for (int rule = 0; rule <= 1; ++rule)
      for (double opacity : {0., 0.003, 0.5, 0.63, 1.})
        for (bool masked : {false, true}) {
          auto c = config(bits, op, opacity);
          c.neutral = (max + 1) / 2;
          c.bias = 126 * (1 << (bits - 8));
          c.inversion_sum = max;
          c.threshold = 3;
          c.weight_rule = rule;
          CHECK(cp_process_plane(&c, in(a.data(), n), in(b.data(), n), masked ? &mask : nullptr, &baseguide, &guide,
                                 out(d.data(), n), rows(n)) == CP_OK);
          for (int i = 0; i < n; ++i) {
            // Reference uses long double, independent of the double implementation.
            long double w = opacity;
            if (rule) {
              auto level = std::floor(w * max + 0.5L);
              w = std::floor((masked ? m[i] : max) * level / max + 0.5L) / max;
            } else if (masked)
              w *= static_cast<long double>(m[i]) / max;
            long double target = b[i];
            switch (op) {
              case CP_ADD:
                target = static_cast<long double>(a[i]) + b[i];
                break;
              case CP_SUBTRACT:
                target = static_cast<long double>(a[i]) - b[i];
                break;
              case CP_PRODUCT:
                target = static_cast<uint64_t>(a[i]) * b[i] / max;
                break;
              case CP_INVERT_MIX:
                target = max - b[i];
                break;
              case CP_GUIDED_MULTIPLY:
                if (rule) {
                  const auto dark = std::floor(w * (max - g[i]) + 0.5L);
                  target = c.neutral;
                  w = dark / max;
                } else
                  target = c.neutral + (static_cast<long double>(a[i]) - c.neutral) * g[i] / max;
                break;
              case CP_SELECT_LIGHTER:
                if (!(int(g[i]) > int(a[i]) + 3))
                  w = 0;
                break;
              case CP_SELECT_DARKER:
                if (!(int(g[i]) < int(a[i]) - 3))
                  w = 0;
                break;
              case CP_DIFFERENCE:
                target = static_cast<long double>(a[i]) - b[i] + c.bias;
                break;
            }
            const auto expected = rounded(static_cast<double>((1 - w) * a[i] + w * target), max);
            equal(d[i], expected, 1); // half-code ties are also checked exactly below.
          }
        }
  auto c = config(bits, CP_MIX, 0.5);
  T aa = 0, bb = 1, dd = 0;
  CHECK(cp_process_plane(&c, in(&aa, 1), in(&bb, 1), nullptr, nullptr, nullptr, out(&dd, 1), rows(1)) == CP_OK);
  equal(dd, 1);
  c.operation = CP_PRODUCT;
  c.opacity = 1;
  aa = bb = max;
  CHECK(cp_process_plane(&c, in(&aa, 1), in(&bb, 1), nullptr, nullptr, nullptr, out(&dd, 1), rows(1)) == CP_OK);
  equal(dd, max);
}
static void arithmetic() {
  arithmetic_depth<uint8_t>(8);
  for (int bits = 9; bits <= 16; ++bits)
    arithmetic_depth<uint16_t>(bits);
}

static void compat() {
  // Exhaust all 8-bit base/source pairs at endpoint, interior and near-endpoint masks.
  std::vector<uint8_t> a(65536), b(65536), m(65536), d(65536);
  for (int i = 0; i < 65536; ++i) {
    a[i] = i >> 8;
    b[i] = i & 255;
  }
  auto mask = in(m.data(), 65536);
  for (int opacity : {0, 1, 127, 128, 255, 256})
    for (int mv : {-1, 0, 1, 127, 128, 254, 255}) {
      std::fill(m.begin(), m.end(), uint8_t(std::max(0, mv)));
      CHECK(cp_blend_compat(fmt(8), in(a.data(), 65536), in(b.data(), 65536), mv < 0 ? nullptr : &mask,
                            out(d.data(), 65536), rows(65536), opacity) == CP_OK);
      for (int i = 0; i < 65536; ++i) {
        int w = mv < 0 ? opacity : (mv * opacity) / 256;
        const int expected = mv == 255 && opacity == 256 ? b[i] : (int(a[i]) * (256 - w) + int(b[i]) * w + 128) / 256;
        equal(d[i], expected);
      }
    }
  for (int bits = 9; bits <= 16; ++bits) {
    uint16_t a16 = 0, b16 = (1 << bits) - 1, m16 = b16, d16 = 0;
    auto mm = in(&m16, 1);
    CHECK(cp_blend_compat(fmt(bits), in(&a16, 1), in(&b16, 1), &mm, out(&d16, 1), rows(1), 256) == CP_OK);
    equal(d16, b16);
    CHECK(cp_blend_compat(fmt(bits), in(&a16, 1), in(&b16, 1), &mm, out(&d16, 1), rows(1), 128) == CP_OK);
    equal(d16, (1 << (bits - 1)) - 1);
  }
}

template <class T>
static cp_const_yuv yin(const T* p) {
  return {in(p, 1), in(p + 1, 1), in(p + 2, 1)};
}
template <class T>
static cp_yuv yout(T* p) {
  return {out(p, 1), out(p + 1, 1), out(p + 2, 1)};
}
template <class T>
static void yuv_depth(int bits) {
  int max = (1 << bits) - 1, half = (max + 1) / 2;
  for (int op = 0; op <= 5; ++op) {
    T a[3] = {T(half), T(half), T(half)}, b[3] = {T(half), T(half), T(half)}, d[3] = {},
      m[3] = {T(max), T(max), T(max)};
    cp_yuv_config c = {fmt(bits), op, 1};
    CHECK(cp_process_yuv(&c, yin(a), yin(b), nullptr, yout(d), rows(1)) == CP_OK);
    double expected = half;
    if (op == CP_YUV_ADD)
      expected = max;
    if (op == CP_YUV_SUBTRACT)
      expected = 0;
    if (op == CP_YUV_EXCLUSION)
      expected = std::floor(2.0 * half * (max - half) / max);
    equal(d[0], expected);
    auto mm = yin(m);
    T d2[3] = {};
    CHECK(cp_process_yuv(&c, yin(a), yin(b), &mm, yout(d2), rows(1)) == CP_OK);
    CHECK(std::memcmp(d, d2, sizeof(d)) == 0);
    c.opacity = 0;
    CHECK(cp_process_yuv(&c, yin(a), yin(b), nullptr, yout(d), rows(1)) == CP_OK);
    CHECK(std::memcmp(a, d, sizeof(a)) == 0);
  }
  // Superwhite/desaturation transition at max+1, and fully white beyond it.
  T a[3] = {T(max), 0, T(max)}, b[3] = {T((max + 1) / 8 + 1), T(half), T(half)}, d[3] = {};
  cp_yuv_config c = {fmt(bits), CP_YUV_ADD, 1};
  CHECK(cp_process_yuv(&c, yin(a), yin(b), nullptr, yout(d), rows(1)) == CP_OK);
  equal(d[0], max);
  equal(d[1], half);
  equal(d[2], half);
}
static void yuv() {
  yuv_depth<uint8_t>(8);
  for (int b = 9; b <= 16; ++b)
    yuv_depth<uint16_t>(b);
}

static void sampling() {
  uint16_t s[15] = {0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400};
  for (int sx : {1, 2, 4})
    for (int sy : {1, 2})
      for (int placement = 0; placement < 3; ++placement) {
        if (sy == 2 && sx != 2)
          continue;
        for (int phase : {-3, 0, 1, 4}) {
          cp_sampling c = {5, 3, sx, sy, placement, phase, -1};
          uint16_t d[12] = {}, band[12] = {};
          CHECK(cp_resample_mask(fmt(16), in(s, 5), out(d, 4), &c, rows(4, 3)) == CP_OK);
          for (int yy = 0; yy < 3; ++yy)
            CHECK(cp_resample_mask(fmt(16), in(s, 5), out(band, 4), &c, {4, 3, yy, 1}) == CP_OK);
          CHECK(std::memcmp(d, band, sizeof(d)) == 0);
          // Independent explicit tap list.
          for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 4; ++x) {
              std::vector<std::array<int, 3>> taps;
              if (sx == 1 || (placement == CP_TOP_LEFT && sx != 4))
                taps.push_back({0, 0, 1});
              else if (sx == 4 || placement == CP_CENTER)
                for (int j = 0; j < sy; ++j)
                  for (int i = 0; i < sx; ++i)
                    taps.push_back({i, j, 1});
              else
                for (int j = 0; j < sy; ++j) {
                  taps.push_back({-1, j, 1});
                  taps.push_back({0, j, 2});
                  taps.push_back({1, j, 1});
                }
              int sum = 0, weight = 0;
              for (auto t : taps) {
                sum += s[std::clamp(y * sy - 1 + t[1], 0, 2) * 5 + std::clamp(x * sx + phase + t[0], 0, 4)] * t[2];
                weight += t[2];
              }
              equal(d[y * 4 + x], (sum + weight / 2) / weight);
            }
        }
      }
}

static void utilities() {
  uint8_t packed[8] = {0, 0, 255, 73, 255, 255, 255, 91}; // BGRA
  cp_const_rgb rgb = {in(packed + 2, 8, 4), in(packed + 1, 8, 4), in(packed, 8, 4)};
  uint8_t luma[2] = {};
  CHECK(cp_rgb_luma(fmt(8), rgb, out(luma, 2), rows(2), CP_LUMA_FLOOR) == CP_OK);
  equal(luma[0], 76);
  equal(luma[1], 255);
  const double key[3] = {255, 0, 0}, tol[3] = {0, 0, 0};
  CHECK(cp_color_key(fmt(8), rgb, in(packed + 3, 8, 4), out(packed + 3, 8, 4), rows(2), key, tol) == CP_OK);
  equal(packed[3], 0);
  equal(packed[7], 91);
  equal(packed[2], 255);
  uint16_t c[3] = {0, 128, 255}, d[3] = {};
  CHECK(cp_affine(fmt(9), in(c, 3), out(d, 3), rows(3), -1, 512) == CP_OK);
  equal(d[0], 511);
  equal(d[1], 384);
  CHECK(cp_fill(fmt(9), out(d, 3), rows(3), 1000) == CP_OK);
  equal(d[2], 511);
  cp_overlap overlap = {};
  CHECK(cp_intersect(10, 10, 5, 5, -2, 8, &overlap) == CP_OK);
  equal(overlap.base_y, 8);
  equal(overlap.source_x, 2);
  equal(overlap.width, 3);
  equal(overlap.height, 2);
  CHECK(cp_intersect(10, 10, 5, 5, std::numeric_limits<int>::min(), 0, &overlap) == CP_OK);
  equal(overlap.width, 0);
}

static void boundaries() {
  // Odd width, packed step, independent negative strides, canaries in every gap.
  std::vector<uint16_t> a(75, 0xABCD), b(90, 0xABCD), d(100, 0xABCD);
  cp_const_plane ap = {a.data() + 50, -40, 6}, bp = {b.data() + 5, 50, 8};
  cp_plane dp = {d.data() + 55, -44, 6};
  auto c = config(16, CP_MIX, 0.5);
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 5; ++x) {
      a[50 - 20 * y + 3 * x] = 0;
      b[5 + 25 * y + 4 * x] = 65535;
    }
  CHECK(cp_process_plane(&c, ap, bp, nullptr, nullptr, nullptr, dp, {5, 3, 1, 1}) == CP_OK);
  for (int i = 0; i < 100; ++i) {
    bool written = false;
    for (int x = 0; x < 5; ++x)
      written |= i == 33 + 3 * x;
    equal(d[i], written ? 32768 : 0xABCD);
  }
  const auto saved = d;
  cp_const_plane bad = ap;
  bad.stride = 1;
  CHECK(cp_process_plane(&c, bad, bp, nullptr, nullptr, nullptr, dp, rows(5, 3)) == CP_INVALID_ARGUMENT);
  CHECK(saved == d);
  bad = ap;
  bad.step = std::numeric_limits<ptrdiff_t>::max() - 1;
  CHECK(cp_copy(fmt(16), bad, dp, rows(5, 3)) == CP_INVALID_ARGUMENT);
  CHECK(saved == d);
  CHECK(cp_process_plane(&c, {}, {}, nullptr, nullptr, nullptr, {}, {5, 3, 3, 0}) == CP_OK);
  CHECK(cp_process_plane(&c, ap, bp, nullptr, nullptr, nullptr, dp, {5, 3, 3, 1}) == CP_INVALID_ARGUMENT);
  c.opacity = std::numeric_limits<double>::quiet_NaN();
  CHECK(cp_process_plane(&c, ap, bp, nullptr, nullptr, nullptr, dp, rows(5, 3)) == CP_INVALID_ARGUMENT);
  CHECK(saved == d);
}

static void floating() {
  uint32_t bits[2] = {0x80000000, 0x7fc12345};
  float a[2], b[2] = {1, 2}, d[2];
  std::memcpy(a, bits, sizeof(a));
  auto c = config(32, CP_MIX, 0);
  CHECK(cp_process_plane(&c, in(a, 2), in(b, 2), nullptr, nullptr, nullptr, out(d, 2), rows(2)) == CP_OK);
  CHECK(std::memcmp(a, d, sizeof(a)) == 0);
  c.opacity = 1;
  CHECK(cp_process_plane(&c, in(b, 2), in(a, 2), nullptr, nullptr, nullptr, out(d, 2), rows(2)) == CP_OK);
  CHECK(std::memcmp(a, d, sizeof(a)) == 0);
  float ay[3] = {1, 0.3f, -0.4f}, by[3] = {0.2f, 0, 0}, dy[3];
  cp_yuv_config yc = {fmt(32), CP_YUV_ADD, 1};
  CHECK(cp_process_yuv(&yc, yin(ay), yin(by), nullptr, yout(dy), rows(1)) == CP_OK);
  equal(dy[0], 1);
  equal(dy[1], 0);
  equal(dy[2], 0);
  ay[0] = -0.5f;
  by[0] = 0.1f;
  CHECK(cp_process_yuv(&yc, yin(ay), yin(by), nullptr, yout(dy), rows(1)) == CP_OK);
  equal(dy[0], -0.4, 1e-7);
  equal(dy[1], 0.3, 1e-7);
  yc.operation = CP_YUV_DIFFERENCE;
  CHECK(cp_process_yuv(&yc, yin(ay), yin(by), nullptr, yout(dy), rows(1)) == CP_UNSUPPORTED);
  // A half-ULP threshold must round in the same domain as the upstream guide.
  float ga = 1, gb = 1, aa = 0, bb = 1, dd = 0;
  auto gav = in(&ga, 1), gbv = in(&gb, 1);
  c = config(32, CP_SELECT_LIGHTER);
  c.inclusive = 1;
  c.threshold = std::ldexp(1.0, -25);
  CHECK(cp_process_plane(&c, in(&aa, 1), in(&bb, 1), nullptr, &gav, &gbv, out(&dd, 1), rows(1)) == CP_OK);
  equal(dd, 1);
  c.inclusive = 0;
  CHECK(cp_process_plane(&c, in(&aa, 1), in(&bb, 1), nullptr, &gav, &gbv, out(&dd, 1), rows(1)) == CP_OK);
  equal(dd, 0);
  std::mt19937 random(889);
  for (int iteration = 0; iteration < 2000; ++iteration) {
    float r = float(random() % 10001) / 10000, g = float(random() % 10001) / 10000,
          bl = float(random() % 10001) / 10000, lum = 0;
    CHECK(cp_rgb_luma(fmt(32), {in(&r, 1), in(&g, 1), in(&bl, 1)}, out(&lum, 1), rows(1), CP_LUMA_FLOOR) == CP_OK);
    float expected = 0.114f * bl + 0.587f * g + 0.299f * r;
    equal(lum, expected);
    float samples[4] = {r, g, bl, lum}, filtered = 0;
    cp_sampling s = {2, 2, 2, 2, CP_CENTER, 0, 0};
    CHECK(cp_resample_mask(fmt(32), in(samples, 2), out(&filtered, 1), &s, rows(1)) == CP_OK);
    expected = (r + g + bl + lum) * 0.25f;
    equal(filtered, expected);
  }
  float excursions[5] = {-2, -0.25f, 0.2f, 2, std::numeric_limits<float>::quiet_NaN()}, clipped[5] = {};
  CHECK(cp_clamp(fmt(32), in(excursions, 5), out(clipped, 5), rows(5), -0.5, 0.5) == CP_OK);
  equal(clipped[0], -.5);
  equal(clipped[1], -.25);
  equal(clipped[2], .2, 1e-7);
  equal(clipped[3], .5);
  equal(clipped[4], -.5);
}

static void composition() {
  // Packed monochrome Layer Subtract blends every BGRA channel toward the
  // FLOOR luma of inverted source RGB, including the alpha target. Keep the
  // original alpha immutable; monochrome does not mean opacity-only weighting.
  uint8_t packed[8] = {3, 81, 199, 17, 240, 11, 89, 233}, inv[8] = {}, luma[2] = {}, dest[8] = {};
  cp_const_plane channels[4];
  for (int p = 0; p < 4; ++p) {
    channels[p] = {packed + p, 8, 4};
    CHECK(cp_affine(fmt(8), channels[p], {inv + p, 8, 4}, rows(2), -1, 255) == CP_OK);
  }
  CHECK(cp_rgb_luma(fmt(8), {{inv + 2, 8, 4}, {inv + 1, 8, 4}, {inv, 8, 4}}, out(luma, 2), rows(2), CP_LUMA_FLOOR) ==
        CP_OK);
  auto mono = config(8, CP_MIX, 0.5);
  mono.weight_rule = CP_WEIGHT_CODE;
  for (int p = 0; p < 4; ++p)
    CHECK(cp_process_plane(&mono, channels[p], in(luma, 2), &channels[3], nullptr, nullptr, {dest + p, 8, 4},
                           rows(2)) == CP_OK);
  for (int x = 0; x < 2; ++x)
    for (int p = 0; p < 4; ++p) {
      const int lum =
          (9798 * (255 - packed[x * 4 + 2]) + 19234 * (255 - packed[x * 4 + 1]) + 3736 * (255 - packed[x * 4])) / 32768;
      const int weight = (packed[x * 4 + 3] * 128 + 127) / 255;
      equal(dest[x * 4 + p], (packed[x * 4 + p] * (255 - weight) + lum * weight + 127) / 255);
    }
  // Layer Subtract must use ORIGINAL alpha as weight and inverted alpha as target.
  uint8_t base[4] = {40, 60, 80, 100}, source[4] = {10, 20, 30, 128}, d[4] = {};
  const auto mask = in(source + 3, 1);
  auto c = config(8, CP_INVERT_MIX, 0.5);
  c.inversion_sum = 255;
  c.weight_rule = CP_WEIGHT_CODE;
  for (int p = 0; p < 4; ++p)
    CHECK(cp_process_plane(&c, in(base + p, 1), in(source + p, 1), &mask, nullptr, nullptr, out(d + p, 1), rows(1)) ==
          CP_OK);
  for (int p = 0; p < 4; ++p)
    equal(d[p], (base[p] * (255 - 64) + (255 - source[p]) * 64 + 127) / 255);
  // Inclusive Overlay and strict Layer must differ on equal Y despite distinct UV.
  uint8_t ay = 100, by = 100, au = 0, bu = 255, result = 0;
  auto ga = in(&ay, 1), gb = in(&by, 1);
  c = config(8, CP_SELECT_LIGHTER);
  c.inclusive = 1;
  CHECK(cp_process_plane(&c, in(&au, 1), in(&bu, 1), nullptr, &ga, &gb, out(&result, 1), rows(1)) == CP_OK);
  equal(result, 255);
  c.inclusive = 0;
  CHECK(cp_process_plane(&c, in(&au, 1), in(&bu, 1), nullptr, &ga, &gb, out(&result, 1), rows(1)) == CP_OK);
  equal(result, 0);
  // Subsampled decision uses rounded average of ORIGINAL Y, never updated luma.
  uint8_t source_y[4] = {0, 255, 0, 255}, base_y[4] = {100, 100, 100, 100}, sg[2], bg[2];
  cp_sampling s = {4, 1, 2, 1, CP_CENTER, 0, 0};
  CHECK(cp_resample_mask(fmt(8), in(source_y, 4), out(sg, 2), &s, rows(2)) == CP_OK);
  CHECK(cp_resample_mask(fmt(8), in(base_y, 4), out(bg, 2), &s, rows(2)) == CP_OK);
  equal(sg[0], 128);
  equal(bg[0], 100);
}

template <class T, class S>
static void fixture_case(int bits, int mode, bool masked, float opacity, const S samples[12]) {
  T a[3], b[3], m[3], d[3] = {};
  for (int p = 0; p < 3; ++p) {
    a[p] = T(samples[p]);
    b[p] = T(samples[3 + p]);
    m[p] = T(samples[6 + p]);
  }
  if (mode < 6) {
    cp_yuv_config c = {fmt(bits), mode, opacity};
    auto mm = yin(m);
    CHECK(cp_process_yuv(&c, yin(a), yin(b), masked ? &mm : nullptr, yout(d), rows(1)) == CP_OK);
  } else
    for (int p = 0; p < 3; ++p) {
      auto c = config(bits, mode == 6 ? CP_GUIDED_MULTIPLY : mode == 7 ? CP_SELECT_LIGHTER : CP_SELECT_DARKER, opacity);
      c.neutral = p ? (1 << (bits - 1)) : 0;
      c.inclusive = !masked && opacity == 1;
      c.weight_rule = mode == 6 ? CP_WEIGHT_CONTINUOUS : CP_WEIGHT_CODE;
      auto mm = in(m + p, 1), ga = in(a, 1), gb = in(b, 1);
      CHECK(cp_process_plane(&c, in(a + p, 1), in(b + p, 1), masked ? &mm : nullptr, &ga, &gb, out(d + p, 1),
                             rows(1)) == CP_OK);
    }
  for (int p = 0; p < 3; ++p)
    equal(d[p], samples[9 + p], bits == 32 ? 8e-7 : mode < 6 ? 0 : 1);
}
template <class T, class S>
static void layer_fixture(int bits, int mode, bool chroma, bool alpha, float opacity, const S s[12]) {
  T a[4], b[4], d[4];
  for (int p = 0; p < 4; ++p) {
    a[p] = T(s[p]);
    b[p] = T(s[4 + p]);
    d[p] = a[p];
  }
  T ag = 0, bg = 0;
  CHECK(cp_rgb_luma(fmt(bits), {in(a + 2, 1), in(a, 1), in(a + 1, 1)}, out(&ag, 1), rows(1), CP_LUMA_FLOOR) == CP_OK);
  CHECK(cp_rgb_luma(fmt(bits), {in(b + 2, 1), in(b, 1), in(b + 1, 1)}, out(&bg, 1), rows(1), CP_LUMA_FLOOR) == CP_OK);
  for (int p = 0; p < (alpha ? 4 : 3); ++p) {
    auto c = config(bits,
                    mode == 0   ? CP_MIX
                    : mode == 1 ? CP_PRODUCT
                    : mode == 2 ? CP_SELECT_LIGHTER
                                : CP_SELECT_DARKER,
                    opacity);
    c.weight_rule = CP_WEIGHT_CODE;
    c.threshold = bits == 32 ? 3.0f / 255 : 3;
    auto mask = in(b + 3, 1), ga = in(&ag, 1), gb = in(&bg, 1);
    CHECK(cp_process_plane(&c, in(a + p, 1), (!chroma && p < 3) ? gb : in(b + p, 1), alpha ? &mask : nullptr, &ga, &gb,
                           out(d + p, 1), rows(1)) == CP_OK);
  }
  for (int p = 0; p < 4; ++p)
    equal(d[p], s[8 + p], bits == 32 ? 2e-7 : 0);
}
template <class T>
static void subsample_fixture(int bits, int mode, int plane, bool alpha, float opacity, const int s[6]) {
  T a = T(s[0]), b = T(s[1]), ag = T(s[2]), bg = T(s[3]), m = T(s[4]), d = 0;
  auto c = config(bits,
                  mode == 0   ? (plane ? CP_MIX : CP_PRODUCT)
                  : mode == 1 ? CP_GUIDED_MULTIPLY
                  : mode == 2 ? CP_SELECT_LIGHTER
                              : CP_SELECT_DARKER,
                  opacity);
  c.weight_rule = CP_WEIGHT_CODE;
  c.threshold = 3;
  c.neutral = plane ? (1 << (bits - 1)) : 0;
  auto mask = in(&m, 1), ga = in(&ag, 1), gb = in(&bg, 1);
  CHECK(cp_process_plane(&c, in(&a, 1), in(&b, 1), alpha ? &mask : nullptr, &ga, &gb, out(&d, 1), rows(1)) == CP_OK);
  equal(d, s[5], mode == 1 ? 1 : 0);
}
static void upstream() {
  std::ifstream file(std::string(CP_FIXTURES) + "/overlay.csv");
  CHECK(file.good());
  std::string line;
  int count = 0;
  while (std::getline(file, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream data(line);
    int bits, mode, masked, samples[12];
    float opacity;
    CHECK(bool(data >> bits >> mode >> masked >> opacity));
    for (auto& sample : samples)
      CHECK(bool(data >> sample));
    if (bits == 8)
      fixture_case<uint8_t>(bits, mode, masked != 0, opacity, samples);
    else
      fixture_case<uint16_t>(bits, mode, masked != 0, opacity, samples);
    ++count;
  }
  CHECK(count == 3600);
  std::ifstream layer(std::string(CP_FIXTURES) + "/layer.csv");
  CHECK(layer.good());
  count = 0;
  while (std::getline(layer, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream data(line);
    int bits, mode, chroma, alpha, s[12];
    float opacity;
    CHECK(bool(data >> bits >> mode >> chroma >> alpha >> opacity));
    for (auto& sample : s)
      CHECK(bool(data >> sample));
    if (bits == 8)
      layer_fixture<uint8_t>(bits, mode, chroma != 0, alpha != 0, opacity, s);
    else
      layer_fixture<uint16_t>(bits, mode, chroma != 0, alpha != 0, opacity, s);
    ++count;
  }
  CHECK(count == 2400);
  std::ifstream subsample(std::string(CP_FIXTURES) + "/subsample.csv");
  CHECK(subsample.good());
  count = 0;
  while (std::getline(subsample, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream data(line);
    int bits, mode, plane, alpha, s[6];
    float opacity;
    CHECK(bool(data >> bits >> mode >> plane >> alpha >> opacity));
    for (auto& sample : s)
      CHECK(bool(data >> sample));
    if (bits == 8)
      subsample_fixture<uint8_t>(bits, mode, plane, alpha != 0, opacity, s);
    else
      subsample_fixture<uint16_t>(bits, mode, plane, alpha != 0, opacity, s);
    ++count;
  }
  CHECK(count == 9600);
  for (bool rgb : {false, true}) {
    std::ifstream floating(std::string(CP_FIXTURES) + (rgb ? "/float-layer.csv" : "/float-overlay.csv"));
    CHECK(floating.good());
    count = 0;
    while (std::getline(floating, line)) {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream data(line);
      int bits, mode, flag, alpha = 0;
      float opacity, s[12];
      CHECK(bool(data >> bits >> mode >> flag));
      CHECK(bits == 32);
      if (rgb)
        CHECK(bool(data >> alpha));
      CHECK(bool(data >> opacity));
      for (auto& sample : s)
        CHECK(bool(data >> sample));
      if (rgb)
        layer_fixture<float>(32, mode, flag != 0, alpha != 0, opacity, s);
      else
        fixture_case<float>(32, mode, flag != 0, opacity, s);
      ++count;
    }
    CHECK(count == (rgb ? 480 : 160));
  }
}

static void concurrent_bands() {
  std::vector<uint16_t> a(1000, 100), b(1000, 500), d(1000, 0);
  const auto c = config(10, CP_MIX, 0.25);
  std::vector<std::future<int>> jobs;
  for (int i = 0; i < 4; ++i)
    jobs.push_back(std::async(std::launch::async, [&, i] {
      for (int row = i; row < 10; row += 4) {
        const int status = cp_process_plane(&c, in(a.data(), 100), in(b.data(), 100), nullptr, nullptr, nullptr,
                                            out(d.data(), 100), {100, 10, row, 1});
        if (status != CP_OK)
          return status;
      }
      return int(CP_OK);
    }));
  for (auto& j : jobs)
    CHECK(j.get() == CP_OK);
  for (auto v : d)
    equal(v, 200);
}
int main(int argc, char** argv) {
  if (argc != 2)
    return 2;
  const std::pair<const char*, void (*)()> suites[] = {
      {"arithmetic", arithmetic},        {"compat", compat},         {"yuv", yuv},        {"sampling", sampling},
      {"utilities", utilities},          {"boundaries", boundaries}, {"float", floating}, {"composition", composition},
      {"concurrency", concurrent_bands}, {"upstream", upstream}};
  try {
    for (auto s : suites)
      if (s.first == std::string(argv[1])) {
        s.second();
        std::cout << s.first << ": " << checks << " numeric checks passed\n";
        return 0;
      }
  } catch (const std::exception& e) {
    std::cerr << argv[1] << ": " << e.what() << "\n";
    return 1;
  }
  return 2;
}
