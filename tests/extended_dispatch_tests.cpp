// SPDX-License-Identifier: GPL-2.0-or-later
#include <composite/composite.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>
#define CHECK(e)                                                                                                       \
  do {                                                                                                                 \
    if (!(e)) {                                                                                                        \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #e);                                                             \
      std::abort();                                                                                                    \
    }                                                                                                                  \
  } while (0)
template <class T>
struct Image {
  int w, h, step, pitch, origin;
  std::vector<T> data;
  Image(int width, int height, int spacing, bool negative, int padding = 9)
      : w(width), h(height), step(spacing), pitch(width * spacing + padding),
        origin(negative ? (height - 1) * pitch : 0), data(size_t(pitch) * height, T(17)) {
    if (negative)
      pitch = -pitch;
  }
  cp_const_plane in() const {
    return {data.data() + origin, pitch * ptrdiff_t(sizeof(T)), step * ptrdiff_t(sizeof(T))};
  }
  cp_plane out() { return {data.data() + origin, pitch * ptrdiff_t(sizeof(T)), step * ptrdiff_t(sizeof(T))}; }
  cp_rows rows() const { return {w, h, 0, h}; }
  void populate(int bits, unsigned seed, bool mask = false) {
    std::mt19937 rng(seed);
    for (auto& v : data) {
      if constexpr (std::is_same<T, float>::value)
        v = mask ? (rng() % 65536) / 65535.f : float(int(rng() % 98305) - 16384) / 65536.f;
      else
        v = T(rng() & ((1u << bits) - 1));
    }
  }
};
template <class T>
void same(const Image<T>& a, const Image<T>& b, const char* label, int tolerance = 0) {
  CHECK(a.data.size() == b.data.size());
  for (size_t i = 0; i < a.data.size(); ++i)
    if (std::memcmp(&a.data[i], &b.data[i], sizeof(T))) {
      bool pixel = false;
      for (int y = 0; y < a.h; ++y) {
        const auto x = ptrdiff_t(i) - a.origin - y * a.pitch;
        pixel = pixel || (x >= 0 && x < a.w * a.step && x % a.step == 0);
      }
      if (tolerance && pixel && std::abs(double(a.data[i]) - double(b.data[i])) <= tolerance)
        continue;
      if constexpr (std::is_same<T, float>::value)
        if (std::isnan(a.data[i]) && std::isnan(b.data[i]))
          continue;
      std::fprintf(stderr, "%s sample %zu got %.17g expected %.17g\n", label, i, double(a.data[i]), double(b.data[i]));
      CHECK(false);
    }
}
static cp_format format(int bits) {
  return {bits == 8 ? CP_U8 : bits == 32 ? CP_F32 : CP_U16, bits};
}
template <class T>
void Utilities(const cp_kernels* k, int bits, int width, int step, bool negative) {
  const auto f = format(bits);
  const double max = bits == 32 ? 1.0 : (1u << bits) - 1;
  Image<T> a(width, 4, step, negative), b = a, c = a, alpha = a, got = a, ref = a;
  a.populate(bits, 19);
  b.populate(bits, 23);
  c.populate(bits, 41);
  alpha.populate(bits, 88, true);
  const cp_const_rgb rgb = {a.in(), b.in(), c.in()};
  const cp_rows band = {width, 4, 1, 2};
  for (const auto params : {std::array<double, 2>{-1, max}, {0, max / 2}, {.75, max / 8}, {1, 0}, {1.5, -max / 4}}) {
    CHECK(k->affine(f, a.in(), got.out(), band, params[0], params[1]) == CP_OK);
    CHECK(cp_affine(f, a.in(), ref.out(), band, params[0], params[1]) == CP_OK);
    same(got, ref, "affine");
    std::copy(a.data.begin(), a.data.end(), got.data.begin());
    std::copy(a.data.begin(), a.data.end(), ref.data.begin());
    CHECK(k->affine(f, got.in(), got.out(), band, params[0], params[1]) == CP_OK);
    CHECK(cp_affine(f, ref.in(), ref.out(), band, params[0], params[1]) == CP_OK);
    same(got, ref, "affine in-place");
  }
  for (const auto params : {std::array<double, 2>{0, max}, {max / 4, max / 2}, {-max / 2, max * 2}, {0, 0}}) {
    CHECK(k->clamp(f, a.in(), got.out(), band, params[0], params[1]) == CP_OK);
    CHECK(cp_clamp(f, a.in(), ref.out(), band, params[0], params[1]) == CP_OK);
    same(got, ref, "clamp");
  }
  for (int rounding : {CP_LUMA_FLOOR, CP_LUMA_NEAREST}) {
    CHECK(k->rgb_luma(f, rgb, got.out(), band, rounding) == CP_OK);
    CHECK(cp_rgb_luma(f, rgb, ref.out(), band, rounding) == CP_OK);
    same(got, ref, "luma");
  }
  const double key[3] = {max / 2, max / 2, max / 2};
  for (double tolerance : {0.0, max / 8, max / 2, max}) {
    const double tol[3] = {tolerance, tolerance, tolerance};
    CHECK(k->color_key(f, rgb, alpha.in(), got.out(), band, key, tol) == CP_OK);
    CHECK(cp_color_key(f, rgb, alpha.in(), ref.out(), band, key, tol) == CP_OK);
    same(got, ref, "color key");
  }
  const auto before = got;
  const double bad[3] = {0, -1, 0};
  CHECK(k->affine(f, a.in(), got.out(), band, INFINITY, 0) == CP_INVALID_ARGUMENT);
  CHECK(k->clamp(f, a.in(), got.out(), band, 1, 0) == CP_INVALID_ARGUMENT);
  CHECK(k->rgb_luma(f, rgb, got.out(), band, 2) == CP_INVALID_ARGUMENT);
  CHECK(k->color_key(f, rgb, alpha.in(), got.out(), band, key, bad) == CP_INVALID_ARGUMENT);
  same(got, before, "invalid utilities");
}
template <class T>
void Yuv(const cp_kernels* k, int bits, int width, int step, bool negative) {
  const auto f = format(bits);
  Image<T> proto(width, 4, step, negative);
  std::array<Image<T>, 3> a = {proto, proto, proto}, b = a, m = a, got = a, ref = a;
  for (int p = 0; p < 3; ++p) {
    a[p].populate(bits, 127 + p);
    b[p].populate(bits, 171 + p);
    m[p].populate(bits, 313 + p, true);
  }
  const cp_const_yuv av = {a[0].in(), a[1].in(), a[2].in()}, bv = {b[0].in(), b[1].in(), b[2].in()},
                     mv = {m[0].in(), m[1].in(), m[2].in()};
  const cp_rows band = {width, 4, 1, 2};
  for (int mode = CP_YUV_ADD; mode <= CP_YUV_MULTIPLY; ++mode)
    for (double opacity : {0.0, .003, .17, .5, .63, 1.0})
      for (bool masked : {false, true}) {
        cp_yuv_config c = {f, mode, opacity};
        const int tolerance = bits != 32 && mode == CP_YUV_MULTIPLY && opacity * 256 != std::floor(opacity * 256) ? 1 : 0;
        const cp_yuv gv = {got[0].out(), got[1].out(), got[2].out()}, rv = {ref[0].out(), ref[1].out(), ref[2].out()};
        const int expected = bits == 32 && mode > CP_YUV_SUBTRACT && mode != CP_YUV_MULTIPLY ? CP_UNSUPPORTED : CP_OK;
        CHECK(k->process_yuv(&c, av, bv, masked ? &mv : nullptr, gv, band) == expected);
        CHECK(cp_process_yuv(&c, av, bv, masked ? &mv : nullptr, rv, band) == expected);
        for (int p = 0; p < 3; ++p) {
          char context[160];
          std::snprintf(context, sizeof(context),
                        "yuv bits=%d width=%d step=%d negative=%d mode=%d opacity=%g mask=%d plane=%d", bits, width,
                        step, negative, mode, opacity, masked, p);
          same(got[p], ref[p], context, tolerance);
          if (mode == CP_YUV_MULTIPLY) {
            const double center = p == 0 || bits == 32 ? 0 : double(1u << (bits - 1));
            const cp_plane_config plane{f, CP_GUIDED_MULTIPLY, opacity, center, 0, 0, 0, 0, CP_WEIGHT_CONTINUOUS};
            const auto mask_plane = m[p].in(), guide = b[0].in();
            CHECK(cp_process_plane(&plane, a[p].in(), b[p].in(), masked ? &mask_plane : nullptr, nullptr, &guide,
                                   ref[p].out(), band) == CP_OK);
            same(got[p], ref[p], "fused versus plane multiply", tolerance);
          }
        }
        // Snapshot guides, masks and all channels must precede writes within each pixel.
        for (int p = 0; p < 3; ++p) {
          std::copy(a[p].data.begin(), a[p].data.end(), got[p].data.begin());
          std::copy(a[p].data.begin(), a[p].data.end(), ref[p].data.begin());
        }
        CHECK(k->process_yuv(&c, {got[0].in(), got[1].in(), got[2].in()}, bv, masked ? &mv : nullptr, gv, band) ==
              expected);
        CHECK(cp_process_yuv(&c, {ref[0].in(), ref[1].in(), ref[2].in()}, bv, masked ? &mv : nullptr, rv, band) ==
              expected);
        for (int p = 0; p < 3; ++p)
          same(got[p], ref[p], "yuv in-place", tolerance);
        if (mode == CP_YUV_MULTIPLY) {
          // Source Y is also the guide, and masks may alias the destination.
          // All three channels must see the original guide before Y is stored.
          for (int p = 0; p < 3; ++p) {
            std::copy(m[p].data.begin(), m[p].data.end(), got[p].data.begin());
            std::copy(m[p].data.begin(), m[p].data.end(), ref[p].data.begin());
          }
          const cp_const_yuv gs{got[0].in(), got[1].in(), got[2].in()};
          const cp_const_yuv rs{ref[0].in(), ref[1].in(), ref[2].in()};
          CHECK(k->process_yuv(&c, av, gs, masked ? &gs : nullptr, gv, band) == CP_OK);
          CHECK(cp_process_yuv(&c, av, rs, masked ? &rs : nullptr, rv, band) == CP_OK);
          for (int p = 0; p < 3; ++p)
            same(got[p], ref[p], "multiply source and mask in-place", tolerance);
        }
      }
  cp_yuv_config c = {f, CP_YUV_ADD, .5};
  auto invalid = av;
  invalid.v.data = nullptr;
  const auto before = got;
  CHECK(k->process_yuv(&c, invalid, bv, &mv, {got[0].out(), got[1].out(), got[2].out()}, band) == CP_INVALID_ARGUMENT);
  for (int p = 0; p < 3; ++p)
    same(got[p], before[p], "invalid yuv");
}
template <class T>
void Sampling(const cp_kernels* k, int bits, int width, int step, bool negative) {
  const auto f = format(bits);
  Image<T> source(width, 5, step, negative), got(width + 3, 4, step, !negative), ref = got;
  source.populate(bits, 575, true);
  // Interior box sampling exercises the direct path, with nonzero row ranges,
  // shifted origins, opposite signed strides, vector tails and untouched padding.
  for (int origin : {0, 1})
    for (int vertical : {1, 2}) {
      const int output_width = (width - origin) / 2;
      if (output_width <= 0)
        continue;
      const cp_sampling interior{width, 5, 2, vertical, CP_CENTER, origin, 0};
      const cp_rows band{output_width, 4, 1, 1};
      CHECK(k->resample_mask(f, source.in(), got.out(), &interior, band) == CP_OK);
      CHECK(cp_resample_mask(f, source.in(), ref.out(), &interior, band) == CP_OK);
      same(got, ref, "interior box");
    }
  for (int sx : {1, 2, 4})
    for (int sy : {1, 2})
      for (int placement : {CP_CENTER, CP_MPEG2, CP_TOP_LEFT})
        for (int ox : {-3, 0, 1, width + 1})
          for (int oy : {-1, 0, 4}) {
            if (sy == 2 && sx != 2)
              continue;
            const cp_sampling s = {width, 5, sx, sy, placement, ox, oy};
            const cp_rows band = {width + 3, 4, 1, 2};
            CHECK(k->resample_mask(f, source.in(), got.out(), &s, band) == CP_OK);
            CHECK(cp_resample_mask(f, source.in(), ref.out(), &s, band) == CP_OK);
            same(got, ref, "sampling");
          }
  cp_sampling s = {width, 5, 2, 2, CP_MPEG2, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
  CHECK(k->resample_mask(f, source.in(), got.out(), &s, got.rows()) == CP_OK);
  CHECK(cp_resample_mask(f, source.in(), ref.out(), &s, ref.rows()) == CP_OK);
  same(got, ref, "extreme phase");
  const auto before = got;
  s.subsample_x = 3;
  CHECK(k->resample_mask(f, source.in(), got.out(), &s, got.rows()) == CP_INVALID_ARGUMENT);
  same(got, before, "invalid sampling");
}
static void FloatYuvEdges(const cp_kernels* k) {
  // Include neighbors of both luma transitions, signed zeros and finite HDR
  // values. Width 257 covers full vectors, tails and stepped staging tiles.
  const float over = 32.0f / 255;
  const float values[]{-0.0f,
                       0.0f,
                       -over,
                       std::nextafter(-over, 0.0f),
                       std::nextafter(-over, -1.0f),
                       1.0f,
                       std::nextafter(1.0f, 0.0f),
                       std::nextafter(1.0f, 2.0f),
                       1 + over,
                       std::nextafter(1 + over, 1.0f),
                       std::nextafter(1 + over, 2.0f),
                       -.5f,
                       .5f,
                       -4.0f,
                       4.0f,
                       std::numeric_limits<float>::max()};
  for (int step : {1, 4})
    for (bool negative : {false, true}) {
      Image<float> proto(257, 3, step, negative);
      std::array<Image<float>, 3> a{proto, proto, proto}, b = a, m = a;
      for (int p = 0; p < 3; ++p)
        for (int y = 0; y < 3; ++y)
          for (int x = 0; x < 257; ++x) {
            const int i = proto.origin + y * proto.pitch + x * step;
            a[p].data[i] = values[(x + p + y) % 16];
            b[p].data[i] = values[(x / 16 + p + y) % 16];
            m[p].data[i] = x % 5 == 0 ? 0.0f : float((x + p) % 7) / 6;
          }
      const cp_const_yuv av{a[0].in(), a[1].in(), a[2].in()}, bv{b[0].in(), b[1].in(), b[2].in()},
          mv{m[0].in(), m[1].in(), m[2].in()};
      const cp_rows band{257, 3, 1, 1};
      for (int op : {CP_YUV_ADD, CP_YUV_SUBTRACT})
        for (double opacity : {0.0, .003, .17, .5, .63, 1.0})
          for (bool masked : {false, true}) {
            const cp_yuv_config c{format(32), op, opacity};
            auto got = b, ref = b;
            const cp_const_yuv gs{got[0].in(), got[1].in(), got[2].in()}, rs{ref[0].in(), ref[1].in(), ref[2].in()};
            CHECK(k->process_yuv(&c, av, gs, masked ? &mv : nullptr, {got[0].out(), got[1].out(), got[2].out()},
                                 band) == CP_OK);
            CHECK(cp_process_yuv(&c, av, rs, masked ? &mv : nullptr, {ref[0].out(), ref[1].out(), ref[2].out()},
                                 band) == CP_OK);
            for (int p = 0; p < 3; ++p)
              same(got[p], ref[p], "float YUV edges and source in-place");
            got = m;
            ref = m;
            const cp_const_yuv gm{got[0].in(), got[1].in(), got[2].in()}, rm{ref[0].in(), ref[1].in(), ref[2].in()};
            CHECK(k->process_yuv(&c, av, bv, &gm, {got[0].out(), got[1].out(), got[2].out()}, band) == CP_OK);
            CHECK(cp_process_yuv(&c, av, bv, &rm, {ref[0].out(), ref[1].out(), ref[2].out()}, band) == CP_OK);
            for (int p = 0; p < 3; ++p)
              same(got[p], ref[p], "float YUV mask in-place");
          }
    }
}

static void FloatSpecials(const cp_kernels* k) {
  Image<float> a(65, 1, 1, false, 0), alpha = a, got = a, ref = a;
  const uint32_t values[] = {0x80000000, 0, 0x7fc12345, 0xff800000, 0x7f800000, 0x3f800000};
  for (size_t i = 0; i < a.data.size(); ++i) {
    std::memcpy(&a.data[i], &values[i % 6], 4);
    std::memcpy(&alpha.data[i], &values[(i + 2) % 6], 4);
  }
  const auto f = format(32);
  CHECK(k->clamp(f, a.in(), got.out(), a.rows(), -0.0, 1) == CP_OK);
  CHECK(cp_clamp(f, a.in(), ref.out(), a.rows(), -0.0, 1) == CP_OK);
  same(got, ref, "float clamp specials");
  // Exact and non-binary32 bounds must agree on zero signs and rounding.
  const double huge = std::numeric_limits<double>::max();
  for (const auto& bounds : {std::array<double, 2>{0.0, -0.0},
                             {-0.0, -0.0},
                             {-1, 1},
                             {std::nextafter(0.0, -1.0), std::nextafter(1.0, 2.0)},
                             {.1, .9},
                             {-huge, huge}}) {
    CHECK(k->clamp(f, a.in(), got.out(), a.rows(), bounds[0], bounds[1]) == CP_OK);
    CHECK(cp_clamp(f, a.in(), ref.out(), a.rows(), bounds[0], bounds[1]) == CP_OK);
    same(got, ref, "float clamp bound rounding");
  }
  const cp_const_rgb rgb = {a.in(), a.in(), a.in()};
  const double key[] = {0, 0, 0}, tol[] = {0, 0, 0};
  CHECK(k->color_key(f, rgb, alpha.in(), got.out(), a.rows(), key, tol) == CP_OK);
  CHECK(cp_color_key(f, rgb, alpha.in(), ref.out(), a.rows(), key, tol) == CP_OK);
  CHECK(std::memcmp(got.data.data(), ref.data.data(), got.data.size() * 4) == 0);
  Image<float> mask = a;
  std::fill(mask.data.begin(), mask.data.end(), 0.0f);
  cp_yuv_config yc = {f, CP_YUV_ADD, 1};
  auto ay = a, au = a, av = a, gy = a, gu = a, gv = a, ry = a, ru = a, rv = a;
  const cp_const_yuv base = {ay.in(), au.in(), av.in()}, m = {mask.in(), mask.in(), mask.in()};
  CHECK(k->process_yuv(&yc, base, base, &m, {gy.out(), gu.out(), gv.out()}, a.rows()) == CP_OK);
  CHECK(cp_process_yuv(&yc, base, base, &m, {ry.out(), ru.out(), rv.out()}, a.rows()) == CP_OK);
  CHECK(std::memcmp(gy.data.data(), ry.data.data(), gy.data.size() * 4) == 0);
  CHECK(std::memcmp(gu.data.data(), ru.data.data(), gu.data.size() * 4) == 0);
  CHECK(std::memcmp(gv.data.data(), rv.data.data(), gv.data.size() * 4) == 0);
  yc.operation = CP_YUV_MULTIPLY;
  CHECK(k->process_yuv(&yc, base, base, &m, {gy.out(), gu.out(), gv.out()}, a.rows()) == CP_OK);
  for (const auto* channel : {&gy, &gu, &gv})
    CHECK(std::memcmp(channel->data.data(), a.data.data(), a.data.size() * 4) == 0);
  // Regression: MSVC intrinsic folding must not leave -0 in neutralized UV.
  std::fill(ay.data.begin(), ay.data.end(), -0.5f);
  std::fill(au.data.begin(), au.data.end(), -0.25f);
  std::fill(av.data.begin(), av.data.end(), -0.25f);
  std::fill(alpha.data.begin(), alpha.data.end(), 1.0f);
  yc.operation = CP_YUV_SUBTRACT;
  CHECK(k->process_yuv(&yc, base, {alpha.in(), alpha.in(), alpha.in()}, nullptr, {gy.out(), gu.out(), gv.out()},
                       a.rows()) == CP_OK);
  for (const auto* channel : {&gy, &gu, &gv})
    for (float value : channel->data) {
      uint32_t bits;
      std::memcpy(&bits, &value, 4);
      CHECK(bits == 0);
    }
}
void FloatSteppedViews(const cp_kernels* k) {
  Image<float> red(67, 3, 2, false), green(67, 3, 3, true), blue(67, 3, 4, false), alpha(67, 3, 5, true);
  Image<float> got(67, 3, 7, true), ref = got;
  constexpr uint32_t values[]{0,           0x80000000u, 1,           0x80000001u, 0x3f000000u,
                              0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u, 0x7fc12345u};
  int channel = 0;
  for (auto* input : {&red, &green, &blue, &alpha}) {
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 67; ++x)
        std::memcpy(&input->data[input->origin + y * input->pitch + x * input->step], &values[(x + y + channel) % 10],
                    4);
    ++channel;
  }
  const auto f = format(32);
  CHECK(k->affine(f, red.in(), got.out(), got.rows(), -.7, .1) == CP_OK);
  CHECK(cp_affine(f, red.in(), ref.out(), ref.rows(), -.7, .1) == CP_OK);
  same(got, ref, "stepped affine mixed views");
  CHECK(k->clamp(f, green.in(), got.out(), got.rows(), -0.0, 1) == CP_OK);
  CHECK(cp_clamp(f, green.in(), ref.out(), ref.rows(), -0.0, 1) == CP_OK);
  same(got, ref, "stepped clamp signed zeros");
  const cp_const_rgb rgb{red.in(), green.in(), blue.in()};
  const double key[]{.1, .5, 0}, tolerance[]{.4, .5, 1};
  CHECK(k->color_key(f, rgb, alpha.in(), got.out(), got.rows(), key, tolerance) == CP_OK);
  CHECK(cp_color_key(f, rgb, alpha.in(), ref.out(), ref.rows(), key, tolerance) == CP_OK);
  // Color key must preserve untouched alpha bits, including NaN payloads.
  CHECK(std::memcmp(got.data.data(), ref.data.data(), got.data.size() * 4) == 0);
  auto inplace = red, expected = red;
  CHECK(k->affine(f, inplace.in(), inplace.out(), inplace.rows(), -.7, .1) == CP_OK);
  CHECK(cp_affine(f, expected.in(), expected.out(), expected.rows(), -.7, .1) == CP_OK);
  same(inplace, expected, "stepped affine in-place");

  const ptrdiff_t huge = (std::numeric_limits<ptrdiff_t>::max() - 4) & ~ptrdiff_t(3);
  std::array<float, 129> input, actual{}, reference{};
  input.fill(.25f);
  const cp_const_plane iv{input.data(), 4, huge};
  const cp_plane ov{actual.data(), 4, huge}, rv{reference.data(), 4, huge};
  const cp_rows single{1, 129, 0, 129};
  CHECK(k->affine(f, iv, ov, single, 2, .1) == CP_OK);
  CHECK(cp_affine(f, iv, rv, single, 2, .1) == CP_OK);
  CHECK(actual == reference);
  CHECK(k->color_key(f, {iv, iv, iv}, iv, ov, single, key, tolerance) == CP_OK);
  CHECK(cp_color_key(f, {iv, iv, iv}, iv, rv, single, key, tolerance) == CP_OK);
  CHECK(actual == reference);
}

void FloatKeyThresholds(const cp_kernels* k) {
  Image<float> values(4097, 1, 1, false), alpha = values, got = values, ref = values;
  std::mt19937 rng(7771);
  for (size_t i = 0; i < values.data.size(); ++i) {
    const uint32_t pixel = rng(), payload = 0x7fc00000u | (rng() & 0x3fffffu);
    std::memcpy(&values.data[i], &pixel, 4);
    std::memcpy(&alpha.data[i], &payload, 4);
  }
  const double huge = std::numeric_limits<double>::max(), maxfloat = std::numeric_limits<float>::max();
  const double cases[][2]{{0, 0},
                          {-.0, -.0},
                          {.1, 0},
                          {double(.1f), 0},
                          {.1, .2},
                          {.5, std::nextafter(.25, 0.0)},
                          {.5, std::nextafter(.25, 1.0)},
                          {huge, huge},
                          {huge, std::nextafter(huge, 0.0)},
                          {-huge, huge},
                          {1e100, 1e100},
                          {-1e100, 1e100},
                          {1e-300, 0},
                          {1e-300, 1e-300},
                          {maxfloat, 0},
                          {0, maxfloat}};
  for (const auto& bounds : cases) {
    size_t i = 0;
    for (double candidate : {bounds[0], bounds[0] - bounds[1], bounds[0] + bounds[1]})
      if (std::isfinite(candidate) && std::abs(candidate) <= maxfloat) {
        const float center = static_cast<float>(candidate);
        values.data[i++] = center;
        values.data[i++] = std::nextafter(center, -std::numeric_limits<float>::infinity());
        values.data[i++] = std::nextafter(center, std::numeric_limits<float>::infinity());
      }
    const double key[]{bounds[0], bounds[0], bounds[0]}, tolerance[]{bounds[1], bounds[1], bounds[1]};
    const cp_const_rgb rgb{values.in(), values.in(), values.in()};
    CHECK(k->color_key(format(32), rgb, alpha.in(), got.out(), values.rows(), key, tolerance) == CP_OK);
    CHECK(cp_color_key(format(32), rgb, alpha.in(), ref.out(), values.rows(), key, tolerance) == CP_OK);
    CHECK(std::memcmp(got.data.data(), ref.data.data(), got.data.size() * 4) == 0);
  }
  for (bool negative : {false, true})
    for (bool bgra : {false, true}) {
      Image<float> packed(67, 3, 4, negative, 0);
      packed.populate(32, 913, true);
      for (size_t i = 3; i < packed.data.size(); i += 4) {
        const uint32_t payload = 0x7fc00000u | static_cast<uint32_t>(i);
        std::memcpy(&packed.data[i], &payload, 4);
      }
      auto expected = packed;
      const auto apply = [&](Image<float>& frame, const cp_kernels* table) {
        auto red = frame.in(), green = red, blue = red, alpha_view = red;
        red.data = static_cast<const float*>(red.data) + (bgra ? 2 : 0);
        green.data = static_cast<const float*>(green.data) + 1;
        blue.data = static_cast<const float*>(blue.data) + (bgra ? 0 : 2);
        alpha_view.data = static_cast<const float*>(alpha_view.data) + 3;
        auto out = frame.out();
        out.data = static_cast<float*>(out.data) + 3;
        const double key[]{.5, .5, .5}, tolerance[]{.25, .25, .25};
        CHECK(table->color_key(format(32), {red, green, blue}, alpha_view, out, frame.rows(), key, tolerance) == CP_OK);
      };
      apply(packed, k);
      apply(expected, cp_get_kernels(0));
      CHECK(std::memcmp(packed.data.data(), expected.data.data(), packed.data.size() * 4) == 0);
    }
}

void FloatMultiplySpecials(const cp_kernels* k) {
  constexpr uint32_t patterns[]{0,           0x80000000u, 1,           0x80000001u, 0x007fffffu,
                                0x00800000u, 0x3f000000u, 0xbf000000u, 0x3f800000u, 0xbf800000u,
                                0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc12345u};
  constexpr int count = int(sizeof(patterns) / sizeof(patterns[0]));
  Image<float> a(count * count, 1, 1, false), guide = a, y = a, u = a, v = a, ry = a, ru = a, rv = a;
  for (int i = 0; i < count * count; ++i) {
    std::memcpy(&a.data[i], &patterns[i % count], 4);
    std::memcpy(&guide.data[i], &patterns[i / count], 4);
  }
  const cp_yuv_config c{format(32), CP_YUV_MULTIPLY, 1};
  const cp_const_yuv base{a.in(), a.in(), a.in()}, source{guide.in(), guide.in(), guide.in()};
  CHECK(k->process_yuv(&c, base, source, nullptr, {y.out(), u.out(), v.out()}, a.rows()) == CP_OK);
  CHECK(cp_process_yuv(&c, base, source, nullptr, {ry.out(), ru.out(), rv.out()}, a.rows()) == CP_OK);
  same(y, ry, "float multiply special Y");
  same(u, ru, "float multiply special U");
  same(v, rv, "float multiply special V");
}

// Full U8 base/guide cross product and wide U16 inputs catch rare rounding
// boundaries that short layout tests cannot. Compare against the original
// single-plane scalar definition, including adjacent binary64 opacities.
template <class T>
void MultiplyRounding(const cp_kernels* k, int bits) {
  const int width = 65539;
  const unsigned maximum = (1u << bits) - 1;
  Image<T> a(width, 1, 1, false), guide = a, mask = a, got = a, ref = a;
  std::mt19937 rng(9923 + bits);
  for (int x = 0; x < width; ++x) {
    a.data[x] = T(bits == 8 ? x & 255 : rng() & maximum);
    guide.data[x] = T(bits == 8 ? x >> 8 : rng() & maximum);
    mask.data[x] = T(rng() & maximum);
  }
  if (bits == 16) {
    // n*(M*M+(guide-M)*mask) mod M*M = (M*M +/- 1)/2.
    // At opacity 1, these chroma results are only 1/(2*M*M) from a half.
    constexpr uint16_t cases[][3]{
        {23296, 9556, 29026},  {41887, 45902, 43744}, {43842, 7499, 46033},  {48732, 17972, 12509},
        {35341, 21128, 62942}, {19509, 15212, 64841}, {19845, 2089, 40436},  {46809, 48998, 47564},
        {11512, 43612, 12751}, {32571, 42581, 4274},  {15946, 34427, 9188},  {49651, 36028, 21023},
        {32776, 57079, 31744}, {10341, 39113, 29198}, {32992, 49663, 604},   {36669, 41968, 65239},
        {61887, 4279, 29858},  {4420, 19294, 1784},   {28239, 36649, 36752}, {51006, 26269, 10936},
        {29286, 14179, 64739}, {49522, 23486, 253},   {51346, 28127, 49838}, {60685, 33737, 45418},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      a.data[i] = T(cases[i][0]);
      guide.data[i] = T(cases[i][1]);
      mask.data[i] = T(cases[i][2]);
    }
  }
  std::vector<double> opacities{0.0, std::nextafter(.5, 0.0),  .5, std::nextafter(.5, 1.0), .625,
                                .17, .37, std::nextafter(1.0, 0.0), 1.0};
  if (bits == 8)
    for (int level = 1; level < 256; ++level)
      opacities.push_back(level / 256.0);
  else
    for (int level : {1, 3, 5, 17, 63, 85, 127, 129, 170, 192, 221, 254, 255})
      opacities.push_back(level / 256.0);
  for (double opacity : opacities) {
    const cp_yuv_config config{format(bits), CP_YUV_MULTIPLY, opacity};
    for (bool masked : {false, true}) {
      got = a;
      auto u = a, v = a;
      const cp_const_yuv masks{mask.in(), mask.in(), mask.in()};
      CHECK(k->process_yuv(&config, {got.in(), u.in(), v.in()}, {guide.in(), guide.in(), guide.in()},
                           masked ? &masks : nullptr, {got.out(), u.out(), v.out()}, a.rows()) == CP_OK);
      for (int channel = 0; channel < 3; ++channel) {
        const cp_plane_config plane{
            format(bits),        CP_GUIDED_MULTIPLY, opacity, channel ? double(1u << (bits - 1)) : 0, 0, 0, 0, 0,
            CP_WEIGHT_CONTINUOUS};
        const auto mv = mask.in(), gv = guide.in();
        CHECK(cp_process_plane(&plane, a.in(), guide.in(), masked ? &mv : nullptr, nullptr, &gv, ref.out(), a.rows()) ==
              CP_OK);
        same(channel == 0 ? got : channel == 1 ? u : v, ref, "multiply rounding boundary", opacity * 256 != std::floor(opacity * 256) ? 1 : 0);
      }
    }
  }
}

template <class T>
void IntegerKeyThresholds(const cp_kernels* k, int bits) {
  const int width = 1 << bits;
  const double maximum = width - 1;
  Image<T> colors(width, 1, 1, false), alpha = colors, got = colors, ref = colors;
  for (int x = 0; x < width; ++x) {
    colors.data[x] = T(x);
    alpha.data[x] = T((x * 137 + 23) & (width - 1));
  }
  const double huge = std::numeric_limits<double>::max();
  const double cases[][2]{{0, 0},
                          {maximum, 0},
                          {.1, 0},
                          {maximum / 2, .5},
                          {maximum / 2, std::nextafter(.5, 0.)},
                          {maximum / 2, std::nextafter(.5, 1.)},
                          {huge, huge},
                          {-huge, huge},
                          {huge, std::nextafter(huge, 0.)},
                          {1e100, 1e100},
                          {-1e100, 1e100},
                          {0, maximum}};
  for (const auto& values : cases) {
    const double key[3]{values[0], values[0], values[0]}, tolerance[3]{values[1], values[1], values[1]};
    const cp_const_rgb rgb{colors.in(), colors.in(), colors.in()};
    CHECK(cp_color_key(format(bits), rgb, alpha.in(), ref.out(), colors.rows(), key, tolerance) == CP_OK);
    CHECK(k->color_key(format(bits), rgb, alpha.in(), got.out(), colors.rows(), key, tolerance) == CP_OK);
    same(got, ref, "integer key exact thresholds");
    got = alpha;
    CHECK(k->color_key(format(bits), rgb, got.in(), got.out(), colors.rows(), key, tolerance) == CP_OK);
    same(got, ref, "integer key alpha alias");
  }
}

template <class T>
void GuidedLimits(const cp_kernels* k, int bits) {
  const auto f = format(bits);
  const double maximum = bits == 32 ? 1 : (1u << bits) - 1;
  Image<T> a(67, 3, 1, true), guide = a, mask = a, got = a, ref = a;
  a.populate(bits, 137);
  guide.populate(bits, 131);
  for (size_t i = 0; i < mask.data.size(); ++i) {
    mask.data[i] = T(i % 3 == 0 ? 0 : i % 3 == 1 ? maximum : maximum / 3);
    if (i % 7 == 0)
      a.data[i] = std::numeric_limits<T>::max();
  }
  const double huge = std::numeric_limits<double>::max();
  for (double neutral : {-huge, -0.0, .1, maximum / 2, huge})
    for (double opacity : {.17, 1.0})
      for (bool masked : {false, true}) {
        const cp_plane_config c{f, CP_GUIDED_MULTIPLY, opacity, neutral, 0, 0, 0, 0, CP_WEIGHT_CONTINUOUS};
        const auto verify = [&](const char* label) {
          const bool approximate = bits != 32 && opacity > 0 && opacity < 1 &&
                                   neutral >= 0 && neutral <= maximum;
          same(got, ref, label, approximate ? 1 : 0);
          if (approximate) {
            for (int y = 0; y < a.h; ++y)
              for (int x = 0; x < a.w; ++x) {
                const int i = a.origin + y * a.pitch + x * a.step;
                // Approximation never changes zero-mask copies or the
                // noncanonical-input fallback, including aliased inputs.
                if (a.data[i] > maximum || guide.data[i] > maximum ||
                    (masked && (mask.data[i] == 0 || mask.data[i] > maximum)))
                  CHECK(got.data[i] == ref.data[i]);
              }
          }
        };
        const auto mv = mask.in(), gv = guide.in();
        CHECK(k->process_plane(&c, a.in(), guide.in(), masked ? &mv : nullptr, nullptr, &gv, got.out(), a.rows()) ==
              CP_OK);
        CHECK(cp_process_plane(&c, a.in(), guide.in(), masked ? &mv : nullptr, nullptr, &gv, ref.out(), a.rows()) ==
              CP_OK);
        verify("guided extreme neutral and endpoint codes");
        // The guide and mask may be exact aliases of the destination.
        for (bool alias_mask : {false, true}) {
          got = ref = alias_mask ? mask : guide;
          const auto gm = alias_mask ? got.in() : mv, rm = alias_mask ? ref.in() : mv;
          const auto gg = alias_mask ? gv : got.in(), rg = alias_mask ? gv : ref.in();
          CHECK(k->process_plane(&c, a.in(), guide.in(), masked ? &gm : nullptr, nullptr, &gg, got.out(), a.rows()) ==
                CP_OK);
          CHECK(cp_process_plane(&c, a.in(), guide.in(), masked ? &rm : nullptr, nullptr, &rg, ref.out(), a.rows()) ==
                CP_OK);
          verify("guided guide/mask in-place");
        }
      }
}

template <class T>
void IntegerClampLimits(const cp_kernels* k, int bits) {
  const int width = 1 << bits;
  const double maximum = width - 1, huge = std::numeric_limits<double>::max();
  Image<T> source(width, 1, 1, false), got = source, ref = source;
  for (int x = 0; x < width; ++x)
    source.data[x] = T(x);
  const double limits[][2]{{-huge, huge},
                           {-huge, -1},
                           {maximum + 1, huge},
                           {0, maximum},
                           {.5, .5},
                           {std::nextafter(.5, 0.), std::nextafter(.5, 1.)},
                           {1.5, maximum - 1.5},
                           {std::nextafter(1.5, 0.), std::nextafter(maximum - 1.5, maximum)},
                           {maximum / 4 + .49, maximum * .75 + .51}};
  for (const auto& range : limits) {
    CHECK(cp_clamp(format(bits), source.in(), ref.out(), source.rows(), range[0], range[1]) == CP_OK);
    CHECK(k->clamp(format(bits), source.in(), got.out(), source.rows(), range[0], range[1]) == CP_OK);
    same(got, ref, "integer clamp quantized limits");
    got = source;
    CHECK(k->clamp(format(bits), got.in(), got.out(), source.rows(), range[0], range[1]) == CP_OK);
    same(got, ref, "integer clamp in-place limits");
  }
  // Include all storage codes, even values beyond a sub-16-bit format's max.
  const int codes = 1 << (sizeof(T) * 8);
  Image<T> input(codes + 1, 1, 1, false), inverted = input, expected = input;
  for (int x = 0; x <= codes; ++x)
    input.data[x] = T(x);
  CHECK(k->affine(format(bits), input.in(), inverted.out(), input.rows(), -1, maximum) == CP_OK);
  CHECK(cp_affine(format(bits), input.in(), expected.out(), input.rows(), -1, maximum) == CP_OK);
  same(inverted, expected, "integer inversion storage codes");
  inverted = input;
  CHECK(k->affine(format(bits), inverted.in(), inverted.out(), input.rows(), -1, maximum) == CP_OK);
  same(inverted, expected, "integer inversion in-place storage codes");
}

template <class T>
void IntegerYuvEdges(const cp_kernels* k, int bits) {
  const int width = bits == 8 ? 65536 : 4097;
  const unsigned maximum = (1u << bits) - 1, over = (maximum + 1) / 8;
  const Image<T> empty(width, 1, 1, false);
  std::array<Image<T>, 3> a{empty, empty, empty}, b = a, m = a, got = a, ref = a;
  for (int p = 0; p < 3; ++p) {
    a[p].populate(bits, 991 + p);
    b[p].populate(bits, 871 + p);
    m[p].populate(bits, 771 + p);
  }
  // Cover every U8 chroma/weight pair, plus alternating zero/full endpoints.
  for (int x = 0; x < width; ++x) {
    if (bits == 8) {
      b[1].data[x] = T(x & 255);
      m[1].data[x] = T(x >> 8);
      b[2].data[x] = T(maximum - (x & 255));
      m[2].data[x] = T(maximum - (x >> 8));
    }
    b[0].data[x] = T(x % (2 * over + 3));
    if (x % 7 == 0)
      m[0].data[x] = T(x % 2 ? maximum : 0);
  }
  a[1].data[0] = 0;
  b[1].data[0] = T(maximum);
  m[1].data[0] = T(maximum); // Difference's largest positive delta.
  for (int operation :
       {CP_YUV_ADD, CP_YUV_SUBTRACT, CP_YUV_SOFT_LIGHT, CP_YUV_HARD_LIGHT, CP_YUV_DIFFERENCE, CP_YUV_EXCLUSION}) {
    for (int x = 0; x < width; ++x)
      a[0].data[x] = T(operation == CP_YUV_ADD || (operation > CP_YUV_SUBTRACT && x % 2) ? maximum - x % 3 : x % 3);
    if (operation > CP_YUV_SUBTRACT)
      for (int x = 0; x < width; ++x) {
        const unsigned offset = unsigned(x) % (over + 3);
        b[0].data[x] = T(x % 4 == 0   ? 0
                         : x % 4 == 1 ? maximum
                         : x % 4 == 2 ? (maximum + 1) / 2 - offset
                                      : (maximum + 1) / 2 + offset);
      }
    for (double opacity : {0., std::nextafter(.5, 0.), .625, 1.})
      for (bool masked : {false, true}) {
        const cp_yuv_config config{format(bits), operation, opacity};
        const cp_const_yuv masks{m[0].in(), m[1].in(), m[2].in()}, base{a[0].in(), a[1].in(), a[2].in()},
            source{b[0].in(), b[1].in(), b[2].in()};
        ref = a;
        got = a;
        CHECK(cp_process_yuv(&config, base, source, masked ? &masks : nullptr,
                             {ref[0].out(), ref[1].out(), ref[2].out()}, empty.rows()) == CP_OK);
        CHECK(k->process_yuv(&config, {got[0].in(), got[1].in(), got[2].in()}, source, masked ? &masks : nullptr,
                             {got[0].out(), got[1].out(), got[2].out()}, empty.rows()) == CP_OK);
        for (int p = 0; p < 3; ++p)
          same(got[p], ref[p], "integer YUV desaturation and delta boundaries");
      }
  }
}

int main() {
  std::vector<int64_t> targets = {0};
  for (int64_t remaining = cp_supported_targets(); remaining; remaining &= remaining - 1)
    targets.push_back(remaining & -remaining);
  for (auto target : targets) {
    const auto* k = cp_get_kernels(target);
    CHECK(k && k->process_yuv && k->resample_mask && k->affine && k->clamp && k->rgb_luma && k->color_key);
    std::printf("extended target 0x%llx\n", static_cast<unsigned long long>(target));
    std::fflush(stdout);
    for (int width : {1, 3, 7, 16, 31, 65, 129})
      for (int step : {1, 2, 3, 4})
        for (bool negative : {false, true}) {
          Utilities<uint8_t>(k, 8, width, step, negative);
          Yuv<uint8_t>(k, 8, width, step, negative);
          Sampling<uint8_t>(k, 8, width, step, negative);
          for (int bits = 9; bits <= 16; ++bits) {
            Utilities<uint16_t>(k, bits, width, step, negative);
            Yuv<uint16_t>(k, bits, width, step, negative);
            Sampling<uint16_t>(k, bits, width, step, negative);
          }
          Utilities<float>(k, 32, width, step, negative);
          Yuv<float>(k, 32, width, step, negative);
          Sampling<float>(k, 32, width, step, negative);
        }
    for (int width : {255, 256, 257, 513})
      for (bool negative : {false, true}) {
        Yuv<uint8_t>(k, 8, width, 4, negative);
        Yuv<uint16_t>(k, 16, width, 4, negative);
        Yuv<float>(k, 32, width, 4, negative);
      }
    MultiplyRounding<uint8_t>(k, 8);
    for (int bits = 9; bits <= 16; ++bits)
      MultiplyRounding<uint16_t>(k, bits);
    IntegerYuvEdges<uint8_t>(k, 8);
    for (int bits = 9; bits <= 16; ++bits)
      IntegerYuvEdges<uint16_t>(k, bits);
    GuidedLimits<uint8_t>(k, 8);
    for (int bits = 9; bits <= 16; ++bits)
      GuidedLimits<uint16_t>(k, bits);
    GuidedLimits<float>(k, 32);
    IntegerClampLimits<uint8_t>(k, 8);
    for (int bits = 9; bits <= 16; ++bits)
      IntegerClampLimits<uint16_t>(k, bits);
    IntegerKeyThresholds<uint8_t>(k, 8);
    for (int bits = 9; bits <= 16; ++bits)
      IntegerKeyThresholds<uint16_t>(k, bits);
    FloatKeyThresholds(k);
    FloatSteppedViews(k);
    FloatMultiplySpecials(k);
    FloatYuvEdges(k);
    FloatSpecials(k);
  }
  std::thread a([] { Sampling<float>(cp_get_kernels(0), 32, 129, 4, true); });
  std::thread b([] { Sampling<float>(cp_get_kernels(CP_TARGET_NATIVE), 32, 129, 4, true); });
  a.join();
  b.join();
  std::puts("extended dispatch tests passed");
}
