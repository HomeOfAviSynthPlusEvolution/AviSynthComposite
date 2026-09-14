// SPDX-License-Identifier: GPL-2.0-or-later
#include <composite/composite.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#define CHECK(expr)                                                                                                    \
  do {                                                                                                                 \
    if (!(expr)) {                                                                                                     \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr);                                                          \
      std::abort();                                                                                                    \
    }                                                                                                                  \
  } while (0)

template <class T>
void CheckLayout(const cp_kernels* table, int bits, int width, int step, bool negative) {
  const int stride = width * step + 19, height = 5;
  const size_t size = static_cast<size_t>(stride) * height;
  const cp_format f = {sizeof(T) == 1 ? CP_U8 : sizeof(T) == 2 ? CP_U16 : CP_F32, bits};
  const uint32_t max = bits == 32 ? 65535 : (1u << bits) - 1;
  std::vector<T> a(size), b(size), mask(size), actual(size, T(23)), expected = actual;
  std::mt19937 rng(432 + bits + width + step);
  for (size_t i = 0; i < size; ++i) {
    a[i] = T(rng() & max);
    b[i] = T(rng() & max);
    mask[i] = T(rng() & max);
  }
  const int origin = negative ? stride * (height - 1) + 1 : 1;
  const ptrdiff_t pitch = (negative ? -stride : stride) * ptrdiff_t(sizeof(T));
  const ptrdiff_t spacing = step * sizeof(T);
  const cp_const_plane pa = {a.data() + origin, pitch, spacing}, pb = {b.data() + origin, pitch, spacing},
                       pm = {mask.data() + origin, pitch, spacing};
  cp_plane pd = {actual.data() + origin, pitch, spacing}, pe = {expected.data() + origin, pitch, spacing};
  const cp_rows rows = {width, height, 1, 3};
  const auto same = [&] {
    CHECK(std::memcmp(actual.data(), expected.data(), size * sizeof(T)) == 0);
  };
  CHECK(table->copy(f, pa, pd, rows) == cp_copy(f, pa, pe, rows));
  same();
  CHECK(table->copy(f, {pd.data, pd.stride, pd.step}, pd, rows) == CP_OK);
  same();
  for (double v : {-13.0, 0.0, -0.0, 0.5, 193.5, 65535.0, 99999.0}) {
    CHECK(table->fill(f, pd, rows, v) == cp_fill(f, pe, rows, v));
    same();
  }
  if (bits != 32)
    for (int opacity : {0, 1, 37, 127, 128, 129, 255, 256})
      for (bool masked : {false, true}) {
        for (size_t i = 0; i < size; ++i)
          mask[i] = T(i % 7 == 0 ? max : i % 7 == 1 ? 0 : rng() & max);
        CHECK(table->blend_compat(f, pa, pb, masked ? &pm : nullptr, pd, rows, opacity) == CP_OK);
        CHECK(cp_blend_compat(f, pa, pb, masked ? &pm : nullptr, pe, rows, opacity) == CP_OK);
        same();
        // Exact in-place base, including signed stride and scalar layout fallback.
        actual = a;
        expected = a;
        CHECK(table->blend_compat(f, {pd.data, pd.stride, pd.step}, pb, masked ? &pm : nullptr, pd, rows, opacity) ==
              CP_OK);
        CHECK(cp_blend_compat(f, {pe.data, pe.stride, pe.step}, pb, masked ? &pm : nullptr, pe, rows, opacity) ==
              CP_OK);
        same();
      }
  if (bits != 32) {
    const cp_plane_config blend{f, CP_MIX, 20481.0 / 32768, 0, 0, 0, 0, 0, CP_WEIGHT_CONTINUOUS};
    CHECK(table->process_plane(&blend, pa, pb, nullptr, nullptr, nullptr, pd, rows) == CP_OK);
    CHECK(cp_process_plane(&blend, pa, pb, nullptr, nullptr, nullptr, pe, rows) == CP_OK);
    same();
    actual = a;
    expected = a;
    CHECK(table->process_plane(&blend, {pd.data, pitch, spacing}, pb, nullptr, nullptr, nullptr, pd, rows) == CP_OK);
    CHECK(cp_process_plane(&blend, {pe.data, pitch, spacing}, pb, nullptr, nullptr, nullptr, pe, rows) == CP_OK);
    same();
  }
  const auto saved = actual;
  cp_const_plane invalid = pa;
  invalid.stride = 0;
  CHECK(table->copy(f, invalid, pd, rows) == CP_INVALID_ARGUMENT);
  CHECK(table->fill(f, pd, rows, std::numeric_limits<double>::quiet_NaN()) == CP_INVALID_ARGUMENT);
  CHECK(table->blend_compat(f, pa, pb, nullptr, pd, rows, -1) == CP_INVALID_ARGUMENT);
  CHECK(actual == saved);
}

static void Exhaustive(const cp_kernels* table) {
  constexpr int width = 65536;
  std::vector<uint8_t> a(width), b(width), m(width), d(width), ref(width);
  for (int i = 0; i < width; ++i) {
    a[i] = uint8_t(i >> 8);
    b[i] = uint8_t(i);
  }
  cp_format f = {CP_U8, 8};
  cp_const_plane pa = {a.data(), width, 1}, pb = {b.data(), width, 1}, pm = {m.data(), width, 1};
  cp_rows rows = {width, 1, 0, 1};
  for (int opacity : {0, 1, 127, 128, 255, 256})
    for (int mask : {-1, 0, 1, 127, 128, 254, 255}) {
      std::fill(m.begin(), m.end(), uint8_t(mask));
      CHECK(table->blend_compat(f, pa, pb, mask < 0 ? nullptr : &pm, {d.data(), width, 1}, rows, opacity) == CP_OK);
      CHECK(cp_blend_compat(f, pa, pb, mask < 0 ? nullptr : &pm, {ref.data(), width, 1}, rows, opacity) == CP_OK);
      CHECK(d == ref);
    }
}
template <class T>
static void Arithmetic(const cp_kernels* table, int bits, int width, int step, bool negative) {
  const int stride = width * step + 7, height = 4, origin = negative ? 1 + stride * (height - 1) : 1;
  const size_t size = static_cast<size_t>(stride) * height;
  const double max = bits == 32 ? 1.0 : (1u << bits) - 1;
  const cp_format f = {sizeof(T) == 1 ? CP_U8 : sizeof(T) == 2 ? CP_U16 : CP_F32, bits};
  std::vector<T> a(size), b(size), m(size), ga(size), gb(size), actual(size, T(19)), expected = actual;
  std::mt19937 rng(718 + bits + width);
  for (size_t i = 0; i < size; ++i) {
    const auto sample = [&]() -> T {
      return bits == 32 ? T((int(rng() % 131073) - 32768) / 65536.0) : T(rng() & uint32_t(max));
    };
    a[i] = sample();
    b[i] = sample();
    ga[i] = sample();
    gb[i] = i % 5 == 0 ? ga[i] : sample();
    m[i] = T(i % 7 == 0 ? max : i % 7 == 1 ? 0 : max * (rng() % 65536) / 65535);
  }
  const ptrdiff_t pitch = (negative ? -stride : stride) * ptrdiff_t(sizeof(T)), spacing = step * sizeof(T);
  const cp_const_plane pa = {a.data() + origin, pitch, spacing}, pb = {b.data() + origin, pitch, spacing},
                       pm = {m.data() + origin, pitch, spacing};
  const cp_const_plane pag = {ga.data() + origin, pitch, spacing}, pbg = {gb.data() + origin, pitch, spacing};
  const cp_plane pd = {actual.data() + origin, pitch, spacing}, pe = {expected.data() + origin, pitch, spacing};
  const cp_rows rows = {width, height, 1, 2};
  for (int op = CP_MIX; op <= CP_DIFFERENCE; ++op)
    for (int rule : {CP_WEIGHT_CONTINUOUS, CP_WEIGHT_CODE})
      for (double opacity : {0.0, 0.17, 0.5, 1.0})
        for (int inclusive : {0, 1})
          for (bool masked : {false, true}) {
            cp_plane_config c = {f,   op,        opacity, bits == 32 ? 0.0 : double(1 << (bits - 1)), max, max / 2,
                                 0.0, inclusive, rule};
            if (inclusive)
              c.threshold = bits == 32 ? double(std::numeric_limits<float>::epsilon() / 2) : 7;
            CHECK(table->process_plane(&c, pa, pb, masked ? &pm : nullptr, &pag, &pbg, pd, rows) == CP_OK);
            CHECK(cp_process_plane(&c, pa, pb, masked ? &pm : nullptr, &pag, &pbg, pe, rows) == CP_OK);
            if (bits != 32 && (op == CP_MIX || op == CP_INVERT_MIX || op == CP_PRODUCT || op == CP_ADD || op == CP_SUBTRACT || (op == CP_GUIDED_MULTIPLY && opacity < 1)) &&
                rule == CP_WEIGHT_CONTINUOUS && opacity != 0) {
              for (int y = rows.first; y < rows.first + rows.count; ++y)
                for (int x = 0; x < width; ++x) {
                  const int i = origin + (negative ? -stride : stride) * y + x * step;
                  const bool exact = (masked && m[i] == 0) ||
                      (op == CP_MIX && opacity == 1 && (!masked || m[i] == max));
                  CHECK(std::abs(int(actual[i]) - int(expected[i])) <= (exact ? 0 : 1));
                  expected[i] = actual[i]; // Remaining bytes still require exact canary equality.
                }
            }
            if (bits == 32 && (op == CP_MIX || op == CP_PRODUCT) && masked && step == 1 && opacity > 0) {
              for (int y = rows.first; y < rows.first + rows.count; ++y)
                for (int x = 0; x < width; ++x) {
                  const int i = origin + (negative ? -stride : stride) * y + x;
                  if (m[i] != 0 && !(opacity == 1 && m[i] == 1) && std::isfinite(double(actual[i])) && std::isfinite(double(expected[i]))) {
                    const double w = opacity * m[i];
                    const double scale = op == CP_MIX ? std::abs(double(a[i])) * (1-w) + std::abs(double(b[i])) * w
                        : std::abs(double(a[i])) * ((1-w) + std::abs(double(b[i])) * w);
                    CHECK(std::abs(double(actual[i]) - double(expected[i])) <= 16 * std::numeric_limits<float>::epsilon() * std::max(1.0, scale));
                    expected[i] = actual[i];
                  }
                }
            }
            if (std::memcmp(actual.data(), expected.data(), size * sizeof(T)) != 0) {
              for (size_t i = 0; i < size; ++i)
                if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0)
                  std::fprintf(stderr,
                               "bits=%d width=%d op=%d rule=%d opacity=%g masked=%d at=%zu got=%.17g expected=%.17g\n",
                               bits, width, op, rule, opacity, masked, i, double(actual[i]), double(expected[i]));
              CHECK(false);
            }
          }
  CHECK(table->process_plane(nullptr, pa, pb, nullptr, nullptr, nullptr, pd, rows) == CP_INVALID_ARGUMENT);
}

static void FloatEndpoints(const cp_kernels* table) {
  constexpr int n = 65;
  float a[n], b[n], m[n], actual[n], expected[n];
  for (int i = 0; i < n; ++i) {
    const uint32_t av = i % 2 ? 0x7fc12345 : 0x80000000, bv = i % 2 ? 0x7fc54321 : 0;
    std::memcpy(a + i, &av, 4);
    std::memcpy(b + i, &bv, 4);
    m[i] = i % 3 == 0 ? 0.0f : 1.0f;
  }
  cp_plane_config c = {{CP_F32, 32}, CP_MIX, 1, 0, 0, 0, 0, 0, CP_WEIGHT_CONTINUOUS};
  cp_const_plane pa = {a, n * 4, 4}, pb = {b, n * 4, 4}, pm = {m, n * 4, 4};
  for (double opacity : {0.0, 1.0}) {
    c.opacity = opacity;
    CHECK(table->process_plane(&c, pa, pb, &pm, nullptr, nullptr, {actual, n * 4, 4}, {n, 1, 0, 1}) == CP_OK);
    CHECK(cp_process_plane(&c, pa, pb, &pm, nullptr, nullptr, {expected, n * 4, 4}, {n, 1, 0, 1}) == CP_OK);
    CHECK(std::memcmp(actual, expected, sizeof(actual)) == 0);
  }
}
static void FloatBlendSpecials(const cp_kernels* table) {
  // Finite samples: signed zero, one 16-bit-scale step, ordinary colors and HDR100.
  // NaN/Inf are separate propagation cases, not brightness requirements.
  constexpr int n = 67;
  const uint32_t patterns[] = {0,          0x80000000, 0x37800000, 0xb7800000, 0x3f000000, 0xbf800000,
                               0x42c80000, 0xc2c80000, 0x7f800000, 0xff800000, 0x7fc12345};
  float a[n], b[n], m[n], actual[n], expected[n];
  for (int i = 0; i < n; ++i) {
    std::memcpy(a + i, &patterns[i % 11], 4);
    std::memcpy(b + i, &patterns[(i / 3 + 4) % 11], 4);
    m[i] = i % 5 == 0 ? 0 : i % 5 == 1 ? 1 : i % 5 == 2 ? .3f : i % 5 == 3 ? .25f : .75f;
  }
  const cp_const_plane pa{a, n * 4, 4}, pb{b, n * 4, 4}, pm{m, n * 4, 4};
  for (int op : {CP_MIX, CP_PRODUCT, CP_GUIDED_MULTIPLY, CP_DIFFERENCE, CP_ADD, CP_SUBTRACT, CP_INVERT_MIX})
    for (double neutral : {-0.0, 0.0, .1, .5, -1.0})
      for (double opacity : {0.0, .17, .625, 1.0})
        for (bool masked : {false, true}) {
          const cp_plane_config c{{CP_F32, 32}, op, opacity, neutral, neutral, neutral, 0, 0, CP_WEIGHT_CONTINUOUS};
          CHECK(table->process_plane(&c, pa, pb, masked ? &pm : nullptr, nullptr, &pb, {actual, n * 4, 4},
                                     {n, 1, 0, 1}) == CP_OK);
          CHECK(cp_process_plane(&c, pa, pb, masked ? &pm : nullptr, nullptr, &pb, {expected, n * 4, 4},
                                 {n, 1, 0, 1}) == CP_OK);
          for (int i = 0; i < n; ++i) {
            // Non-endpoint arithmetic may choose a different quiet NaN payload.
            if ((!masked || m[i] * opacity != 0) && opacity != 0 && std::isnan(actual[i]) && std::isnan(expected[i]))
              continue;
            if ((op == CP_MIX || op == CP_PRODUCT) && masked && opacity > 0 && m[i] != 0 &&
                !(opacity == 1 && m[i] == 1) && std::isfinite(actual[i]) && std::isfinite(expected[i])) {
              const double w = opacity * m[i];
              const double scale = op == CP_MIX ? std::abs(double(a[i]))*(1-w)+std::abs(double(b[i]))*w
                  : std::abs(double(a[i]))*((1-w)+std::abs(double(b[i]))*w);
              CHECK(std::abs(double(actual[i])-expected[i]) <= 16*std::numeric_limits<float>::epsilon()*std::max(1.0,scale));
              continue;
            }
            if (std::memcmp(actual + i, expected + i, 4) != 0)
              std::fprintf(stderr, "FloatBlendSpecials op=%d neutral=%g opacity=%g mask=%d i=%d a=%a b=%a m=%a got=%a ref=%a\n",
                           op, neutral, opacity, int(masked), i, double(a[i]), double(b[i]), double(m[i]), double(actual[i]), double(expected[i]));
            CHECK(std::memcmp(actual + i, expected + i, 4) == 0);
          }
        }
}
// All U8 input pairs with constant masks and neighboring opacities cover
// both sides of rounding boundaries, including dense half-integer cases.
void U8MaskedMixRounding(const cp_kernels* k, int operation = CP_MIX) {
  constexpr int count = 65537;
  std::vector<uint8_t> a(count), b(count), mask(count), got(count), expected(count);
  for (int i = 0; i < count; ++i) {
    a[i] = uint8_t(i);
    b[i] = uint8_t(i >> 8);
  }
  const cp_const_plane pa{a.data(), count, 1}, pb{b.data(), count, 1}, pm{mask.data(), count, 1};
  const cp_rows rows{count, 1, 0, 1};
  cp_plane_config c{};
  c.format = {CP_U8, 8};
  c.operation = operation;
  c.inversion_sum = 255;
  c.weight_rule = CP_WEIGHT_CONTINUOUS;
  for (double opacity : {1.0 / 65535, .1, .17, std::nextafter(.5, 0.), .5,
                         std::nextafter(.5, 1.), .625, 1.0 - 1.0 / 65535, 1.}) {
    c.opacity = opacity;
    for (uint8_t m : {0, 1, 51, 85, 127, 128, 170, 254, 255}) {
      std::fill(mask.begin(), mask.end(), m);
      CHECK(cp_process_plane(&c, pa, pb, &pm, nullptr, nullptr, {expected.data(), count, 1}, rows) == CP_OK);
      CHECK(k->process_plane(&c, pa, pb, &pm, nullptr, nullptr, {got.data(), count, 1}, rows) == CP_OK);
      for (int i = 0; i < count; ++i)
        CHECK(std::abs(int(got[i]) - int(expected[i])) <= ((m == 0 || (opacity == 1 && m == 255)) ? 0 : 1));
    }
  }
}
template <class T>
void ContinuousIntegerRounding(const cp_kernels* k, int bits) {
  constexpr int count = 65541;
  std::vector<T> a(count), b(count), got(count), expected(count);
  for (int i = 0; i < count; ++i) {
    a[i] = T(i & ((1u << bits) - 1));
    b[i] = sizeof(T) == 1 ? T(i >> 8) : T((uint32_t(i) * 40503u + 19u) & ((1u << bits) - 1));
  }
  const T maximum = T((1u << bits) - 1);
  a[65536] = b[65536] = maximum;
  a[65537] = maximum;
  b[65537] = 0;
  a[65538] = 0;
  b[65538] = maximum;
  a[65539] = maximum / 2;
  b[65539] = maximum;
  // Enumerate every U8 pair, or every U16 base code with a permuted source.
  // Narrow U16 inputs obey the declared-depth precondition.
  const ptrdiff_t pitch = count * ptrdiff_t(sizeof(T));
  const cp_const_plane pa{a.data(), pitch, sizeof(T)}, pb{b.data(), pitch, sizeof(T)};
  const cp_rows rows{count, 1, 0, 1};
  cp_plane_config c{};
  c.format = {sizeof(T) == 1 ? CP_U8 : CP_U16, bits};
  c.weight_rule = CP_WEIGHT_CONTINUOUS;
  const auto check = [&]() {
    CHECK(cp_process_plane(&c, pa, pb, nullptr, nullptr, nullptr, {expected.data(), pitch, sizeof(T)}, rows) == CP_OK);
    CHECK(k->process_plane(&c, pa, pb, nullptr, nullptr, nullptr, {got.data(), pitch, sizeof(T)}, rows) == CP_OK);
    if ((c.operation == CP_DIFFERENCE && c.bias >= 0 && c.bias <= 65535 && c.bias == std::floor(c.bias)) || c.operation == CP_PRODUCT || c.operation == CP_ADD || c.operation == CP_SUBTRACT || (c.operation == CP_INVERT_MIX && c.inversion_sum >= 0 && c.inversion_sum <= 65535 &&
        c.inversion_sum == std::floor(c.inversion_sum))) {
      for (int i = 0; i < count; ++i)
        CHECK(std::abs(int(got[i]) - int(expected[i])) <= 1);
    } else {
      CHECK(got == expected);
    }
  };
  for (int operation : {CP_PRODUCT, CP_ADD, CP_SUBTRACT}) {
    c.operation = operation;
    for (double opacity : {.17, 1. / 32768, std::nextafter(.5, 0.), .5, .625, std::nextafter(.625, 1.), 32767. / 32768}) {
      c.opacity = opacity;
      check();
    }
    if (operation != CP_PRODUCT || bits == 8 || bits == 16) {
      c.opacity = 1;
      check();
    }
  }
  // Offset bounds cover negative intermediate targets, upper clipping, the
  // chroma inversion sum 65536, and fractional/out-of-range fallback paths.
  for (int operation : {CP_INVERT_MIX, CP_DIFFERENCE}) {
    c.operation = operation;
    for (double offset : {-1., 0., .5, double(1u << (bits - 1)), double((1u << bits) - 1), double(1u << bits), 65535.,
                          65536., 65537.}) {
      c.inversion_sum = c.bias = offset;
      for (double opacity : {0., .17, 1. / 32768, .5, .625, std::nextafter(.625, 1.), 32767. / 32768, 1.}) {
        c.opacity = opacity;
        check();
      }
    }
    c.inversion_sum = c.bias = 65536;
    c.opacity = .625;
    check();
    for (bool alias_base : {false, true}) {
      got = alias_base ? a : b;
      const cp_const_plane alias{got.data(), pitch, sizeof(T)};
      CHECK(k->process_plane(&c, alias_base ? alias : pa, alias_base ? pb : alias, nullptr, nullptr, nullptr,
                             {got.data(), pitch, sizeof(T)}, rows) == CP_OK);
      CHECK(got == expected);
    }
  }
}
template <class T>
static void QuantizedMix(const cp_kernels* k, int bits, int operation = CP_MIX) {
  const int count = 65539, maximum = (1 << bits) - 1;
  std::vector<T> a(count), b(count), got(count), ref(count);
  std::mt19937 rng(2718 + bits);
  for (int i = 0; i < count; ++i) {
    // Exhaust every U8 pair and sample the declared U16 code range.
    a[i] = sizeof(T) == 1 ? T(i >> 8) : T(rng() & maximum);
    b[i] = sizeof(T) == 1 ? T(i) : T(rng() & maximum);
  }
  a[count - 3] = 0; b[count - 3] = T(maximum);
  a[count - 2] = T(maximum); b[count - 2] = 0;
  a[count - 1] = 100; b[count - 1] = 50;
  const ptrdiff_t pitch = count * sizeof(T);
  const cp_const_plane pa{a.data(), pitch, sizeof(T)}, pb{b.data(), pitch, sizeof(T)};
  const cp_rows rows{count, 1, 0, 1};
  std::vector<double> weights{0, 1, .17, .625, .001, .999, .123456789};
  for (int w : {0, 1, 5570, 16383, 32766, 32767}) {
    const double midpoint = (w + .5) / 32768;
    weights.push_back(std::nextafter(midpoint, 0.0));
    weights.push_back(midpoint);
    weights.push_back(std::nextafter(midpoint, 1.0));
  }
  for (double opacity : weights) {
    cp_plane_config c{};
    c.format = {sizeof(T) == 1 ? CP_U8 : CP_U16, bits};
    c.operation = operation; c.opacity = opacity; c.weight_rule = CP_WEIGHT_CONTINUOUS;
    c.inversion_sum = maximum;
    CHECK(cp_process_plane(&c, pa, pb, nullptr, nullptr, nullptr, {ref.data(), pitch, sizeof(T)}, rows) == CP_OK);
    for (int alias = 0; alias != 3; ++alias) {
      got = alias == 2 ? b : a;
      const cp_const_plane in{got.data(), pitch, sizeof(T)};
      CHECK(k->process_plane(&c, alias == 1 ? in : pa, alias == 2 ? in : pb, nullptr, nullptr, nullptr,
                              {got.data(), pitch, sizeof(T)}, rows) == CP_OK);
      for (int i = 0; i < count; ++i) {
        CHECK(std::abs(int(got[i]) - int(ref[i])) <= (opacity == 0 || opacity == 1 ? 0 : 1));
        if (opacity != 0 && opacity != 1 && opacity * 32768 != std::floor(opacity * 32768))
          CHECK(got[i] <= maximum);
      }
    }
  }
}

template <class T>
static void QuantizedMasked(const cp_kernels* k, int bits) {
  constexpr int count = 65539;
  const int max = (1 << bits) - 1;
  std::vector<T> a(count), b(count), m(count), got(count), ref(count);
  std::mt19937 rng(bits + 777);
  for (int i = 0; i < count; ++i) {
    a[i] = T(rng() & max); b[i] = T(rng() & max); m[i] = T(i & max);
  }
  const ptrdiff_t pitch = count * sizeof(T);
  const cp_const_plane pa{a.data(), pitch, sizeof(T)}, pb{b.data(), pitch, sizeof(T)}, pm{m.data(), pitch, sizeof(T)};
  const cp_rows rows{count, 1, 0, 1};
  for (int op : {CP_MIX, CP_INVERT_MIX, CP_PRODUCT, CP_ADD, CP_SUBTRACT, CP_DIFFERENCE})
    for (double opacity : {0., .17, .5, .625, 1.0 - 1.0 / 65535, 1.})
      for (bool endpoints : {false, true}) {
        // Exercise legal endpoint codes in both a full vector and the tail.
        a[17] = endpoints ? T(max) : T(max / 2);
        m[count - 1] = endpoints ? T(max) : T(0);
        cp_plane_config c{};
        c.format = {sizeof(T) == 1 ? CP_U8 : CP_U16, bits};
        c.operation = op; c.opacity = opacity; c.inversion_sum = max; c.bias = (max + 1) / 2; c.weight_rule = CP_WEIGHT_CONTINUOUS;
        CHECK(cp_process_plane(&c, pa, pb, &pm, nullptr, nullptr, {ref.data(), pitch, sizeof(T)}, rows) == CP_OK);
        for (int alias = 0; alias < 4; ++alias) {
          got = alias == 2 ? b : alias == 3 ? m : a;
          const cp_const_plane in{got.data(), pitch, sizeof(T)};
          const cp_const_plane mask_in = alias == 3 ? in : pm;
          CHECK(k->process_plane(&c, alias == 1 ? in : pa, alias == 2 ? in : pb, &mask_in, nullptr, nullptr,
                                  {got.data(), pitch, sizeof(T)}, rows) == CP_OK);
          for (int i = 0; i < count; ++i) {
            const bool exact = opacity == 0 || m[i] == 0 || (opacity == 1 && m[i] == max);
            if (std::abs(int(got[i]) - int(ref[i])) > (exact ? 0 : 1))
              std::fprintf(stderr, "quantized op=%d bits=%d opacity=%.17g alias=%d i=%d a=%u b=%u m=%u got=%u ref=%u\n", op, bits, opacity, alias, i, unsigned(a[i]), unsigned(b[i]), unsigned(m[i]), unsigned(got[i]), unsigned(ref[i]));
            CHECK(std::abs(int(got[i]) - int(ref[i])) <= (exact ? 0 : 1));
          }
        }
      }
}

template <class T>
static void GuidedBounded(const cp_kernels* k, int bits) {
  constexpr int count = 65539;
  const int max = (1 << bits) - 1;
  std::vector<T> a(count), b(count), m(count), got(count), ref(count);
  std::mt19937 rng(bits + 777);
  for (int i = 0; i < count; ++i) {
    a[i] = T(rng() & max); b[i] = T(rng() & max); m[i] = T(i & max);
  }
  const ptrdiff_t pitch = count * sizeof(T);
  const cp_const_plane pa{a.data(), pitch, sizeof(T)}, pb{b.data(), pitch, sizeof(T)}, pm{m.data(), pitch, sizeof(T)};
  const cp_rows rows{count, 1, 0, 1};
  for (double neutral : {0., .1, double(max) / 2, double(max), -1., double(max) + 1.})
    for (double opacity : {0., .17, .5, .625, 1.0 - 1.0 / 65535, 1.})
      for (bool endpoints : {false, true}) {
        // Exercise legal endpoint codes in both a full vector and the tail.
        b[33] = endpoints ? T(max) : T(max / 3);
        a[17] = endpoints ? T(max) : T(max / 2);
        m[count - 1] = endpoints ? T(max) : T(0);
        cp_plane_config c{};
        c.format = {sizeof(T) == 1 ? CP_U8 : CP_U16, bits};
        const int op = CP_GUIDED_MULTIPLY; c.operation = op; c.neutral = neutral; c.opacity = opacity; c.inversion_sum = max; c.bias = (max + 1) / 2; c.weight_rule = CP_WEIGHT_CONTINUOUS;
        CHECK(cp_process_plane(&c, pa, pb, &pm, nullptr, &pb, {ref.data(), pitch, sizeof(T)}, rows) == CP_OK);
        for (int alias = 0; alias < 4; ++alias) {
          got = alias == 2 ? b : alias == 3 ? m : a;
          const cp_const_plane in{got.data(), pitch, sizeof(T)};
          const cp_const_plane mask_in = alias == 3 ? in : pm;
          CHECK(k->process_plane(&c, alias == 1 ? in : pa, alias == 2 ? in : pb, &mask_in, nullptr, &(alias == 2 ? in : pb),
                                  {got.data(), pitch, sizeof(T)}, rows) == CP_OK);
          for (int i = 0; i < count; ++i) {
            const bool exact = opacity == 0 || opacity == 1 || neutral < 0 || neutral > max || m[i] == 0 || (opacity == 1 && m[i] == max);
            if (std::abs(int(got[i]) - int(ref[i])) > (exact ? 0 : 1))
              std::fprintf(stderr, "quantized op=%d bits=%d opacity=%.17g alias=%d i=%d a=%u b=%u m=%u got=%u ref=%u\n", op, bits, opacity, alias, i, unsigned(a[i]), unsigned(b[i]), unsigned(m[i]), unsigned(got[i]), unsigned(ref[i]));
            CHECK(std::abs(int(got[i]) - int(ref[i])) <= (exact ? 0 : 1));
          }
        }
      }
}

static void FloatBlendZeroSigns(const cp_kernels* k) {
  constexpr int n=67;
  for (int operation : {CP_MIX, CP_PRODUCT}) {
    std::vector<float> a(n, -0.f);
    std::vector<float> b(n, operation==CP_MIX ? -0.f : 0.f), m(n,1.f), got(n), ref(n);
    const cp_const_plane pa{a.data(),n*4,4}, pb{b.data(),n*4,4}, pm{m.data(),n*4,4};
    cp_plane_config c{}; c.format={CP_F32,32}; c.operation=static_cast<cp_operation>(operation);
    c.opacity=operation==CP_MIX ? .9 : .625; c.weight_rule=CP_WEIGHT_CONTINUOUS;
    const cp_rows r{n,1,0,1};
    CHECK(k->process_plane(&c,pa,pb,&pm,nullptr,nullptr,{got.data(),n*4,4},r)==CP_OK);
    CHECK(cp_process_plane(&c,pa,pb,&pm,nullptr,nullptr,{ref.data(),n*4,4},r)==CP_OK);
    CHECK(std::memcmp(got.data(),ref.data(),n*4)==0);
  }
}

static void FiniteFloatBlends(const cp_kernels* k) {
  constexpr int width = 2051, pitch = width + 5, size = pitch * 2;
  std::mt19937 rng(89271);
  double max_normal_error = 0;
  for (int operation : {CP_MIX, CP_PRODUCT})
    for (bool negative : {false, true})
      for (double opacity : {0., 1.0 / 65535, .003, .17, .5, .625, .75, .75 + 1.0 / 65535, .9, 1.0 - 1.0 / 65535, 1.})
        for (int scenario = 0; scenario < 7; ++scenario) {
          std::vector<float> a(size), b(size), m(size), got(size), ref(size);
          const std::vector<float> untouched(size, 19.f);
          for (int i = 0; i < size; ++i) {
            a[i] = float(int(rng() % 131073) - 65536) / 65536.f;
            b[i] = float(rng() % 65536) / 65535.f;
            m[i] = i % 7 == 0 ? 0.f : i % 7 == 1 ? 1.f : float(rng() % 65536) / 65535.f;
            if (scenario == 1) b[i] = -a[i]; // cancellation
            if (scenario == 2) a[i] = std::ldexp(a[i], i % 23 - 16);
            if (scenario == 3) {
              a[i] = i % 3 == 0 ? -0.f : i % 3 == 1 ? 100.f : -4.f;
              b[i] = i % 4 == 0 ? -2.f : i % 4 == 1 ? 4.f : b[i];
            }
          }
          if (scenario == 4) for (int i=0;i<size;++i) { a[i] *= 100; b[i] = i%5==0 ? -.125f : b[i]*100; }
          if (scenario == 5) for (int i=0;i<size;++i) { a[i] = std::ldexp(a[i], i%23-16); b[i] = std::ldexp(b[i], (i*37)%23-16); }
          if (scenario == 6) for (int i=0;i<size;++i) { b[i] = i%3==0 ? 1.f/65535 : i%3==1 ? 0.f : -float(std::min(100.0, (1-opacity)/std::max(opacity,1.0/65535))); m[i]=1; }
          const int origin = negative ? pitch : 0;
          const ptrdiff_t stride = (negative ? -pitch : pitch) * sizeof(float);
          for (int alias = 0; alias < 4; ++alias) {
            const auto& initial = alias == 1 ? a : alias == 2 ? b : alias == 3 ? m : untouched;
            std::copy(initial.begin(), initial.end(), got.begin());
            std::copy(initial.begin(), initial.end(), ref.begin());
            const auto in = [&](const std::vector<float>& v) { return cp_const_plane{v.data() + origin, stride, 4}; };
            const auto ga = in(alias == 1 ? got : a), ra = in(alias == 1 ? ref : a);
            const auto gb = in(alias == 2 ? got : b), rb = in(alias == 2 ? ref : b);
            const auto gm = in(alias == 3 ? got : m), rm = in(alias == 3 ? ref : m);
            for (int rule : {CP_WEIGHT_CONTINUOUS, CP_WEIGHT_CODE}) {
              std::copy(initial.begin(), initial.end(), got.begin());
              std::copy(initial.begin(), initial.end(), ref.begin());
              const cp_plane_config c{{CP_F32,32}, operation, opacity, 0, 0, 0, 0, 0, rule};
              CHECK(k->process_plane(&c, ga, gb, &gm, nullptr, nullptr, {got.data()+origin,stride,4}, {width,2,0,2}) == CP_OK);
              CHECK(cp_process_plane(&c, ra, rb, &rm, nullptr, nullptr, {ref.data()+origin,stride,4}, {width,2,0,2}) == CP_OK);
              for (int i = 0; i < size; ++i) {
                const bool pixel = i % pitch < width;
                const bool eligible = pixel && opacity > 0 && m[i] != 0 && !(opacity==1 && m[i]==1) &&
                    std::isfinite(a[i]) && std::isfinite(b[i]) && std::isfinite(got[i]) && std::isfinite(ref[i]);
                if (eligible) {
                  const double error = std::abs(double(got[i]) - ref[i]);
                  const double w = opacity*m[i];
                  const double scale = operation==CP_MIX ? std::abs(double(a[i]))*(1-w)+std::abs(double(b[i]))*w
                      : std::abs(double(a[i]))*((1-w)+std::abs(double(b[i]))*w);
                  const double limit = 16*std::numeric_limits<float>::epsilon()*std::max(1.0,scale);
                  if (!(error<=limit)) std::fprintf(stderr,"float op=%d o=%.17g a=%.9g b=%.9g m=%.9g got=%.9g ref=%.9g limit=%.9g\n",operation,opacity,a[i],b[i],m[i],got[i],ref[i],limit);
                  CHECK(error <= limit);
                  if (scenario == 0) max_normal_error = std::max(max_normal_error, error);
                } else if (!(std::isnan(got[i]) && std::isnan(ref[i])))
                  CHECK(std::memcmp(&got[i], &ref[i], sizeof(float)) == 0);
              }
            }
          }
        }
  std::printf("bounded float blends normalized max error %.9g\n", max_normal_error);
}

void CheckDenseCopy(const cp_kernels* table) {
  for (int bytes : {1, 2, 4})
    for (int padding : {0, 7})
      for (bool negative : {false, true}) {
        constexpr int width = 65, height = 5;
        const int stride = (width + padding) * bytes;
        std::vector<uint32_t> source((stride * height + 3) / 4), dest(source.size()), expected(source.size());
        auto* src = reinterpret_cast<uint8_t*>(source.data());
        auto* dst = reinterpret_cast<uint8_t*>(dest.data());
        auto* ref = reinterpret_cast<uint8_t*>(expected.data());
        for (int i = 0; i < stride * height; ++i) src[i] = uint8_t(i * 17);
        const int origin = negative ? stride * (height - 1) : 0;
        const ptrdiff_t pitch = negative ? -stride : stride;
        for (int y = 1; y < 4; ++y)
          std::memcpy(ref + origin + y * pitch, src + origin + y * pitch, width * bytes);
        const cp_format f{bytes == 1 ? CP_U8 : bytes == 2 ? CP_U16 : CP_F32, bytes * 8};
        CHECK(table->copy(f, {src + origin, pitch, bytes}, {dst + origin, pitch, bytes}, {width, height, 1, 3}) == CP_OK);
        CHECK(dest == expected);
      }
}

void U8AverageBounds(const cp_kernels* table) {
  for (int width : {1, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1920})
    for (bool negative : {false, true})
      for (int alias : {0, 1, 2}) {
        const int height = 5, pitch = width + 13;
        std::vector<uint8_t> a(pitch * height + 2), b(a.size()), out(a.size(), 0xAD);
        for (size_t i = 0; i < a.size(); ++i) {
          a[i] = uint8_t(i * 37);
          b[i] = uint8_t(i * 19 + 3);
        }
        auto& dest = alias == 1 ? a : alias == 2 ? b : out;
        auto expected = dest;
        const int origin = 1 + (negative ? pitch * (height - 1) : 0);
        const ptrdiff_t stride = negative ? -pitch : pitch;
        for (int y = 1; y < 4; ++y)
          for (int x = 0; x < width; ++x) {
            const ptrdiff_t i = origin + y * stride + x;
            expected[i] = uint8_t((unsigned(a[i]) + b[i] + 1) >> 1);
          }
        cp_plane_config config{};
        config.format = {CP_U8, 8};
        config.operation = CP_MIX;
        config.opacity = .5;
        CHECK(table->process_plane(&config, {a.data() + origin, stride, 1}, {b.data() + origin, stride, 1}, nullptr,
                                   nullptr, nullptr, {dest.data() + origin, stride, 1},
                                   {width, height, 1, 3}) == CP_OK);
        CHECK(dest == expected);
      }
}

int main(int argc, char** argv) {
  CHECK(cp_get_kernels(CP_TARGET_C));
  CHECK(cp_get_kernels(CP_TARGET_NATIVE));
  CHECK(cp_choose_target(0) == CP_TARGET_C);
  const int64_t supported = cp_supported_targets();
  CHECK((supported & cp_compiled_targets()) == supported);
  CHECK(!cp_get_kernels(INT64_C(-2)));
  CHECK(!cp_get_kernels(INT64_C(3)));
  for (int bit = 0; bit < 63; ++bit) {
    const int64_t target = INT64_C(1) << bit;
    CHECK((cp_get_kernels(target) != nullptr) == ((supported & target) != 0));
  }
  std::vector<int64_t> targets = {CP_TARGET_C};
  for (int64_t remaining = supported; remaining; remaining &= remaining - 1)
    targets.push_back(remaining & -remaining);
  if (argc > 1 && std::strcmp(argv[1], "u16-mix") == 0) {
    for (const int64_t target : targets)
      for (int bits = 9; bits <= 16; ++bits) {
        const auto* table = cp_get_kernels(target);
        QuantizedMix<uint16_t>(table, bits);
        for (int width : {1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257})
          for (int step : {1, 4})
            for (bool negative : {false, true})
              CheckLayout<uint16_t>(table, bits, width, step, negative);
      }
    std::puts("U16 mix tests passed");
    return 0;
  }
  if (argc > 1 && std::strcmp(argv[1], "u8-average") == 0) {
    for (const int64_t target : targets)
      U8AverageBounds(cp_get_kernels(target));
    std::puts("U8 average boundaries passed");
    return 0;
  }
  for (const int64_t target : targets) {
    U8AverageBounds(cp_get_kernels(target));
    CheckDenseCopy(cp_get_kernels(target));
    const auto* table = cp_get_kernels(target);
    CHECK(table);
    CHECK(table == cp_get_kernels(target));
    std::printf("target 0x%llx\n", static_cast<unsigned long long>(target));
    for (int width : {1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129})
      for (int step : {1, 4})
        for (bool negative : {false, true}) {
          CheckLayout<uint8_t>(table, 8, width, step, negative);
          for (int bits = 9; bits <= 16; ++bits)
            CheckLayout<uint16_t>(table, bits, width, step, negative);
          CheckLayout<float>(table, 32, width, step, negative);
        }
    Exhaustive(table);
    for (int width : {1, 7, 31, 257})
      for (int step : {1, 4})
        for (bool negative : {false, true}) {
          Arithmetic<uint8_t>(table, 8, width, step, negative);
          for (int bits = 9; bits <= 16; ++bits)
            Arithmetic<uint16_t>(table, bits, width, step, negative);
          Arithmetic<float>(table, 32, width, step, negative);
        }
    U8MaskedMixRounding(table);
    U8MaskedMixRounding(table, CP_INVERT_MIX);
    U8MaskedMixRounding(table, CP_PRODUCT);
    U8MaskedMixRounding(table, CP_ADD);
    U8MaskedMixRounding(table, CP_SUBTRACT);
    QuantizedMasked<uint8_t>(table, 8);
    GuidedBounded<uint8_t>(table, 8);
    for (int bits = 9; bits <= 16; ++bits)
      QuantizedMasked<uint16_t>(table, bits);
    for (int bits = 9; bits <= 16; ++bits) GuidedBounded<uint16_t>(table, bits);
    QuantizedMix<uint8_t>(table, 8);
    QuantizedMix<uint8_t>(table, 8, CP_INVERT_MIX);
    for (int bits = 9; bits <= 16; ++bits)
      QuantizedMix<uint16_t>(table, bits);
    for (int bits = 9; bits <= 16; ++bits)
      QuantizedMix<uint16_t>(table, bits, CP_INVERT_MIX);
    ContinuousIntegerRounding<uint8_t>(table, 8);
    for (int bits = 9; bits <= 16; ++bits)
      ContinuousIntegerRounding<uint16_t>(table, bits);
    FloatBlendZeroSigns(table);
    FiniteFloatBlends(table);
    FloatEndpoints(table);
    FloatBlendSpecials(table);
  }
  // Different instance policies can execute concurrently without shared dispatch mutation.
  std::thread a([&] { Exhaustive(cp_get_kernels(CP_TARGET_C)); });
  std::thread b([&] { Exhaustive(cp_get_kernels(CP_TARGET_NATIVE)); });
  a.join();
  b.join();
  std::puts("dispatch tests passed");
}
