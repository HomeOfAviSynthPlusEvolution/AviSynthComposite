// SPDX-License-Identifier: GPL-2.0-or-later
#include <composite/composite.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

template <class T>
void Verify(int bits, int weights) {
  constexpr int n = 65536;
  const uint64_t max = (UINT64_C(1) << bits) - 1;
  std::vector<T> a(n), b(n), mask(n), out(n), expected(n);
  std::mt19937 rng(821 + bits);
  for (int i = 0; i < n; ++i) {
    a[i] = T(bits == 8 ? i / 256 : rng() & max);
    b[i] = T(bits == 8 ? i % 256 : rng() & max);
    if (bits != 8 && i < 256) {
      a[i] = T(max - i / 16);
      b[i] = T(max - i % 16);
    }
  }
  const cp_const_plane pa{a.data(), n * sizeof(T), sizeof(T)}, pb{b.data(), n * sizeof(T), sizeof(T)},
      pm{mask.data(), n * sizeof(T), sizeof(T)};
  const cp_plane po{out.data(), n * sizeof(T), sizeof(T)};
  std::vector<int64_t> targets{CP_TARGET_C};
  for (auto remaining = cp_supported_targets(); remaining; remaining &= remaining - 1)
    targets.push_back(remaining & -remaining);
  for (int i = 0; i < n; ++i)
    expected[i] = T((uint64_t(a[i]) + b[i] + 1) / 2);
  const cp_plane_config average{{bits == 8 ? CP_U8 : CP_U16, bits}, CP_MIX, .5, 0, 0, 0, 0, 0, CP_WEIGHT_CONTINUOUS};
  for (const auto target : targets) {
    const auto* k = cp_get_kernels(target);
    if (k->process_plane(&average, pa, pb, nullptr, nullptr, nullptr, po, {n, 1, 0, 1}) != CP_OK || out != expected ||
        k->blend_compat(average.format, pa, pb, nullptr, po, {n, 1, 0, 1}, 128) != CP_OK || out != expected)
      std::abort();
  }
  for (int weight : {1, 127, 128, 8192, 20480, 20481, 24576, 32640, 32767}) {
    auto blend = average;
    blend.opacity = weight / 32768.0;
    for (int i = 0; i < n; ++i)
      expected[i] = T((uint64_t(a[i]) * (32768 - weight) + uint64_t(b[i]) * weight + 16384) / 32768);
    for (const auto target : targets)
      if (cp_get_kernels(target)->process_plane(&blend, pa, pb, nullptr, nullptr, nullptr, po, {n, 1, 0, 1}) != CP_OK ||
          out != expected)
        std::abort();
  }
  for (double opacity : {0., .37, .625, 1.})
    for (uint64_t neutral : {UINT64_C(0), (max + 1) / 2, max})
      for (bool masked : {false, true}) {
        for (int i = 0; i < n; ++i)
          mask[i] = T(rng() & max);
        const uint64_t level = uint64_t(std::floor(opacity * max + .5));
        for (int i = 0; i < n; ++i) {
          const uint64_t weight = masked ? (uint64_t(mask[i]) * level + max / 2) / max : level;
          const uint64_t darken = (weight * (max - b[i]) + max / 2) / max;
          expected[i] = T((uint64_t(a[i]) * (max - darken) + neutral * darken + max / 2) / max);
        }
        const cp_plane_config guided{average.format, CP_GUIDED_MULTIPLY, opacity, double(neutral), 0, 0, 0, 0,
                                     CP_WEIGHT_CODE};
        for (const auto target : targets)
          if (cp_get_kernels(target)->process_plane(&guided, pa, pb, masked ? &pm : nullptr, nullptr, &pb, po,
                                                    {n, 1, 0, 1}) != CP_OK ||
              out != expected)
            std::abort();
      }
  for (int weight = 0; weight < weights; ++weight) {
    for (int i = 0; i < n; ++i)
      mask[i] = T(bits == 8 ? weight : weight == 0 ? max : weight == 1 ? 0 : rng() & max);
    for (int op : {CP_MIX, CP_PRODUCT}) {
      const double opacity = bits == 8 || weight < 2 ? 1.0 : .625;
      const uint64_t level = uint64_t(std::floor(opacity * max + .5));
      for (int i = 0; i < n; ++i) {
        const uint64_t m = (uint64_t(mask[i]) * level + max / 2) / max;
        const uint64_t target = op == CP_PRODUCT ? uint64_t(a[i]) * b[i] / max : b[i];
        expected[i] = T((uint64_t(a[i]) * (max - m) + target * m + max / 2) / max);
      }
      cp_plane_config c{{bits == 8 ? CP_U8 : CP_U16, bits}, op, opacity, 0, 0, 0, 0, 0, CP_WEIGHT_CODE};
      for (const auto target : targets) {
        if (cp_get_kernels(target)->process_plane(&c, pa, pb, &pm, nullptr, nullptr, po, {n, 1, 0, 1}) != CP_OK ||
            out != expected) {
          std::fprintf(stderr, "code blend failed: bits=%d weight=%d op=%d target=%lld\n", bits, weight, op,
                       static_cast<long long>(target));
          std::abort();
        }
        if (opacity == 1) {
          c.weight_rule = CP_WEIGHT_CONTINUOUS;
          if (cp_get_kernels(target)->process_plane(&c, pa, pb, &pm, nullptr, nullptr, po, {n, 1, 0, 1}) != CP_OK ||
              out != expected)
            std::abort();
          c.weight_rule = CP_WEIGHT_CODE;
        }
      }
    }
  }
}
void VerifyQ15Differences() {
  // Every signed U8 difference at every Q15 weight. Integer translation of
  // both inputs leaves rounding unchanged; 511 samples also exercise tails.
  constexpr int n = 511;
  std::vector<uint8_t> a(n), b(n), out(n), expected(n);
  for (int i = 0; i < n; ++i) {
    a[i] = uint8_t(std::max(255 - i, 0));
    b[i] = uint8_t(std::max(i - 255, 0));
  }
  const cp_const_plane pa{a.data(), n, 1}, pb{b.data(), n, 1};
  const cp_plane po{out.data(), n, 1};
  std::vector<int64_t> targets{CP_TARGET_C};
  for (auto remaining = cp_supported_targets(); remaining; remaining &= remaining - 1)
    targets.push_back(remaining & -remaining);
  for (int weight = 0; weight <= 32768; ++weight) {
    const cp_plane_config c{{CP_U8, 8}, CP_MIX, weight / 32768.0, 0, 0, 0, 0, 0, CP_WEIGHT_CONTINUOUS};
    for (int i = 0; i < n; ++i)
      expected[i] = uint8_t((unsigned(a[i]) * (32768 - weight) + unsigned(b[i]) * weight + 16384) / 32768);
    for (const auto target : targets)
      if (cp_get_kernels(target)->process_plane(&c, pa, pb, nullptr, nullptr, nullptr, po, {n, 1, 0, 1}) != CP_OK ||
          out != expected) {
        std::fprintf(stderr, "Q15 difference mismatch: weight=%d target=%lld\n", weight, (long long)target);
        std::abort();
      }
  }
}

int main() {
  VerifyQ15Differences();
  // All 256^3 base/source/mask combinations for both 8-bit operations.
  // Wider depths include maximum-product overflow boundaries and random masks.
  Verify<uint8_t>(8, 256);
  for (int bits = 9; bits <= 16; ++bits)
    Verify<uint16_t>(bits, 4);
  std::puts("code blend integer reference passed");
}
