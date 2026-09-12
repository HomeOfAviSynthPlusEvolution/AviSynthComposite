// SPDX-License-Identifier: GPL-2.0-or-later
// With the inherited AviSynth linking exception; see LICENSE.
#include <composite/composite.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#ifdef CP_BENCH_UPSTREAM
#include "layer_kernels.h"
#include "upstream_commit.h"
extern "C" void cp_bench_yuv_add(int, bool, bool, unsigned char**, unsigned char**, unsigned char*, int, int, int,
                                 double);
#endif
#ifdef CP_BENCH_AVX2
extern "C" void cp_bench_average_avx2(int, unsigned char*, const unsigned char*, int, int, int);
extern "C" void cp_bench_overlay_mul(int, double, bool, int, int, const unsigned char*, unsigned char**,
                                     const unsigned char*, int);
extern "C" void cp_bench_sample420_avx2(int, unsigned char*, const unsigned char*, int, int, int);
extern "C" void cp_bench_masked_avx2(int, unsigned char*, const unsigned char*, const unsigned char*, int, int, int,
                                     double);
extern "C" void cp_bench_weighted_avx2(int, unsigned char*, const unsigned char*, int, int, int, double);
#endif

namespace {
using Clock = std::chrono::steady_clock;
int width = 1920, height = 1080, trials = 7;
bool all_targets = false, packed = false;
double blend_opacity = .625;
int only_step = 0, only_bits = 0;
std::string filter;
int measured = 0;
volatile double sink = 0;
std::vector<int64_t> targets;
void check(int status) {
  if (status != CP_OK)
    throw std::runtime_error("kernel rejected benchmark input");
}
template <class T>
struct Frame {
  std::array<std::vector<T>, 4> p;
  int step;
  explicit Frame(int s) : step(s) {
    for (int i = 0; i < (packed ? 1 : 4); ++i)
      p[i].resize(size_t(width) * height * step);
  }
  cp_const_plane read(int c) const {
    return {packed ? p[0].data() + c : p[c].data(), ptrdiff_t(size_t(width) * step * sizeof(T)), ptrdiff_t(step * sizeof(T))};
  }
  cp_plane write(int c) {
    return {packed ? p[0].data() + c : p[c].data(), ptrdiff_t(size_t(width) * step * sizeof(T)), ptrdiff_t(step * sizeof(T))};
  }
  void randomize(std::mt19937& rng, int bits) {
    for (auto& v : p)
      for (auto& x : v) {
        if constexpr (std::is_same<T, float>::value)
          x = float(rng() & 65535) / 65535.0f;
        else
          x = T(rng() & ((1u << bits) - 1));
      }
  }
};
template <class T>
void run(int bits, int step) {
  cp_format f{std::is_same<T, float>::value ? CP_F32 : sizeof(T) == 1 ? CP_U8 : CP_U16, bits};
  Frame<T> base(step), source(step), output(step), reference(step);
  std::mt19937 rng(14731 + bits + step);
  base.randomize(rng, bits);
  source.randomize(rng, bits);
  const cp_rows rows{width, height, 0, height};
  const double maximum = bits == 32 ? 1.0 : double((1u << bits) - 1);
  for (const std::string workload : {"continuous_product",
                                     "continuous_add",
                                     "continuous_subtract",
                                     "continuous_invert",
                                     "continuous_difference",
                                     "mix",
                                     "code_mix",
                                     "code_guided",
                                     "product",
                                     "guided",
                                     "compat",
                                     "layer_add_rgb",
                                     "layer_mul_rgb",
                                     "overlay_mul",
                                     "sample420",
                                     "yuv_add",
                                     "yuv_subtract",
                                     "yuv_soft",
                                     "yuv_hard",
                                     "yuv_difference",
                                     "yuv_exclusion",
                                     "luma",
                                     "affine",
                                     "clamp",
                                     "copy",
                                     "fill",
                                     "key"}) {
    if (!filter.empty() && workload != filter)
      continue;
    if (bits == 32 && workload == "compat")
      continue;
    if (step != 1 && workload.find("layer_") == 0)
      continue;
    const bool yuv = workload.find("yuv_") == 0;
    const bool utility = workload == "luma" || workload == "affine" || workload == "clamp" || workload == "copy" ||
                         workload == "fill" || workload == "key";
    if (bits == 32 && yuv && workload != "yuv_add" && workload != "yuv_subtract")
      continue;
    for (bool masked : {false, true}) {
      if (masked && (workload == "sample420" || utility))
        continue;
      const bool rgb = workload.find("layer_") == 0;
      const int channels = rgb || yuv || workload == "overlay_mul" ? 3 : 1;
      cp_plane_config c{};
      c.format = f;
      c.operation =
          workload == "continuous_add"                                                               ? CP_ADD
          : workload == "continuous_subtract"                                                        ? CP_SUBTRACT
          : workload == "continuous_invert"                                                          ? CP_INVERT_MIX
          : workload == "continuous_difference"                                                      ? CP_DIFFERENCE
          : workload == "product" || workload == "continuous_product" || workload == "layer_mul_rgb" ? CP_PRODUCT
          : (workload == "guided" || workload == "code_guided" || workload == "overlay_mul") ? CP_GUIDED_MULTIPLY
                                                                                             : CP_MIX;
      c.opacity = blend_opacity;
      c.inversion_sum = maximum;
      c.bias = bits == 32 ? 0 : (maximum + 1) / 2;
      c.neutral = workload == "code_guided" ? (bits == 32 ? 0 : (maximum + 1) / 2) : maximum / 2;
      c.weight_rule = rgb || workload == "code_mix" || workload == "code_guided" || workload == "product"
                          ? CP_WEIGHT_CODE
                          : CP_WEIGHT_CONTINUOUS;
      const auto mask = source.read(3), guide = source.read(workload == "overlay_mul" ? 0 : 2);
      const auto invoke = [&](const cp_kernels* k, Frame<T>& dst) {
        if (workload == "overlay_mul") {
          const cp_yuv_config config{f, CP_YUV_MULTIPLY, blend_opacity};
          const cp_const_yuv masks{mask, mask, mask};
          check(k->process_yuv(&config, {dst.read(0), dst.read(1), dst.read(2)},
                               {source.read(0), source.read(1), source.read(2)}, masked ? &masks : nullptr,
                               {dst.write(0), dst.write(1), dst.write(2)}, rows));
        } else if (yuv) {
          const int op = workload == "yuv_add"          ? CP_YUV_ADD
                         : workload == "yuv_subtract"   ? CP_YUV_SUBTRACT
                         : workload == "yuv_soft"       ? CP_YUV_SOFT_LIGHT
                         : workload == "yuv_hard"       ? CP_YUV_HARD_LIGHT
                         : workload == "yuv_difference" ? CP_YUV_DIFFERENCE
                                                        : CP_YUV_EXCLUSION;
          const cp_yuv_config config{f, op, blend_opacity};
          const cp_const_yuv masks{mask, mask, mask};
          check(k->process_yuv(&config, {dst.read(0), dst.read(1), dst.read(2)},
                               {source.read(0), source.read(1), source.read(2)}, masked ? &masks : nullptr,
                               {dst.write(0), dst.write(1), dst.write(2)}, rows));
        } else if (utility) {
          if (workload == "luma")
            check(
                k->rgb_luma(f, {source.read(0), source.read(1), source.read(2)}, dst.write(0), rows, CP_LUMA_NEAREST));
          else if (workload == "affine")
            check(k->affine(f, source.read(0), dst.write(0), rows, -1, maximum));
          else if (workload == "clamp")
            check(k->clamp(f, source.read(0), dst.write(0), rows, maximum / 4, maximum * .75));
          else if (workload == "copy")
            check(k->copy(f, source.read(0), dst.write(0), rows));
          else if (workload == "fill")
            check(k->fill(f, dst.write(0), rows, maximum / 2));
          else {
            const double key[3]{maximum / 2, maximum / 2, maximum / 2};
            const double tolerance[3]{maximum / 4, maximum / 4, maximum / 4};
            check(k->color_key(f, {source.read(0), source.read(1), source.read(2)}, mask, dst.write(0), rows, key,
                               tolerance));
          }
        } else if (workload == "sample420") {
          const cp_sampling s{width, height, 2, 2, CP_CENTER, 0, 0};
          check(k->resample_mask(f, source.read(0), dst.write(0), &s, {width / 2, height / 2, 0, height / 2}));
        } else if (workload == "compat") {
          check(k->blend_compat(f, dst.read(0), source.read(0), masked ? &mask : nullptr, dst.write(0), rows,
                                int(std::floor(blend_opacity * 256 + .5))));
        } else {
          for (int p = 0; p < channels; ++p) {
            check(k->process_plane(&c, dst.read(p), source.read(p), masked ? &mask : nullptr, nullptr, &guide,
                                   dst.write(p), rows));
          }
        }
      };
      reference = base;
      invoke(cp_get_kernels(CP_TARGET_C), reference);
      const auto measure = [&](const std::string& backend, auto call, double tolerance) {
        output = base;
        call();
        double error = 0, checksum = 0;
        for (int p = 0; p < 4; ++p)
          for (size_t i = 0; i < output.p[p].size(); ++i) {
            const double difference = std::abs(double(output.p[p][i]) - double(reference.p[p][i]));
            if (!std::isfinite(difference))
              throw std::runtime_error(workload + " " + backend + " non-finite output difference");
            error = std::max(error, difference);
            checksum += double(output.p[p][i]);
          }
        if (!(error <= tolerance))
          throw std::runtime_error(workload + " " + backend + " output error " + std::to_string(error));
        if (tolerance == 0)
          for (int p = 0; p < 4; ++p)
            if (!output.p[p].empty() &&
                std::memcmp(output.p[p].data(), reference.p[p].data(), output.p[p].size() * sizeof(T)) != 0)
              throw std::runtime_error(workload + " " + backend + " bitwise mismatch");
        std::vector<double> ms;
        for (int t = -2; t < trials; ++t) {
          output = base; // Reset/allocation, validation and checksum are outside the timed region.
          const auto begin = Clock::now();
          call();
          const auto end = Clock::now();
          // Consume every processed channel after the timer. In particular the
          // inline upstream templates must not lose unobserved stores under
          // optimization or LTO merely because only one pixel was inspected.
          double consumed = 0;
          for (int p = 0; p < channels; ++p)
            for (T value : output.p[p])
              consumed += double(value);
          sink = consumed;
          if (t >= 0)
            ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        }
        std::sort(ms.begin(), ms.end());
        const double median = ms[ms.size() / 2];
        ++measured;
        const double pixels =
            workload == "sample420" ? double(width / 2) * (height / 2) : double(width) * height * channels;
        std::cout << workload << ',' << bits << ',' << step << ',' << masked << ',' << width << ',' << height << ','
                  << backend << ',' << ms.front() << ',' << median << ',' << ms.back() << ',' << pixels / (median * 1e6)
                  << ',' << error << ',' << checksum << ',' << blend_opacity << ',' << (packed ? "packed" : "separate")
                  << '\n';
      };
      for (auto target : targets)
        measure(std::to_string(target), [&] { invoke(cp_get_kernels(target), output); },
                bits != 32 && (workload == "mix" || workload == "continuous_invert") ? 1 : 0);
#ifdef CP_BENCH_AVX2
      if (workload == "overlay_mul" && bits != 32 && step == 1 && (cp_supported_targets() & 512))
        measure(
            "upstream-avx2-overlay-mul",
            [&] {
              BYTE* dst[3];
              for (int p = 0; p < 3; ++p)
                dst[p] = reinterpret_cast<BYTE*>(output.p[p].data());
              cp_bench_overlay_mul(bits, blend_opacity, masked, width, height,
                                   reinterpret_cast<const BYTE*>(source.p[0].data()), dst,
                                   reinterpret_cast<const BYTE*>(source.p[3].data()), width * int(sizeof(T)));
            },
            1); // Upstream evaluates in binary32 before rounding integer codes.
      if (workload == "sample420" && step == 1 && (cp_supported_targets() & 512))
        measure(
            "upstream-avx2-sample420",
            [&] {
              cp_bench_sample420_avx2(bits, reinterpret_cast<unsigned char*>(output.p[0].data()),
                                      reinterpret_cast<const unsigned char*>(source.p[0].data()),
                                      width * int(sizeof(T)), width / 2, height / 2);
            },
            bits == 32 ? 2e-7 : 0);
      if (masked && step == 1 && (cp_supported_targets() & 512) &&
          (workload == "code_mix" || (workload == "mix" && bits == 32)))
        measure(
            "upstream-avx2-masked",
            [&] {
              cp_bench_masked_avx2(bits, reinterpret_cast<unsigned char*>(output.p[0].data()),
                                   reinterpret_cast<const unsigned char*>(source.p[0].data()),
                                   reinterpret_cast<const unsigned char*>(source.p[3].data()), width * int(sizeof(T)),
                                   width, height, blend_opacity);
            },
            bits == 32 ? 2e-7 : 0);
      // Integer upstream weights are fixed-point Q15. Compare only exactly
      // representable weights; do not quantize the continuous C ABI silently.
      if (workload == "mix" && !masked && step == 1 && blend_opacity > 0 && blend_opacity < 1 && blend_opacity != .5 &&
          (cp_supported_targets() & 512) && (bits == 32 || blend_opacity * 32768 == std::floor(blend_opacity * 32768)))
        measure(
            "upstream-avx2-weighted",
            [&] {
              cp_bench_weighted_avx2(bits, reinterpret_cast<unsigned char*>(output.p[0].data()),
                                     reinterpret_cast<const unsigned char*>(source.p[0].data()), width * int(sizeof(T)),
                                     width, height, blend_opacity);
            },
            bits == 32 ? 2e-7 : 0);
      // 512 is the public Highway AVX2 target bit. Never execute AVX2 based
      // solely on compile-time architecture or the native target's bit number.
      if (workload == "mix" && !masked && step == 1 && blend_opacity == .5 && (cp_supported_targets() & 512))
        measure(
            "upstream-avx2-average",
            [&] {
              const int pitch = width * int(sizeof(T));
              cp_bench_average_avx2(sizeof(T), reinterpret_cast<unsigned char*>(output.p[0].data()),
                                    reinterpret_cast<const unsigned char*>(source.p[0].data()), pitch, pitch, height);
            },
            bits == 32 ? 2e-7 : 0);
#endif
#ifdef CP_BENCH_UPSTREAM
      if (bits != 32 && step == 1 && (workload == "yuv_add" || workload == "yuv_subtract") &&
          double(float(blend_opacity)) == blend_opacity)
        measure(
            "upstream-scalar-yuv",
            [&] {
              BYTE* a[3];
              BYTE* b[3];
              for (int p = 0; p < 3; ++p) {
                a[p] = reinterpret_cast<BYTE*>(output.p[p].data());
                b[p] = reinterpret_cast<BYTE*>(source.p[p].data());
              }
              cp_bench_yuv_add(bits, workload == "yuv_add", masked, a, b, reinterpret_cast<BYTE*>(source.p[3].data()),
                               width * int(sizeof(T)), width, height, blend_opacity);
            },
            0);
      if (rgb) {
        measure(
            "upstream-scalar",
            [&] {
              BYTE* dst[4];
              const BYTE* src[4];
              for (int p = 0; p < 4; ++p) {
                dst[p] = reinterpret_cast<BYTE*>(output.p[p].data());
                src[p] = reinterpret_cast<const BYTE*>(source.p[p].data());
              }
              const int pitch = width * int(sizeof(T));
              const auto* alpha = masked ? src[3] : nullptr;
              if constexpr (std::is_same<T, float>::value) {
                if (workload == "layer_add_rgb") {
                  if (masked)
                    layer_planarrgb_add_f_c<true, true, false>(dst, src, alpha, pitch, pitch, pitch, width, height,
                                                               static_cast<float>(blend_opacity));
                  else
                    layer_planarrgb_add_f_c<true, false, false>(dst, src, alpha, pitch, pitch, pitch, width, height,
                                                                static_cast<float>(blend_opacity));
                } else {
                  if (masked)
                    layer_planarrgb_mul_f_c<true, true, false>(dst, src, alpha, pitch, pitch, pitch, width, height,
                                                               static_cast<float>(blend_opacity));
                  else
                    layer_planarrgb_mul_f_c<true, false, false>(dst, src, alpha, pitch, pitch, pitch, width, height,
                                                                static_cast<float>(blend_opacity));
                }
              } else {
                const int level = int(std::floor(blend_opacity * maximum + .5));
                if (workload == "layer_add_rgb") {
                  if (masked)
                    layer_planarrgb_add_c<T, true, true, false>(dst, src, alpha, pitch, pitch, pitch, width, height,
                                                                level, bits);
                  else
                    layer_planarrgb_add_c<T, true, false, false>(dst, src, alpha, pitch, pitch, pitch, width, height,
                                                                 level, bits);
                } else {
                  if (masked)
                    layer_planarrgb_mul_c<T, true, true, false>(dst, src, alpha, pitch, pitch, pitch, width, height,
                                                                level, bits);
                  else
                    layer_planarrgb_mul_c<T, true, false, false>(dst, src, alpha, pitch, pitch, pitch, width, height,
                                                                 level, bits);
                }
              }
            },
            bits == 32 ? 2e-7 : 0);
      }
#endif
    }
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--packed")
        packed = true;
      else if (arg == "--all-targets")
        all_targets = true;
      else if (arg == "--width" && i + 1 < argc)
        width = std::stoi(argv[++i]);
      else if (arg == "--height" && i + 1 < argc)
        height = std::stoi(argv[++i]);
      else if (arg == "--trials" && i + 1 < argc)
        trials = std::stoi(argv[++i]);
      else if (arg == "--workload" && i + 1 < argc)
        filter = argv[++i];
      else if (arg == "--opacity" && i + 1 < argc)
        blend_opacity = std::stod(argv[++i]);
      else if (arg == "--bits" && i + 1 < argc)
        only_bits = std::stoi(argv[++i]);
      else if (arg == "--step" && i + 1 < argc)
        only_step = std::stoi(argv[++i]);
      else
        throw std::runtime_error("usage: composite_bench [--width N] [--height N] [--trials N] [--all-targets] "
                                 "[--workload NAME] [--opacity W] [--step 1|4] [--bits 8|10|16|32] [--packed]");
    }
    if (packed) {
      if (only_step != 0 && only_step != 4)
        throw std::runtime_error("--packed requires --step 4 (or omit --step)");
      only_step = 4;
    }
    if (width < 2 || height < 2 || width > 16384 || height > 16384 || trials < 1 || trials > 101 ||
        !std::isfinite(blend_opacity) || blend_opacity < 0 || blend_opacity > 1 ||
        (only_step != 0 && only_step != 1 && only_step != 4) ||
        (only_bits != 0 && only_bits != 8 && only_bits != 10 && only_bits != 16 && only_bits != 32))
      throw std::runtime_error("invalid dimensions or trial count");
    targets.push_back(CP_TARGET_C);
    const int64_t supported = cp_supported_targets() & cp_compiled_targets();
    for (int64_t bit = 1; bit > 0 && bit <= (INT64_C(1) << 61); bit <<= 1)
      if ((supported & bit) && (all_targets || bit == cp_choose_target(CP_TARGET_NATIVE)))
        targets.push_back(bit);
#ifdef CP_BENCH_UPSTREAM
    std::cerr << "upstream=" << CP_UPSTREAM_COMMIT
              << " (extracted upstream cores; backend labels identify implementation)\n";
#endif
    std::cerr << "compiled=" << cp_compiled_targets() << " supported=" << supported << " trials=" << trials << '\n';
    std::cout << std::setprecision(12)
              << "workload,bits,step,mask,width,height,target,min_ms,median_ms,max_ms,gsamples_per_s,max_error,"
                 "checksum,opacity,layout\n";
    for (int step : {1, 4}) {
      if (only_step && step != only_step)
        continue;
      if (!only_bits || only_bits == 8)
        run<uint8_t>(8, step);
      if (!only_bits || only_bits == 10)
        run<uint16_t>(10, step);
      if (!only_bits || only_bits == 16)
        run<uint16_t>(16, step);
      if (!only_bits || only_bits == 32)
        run<float>(32, step);
    }
    if (!measured)
      throw std::runtime_error("unknown workload: " + filter);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
