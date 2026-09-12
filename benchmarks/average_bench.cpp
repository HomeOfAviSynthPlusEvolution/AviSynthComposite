// SPDX-License-Identifier: GPL-2.0-or-later
// Paired, batched 16-bit averaging microbenchmark. See README for cache semantics.
#include <composite/composite.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef CP_BENCH_AVX2
#include <upstream_commit.h>
extern "C" void cp_bench_average_avx2(int, unsigned char*, const unsigned char*, int, int, int);
#endif
namespace {
using Clock = std::chrono::steady_clock;
volatile uint64_t consumed;
struct Frame {
  std::vector<uint16_t> base, source, output;
};
struct Backend {
  std::string name;
  const cp_kernels* kernels;
};
int width = 1920, height = 1080, slots = 1, batch = 64, rounds = 30;
void check(bool ok) {
  if (!ok)
    throw std::runtime_error("average benchmark validation failed");
}
} // namespace
int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (i + 1 == argc)
        throw std::runtime_error("option requires a value");
      const int value = std::stoi(argv[++i]);
      if (arg == "--width")
        width = value;
      else if (arg == "--height")
        height = value;
      else if (arg == "--slots")
        slots = value;
      else if (arg == "--batch")
        batch = value;
      else if (arg == "--rounds")
        rounds = value;
      else
        throw std::runtime_error("options: --width N --height N --slots N --batch N --rounds N");
    }
    check(width > 0 && width <= 8192 && height > 0 && height <= 8192 && slots > 0 && slots <= 64 && batch > 0 &&
          batch <= 4096 && batch % slots == 0 && rounds >= 3 && rounds <= 1000);
    const size_t pixels = size_t(width) * height;
    std::vector<Frame> frames(slots);
    std::mt19937 rng(7331);
    for (auto& f : frames) {
      f.base.resize(pixels);
      f.source.resize(pixels);
      f.output.resize(pixels);
      for (size_t x = 0; x < pixels; ++x) {
        f.base[x] = uint16_t(rng());
        f.source[x] = uint16_t(rng());
      }
    }
    const auto native = cp_choose_target(CP_TARGET_NATIVE);
    std::vector<Backend> backends{{"own-native-" + std::to_string(native), cp_get_kernels(native)}};
    if ((cp_supported_targets() & 512) && native != 512)
      backends.push_back({"own-avx2", cp_get_kernels(512)});
#ifdef CP_BENCH_AVX2
    if (cp_supported_targets() & 512)
      backends.push_back({"upstream-avx2", nullptr});
#endif
    const cp_rows rows{width, height, 0, height};
    const ptrdiff_t pitch = width * sizeof(uint16_t);
    const cp_plane_config config{{CP_U16, 16}, CP_MIX, .5, 0, 0, 0, 0, 0, CP_WEIGHT_CONTINUOUS};
    const auto call = [&](const Backend& b, Frame& f) {
      if (b.kernels) {
        check(b.kernels->process_plane(&config, {f.output.data(), pitch, 2}, {f.source.data(), pitch, 2}, nullptr,
                                       nullptr, nullptr, {f.output.data(), pitch, 2}, rows) == CP_OK);
      } else {
#ifdef CP_BENCH_AVX2
        cp_bench_average_avx2(2, reinterpret_cast<unsigned char*>(f.output.data()),
                              reinterpret_cast<const unsigned char*>(f.source.data()), int(pitch), int(pitch), height);
#endif
      }
    };
    const auto block = [&](const Backend& b) {
      for (auto& f : frames)
        std::copy(f.base.begin(), f.base.end(), f.output.begin());
      // Identical per-backend cache preparation: touch the full active ring twice
      // with the actual kernel, not with a checksum or a synthetic cache flush.
      for (int i = 0; i < 2 * slots; ++i)
        call(b, frames[i % slots]);
      const auto start = Clock::now();
      for (int i = 0; i < batch; ++i)
        call(b, frames[i % slots]);
      const auto end = Clock::now();
      // Repeated ceil-average: source + ceil((base-source)/2^k).
      // Verify all output independently, including after values have converged.
      const int k = std::min(16, 2 + batch / slots);
      const uint32_t scale = 1u << k;
      uint64_t sum = 0;
      for (const auto& f : frames)
        for (size_t x = 0; x < pixels; ++x) {
          const uint32_t a = f.base[x], s = f.source[x];
          const uint32_t expected = a >= s ? s + (a - s + scale - 1) / scale : s - (s - a) / scale;
          check(f.output[x] == expected);
          sum += f.output[x];
        }
      consumed = sum;
      return std::chrono::duration<double, std::milli>(end - start).count();
    };
    // Time-based initial warmup. Correctness checking stays outside timed blocks.
    const auto warm_until = Clock::now() + std::chrono::seconds(1);
    do {
      for (const auto& b : backends)
        block(b);
    } while (Clock::now() < warm_until);
    std::vector<size_t> order(backends.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::cerr << "U16 average, width=" << width << " height=" << height << " slots=" << slots
              << " active_MiB=" << double(pixels * 4 * slots) / (1024 * 1024) << " batch=" << batch
              << " rounds=" << rounds << " supported=" << cp_supported_targets() << '\n';
#ifdef CP_BENCH_AVX2
    std::cerr << "upstream_commit=" << CP_UPSTREAM_COMMIT << '\n';
#endif
    std::cout << "round,position,backend,width,height,slots,batch,batch_ms,per_call_ms\n" << std::setprecision(12);
    for (int round = 0; round < rounds; ++round) {
      // Within each complete group every backend occupies every position once;
      // randomize the base order between groups (deterministic seed).
      if (round % int(order.size()) == 0)
        std::shuffle(order.begin(), order.end(), rng);
      else
        std::rotate(order.begin(), order.begin() + 1, order.end());
      for (size_t position = 0; position < order.size(); ++position) {
        const auto& b = backends[order[position]];
        const double ms = block(b);
        std::cout << round << ',' << position << ',' << b.name << ',' << width << ',' << height << ',' << slots << ','
                  << batch << ',' << ms << ',' << ms / batch << '\n';
      }
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
