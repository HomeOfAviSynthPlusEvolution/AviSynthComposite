// SPDX-License-Identifier: GPL-2.0-or-later
#include <composite/composite.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#define CHECK(e)                                                                                                       \
  do {                                                                                                                 \
    if (!(e)) {                                                                                                        \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #e);                                                             \
      std::abort();                                                                                                    \
    }                                                                                                                  \
  } while (0)
class Guarded {
public:
  unsigned char* allocation;
  unsigned char* data;
  size_t page, size, allocation_size;
  explicit Guarded(size_t bytes, bool at_end = true) : size(bytes) {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    page = info.dwPageSize;
#else
    page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    const size_t interior = ((size + page - 1) / page) * page;
    allocation_size = interior + 2 * page;
#if defined(_WIN32)
    allocation =
        static_cast<unsigned char*>(VirtualAlloc(nullptr, allocation_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    CHECK(allocation);
    DWORD old;
    CHECK(VirtualProtect(allocation, page, PAGE_NOACCESS, &old));
    CHECK(VirtualProtect(allocation + page + interior, page, PAGE_NOACCESS, &old));
#else
    allocation = static_cast<unsigned char*>(
        mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(allocation != MAP_FAILED);
    CHECK(mprotect(allocation, page, PROT_NONE) == 0);
    CHECK(mprotect(allocation + page + interior, page, PROT_NONE) == 0);
#endif
    data = at_end ? allocation + page + interior - size : allocation + page;
    std::memset(data, 0xCD, size);
  }
  Guarded(const Guarded&) = delete;
  Guarded& operator=(const Guarded&) = delete;
  ~Guarded() {
#if defined(_WIN32)
    VirtualFree(allocation, 0, MEM_RELEASE);
#else
    munmap(allocation, allocation_size);
#endif
  }
};
template <class T>
void run(const cp_kernels* k, int bits, int width, int step, bool negative) {
  const size_t stride = size_t(width - 1) * step * sizeof(T) + sizeof(T), span = stride * 2;
  Guarded a(span, !negative), b(span), m(span, !negative), out(span);
  const ptrdiff_t pitch = negative ? -ptrdiff_t(stride) : ptrdiff_t(stride);
  const size_t origin = negative ? stride : 0;
  cp_const_plane av = {a.data + origin, pitch, step * ptrdiff_t(sizeof(T))};
  cp_const_plane bv = {b.data + origin, pitch, step * ptrdiff_t(sizeof(T))};
  cp_const_plane mv = {m.data + origin, pitch, step * ptrdiff_t(sizeof(T))};
  cp_plane ov = {out.data + origin, pitch, step * ptrdiff_t(sizeof(T))};
  cp_format f = {bits == 8 ? CP_U8 : bits == 32 ? CP_F32 : CP_U16, bits};
  cp_rows r = {width, 2, 0, 2};
  const T max = T(bits == 32 ? 1 : (1u << bits) - 1);
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < width; ++x) {
      const size_t offset = y * stride + size_t(x) * step * sizeof(T);
      const T v = T((x % 3) * max / 2), alpha = T(x % 2 ? max : 0);
      std::memcpy(a.data + offset, &v, sizeof(T));
      std::memcpy(b.data + offset, &max, sizeof(T));
      std::memcpy(m.data + offset, &alpha, sizeof(T));
    }
  CHECK(k->copy(f, av, ov, r) == CP_OK);
  CHECK(k->fill(f, ov, r, max) == CP_OK);
  if (bits != 32)
    CHECK(k->blend_compat(f, av, bv, &mv, ov, r, 255) == CP_OK);
  for (int op = CP_MIX; op <= CP_DIFFERENCE; ++op) {
    cp_plane_config c = {f, op, .5, 0, double(max), 0, 0, 1, CP_WEIGHT_CODE};
    CHECK(k->process_plane(&c, av, bv, &mv, &av, &bv, ov, r) == CP_OK);
  }
  for (int operation : {CP_PRODUCT, CP_ADD, CP_SUBTRACT, CP_INVERT_MIX, CP_DIFFERENCE}) {
    const cp_plane_config arithmetic{f, operation, .625, 0, double(max), double(max / 2), 0, 0, CP_WEIGHT_CONTINUOUS};
    CHECK(k->process_plane(&arithmetic, av, bv, nullptr, nullptr, nullptr, ov, r) == CP_OK);
  }
  CHECK(k->affine(f, av, ov, r, -1, max) == CP_OK);
  const cp_plane_config quantized{f, CP_MIX, .17, 0, 0, 0, 0, 0, CP_WEIGHT_CONTINUOUS};
  CHECK(k->process_plane(&quantized, av, bv, nullptr, nullptr, nullptr, ov, r) == CP_OK);
  CHECK(k->process_plane(&quantized, av, bv, &mv, nullptr, nullptr, ov, r) == CP_OK);
  const cp_plane_config masked_invert{f, CP_INVERT_MIX, .17, 0, double(max), 0, 0, 0, CP_WEIGHT_CONTINUOUS};
  CHECK(k->process_plane(&masked_invert, av, bv, &mv, nullptr, nullptr, ov, r) == CP_OK);
  CHECK(k->clamp(f, av, ov, r, 0, max) == CP_OK);
  CHECK(k->rgb_luma(f, {av, bv, av}, ov, r, CP_LUMA_NEAREST) == CP_OK);
  const double key[] = {double(max), double(max), double(max)}, tol[] = {.5, .5, .5};
  CHECK(k->color_key(f, {av, bv, av}, mv, ov, r, key, tol) == CP_OK);
  for (int sx : {1, 2, 4})
    for (int placement : {CP_CENTER, CP_MPEG2, CP_TOP_LEFT}) {
      cp_sampling s = {width, 2, sx, sx == 2 ? 2 : 1, placement, -1, 0};
      CHECK(k->resample_mask(f, av, ov, &s, r) == CP_OK);
    }
  Guarded u(span), v(span);
  cp_plane uv = {u.data + origin, pitch, step * ptrdiff_t(sizeof(T))},
           vv = {v.data + origin, pitch, step * ptrdiff_t(sizeof(T))};
  cp_const_yuv masks = {mv, mv, mv};
  for (int op = CP_YUV_ADD; op <= CP_YUV_MULTIPLY; ++op) {
    if (bits == 32 && op > CP_YUV_SUBTRACT && op != CP_YUV_MULTIPLY)
      continue;
    cp_yuv_config c = {f, op, .63};
    CHECK(k->process_yuv(&c, {av, av, av}, {bv, bv, bv}, &masks, {ov, uv, vv}, r) == CP_OK);
  }
  // No neighboring interleaved channel may be written during scatter stores.
  for (const auto* p : {out.data, u.data, v.data})
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < width - 1; ++x)
        for (size_t gap = sizeof(T); gap < size_t(step) * sizeof(T); ++gap)
          CHECK(p[y * stride + size_t(x) * step * sizeof(T) + gap] == 0xCD);
}
// Full interior box rows end exactly at a guard page, including SIMD tails.
template <class T>
void box_sampling(const cp_kernels* k, int bits, int width, bool negative, int step = 1) {
  const size_t row_bytes = (size_t(width * 2 - 1) * step + 1) * sizeof(T);
  const size_t output_samples = size_t(width - 1) * step + 1;
  Guarded source(row_bytes * 2, !negative), out(output_samples * sizeof(T));
  std::memset(out.data, 0, out.size);
  const T maximum = bits == 32 ? T(1) : T((1u << bits) - 1);
  for (int i = 0; i < width * 4; ++i) {
    const T value = i % 5 == 0 ? maximum : T(i % 7);
    const size_t offset = size_t(i / (width * 2)) * row_bytes + size_t(i % (width * 2)) * step * sizeof(T);
    std::memcpy(source.data + offset, &value, sizeof(T));
  }
  const cp_const_plane input{source.data + (negative ? row_bytes : 0),
                             negative ? -ptrdiff_t(row_bytes) : ptrdiff_t(row_bytes), step * ptrdiff_t(sizeof(T))};
  const cp_plane output{out.data, ptrdiff_t(out.size), step * ptrdiff_t(sizeof(T))};
  std::vector<T> reference(output_samples);
  const cp_plane expected{reference.data(), ptrdiff_t(out.size), step * ptrdiff_t(sizeof(T))};
  const cp_format f{bits == 32 ? CP_F32 : bits == 8 ? CP_U8 : CP_U16, bits};
  for (int vertical : {1, 2}) {
    const cp_sampling sampling{width * 2, 2, 2, vertical, CP_CENTER, 0, 0};
    CHECK(k->resample_mask(f, input, output, &sampling, {width, 1, 0, 1}) == CP_OK);
    CHECK(cp_resample_mask(f, input, expected, &sampling, {width, 1, 0, 1}) == CP_OK);
    CHECK(std::memcmp(out.data, reference.data(), out.size) == 0);
  }
}

template <class T>
void packed_key(const cp_kernels* k, int bits, int width, bool negative, bool bgra) {
  constexpr size_t bytes = sizeof(T), pixel_bytes = 4 * bytes;
  const T maximum = bits == 32 ? T(1) : T((1u << bits) - 1);
  const size_t stride = size_t(width) * pixel_bytes, span = stride * 2;
  Guarded source(span, !negative), out(span);
  for (int i = 0; i < width * 2; ++i) {
    const T pixel[4]{i % 2 ? T(maximum / 2) : maximum, T(maximum / 2), T(maximum / 2), maximum};
    std::memcpy(source.data + size_t(i) * pixel_bytes, pixel, pixel_bytes);
  }
  const size_t origin = negative ? stride : 0;
  const ptrdiff_t pitch = negative ? -ptrdiff_t(stride) : ptrdiff_t(stride);
  const cp_const_plane red{source.data + origin + (bgra ? 2 * bytes : 0), pitch, pixel_bytes},
      green{source.data + origin + bytes, pitch, pixel_bytes},
      blue{source.data + origin + (bgra ? 0 : 2 * bytes), pitch, pixel_bytes},
      alpha{source.data + origin + 3 * bytes, pitch, pixel_bytes};
  const cp_plane destination{out.data + origin + 3 * bytes, pitch, pixel_bytes};
  const double key[]{maximum / 2., maximum / 2., maximum / 2.}, tolerance[]{maximum / 4., maximum / 4., maximum / 4.};
  const cp_format f{bits == 32 ? CP_F32 : bits == 8 ? CP_U8 : CP_U16, bits};
  CHECK(k->color_key(f, {red, green, blue}, alpha, destination, {width, 2, 0, 2}, key, tolerance) == CP_OK);
  for (int i = 0; i < width * 2; ++i) {
    T alpha_value;
    std::memcpy(&alpha_value, out.data + size_t(i) * pixel_bytes + 3 * bytes, bytes);
    CHECK(alpha_value == (i % 2 ? T(0) : maximum));
    for (size_t byte = 0; byte < 3 * bytes; ++byte)
      CHECK(out.data[size_t(i) * pixel_bytes + byte] == 0xCD);
  }
}

// The final fourth channel lies in a no-access page. Every intermediate
// fourth channel is a canary, so masked stores must leave it unchanged.
template <class T>
void packed_yuv(const cp_kernels* k, int bits, int width, bool negative, bool reverse) {
  const size_t stride = size_t(width) * 4 * sizeof(T), span = stride * 2 - sizeof(T);
  Guarded source(span), out(span);
  const double maximum = bits == 32 ? 1.0 : double((1u << bits) - 1);
  for (int i = 0; i < width * 2; ++i)
    for (int p = 0; p < 3; ++p) {
      const T value = T(maximum * ((i * 31 + p * 17) % 257) / 256);
      std::memcpy(source.data + (size_t(i) * 4 + p) * sizeof(T), &value, sizeof(T));
    }
  const size_t origin = negative ? stride : 0;
  const ptrdiff_t pitch = negative ? -ptrdiff_t(stride) : ptrdiff_t(stride);
  const auto input = [&](unsigned char* ptr) {
    return cp_const_yuv{{ptr + origin + (reverse ? 2 : 0) * sizeof(T), pitch, 4 * sizeof(T)},
                        {ptr + origin + sizeof(T), pitch, 4 * sizeof(T)},
                        {ptr + origin + (reverse ? 0 : 2) * sizeof(T), pitch, 4 * sizeof(T)}};
  };
  const auto output = [&](unsigned char* ptr) {
    const auto v = input(ptr);
    return cp_yuv{{const_cast<void*>(v.y.data), v.y.stride, v.y.step},
                  {const_cast<void*>(v.u.data), v.u.stride, v.u.step},
                  {const_cast<void*>(v.v.data), v.v.stride, v.v.step}};
  };
  const auto src = input(source.data);
  const cp_const_yuv masks{src.y, src.y, src.y};
  for (int op = CP_YUV_ADD; op <= CP_YUV_MULTIPLY; ++op) {
    if (bits == 32 && op > CP_YUV_SUBTRACT && op != CP_YUV_MULTIPLY)
      continue;
    for (bool masked : {false, true}) {
      std::vector<unsigned char> ref(source.data, source.data + span);
      std::memcpy(out.data, source.data, span);
      const cp_yuv_config config{{bits == 32 ? CP_F32 : bits == 8 ? CP_U8 : CP_U16, bits}, op, .625};
      CHECK(cp_process_yuv(&config, input(ref.data()), src, masked ? &masks : nullptr, output(ref.data()),
                           {width, 2, 0, 2}) == CP_OK);
      CHECK(k->process_yuv(&config, input(out.data), src, masked ? &masks : nullptr, output(out.data),
                           {width, 2, 0, 2}) == CP_OK);
      if (bits != 32 && op == CP_YUV_MULTIPLY) {
        for (int i = 0; i < width * 2; ++i)
          for (int p = 0; p < 3; ++p) {
            const size_t offset = (size_t(i) * 4 + p) * sizeof(T);
            T actual, expected;
            std::memcpy(&actual, out.data + offset, sizeof(T));
            std::memcpy(&expected, ref.data() + offset, sizeof(T));
            CHECK(std::abs(int(actual) - int(expected)) <= 1);
            std::memcpy(ref.data() + offset, &actual, sizeof(T));
          }
      }
      // All gaps/canaries still compare byte-for-byte.
      CHECK(std::memcmp(ref.data(), out.data, span) == 0);
    }
  }
}

void large_masked_mix(const cp_kernels* k, int width) {
  Guarded a(width), b(width), mask(width), out(width);
  std::vector<unsigned char> reference(width);
  const cp_const_plane pa{a.data, width, 1}, pb{b.data, width, 1}, pm{mask.data, width, 1};
  const cp_plane_config config{{CP_U8, 8}, CP_MIX, .625, 0, 0, 0, 0, 0, CP_WEIGHT_CONTINUOUS};
  const cp_rows rows{width, 1, 0, 1};
  for (unsigned char* destination : {out.data, a.data, b.data, mask.data}) {
    for (int i = 0; i < width; ++i) {
      a.data[i] = static_cast<unsigned char>(i * 29);
      b.data[i] = static_cast<unsigned char>(i * 37 + 81);
      mask.data[i] = static_cast<unsigned char>(i * 73);
    }
    CHECK(cp_process_plane(&config, pa, pb, &pm, nullptr, nullptr, {reference.data(), width, 1}, rows) == CP_OK);
    CHECK(k->process_plane(&config, pa, pb, &pm, nullptr, nullptr, {destination, width, 1}, rows) == CP_OK);
    for (int i = 0; i < width; ++i) {
      // Continuous integer mixing permits one code, including the NEON
      // byte-weight path. Aliasing and inaccessible guard pages stay covered.
      const int tolerance = static_cast<unsigned char>(i * 73) == 0 ? 0 : 1;
      CHECK(std::abs(int(destination[i]) - int(reference[i])) <= tolerance);
    }
  }
}

int main() {
  std::vector<int64_t> targets = {0};
  for (int64_t rest = cp_supported_targets(); rest; rest &= rest - 1)
    targets.push_back(rest & -rest);
  for (auto target : targets)
    for (int width : {65536, 65537, 65543})
      large_masked_mix(cp_get_kernels(target), width);
  for (auto target : targets)
    for (int width : {1, 3, 7, 15, 16, 17, 31, 32, 33, 65, 127})
      for (bool negative : {false, true}) {
        box_sampling<uint8_t>(cp_get_kernels(target), 8, width, negative);
        for (int bits = 9; bits <= 16; ++bits)
          box_sampling<uint16_t>(cp_get_kernels(target), bits, width, negative);
        box_sampling<float>(cp_get_kernels(target), 32, width, negative);
        box_sampling<float>(cp_get_kernels(target), 32, width, negative, 4);
      }
  for (auto target : targets)
    for (int width : {1, 3, 7, 15, 16, 17, 31, 32, 33, 65})
      for (int step : {1, 2, 3, 4})
        for (bool negative : {false, true}) {
          const auto* k = cp_get_kernels(target);
          run<uint8_t>(k, 8, width, step, negative);
          run<uint16_t>(k, 16, width, step, negative);
          run<float>(k, 32, width, step, negative);
        }
  for (auto target : targets)
    for (int width : {64, 65, 95, 96, 127, 128})
      for (bool negative : {false, true})
        for (bool bgra : {false, true}) {
          packed_key<uint8_t>(cp_get_kernels(target), 8, width, negative, bgra);
          packed_key<uint16_t>(cp_get_kernels(target), 16, width, negative, bgra);
          packed_key<float>(cp_get_kernels(target), 32, width, negative, bgra);
        }
  for (auto target : targets)
    for (int width : {63, 64, 65, 127})
      for (bool negative : {false, true})
        for (bool reverse : {false, true}) {
          packed_yuv<uint8_t>(cp_get_kernels(target), 8, width, negative, reverse);
          packed_yuv<uint16_t>(cp_get_kernels(target), 16, width, negative, reverse);
          packed_yuv<float>(cp_get_kernels(target), 32, width, negative, reverse);
        }
  std::puts("guard page tests passed");
}
