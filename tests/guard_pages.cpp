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
  size_t page, size;
  explicit Guarded(size_t bytes, bool at_end = true) : size(bytes) {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    page = info.dwPageSize;
    allocation = static_cast<unsigned char*>(VirtualAlloc(nullptr, page * 3, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    CHECK(allocation);
    DWORD old;
    CHECK(VirtualProtect(allocation, page, PAGE_NOACCESS, &old));
    CHECK(VirtualProtect(allocation + page * 2, page, PAGE_NOACCESS, &old));
#else
    page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    allocation = static_cast<unsigned char*>(
        mmap(nullptr, page * 3, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(allocation != MAP_FAILED);
    CHECK(mprotect(allocation, page, PROT_NONE) == 0);
    CHECK(mprotect(allocation + page * 2, page, PROT_NONE) == 0);
#endif
    CHECK(size <= page);
    data = at_end ? allocation + page * 2 - size : allocation + page;
    std::memset(data, 0xCD, size);
  }
  Guarded(const Guarded&) = delete;
  Guarded& operator=(const Guarded&) = delete;
  ~Guarded() {
#if defined(_WIN32)
    VirtualFree(allocation, 0, MEM_RELEASE);
#else
    munmap(allocation, page * 3);
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
  CHECK(k->affine(f, av, ov, r, -1, max) == CP_OK);
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
      CHECK(std::memcmp(ref.data(), out.data, span) == 0);
    }
  }
}

int main() {
  std::vector<int64_t> targets = {0};
  for (int64_t rest = cp_supported_targets(); rest; rest &= rest - 1)
    targets.push_back(rest & -rest);
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
