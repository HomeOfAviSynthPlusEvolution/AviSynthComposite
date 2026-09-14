// SPDX-License-Identifier: GPL-2.0-or-later
// With the inherited AviSynth linking exception; see LICENSE.
// Explicit target table pattern follows AviSynthConvertAudio/ConvertVideo.
#include "common.h"
#include <type_traits>

#ifndef CP_SCALAR_ONLY
#include <hwy/targets.h>
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace cp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
#if HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128
// Stepped channels never grant access to their neighboring channels. Gather and
// scatter only the declared samples; full contiguous vectors use direct loads.
template <class D>
hn::VFromD<D> LoadChannel(D d, cp_const_plane p, int x, int y, size_t count) {
  using T = hn::TFromD<D>;
  const auto* source = reinterpret_cast<const T*>(address(p, x, y));
  if (p.step == sizeof(T) && count == hn::Lanes(d))
    return hn::LoadU(d, source);
  HWY_ALIGN T values[hn::MaxLanes(d)] = {};
  for (size_t i = 0; i < count; ++i)
    std::memcpy(values + i, address(p, x + static_cast<int>(i), y), sizeof(T));
  return hn::LoadU(d, values);
}
template <class D>
void StoreChannel(hn::VFromD<D> v, D d, cp_plane p, int x, int y, size_t count) {
  using T = hn::TFromD<D>;
  auto* output = reinterpret_cast<T*>(address(p, x, y));
  if (p.step == sizeof(T) && count == hn::Lanes(d)) {
    hn::StoreU(v, d, output);
    return;
  }
  HWY_ALIGN T values[hn::MaxLanes(d)];
  hn::StoreU(v, d, values);
  for (size_t i = 0; i < count; ++i)
    std::memcpy(address(p, x + static_cast<int>(i), y), values + i, sizeof(T));
}
#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
// AVX512BW byte masks suppress accesses to the neighboring stepped channel,
// including the byte immediately beyond the final declared sample.
template <bool average>
int SteppedU8Rows(cp_const_plane a, cp_const_plane b, cp_plane output, cp_rows r, uint32_t weight = 0) {
  if constexpr (!average) {
    if (weight > 16384) {
      std::swap(a, b);
      weight = 32768 - weight;
    }
  }
  const hn::ScalableTag<uint8_t> d;
  const hn::ScalableTag<int16_t> di;
  const size_t n = hn::Lanes(di), width = size_t(r.width);
  const auto even = hn::Eq(hn::And(hn::Iota(d, 0), hn::Set(d, uint8_t(1))), hn::Zero(d));
  for (int y = r.first; y < r.first + r.count; ++y)
    for (size_t x = 0; x < width; x += n) {
      const auto active = hn::And(even, hn::FirstN(d, 2 * std::min(n, width - x)));
      const auto av = hn::MaskedLoad(active, d, address(a, int(x), y));
      const auto bv = hn::MaskedLoad(active, d, address(b, int(x), y));
      auto value = av;
      if constexpr (average)
        value = hn::AverageRound(av, bv);
      else {
        const auto base = hn::BitCast(di, av), diff = hn::Sub(hn::BitCast(di, bv), base);
        value = hn::BitCast(d, hn::Add(base, hn::MulFixedPoint15(diff, hn::Set(di, int16_t(weight)))));
      }
      hn::BlendedStore(value, active, d, address(output, int(x), y));
    }
  return CP_OK;
}
#endif

// Isolate the contiguous loop from stepped-channel staging and descriptor reloads.
HWY_NOINLINE int AverageU8Contiguous(const uint8_t* ap, const uint8_t* bp, uint8_t* dst, ptrdiff_t a_stride,
                                     ptrdiff_t b_stride, ptrdiff_t dst_stride, int width, int count) {
  const hn::ScalableTag<uint8_t> d;
  const size_t n = hn::Lanes(d), end = size_t(width) - size_t(width) % n;
  for (int y = 0; y < count; ++y) {
    size_t x = 0;
    for (; x < end; x += n)
      hn::StoreU(hn::AverageRound(hn::LoadU(d, ap + x), hn::LoadU(d, bp + x)), d, dst + x);
    for (; x < size_t(width); ++x)
      dst[x] = uint8_t((unsigned(ap[x]) + bp[x] + 1) >> 1);
    if (y + 1 < count) {
      ap += a_stride;
      bp += b_stride;
      dst += dst_stride;
    }
  }
  return CP_OK;
}

template <class T>
int AverageRows(cp_const_plane a, cp_const_plane b, cp_plane output, cp_rows r) {
  if constexpr (std::is_same<T, uint8_t>::value)
    if (a.step == 1 && b.step == 1 && output.step == 1)
      return AverageU8Contiguous(address(a, 0, r.first), address(b, 0, r.first), address(output, 0, r.first), a.stride,
                                 b.stride, output.stride, r.width, r.count);
#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  if constexpr (std::is_same<T, uint8_t>::value)
    if (a.step == 2 && b.step == 2 && output.step == 2)
      return SteppedU8Rows<true>(a, b, output, r);
#endif
  const hn::ScalableTag<T> d;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width);
  const bool contiguous = a.step == sizeof(T) && b.step == sizeof(T) && output.step == sizeof(T);
  const size_t end = width - width % n;
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    if (contiguous) {
      const auto* ap = reinterpret_cast<const T*>(address(a, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(b, 0, y));
      auto* dst = reinterpret_cast<T*>(address(output, 0, y));
      for (; x < end; x += n)
        hn::StoreU(hn::AverageRound(hn::LoadU(d, ap + x), hn::LoadU(d, bp + x)), d, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      const auto av = LoadChannel(d, a, static_cast<int>(x), y, count);
      const auto bv = LoadChannel(d, b, static_cast<int>(x), y, count);
      StoreChannel(hn::AverageRound(av, bv), d, output, static_cast<int>(x), y, count);
    }
  }
  return CP_OK;
}

// For M=2^bits-1 and 0<=v<=M*M+floor(M/2), floor(v/M) equals
// (v+1+(v>>bits))>>bits. Write v=q*M+r: the inner shift is q-[r<q],
// so the final numerator is q*2^bits+r+1-[r<q], whose remainder
// is in [0, M]. At bits<=16 even the largest numerator fits uint32_t.
// This avoids division and preserves every integer rounding boundary.
template <class D>
hn::VFromD<D> DivideCode(D d, hn::VFromD<D> v, int bits) {
  return hn::ShiftRightSame(hn::Add(hn::Add(v, hn::Set(d, 1)), hn::ShiftRightSame(v, bits)), bits);
}

// Construct vector constants inside arithmetic lambdas from scalar values.
// Captured vector wrappers can make MSVC spill/reassemble their halves in hot loops.
// Exact dyadic weights need no floating-point arithmetic. Convex blends sum
// to at most 65535*32768+16384; Add/Subtract bounds are documented below.
// The MIX dispatcher may also round continuous opacity to Q15 (<=1 LSB).
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES || HWY_ARCH_X86
int MixU8Q15Rows(cp_const_plane a, cp_const_plane b, cp_plane output, cp_rows r, uint32_t weight) {
#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  if (a.step == 2 && b.step == 2 && output.step == 2)
    return SteppedU8Rows<false>(a, b, output, r, weight);
#endif
  // a + round((b-a)*w/32768) is the same Q15 convex blend. Swap inputs
  // so w <= 16384 fits signed lanes; differences are in [-255, 255].
  // MulFixedPoint15 rounds ties upwards, including negative differences.
  if (weight > 16384) {
    std::swap(a, b);
    weight = 32768 - weight;
  }
  const hn::ScalableTag<int16_t> d;
  const hn::Rebind<uint8_t, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
  const bool contiguous = a.step == 1 && b.step == 1 && output.step == 1;
  const auto blend = [&](auto av, auto bv) HWY_ATTR {
    const auto base = hn::PromoteTo(d, av);
    const auto diff = hn::Sub(hn::PromoteTo(d, bv), base);
    return hn::DemoteTo(dt, hn::Add(base, hn::MulFixedPoint15(diff, hn::Set(d, int16_t(weight)))));
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    if (contiguous) {
      const auto* ap = address(a, 0, y);
      const auto* bp = address(b, 0, y);
      auto* dst = address(output, 0, y);
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x)), dt, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(dt, a, int(x), y, count), LoadChannel(dt, b, int(x), y, count)),
                   dt, output, int(x), y, count);
    }
  }
  return CP_OK;
}
#endif

#if HWY_ARCH_X86
int MixU16Q15Rows(cp_const_plane a, cp_const_plane b, cp_plane output, cp_rows r, uint32_t weight,
                  uint32_t maximum_code) {
  const hn::ScalableTag<uint16_t> d;
  const hn::Rebind<int16_t, decltype(d)> di16;
  const hn::Repartition<int32_t, decltype(d)> di32;
  const hn::RebindToUnsigned<decltype(di32)> du32;
  const size_t n = hn::Lanes(d), width = size_t(r.width);
  const auto pivot = hn::Set(d, uint16_t(32768));
  const auto weights = hn::BitCast(di16, hn::Set(du32, ((weight << 16) | (32768 - weight))));
  const auto blend = [&](auto av, auto bv) HWY_ATTR {
    // Quantized endpoints cannot be represented as positive signed-16 weights.
    if (weight == 0 || weight == 32768)
      return hn::Min(weight == 0 ? av : bv, hn::Set(d, uint16_t(maximum_code)));
    // XOR subtracts 32768 from each unsigned sample. Since the two weights sum
    // to 32768, signed pairwise multiply-add followed by the same Q15 rounding
    // produces the biased result without a difference/carry/sign dependency chain.
    const auto as = hn::BitCast(di16, hn::Xor(av, pivot));
    const auto bs = hn::BitCast(di16, hn::Xor(bv, pivot));
    const auto round = hn::Set(di32, 16384);
    const auto lo =
        hn::ShiftRight<15>(hn::Add(hn::WidenMulPairwiseAdd(di32, hn::InterleaveLower(as, bs), weights), round));
    const auto hi =
        hn::ShiftRight<15>(hn::Add(hn::WidenMulPairwiseAdd(di32, hn::InterleaveUpper(di16, as, bs), weights), round));
    // Interleave and reordered demotion have matching 128-bit block order on x86.
    const auto value = hn::Xor(hn::BitCast(d, hn::ReorderDemote2To(di16, lo, hi)), pivot);
    return hn::Min(value, hn::Set(d, uint16_t(maximum_code)));
  };
  const bool contiguous = a.step == 2 && b.step == 2 && output.step == 2;
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    if (contiguous) {
      const auto* ap = reinterpret_cast<const uint16_t*>(address(a, 0, y));
      const auto* bp = reinterpret_cast<const uint16_t*>(address(b, 0, y));
      auto* dst = reinterpret_cast<uint16_t*>(address(output, 0, y));
      for (; x + n <= width; x += n)
        hn::StoreU(blend(hn::LoadU(d, ap + x), hn::LoadU(d, bp + x)), d, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(d, a, int(x), y, count), LoadChannel(d, b, int(x), y, count)), d, output, int(x),
                   y, count);
    }
  }
  return CP_OK;
}
#endif

template <class T, int operation = CP_MIX, int fractional_bits = 15>
int WeightedRows(cp_const_plane a, cp_const_plane b, cp_plane output, cp_rows r, uint32_t weight,
                 uint32_t maximum_code = std::numeric_limits<T>::max(), uint32_t offset = 0) {
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES || HWY_ARCH_X86
  if constexpr (std::is_same<T, uint8_t>::value && operation == CP_MIX && fractional_bits == 15)
    return MixU8Q15Rows(a, b, output, r, weight);
#endif
#if HWY_ARCH_X86
  if constexpr (std::is_same<T, uint16_t>::value && operation == CP_MIX && fractional_bits == 15)
    return MixU16Q15Rows(a, b, output, r, weight, maximum_code);
#endif
  const hn::ScalableTag<uint32_t> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const bool contiguous = a.step == sizeof(T) && b.step == sizeof(T) && output.step == sizeof(T);
  const auto blend = [&](auto av, auto bv) HWY_ATTR {
    const auto w = hn::Set(d, weight), inv = hn::Set(d, (1u << fractional_bits) - weight),
               round = hn::Set(d, 1u << (fractional_bits - 1));
    const auto base = hn::PromoteTo(d, av);
    auto target = hn::PromoteTo(d, bv);
    if constexpr (operation == CP_PRODUCT)
      target = DivideCode(d, hn::Mul(base, target), sizeof(T) * 8);
    if constexpr (operation == CP_INVERT_MIX || operation == CP_DIFFERENCE) {
      // For offset <= 65536, even Difference's largest positive numerator
      // plus rounding is 65535*32768 + 65536*32768 + 16384 < 2^32.
      // Clamp the unsigned subtraction before rounding to avoid underflow.
      const auto positive = hn::Add(operation == CP_DIFFERENCE ? hn::ShiftLeft<15>(base) : hn::Mul(base, inv),
                                    hn::Mul(hn::Set(d, offset), w));
      const auto negative = hn::Mul(target, w);
      const auto value = hn::ShiftRight<fractional_bits>(hn::Add(hn::Sub(hn::Max(positive, negative), negative), round));
      return hn::DemoteTo(dt, hn::Min(value, hn::Set(d, maximum_code)));
    } else if constexpr (operation == CP_SUBTRACT) {
      const hn::Rebind<int32_t, hn::ScalableTag<uint32_t>> di;
      // Both products are <= 65535*32768, so their signed difference and
      // rounding addition fit int32_t, even for noncanonical U16 samples.
      const auto sum =
          hn::Add(hn::BitCast(di, hn::Sub(hn::ShiftLeft<15>(base), hn::Mul(target, w))), hn::Set(di, 16384));
      const auto value = hn::Min(hn::Set(di, int32_t(maximum_code)), hn::Max(hn::Zero(di), hn::ShiftRight<15>(sum)));
      return hn::DemoteTo(dt, value);
    } else {
      auto value = hn::ShiftRight<15>(hn::Add(
          hn::Add(operation == CP_ADD ? hn::ShiftLeft<15>(base) : hn::Mul(base, inv), hn::Mul(target, w)), round));
      if constexpr (operation == CP_ADD || (operation == CP_MIX && sizeof(T) != 1))
        value = hn::Min(value, hn::Set(d, maximum_code));
      return hn::DemoteTo(dt, value);
    }
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    if (contiguous) {
      const auto* ap = reinterpret_cast<const T*>(address(a, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(b, 0, y));
      auto* dst = reinterpret_cast<T*>(address(output, 0, y));
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x)), dt, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(
          blend(LoadChannel(dt, a, static_cast<int>(x), y, count), LoadChannel(dt, b, static_cast<int>(x), y, count)),
          dt, output, static_cast<int>(x), y, count);
    }
  }
  return CP_OK;
}

#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES
template <int operation, bool masked>
HWY_NOINLINE int NarrowContinuousU8(const cp_plane_config* c, cp_const_plane a, cp_const_plane b,
                                    cp_const_plane mask, cp_plane output, cp_rows r) {
  const hn::ScalableTag<uint16_t> d;
  const hn::Rebind<int16_t, decltype(d)> di;
  const hn::Rebind<uint8_t, decltype(d)> dt;
  // Q15 opacity error is <= 1/32768 (including the signed upper clamp).
  // Rounding mask*opacity to one byte adds <= .5/255 to the weight:
  // the U8 output perturbation is <= .5 + 255/32768 < .508 codes.
  const int16_t opacity = int16_t(std::min(32767.0, std::floor(c->opacity * 32768 + .5)));
  const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
  const bool contiguous = a.step == 1 && b.step == 1 && output.step == 1 && (!masked || mask.step == 1);
  const auto blend = [&](auto ac, auto bc, auto mc) HWY_ATTR {
    const auto av = hn::PromoteTo(d, ac);
    auto bv = hn::PromoteTo(d, bc);
    if constexpr (operation == CP_PRODUCT)
      bv = DivideCode(d, hn::Mul(av, bv), 8);
    else if constexpr (operation == CP_INVERT_MIX)
      bv = hn::Sub(hn::Set(d, 255), bv);
    const auto weight = hn::BitCast(d, hn::MulFixedPoint15(hn::PromoteTo(di, mc), hn::Set(di, opacity)));
    const auto sum = hn::Add(hn::Add(hn::Mul(av, hn::Sub(hn::Set(d, 255), weight)), hn::Mul(bv, weight)), hn::Set(d, 127));
    return hn::DemoteTo(dt, DivideCode(d, sum, 8));
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    if (contiguous) {
      const auto* ap = address(a, 0, y); const auto* bp = address(b, 0, y);
      const auto* mp = masked ? address(mask, 0, y) : nullptr; auto* dst = address(output, 0, y);
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), masked ? hn::LoadU(dt, mp + x) : hn::Set(dt, 255)), dt, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(dt, a, int(x), y, count), LoadChannel(dt, b, int(x), y, count),
                         masked ? LoadChannel(dt, mask, int(x), y, count) : hn::Set(dt, 255)), dt, output, int(x), y, count);
    }
  }
  return CP_OK;
}
#endif

// Quantize the combined continuous weight once, not opacity and mask separately.
// For canonical codes, float multiplication plus Q16 rounding contributes less
// than 0.51 output codes at 16 bits. Integer accumulation is exact and bounded
// by 65535*65536+32768 < 2^32, so final rounding differs by at most one code.
template <class T, bool invert>
HWY_NOINLINE int MaskedContinuousRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b,
                                     cp_const_plane mask, cp_plane output, cp_rows r) {
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES
  if constexpr (std::is_same<T, uint8_t>::value)
    return NarrowContinuousU8<invert ? CP_INVERT_MIX : CP_MIX, true>(c, a, b, mask, output, r);
#endif
  const hn::ScalableTag<uint32_t> d;
  const hn::Rebind<T, decltype(d)> dt;
  const hn::Rebind<float, decltype(d)> df;
  const hn::Rebind<int32_t, decltype(d)> di;
  const uint32_t maximum_code = static_cast<uint32_t>(maximum(c->format));
  const float factor = static_cast<float>(c->opacity * 65536 / maximum_code);
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const bool contiguous = a.step == sizeof(T) && b.step == sizeof(T) &&
                          mask.step == sizeof(T) && output.step == sizeof(T);
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto blend = [&](auto ac, auto bc, auto mc, size_t, size_t) HWY_ATTR {
      const auto av = hn::PromoteTo(d, ac), mv = hn::PromoteTo(d, mc);
      auto bv = hn::PromoteTo(d, bc);
      const auto max = hn::Set(d, maximum_code);
      if constexpr (invert)
        bv = hn::Sub(max, bv);
      const auto scaled = hn::Mul(hn::ConvertTo(df, mv), hn::Set(df, factor));
      const auto weight = hn::BitCast(d, hn::ConvertTo(di, hn::Add(scaled, hn::Set(df, .5f))));
      const auto sum = hn::Add(hn::Add(hn::Mul(av, hn::Sub(hn::Set(d, 65536), weight)), hn::Mul(bv, weight)),
                               hn::Set(d, 32768));
      return hn::DemoteTo(dt, hn::ShiftRight<16>(sum));
    };
    size_t x = 0;
    if (contiguous) {
      const auto* ap = reinterpret_cast<const T*>(address(a, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(b, 0, y));
      const auto* mp = reinterpret_cast<const T*>(address(mask, 0, y));
      auto* dst = reinterpret_cast<T*>(address(output, 0, y));
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), hn::LoadU(dt, mp + x), x, n), dt, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(dt, a, int(x), y, count), LoadChannel(dt, b, int(x), y, count),
                           LoadChannel(dt, mask, int(x), y, count), x, count), dt, output, int(x), y, count);
    }
  }
  return CP_OK;
}

// Preserve the floored product before quantizing the continuous blend weight.
template <class T, bool masked>
HWY_NOINLINE int ProductContinuousRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b,
                                     cp_const_plane mask, cp_plane output, cp_rows r) {
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES
  if constexpr (std::is_same<T, uint8_t>::value)
    return NarrowContinuousU8<CP_PRODUCT, masked>(c, a, b, mask, output, r);
#endif
  const hn::ScalableTag<uint32_t> d;
  const hn::Rebind<T, decltype(d)> dt;
  const hn::Rebind<float, decltype(d)> df;
  const hn::Rebind<int32_t, decltype(d)> di;
  const uint32_t maximum_code = static_cast<uint32_t>(maximum(c->format));
  const float factor = static_cast<float>(c->opacity * 65536 / maximum_code);
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const bool contiguous = a.step == sizeof(T) && b.step == sizeof(T) &&
                          (!masked || mask.step == sizeof(T)) && output.step == sizeof(T);
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto blend = [&](auto ac, auto bc, auto mc, size_t, size_t) HWY_ATTR {
      const auto av = hn::PromoteTo(d, ac), mv = hn::PromoteTo(d, mc);
      auto bv = hn::PromoteTo(d, bc);
      bv = DivideCode(d, hn::Mul(av, bv), c->format.bits);
      const auto scaled = hn::Mul(hn::ConvertTo(df, mv), hn::Set(df, factor));
      const auto weight = hn::BitCast(d, hn::ConvertTo(di, hn::Add(scaled, hn::Set(df, .5f))));
      const auto sum = hn::Add(hn::Add(hn::Mul(av, hn::Sub(hn::Set(d, 65536), weight)), hn::Mul(bv, weight)),
                               hn::Set(d, 32768));
      return hn::DemoteTo(dt, hn::ShiftRight<16>(sum));
    };
    size_t x = 0;
    if (contiguous) {
      const auto* ap = reinterpret_cast<const T*>(address(a, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(b, 0, y));
      const auto* mp = masked ? reinterpret_cast<const T*>(address(mask, 0, y)) : nullptr;
      auto* dst = reinterpret_cast<T*>(address(output, 0, y));
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), masked ? hn::LoadU(dt, mp + x) : hn::Set(dt, T(maximum_code)), x, n), dt, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(dt, a, int(x), y, count), LoadChannel(dt, b, int(x), y, count),
                           masked ? LoadChannel(dt, mask, int(x), y, count) : hn::Set(dt, T(maximum_code)), x, count), dt, output, int(x), y, count);
    }
  }
  return CP_OK;
}

// Vector-backed comparison masks contain only zero or all-one lanes. Truncating
// their bits is exact and avoids the general saturating i64 demotion sequence.
// Targets with compact mask registers retain Highway's native mask conversion.
template <class DTo, class DFrom, class M>
HWY_INLINE auto NarrowMask(DTo to, DFrom from, M mask) {
#if HWY_TARGET == HWY_AVX2 || HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
  const hn::RebindToUnsigned<DTo> du_to;
  const hn::RebindToUnsigned<DFrom> du_from;
  return hn::MaskFromVec(hn::BitCast(to, hn::TruncateTo(du_to, hn::BitCast(du_from, hn::VecFromMask(from, mask)))));
#else
  return hn::DemoteMaskTo(to, from, mask);
#endif
}

// Bounded continuous guided blend; extrapolation retains the reference path.
template <class T, bool masked>
HWY_NOINLINE int GuidedContinuousRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b,
                                     cp_const_plane mask, cp_plane output, cp_rows r) {
  const hn::ScalableTag<uint32_t> d;
  const hn::Rebind<T, decltype(d)> dt;
  const hn::Rebind<float, decltype(d)> df;
  const hn::Rebind<int32_t, decltype(d)> di;
  const uint32_t maximum_code = static_cast<uint32_t>(maximum(c->format));
  const float factor = static_cast<float>(1.0 / maximum_code);
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const bool contiguous = a.step == sizeof(T) && b.step == sizeof(T) &&
                          (!masked || mask.step == sizeof(T)) && output.step == sizeof(T);
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto blend = [&](auto ac, auto bc, auto mc, size_t, size_t) HWY_ATTR {
      const auto av = hn::PromoteTo(d, ac), mv = hn::PromoteTo(d, mc);
      auto bv = hn::PromoteTo(d, bc);
      // With canonical codes, bounded neutral and interior opacity, all
      // magnitudes are <= M. A conservative 32*float_epsilon*M error bound
      // is < 0.25 code through 16 bits, hence final rounding differs <= 1.
      const auto af = hn::ConvertTo(df, av), bf = hn::ConvertTo(df, bv);
      const auto neutral = hn::Set(df, static_cast<float>(c->neutral));
      const auto inv = hn::Set(df, factor), one = hn::Set(df, 1);
      const auto w = masked ? hn::Mul(hn::Set(df, static_cast<float>(c->opacity)), hn::Mul(hn::ConvertTo(df, mv), inv))
                            : hn::Set(df, static_cast<float>(c->opacity));
      const auto scale = hn::Sub(one, hn::Mul(w, hn::Sub(one, hn::Mul(bf, inv))));
      auto result = hn::Add(neutral, hn::Mul(hn::Sub(af, neutral), scale));
      result = hn::Min(hn::Set(df, static_cast<float>(maximum_code)), hn::Max(hn::Zero(df), result));
      const auto code = hn::DemoteTo(dt, hn::ConvertTo(di, hn::Add(result, hn::Set(df, .5f))));
      return hn::IfThenElse(NarrowMask(dt, d, hn::Eq(mv, hn::Zero(d))), ac, code);
    };
    size_t x = 0;
    if (contiguous) {
      const auto* ap = reinterpret_cast<const T*>(address(a, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(b, 0, y));
      const auto* mp = masked ? reinterpret_cast<const T*>(address(mask, 0, y)) : nullptr;
      auto* dst = reinterpret_cast<T*>(address(output, 0, y));
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), masked ? hn::LoadU(dt, mp + x) : hn::Set(dt, T(maximum_code)), x, n), dt, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(dt, a, int(x), y, count), LoadChannel(dt, b, int(x), y, count),
                           masked ? LoadChannel(dt, mask, int(x), y, count) : hn::Set(dt, T(maximum_code)), x, count), dt, output, int(x), y, count);
    }
  }
  return CP_OK;
}

// Quantize bounded affine deltas once; clip only the final sample.
template <class T, bool masked, int operation>
HWY_NOINLINE int ArithmeticContinuousRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b,
                                     cp_const_plane mask, cp_plane output, cp_rows r) {
  const hn::ScalableTag<uint32_t> d;
  const hn::Rebind<T, decltype(d)> dt;
  const hn::Rebind<float, decltype(d)> df;
  const hn::Rebind<int32_t, decltype(d)> di;
  const uint32_t maximum_code = static_cast<uint32_t>(maximum(c->format));
  const float factor = static_cast<float>(c->opacity * 65536 / maximum_code);
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const bool contiguous = a.step == sizeof(T) && b.step == sizeof(T) &&
                          (!masked || mask.step == sizeof(T)) && output.step == sizeof(T);
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto blend = [&](auto ac, auto bc, auto mc, size_t, size_t) HWY_ATTR {
      const auto av = hn::PromoteTo(d, ac), mv = hn::PromoteTo(d, mc);
      auto bv = hn::PromoteTo(d, bc);
      const auto max = hn::Set(d, maximum_code);
      auto negative = hn::Eq(hn::Zero(d), hn::Set(d, 1));
      if constexpr (operation == CP_SUBTRACT)
        negative = hn::Eq(hn::Zero(d), hn::Zero(d));
      if constexpr (operation == CP_DIFFERENCE) {
        const auto bias = hn::Set(d, static_cast<uint32_t>(c->bias));
        negative = hn::Lt(bias, bv);
        bv = hn::IfThenElse(negative, hn::Sub(bv, bias), hn::Sub(bias, bv));
      }
      const auto scaled = hn::Mul(hn::ConvertTo(df, mv), hn::Set(df, factor));
      const auto weight = hn::BitCast(d, hn::ConvertTo(di, hn::Add(scaled, hn::Set(df, .5f))));
      // A negative half tie rounds toward +infinity, so its magnitude
      // rounds down. Both numerators fit uint32_t through 16 bits.
      const auto rounding = hn::IfThenElse(negative, hn::Set(d, 32767), hn::Set(d, 32768));
      const auto delta = hn::ShiftRight<16>(hn::Add(hn::Mul(bv, weight), rounding));
      const auto lower = hn::IfThenElse(hn::Lt(av, delta), hn::Zero(d), hn::Sub(av, delta));
      const auto upper = hn::Min(max, hn::Add(av, delta));
      return hn::DemoteTo(dt, hn::IfThenElse(negative, lower, upper));
    };
    size_t x = 0;
    if (contiguous) {
      const auto* ap = reinterpret_cast<const T*>(address(a, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(b, 0, y));
      const auto* mp = masked ? reinterpret_cast<const T*>(address(mask, 0, y)) : nullptr;
      auto* dst = reinterpret_cast<T*>(address(output, 0, y));
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), masked ? hn::LoadU(dt, mp + x) : hn::Set(dt, T(maximum_code)), x, n), dt, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(dt, a, int(x), y, count), LoadChannel(dt, b, int(x), y, count),
                           masked ? LoadChannel(dt, mask, int(x), y, count) : hn::Set(dt, T(maximum_code)), x, count), dt, output, int(x), y, count);
    }
  }
  return CP_OK;
}

template <class T, bool product, bool masked, bool guided = false, int fixed_bits = 0>
int CodeRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask, cp_plane output,
             cp_rows r) {
  if constexpr (guided && sizeof(T) == 2 && fixed_bits == 0) {
    if (c->format.bits == 10) return CodeRows<T, product, masked, guided, 10>(c, a, b, mask, output, r);
    if (c->format.bits == 16) return CodeRows<T, product, masked, guided, 16>(c, a, b, mask, output, r);
  }
  // 8-bit sums including the exact division correction fit in uint16_t.
  // Retaining narrow lanes doubles the number of samples processed per vector.
  using Acc = typename std::conditional<sizeof(T) == 1, uint16_t, uint32_t>::type;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const int bits = sizeof(T) == 1 ? 8 : fixed_bits ? fixed_bits : c->format.bits;
  const uint32_t maximum = (1u << bits) - 1;
  const Acc level_value = static_cast<Acc>(std::floor(c->opacity * maximum + .5));
  const bool contiguous =
      a.step == sizeof(T) && b.step == sizeof(T) && output.step == sizeof(T) && (!masked || mask->step == sizeof(T));
  const auto blend = [&](auto ac, auto bc, auto mc) HWY_ATTR {
    const auto max = hn::Set(d, static_cast<Acc>(maximum)), half = hn::Set(d, static_cast<Acc>(maximum / 2));
    const auto level = hn::Set(d, level_value);
    const auto av = hn::PromoteTo(d, ac);
    auto bv = hn::PromoteTo(d, bc), weight = level;
    if constexpr (masked) {
      const auto mv = hn::PromoteTo(d, mc);
      weight = c->opacity == 1 ? mv : DivideCode(d, hn::Add(hn::Mul(mv, level), half), bits);
    }
    if constexpr (guided) {
      // Code-domain guided multiply first quantizes its darkening weight.
      // Both divisions have odd denominators, so binary64 evaluation cannot
      // cross a rounding midpoint for these bounded integral numerators.
      weight = DivideCode(d, hn::Add(hn::Mul(weight, hn::Sub(max, bv)), half), bits);
      bv = hn::Set(d, static_cast<Acc>(c->neutral));
    }
    if constexpr (product)
      bv = DivideCode(d, hn::Mul(av, bv), bits);
    const auto sum = hn::Add(hn::Add(hn::Mul(av, hn::Sub(max, weight)), hn::Mul(bv, weight)), half);
    return hn::DemoteTo(dt, DivideCode(d, sum, bits));
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    if (contiguous) {
      const auto* ap = reinterpret_cast<const T*>(address(a, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(b, 0, y));
      const auto* mp = masked ? reinterpret_cast<const T*>(address(*mask, 0, y)) : nullptr;
      auto* dst = reinterpret_cast<T*>(address(output, 0, y));
      if constexpr (sizeof(T) == 1) {
        for (; x + 2 * n <= end; x += 2 * n) {
          const auto first = blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x),
                                   masked ? hn::LoadU(dt, mp + x) : hn::Zero(dt));
          const auto second = blend(hn::LoadU(dt, ap + x + n), hn::LoadU(dt, bp + x + n),
                                    masked ? hn::LoadU(dt, mp + x + n) : hn::Zero(dt));
          hn::StoreU(first, dt, dst + x);
          hn::StoreU(second, dt, dst + x + n);
        }
      }
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), masked ? hn::LoadU(dt, mp + x) : hn::Zero(dt)),
                   dt, dst + x);
    }
    if constexpr (sizeof(T) == 1) {
      for (; x + n <= width; x += n)
        StoreChannel(blend(LoadChannel(dt, a, int(x), y, n), LoadChannel(dt, b, int(x), y, n),
                           masked ? LoadChannel(dt, *mask, int(x), y, n) : hn::Zero(dt)),
                     dt, output, int(x), y, n);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(dt, a, static_cast<int>(x), y, count),
                         LoadChannel(dt, b, static_cast<int>(x), y, count),
                         masked ? LoadChannel(dt, *mask, static_cast<int>(x), y, count) : hn::Zero(dt)),
                   dt, output, static_cast<int>(x), y, count);
    }
  }
  return CP_OK;
}
template <class T, int fixed_bits = 0>
int CodePlaneRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
                  cp_plane output, cp_rows r) {
  if constexpr (sizeof(T) == 2 && fixed_bits == 0) {
    if (c->format.bits == 10)
      return CodePlaneRows<T, 10>(c, a, b, mask, output, r);
    if (c->format.bits == 16)
      return CodePlaneRows<T, 16>(c, a, b, mask, output, r);
  }
  if (c->operation == CP_PRODUCT)
    return mask ? CodeRows<T, true, true, false, fixed_bits>(c, a, b, mask, output, r)
                : CodeRows<T, true, false, false, fixed_bits>(c, a, b, mask, output, r);
  return mask ? CodeRows<T, false, true, false, fixed_bits>(c, a, b, mask, output, r)
              : CodeRows<T, false, false, false, fixed_bits>(c, a, b, mask, output, r);
}

#if HWY_HAVE_FLOAT64
template <class T, class D>
hn::VFromD<D> LoadDouble(D d, cp_const_plane p, int x, int y, size_t count) {
  const hn::Rebind<T, D> dt;
  if constexpr (std::is_same<T, float>::value)
    return hn::PromoteTo(d, LoadChannel(dt, p, x, y, count));
  else {
    const hn::Rebind<uint32_t, D> du;
    const hn::Rebind<float, D> df;
    return hn::PromoteTo(d, hn::ConvertTo(df, hn::PromoteTo(du, LoadChannel(dt, p, x, y, count))));
  }
}
// Continuous integer blends, including non-dyadic opacity and masks. Preserve
// double division/multiplication order and final rounding exactly as the C path.
template <class T, int operation, bool masked>
int IntegerBlendRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
                     cp_plane output, cp_rows r) {
  const hn::ScalableTag<double> d;
  const hn::Rebind<T, decltype(d)> dt;
  const hn::Rebind<uint32_t, decltype(d)> du;
  const hn::Rebind<int32_t, decltype(d)> di;
  const hn::Rebind<float, decltype(d)> df;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width);
  const double maximum_value = maximum(c->format), opacity_value = c->opacity, neutral_value = c->neutral;
  const auto promote = [&](auto v) HWY_ATTR {
    return hn::PromoteTo(d, hn::ConvertTo(df, hn::PromoteTo(du, v)));
  };
  const auto blend = [&](auto ac, auto bc, auto mc) HWY_ATTR {
    const auto max = hn::Set(d, maximum_value), zero = hn::Zero(d), half = hn::Set(d, .5);
    const auto opacity = hn::Set(d, opacity_value), neutral = hn::Set(d, neutral_value);
    const auto av = promote(ac), bv = promote(bc);
    const auto w = masked ? hn::Mul(opacity, hn::Div(promote(mc), max)) : opacity;
    auto target = bv;
    if constexpr (operation == CP_GUIDED_MULTIPLY)
      target = hn::Add(neutral, hn::Div(hn::Mul(hn::Sub(av, neutral), bv), max));
    auto v = hn::Add(av, hn::Mul(hn::Sub(target, av), w));
    if constexpr (operation == CP_GUIDED_MULTIPLY) {
      v = hn::IfThenElse(hn::Eq(w, hn::Set(d, 1)), target, v);
      v = hn::IfThenElse(hn::IsNaN(v), zero, v);
    }
    v = hn::Min(max, hn::Max(zero, v));
    auto code = hn::DemoteTo(dt, hn::ConvertTo(di, hn::DemoteTo(df, hn::Floor(hn::Add(v, half)))));
    // Preserve exact copy endpoints for valid input samples.
    if constexpr (operation == CP_MIX)
      code = hn::IfThenElse(NarrowMask(dt, d, hn::Eq(w, hn::Set(d, 1))), bc, code);
    return hn::IfThenElse(NarrowMask(dt, d, hn::Eq(w, zero)), ac, code);
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto* ap = reinterpret_cast<const T*>(address(a, 0, y));
    const auto* bp = reinterpret_cast<const T*>(address(b, 0, y));
    const auto* mp = masked ? reinterpret_cast<const T*>(address(*mask, 0, y)) : nullptr;
    auto* dst = reinterpret_cast<T*>(address(output, 0, y));
    size_t x = 0;
    for (; x + n <= width; x += n)
      hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), masked ? hn::LoadU(dt, mp + x) : hn::Zero(dt)), dt,
                 dst + x);
    if (x < width) {
      const size_t count = width - x;
      StoreChannel(blend(LoadChannel(dt, a, static_cast<int>(x), y, count),
                         LoadChannel(dt, b, static_cast<int>(x), y, count),
                         masked ? LoadChannel(dt, *mask, static_cast<int>(x), y, count) : hn::Zero(dt)),
                   dt, output, static_cast<int>(x), y, count);
    }
  }
  return CP_OK;
}

// Common float blends retain reference double arithmetic outside the documented
// finite-input fast paths. Layout and
// operation dispatch happen before the row loop, not once per vector.
template <int operation, bool masked, bool full_product = false, int center_mode = 0>
int FloatBlendRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
                   cp_plane output, cp_rows r) {
  const hn::ScalableTag<double> d;
  const hn::Rebind<float, decltype(d)> df;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width);
  const double opacity_value = c->opacity, neutral_value = center_mode == 1 ? 0.0 : center_mode == 2 ? .5 : c->neutral, bias_value = c->bias, inversion_value = c->inversion_sum;
  const auto blend = [&](auto af, auto bf, auto mf) HWY_ATTR {
    const auto zero = hn::Zero(d), one = hn::Set(d, 1), opacity = hn::Set(d, opacity_value);
    const auto neutral = hn::Set(d, neutral_value);
    const auto av = hn::PromoteTo(d, af), bv = hn::PromoteTo(d, bf);
    const auto w = masked ? hn::Mul(opacity, hn::PromoteTo(d, mf)) : opacity;
    auto target = bv;
    if constexpr (operation == CP_PRODUCT)
      target = hn::Mul(av, bv);
    else if constexpr (operation == CP_INVERT_MIX)
      target = hn::Sub(hn::Set(d, inversion_value), bv);
    else if constexpr (operation == CP_SUBTRACT)
      target = hn::Sub(av, bv);
    else if constexpr (operation == CP_ADD)
      target = hn::Add(av, bv);
    else if constexpr (operation == CP_DIFFERENCE)
      target = hn::Add(hn::Sub(av, bv), hn::Set(d, bias_value));
    else if constexpr (operation == CP_GUIDED_MULTIPLY) {
      target = hn::Add(neutral, hn::Mul(hn::Sub(av, neutral), bv));
#if defined(_MSC_VER) && !defined(__clang__)
      // MSVC can fold away the +0 neutral even under /fp:strict. The scalar
      // expression adds +0, so an exact zero target must have a positive sign.
      if constexpr (center_mode == 1)
        target = hn::IfThenElse(hn::Eq(target, zero), zero, target);
#endif
    }
    // Unmasked MIX copy endpoints were already handled by Plane.
    if constexpr ((operation == CP_PRODUCT || operation == CP_GUIDED_MULTIPLY) && !masked) {
      // Plane handles zero opacity and dispatches the full-product endpoint.
      // Keep invariant endpoint comparisons and selections out of the hot loop.
      if constexpr (full_product)
        return hn::DemoteTo(df, target);
      return hn::DemoteTo(df, hn::Add(av, hn::Mul(hn::Sub(target, av), opacity)));
    } else if constexpr (operation == CP_MIX && !masked) {
      return hn::DemoteTo(df, hn::Add(av, hn::Mul(hn::Sub(bv, av), opacity)));
    } else {
      auto v = hn::DemoteTo(df, hn::IfThenElse(hn::Eq(w, one), target, hn::Add(av, hn::Mul(hn::Sub(target, av), w))));
      if constexpr (operation == CP_MIX)
        v = hn::IfThenElse(NarrowMask(df, d, hn::Eq(w, one)), bf, v);
      return hn::IfThenElse(NarrowMask(df, d, hn::Eq(w, zero)), af, v);
    }
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto* ap = reinterpret_cast<const float*>(address(a, 0, y));
    const auto* bp = reinterpret_cast<const float*>(address(b, 0, y));
    const auto* mp = masked ? reinterpret_cast<const float*>(address(*mask, 0, y)) : nullptr;
    auto* dst = reinterpret_cast<float*>(address(output, 0, y));
    size_t x = 0;
    if constexpr ((operation == CP_MIX || operation == CP_PRODUCT) && masked) {
      if (opacity_value > 0 && opacity_value <= 1) {
        const hn::ScalableTag<float> fast;
        const size_t fn = hn::Lanes(fast);
        const auto onef = hn::Set(fast, 1.f), zerof = hn::Zero(fast);
        const auto of = hn::Set(fast, float(opacity_value));
        const auto io = hn::Set(fast, float(1.0 - opacity_value));
        const auto safe = hn::Set(fast, std::numeric_limits<float>::max() *
                                       (1.f - 64 * std::numeric_limits<float>::epsilon()));
        for (; x + fn <= width; x += fn) {
          const auto af = hn::LoadU(fast, ap + x), bf = hn::LoadU(fast, bp + x);
          const auto mf = hn::LoadU(fast, mp + x);
          const auto wf = hn::Mul(of, mf);
          auto result = af;
          auto exact_zero = hn::Eq(af, zerof);
          if constexpr (operation == CP_PRODUCT) {
            // Independent opacity complement preserves near-full weights.
            const auto factor = opacity_value <= .75
                ? hn::Add(onef, hn::Mul(wf, hn::Sub(bf, onef)))
                : hn::Add(hn::Sub(onef, mf), hn::Mul(mf, hn::Add(io, hn::Mul(of, bf))));
            result = hn::Mul(af, factor);
            exact_zero = hn::Or(exact_zero, hn::Eq(factor, zerof));
          } else {
            if (opacity_value <= .75)
              result = hn::Add(af, hn::Mul(hn::Sub(bf, af), wf));
            else {
              const auto keep = hn::Add(hn::Sub(onef, mf), hn::Mul(mf, io));
              result = hn::Add(hn::Mul(af, keep), hn::Mul(bf, wf));
            }
          }
          // Nonfinite inputs necessarily propagate to the candidate result.
          auto valid = hn::Le(hn::Abs(result), safe);
          // Preserve the reference subtraction order when near-full blending
          // loses over 20 bits through cancellation of a very large base.
          if (opacity_value > .75 && opacity_value < 1)
            valid = hn::And(valid, hn::Le(hn::Abs(af), hn::Mul(hn::Set(fast, 1048576.f), hn::Max(onef, hn::Abs(result)))));
          if (hn::AllTrue(fast, valid)) {
            if constexpr (operation == CP_PRODUCT)
              result = hn::IfThenElse(hn::And(hn::Eq(result, zerof), exact_zero), zerof, result);
            else if (opacity_value > .75)
              result = hn::IfThenElse(hn::And(hn::Eq(af, zerof), hn::Eq(bf, zerof)), zerof, result);
            if (opacity_value == 1) {
              const auto endpoint = operation == CP_MIX ? bf : hn::Mul(af, bf);
              result = hn::IfThenElse(hn::Eq(mf, onef), endpoint, result);
            }
            hn::StoreU(hn::IfThenElse(hn::Eq(mf, zerof), af, result), fast, dst + x);
          } else {
            for (size_t half = 0; half < fn; half += n)
              hn::StoreU(blend(hn::LoadU(df, ap + x + half), hn::LoadU(df, bp + x + half),
                              hn::LoadU(df, mp + x + half)), df, dst + x + half);
          }
        }
      }
    }
    for (; x + n <= width; x += n)
      hn::StoreU(blend(hn::LoadU(df, ap + x), hn::LoadU(df, bp + x), masked ? hn::LoadU(df, mp + x) : hn::Zero(df)), df,
                 dst + x);
    if (x < width) {
      const size_t count = width - x;
      StoreChannel(blend(LoadChannel(df, a, static_cast<int>(x), y, count),
                         LoadChannel(df, b, static_cast<int>(x), y, count),
                         masked ? LoadChannel(df, *mask, static_cast<int>(x), y, count) : hn::Zero(df)),
                   df, output, static_cast<int>(x), y, count);
    }
  }
  return CP_OK;
}

template <bool masked, int center_mode>
int GuidedFloatRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane guide, const cp_const_plane* mask,
                    cp_plane output, cp_rows r) {
  if constexpr (!masked) {
    if (c->opacity == 1)
      return FloatBlendRows<CP_GUIDED_MULTIPLY, false, true, center_mode>(c, a, guide, nullptr, output, r);
  }
  return FloatBlendRows<CP_GUIDED_MULTIPLY, masked, false, center_mode>(c, a, guide, mask, output, r);
}

template <class T, int fixed_operation = -1>
int PlaneRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
              const cp_const_plane* ga, const cp_const_plane* gb, cp_plane output, cp_rows r) {
  const int operation = fixed_operation < 0 ? c->operation : fixed_operation;
  const hn::ScalableTag<double> d;
  const hn::Rebind<float, decltype(d)> df;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const bool select = operation == CP_SELECT_LIGHTER || operation == CP_SELECT_DARKER;
  const bool guided = select || operation == CP_GUIDED_MULTIPLY;
  const bool code = !std::is_same<T, float>::value && c->weight_rule == CP_WEIGHT_CODE;
  const double maximum = cp::maximum(c->format);
  const auto zero = hn::Zero(d), one = hn::Set(d, 1.0), max = hn::Set(d, maximum), half = hn::Set(d, 0.5);
  const auto neutral = hn::Set(d, c->neutral), opacity = hn::Set(d, c->opacity);
  const auto opacity_code = hn::Set(d, std::floor(c->opacity * maximum + 0.5));
  const auto mix = [&](auto av, auto bv, auto w) HWY_ATTR {
    const auto zero = hn::Zero(d), one = hn::Set(d, 1);
    return hn::IfThenElse(hn::Eq(w, zero), av,
                          hn::IfThenElse(hn::Eq(w, one), bv, hn::Add(av, hn::Mul(hn::Sub(bv, av), w))));
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    for (size_t x = 0; x < end; x += n) {
      const auto av = LoadDouble<T>(d, a, static_cast<int>(x), y, n),
                 bv = LoadDouble<T>(d, b, static_cast<int>(x), y, n);
      const auto mv = mask ? LoadDouble<T>(d, *mask, static_cast<int>(x), y, n) : max;
      auto w = code ? hn::Div(hn::Floor(hn::Add(hn::Div(hn::Mul(mv, opacity_code), max), half)), max)
                    : hn::Mul(opacity, hn::Div(mv, max));
      const auto guide = guided ? LoadDouble<T>(d, *gb, static_cast<int>(x), y, n) : zero;
      if (select) {
        const bool lighter = operation == CP_SELECT_LIGHTER;
        double threshold = lighter ? c->threshold : -c->threshold;
        if constexpr (std::is_same<T, float>::value)
          threshold = static_cast<float>(threshold);
        auto boundary = hn::Add(LoadDouble<T>(d, *ga, static_cast<int>(x), y, n), hn::Set(d, threshold));
        if constexpr (std::is_same<T, float>::value)
          boundary = hn::PromoteTo(d, hn::DemoteTo(df, boundary));
        const auto accepted = lighter ? (c->inclusive ? hn::Ge(guide, boundary) : hn::Gt(guide, boundary))
                                      : (c->inclusive ? hn::Le(guide, boundary) : hn::Lt(guide, boundary));
        w = hn::IfThenElse(accepted, w, zero);
      }
      auto target = bv;
      switch (operation) {
        case CP_ADD:
          target = hn::Add(av, bv);
          break;
        case CP_SUBTRACT:
          target = hn::Sub(av, bv);
          break;
        case CP_PRODUCT:
          target = hn::Div(hn::Mul(av, bv), max);
          if constexpr (!std::is_same<T, float>::value)
            target = hn::Floor(target);
          break;
        case CP_INVERT_MIX:
          target = hn::Sub(hn::Set(d, c->inversion_sum), bv);
          break;
        case CP_GUIDED_MULTIPLY:
          target = hn::Add(neutral, hn::Div(hn::Mul(hn::Sub(av, neutral), guide), max));
          break;
        case CP_DIFFERENCE:
          target = hn::Add(hn::Sub(av, bv), hn::Set(d, c->bias));
          break;
        default:
          break;
      }
      auto result = mix(av, target, w);
      if (code && operation == CP_GUIDED_MULTIPLY) {
        const auto darken = hn::Div(hn::Floor(hn::Add(hn::Mul(w, hn::Sub(max, guide)), half)), max);
        result = mix(av, neutral, darken);
      }
      if constexpr (std::is_same<T, float>::value) {
        auto value = hn::DemoteTo(df, result);
        // Keep source bits for exact copy endpoints (including NaN payloads).
        if (operation == CP_MIX || select)
          value =
              hn::IfThenElse(NarrowMask(df, d, hn::Eq(w, one)), LoadChannel(df, b, static_cast<int>(x), y, n), value);
        value =
            hn::IfThenElse(NarrowMask(df, d, hn::Eq(w, zero)), LoadChannel(df, a, static_cast<int>(x), y, n), value);
        StoreChannel(value, df, output, static_cast<int>(x), y, n);
      } else {
        result = hn::IfThenElse(hn::IsNaN(result), zero, hn::Min(max, hn::Max(zero, result)));
        const auto rounded = hn::DemoteTo(df, hn::Floor(hn::Add(result, half)));
        const hn::Rebind<int32_t, decltype(d)> di;
        StoreChannel(hn::DemoteTo(dt, hn::ConvertTo(di, rounded)), dt, output, static_cast<int>(x), y, n);
      }
    }
  }
  if (end == width)
    return CP_OK;
  // Only complete vectors are loaded; tails retain identical scalar arithmetic.
  const auto shift = [&](cp_const_plane p) HWY_ATTR {
    p.data = address(p, static_cast<int>(end), 0);
    return p;
  };
  a = shift(a);
  b = shift(b);
  output.data = address(output, static_cast<int>(end), 0);
  cp_const_plane m{}, ag{}, bg{};
  if (mask)
    m = shift(*mask);
  if (select)
    ag = shift(*ga);
  if (guided)
    bg = shift(*gb);
  r.width -= static_cast<int>(end);
  return cp_process_plane(c, a, b, mask ? &m : nullptr, select ? &ag : nullptr, guided ? &bg : nullptr, output, r);
}
#endif
int Copy(cp_format f, cp_const_plane source, cp_plane output, cp_rows r);
int Compat(cp_format f, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask, cp_plane output, cp_rows r,
           int opacity);
// Keep offset kernels out of the already large Plane dispatcher so adding
// specializations does not disrupt optimization of its existing stepped loops.
HWY_NOINLINE int OffsetPlaneRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, cp_plane output,
                                 cp_rows r, uint32_t weight, uint32_t offset) {
  const auto max = static_cast<uint32_t>(maximum(c->format));
  if (c->format.storage == CP_U8)
    return c->operation == CP_INVERT_MIX ? WeightedRows<uint8_t, CP_INVERT_MIX>(a, b, output, r, weight, max, offset)
                                         : WeightedRows<uint8_t, CP_DIFFERENCE>(a, b, output, r, weight, max, offset);
  return c->operation == CP_INVERT_MIX ? WeightedRows<uint16_t, CP_INVERT_MIX>(a, b, output, r, weight, max, offset)
                                       : WeightedRows<uint16_t, CP_DIFFERENCE>(a, b, output, r, weight, max, offset);
}
HWY_NOINLINE int QuantizedInvertRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, cp_plane output,
                                     cp_rows r) {
  const auto weight = static_cast<uint32_t>(std::floor(c->opacity * 65536 + .5));
  const auto max = static_cast<uint32_t>(maximum(c->format));
  const auto offset = static_cast<uint32_t>(c->inversion_sum);
  if (c->format.storage == CP_U8)
    return WeightedRows<uint8_t, CP_INVERT_MIX, 16>(a, b, output, r, weight, max, offset);
  return WeightedRows<uint16_t, CP_INVERT_MIX, 16>(a, b, output, r, weight, max, offset);
}
int Plane(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
          const cp_const_plane* ga, const cp_const_plane* gb, cp_plane output, cp_rows r) {
  const int status = check_plane(c, a, b, mask, ga, gb, output, r);
  if (status != CP_OK || !r.count)
    return status;
  if (c->opacity == 0)
    return Copy(c->format, a, output, r);
  if (c->operation == CP_MIX && !mask && c->opacity == 1)
    return Copy(c->format, b, output, r);
  if (c->operation == CP_MIX && !mask && c->opacity == .5 && c->weight_rule == CP_WEIGHT_CONTINUOUS &&
      c->format.storage != CP_F32) {
    if (c->format.storage == CP_U8)
      return AverageRows<uint8_t>(a, b, output, r);
    return AverageRows<uint16_t>(a, b, output, r);
  }
  const double weight32768 = c->opacity * 32768;
  if (c->operation == CP_MIX && !mask && c->weight_rule == CP_WEIGHT_CONTINUOUS && c->format.storage != CP_F32 &&
      weight32768 == std::floor(weight32768)) {
    const auto max = static_cast<uint32_t>(maximum(c->format));
    if (c->format.storage == CP_U8)
      return WeightedRows<uint8_t>(a, b, output, r, static_cast<uint32_t>(weight32768), max);
    return WeightedRows<uint16_t>(a, b, output, r, static_cast<uint32_t>(weight32768), max);
  }
  // Q15 rounding changes a convex blend by at most 65535/65536 < 1
  // code before output rounding. Keep the exact opacity endpoints above;
  // fractional weights which quantize to endpoints still clamp narrow U16.
  if (c->operation == CP_MIX && !mask && c->weight_rule == CP_WEIGHT_CONTINUOUS && c->format.storage != CP_F32) {
    const auto weight = static_cast<uint32_t>(std::floor(weight32768 + .5));
    const auto max = static_cast<uint32_t>(maximum(c->format));
    if (c->format.storage == CP_U8)
      return WeightedRows<uint8_t>(a, b, output, r, weight, max);
    return WeightedRows<uint16_t>(a, b, output, r, weight, max);
  }
  // A floored integer product blended with an exact k/32768 weight is
  // integral arithmetic. Full-range storage keeps both the product division
  // and weighted sum within uint32_t. Narrower U16 formats use the
  // general depth-aware product kernel.
  if (c->operation == CP_PRODUCT && !mask && c->weight_rule == CP_WEIGHT_CONTINUOUS &&
      (c->format.storage == CP_U8 || (c->format.storage == CP_U16 && c->format.bits == 16)) &&
      weight32768 == std::floor(weight32768)) {
    if (c->format.storage == CP_U8)
      return WeightedRows<uint8_t, CP_PRODUCT>(a, b, output, r, static_cast<uint32_t>(weight32768));
    return WeightedRows<uint16_t, CP_PRODUCT>(a, b, output, r, static_cast<uint32_t>(weight32768));
  }
  // Add's largest rounded numerator is 2*65535*32768+16384 < 2^32.
  // Subtract uses a signed difference. The dyadic double expression is exact
  // in either case, and clamping follows the sum, including narrow U16 formats.
  if (c->operation == CP_DIFFERENCE && c->format.storage != CP_F32 &&
      c->weight_rule == CP_WEIGHT_CONTINUOUS && c->bias >= 0 && c->bias <= 65535 && c->bias == std::floor(c->bias)) {
    if (c->format.storage == CP_U8)
      return mask ? ArithmeticContinuousRows<uint8_t, true, CP_DIFFERENCE>(c, a, b, *mask, output, r)
                  : ArithmeticContinuousRows<uint8_t, false, CP_DIFFERENCE>(c, a, b, {}, output, r);
    return mask ? ArithmeticContinuousRows<uint16_t, true, CP_DIFFERENCE>(c, a, b, *mask, output, r)
                : ArithmeticContinuousRows<uint16_t, false, CP_DIFFERENCE>(c, a, b, {}, output, r);
  }
  if ((c->operation == CP_ADD || c->operation == CP_SUBTRACT) && !mask && c->weight_rule == CP_WEIGHT_CONTINUOUS &&
      c->format.storage != CP_F32 && weight32768 == std::floor(weight32768)) {
    const auto max = static_cast<uint32_t>(maximum(c->format));
    const auto weight = static_cast<uint32_t>(weight32768);
    if (c->format.storage == CP_U8)
      return c->operation == CP_ADD ? WeightedRows<uint8_t, CP_ADD>(a, b, output, r, weight, max)
                                    : WeightedRows<uint8_t, CP_SUBTRACT>(a, b, output, r, weight, max);
    return c->operation == CP_ADD ? WeightedRows<uint16_t, CP_ADD>(a, b, output, r, weight, max)
                                  : WeightedRows<uint16_t, CP_SUBTRACT>(a, b, output, r, weight, max);
  }
  if ((c->operation == CP_INVERT_MIX || c->operation == CP_DIFFERENCE) && !mask &&
      c->weight_rule == CP_WEIGHT_CONTINUOUS && c->format.storage != CP_F32 && weight32768 == std::floor(weight32768)) {
    const double offset = c->operation == CP_INVERT_MIX ? c->inversion_sum : c->bias;
    // Include the 65536 chroma inversion sum while keeping all intermediate
    // values bounded. Fractional or larger offsets retain binary64 evaluation.
    if (offset >= 0 && offset <= 65536 && offset == std::floor(offset))
      return OffsetPlaneRows(c, a, b, output, r, static_cast<uint32_t>(weight32768), static_cast<uint32_t>(offset));
  }
  if (c->operation == CP_DIFFERENCE && c->format.storage != CP_F32 &&
      c->weight_rule == CP_WEIGHT_CONTINUOUS && c->bias >= 0 && c->bias <= 65535 && c->bias == std::floor(c->bias)) {
    if (c->format.storage == CP_U8)
      return mask ? ArithmeticContinuousRows<uint8_t, true, CP_DIFFERENCE>(c, a, b, *mask, output, r)
                  : ArithmeticContinuousRows<uint8_t, false, CP_DIFFERENCE>(c, a, b, {}, output, r);
    return mask ? ArithmeticContinuousRows<uint16_t, true, CP_DIFFERENCE>(c, a, b, *mask, output, r)
                : ArithmeticContinuousRows<uint16_t, false, CP_DIFFERENCE>(c, a, b, {}, output, r);
  }
  if ((c->operation == CP_ADD || c->operation == CP_SUBTRACT) &&
      c->format.storage != CP_F32 && c->weight_rule == CP_WEIGHT_CONTINUOUS) {
    if (c->format.storage == CP_U8) {
      if (c->operation == CP_ADD)
        return mask ? ArithmeticContinuousRows<uint8_t, true, CP_ADD>(c, a, b, *mask, output, r)
                    : ArithmeticContinuousRows<uint8_t, false, CP_ADD>(c, a, b, {}, output, r);
      return mask ? ArithmeticContinuousRows<uint8_t, true, CP_SUBTRACT>(c, a, b, *mask, output, r)
                  : ArithmeticContinuousRows<uint8_t, false, CP_SUBTRACT>(c, a, b, {}, output, r);
    }
    if (c->operation == CP_ADD)
      return mask ? ArithmeticContinuousRows<uint16_t, true, CP_ADD>(c, a, b, *mask, output, r)
                  : ArithmeticContinuousRows<uint16_t, false, CP_ADD>(c, a, b, {}, output, r);
    return mask ? ArithmeticContinuousRows<uint16_t, true, CP_SUBTRACT>(c, a, b, *mask, output, r)
                : ArithmeticContinuousRows<uint16_t, false, CP_SUBTRACT>(c, a, b, {}, output, r);
  }
  if (c->operation == CP_PRODUCT && c->format.storage != CP_F32 &&
      c->weight_rule == CP_WEIGHT_CONTINUOUS) {
    if (c->format.storage == CP_U8)
      return mask ? ProductContinuousRows<uint8_t, true>(c, a, b, *mask, output, r)
                  : ProductContinuousRows<uint8_t, false>(c, a, b, {}, output, r);
    return mask ? ProductContinuousRows<uint16_t, true>(c, a, b, *mask, output, r)
                : ProductContinuousRows<uint16_t, false>(c, a, b, {}, output, r);
  }
  if (mask && c->format.storage != CP_F32 && c->weight_rule == CP_WEIGHT_CONTINUOUS &&
      (c->operation == CP_MIX || (c->operation == CP_INVERT_MIX && c->inversion_sum == maximum(c->format)))) {
    if (c->format.storage == CP_U8)
      return c->operation == CP_MIX ? MaskedContinuousRows<uint8_t, false>(c, a, b, *mask, output, r)
                                    : MaskedContinuousRows<uint8_t, true>(c, a, b, *mask, output, r);
    return c->operation == CP_MIX ? MaskedContinuousRows<uint16_t, false>(c, a, b, *mask, output, r)
                                  : MaskedContinuousRows<uint16_t, true>(c, a, b, *mask, output, r);
  }
  // Q16 bounds the error even when offset-a-b spans [-131070,65535].
  // The positive convex sum and its rounding fit uint32_t for offset<=65535.
  if (c->operation == CP_INVERT_MIX && !mask && c->weight_rule == CP_WEIGHT_CONTINUOUS &&
      c->format.storage != CP_F32 && c->inversion_sum >= 0 && c->inversion_sum <= 65535 &&
      c->inversion_sum == std::floor(c->inversion_sum))
    return QuantizedInvertRows(c, a, b, output, r);

  if (c->format.storage != CP_F32 && (c->weight_rule == CP_WEIGHT_CODE || c->opacity == 1) &&
      (c->operation == CP_MIX || c->operation == CP_PRODUCT)) {
    if (c->format.storage == CP_U8)
      return CodePlaneRows<uint8_t>(c, a, b, mask, output, r);
    return CodePlaneRows<uint16_t>(c, a, b, mask, output, r);
  }
  if (c->format.storage != CP_F32 && c->operation == CP_GUIDED_MULTIPLY &&
      c->weight_rule == CP_WEIGHT_CONTINUOUS && c->opacity > 0 && c->opacity < 1 &&
      c->neutral >= 0 && c->neutral <= maximum(c->format)) {
    if (c->format.storage == CP_U8)
      return mask ? GuidedContinuousRows<uint8_t, true>(c, a, *gb, *mask, output, r)
                  : GuidedContinuousRows<uint8_t, false>(c, a, *gb, {}, output, r);
    return mask ? GuidedContinuousRows<uint16_t, true>(c, a, *gb, *mask, output, r)
                : GuidedContinuousRows<uint16_t, false>(c, a, *gb, {}, output, r);
  }
  if (c->format.storage != CP_F32 && c->operation == CP_GUIDED_MULTIPLY && c->weight_rule == CP_WEIGHT_CODE &&
      c->neutral >= 0 && c->neutral <= maximum(c->format) && c->neutral == std::floor(c->neutral)) {
    if (c->format.storage == CP_U8)
      return mask ? CodeRows<uint8_t, false, true, true>(c, a, *gb, mask, output, r)
                  : CodeRows<uint8_t, false, false, true>(c, a, *gb, nullptr, output, r);
    return mask ? CodeRows<uint16_t, false, true, true>(c, a, *gb, mask, output, r)
                : CodeRows<uint16_t, false, false, true>(c, a, *gb, nullptr, output, r);
  }
#if HWY_HAVE_FLOAT64
  if (c->format.storage != CP_F32 && c->operation == CP_MIX && c->weight_rule == CP_WEIGHT_CONTINUOUS &&
      a.step == bytes(c->format) && b.step == bytes(c->format) && output.step == bytes(c->format) &&
      (!mask || mask->step == bytes(c->format))) {
    if (c->format.storage == CP_U8)
      return mask ? IntegerBlendRows<uint8_t, CP_MIX, true>(c, a, b, mask, output, r)
                  : IntegerBlendRows<uint8_t, CP_MIX, false>(c, a, b, mask, output, r);
    return mask ? IntegerBlendRows<uint16_t, CP_MIX, true>(c, a, b, mask, output, r)
                : IntegerBlendRows<uint16_t, CP_MIX, false>(c, a, b, mask, output, r);
  }
  if (c->format.storage != CP_F32 && c->operation == CP_GUIDED_MULTIPLY && c->weight_rule == CP_WEIGHT_CONTINUOUS &&
      a.step == bytes(c->format) && gb->step == bytes(c->format) && output.step == bytes(c->format) &&
      (!mask || mask->step == bytes(c->format))) {
    if (c->format.storage == CP_U8)
      return mask ? IntegerBlendRows<uint8_t, CP_GUIDED_MULTIPLY, true>(c, a, *gb, mask, output, r)
                  : IntegerBlendRows<uint8_t, CP_GUIDED_MULTIPLY, false>(c, a, *gb, nullptr, output, r);
    return mask ? IntegerBlendRows<uint16_t, CP_GUIDED_MULTIPLY, true>(c, a, *gb, mask, output, r)
                : IntegerBlendRows<uint16_t, CP_GUIDED_MULTIPLY, false>(c, a, *gb, nullptr, output, r);
  }
  if (c->format.storage == CP_F32 && c->operation == CP_GUIDED_MULTIPLY) {
    if (a.step != 4 || gb->step != 4 || output.step != 4 || (mask && mask->step != 4))
      return PlaneRows<float, CP_GUIDED_MULTIPLY>(c, a, b, mask, ga, gb, output, r);
    if (c->neutral == 0 && !std::signbit(c->neutral))
      return mask ? GuidedFloatRows<true, 1>(c, a, *gb, mask, output, r)
                  : GuidedFloatRows<false, 1>(c, a, *gb, nullptr, output, r);
    if (c->neutral == .5)
      return mask ? GuidedFloatRows<true, 2>(c, a, *gb, mask, output, r)
                  : GuidedFloatRows<false, 2>(c, a, *gb, nullptr, output, r);
    return mask ? GuidedFloatRows<true, 0>(c, a, *gb, mask, output, r)
                : GuidedFloatRows<false, 0>(c, a, *gb, nullptr, output, r);
  }
  if (c->format.storage == CP_F32 && c->operation == CP_INVERT_MIX) {
    if (a.step == 4 && b.step == 4 && output.step == 4 && (!mask || mask->step == 4))
      return mask ? FloatBlendRows<CP_INVERT_MIX, true>(c, a, b, mask, output, r)
                  : FloatBlendRows<CP_INVERT_MIX, false>(c, a, b, nullptr, output, r);
    // Keep the established stepped memory loop, specializing only its operation.
    return PlaneRows<float, CP_INVERT_MIX>(c, a, b, mask, ga, gb, output, r);
  }
  if (c->format.storage == CP_F32 && c->operation == CP_SUBTRACT) {
    if (a.step == 4 && b.step == 4 && output.step == 4 && (!mask || mask->step == 4))
      return mask ? FloatBlendRows<CP_SUBTRACT, true>(c, a, b, mask, output, r)
                  : FloatBlendRows<CP_SUBTRACT, false>(c, a, b, nullptr, output, r);
    // Keep the established stepped memory loop, specializing only its operation.
    return PlaneRows<float, CP_SUBTRACT>(c, a, b, mask, ga, gb, output, r);
  }
  if (c->format.storage == CP_F32 && c->operation == CP_ADD && a.step == 4 && b.step == 4 &&
      output.step == 4 && (!mask || mask->step == 4))
    return mask ? FloatBlendRows<CP_ADD, true>(c, a, b, mask, output, r)
                : FloatBlendRows<CP_ADD, false>(c, a, b, nullptr, output, r);
  if (c->format.storage == CP_F32 && c->operation == CP_DIFFERENCE && a.step == 4 && b.step == 4 &&
      output.step == 4 && (!mask || mask->step == 4))
    return mask ? FloatBlendRows<CP_DIFFERENCE, true>(c, a, b, mask, output, r)
                : FloatBlendRows<CP_DIFFERENCE, false>(c, a, b, nullptr, output, r);
  if (c->format.storage == CP_F32 && a.step == 4 && b.step == 4 && output.step == 4 && (!mask || mask->step == 4)) {
    if (c->operation == CP_MIX)
      return mask ? FloatBlendRows<CP_MIX, true>(c, a, b, mask, output, r)
                  : FloatBlendRows<CP_MIX, false>(c, a, b, mask, output, r);
    if (c->operation == CP_PRODUCT) {
      if (mask)
        return FloatBlendRows<CP_PRODUCT, true>(c, a, b, mask, output, r);
      return c->opacity == 1 ? FloatBlendRows<CP_PRODUCT, false, true>(c, a, b, nullptr, output, r)
                             : FloatBlendRows<CP_PRODUCT, false>(c, a, b, nullptr, output, r);
    }
  }
  if (c->format.storage == CP_U8)
    return PlaneRows<uint8_t>(c, a, b, mask, ga, gb, output, r);
  if (c->format.storage == CP_U16)
    return PlaneRows<uint16_t>(c, a, b, mask, ga, gb, output, r);
  return PlaneRows<float>(c, a, b, mask, ga, gb, output, r);
#else
  return cp_process_plane(c, a, b, mask, ga, gb, output, r);
#endif
}
template <class T, int mask_mode = -1, int fixed_bits = 0>
void CompatRows(cp_format f, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask, cp_plane output, cp_rows r,
                int opacity) {
  if constexpr (mask_mode < 0) {
    if (mask)
      return CompatRows<T, 1>(f, a, b, mask, output, r, opacity);
    return CompatRows<T, 0>(f, a, b, mask, output, r, opacity);
  }
  if constexpr (sizeof(T) == 2 && fixed_bits == 0) {
    if (f.bits == 10)
      return CompatRows<T, mask_mode, 10>(f, a, b, mask, output, r, opacity);
    if (f.bits == 16)
      return CompatRows<T, mask_mode, 16>(f, a, b, mask, output, r, opacity);
  }
  const int bits = sizeof(T) == 1 ? 8 : fixed_bits ? fixed_bits : f.bits;
  constexpr bool masked = mask_mode != 0;
  // An 8-bit weighted sum, including rounding, is at most 65408.
  // Narrow accumulators therefore double the number of exact byte results.
  using Acc = std::conditional_t<std::is_same<T, uint8_t>::value, uint16_t, uint32_t>;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const Acc scale = static_cast<Acc>(1u << bits);
  const bool contiguous =
      a.step == sizeof(T) && b.step == sizeof(T) && output.step == sizeof(T) && (!masked || mask->step == sizeof(T));
  const auto vs = hn::Set(d, scale), vo = hn::Set(d, static_cast<Acc>(opacity));
  const auto blend = [&](auto ac, auto bc, auto mc) HWY_ATTR {
    const auto av = hn::PromoteTo(d, ac), bv = hn::PromoteTo(d, bc);
    auto result = hn::Zero(d);
    if constexpr (mask_mode == 0) {
      result = hn::ShiftRight<8>(
          hn::Add(hn::Add(hn::Mul(av, hn::Set(d, static_cast<Acc>(256 - opacity))), hn::Mul(bv, vo)), hn::Set(d, 128)));
    } else {
      const auto mv = hn::PromoteTo(d, mc);
      const auto weight = opacity == 256 ? mv : hn::ShiftRight<8>(hn::Mul(mv, vo));
      // The 16-bit maximum is 65535*65536+32768, which fits uint32_t.
      result = hn::ShiftRightSame(
          hn::Add(hn::Add(hn::Mul(av, hn::Sub(vs, weight)), hn::Mul(bv, weight)), hn::Set(d, scale / 2)), bits);
      if (opacity == 256)
        result = hn::IfThenElse(hn::Eq(mv, hn::Set(d, scale - 1)), bv, result);
    }
    return hn::DemoteTo(dt, result);
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    if (contiguous) {
      const auto* ap = reinterpret_cast<const T*>(address(a, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(b, 0, y));
      const auto* mp = masked ? reinterpret_cast<const T*>(address(*mask, 0, y)) : nullptr;
      auto* dst = reinterpret_cast<T*>(address(output, 0, y));
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), masked ? hn::LoadU(dt, mp + x) : hn::Zero(dt)), dt,
                   dst + x);
    }
    for (; x + n <= width; x += n)
      StoreChannel(blend(LoadChannel(dt, a, int(x), y, n), LoadChannel(dt, b, int(x), y, n),
                         masked ? LoadChannel(dt, *mask, int(x), y, n) : hn::Zero(dt)),
                   dt, output, int(x), y, n);
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(dt, a, static_cast<int>(x), y, count),
                         LoadChannel(dt, b, static_cast<int>(x), y, count),
                         masked ? LoadChannel(dt, *mask, static_cast<int>(x), y, count) : hn::Zero(dt)),
                   dt, output, static_cast<int>(x), y, count);
    }
  }
}

int Compat(cp_format f, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask, cp_plane output, cp_rows r,
           int opacity) {
  const int bytes = cp::bytes(f);
  if (!bytes || !rows_ok(r) || opacity < 0 || opacity > 256)
    return CP_INVALID_ARGUMENT;
  if (f.storage == CP_F32)
    return CP_UNSUPPORTED;
  if (!r.count)
    return CP_OK;
  if (!plane_ok(a, r, bytes) || !plane_ok(b, r, bytes) || !plane_ok(output, r, bytes) ||
      (mask && !plane_ok(*mask, r, bytes)))
    return CP_INVALID_ARGUMENT;
  if (!mask && opacity == 128) {
    if (f.storage == CP_U8)
      return AverageRows<uint8_t>(a, b, output, r);
    return AverageRows<uint16_t>(a, b, output, r);
  }
  if (f.storage == CP_U8)
    CompatRows<uint8_t>(f, a, b, mask, output, r, opacity);
  else
    CompatRows<uint16_t>(f, a, b, mask, output, r, opacity);
  return CP_OK;
}
template <class T>
void CopyRows(cp_const_plane source, cp_plane output, cp_rows r) {
  const hn::ScalableTag<T> d;
  const size_t n = hn::Lanes(d);
  for (int y = r.first; y < r.first + r.count; ++y)
    for (size_t x = 0; x < static_cast<size_t>(r.width); x += n) {
      const size_t count = std::min(n, static_cast<size_t>(r.width) - x);
      StoreChannel(LoadChannel(d, source, static_cast<int>(x), y, count), d, output, static_cast<int>(x), y, count);
    }
}
int Copy(cp_format f, cp_const_plane source, cp_plane output, cp_rows r) {
  const int bytes = cp::bytes(f);
  if (!bytes || !rows_ok(r))
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!plane_ok(source, r, bytes) || !plane_ok(output, r, bytes))
    return CP_INVALID_ARGUMENT;
  if (source.data == output.data && source.stride == output.stride && source.step == output.step)
    return CP_OK;
  if (source.step == bytes && output.step == bytes) {
    const size_t row_bytes = size_t(r.width) * bytes;
    if (source.stride == ptrdiff_t(row_bytes) && output.stride == ptrdiff_t(row_bytes)) {
      std::memcpy(address(output, 0, r.first), address(source, 0, r.first), row_bytes * size_t(r.count));
      return CP_OK;
    }
    for (int y = r.first; y < r.first + r.count; ++y)
      std::memcpy(address(output, 0, y), address(source, 0, y), size_t(r.width) * bytes);
    return CP_OK;
  }
  if (f.storage == CP_U8)
    CopyRows<uint8_t>(source, output, r);
  else if (f.storage == CP_U16)
    CopyRows<uint16_t>(source, output, r);
  else
    CopyRows<float>(source, output, r);
  return CP_OK;
}

template <class T>
void FillRows(cp_plane output, cp_rows r, T value) {
  if (std::is_same<T, float>::value && output.step != sizeof(T)) {
    for (int y = r.first; y < r.first + r.count; ++y)
      for (int x = 0; x < r.width; ++x)
        std::memcpy(address(output, x, y), &value, sizeof(T));
    return;
  }
  const hn::ScalableTag<T> d;
  const auto v = hn::Set(d, value);
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width);
  for (int y = r.first; y < r.first + r.count; ++y)
    for (size_t x = 0; x < width; x += n)
      StoreChannel(v, d, output, static_cast<int>(x), y, std::min(n, width - x));
}

int Fill(cp_format f, cp_plane output, cp_rows r, double value) {
  const int bytes = cp::bytes(f);
  if (!bytes || !rows_ok(r) || !std::isfinite(value))
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!plane_ok(output, r, bytes))
    return CP_INVALID_ARGUMENT;
  if (f.storage == CP_F32)
    FillRows(output, r, static_cast<float>(value));
  else {
    const auto code = static_cast<uint16_t>(std::floor(std::clamp(value, 0.0, maximum(f)) + 0.5));
    if (f.storage == CP_U8)
      FillRows(output, r, static_cast<uint8_t>(code));
    else
      FillRows(output, r, code);
  }
  return CP_OK;
}
#include "highway_extra-inl.h"
#endif
} // namespace HWY_NAMESPACE
} // namespace cp
HWY_AFTER_NAMESPACE();
#endif

#if defined(CP_SCALAR_ONLY) || HWY_ONCE
int64_t cp_compiled_targets(void) {
  int64_t mask = 0;
#ifndef CP_SCALAR_ONLY
#define CP_TARGET(target, choose) mask |= target;
#include "targets.inc"
#undef CP_TARGET
#endif
  return mask;
}
int64_t cp_supported_targets(void) {
#ifdef CP_SCALAR_ONLY
  return 0;
#else
  return cp_compiled_targets() & hwy::SupportedTargets();
#endif
}
int64_t cp_choose_target(int64_t allowed) {
  const int64_t candidates = allowed & cp_supported_targets();
  return candidates ? candidates & -candidates : CP_TARGET_C;
}
const cp_kernels* cp_get_kernels(int64_t target) {
  if (target == CP_TARGET_NATIVE)
    target = cp_choose_target(CP_TARGET_NATIVE);
  if (target == CP_TARGET_C) {
    static const cp_kernels c = {cp_process_plane, cp_blend_compat, cp_copy,  cp_fill,     cp_process_yuv,
                                 cp_resample_mask, cp_affine,       cp_clamp, cp_rgb_luma, cp_color_key};
    return &c;
  }
  if (target < 0 || (target & (target - 1)) != 0 || !(cp_supported_targets() & target))
    return nullptr;
#ifndef CP_SCALAR_ONLY
  using namespace cp;
  switch (target) {
#define CP_TARGET(target, choose)                                                                                      \
  case target: {                                                                                                       \
    static const cp_kernels table = {choose(Plane),  choose(Compat), choose(Copy),  choose(Fill), choose(Yuv),         \
                                     choose(Sample), choose(Affine), choose(Clamp), choose(Luma), choose(Key)};        \
    return &table;                                                                                                     \
  }
#include "targets.inc"
#undef CP_TARGET
    default:
      break;
  }
#endif
  return nullptr;
}
#endif
