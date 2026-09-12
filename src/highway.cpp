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
template <class T>
int AverageRows(cp_const_plane a, cp_const_plane b, cp_plane output, cp_rows r) {
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
template <class T, int operation = CP_MIX>
int WeightedRows(cp_const_plane a, cp_const_plane b, cp_plane output, cp_rows r, uint32_t weight,
                 uint32_t maximum_code = std::numeric_limits<T>::max(), uint32_t offset = 0) {
  const hn::ScalableTag<uint32_t> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const bool contiguous = a.step == sizeof(T) && b.step == sizeof(T) && output.step == sizeof(T);
  const auto blend = [&](auto av, auto bv) HWY_ATTR {
    const auto w = hn::Set(d, weight), inv = hn::Set(d, 32768 - weight), round = hn::Set(d, 16384);
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
      const auto value = hn::ShiftRight<15>(hn::Add(hn::Sub(hn::Max(positive, negative), negative), round));
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

template <class T, bool product, bool masked, bool guided = false>
int CodeRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask, cp_plane output,
             cp_rows r) {
  // 8-bit sums including the exact division correction fit in uint16_t.
  // Retaining narrow lanes doubles the number of samples processed per vector.
  using Acc = typename std::conditional<sizeof(T) == 1, uint16_t, uint32_t>::type;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const int bits = sizeof(T) == 1 ? 8 : c->format.bits;
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
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), masked ? hn::LoadU(dt, mp + x) : hn::Zero(dt)),
                   dt, dst + x);
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
template <class T>
int CodePlaneRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
                  cp_plane output, cp_rows r) {
  if (c->operation == CP_PRODUCT)
    return mask ? CodeRows<T, true, true>(c, a, b, mask, output, r)
                : CodeRows<T, true, false>(c, a, b, mask, output, r);
  return mask ? CodeRows<T, false, true>(c, a, b, mask, output, r)
              : CodeRows<T, false, false>(c, a, b, mask, output, r);
}

#if HWY_HAVE_FLOAT64
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
    // Copy endpoints retain even noncanonical input codes, as the C ABI does.
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

// Common float blends retain the scalar double evaluation order. Layout and
// operation dispatch happen before the row loop, not once per vector.
template <int operation, bool masked, bool full_product = false>
int FloatBlendRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
                   cp_plane output, cp_rows r) {
  const hn::ScalableTag<double> d;
  const hn::Rebind<float, decltype(d)> df;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width);
  const double opacity_value = c->opacity, neutral_value = c->neutral;
  const auto blend = [&](auto af, auto bf, auto mf) HWY_ATTR {
    const auto zero = hn::Zero(d), one = hn::Set(d, 1), opacity = hn::Set(d, opacity_value);
    const auto neutral = hn::Set(d, neutral_value);
    const auto av = hn::PromoteTo(d, af), bv = hn::PromoteTo(d, bf);
    const auto w = masked ? hn::Mul(opacity, hn::PromoteTo(d, mf)) : opacity;
    auto target = bv;
    if constexpr (operation == CP_PRODUCT)
      target = hn::Mul(av, bv);
    else if constexpr (operation == CP_GUIDED_MULTIPLY)
      target = hn::Add(neutral, hn::Mul(hn::Sub(av, neutral), bv));
    // Unmasked MIX copy endpoints were already handled by Plane.
    if constexpr (operation == CP_PRODUCT && !masked) {
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

template <class T>
int PlaneRows(const cp_plane_config* c, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask,
              const cp_const_plane* ga, const cp_const_plane* gb, cp_plane output, cp_rows r) {
  const hn::ScalableTag<double> d;
  const hn::Rebind<float, decltype(d)> df;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const bool select = c->operation == CP_SELECT_LIGHTER || c->operation == CP_SELECT_DARKER;
  const bool guided = select || c->operation == CP_GUIDED_MULTIPLY;
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
        const bool lighter = c->operation == CP_SELECT_LIGHTER;
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
      switch (c->operation) {
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
      if (code && c->operation == CP_GUIDED_MULTIPLY) {
        const auto darken = hn::Div(hn::Floor(hn::Add(hn::Mul(w, hn::Sub(max, guide)), half)), max);
        result = mix(av, neutral, darken);
      }
      if constexpr (std::is_same<T, float>::value) {
        auto value = hn::DemoteTo(df, result);
        // Keep source bits for exact copy endpoints (including NaN payloads).
        if (c->operation == CP_MIX || select)
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
  // and weighted sum within uint32_t; narrower U16 formats retain their
  // general path because the ABI also permits noncanonical input codes.
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
  // At full opacity, continuous and code mask weights are identical.
  if (c->format.storage != CP_F32 && (c->weight_rule == CP_WEIGHT_CODE || c->opacity == 1) &&
      (c->operation == CP_MIX || c->operation == CP_PRODUCT)) {
    if (c->format.storage == CP_U8)
      return CodePlaneRows<uint8_t>(c, a, b, mask, output, r);
    return CodePlaneRows<uint16_t>(c, a, b, mask, output, r);
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
  if (c->format.storage == CP_F32 && c->operation == CP_GUIDED_MULTIPLY && a.step == 4 && gb->step == 4 &&
      output.step == 4 && (!mask || mask->step == 4))
    return mask ? FloatBlendRows<CP_GUIDED_MULTIPLY, true>(c, a, *gb, mask, output, r)
                : FloatBlendRows<CP_GUIDED_MULTIPLY, false>(c, a, *gb, nullptr, output, r);
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
template <class T>
void CompatRows(cp_format f, cp_const_plane a, cp_const_plane b, const cp_const_plane* mask, cp_plane output, cp_rows r,
                int opacity) {
  // An 8-bit weighted sum, including rounding, is at most 65408.
  // Narrow accumulators therefore double the number of exact byte results.
  using Acc = std::conditional_t<std::is_same<T, uint8_t>::value, uint16_t, uint32_t>;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const Acc scale = static_cast<Acc>(1u << f.bits);
  const bool contiguous =
      a.step == sizeof(T) && b.step == sizeof(T) && output.step == sizeof(T) && (!mask || mask->step == sizeof(T));
  const auto vs = hn::Set(d, scale), vo = hn::Set(d, static_cast<Acc>(opacity));
  const auto blend = [&](auto ac, auto bc, auto mc) HWY_ATTR {
    const auto av = hn::PromoteTo(d, ac), bv = hn::PromoteTo(d, bc);
    auto result = hn::Zero(d);
    if (!mask) {
      result = hn::ShiftRight<8>(
          hn::Add(hn::Add(hn::Mul(av, hn::Set(d, static_cast<Acc>(256 - opacity))), hn::Mul(bv, vo)), hn::Set(d, 128)));
    } else {
      const auto mv = hn::PromoteTo(d, mc);
      const auto weight = opacity == 256 ? mv : hn::ShiftRight<8>(hn::Mul(mv, vo));
      // The 16-bit maximum is 65535*65536+32768, which fits uint32_t.
      result = hn::ShiftRightSame(
          hn::Add(hn::Add(hn::Mul(av, hn::Sub(vs, weight)), hn::Mul(bv, weight)), hn::Set(d, scale / 2)), f.bits);
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
      const auto* mp = mask ? reinterpret_cast<const T*>(address(*mask, 0, y)) : nullptr;
      auto* dst = reinterpret_cast<T*>(address(output, 0, y));
      for (; x < end; x += n)
        hn::StoreU(blend(hn::LoadU(dt, ap + x), hn::LoadU(dt, bp + x), mask ? hn::LoadU(dt, mp + x) : hn::Zero(dt)), dt,
                   dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(blend(LoadChannel(dt, a, static_cast<int>(x), y, count),
                         LoadChannel(dt, b, static_cast<int>(x), y, count),
                         mask ? LoadChannel(dt, *mask, static_cast<int>(x), y, count) : hn::Zero(dt)),
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
  // Smaller byte batches avoid the wide staging buffers on stepped channels.
  // Scaling both weights and the rounding term by 128 preserves /256 rounding.
  if (f.storage == CP_U8 && !mask && (a.step != 1 || b.step != 1 || output.step != 1))
    return WeightedRows<uint8_t>(a, b, output, r, static_cast<uint32_t>(opacity) * 128);
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
