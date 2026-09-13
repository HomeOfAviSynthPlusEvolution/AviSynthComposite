// SPDX-License-Identifier: GPL-2.0-or-later
// Intentionally re-included inside each Highway target namespace.

#if HWY_HAVE_FLOAT64
// SVE vectors and masks are sizeless: keep channels as individual locals.
// References preserve the load-all-before-store ordering required for aliasing.
template <class V>
HWY_INLINE V& VectorChannel(V& y, V& u, V& v, int channel) {
  return channel == 0 ? y : channel == 1 ? u : v;
}

template <class T, class D>
void StoreDouble(hn::VFromD<D> value, D d, cp_format f, cp_plane out, int x, int y, size_t count) {
  const hn::Rebind<float, D> df;
  if constexpr (std::is_same<T, float>::value)
    StoreChannel(hn::DemoteTo(df, value), df, out, x, y, count);
  else {
    const hn::Rebind<int32_t, D> di;
    const hn::Rebind<T, D> dt;
    const auto zero = hn::Zero(d), max = hn::Set(d, maximum(f));
    value = hn::IfThenElse(hn::IsNaN(value), zero, hn::Min(max, hn::Max(zero, value)));
    const auto rounded = hn::DemoteTo(df, hn::Floor(hn::Add(value, hn::Set(d, 0.5))));
    StoreChannel(hn::DemoteTo(dt, hn::ConvertTo(di, rounded)), dt, out, x, y, count);
  }
}
template <class D>
hn::VFromD<D> Mix(D d, hn::VFromD<D> a, hn::VFromD<D> b, hn::VFromD<D> w) {
  return hn::IfThenElse(hn::Eq(w, hn::Zero(d)), a,
                        hn::IfThenElse(hn::Eq(w, hn::Set(d, 1)), b, hn::Add(a, hn::Mul(hn::Sub(b, a), w))));
}
// For an integer numerator |x| <= 2^32 and odd divisor 1..65535, a
// reciprocal estimate plus an exact FMA residual reproduces rounded binary64
// division. The residual is exact (at most 17 significant bits); reciprocal
// error after correction is O(2^-104), far below the quotient's distance from
// a binary64 midpoint (at least one ulp/(2*divisor)). No reduced precision.
// Call only with integer numerators, not arbitrary floating-point pixels.
template <class D>
hn::VFromD<D> DivideIntegral(D, hn::VFromD<D> x, hn::VFromD<D> divisor, hn::VFromD<D> reciprocal) {
#if HWY_NATIVE_FMA
  const auto q = hn::Mul(x, reciprocal);
  return hn::MulAdd(hn::NegMulAdd(q, divisor, x), reciprocal, q);
#else
  (void)reciprocal;
  return hn::Div(x, divisor);
#endif
}

void MultiplyYuvTail(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* masks,
                     cp_yuv output, cp_rows r, size_t end) {
  if (end != static_cast<size_t>(r.width)) {
    const auto offset_in = [=](cp_const_plane p) HWY_ATTR {
      p.data = static_cast<const unsigned char*>(p.data) + end * p.step;
      return p;
    };
    const auto offset_out = [=](cp_plane p) HWY_ATTR {
      p.data = static_cast<unsigned char*>(p.data) + end * p.step;
      return p;
    };
    const cp_const_yuv tail_masks{masks ? offset_in(masks->y) : cp_const_plane{},
                                  masks ? offset_in(masks->u) : cp_const_plane{},
                                  masks ? offset_in(masks->v) : cp_const_plane{}};
    r.width -= static_cast<int>(end);
    cp_process_yuv(c, {offset_in(base.y), offset_in(base.u), offset_in(base.v)},
                   {offset_in(source.y), offset_in(source.u), offset_in(source.v)}, masks ? &tail_masks : nullptr,
                   {offset_out(output.y), offset_out(output.u), offset_out(output.v)}, r);
  }
}

// At full opacity, division by the odd code maximum has no half-integer
// ties. The nearest signed chroma product can therefore use exact uint32
// arithmetic; even 65535*65535+32767 fits. Read every plane before storing.
template <class T>
void MultiplyYuvFull(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, cp_yuv output, cp_rows r) {
  using Acc = typename std::conditional<std::is_same<T, float>::value, float, uint32_t>::type;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const cp_const_plane a[3]{base.y, base.u, base.v};
  const cp_plane out[3]{output.y, output.u, output.v};
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto* gp = reinterpret_cast<const T*>(address(source.y, 0, y));
    const T* ap[3];
    T* dst[3];
    for (int p = 0; p < 3; ++p) {
      ap[p] = reinterpret_cast<const T*>(address(a[p], 0, y));
      dst[p] = reinterpret_cast<T*>(address(out[p], 0, y));
    }
    for (size_t x = 0; x < end; x += n) {
      hn::VFromD<decltype(dt)> result_0, result_1, result_2;
      if constexpr (std::is_same<T, float>::value) {
        const auto guide = hn::LoadU(d, gp + x), zero = hn::Zero(d);
        for (int p = 0; p < 3; ++p) {
          const auto av = hn::LoadU(d, ap[p] + x), product = hn::Mul(av, guide);
          // Binary64 multiplies two binary32 significands exactly. A single
          // float multiply has the same final rounding, including underflow.
          // Normalize true zero products for the scalar +0 neutral addition,
          // but preserve the sign of a nonzero product that rounds to zero.
          const auto true_zero = hn::And(hn::Eq(product, zero), hn::Or(hn::Eq(av, zero), hn::Eq(guide, zero)));
          VectorChannel(result_0, result_1, result_2, p) = hn::IfThenElse(true_zero, zero, product);
        }
      } else {
        const int bits = c->format.bits;
        const auto guide = hn::PromoteTo(d, hn::LoadU(dt, gp + x));
        const auto half = hn::Set(d, (1u << (bits - 1)) - 1);
        for (int p = 0; p < 3; ++p) {
          const auto av = hn::PromoteTo(d, hn::LoadU(dt, ap[p] + x));
          const auto center = hn::Set(d, p ? 1u << (bits - 1) : 0);
          const auto negative = hn::Lt(av, center);
          const auto delta = hn::IfThenElse(negative, hn::Sub(center, av), hn::Sub(av, center));
          const auto magnitude = DivideCode(d, hn::Add(hn::Mul(delta, guide), half), bits);
          VectorChannel(result_0, result_1, result_2, p) =
              hn::DemoteTo(dt, hn::IfThenElse(negative, hn::Sub(center, magnitude), hn::Add(center, magnitude)));
        }
      }
      for (int p = 0; p < 3; ++p)
        hn::StoreU(VectorChannel(result_0, result_1, result_2, p), dt, dst[p] + x);
    }
  }
  MultiplyYuvTail(c, base, source, nullptr, output, r, end);
}

// Dyadic U8 opacity admits an exact uint32 numerator. Only exact ties can
// disagree with the scalar expression's staged double rounding; a compact
// per-call bit table preserves those decisions without repeated double SIMD.
struct U8MultiplyCorrections;
#if HWY_TARGET <= HWY_AVX2
struct U8MultiplyCorrections {
  HWY_ALIGN uint32_t corrections[4096];
  uint32_t keys[16];
  unsigned counts[2], lower[2]{255, 255}, upper[2]{};
  unsigned level;
  explicit U8MultiplyCorrections(unsigned weight) : level(weight) {
    cp::u8_multiply_corrections(level, corrections, keys, counts);
    for (unsigned p = 0; p < 2; ++p)
      if (counts[p] > 8)
        for (unsigned g = 0; g < 256; ++g)
          for (unsigned word = 0; word < 8; ++word)
            if (corrections[p * 2048 + g * 8 + word]) {
              lower[p] = std::min(lower[p], g);
              upper[p] = g;
            }
  }
};
void MultiplyYuvU8DyadicPrepared(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, cp_yuv output,
                                 cp_rows r, const U8MultiplyCorrections& prepared) {
  const auto& corrections = prepared.corrections;
  const auto& keys = prepared.keys;
  const auto& counts = prepared.counts;
  const auto& lower = prepared.lower;
  const auto& upper = prepared.upper;
  const unsigned level = prepared.level;
  const hn::ScalableTag<uint32_t> d;
  const hn::Rebind<int32_t, decltype(d)> di;
  const hn::Rebind<uint8_t, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
  const cp_const_plane a[3]{base.y, base.u, base.v};
  const cp_plane out[3]{output.y, output.u, output.v};
  const auto weight = hn::Set(d, level), inv = hn::Set(d, 255 * (256 - level));
  const auto denominator = hn::Set(d, 255 * 256), half = hn::Set(d, 255 * 128), one = hn::Set(d, 1);
  HWY_ALIGN uint32_t local_words[2][hn::MaxLanes(d)]{};
  for (unsigned p = 0; p < 2; ++p)
    if (counts[p] > 8 && lower[p] == upper[p])
      std::memcpy(local_words[p], corrections + p * 2048 + lower[p] * 8, 8 * sizeof(uint32_t));
  const hn::VFromD<decltype(d)> lookup[2]{hn::LoadU(d, local_words[0]), hn::LoadU(d, local_words[1])};
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto* gp = address(source.y, 0, y);
    const unsigned char* ap[3]{address(a[0], 0, y), address(a[1], 0, y), address(a[2], 0, y)};
    unsigned char* dst[3]{address(out[0], 0, y), address(out[1], 0, y), address(out[2], 0, y)};
    for (size_t x = 0; x < end; x += n) {
      const auto guide = hn::PromoteTo(d, hn::LoadU(dt, gp + x));
      const auto factor = hn::Add(inv, hn::Mul(guide, weight));
      const auto centered = hn::ShiftLeft<7>(hn::Sub(denominator, factor));
      hn::VFromD<decltype(dt)> result_0, result_1, result_2;
      for (int p = 0; p < 3; ++p) {
        const auto av = hn::PromoteTo(d, hn::LoadU(dt, ap[p] + x));
        auto numerator = hn::Mul(av, factor);
        if (p)
          numerator = hn::Add(numerator, centered);
        auto rounded = DivideCode(d, hn::ShiftRight<8>(hn::Add(numerator, half)), 8);
        const auto ties = hn::Eq(hn::Add(numerator, half), hn::Mul(rounded, denominator));
        const unsigned plane = p ? 1 : 0;
        if (counts[plane] <= 8) {
          const auto key = hn::Add(hn::ShiftLeft<8>(guide), av);
          auto correction = hn::Zero(d);
          for (unsigned i = 0; i < counts[plane]; ++i)
            correction =
                hn::Or(correction, hn::IfThenElse(hn::Eq(key, hn::Set(d, keys[plane * 8 + i])), one, hn::Zero(d)));
          rounded = hn::Sub(rounded, correction);
        } else if (lower[plane] == upper[plane]) {
          const auto word = hn::TableLookupLanes(lookup[plane], hn::IndicesFromVec(d, hn::ShiftRight<5>(av)));
          const auto bit = hn::And(word >> hn::And(av, hn::Set(d, 31)), one);
          rounded = hn::Sub(rounded, hn::IfThenElse(hn::Eq(guide, hn::Set(d, lower[plane])), bit, hn::Zero(d)));
        } else if (!hn::AllFalse(d, hn::And(ties, hn::And(hn::Ge(guide, hn::Set(d, lower[plane])),
                                                          hn::Le(guide, hn::Set(d, upper[plane])))))) {
          const auto index = hn::BitCast(di, hn::Add(hn::ShiftLeft<3>(guide), hn::ShiftRight<5>(av)));
          const auto word = hn::GatherIndex(d, corrections + (p ? 2048 : 0), index);
          rounded = hn::Sub(rounded, hn::And(word >> hn::And(av, hn::Set(d, 31)), one));
        }
        VectorChannel(result_0, result_1, result_2, p) = hn::DemoteTo(dt, rounded);
      }
      for (int p = 0; p < 3; ++p)
        hn::StoreU(VectorChannel(result_0, result_1, result_2, p), dt, dst[p] + x);
    }
  }
  MultiplyYuvTail(c, base, source, nullptr, output, r, end);
}

void MultiplyYuvU8Dyadic(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, cp_yuv output, cp_rows r,
                         unsigned level) {
  const U8MultiplyCorrections prepared(level);
  MultiplyYuvU8DyadicPrepared(c, base, source, output, r, prepared);
}
#endif

// Binary32 candidates are accepted only when rounding is unambiguous. With
// integer inputs in [0,M] through 10 bits, use u=epsilon/2. Absolute errors of normalized
// guide, slope, weight and factor are bounded by 2u, 3u, 4u and 9u (up to
// O(u^2)). Centered multiplication/addition adds at most 2u*M; distance
// evaluation and the binary64 reference fit the remaining margin below
// 16u*M = 8*epsilon*M. Tiny/subnormal weights obey the same absolute bound.
// Recompute samples near a half-integer in the original order. This keeps
// exact integer outputs while usually processing twice as many SIMD lanes.
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES
// Quantize the combined darkening factor once, then share it across channels.
// Float evaluation contributes < .063 code through 16 bits; weight rounding
// adds <= .5 code to the final sample, hence total output error remains <=1 LSB.
// Integer convex sums plus rounding and DivideCode's correction fit the lane.
template <class T, bool masked>
void MultiplyYuvNarrow(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source,
                       const cp_const_yuv* masks, cp_yuv output, cp_rows r) {
  using Acc = typename std::conditional<sizeof(T) == 1, uint16_t, uint32_t>::type;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<T, decltype(d)> dt;
  const hn::ScalableTag<float> df;
  const hn::Rebind<uint32_t, decltype(df)> du;
  const unsigned maximum = (1u << c->format.bits) - 1;
  const float scale = float(c->opacity / maximum);
  const auto max = hn::Set(d, Acc(maximum));
  const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
  const bool shared = !masked || (masks->y.data == masks->u.data && masks->y.data == masks->v.data &&
                                  masks->y.stride == masks->u.stride && masks->y.stride == masks->v.stride);
  const auto weight = [&](auto guide, auto mask) HWY_ATTR {
    const auto product = hn::Mul(hn::Sub(max, guide), mask);
    const auto convert = [&](auto v) HWY_ATTR {
      const auto scaled = hn::Mul(hn::ConvertTo(df, v), hn::Set(df, scale));
      return hn::ConvertTo(du, hn::Add(scaled, hn::Set(df, .5f)));
    };
    if constexpr (sizeof(T) == 1) {
      const hn::Half<decltype(d)> dh;
      return hn::Min(max, hn::Combine(d, hn::DemoteTo(dh, convert(hn::PromoteTo(du, hn::UpperHalf(dh, product)))),
                                        hn::DemoteTo(dh, convert(hn::PromoteTo(du, hn::LowerHalf(dh, product))))));
    } else {
      return hn::Min(max, convert(product));
    }
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto load = [&](cp_const_plane p, size_t x) HWY_ATTR {
      return hn::PromoteTo(d, hn::LoadU(dt, reinterpret_cast<const T*>(address(p, 0, y)) + x));
    };
    for (size_t x = 0; x < end; x += n) {
      const auto guide = load(source.y, x);
      const auto ay = load(base.y, x), au = load(base.u, x), av = load(base.v, x);
      const auto wy = weight(guide, masked ? load(masks->y, x) : max);
      const auto wu = shared ? wy : weight(guide, load(masks->u, x));
      const auto wv = shared ? wy : weight(guide, load(masks->v, x));
      const auto finish = [&](auto a, auto w, bool chroma) HWY_ATTR {
        auto sum = hn::Mul(a, hn::Sub(max, w));
        if (chroma) sum = hn::Add(sum, hn::Mul(w, hn::Set(d, Acc((maximum + 1) / 2))));
        return hn::DemoteTo(dt, DivideCode(d, hn::Add(sum, hn::Set(d, Acc(maximum / 2))), c->format.bits));
      };
      const auto yy = finish(ay, wy, false), uu = finish(au, wu, true), vv = finish(av, wv, true);
      hn::StoreU(yy, dt, reinterpret_cast<T*>(address(output.y, 0, y)) + x);
      hn::StoreU(uu, dt, reinterpret_cast<T*>(address(output.u, 0, y)) + x);
      hn::StoreU(vv, dt, reinterpret_cast<T*>(address(output.v, 0, y)) + x);
    }
  }
  MultiplyYuvTail(c, base, source, masks, output, r, end);
}
#endif

template <class T, bool masked, bool approximate = false>
void MultiplyYuvFloatCandidate(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source,
                               const cp_const_yuv* masks, cp_yuv output, cp_rows r) {
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES
  if constexpr (approximate && sizeof(T) == 1) {
    MultiplyYuvNarrow<T, masked>(c, base, source, masks, output, r);
    return;
  }
#endif
  const auto madd = [](auto a, auto b, auto c) HWY_ATTR {
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES
    if constexpr (approximate && !masked) return hn::MulAdd(a, b, c);
#endif
    return hn::Add(hn::Mul(a, b), c);
  };
  const hn::ScalableTag<float> d;
  const hn::Rebind<T, decltype(d)> dt;
  const hn::Rebind<int32_t, decltype(d)> di;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const unsigned maximum = (1u << c->format.bits) - 1;
  const auto zero = hn::Zero(d), one = hn::Set(d, 1), half = hn::Set(d, .5f);
  const auto center = hn::Set(d, (maximum + 1) / 2), opacity = hn::Set(d, float(c->opacity));
  const auto reciprocal = hn::Set(d, 1.f / maximum);
  const auto tolerance = hn::Set(d, 8 * std::numeric_limits<float>::epsilon() * maximum);
  const cp_const_plane a[3]{base.y, base.u, base.v};
  const cp_plane out[3]{output.y, output.u, output.v};
  const cp_const_plane m[3]{masked ? masks->y : cp_const_plane{}, masked ? masks->u : cp_const_plane{},
                            masked ? masks->v : cp_const_plane{}};
  const bool shared = !masked || (m[0].data == m[1].data && m[0].data == m[2].data && m[0].stride == m[1].stride &&
                                  m[0].stride == m[2].stride);
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto* guide_row = reinterpret_cast<const T*>(address(source.y, 0, y));
    const T* rows[3];
    const T* mask_rows[3];
    T* dest[3];
    for (int p = 0; p < 3; ++p) {
      rows[p] = reinterpret_cast<const T*>(address(a[p], 0, y));
      mask_rows[p] = masked ? reinterpret_cast<const T*>(address(m[p], 0, y)) : nullptr;
      dest[p] = reinterpret_cast<T*>(address(out[p], 0, y));
    }
    for (size_t x = 0; x < end; x += n) {
      const auto guide = hn::ConvertTo(d, hn::PromoteTo(di, hn::LoadU(dt, guide_row + x)));
      const auto slope = hn::Sub(hn::Mul(guide, reciprocal), one);
      const auto common_weight =
          masked ? hn::Mul(opacity,
                           hn::Mul(hn::ConvertTo(d, hn::PromoteTo(di, hn::LoadU(dt, mask_rows[0] + x))), reciprocal))
                 : opacity;
      const auto common_factor = madd(slope, common_weight, one);
      if constexpr (approximate) {
        // Keep approximate loops free of channel selection and
        // rounding-boundary work. Load all channels before any aliased store.
        const auto ay = hn::ConvertTo(d, hn::PromoteTo(di, hn::LoadU(dt, rows[0] + x)));
        const auto au = hn::ConvertTo(d, hn::PromoteTo(di, hn::LoadU(dt, rows[1] + x)));
        const auto av = hn::ConvertTo(d, hn::PromoteTo(di, hn::LoadU(dt, rows[2] + x)));
        const auto factor_for = [&](int p) HWY_ATTR {
          if constexpr (masked) {
            if (!shared) {
              const auto mv = hn::ConvertTo(d, hn::PromoteTo(di, hn::LoadU(dt, mask_rows[p] + x)));
              return madd(slope, hn::Mul(opacity, hn::Mul(mv, reciprocal)), one);
            }
          }
          return common_factor;
        };
        const auto uf = factor_for(1), vf = factor_for(2);
        const auto chroma_round = hn::Add(center, half);
        const auto yy = hn::ConvertInRangeTo(di, madd(ay, common_factor, half));
        const auto uu = hn::ConvertInRangeTo(di, madd(hn::Sub(au, center), uf, chroma_round));
        const auto vv = hn::ConvertInRangeTo(di, madd(hn::Sub(av, center), vf, chroma_round));
        hn::StoreU(hn::DemoteTo(dt, yy), dt, dest[0] + x);
        hn::StoreU(hn::DemoteTo(dt, uu), dt, dest[1] + x);
        hn::StoreU(hn::DemoteTo(dt, vv), dt, dest[2] + x);
        continue;
      }
      hn::VFromD<decltype(di)> result_0, result_1, result_2;
      hn::MFromD<decltype(d)> uncertain_0, uncertain_1, uncertain_2;
      for (int p = 0; p < 3; ++p) {
        const auto av = hn::ConvertTo(d, hn::PromoteTo(di, hn::LoadU(dt, rows[p] + x)));
        auto factor = common_factor;
        if (masked && p && !shared) {
          const auto w = hn::Mul(
              opacity, hn::Mul(hn::ConvertTo(d, hn::PromoteTo(di, hn::LoadU(dt, mask_rows[p] + x))), reciprocal));
          factor = hn::Add(one, hn::Mul(slope, w));
        }
        const auto neutral = p ? center : zero;
        const auto candidate = hn::Add(neutral, hn::Mul(hn::Sub(av, neutral), factor));
        VectorChannel(result_0, result_1, result_2, p) = hn::ConvertInRangeTo(di, hn::Add(candidate, half));
        const auto distance =
            hn::Abs(hn::Sub(candidate, hn::ConvertTo(d, VectorChannel(result_0, result_1, result_2, p))));
        VectorChannel(uncertain_0, uncertain_1, uncertain_2, p) = hn::Ge(distance, hn::Sub(half, tolerance));
      }
      if (!approximate && !hn::AllFalse(d, hn::Or(uncertain_0, hn::Or(uncertain_1, uncertain_2)))) {
        HWY_ALIGN int32_t corrected[3][hn::MaxLanes(d)];
        const hn::ScalableTag<double> dd;
        const hn::Rebind<float, decltype(dd)> dh;
        const hn::Rebind<T, decltype(dd)> db;
        const hn::Rebind<int32_t, decltype(dd)> di32;
        const size_t half_lanes = hn::Lanes(dd);
        const auto max64 = hn::Set(dd, maximum), reciprocal64 = hn::Set(dd, 1.0 / maximum);
        const auto opacity64 = hn::Set(dd, c->opacity);
        for (int p = 0; p < 3; ++p) {
          if (hn::AllFalse(d, VectorChannel(uncertain_0, uncertain_1, uncertain_2, p)))
            continue;
          hn::StoreU(VectorChannel(result_0, result_1, result_2, p), di, corrected[p]);
          const auto correct_half = [&](auto upper) HWY_ATTR {
            const auto flags = decltype(upper)::value
                                   ? hn::UpperHalfOfMask(dh, VectorChannel(uncertain_0, uncertain_1, uncertain_2, p))
                                   : hn::LowerHalfOfMask(dh, VectorChannel(uncertain_0, uncertain_1, uncertain_2, p));
            if (hn::AllFalse(dh, flags))
              return;
            const size_t offset = decltype(upper)::value ? half_lanes : 0;
            const auto av = hn::PromoteTo(dd, hn::PromoteTo(di32, hn::LoadU(db, rows[p] + x + offset)));
            const auto gv = hn::PromoteTo(dd, hn::PromoteTo(di32, hn::LoadU(db, guide_row + x + offset)));
            const auto neutral = hn::Set(dd, p ? (maximum + 1) / 2 : 0);
            auto w = opacity64;
            if constexpr (masked) {
              const auto mv = hn::PromoteTo(dd, hn::PromoteTo(di32, hn::LoadU(db, mask_rows[p] + x + offset)));
              w = hn::Mul(opacity64, DivideIntegral(dd, mv, max64, reciprocal64));
            }
            const auto target =
                hn::Add(neutral, DivideIntegral(dd, hn::Mul(hn::Sub(av, neutral), gv), max64, reciprocal64));
            const auto value = Mix(dd, av, target, w);
            const auto rounded = hn::DemoteInRangeTo(di32, hn::Add(value, hn::Set(dd, .5)));
            hn::StoreU(rounded, di32, corrected[p] + offset);
          };
          correct_half(std::false_type{});
          correct_half(std::true_type{});
          VectorChannel(result_0, result_1, result_2, p) = hn::LoadU(di, corrected[p]);
        }
      }
      for (int p = 0; p < 3; ++p)
        hn::StoreU(hn::DemoteTo(dt, VectorChannel(result_0, result_1, result_2, p)), dt, dest[p] + x);
    }
  }
  MultiplyYuvTail(c, base, source, masks, output, r, end);
}

// Split F into 256*hi+lo before multiplication. Then
// floor((a*F + N*(M*256-F) + M*128)/256) equals
// a*hi + N*(M-hi) + floor(((a-N)*lo + M*128)/256).
// The first part is <= M*M; the signed residual fits 25 bits. The final
// sum is <= M*M+floor(M/2), so exact DivideCode uses uint32 lanes throughout.
void MultiplyYuvU16Dyadic(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, cp_yuv output, cp_rows r,
                          unsigned level) {
  const hn::ScalableTag<uint32_t> d;
  const hn::Rebind<uint16_t, decltype(d)> dt;
  const hn::Rebind<int32_t, decltype(d)> di;
  const unsigned bits = c->format.bits, maximum = (1u << bits) - 1;
  const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
  const auto weight = hn::Set(d, level), inv = hn::Set(d, maximum * (256 - level));
  const auto max = hn::Set(d, maximum), lowmask = hn::Set(d, 255), half = hn::Set(d, maximum * 128);
  const cp_const_plane a[3]{base.y, base.u, base.v};
  const cp_plane out[3]{output.y, output.u, output.v};
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto* gp = reinterpret_cast<const uint16_t*>(address(source.y, 0, y));
    const uint16_t* ap[3];
    uint16_t* dst[3];
    for (int p = 0; p < 3; ++p) {
      ap[p] = reinterpret_cast<const uint16_t*>(address(a[p], 0, y));
      dst[p] = reinterpret_cast<uint16_t*>(address(out[p], 0, y));
    }
    for (size_t x = 0; x < end; x += n) {
      const auto guide = hn::PromoteTo(d, hn::LoadU(dt, gp + x));
      const auto factor = hn::Add(inv, hn::Mul(guide, weight));
      const auto hi = hn::ShiftRight<8>(factor), lo = hn::And(factor, lowmask);
      const auto centered = hn::ShiftLeftSame(hn::Sub(max, hi), bits - 1);
      hn::VFromD<decltype(dt)> result_0, result_1, result_2;
      for (int p = 0; p < 3; ++p) {
        const auto av = hn::PromoteTo(d, hn::LoadU(dt, ap[p] + x));
        const auto neutral = hn::Set(d, p ? (maximum + 1) / 2 : 0);
        const auto residual = hn::Add(hn::Mul(hn::Sub(av, neutral), lo), half);
        auto scaled = hn::Add(hn::Mul(av, hi), hn::BitCast(d, hn::ShiftRight<8>(hn::BitCast(di, residual))));
        if (p)
          scaled = hn::Add(scaled, centered);
        const auto rounded = DivideCode(d, scaled, bits);
        VectorChannel(result_0, result_1, result_2, p) = hn::DemoteTo(dt, rounded);
        const auto ties =
            hn::And(hn::Eq(hn::And(residual, lowmask), hn::Zero(d)), hn::Eq(scaled, hn::Mul(rounded, max)));
        if (!hn::AllFalse(d, ties)) {
          HWY_ALIGN uint32_t values[hn::MaxLanes(d)];
          hn::StoreU(rounded, d, values);
          const hn::ScalableTag<double> dd;
          const hn::Rebind<int32_t, decltype(dd)> dh;
          const hn::Rebind<uint16_t, decltype(dd)> db;
          const size_t half_lanes = hn::Lanes(dd);
          const auto max64 = hn::Set(dd, double(maximum)), reciprocal = hn::Set(dd, 1.0 / maximum);
          for (size_t offset = 0; offset < n; offset += half_lanes) {
            const auto a64 = hn::PromoteTo(dd, hn::PromoteTo(dh, hn::LoadU(db, ap[p] + x + offset)));
            const auto g64 = hn::PromoteTo(dd, hn::PromoteTo(dh, hn::LoadU(db, gp + x + offset)));
            const auto center64 = hn::Set(dd, p ? double((maximum + 1) / 2) : 0.0);
            const auto target =
                hn::Add(center64, DivideIntegral(dd, hn::Mul(hn::Sub(a64, center64), g64), max64, reciprocal));
            const auto exact =
                hn::DemoteInRangeTo(dh, hn::Add(Mix(dd, a64, target, hn::Set(dd, c->opacity)), hn::Set(dd, .5)));
            hn::StoreU(exact, dh, reinterpret_cast<int32_t*>(values + offset));
          }
          VectorChannel(result_0, result_1, result_2, p) = hn::DemoteTo(dt, hn::LoadU(d, values));
        }
      }
      for (int p = 0; p < 3; ++p)
        hn::StoreU(VectorChannel(result_0, result_1, result_2, p), dt, dst[p] + x);
    }
  }
  MultiplyYuvTail(c, base, source, nullptr, output, r, end);
}

// High-depth Multiply shares the normalized guide and, when the
// caller uses one mask for all channels, the weight as well. Reassociation
// is certified against a conservative binary64 rounding error bound; rare
// ambiguous vectors use the original expression before any stores.
template <bool masked>
void MultiplyYuvInteger16(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* masks,
                          cp_yuv output, cp_rows r) {
  const hn::ScalableTag<double> d;
  const hn::Rebind<uint16_t, decltype(d)> dt;
  const hn::Rebind<int32_t, decltype(d)> di;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  const double maximum = cp::maximum(c->format);
  const auto max = hn::Set(d, maximum), zero = hn::Zero(d), one = hn::Set(d, 1), half = hn::Set(d, .5);
  const auto center = hn::Set(d, (maximum + 1) / 2), opacity = hn::Set(d, c->opacity);
  const auto reciprocal = hn::Set(d, 1 / maximum);
  const auto tolerance = hn::Set(d, 64 * std::numeric_limits<double>::epsilon() * maximum);
  const cp_const_plane a[3]{base.y, base.u, base.v},
      m[3]{masked ? masks->y : cp_const_plane{}, masked ? masks->u : cp_const_plane{},
           masked ? masks->v : cp_const_plane{}};
  const cp_plane out[3]{output.y, output.u, output.v};
  const bool shared = !masked || (m[0].data == m[1].data && m[0].data == m[2].data && m[0].stride == m[1].stride &&
                                  m[0].stride == m[2].stride);
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto* gp = reinterpret_cast<const uint16_t*>(address(source.y, 0, y));
    const uint16_t* ap[3];
    const uint16_t* mp[3];
    uint16_t* dst[3];
    for (int p = 0; p < 3; ++p) {
      ap[p] = reinterpret_cast<const uint16_t*>(address(a[p], 0, y));
      mp[p] = masked ? reinterpret_cast<const uint16_t*>(address(m[p], 0, y)) : nullptr;
      dst[p] = reinterpret_cast<uint16_t*>(address(out[p], 0, y));
    }
    for (size_t x = 0; x < end; x += n) {
      const auto guide = hn::PromoteTo(d, hn::PromoteTo(di, hn::LoadU(dt, gp + x)));
      const auto slope = hn::MulAdd(guide, reciprocal, hn::Neg(one));
      hn::VFromD<decltype(d)> av_0, av_1, av_2, mv_0, mv_1, mv_2;
      hn::VFromD<decltype(di)> rounded_0, rounded_1, rounded_2;
      auto uncertain = hn::Eq(one, zero);
      mv_0 = masked ? hn::PromoteTo(d, hn::PromoteTo(di, hn::LoadU(dt, mp[0] + x))) : max;
      const auto common_weight = masked ? hn::Mul(opacity, hn::Mul(mv_0, reciprocal)) : opacity;
      const auto common_factor = hn::MulAdd(slope, common_weight, one);
      for (int p = 0; p < 3; ++p) {
        VectorChannel(av_0, av_1, av_2, p) = hn::PromoteTo(d, hn::PromoteTo(di, hn::LoadU(dt, ap[p] + x)));
        auto factor = common_factor;
        if (masked && p && !shared) {
          VectorChannel(mv_0, mv_1, mv_2, p) = hn::PromoteTo(d, hn::PromoteTo(di, hn::LoadU(dt, mp[p] + x)));
          factor = hn::MulAdd(slope, hn::Mul(opacity, hn::Mul(VectorChannel(mv_0, mv_1, mv_2, p), reciprocal)), one);
        } else {
          VectorChannel(mv_0, mv_1, mv_2, p) = mv_0;
        }
        const auto neutral = p ? center : zero;
        const auto result = hn::MulAdd(hn::Sub(VectorChannel(av_0, av_1, av_2, p), neutral), factor, neutral);
        VectorChannel(rounded_0, rounded_1, rounded_2, p) = hn::DemoteInRangeTo(di, hn::Add(result, half));
        const auto distance =
            hn::Abs(hn::Sub(result, hn::PromoteTo(d, VectorChannel(rounded_0, rounded_1, rounded_2, p))));
        uncertain = hn::Or(uncertain, hn::Ge(distance, hn::Sub(half, tolerance)));
      }
      if (!hn::AllFalse(d, uncertain))
        for (int p = 0; p < 3; ++p) {
          const auto neutral = p ? center : zero;
          const auto w = masked
                             ? hn::Mul(opacity, DivideIntegral(d, VectorChannel(mv_0, mv_1, mv_2, p), max, reciprocal))
                             : opacity;
          const auto target =
              hn::Add(neutral, DivideIntegral(d, hn::Mul(hn::Sub(VectorChannel(av_0, av_1, av_2, p), neutral), guide),
                                              max, reciprocal));
          VectorChannel(rounded_0, rounded_1, rounded_2, p) =
              hn::DemoteInRangeTo(di, hn::Add(Mix(d, VectorChannel(av_0, av_1, av_2, p), target, w), half));
        }
      // Exact real results lie in [0, maximum]; candidate error is < .5,
      // so rounded candidates and the fallback both fit the storage domain.
      for (int p = 0; p < 3; ++p)
        hn::StoreU(hn::DemoteTo(dt, VectorChannel(rounded_0, rounded_1, rounded_2, p)), dt, dst[p] + x);
    }
  }
  MultiplyYuvTail(c, base, source, masks, output, r, end);
}

template <class T, bool masked, bool interior = false>
void MultiplyYuvRows(const cp_yuv_config*, cp_const_yuv, cp_const_yuv, const cp_const_yuv*, cp_yuv, cp_rows);

template <class T>
void YuvRows(const cp_yuv_config*, cp_const_yuv, cp_const_yuv, const cp_const_yuv*, cp_yuv, cp_rows);

// AVX-512 masked memory operations do not access the fourth (undeclared)
// channel. Stage their registers for Highway's portable interleave shuffle;
// the optimizer can eliminate the local store/load round trip.
template <class T, bool store>
bool PackedThree(const cp_const_plane* inputs, const cp_plane* outputs, T* const* planes, int x, int y, int count) {
#if HWY_TARGET <= HWY_AVX3
  cp_const_plane view[3];
  for (int p = 0; p < 3; ++p) {
    if constexpr (store)
      view[p] = {outputs[p].data, outputs[p].stride, outputs[p].step};
    else
      view[p] = inputs[p];
  }
  for (int p = 0; p < 3; ++p)
    if (view[p].step != 4 * sizeof(T) || view[p].stride != view[0].stride)
      return false;
  const auto a = reinterpret_cast<uintptr_t>(view[0].data), b = reinterpret_cast<uintptr_t>(view[1].data),
             c = reinterpret_cast<uintptr_t>(view[2].data);
  const bool forward = b == a + sizeof(T) && c == a + 2 * sizeof(T);
  const bool reverse = b == c + sizeof(T) && a == c + 2 * sizeof(T);
  if (!forward && !reverse)
    return false;
  const hn::ScalableTag<T> d;
  const hn::RebindToUnsigned<decltype(d)> du;
  const int lanes = int(hn::Lanes(d));
  const auto mask = hn::RebindMask(d, hn::Ne(hn::And(hn::Iota(du, 0), hn::Set(du, 3)), hn::Set(du, 3)));
  HWY_ALIGN T packed[4 * hn::MaxLanes(d)];
  int i = 0;
  for (; i + lanes <= count; i += lanes) {
    auto* ptr = reinterpret_cast<T*>(const_cast<unsigned char*>(address(view[forward ? 0 : 2], x + i, y)));
    if constexpr (store) {
      hn::StoreInterleaved4(hn::LoadU(d, planes[forward ? 0 : 2] + i), hn::LoadU(d, planes[1] + i),
                            hn::LoadU(d, planes[forward ? 2 : 0] + i), hn::Zero(d), d, packed);
      for (int part = 0; part < 4; ++part)
        hn::BlendedStore(hn::LoadU(d, packed + part * lanes), mask, d, ptr + part * lanes);
    } else {
      for (int part = 0; part < 4; ++part)
        hn::StoreU(hn::MaskedLoad(mask, d, ptr + part * lanes), d, packed + part * lanes);
      hn::VFromD<decltype(d)> first, second, third, ignored;
      hn::LoadInterleaved4(d, packed, first, second, third, ignored);
      hn::StoreU(first, d, planes[forward ? 0 : 2] + i);
      hn::StoreU(second, d, planes[1] + i);
      hn::StoreU(third, d, planes[forward ? 2 : 0] + i);
    }
  }
  for (int p = 0; p < 3; ++p)
    for (int j = i; j < count; ++j) {
      if constexpr (store)
        std::memcpy(address(outputs[p], x + j, y), planes[p] + j, sizeof(T));
      else
        std::memcpy(planes[p] + j, address(inputs[p], x + j, y), sizeof(T));
    }
  return true;
#else
  (void)inputs;
  (void)outputs;
  (void)planes;
  (void)x;
  (void)y;
  (void)count;
  return false;
#endif
}

// Pack only declared samples into a bounded tile. This amortizes strided
// address handling and reuses the contiguous kernels without reading gaps.
// Gather every input before scattering output, including aliased masks/guides.
template <class T, bool masked, bool multiply>
HWY_NOINLINE void YuvSteppedPrepared(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source,
                                     const cp_const_yuv* masks, cp_yuv output, cp_rows r,
                                     const U8MultiplyCorrections* prepared) {
  constexpr int tile = 256;
  HWY_ALIGN T pixels[3][tile], guides[multiply ? 1 : 3][tile], weights[3][tile];
  const cp_const_plane a[3]{base.y, base.u, base.v};
  const cp_plane out[3]{output.y, output.u, output.v};
  const cp_const_plane m[3]{masked ? masks->y : cp_const_plane{}, masked ? masks->u : cp_const_plane{},
                            masked ? masks->v : cp_const_plane{}};
  const bool shared = !masked || (m[0].data == m[1].data && m[0].data == m[2].data && m[0].stride == m[1].stride &&
                                  m[0].stride == m[2].stride && m[0].step == m[1].step && m[0].step == m[2].step);
  const auto in = [tile](const T* ptr) HWY_ATTR {
    return cp_const_plane{ptr, static_cast<ptrdiff_t>(tile * sizeof(T)), sizeof(T)};
  };
  const auto dest = [tile](T* ptr) HWY_ATTR {
    return cp_plane{ptr, static_cast<ptrdiff_t>(tile * sizeof(T)), sizeof(T)};
  };
  const cp_const_yuv packed_base{in(pixels[0]), in(pixels[1]), in(pixels[2])};
  const cp_const_plane b[3]{source.y, source.u, source.v};
  const cp_const_yuv packed_source{in(guides[0]), in(guides[multiply ? 0 : 1]), in(guides[multiply ? 0 : 2])};
  const cp_const_yuv packed_masks{in(weights[0]), in(weights[shared ? 0 : 1]), in(weights[shared ? 0 : 2])};
  const cp_yuv packed_output{dest(pixels[0]), dest(pixels[1]), dest(pixels[2])};
  for (int y = r.first; y < r.first + r.count; ++y)
    for (int x = 0; x < r.width;) {
      const int count = std::min(tile, r.width - x);
      const auto gather = [&](cp_const_plane plane, T* ptr) HWY_ATTR {
        for (int i = 0; i < count; ++i)
          std::memcpy(ptr + i, address(plane, x + i, y), sizeof(T));
      };
      T* base_tiles[3]{pixels[0], pixels[1], pixels[2]};
      T* source_tiles[3]{guides[0], guides[multiply ? 0 : 1], guides[multiply ? 0 : 2]};
      if (multiply || !PackedThree<T, false>(b, nullptr, source_tiles, x, y, count))
        for (int p = 0; p < (multiply ? 1 : 3); ++p)
          gather(b[p], guides[p]);
      if (!PackedThree<T, false>(a, nullptr, base_tiles, x, y, count))
        for (int p = 0; p < 3; ++p)
          gather(a[p], pixels[p]);
      for (int p = 0; p < 3; ++p)
        if (masked && (!shared || p == 0))
          gather(m[p], weights[p]);
      if constexpr (multiply) {
        bool handled = false;
#if HWY_TARGET <= HWY_AVX2
        if constexpr (std::is_same<T, uint8_t>::value && !masked)
          if (prepared) {
            MultiplyYuvU8DyadicPrepared(c, packed_base, packed_source, packed_output, {count, 1, 0, 1}, *prepared);
            handled = true;
          }
#else
        (void)prepared;
#endif
        if (!handled)
          MultiplyYuvRows<T, masked>(c, packed_base, packed_source, masked ? &packed_masks : nullptr, packed_output,
                                     {count, 1, 0, 1});
      } else
        YuvRows<T>(c, packed_base, packed_source, masked ? &packed_masks : nullptr, packed_output, {count, 1, 0, 1});
      if (!PackedThree<T, true>(nullptr, out, base_tiles, x, y, count))
        for (int p = 0; p < 3; ++p)
          for (int i = 0; i < count; ++i)
            std::memcpy(address(out[p], x + i, y), pixels[p] + i, sizeof(T));
      x += count;
    }
}

template <class T, bool masked, bool multiply>
HWY_NOINLINE void YuvStepped(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* masks,
                             cp_yuv output, cp_rows r) {
#if HWY_TARGET <= HWY_AVX2
  if constexpr (std::is_same<T, uint8_t>::value && !masked && multiply) {
    const double level = c->opacity * 256;
    if (int64_t(r.width) * r.count >= 65536 && level > 0 && level < 256 && level == std::floor(level)) {
      const U8MultiplyCorrections prepared{unsigned(level)};
      YuvSteppedPrepared<T, masked, multiply>(c, base, source, masks, output, r, &prepared);
      return;
    }
  }
#endif
  YuvSteppedPrepared<T, masked, multiply>(c, base, source, masks, output, r, nullptr);
}

// Fused Overlay Multiply: share the source luma guide and load all channels
// before writing any channel. Evaluation remains identical to the plane API.
template <class T, bool masked, bool interior>
void MultiplyYuvRows(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* masks,
                     cp_yuv output, cp_rows r) {
  const hn::ScalableTag<double> d;
  using SampleTag = hn::Rebind<T, hn::ScalableTag<double>>;
  const SampleTag dt;
  const hn::Rebind<float, decltype(d)> df;
  const hn::Rebind<uint32_t, decltype(d)> du;
  const hn::Rebind<int32_t, decltype(d)> di;
  const size_t n = hn::Lanes(d);
  const double maximum = cp::maximum(c->format);
  const auto max = hn::Set(d, maximum), zero = hn::Zero(d), opacity = hn::Set(d, c->opacity);
  const auto reciprocal = hn::Set(d, 1.0 / maximum);
  const auto divide = [&](auto value) HWY_ATTR {
    if constexpr (std::is_same<T, float>::value)
      return value; // float scale is exactly one
    else
      return DivideIntegral(d, value, max, reciprocal);
  };
  const cp_const_plane a[3]{base.y, base.u, base.v};
  const cp_plane out[3]{output.y, output.u, output.v};
  const cp_const_plane m[3]{masked ? masks->y : cp_const_plane{}, masked ? masks->u : cp_const_plane{},
                            masked ? masks->v : cp_const_plane{}};
  const auto promote = [=](auto value) HWY_ATTR {
    if constexpr (std::is_same<T, float>::value)
      return hn::PromoteTo(d, value);
    else
      return hn::PromoteTo(d, hn::ConvertTo(df, hn::PromoteTo(du, value)));
  };
  bool contiguous = source.y.step == sizeof(T);
  for (int p = 0; p < 3; ++p)
    contiguous =
        contiguous && a[p].step == sizeof(T) && out[p].step == sizeof(T) && (!masked || m[p].step == sizeof(T));
  if (!contiguous) {
    YuvStepped<T, masked, true>(c, base, source, masks, output, r);
    return;
  }
  if constexpr (!masked) {
    if (contiguous && c->opacity == 1) {
      MultiplyYuvFull<T>(c, base, source, output, r);
      return;
    }
  }
  // For canonical integer inputs the binary32 error is bounded by
  // 8*epsilon*M < 0.063 through 16 bits, hence final rounding differs
  // by at most one code. Keep full-opacity exact kernels.
  if constexpr (!std::is_same<T, float>::value) {
    const double level = c->opacity * 256;
    // Reserve dyadic weights only when the exact integer kernel below can
    // actually run. Other targets/small images use the same <=1 LSB path.
    const bool keep_byte_dyadic = HWY_TARGET <= HWY_AVX2 && sizeof(T) == 1 && !masked &&
                                  int64_t(r.width) * r.count >= 65536 && level == std::floor(level);
    if (c->opacity > 0 && c->opacity < 1 && !keep_byte_dyadic) {
      MultiplyYuvFloatCandidate<T, masked, true>(c, base, source, masks, output, r);
      return;
    }
  }
  if constexpr (std::is_same<T, uint16_t>::value) {
    if (contiguous) {
      if constexpr (!masked) {
        const double level = c->opacity * 256;
        if (level > 0 && level < 256 && level == std::floor(level)) {
          MultiplyYuvU16Dyadic(c, base, source, output, r, unsigned(level));
          return;
        }
      }
      if (c->format.bits <= 10)
        MultiplyYuvFloatCandidate<uint16_t, masked>(c, base, source, masks, output, r);
      else
        MultiplyYuvInteger16<masked>(c, base, source, masks, output, r);
      return;
    }
  }
  if constexpr (std::is_same<T, uint8_t>::value) {
    if (contiguous) {
#if HWY_TARGET <= HWY_AVX2
      if constexpr (!masked) {
        const double level = c->opacity * 256;
        if (int64_t(r.width) * r.count >= 65536 && level > 0 && level < 256 && level == std::floor(level)) {
          MultiplyYuvU8Dyadic(c, base, source, output, r, unsigned(level));
          return;
        }
      }
#endif
      MultiplyYuvFloatCandidate<uint8_t, masked>(c, base, source, masks, output, r);
      return;
    }
  }
  const size_t width = static_cast<size_t>(r.width), end = width - width % n;
  for (int y = r.first; y < r.first + r.count; ++y) {
    const T* ap[3];
    const T* mp[3];
    T* dst[3];
    const auto* gp = reinterpret_cast<const T*>(address(source.y, 0, y));
    for (int p = 0; p < 3; ++p) {
      ap[p] = reinterpret_cast<const T*>(address(a[p], 0, y));
      mp[p] = masked ? reinterpret_cast<const T*>(address(m[p], 0, y)) : nullptr;
      dst[p] = reinterpret_cast<T*>(address(out[p], 0, y));
    }
    const auto block = [&](auto direct, size_t xx, size_t count) HWY_ATTR {
      const int x = static_cast<int>(xx);
      const auto load = [&](cp_const_plane plane, const T* ptr) HWY_ATTR {
        if constexpr (decltype(direct)::value)
          return hn::LoadU(dt, ptr + xx);
        else
          return LoadChannel(dt, plane, x, y, count);
      };
      const auto guide = promote(load(source.y, gp));
      // MSVC can treat decltype of a reference-captured tag as a reference here.
      // Name the tag type directly so VFromD always receives a value type.
      hn::VFromD<SampleTag> values_0, values_1, values_2;
      const auto channel = [&](int p) HWY_ATTR {
        const auto original = load(a[p], ap[p]);
        const auto av = promote(original);
        const auto neutral = hn::Set(d, p == 0 || std::is_same<T, float>::value ? 0 : (maximum + 1) / 2);
        const auto w = masked ? hn::Mul(opacity, divide(promote(load(m[p], mp[p])))) : opacity;
        auto target = hn::Add(neutral, divide(hn::Mul(hn::Sub(av, neutral), guide)));
        if constexpr (std::is_same<T, float>::value)
          // MSVC may fold intrinsic +0 and retain a negative zero product.
          // The scalar neutral addition produces +0; weight-zero still copies
          // the original sample bits below.
          target = hn::IfThenElse(hn::Eq(target, zero), zero, target);
        auto result = Mix(d, av, target, w);
        if constexpr (interior)
          // Only this specialization has 0 < opacity < 1 and no mask.
          result = hn::Add(av, hn::Mul(hn::Sub(target, av), opacity));
        if constexpr (std::is_same<T, float>::value)
          VectorChannel(values_0, values_1, values_2, p) = hn::DemoteTo(dt, result);
        else {
          result = hn::Min(max, hn::Max(zero, result));
          VectorChannel(values_0, values_1, values_2, p) =
              hn::DemoteTo(dt, hn::DemoteInRangeTo(di, hn::Add(result, hn::Set(d, .5))));
        }
        if constexpr (!interior)
          VectorChannel(values_0, values_1, values_2, p) = hn::IfThenElse(NarrowMask(dt, d, hn::Eq(w, zero)), original,
                                                                        VectorChannel(values_0, values_1, values_2, p));
      };
      if constexpr (interior) {
        channel(0);
        channel(1);
        channel(2);
      } else {
        for (int p = 0; p < 3; ++p)
          channel(p);
      }
      for (int p = 0; p < 3; ++p) {
        if constexpr (decltype(direct)::value)
          hn::StoreU(VectorChannel(values_0, values_1, values_2, p), dt, dst[p] + xx);
        else
          StoreChannel(VectorChannel(values_0, values_1, values_2, p), dt, out[p], x, y, count);
      }
    };
    size_t x = 0;
    if (contiguous)
      for (; x < end; x += n)
        block(std::true_type{}, x, n);
    for (; x < width; x += n)
      block(std::false_type{}, x, std::min(n, width - x));
  }
}

// Finite F32 fast path with an input-scale error budget. Independent
// opacity complements avoid cancellation when opacity approaches one.
template <bool masked>
void MultiplyYuvFloatShared(const cp_yuv_config* c, cp_const_yuv a, cp_const_yuv b,
                            const cp_const_yuv* m, cp_yuv out, cp_rows r) {
  const double opacity_value = c->opacity;
  const bool low_opacity = opacity_value <= .75;
  const bool near_full = opacity_value > .75 && opacity_value < 1;
  const hn::ScalableTag<float> d;
  const size_t n = hn::Lanes(d);
  const auto zero = hn::Zero(d), one = hn::Set(d, 1.f);
  const auto opacity = hn::Set(d, float(opacity_value)), inverse = hn::Set(d, float(1.0 - opacity_value));
  const auto safe = hn::Set(d, std::numeric_limits<float>::max() *
                               (1.f - 64 * std::numeric_limits<float>::epsilon()));
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    for (; x + n <= size_t(r.width); x += n) {
      const auto load = [&](cp_const_plane p) HWY_ATTR {
        return hn::LoadU(d, reinterpret_cast<const float*>(address(p, int(x), y)));
      };
      const auto guide = load(b.y), mask = masked ? load(m->y) : one;
      const auto ay = load(a.y), au = load(a.u), av = load(a.v);
      const auto factor = low_opacity
          ? hn::Add(one, hn::Mul(hn::Mul(opacity, mask), hn::Sub(guide, one)))
          : hn::Add(hn::Sub(one, mask), hn::Mul(mask, hn::Add(inverse, hn::Mul(opacity, guide))));
      const auto ry = hn::Mul(ay, factor), ru = hn::Mul(au, factor), rv = hn::Mul(av, factor);
      auto results = hn::And(hn::Le(hn::Abs(ry), safe), hn::And(hn::Le(hn::Abs(ru), safe), hn::Le(hn::Abs(rv), safe)));
      if (near_full) {
        const auto conditioned = [&](auto a, auto r) HWY_ATTR {
          return hn::Le(hn::Abs(a), hn::Mul(hn::Set(d, 1048576.f), hn::Max(one, hn::Abs(r))));
        };
        results = hn::And(results, hn::And(conditioned(ay, ry), hn::And(conditioned(au, ru), conditioned(av, rv))));
      }
      if (!hn::AllTrue(d, results)) {
        const auto input = [&](cp_const_plane p) { return cp_const_plane{address(p, int(x), y), p.stride, p.step}; };
        const auto output = [&](cp_plane p) { return cp_plane{address(p, int(x), y), p.stride, p.step}; };
        const cp_const_yuv aa{input(a.y), input(a.u), input(a.v)}, bb{input(b.y), input(b.u), input(b.v)};
        const cp_const_yuv mm = masked ? cp_const_yuv{input(m->y), input(m->u), input(m->v)} : cp_const_yuv{};
        MultiplyYuvRows<float, masked>(c, aa, bb, masked ? &mm : nullptr, {output(out.y), output(out.u), output(out.v)}, {int(n), 1, 0, 1});
        continue;
      }
      const auto finish = [&](auto original, auto result) HWY_ATTR {
        // Exact zero products add +0; negative underflow must retain -0.
        result = hn::IfThenElse(hn::Or(hn::Eq(original, zero), hn::Eq(factor, zero)), zero, result);
        return hn::IfThenElse(hn::Eq(mask, zero), original, result);
      };
      const auto yy = finish(ay, ry), uu = finish(au, ru), vv = finish(av, rv);
      hn::StoreU(yy, d, reinterpret_cast<float*>(address(out.y, int(x), y)));
      hn::StoreU(uu, d, reinterpret_cast<float*>(address(out.u, int(x), y)));
      hn::StoreU(vv, d, reinterpret_cast<float*>(address(out.v, int(x), y)));
    }
    MultiplyYuvTail(c, a, b, m, out, {r.width, r.height, y, 1}, x);
  }
}

// Integer Add/Subtract have exact code weights. Their odd denominator
// keeps delta rounding away from half ties (at least 1/(2*M)), much farther
// than binary64 normalization/multiplication error. Desaturation divides by
// 2^(bits-3), so its signed floor is an exact arithmetic right shift.
template <class T, bool add>
void IntegerYuvAddSubtract(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* mask,
                           cp_yuv output, cp_rows r) {
  // U8 division correction is <=65407 and signed desaturation products
  // are bounded by 256*32, so 16-bit lanes suffice throughout.
  using Acc = typename std::conditional<sizeof(T) == 1, uint16_t, uint32_t>::type;
  using Signed = typename std::make_signed<Acc>::type;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<Signed, decltype(d)> di;
  const hn::Rebind<T, decltype(d)> dt;
  const int bits = c->format.bits;
  const uint32_t maximum = (1u << bits) - 1, center_code = 1u << (bits - 1), over_code = 1u << (bits - 3);
  const auto half = hn::Set(d, Acc(maximum / 2));
  const auto level = hn::Set(d, Acc(std::floor(c->opacity * maximum + .5)));
  const auto zero = hn::Zero(di), max = hn::Set(di, Signed(maximum)), center = hn::Set(di, Signed(center_code));
  const auto over = hn::Set(di, Signed(over_code));
  const cp_const_plane a[3]{base.y, base.u, base.v}, b[3]{source.y, source.u, source.v};
  const cp_const_plane m[3]{mask ? mask->y : cp_const_plane{}, mask ? mask->u : cp_const_plane{},
                            mask ? mask->v : cp_const_plane{}};
  const cp_plane out[3]{output.y, output.u, output.v};
  const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
  // YuvRows has established contiguous samples. Split only the bounded tail.
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto block = [&](auto direct, size_t xx, size_t count) HWY_ATTR {
      const int x = static_cast<int>(xx);
      const auto load = [&](cp_const_plane p) HWY_ATTR {
        if constexpr (decltype(direct)::value)
          return hn::LoadU(dt, reinterpret_cast<const T*>(address(p, 0, y)) + xx);
        else
          return LoadChannel(dt, p, x, y, count);
      };
      const auto store = [&](auto value, cp_plane p) HWY_ATTR {
        if constexpr (decltype(direct)::value)
          hn::StoreU(value, dt, reinterpret_cast<T*>(address(p, 0, y)) + xx);
        else
          StoreChannel(value, dt, p, x, y, count);
      };
      hn::VFromD<decltype(di)> v_0, v_1, v_2;
      for (int p = 0; p < 3; ++p) {
        const auto av = hn::PromoteTo(di, load(a[p]));
        const auto bv = hn::PromoteTo(di, load(b[p]));
        const auto weight =
            mask ? DivideCode(d, hn::Add(hn::Mul(hn::PromoteTo(d, load(m[p])), level), half),
                              bits)
                 : level;
        const auto difference = hn::Sub(bv, p ? center : zero);
        const auto magnitude = hn::BitCast(d, hn::Abs(difference));
        const auto rounded = hn::BitCast(di, DivideCode(d, hn::Add(hn::Mul(magnitude, weight), half), bits));
        const auto delta = hn::IfThenElse(hn::Lt(difference, zero), hn::Neg(rounded), rounded);
        VectorChannel(v_0, v_1, v_2, p) = add ? hn::Add(av, delta) : hn::Sub(av, delta);
      }
      const auto amount = add ? hn::Sub(hn::Set(di, Signed(maximum + 1 + over_code)), v_0) : hn::Add(over, v_0);
      const auto keep = hn::Min(over, hn::Max(zero, amount));
      v_0 = hn::Min(max, hn::Max(zero, v_0));
      for (int p = 1; p < 3; ++p)
        VectorChannel(v_0, v_1, v_2, p) = hn::Min(
            max,
            hn::Max(zero,
                    hn::Add(center, hn::ShiftRightSame(hn::Mul(hn::Sub(VectorChannel(v_0, v_1, v_2, p), center), keep),
                                                       bits - 3))));
      for (int p = 0; p < 3; ++p)
        store(hn::DemoteTo(dt, VectorChannel(v_0, v_1, v_2, p)), out[p]);
    };
    size_t x = 0;
    for (; x < end; x += n) block(std::true_type{}, x, n);
    if (x < width) block(std::false_type{}, x, width - x);
  }
}

// The artistic targets are integers before mixing. Splitting a delta larger
// than M into M plus its remainder keeps every product and DivideCode correction
// in the same proven unsigned range, including Difference's 1.5*M targets and
// HardLight's -M-1 delta. Odd-denominator rounding has no half ties.
template <class T, int operation>
void IntegerYuvArtistic(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* mask,
                        cp_yuv output, cp_rows r) {
  using Acc = typename std::conditional<sizeof(T) == 1, uint16_t, uint32_t>::type;
  using Signed = typename std::make_signed<Acc>::type;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<Signed, decltype(d)> di;
  const hn::Rebind<T, decltype(d)> dt;
  const int bits = c->format.bits;
  const uint32_t maximum = (1u << bits) - 1, center_code = 1u << (bits - 1), over_code = 1u << (bits - 3);
  const auto half = hn::Set(d, Acc(maximum / 2)), maxu = hn::Set(d, Acc(maximum));
  const auto level = hn::Set(d, Acc(std::floor(c->opacity * maximum + .5)));
  const auto zero = hn::Zero(di), max = hn::Set(di, Signed(maximum)), center = hn::Set(di, Signed(center_code));
  const auto over = hn::Set(di, Signed(over_code));
  const cp_const_plane a[3]{base.y, base.u, base.v}, b[3]{source.y, source.u, source.v};
  const cp_const_plane m[3]{mask ? mask->y : cp_const_plane{}, mask ? mask->u : cp_const_plane{},
                            mask ? mask->v : cp_const_plane{}};
  const cp_plane out[3]{output.y, output.u, output.v};
  const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
  // YuvRows has established contiguous samples. Split only the bounded tail.
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto block = [&](auto direct, size_t xx, size_t count) HWY_ATTR {
      const int x = static_cast<int>(xx);
      const auto load = [&](cp_const_plane p) HWY_ATTR {
        if constexpr (decltype(direct)::value)
          return hn::LoadU(dt, reinterpret_cast<const T*>(address(p, 0, y)) + xx);
        else
          return LoadChannel(dt, p, x, y, count);
      };
      const auto store = [&](auto value, cp_plane p) HWY_ATTR {
        if constexpr (decltype(direct)::value)
          hn::StoreU(value, dt, reinterpret_cast<T*>(address(p, 0, y)) + xx);
        else
          StoreChannel(value, dt, p, x, y, count);
      };
      hn::VFromD<decltype(di)> v_0, v_1, v_2;
      auto guide = hn::Zero(d);
      if constexpr (operation == CP_YUV_EXCLUSION)
        guide = hn::PromoteTo(d, load(b[0]));
      const auto channel = [&](auto index) HWY_ATTR {
        const int p = index;
        const auto av = hn::PromoteTo(di, load(a[p]));
        const auto bv = hn::PromoteTo(di, load(b[p]));
        const auto weight =
            mask ? DivideCode(d, hn::Add(hn::Mul(hn::PromoteTo(d, load(m[p])), level), half),
                              bits)
                 : level;
        auto difference = hn::Sub(bv, center);
        if constexpr (operation == CP_YUV_HARD_LIGHT) {
          if (!p)
            difference = hn::Add(difference, difference);
        } else if constexpr (operation == CP_YUV_DIFFERENCE) {
          difference = hn::Sub(hn::Add(hn::Abs(hn::Sub(av, bv)), center), av);
        } else if constexpr (operation == CP_YUV_EXCLUSION) {
          const auto au = hn::BitCast(d, av);
          const auto numerator = hn::Add(hn::Mul(hn::Sub(maxu, au), guide), hn::Mul(hn::Sub(maxu, guide), au));
          difference = hn::Sub(hn::BitCast(di, DivideCode(d, numerator, bits)), av);
        }
        auto magnitude = hn::BitCast(d, hn::Abs(difference));
        const auto large = hn::Gt(magnitude, maxu);
        magnitude = hn::IfThenElse(large, hn::Sub(magnitude, maxu), magnitude);
        auto quotient = DivideCode(d, hn::Add(hn::Mul(magnitude, weight), half), bits);
        quotient = hn::Add(quotient, hn::IfThenElse(large, weight, hn::Zero(d)));
        const auto rounded = hn::BitCast(di, quotient);
        const auto delta = hn::IfThenElse(hn::Lt(difference, zero), hn::Neg(rounded), rounded);
        return hn::Add(av, delta);
      };
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES
      if constexpr (operation == CP_YUV_DIFFERENCE || operation == CP_YUV_EXCLUSION) {
        v_0 = channel(std::integral_constant<int, 0>{});
        v_1 = channel(std::integral_constant<int, 1>{});
        v_2 = channel(std::integral_constant<int, 2>{});
      } else
#endif
      {
        for (int p = 0; p < 3; ++p)
          VectorChannel(v_0, v_1, v_2, p) = channel(p);
      }
      const auto upper = hn::Sub(hn::Set(di, Signed(maximum + 1 + over_code)), v_0);
      const auto lower = hn::Add(over, v_0);
      const auto keep = hn::Max(zero, hn::Min(over, hn::Min(upper, lower)));
      v_0 = hn::Min(max, hn::Max(zero, v_0));
      for (int p = 1; p < 3; ++p)
        VectorChannel(v_0, v_1, v_2, p) = hn::Min(
            max,
            hn::Max(zero,
                    hn::Add(center, hn::ShiftRightSame(hn::Mul(hn::Sub(VectorChannel(v_0, v_1, v_2, p), center), keep),
                                                       bits - 3))));
      for (int p = 0; p < 3; ++p)
        store(hn::DemoteTo(dt, VectorChannel(v_0, v_1, v_2, p)), out[p]);
    };
    size_t x = 0;
    for (; x < end; x += n) block(std::true_type{}, x, n);
    if (x < width) block(std::false_type{}, x, width - x);
  }
}

// Float YUV supports only Add/Subtract here. Resolve operation and masking
// once, and use direct contiguous loads except for the bounded final vector.
// Arithmetic stays binary64, including the original desaturation division.
template <bool add, bool masked, bool shared = false>
void FloatYuvAddSubtract(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* masks,
                         cp_yuv output, cp_rows r) {
  const hn::ScalableTag<double> d;
  using SampleTag = hn::Rebind<float, hn::ScalableTag<double>>;
  const SampleTag dt;
  const double opacity_value = c->opacity;
  const cp_const_plane a[3]{base.y, base.u, base.v}, b[3]{source.y, source.u, source.v};
  const cp_const_plane m[3]{masked ? masks->y : cp_const_plane{}, masked ? masks->u : cp_const_plane{},
                            masked ? masks->v : cp_const_plane{}};
  if constexpr (masked && !shared) {
    if (m[0].data == m[1].data && m[0].data == m[2].data &&
        m[0].stride == m[1].stride && m[0].stride == m[2].stride &&
        m[0].step == m[1].step && m[0].step == m[2].step) {
      FloatYuvAddSubtract<add, masked, true>(c, base, source, masks, output, r);
      return;
    }
  }
  const cp_plane out[3]{output.y, output.u, output.v};
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width), end = width - width % n;
  for (int y = r.first; y < r.first + r.count; ++y) {
    const float* ap[3];
    const float* bp[3];
    const float* mp[3];
    float* dst[3];
    for (int p = 0; p < 3; ++p) {
      ap[p] = reinterpret_cast<const float*>(address(a[p], 0, y));
      bp[p] = reinterpret_cast<const float*>(address(b[p], 0, y));
      mp[p] = masked ? reinterpret_cast<const float*>(address(m[p], 0, y)) : nullptr;
      dst[p] = reinterpret_cast<float*>(address(out[p], 0, y));
    }
    const auto block = [&](auto direct, size_t xx, size_t count) HWY_ATTR {
      const auto zero = hn::Zero(d), one = hn::Set(d, 1);
      const auto over = hn::Set(d, 32.0 / 255), upper_limit = hn::Set(d, 1 + 32.0 / 255);
      const int x = static_cast<int>(xx);
      const auto load = [&](cp_const_plane plane, const float* ptr) HWY_ATTR {
        if constexpr (decltype(direct)::value)
          return hn::LoadU(dt, ptr + xx);
        else
          return LoadChannel(dt, plane, x, y, count);
      };
      const auto original_y = load(a[0], ap[0]), original_u = load(a[1], ap[1]), original_v = load(a[2], ap[2]);
      const auto opacity = hn::Set(d, opacity_value);
      const auto common_weight = masked && shared
          ? hn::Mul(opacity, hn::PromoteTo(d, load(m[0], mp[0]))) : opacity;
      auto unchanged = shared ? hn::Eq(common_weight, zero) : hn::Eq(zero, zero);
      const auto blend = [&](auto original, int p) HWY_ATTR {
        const auto av = hn::PromoteTo(d, original), bv = hn::PromoteTo(d, load(b[p], bp[p]));
        const auto weight = masked && !shared ? hn::Mul(opacity, hn::PromoteTo(d, load(m[p], mp[p]))) : common_weight;
        if (!shared)
          unchanged = hn::And(unchanged, hn::Eq(weight, zero));
        const auto delta = hn::Mul(bv, weight);
        return add ? hn::Add(av, delta) : hn::Sub(av, delta);
      };
      auto vy = blend(original_y, 0);
      const auto vu = blend(original_u, 1), vv = blend(original_v, 2);
      const auto overflow = add ? hn::Gt(vy, one) : hn::Lt(vy, zero);
      const auto fade = add ? hn::Div(hn::Sub(upper_limit, vy), over) : hn::Add(one, hn::Div(vy, over));
      // Ordered comparisons reproduce std::max(0, NaN), and avoid clamping
      // the opposite luma endpoint (float Add/Subtract deliberately differ).
      const auto keep = hn::IfThenElse(overflow, hn::IfThenElse(hn::Gt(fade, zero), fade, zero), one);
      vy = hn::IfThenElse(overflow, add ? one : zero, vy);
      const auto desaturate = [&](auto value, auto keep) HWY_ATTR {
        const auto zero = hn::Zero(d), one = hn::Set(d, 1);
        auto faded = hn::Mul(value, keep);
        // Scalar chroma includes +0 after multiplication; retain its zero sign.
        faded = hn::IfThenElse(hn::Eq(faded, zero), zero, faded);
        return hn::IfThenElse(hn::Ne(keep, one), faded, value);
      };
      // All source, base and mask channels have been loaded before any store,
      // including original endpoint bits needed for exact in-place execution.
      const auto store = [&](auto original, auto value, auto unchanged, int p) HWY_ATTR {
        const auto result = hn::IfThenElse(NarrowMask(dt, d, unchanged), original, hn::DemoteTo(dt, value));
        if constexpr (decltype(direct)::value)
          hn::StoreU(result, dt, dst[p] + xx);
        else
          StoreChannel(result, dt, out[p], x, y, count);
      };
      store(original_y, vy, unchanged, 0);
      store(original_u, desaturate(vu, keep), unchanged, 1);
      store(original_v, desaturate(vv, keep), unchanged, 2);
    };
    size_t x = 0;
    for (; x < end; x += n)
      block(std::true_type{}, x, n);
    if (x < width)
      block(std::false_type{}, x, width - x);
  }
}

template <class T>
void YuvRows(const cp_yuv_config* c, cp_const_yuv base, cp_const_yuv source, const cp_const_yuv* mask, cp_yuv output,
             cp_rows r) {
  const cp_const_plane inputs[6]{base.y, base.u, base.v, source.y, source.u, source.v};
  const cp_plane outputs[3]{output.y, output.u, output.v};
  const cp_const_plane masks[3]{mask ? mask->y : cp_const_plane{}, mask ? mask->u : cp_const_plane{},
                                mask ? mask->v : cp_const_plane{}};
  bool contiguous = true;
  for (const auto& input : inputs)
    contiguous = contiguous && input.step == sizeof(T);
  for (int p = 0; p < 3; ++p)
    contiguous = contiguous && outputs[p].step == sizeof(T) && (!mask || masks[p].step == sizeof(T));
  if (!contiguous) {
    if (mask)
      YuvStepped<T, true, false>(c, base, source, mask, output, r);
    else
      YuvStepped<T, false, false>(c, base, source, mask, output, r);
    return;
  }
  if constexpr (!std::is_same<T, float>::value) {
    switch (c->operation) {
      case CP_YUV_ADD:
        IntegerYuvAddSubtract<T, true>(c, base, source, mask, output, r);
        return;
      case CP_YUV_SUBTRACT:
        IntegerYuvAddSubtract<T, false>(c, base, source, mask, output, r);
        return;
      case CP_YUV_SOFT_LIGHT:
        IntegerYuvArtistic<T, CP_YUV_SOFT_LIGHT>(c, base, source, mask, output, r);
        return;
      case CP_YUV_HARD_LIGHT:
        IntegerYuvArtistic<T, CP_YUV_HARD_LIGHT>(c, base, source, mask, output, r);
        return;
      case CP_YUV_DIFFERENCE:
        IntegerYuvArtistic<T, CP_YUV_DIFFERENCE>(c, base, source, mask, output, r);
        return;
      case CP_YUV_EXCLUSION:
        IntegerYuvArtistic<T, CP_YUV_EXCLUSION>(c, base, source, mask, output, r);
        return;
    }
  } else {
    if (c->operation == CP_YUV_ADD) {
      if (mask)
        FloatYuvAddSubtract<true, true>(c, base, source, mask, output, r);
      else
        FloatYuvAddSubtract<true, false>(c, base, source, mask, output, r);
    } else {
      if (mask)
        FloatYuvAddSubtract<false, true>(c, base, source, mask, output, r);
      else
        FloatYuvAddSubtract<false, false>(c, base, source, mask, output, r);
    }
  }
}
#endif
int Yuv(const cp_yuv_config* c, cp_const_yuv a, cp_const_yuv b, const cp_const_yuv* m, cp_yuv out, cp_rows r) {
  if (!c || !bytes(c->format) || !rows_ok(r) || !opacity_ok(c->opacity) || c->operation < CP_YUV_ADD ||
      c->operation > CP_YUV_MULTIPLY)
    return CP_INVALID_ARGUMENT;
  if (c->format.storage == CP_F32 && c->operation > CP_YUV_SUBTRACT && c->operation != CP_YUV_MULTIPLY)
    return CP_UNSUPPORTED;
  if (!r.count)
    return CP_OK;
  const int size = bytes(c->format);
  if (!plane_ok(a.y, r, size) || !plane_ok(a.u, r, size) || !plane_ok(a.v, r, size) || !plane_ok(b.y, r, size) ||
      !plane_ok(b.u, r, size) || !plane_ok(b.v, r, size) || !plane_ok(out.y, r, size) || !plane_ok(out.u, r, size) ||
      !plane_ok(out.v, r, size) ||
      (m && (!plane_ok(m->y, r, size) || !plane_ok(m->u, r, size) || !plane_ok(m->v, r, size))))
    return CP_INVALID_ARGUMENT;
#if HWY_HAVE_FLOAT64
  if (c->operation == CP_YUV_MULTIPLY) {
    if (c->format.storage == CP_F32 && m && c->opacity > 0 && c->opacity <= 1 &&
        m->y.data == m->u.data && m->y.data == m->v.data &&
        m->y.stride == m->u.stride && m->y.stride == m->v.stride &&
        m->y.step == 4 && m->u.step == 4 && m->v.step == 4 &&
        a.y.step == 4 && a.u.step == 4 && a.v.step == 4 && b.y.step == 4 &&
        out.y.step == 4 && out.u.step == 4 && out.v.step == 4) {
      MultiplyYuvFloatShared<true>(c, a, b, m, out, r);
      return CP_OK;
    }
    if (c->format.storage == CP_F32 && !m && c->opacity > 0 && c->opacity < 1 &&
        a.y.step == 4 && a.u.step == 4 && a.v.step == 4 && b.y.step == 4 &&
        out.y.step == 4 && out.u.step == 4 && out.v.step == 4) {
      MultiplyYuvFloatShared<false>(c, a, b, nullptr, out, r);
      return CP_OK;
    }
    if (c->format.storage == CP_U8) {
      if (m)
        MultiplyYuvRows<uint8_t, true>(c, a, b, m, out, r);
      else
        MultiplyYuvRows<uint8_t, false>(c, a, b, m, out, r);
    } else if (c->format.storage == CP_U16) {
      if (m)
        MultiplyYuvRows<uint16_t, true>(c, a, b, m, out, r);
      else
        MultiplyYuvRows<uint16_t, false>(c, a, b, m, out, r);
    } else {
      if (m)
        MultiplyYuvRows<float, true>(c, a, b, m, out, r);
      else
        MultiplyYuvRows<float, false>(c, a, b, m, out, r);
    }
    return CP_OK;
  }
  if (c->format.storage == CP_U8)
    YuvRows<uint8_t>(c, a, b, m, out, r);
  else if (c->format.storage == CP_U16)
    YuvRows<uint16_t>(c, a, b, m, out, r);
  else
    YuvRows<float>(c, a, b, m, out, r);
  return CP_OK;
#else
  return cp_process_yuv(c, a, b, m, out, r);
#endif
}

template <class D>
hn::VFromD<D> SampleTap(D d, cp_const_plane source, const cp_sampling* s, int x, int y, int dx, int dy, size_t count) {
  using T = hn::TFromD<D>;
  const int64_t sx = int64_t(x) * s->subsample_x + s->origin_x + dx;
  const int sy =
      static_cast<int>(std::clamp<int64_t>(int64_t(y) * s->subsample_y + s->origin_y + dy, 0, s->source_height - 1));
  const int64_t last = sx + static_cast<int64_t>(count - 1) * s->subsample_x;
  // A contiguous source grants access to every sample in this span. Deinterleave
  // an interior vector instead of staging each downsampled tap separately.
  // This permission does not extend to neighboring channels of stepped views.
  if (source.step == sizeof(T) && count == hn::Lanes(d) && sx >= 0 &&
      sx + static_cast<int64_t>(count) * s->subsample_x <= s->source_width) {
    const auto* p = reinterpret_cast<const T*>(address(source, static_cast<int>(sx), sy));
    if (s->subsample_x == 2) {
      hn::VFromD<D> even, odd;
      hn::LoadInterleaved2(d, p, even, odd);
      return even;
    }
    if (s->subsample_x == 4) {
      hn::VFromD<D> a, b, c, e;
      hn::LoadInterleaved4(d, p, a, b, c, e);
      return a;
    }
  }
  if (sx >= 0 && last < s->source_width && source.step <= std::numeric_limits<ptrdiff_t>::max() / s->subsample_x) {
    source.data = address(source, static_cast<int>(sx), sy);
    source.step *= s->subsample_x;
    return LoadChannel(d, source, 0, 0, count);
  }
  HWY_ALIGN T values[hn::MaxLanes(d)] = {};
  for (size_t i = 0; i < count; ++i) {
    const int px =
        static_cast<int>(std::clamp<int64_t>(sx + static_cast<int64_t>(i) * s->subsample_x, 0, s->source_width - 1));
    std::memcpy(values + i, address(source, px, sy), sizeof(T));
  }
  return hn::LoadU(d, values);
}
// Interior CENTER 422/420 box sampling: load adjacent taps together. The caller
// proves the whole rectangle is inside the source, so full vectors need no clamp.
template <class T>
void BoxRows(cp_const_plane source, cp_plane out, const cp_sampling* s, cp_rows r) {
  using Acc = typename std::conditional<std::is_same<T, float>::value, float,
                                        typename std::conditional<sizeof(T) == 1, uint16_t, uint32_t>::type>::type;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d), width = static_cast<size_t>(r.width);
  const auto promote = [&](auto v) HWY_ATTR {
    if constexpr (std::is_same<T, float>::value)
      return v;
    else
      return hn::PromoteTo(d, v);
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    const int sy = static_cast<int>(int64_t(s->origin_y) + int64_t(y) * s->subsample_y);
    const auto* row0 = reinterpret_cast<const T*>(address(source, s->origin_x, sy));
    const auto* row1 = s->subsample_y == 2 ? reinterpret_cast<const T*>(address(source, s->origin_x, sy + 1)) : row0;
    auto* dst = reinterpret_cast<T*>(address(out, 0, y));
    for (size_t x = 0; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      if constexpr (!std::is_same<T, float>::value) {
        if (count == n) {
          // Widen adjacent integer pairs directly, without deinterleaving
          // and promoting each tap. Even four full-range U16 taps fit U32.
          const hn::Repartition<T, decltype(d)> pairs;
          auto sum = hn::SumsOf2(hn::LoadU(pairs, row0 + 2 * x));
          if (s->subsample_y == 2)
            sum = hn::Add(sum, hn::SumsOf2(hn::LoadU(pairs, row1 + 2 * x)));
          const auto value = s->subsample_y == 2 ? hn::ShiftRight<2>(hn::Add(sum, hn::Set(d, 2)))
                                                 : hn::ShiftRight<1>(hn::Add(sum, hn::Set(d, 1)));
          hn::StoreU(hn::DemoteTo(dt, value), dt, dst + x);
          continue;
        }
      }
      hn::VFromD<decltype(dt)> a, b, c = hn::Zero(dt), e = hn::Zero(dt);
      if (count == n) {
        hn::LoadInterleaved2(dt, row0 + 2 * x, a, b);
        if (s->subsample_y == 2)
          hn::LoadInterleaved2(dt, row1 + 2 * x, c, e);
      } else {
        a = SampleTap(dt, source, s, static_cast<int>(x), y, 0, 0, count);
        b = SampleTap(dt, source, s, static_cast<int>(x), y, 1, 0, count);
        if (s->subsample_y == 2) {
          c = SampleTap(dt, source, s, static_cast<int>(x), y, 0, 1, count);
          e = SampleTap(dt, source, s, static_cast<int>(x), y, 1, 1, count);
        }
      }
      auto sum = hn::Add(promote(a), promote(b));
      // Preserve scalar binary32 left-to-right addition, including guide masks.
      if (s->subsample_y == 2)
        sum = hn::Add(hn::Add(sum, promote(c)), promote(e));
      hn::VFromD<decltype(dt)> value;
      if constexpr (std::is_same<T, float>::value)
        value = hn::Mul(sum, hn::Set(d, s->subsample_y == 2 ? .25f : .5f));
      else
        value = hn::DemoteTo(dt, s->subsample_y == 2 ? hn::ShiftRight<2>(hn::Add(sum, hn::Set(d, 2)))
                                                     : hn::ShiftRight<1>(hn::Add(sum, hn::Set(d, 1))));
      if (count == n)
        hn::StoreU(value, dt, dst + x);
      else
        StoreChannel(value, dt, out, static_cast<int>(x), y, count);
    }
  }
}

template <class T>
void SampleRows(cp_const_plane source, cp_plane out, const cp_sampling* s, cp_rows r) {
  if constexpr (std::is_same<T, float>::value) {
    if ((source.step != sizeof(T) || out.step != sizeof(T)) && s->placement == CP_CENTER && s->subsample_x == 2 &&
        s->origin_x >= 0 && int64_t(s->origin_x) + int64_t(r.width) * 2 <= s->source_width &&
        int64_t(s->origin_y) + int64_t(r.first) * s->subsample_y >= 0 &&
        int64_t(s->origin_y) + int64_t(r.first + r.count) * s->subsample_y <= s->source_height) {
      // The complete box is interior. Read only the selected samples; staging
      // four separate tap vectors costs more than the arithmetic on sparse views.
      const auto box = [&](auto vertical) HWY_ATTR {
        const int width = r.width, first = r.first, end = r.first + r.count;
        const int origin_x = s->origin_x, origin_y = s->origin_y;
        const ptrdiff_t input_step = source.step, output_step = out.step;
        for (int y = first; y < end; ++y) {
          const int sy = static_cast<int>(int64_t(origin_y) + int64_t(y) * decltype(vertical)::value);
          const auto* row0 = address(source, origin_x, sy);
          const auto* row1 = decltype(vertical)::value == 2 ? address(source, origin_x, sy + 1) : row0;
          auto* destination = address(out, 0, y);
          for (int x = 0; x < width; ++x) {
            const ptrdiff_t offset = ptrdiff_t(x * 2) * input_step;
            float a, b;
            std::memcpy(&a, row0 + offset, sizeof(a));
            std::memcpy(&b, row0 + offset + input_step, sizeof(b));
            float value = a + b;
            if constexpr (decltype(vertical)::value == 2) {
              float c, e;
              std::memcpy(&c, row1 + offset, sizeof(c));
              std::memcpy(&e, row1 + offset + input_step, sizeof(e));
              value = ((value + c) + e) * .25f;
            } else
              value *= .5f;
            std::memcpy(destination + ptrdiff_t(x) * output_step, &value, sizeof(value));
          }
        }
      };
      if (s->subsample_y == 2)
        box(std::integral_constant<int, 2>{});
      else
        box(std::integral_constant<int, 1>{});
      return;
    }
  }
  if (source.step == sizeof(T) && out.step == sizeof(T) && s->placement == CP_CENTER && s->subsample_x == 2 &&
      s->origin_x >= 0 && int64_t(s->origin_x) + int64_t(r.width) * 2 <= s->source_width &&
      int64_t(s->origin_y) + int64_t(r.first) * s->subsample_y >= 0 &&
      int64_t(s->origin_y) + int64_t(r.first + r.count) * s->subsample_y <= s->source_height) {
    BoxRows<T>(source, out, s, r);
    return;
  }
  using Acc = typename std::conditional<std::is_same<T, float>::value, float, uint32_t>::type;
  const hn::ScalableTag<Acc> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d);
  for (int y = r.first; y < r.first + r.count; ++y)
    for (size_t xx = 0; xx < static_cast<size_t>(r.width); xx += n) {
      const int x = static_cast<int>(xx);
      const size_t count = std::min(n, static_cast<size_t>(r.width) - xx);
      const auto tap = [&](int dx, int dy) HWY_ATTR {
        const auto v = SampleTap(dt, source, s, x, y, dx, dy, count);
        if constexpr (std::is_same<T, float>::value)
          return v;
        else
          return hn::PromoteTo(d, v);
      };
      auto value = tap(0, 0);
      int shift = 0;
      if (s->subsample_x == 1 || (s->placement == CP_TOP_LEFT && s->subsample_x != 4)) {
      } else if (s->subsample_x == 4) {
        value = hn::Add(hn::Add(hn::Add(value, tap(1, 0)), tap(2, 0)), tap(3, 0));
        shift = 2;
      } else if (s->placement == CP_CENTER) {
        value = hn::Add(value, tap(1, 0));
        shift = 1;
        if (s->subsample_y == 2) {
          value = hn::Add(hn::Add(value, tap(0, 1)), tap(1, 1));
          shift = 2;
        }
      } else if (s->subsample_y == 2) {
        const auto left = hn::Add(tap(-1, 0), tap(-1, 1)), mid = hn::Add(value, tap(0, 1)),
                   right = hn::Add(tap(1, 0), tap(1, 1));
        value = hn::Add(hn::Add(left, hn::Mul(hn::Set(d, 2), mid)), right);
        shift = 3;
      } else {
        value = hn::Add(hn::Add(tap(-1, 0), hn::Mul(hn::Set(d, 2), value)), tap(1, 0));
        shift = 2;
      }
      if constexpr (std::is_same<T, float>::value) {
        if (shift)
          value = hn::Mul(value, hn::Set(d, 1.0f / static_cast<float>(1 << shift)));
        StoreChannel(value, d, out, x, y, count);
      } else {
        if (shift)
          value = hn::ShiftRightSame(hn::Add(value, hn::Set(d, 1u << (shift - 1))), shift);
        StoreChannel(hn::DemoteTo(dt, value), dt, out, x, y, count);
      }
    }
}
int Sample(cp_format f, cp_const_plane source, cp_plane out, const cp_sampling* s, cp_rows r) {
  if (!s || !bytes(f) || !rows_ok(r) || s->source_width <= 0 || s->source_height <= 0 ||
      (s->subsample_x != 1 && s->subsample_x != 2 && s->subsample_x != 4) ||
      (s->subsample_y != 1 && s->subsample_y != 2) || (s->subsample_y == 2 && s->subsample_x != 2) ||
      s->placement < CP_CENTER || s->placement > CP_TOP_LEFT)
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  const cp_rows input = {s->source_width, s->source_height, 0, s->source_height};
  if (!plane_ok(source, input, bytes(f)) || !plane_ok(out, r, bytes(f)))
    return CP_INVALID_ARGUMENT;
  if (f.storage == CP_U8)
    SampleRows<uint8_t>(source, out, s, r);
  else if (f.storage == CP_U16)
    SampleRows<uint16_t>(source, out, s, r);
  else
    SampleRows<float>(source, out, s, r);
  return CP_OK;
}

// Gather/scatter access only the declared float samples, never neighboring
// channels or padding. Limit each vector's positive element indices to int32;
// huge legal steps and tails retain the bounded scalar path.
constexpr bool kNativeFloatGather = HWY_TARGET <= HWY_AVX2 || HWY_TARGET == HWY_SVE || HWY_TARGET == HWY_SVE2 ||
                                    HWY_TARGET == HWY_SVE_256 || HWY_TARGET == HWY_SVE2_128 || HWY_TARGET == HWY_RVV;
template <class D>
bool FloatIndicesFit(D d, ptrdiff_t step) {
  return hn::Lanes(d) > 1 && step / 4 <= std::numeric_limits<int32_t>::max() / ptrdiff_t(hn::Lanes(d) - 1);
}
template <class D>
auto FloatIndices(D, ptrdiff_t step) {
  const hn::Rebind<int32_t, D> di;
  return hn::Mul(hn::Iota(di, 0), hn::Set(di, static_cast<int32_t>(step / 4)));
}
template <class D>
auto GatherFloat(D d, const float* p, hn::VFromD<hn::Rebind<int32_t, D>> indices, ptrdiff_t step) {
  return step == 4 ? hn::LoadU(d, p) : hn::GatherIndex(d, p, indices);
}
template <class D>
void ScatterFloat(hn::VFromD<D> value, D d, float* p, hn::VFromD<hn::Rebind<int32_t, D>> indices, ptrdiff_t step) {
  if (step == 4)
    hn::StoreU(value, d, p);
  else
    hn::ScatterIndex(value, d, p, indices);
}

#if HWY_HAVE_FLOAT64
// Multiplication by 0 or +/-1 is exact, so representable offsets permit
// binary32 arithmetic without introducing an intermediate rounding error.
// Keep the multiply/add (rather than copy/fill) for signed zero and Inf/NaN.
HWY_NOINLINE void FloatSimpleAffineRows(cp_const_plane source, cp_plane out, cp_rows r, float first, float second) {
  // Use at most eight floats even on wider targets: wider arithmetic alone
  // does not improve large-plane throughput. Do not specialize by image size.
  const hn::CappedTag<float, 8> d;
  const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
  const auto a = hn::Set(d, first), b = hn::Set(d, second);
  const bool contiguous = source.step == sizeof(float) && out.step == sizeof(float);
  const ptrdiff_t input_step = source.step, output_step = out.step;
  for (int y = r.first; y < r.first + r.count; ++y) {
    const auto* src = address(source, 0, y);
    auto* dst = address(out, 0, y);
    size_t x = 0;
    if (contiguous) {
      const auto* input = reinterpret_cast<const float*>(src);
      auto* output = reinterpret_cast<float*>(dst);
      for (; x < end; x += n)
        hn::StoreU(hn::Add(hn::Mul(hn::LoadU(d, input + x), a), b), d, output + x);
      // Keep sparse-tail induction variables out of the contiguous SIMD loop.
      for (; x < width; ++x)
        output[x] = input[x] * first + second;
      continue;
    }
    for (; x < width; ++x) {
      float value;
      std::memcpy(&value, src + ptrdiff_t(x) * input_step, sizeof(value));
      value = value * first + second;
      std::memcpy(dst + ptrdiff_t(x) * output_step, &value, sizeof(value));
    }
  }
}

template <class T, bool Clamp>
void UnaryRows(cp_format f, cp_const_plane source, cp_plane out, cp_rows r, double first, double second) {
  if constexpr (!Clamp && std::is_same<T, float>::value) {
    if (first == -1 && second == 1) {
      // Exact binary32 inversion: avoid promotion, multiplication and demotion.
      // Sparse views authorize only their selected samples, not adjacent lanes.
      const hn::ScalableTag<float> d;
      const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
      const auto one = hn::Set(d, 1.0f);
      const bool contiguous = source.step == sizeof(float) && out.step == sizeof(float);
      for (int y = r.first; y < r.first + r.count; ++y) {
        size_t x = 0;
        if (contiguous) {
          const auto* src = reinterpret_cast<const float*>(address(source, 0, y));
          auto* dst = reinterpret_cast<float*>(address(out, 0, y));
          for (; x < end; x += n)
            hn::StoreU(hn::Sub(one, hn::LoadU(d, src + x)), d, dst + x);
        }
        for (; x < width; ++x) {
          float value;
          std::memcpy(&value, address(source, static_cast<int>(x), y), sizeof(value));
          value = 1.0f - value;
          std::memcpy(address(out, static_cast<int>(x), y), &value, sizeof(value));
        }
      }
      return;
    }
    if ((first == 0 || first == 1 || first == -1) && std::abs(second) <= std::numeric_limits<float>::max() &&
        double(static_cast<float>(second)) == second) {
      FloatSimpleAffineRows(source, out, r, static_cast<float>(first), static_cast<float>(second));
      return;
    }
    // Keep binary64 arithmetic, but separate complete contiguous vectors from
    // sparse samples and tails. No layout/count checks belong in the hot loop.
    const hn::ScalableTag<double> d;
    const hn::Rebind<float, decltype(d)> df;
    const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
    const auto a = hn::Set(d, first), b = hn::Set(d, second);
    const bool contiguous = source.step == sizeof(float) && out.step == sizeof(float);
    const ptrdiff_t input_step = source.step, output_step = out.step;
    for (int y = r.first; y < r.first + r.count; ++y) {
      const auto* src = address(source, 0, y);
      auto* dst = address(out, 0, y);
      size_t x = 0;
      if (contiguous) {
        const auto* input = reinterpret_cast<const float*>(src);
        auto* output = reinterpret_cast<float*>(dst);
        for (; x < end; x += n) {
          const auto value = hn::PromoteTo(d, hn::LoadU(df, input + x));
          hn::StoreU(hn::DemoteTo(df, hn::Add(hn::Mul(value, a), b)), df, output + x);
        }
      }
      for (; x < width; ++x) {
        float value;
        std::memcpy(&value, src + ptrdiff_t(x) * input_step, sizeof(value));
        value = static_cast<float>(double(value) * first + second);
        std::memcpy(dst + ptrdiff_t(x) * output_step, &value, sizeof(value));
      }
    }
    return;
  }
  if constexpr (!std::is_same<T, float>::value) {
    const double maximum = cp::maximum(f);
    if (Clamp || (first == -1 && second == maximum)) {
      // Quantization commutes with integer clamping. Inversion is an exact
      // saturated subtraction, including noncanonical input codes above max.
      const T lower = T(std::floor(std::clamp(first, 0.0, maximum) + .5));
      const T upper = T(std::floor(std::clamp(second, 0.0, maximum) + .5));
      const hn::ScalableTag<T> d;
      const size_t n = hn::Lanes(d);
      const auto low = hn::Set(d, lower), high = hn::Set(d, upper);
      for (int y = r.first; y < r.first + r.count; ++y)
        for (size_t xx = 0; xx < size_t(r.width); xx += n) {
          const int x = static_cast<int>(xx);
          const size_t count = std::min(n, size_t(r.width) - xx);
          const auto value = LoadChannel(d, source, x, y, count);
          const auto result = Clamp ? hn::Min(high, hn::Max(low, value)) : hn::SaturatedSub(high, value);
          StoreChannel(result, d, out, x, y, count);
        }
      return;
    }
  }
  if constexpr (Clamp && std::is_same<T, float>::value) {
    // Exactly representable bounds permit binary32 comparisons at twice the
    // lane count. Ordered selects preserve signed zero, and NaN maps to low.
    // Other bounds retain binary64 comparisons (rounding a bound can matter).
    if (std::abs(first) <= std::numeric_limits<float>::max() && std::abs(second) <= std::numeric_limits<float>::max()) {
      const float lower = static_cast<float>(first), upper = static_cast<float>(second);
      if (double(lower) == first && double(upper) == second) {
        const hn::ScalableTag<float> d;
        const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
        const bool contiguous = source.step == sizeof(float) && out.step == sizeof(float);
        const auto clamp = [&](auto value) HWY_ATTR {
          const auto low = hn::Set(d, lower), high = hn::Set(d, upper);
          return hn::IfThenElse(
              hn::IsNaN(value), low,
              hn::IfThenElse(hn::Lt(value, low), low, hn::IfThenElse(hn::Gt(value, high), high, value)));
        };
        for (int y = r.first; y < r.first + r.count; ++y) {
          size_t x = 0;
          if (contiguous) {
            const auto* src = reinterpret_cast<const float*>(address(source, 0, y));
            auto* dst = reinterpret_cast<float*>(address(out, 0, y));
            for (; x < end; x += n)
              hn::StoreU(clamp(hn::LoadU(d, src + x)), d, dst + x);
          }
          for (; x < width; x += n) {
            const size_t count = std::min(n, width - x);
            StoreChannel(clamp(LoadChannel(d, source, static_cast<int>(x), y, count)), d, out, static_cast<int>(x), y,
                         count);
          }
        }
        return;
      }
    }
  }
  // A single strided channel does not authorize reading adjacent channels.
  // Direct typed accesses avoid packing every vector into temporary arrays.
  if constexpr (std::is_same<T, float>::value) {
    if (source.step != sizeof(T) || out.step != sizeof(T)) {
      for (int y = r.first; y < r.first + r.count; ++y)
        for (int x = 0; x < r.width; ++x) {
          float sample;
          std::memcpy(&sample, address(source, x, y), sizeof(sample));
          double v = sample;
          if constexpr (Clamp)
            v = std::isnan(v) ? first : std::clamp(v, first, second);
          else
            v = v * first + second;
          sample = static_cast<float>(v);
          std::memcpy(address(out, x, y), &sample, sizeof(sample));
        }
      return;
    }
  }
  const hn::ScalableTag<double> d;
  const size_t n = hn::Lanes(d);
  const auto a = hn::Set(d, first), b = hn::Set(d, second);
  for (int y = r.first; y < r.first + r.count; ++y)
    for (size_t xx = 0; xx < static_cast<size_t>(r.width); xx += n) {
      const int x = static_cast<int>(xx);
      const size_t count = std::min(n, static_cast<size_t>(r.width) - xx);
      auto value = LoadDouble<T>(d, source, x, y, count);
      if constexpr (Clamp) {
        // Ordered selects retain signed zero on equality, just like std::clamp.
        value = hn::IfThenElse(hn::IsNaN(value), a,
                               hn::IfThenElse(hn::Lt(value, a), a, hn::IfThenElse(hn::Gt(value, b), b, value)));
      } else
        value = hn::Add(hn::Mul(value, a), b);
      StoreDouble<T>(value, d, f, out, x, y, count);
    }
}
#endif
template <bool Clamp>
int Unary(cp_format f, cp_const_plane source, cp_plane out, cp_rows r, double a, double b) {
  if (!bytes(f) || !rows_ok(r) || !std::isfinite(a) || !std::isfinite(b) || (Clamp && a > b))
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!plane_ok(source, r, bytes(f)) || !plane_ok(out, r, bytes(f)))
    return CP_INVALID_ARGUMENT;
#if HWY_HAVE_FLOAT64
  if (f.storage == CP_U8)
    UnaryRows<uint8_t, Clamp>(f, source, out, r, a, b);
  else if (f.storage == CP_U16)
    UnaryRows<uint16_t, Clamp>(f, source, out, r, a, b);
  else
    UnaryRows<float, Clamp>(f, source, out, r, a, b);
  return CP_OK;
#else
  return Clamp ? cp_clamp(f, source, out, r, a, b) : cp_affine(f, source, out, r, a, b);
#endif
}
int Affine(cp_format f, cp_const_plane source, cp_plane out, cp_rows r, double a, double b) {
  return Unary<false>(f, source, out, r, a, b);
}
int Clamp(cp_format f, cp_const_plane source, cp_plane out, cp_rows r, double a, double b) {
  return Unary<true>(f, source, out, r, a, b);
}

void FloatLumaRows(cp_const_rgb rgb, cp_plane out, cp_rows r) {
  if (rgb.r.step != 4 || rgb.g.step != 4 || rgb.b.step != 4 || out.step != 4) {
    for (int y = r.first; y < r.first + r.count; ++y)
      for (int x = 0; x < r.width; ++x) {
        float rr, gg, bb;
        std::memcpy(&rr, address(rgb.r, x, y), 4);
        std::memcpy(&gg, address(rgb.g, x, y), 4);
        std::memcpy(&bb, address(rgb.b, x, y), 4);
        const float value = .114f * bb + .587f * gg + .299f * rr;
        std::memcpy(address(out, x, y), &value, 4);
      }
    return;
  }
  const hn::ScalableTag<float> d;
  const size_t n = hn::Lanes(d);
  for (int y = r.first; y < r.first + r.count; ++y)
    for (size_t xx = 0; xx < size_t(r.width); xx += n) {
      const int x = static_cast<int>(xx);
      const size_t count = std::min(n, size_t(r.width) - xx);
      const auto rr = LoadChannel(d, rgb.r, x, y, count), gg = LoadChannel(d, rgb.g, x, y, count),
                 bb = LoadChannel(d, rgb.b, x, y, count);
      const auto value = hn::Add(hn::Add(hn::Mul(hn::Set(d, .114f), bb), hn::Mul(hn::Set(d, .587f), gg)),
                                 hn::Mul(hn::Set(d, .299f), rr));
      StoreChannel(value, d, out, x, y, count);
    }
}

#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES
void IntegerLumaU8Rows(cp_const_rgb rgb, cp_plane out, cp_rows r, int rounding) {
  const hn::ScalableTag<uint32_t> d;
  const hn::ScalableTag<uint16_t> d16;
  const hn::Half<decltype(d16)> dh16;
  const hn::Rebind<uint8_t, decltype(d16)> d8;
  const size_t n = hn::Lanes(d16), width = size_t(r.width), end = width - width % n;
  const bool contiguous = rgb.r.step == 1 && rgb.g.step == 1 && rgb.b.step == 1 && out.step == 1;
  const auto sum = [&](auto rv, auto gv, auto bv) HWY_ATTR {
    auto v = hn::Add(hn::Mul(hn::PromoteTo(d, rv), hn::Set(d, 9798)),
                     hn::Mul(hn::PromoteTo(d, gv), hn::Set(d, 19234)));
    v = hn::Add(v, hn::Mul(hn::PromoteTo(d, bv), hn::Set(d, 3736)));
    return hn::DemoteTo(dh16, hn::ShiftRight<15>(hn::Add(v, hn::Set(d, rounding == CP_LUMA_NEAREST ? 16384 : 0))));
  };
  const auto luma = [&](auto rr, auto gg, auto bb) HWY_ATTR {
    const auto rv = hn::PromoteTo(d16, rr), gv = hn::PromoteTo(d16, gg), bv = hn::PromoteTo(d16, bb);
    const auto low = sum(hn::LowerHalf(dh16, rv), hn::LowerHalf(dh16, gv), hn::LowerHalf(dh16, bv));
    const auto high = sum(hn::UpperHalf(dh16, rv), hn::UpperHalf(dh16, gv), hn::UpperHalf(dh16, bv));
    return hn::DemoteTo(d8, hn::Combine(d16, high, low));
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    if (contiguous) {
      const auto* rp = address(rgb.r, 0, y);
      const auto* gp = address(rgb.g, 0, y);
      const auto* bp = address(rgb.b, 0, y);
      auto* dst = address(out, 0, y);
      for (; x < end; x += n)
        hn::StoreU(luma(hn::LoadU(d8, rp + x), hn::LoadU(d8, gp + x), hn::LoadU(d8, bp + x)), d8, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      StoreChannel(luma(LoadChannel(d8, rgb.r, int(x), y, count), LoadChannel(d8, rgb.g, int(x), y, count),
                        LoadChannel(d8, rgb.b, int(x), y, count)), d8, out, int(x), y, count);
    }
  }
}
#endif

template <class T>
void IntegerLumaRows(cp_const_rgb rgb, cp_plane out, cp_rows r, int rounding) {
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_BF16 || HWY_TARGET == HWY_NEON_WITHOUT_AES
  if constexpr (std::is_same<T, uint8_t>::value) {
    IntegerLumaU8Rows(rgb, out, r, rounding);
    return;
  }
#endif
  using AccTag = hn::ScalableTag<uint32_t>;
  const AccTag d;
  const hn::Rebind<T, AccTag> dt;
  const size_t n = hn::Lanes(d), width = size_t(r.width), end = width - width % n;
  const bool contiguous =
      rgb.r.step == sizeof(T) && rgb.g.step == sizeof(T) && rgb.b.step == sizeof(T) && out.step == sizeof(T);
  const auto luma = [&](auto rr, auto gg, auto bb) HWY_ATTR {
    hn::VFromD<AccTag> rg;
    if constexpr (std::is_same<T, uint8_t>::value && !HWY_ARCH_ARM) {
      // U8 samples fit signed 16-bit lanes. One pairwise multiply-add
      // replaces two 32-bit products without changing the integer sum.
      const hn::Repartition<int16_t, AccTag> ds;
      const hn::Rebind<int32_t, AccTag> di;
      const auto pairs = hn::Or(hn::PromoteTo(d, rr), hn::ShiftLeft<16>(hn::PromoteTo(d, gg)));
      const auto weights = hn::Set(d, 9798u | (19234u << 16));
      rg = hn::BitCast(d, hn::WidenMulPairwiseAdd(di, hn::BitCast(ds, pairs), hn::BitCast(ds, weights)));
    } else {
      rg = hn::Add(hn::Mul(hn::Set(d, 9798), hn::PromoteTo(d, rr)), hn::Mul(hn::Set(d, 19234), hn::PromoteTo(d, gg)));
    }
    auto value = hn::Add(rg, hn::Mul(hn::Set(d, 3736), hn::PromoteTo(d, bb)));
    value = hn::ShiftRight<15>(hn::Add(value, hn::Set(d, rounding == CP_LUMA_NEAREST ? 16384 : 0)));
    return hn::DemoteTo(dt, value);
  };
  for (int y = r.first; y < r.first + r.count; ++y) {
    size_t x = 0;
    if (contiguous) {
      const auto* rp = reinterpret_cast<const T*>(address(rgb.r, 0, y));
      const auto* gp = reinterpret_cast<const T*>(address(rgb.g, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(rgb.b, 0, y));
      auto* dst = reinterpret_cast<T*>(address(out, 0, y));
      for (; x < end; x += n)
        hn::StoreU(luma(hn::LoadU(dt, rp + x), hn::LoadU(dt, gp + x), hn::LoadU(dt, bp + x)), dt, dst + x);
    }
    for (; x < width; x += n) {
      const size_t count = std::min(n, width - x);
      const int xx = static_cast<int>(x);
      StoreChannel(luma(LoadChannel(dt, rgb.r, xx, y, count), LoadChannel(dt, rgb.g, xx, y, count),
                        LoadChannel(dt, rgb.b, xx, y, count)),
                   dt, out, xx, y, count);
    }
  }
}

int Luma(cp_format f, cp_const_rgb rgb, cp_plane out, cp_rows r, int rounding) {
  if (!bytes(f) || !rows_ok(r) || (rounding != CP_LUMA_FLOOR && rounding != CP_LUMA_NEAREST))
    return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!plane_ok(rgb.r, r, bytes(f)) || !plane_ok(rgb.g, r, bytes(f)) || !plane_ok(rgb.b, r, bytes(f)) ||
      !plane_ok(out, r, bytes(f)))
    return CP_INVALID_ARGUMENT;
  if (f.storage == CP_U8)
    IntegerLumaRows<uint8_t>(rgb, out, r, rounding);
  else if (f.storage == CP_U16)
    IntegerLumaRows<uint16_t>(rgb, out, r, rounding);
  else
    FloatLumaRows(rgb, out, r);
  return CP_OK;
}
#if HWY_HAVE_FLOAT64
// All four lanes are declared inputs here, so a matching RGBA/BGRA view
// permits interleaved loads without reading an undeclared neighboring channel.
template <class T>
int PackedOrder(cp_const_rgb rgb, cp_const_plane alpha) {
  constexpr uintptr_t bytes = sizeof(T);
  if (rgb.r.step != 4 * bytes || rgb.g.step != 4 * bytes || rgb.b.step != 4 * bytes || alpha.step != 4 * bytes ||
      rgb.r.stride != rgb.g.stride || rgb.r.stride != rgb.b.stride || rgb.r.stride != alpha.stride)
    return 0;
  const uintptr_t r = reinterpret_cast<uintptr_t>(rgb.r.data), g = reinterpret_cast<uintptr_t>(rgb.g.data),
                  b = reinterpret_cast<uintptr_t>(rgb.b.data), a = reinterpret_cast<uintptr_t>(alpha.data);
  if (r <= std::numeric_limits<uintptr_t>::max() - 3 * bytes && g == r + bytes && b == r + 2 * bytes &&
      a == r + 3 * bytes)
    return 1;
  if (b <= std::numeric_limits<uintptr_t>::max() - 3 * bytes && g == b + bytes && r == b + 2 * bytes &&
      a == b + 3 * bytes)
    return 2;
  return 0;
}

float FloatFromRank(uint32_t rank) {
  const uint32_t bits = rank & 0x80000000u ? rank ^ 0x80000000u : ~rank;
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

// Enumerating finite binary32 values in numeric order makes each binary64
// subtraction predicate monotone. Binary searches find the exact inclusive
// float interval accepted by abs(double(pixel)-key)<=tolerance. In particular,
// do not approximate the limits with key +/- tolerance: cancellation or a
// very large key can otherwise change the original rounded subtraction.
bool FloatKeyBounds(double key, double tolerance, float& lower, float& upper) {
  const auto search = [&](bool after) HWY_ATTR {
    uint32_t lo = 0x00800000u, hi = 0xff800000u; // [-FLT_MAX, +FLT_MAX], exclusive end
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      const double delta = double(FloatFromRank(mid)) - key;
      if (after ? delta > tolerance : delta >= -tolerance)
        hi = mid;
      else
        lo = mid + 1;
    }
    return lo;
  };
  const uint32_t begin = search(false), end = search(true);
  if (begin >= end)
    return false;
  lower = FloatFromRank(begin);
  upper = FloatFromRank(end - 1);
  return true;
}

template <class D>
void FloatKeyRows(D d, cp_const_rgb rgb, cp_const_plane alpha, cp_plane out, cp_rows r, const double* key,
                  const double* tolerance) {
  float lower[3], upper[3];
  for (int p = 0; p < 3; ++p)
    if (!FloatKeyBounds(key[p], tolerance[p], lower[p], upper[p])) {
      CopyRows<float>(alpha, out, r);
      return;
    }
  const auto lr = hn::Set(d, lower[0]), lg = hn::Set(d, lower[1]), lb = hn::Set(d, lower[2]);
  const auto ur = hn::Set(d, upper[0]), ug = hn::Set(d, upper[1]), ub = hn::Set(d, upper[2]);
  const int packed_order = PackedOrder<float>(rgb, alpha);
  const bool indexed = kNativeFloatGather && FloatIndicesFit(d, rgb.r.step) && FloatIndicesFit(d, rgb.g.step) &&
                       FloatIndicesFit(d, rgb.b.step) && FloatIndicesFit(d, alpha.step) && FloatIndicesFit(d, out.step);
  // For a huge legal step, zero indices are unused and cannot overflow int32.
  const auto ri = FloatIndices(d, indexed ? rgb.r.step : 4), gi = FloatIndices(d, indexed ? rgb.g.step : 4),
             bi = FloatIndices(d, indexed ? rgb.b.step : 4), ai = FloatIndices(d, indexed ? alpha.step : 4),
             oi = FloatIndices(d, indexed ? out.step : 4);
  const size_t n = hn::Lanes(d);
  for (int y = r.first; y < r.first + r.count; ++y)
    for (size_t xx = 0; xx < size_t(r.width); xx += n) {
      const int x = static_cast<int>(xx);
      const size_t count = std::min(n, size_t(r.width) - xx);
      hn::VFromD<decltype(d)> rr, gg, bb, av;
      if (packed_order && count == n) {
        const auto first = packed_order == 1 ? rgb.r : rgb.b;
        hn::LoadInterleaved4(d, reinterpret_cast<const float*>(address(first, x, y)), rr, gg, bb, av);
        if (packed_order == 2) {
          const auto saved = rr;
          rr = bb;
          bb = saved;
        }
      } else if (indexed && count == n) {
        rr = GatherFloat(d, reinterpret_cast<const float*>(address(rgb.r, x, y)), ri, rgb.r.step);
        gg = GatherFloat(d, reinterpret_cast<const float*>(address(rgb.g, x, y)), gi, rgb.g.step);
        bb = GatherFloat(d, reinterpret_cast<const float*>(address(rgb.b, x, y)), bi, rgb.b.step);
        av = GatherFloat(d, reinterpret_cast<const float*>(address(alpha, x, y)), ai, alpha.step);
      } else {
        rr = LoadChannel(d, rgb.r, x, y, count);
        gg = LoadChannel(d, rgb.g, x, y, count);
        bb = LoadChannel(d, rgb.b, x, y, count);
        av = LoadChannel(d, alpha, x, y, count);
      }
      // Bounds are finite. Ordered comparisons reject NaN and infinities;
      // the alpha selection preserves every original payload on a miss.
      const auto hit =
          hn::And(hn::And(hn::And(hn::Ge(rr, lr), hn::Le(rr, ur)), hn::And(hn::Ge(gg, lg), hn::Le(gg, ug))),
                  hn::And(hn::Ge(bb, lb), hn::Le(bb, ub)));
      const auto value = hn::IfThenElse(hit, hn::Zero(d), av);
      if (indexed && count == n)
        ScatterFloat(value, d, reinterpret_cast<float*>(address(out, x, y)), oi, out.step);
      else
        StoreChannel(value, d, out, x, y, count);
    }
}

// Integer channels use the same monotone rounded-subtraction predicate as
// float thresholds. Search the finite code domain once, then compare native
// integer lanes without converting every pixel to double.
template <class T>
void IntegerKeyRows(cp_format f, cp_const_rgb rgb, cp_const_plane alpha, cp_plane out, cp_rows r, const double* key,
                    const double* tolerance) {
  T lower[3], upper[3];
  const uint32_t end_code = 1u << f.bits;
  for (int p = 0; p < 3; ++p) {
    const auto search = [&](bool after) HWY_ATTR {
      uint32_t lo = 0, hi = end_code;
      while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const double delta = double(mid) - key[p];
        if (after ? delta > tolerance[p] : delta >= -tolerance[p])
          hi = mid;
        else
          lo = mid + 1;
      }
      return lo;
    };
    const uint32_t begin = search(false), end = search(true);
    if (begin >= end) {
      CopyRows<T>(alpha, out, r);
      return;
    }
    lower[p] = T(begin);
    upper[p] = T(end - 1);
  }
  const hn::ScalableTag<T> d;
  const size_t n = hn::Lanes(d);
  const auto lr = hn::Set(d, lower[0]), lg = hn::Set(d, lower[1]), lb = hn::Set(d, lower[2]);
  const auto ur = hn::Set(d, upper[0]), ug = hn::Set(d, upper[1]), ub = hn::Set(d, upper[2]);
  // Hoist layout and tail handling out of the full planar vector loop.
  if (rgb.r.step == sizeof(T) && rgb.g.step == sizeof(T) && rgb.b.step == sizeof(T) && alpha.step == sizeof(T) &&
      out.step == sizeof(T)) {
    const size_t full = size_t(r.width) / n * n;
    for (int y = r.first; y < r.first + r.count; ++y) {
      const auto* rp = reinterpret_cast<const T*>(address(rgb.r, 0, y));
      const auto* gp = reinterpret_cast<const T*>(address(rgb.g, 0, y));
      const auto* bp = reinterpret_cast<const T*>(address(rgb.b, 0, y));
      const auto* ap = reinterpret_cast<const T*>(address(alpha, 0, y));
      auto* op = reinterpret_cast<T*>(address(out, 0, y));
      const auto apply = [&](auto rr, auto gg, auto bb, auto av) HWY_ATTR {
        const auto hit =
            hn::And(hn::And(hn::And(hn::Ge(rr, lr), hn::Le(rr, ur)), hn::And(hn::Ge(gg, lg), hn::Le(gg, ug))),
                    hn::And(hn::Ge(bb, lb), hn::Le(bb, ub)));
        return hn::IfThenElse(hit, hn::Zero(d), av);
      };
      for (size_t x = 0; x < full; x += n)
        hn::StoreU(apply(hn::LoadU(d, rp + x), hn::LoadU(d, gp + x), hn::LoadU(d, bp + x), hn::LoadU(d, ap + x)), d,
                   op + x);
      if (full != size_t(r.width)) {
        const int x = static_cast<int>(full);
        const size_t count = size_t(r.width) - full;
        StoreChannel(apply(LoadChannel(d, rgb.r, x, y, count), LoadChannel(d, rgb.g, x, y, count),
                           LoadChannel(d, rgb.b, x, y, count), LoadChannel(d, alpha, x, y, count)),
                     d, out, x, y, count);
      }
    }
    return;
  }
  const int order = PackedOrder<T>(rgb, alpha);
  for (int y = r.first; y < r.first + r.count; ++y)
    for (size_t xx = 0; xx < size_t(r.width); xx += n) {
      const int x = static_cast<int>(xx);
      const size_t count = std::min(n, size_t(r.width) - xx);
      hn::VFromD<decltype(d)> rr, gg, bb, av;
      if (order && count == n) {
        const auto first = order == 1 ? rgb.r : rgb.b;
        hn::LoadInterleaved4(d, reinterpret_cast<const T*>(address(first, x, y)), rr, gg, bb, av);
        if (order == 2) {
          const auto saved = rr;
          rr = bb;
          bb = saved;
        }
      } else {
        rr = LoadChannel(d, rgb.r, x, y, count);
        gg = LoadChannel(d, rgb.g, x, y, count);
        bb = LoadChannel(d, rgb.b, x, y, count);
        av = LoadChannel(d, alpha, x, y, count);
      }
      const auto hit =
          hn::And(hn::And(hn::And(hn::Ge(rr, lr), hn::Le(rr, ur)), hn::And(hn::Ge(gg, lg), hn::Le(gg, ug))),
                  hn::And(hn::Ge(bb, lb), hn::Le(bb, ub)));
      StoreChannel(hn::IfThenElse(hit, hn::Zero(d), av), d, out, x, y, count);
    }
}

template <class T>
void KeyRows(cp_format f, cp_const_rgb rgb, cp_const_plane alpha, cp_plane out, cp_rows r, const double* key,
             const double* tolerance) {
  if constexpr (std::is_same<T, float>::value) {
    // Keep the existing path for tiny requests, where bound construction
    // would cost more than the per-sample comparisons.
    if (int64_t(r.width) * r.count >= 128) {
      if (!PackedOrder<float>(rgb, alpha) &&
          (rgb.r.step != 4 || rgb.g.step != 4 || rgb.b.step != 4 || alpha.step != 4 || out.step != 4))
        FloatKeyRows(hn::CappedTag<float, 8>(), rgb, alpha, out, r, key, tolerance);
      else
        FloatKeyRows(hn::ScalableTag<float>(), rgb, alpha, out, r, key, tolerance);
      return;
    }
  }
  if constexpr (!std::is_same<T, float>::value) {
    if (int64_t(r.width) * r.count >= 128) {
      IntegerKeyRows<T>(f, rgb, alpha, out, r, key, tolerance);
      return;
    }
  }
  const hn::ScalableTag<double> d;
  const hn::Rebind<T, decltype(d)> dt;
  const size_t n = hn::Lanes(d);
  for (int y = r.first; y < r.first + r.count; ++y)
    for (size_t xx = 0; xx < static_cast<size_t>(r.width); xx += n) {
      const int x = static_cast<int>(xx);
      const size_t count = std::min(n, static_cast<size_t>(r.width) - xx);
      auto hit =
          hn::Le(hn::Abs(hn::Sub(LoadDouble<T>(d, rgb.r, x, y, count), hn::Set(d, key[0]))), hn::Set(d, tolerance[0]));
      hit = hn::And(hit, hn::Le(hn::Abs(hn::Sub(LoadDouble<T>(d, rgb.g, x, y, count), hn::Set(d, key[1]))),
                                hn::Set(d, tolerance[1])));
      hit = hn::And(hit, hn::Le(hn::Abs(hn::Sub(LoadDouble<T>(d, rgb.b, x, y, count), hn::Set(d, key[2]))),
                                hn::Set(d, tolerance[2])));
      StoreChannel(hn::IfThenElse(NarrowMask(dt, d, hit), hn::Zero(dt), LoadChannel(dt, alpha, x, y, count)), dt, out,
                   x, y, count);
    }
}
#endif
int Key(cp_format f, cp_const_rgb rgb, cp_const_plane alpha, cp_plane out, cp_rows r, const double* key,
        const double* tolerance) {
  if (!bytes(f) || !rows_ok(r) || !key || !tolerance)
    return CP_INVALID_ARGUMENT;
  for (int i = 0; i < 3; ++i)
    if (!std::isfinite(key[i]) || !std::isfinite(tolerance[i]) || tolerance[i] < 0)
      return CP_INVALID_ARGUMENT;
  if (!r.count)
    return CP_OK;
  if (!plane_ok(rgb.r, r, bytes(f)) || !plane_ok(rgb.g, r, bytes(f)) || !plane_ok(rgb.b, r, bytes(f)) ||
      !plane_ok(alpha, r, bytes(f)) || !plane_ok(out, r, bytes(f)))
    return CP_INVALID_ARGUMENT;
#if HWY_HAVE_FLOAT64
  if (f.storage == CP_U8)
    KeyRows<uint8_t>(f, rgb, alpha, out, r, key, tolerance);
  else if (f.storage == CP_U16)
    KeyRows<uint16_t>(f, rgb, alpha, out, r, key, tolerance);
  else
    KeyRows<float>(f, rgb, alpha, out, r, key, tolerance);
  return CP_OK;
#else
  return cp_color_key(f, rgb, alpha, out, r, key, tolerance);
#endif
}
