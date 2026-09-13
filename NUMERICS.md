# Composite numerical behavior

[Project overview](README.md) · [Public contracts](include/composite)

For unmasked integer `CP_MIX` with `CP_WEIGHT_CONTINUOUS`, SIMD backends may
round opacity to Q15 (`round(opacity * 32768) / 32768`). Results differ from
the double scalar reference by at most **1 LSB per call**, including rounding
boundaries. Unmasked integer continuous `CP_INVERT_MIX` may use Q16 weights
when `inversion_sum` is integral and in [0,65535], also with at most 1 LSB error.
Masked integer continuous MIX and INVERT_MIX (`inversion_sum == maximum`)
may similarly quantize their combined effective weight to Q16 within 1 LSB.
Exact zero-weight and
full-weight endpoints are preserved. Integer YUV Overlay Multiply also permits 1 LSB for interior opacity, as
described below. These allowances apply only to the operations documented here. Integer code weights remain exact; F32 has a separate allowance below. Repeated
operations can accumulate error. Select `CP_TARGET_C` or call `cp_process_plane`
for reference arithmetic; increasing working bit depth before processing reduces
the normalized size of a code-value error.

Integer storage is U8/8 or U16/9–16 bits; float storage is F32/32. Float chroma is
centered at zero; integer chroma is centered at `2^(bits-1)`. Mask channels always
use nonnegative opacity values, even when attached to chroma. Consult headers for
supported operations, finite-value requirements, rounding and aliasing rules.

Integer YUV Overlay Multiply may use binary32 SIMD for interior opacity
(0 < opacity < 1). For canonical input codes, the final output differs
from the scalar reference by at most 1 LSB. Zero and full opacity retain their
existing exact behavior. F32 has the separate allowance below.

Integer continuous PRODUCT may quantize its blend weight to Q16, with at most
1 LSB output difference. The product is still floored before blending.

Integer continuous ADD and SUBTRACT may quantize the effective weight to Q16.
The final clipped output differs by at most 1 LSB; exact zero/full-weight endpoints remain exact.

Integer continuous DIFFERENCE may use Q16 weights when bias is an integer in
[0, 65535]. Its final clipped result differs by at most 1 LSB. Fractional or
out-of-range bias retains reference arithmetic.

Integer continuous GUIDED_MULTIPLY may use float SIMD for 0 < opacity < 1
and neutral in [0, maximum]. Final output differs by at most 1 LSB. Full
opacity and out-of-range neutral retain the
reference calculation; zero mask preserves the original code exactly.

Integer input samples, guides and masks must fit the declared bit depth, as
required by `types.h` (10-bit: 0..1023, not the narrower video limited range).
SIMD kernels do not scan for violations or guarantee scalar-equivalent results
for them. Geometry and descriptor validation, and bounds-safe access, remain
unchanged. F32 color excursions remain supported under their existing contract.

F32 SIMD masked MIX and PRODUCT on contiguous planes, and YUV Overlay Multiply
on contiguous planes with no mask or one shared mask, may use binary32
arithmetic for every opacity in (0,1], including negative colors and HDR.
The allowed error against the scalar result is
`16 * FLT_EPSILON * max(1, S)`, with double-precision weight
`w = opacity * mask` (or `opacity` without a mask):

- MIX: `S = abs(a)*(1-w) + abs(b)*w`.
- PRODUCT: `S = abs(a)*((1-w) + abs(b)*w)`.
- YUV Multiply uses the PRODUCT scale for each base channel with source Y as `b`.

For normalized nonnegative inputs this is at most 1.91e-6, or 0.125 of a
16-bit code. This is an error allowance, not a claim that all cases reach it.
Zero mask copies input bits, and full-weight endpoints retain reference semantics.
Both F32 weight rules share this allowance; integer CODE behavior is unchanged.
Nonfinite inputs or results, results within 64 float epsilons of overflow, and
near-full blends whose base exceeds `2^20 * max(1, abs(candidate))` retain the
reference calculation for the vector block. Other layouts retain existing kernels.
Use `cp_process_plane`, `cp_process_yuv`, or `CP_TARGET_C` for reference arithmetic.
