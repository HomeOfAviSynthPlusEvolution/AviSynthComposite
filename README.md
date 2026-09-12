# AviSynth — Composite

Independent image blending and compositing kernels for Merge, Overlay,
Layer and related channel/mask operations. Public headers are C-compatible;
implementation is C++17 and builds as the `AviSynth::Composite` static target.
There is no AviSynth SDK dependency or host integration in this repository.

This is an initial development implementation. The public interface may change;
build matching headers and sources together rather than mixing binary versions.

## Build

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

CMake 3.24+, a C++17 compiler and Ninja are sufficient. Tests have no downloaded
dependencies. Embedded builds default tests off; use `CP_BUILD_TESTS` to override.
With `CP_BUILD_TESTS=ON`, the library and test references disable fused
multiply-add for reproducible exact comparisons. Production builds use
`CP_BUILD_TESTS=OFF` and permit fused multiply-add; use a separate Release build
directory for production.
The optional `CP_SANITIZERS` switch instruments library and consumers with ASan
and UBSan on supported non-MSVC Clang/GCC toolchains.

Highway 1.4.0 is vendored for offline builds. An existing compatible `hwy` target
is reused when embedded beside ConvertAudio/ConvertVideo. `CP_SCALAR_ONLY=ON`
removes both SIMD targets and the Highway dependency entirely.

## Backend selection

Direct operation functions always use the scalar implementation. Obtain an
immutable `cp_kernels` table with `cp_get_kernels(cp_choose_target(allowed_bits))`
to select a backend per instance. Target 0 means C, -1 means native, and positive
single bits use Highway's target identifiers. No process-wide target override is
used. Invalid or unsupported explicit targets return null.

The SIMD table covers all nine plane modes, coupled YUV operations, compatibility
blend, copy/fill, guide resampling, affine/clamp, RGB luma and color key. Stepped
channels and partial vectors use bounded staging where a direct vector access
would touch neighboring samples. No padding is required. Targets without FP64
retain scalar fallback for double-precision arithmetic; other kernels still use
SIMD. Rectangle intersection is scalar constant-time geometry.

Every mode remains available in scalar-only builds. Cache the table after applying
the host CPU policy; target discovery is unnecessary per frame. SIMD coverage does
not imply a speedup for every mode or layout; benchmarks are not yet established.

## Interface

Include `composite/composite.h`, or individual operation headers. All images are
explicit channel views: `stride` is signed bytes between rows and `step` is
positive bytes between samples. This handles planar channels, BGRA/RGBA channels,
and YUY2 channels without duplicate arithmetic kernels or a full-frame repack.
Each pointer denotes logical row zero, including partial row-band execution.

The library provides building blocks with explicit mathematical semantics:

- `cp_process_plane`: weighted/masked mix, addition/subtraction, product,
  inverted-target mix, luma-guided multiply, threshold selection and biased difference.
- `cp_process_yuv`: coupled full-resolution YUV artistic operations with AviSynth
  overshoot desaturation; Soft/Hard Light are AviSynth's additive definitions.
- `cp_resample_mask`: bounded chroma mask/guide preparation for 444, 422, 420,
  411, centered/MPEG2/top-left placement and signed sampling phase.
- Channel copy/fill, affine inversion, explicit clamp, RGB luma, color key and
  rectangle intersection.
- `cp_blend_compat`: separate historical Minus integer blend arithmetic.

Alpha is an ordinary independently addressable channel. Weight masks are separate
from blend targets, so Layer Subtract can use original alpha as weight and inverted
alpha as the target. No implicit Porter-Duff alpha equation is substituted.
Unused channels are left untouched by individual plane calls. For multi-channel
composition, preserve original guides and weight masks until all dependent channels
have finished. No function allocates memory or modifies global dispatch state.

For unmasked integer `CP_MIX` with `CP_WEIGHT_CONTINUOUS`, SIMD backends may
round opacity to Q15 (`round(opacity * 32768) / 32768`). Results differ from
the double scalar reference by at most **1 LSB per call**, including rounding
boundaries. Unmasked integer continuous `CP_INVERT_MIX` may use Q16 weights
when `inversion_sum` is integral and in [0,65535], also with at most 1 LSB error.
Masked integer continuous MIX and INVERT_MIX (`inversion_sum == maximum`)
may similarly quantize their combined effective weight to Q16 within 1 LSB.
Exact zero-weight and
full-weight endpoints are preserved. Integer YUV Overlay Multiply also permits 1 LSB for interior opacity, as
described below. These allowances do not extend to other operations (except additional integer operations documented below), code weights,
or F32 except for the separately documented allowances below. Repeated
operations can accumulate error. Select `CP_TARGET_C` or call `cp_process_plane`
for reference arithmetic; increasing working bit depth before processing reduces
the normalized size of a code-value error.

Integer storage is U8/8 or U16/9–16 bits; float storage is F32/32. Float chroma is
centered at zero; integer chroma is centered at `2^(bits-1)`. Mask channels always
use nonnegative opacity values, even when attached to chroma. Consult headers for
supported operations, finite-value requirements, rounding and aliasing rules.

Normal kernels follow reviewed arithmetic, with explicit code-scale weight
quantization where needed. They do not promise identical output to every historic
SIMD path. Compatibility is a separate function. Upstream comparisons and local
research stay outside version control in `tmp/` and `docs/` respectively.

## Kernel benchmarks

Configure with `-DCP_BUILD_BENCHMARKS=ON` in a Release build. Run
`composite_bench --width 1920 --height 1080 --trials 7 --all-targets` and redirect
stdout to CSV. Without `--all-targets`, only C and the native target are measured.
Target numbers are the public Highway bits, with zero denoting C. Workloads
cover plane mix/product/guided multiply, compat, RGB Layer Add/Mul, YUV artistic
modes, centered 420 sampling and all pixel utility interfaces, at 8/16/32 bits
with contiguous and stepped channel views. Use `--workload NAME` to repeat a
single workload (names appear in the first CSV column). `--opacity W` controls
blend opacity and `--step 1` restricts runs to contiguous planes. 10-bit storage
is included alongside 8/16/32 bits in the current matrix.

Optional upstream scalar comparisons are prepared from a local Git repository:

```sh
python benchmarks/prepare_upstream.py ../AviSynthMinus --ref upstream/master
cmake -S . -B build/benchmark -DCP_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DCP_UPSTREAM_BENCH_DIR=/absolute/path/to/AviSynthComposite/tmp/benchmark-upstream
cmake --build build/benchmark --config Release
```

The extractor records the resolved commit and retains upstream notices in ignored
temporary headers. It does not fetch; update the local ref explicitly when desired.
These comparisons measure actual Layer Add/Mul scalar templates, with compiler
auto-vectorization enabled, not upstream hand-written SIMD or complete filters.
For `--workload mix --opacity 0.5 --step 1`, x86 builds additionally compare the
actual upstream AVX2 average core when the AVX2 target is supported. Its source
is extracted to the temporary directory and compiled separately with AVX2 enabled;
the benchmark driver retains its baseline instruction set. This reference applies
to unmasked averaging. Non-half, non-endpoint unmasked MIX additionally compares
upstream's shared Merge/Overlay weighted AVX2 core. Integer comparisons require
an exactly representable k/32768 opacity and bit-exact outputs; other integer
weights are skipped rather than quantized. Float comparisons permit 2e-7 absolute
error on normalized finite inputs, not a general float error guarantee. The
weighted float core requires FMA, included in Highway's AVX2 target requirements.
For masked `code_mix` (or float `mix`), the MASK444 upstream AVX2 comparison
includes upstream row preparation and scratch allocation inside the timed call.
Integer code weights must match exactly; continuous integer mask weights have a
different contract and are not compared to this reference. Float comparisons use
the normalized-input tolerance above. `sample420` additionally compares the full-opacity upstream CENTER box row core.
Integer outputs must match exactly. Upstream float SIMD groups its additions
differently from its scalar tail; comparison allows 2e-7 on normalized inputs,
while our kernels retain the scalar tap order bit-for-bit. Other subsampled mask
modes are not yet compared to upstream. Use `--bits 8|10|16|32` to focus the format
without changing the workload or validation.
RGB comparisons process three channels; optional source alpha weights all three
without modifying alpha. Stepped single-channel tests do not grant access to other
channels and are not a fused packed-RGBA benchmark.
Use `--packed` to place all four channels in a single interleaved RGBA allocation
with a four-sample pixel step. This differs from `--step 4`, which uses four
separate sparse planes and has a larger working set. The CSV `layout` column
identifies `packed` or `separate`; compare like layouts. Packed runs validate the
entire allocation, including untouched channels, and omit contiguous upstream
references. For example:

```sh
composite_bench --workload key --bits 32 --packed --trials 11 > tmp/packed-key.csv
```

`code_guided` measures the code-weight guided multiply path with an integral
chroma neutral (or zero for float); `guided` retains continuous weights and its
existing half-maximum neutral. These are distinct arithmetic contracts.

Inputs are deterministic. Before timing, every backend is checked against the C
reference, including untouched samples; local backends must be bit-exact except
for the documented 1 LSB allowance for integer continuous MIX and
INVERT_MIX, and integer Overlay Multiply. Upstream
integer Layer results must be exact and float results within 2e-7 (the existing
Layer fixture tolerance). Any mismatch fails the run. Each timed call follows a
reset from the same base, with two warmups and the requested measured trials;
reset, allocation, validation and checksum are excluded from timing. CSV reports
min/median/max milliseconds and processed samples per second. Resetting warms
memory, so these are kernel microbenchmarks, not cold-cache streaming throughput.
Record CPU, compiler, build flags and Git revision alongside results. Run benchmarks
alone rather than concurrently with builds/tests, and repeat before judging small
differences.

### Batched average comparison

`composite_average_bench` focuses on contiguous 16-bit, 50% unmasked MIX of a
1920x1080 single plane. It compares the native target, AVX2 when available, and
the extracted upstream AVX2 core when configured above.

```sh
composite_average_bench --slots 1 --batch 64 --rounds 30 > tmp/average-hot.csv
composite_average_bench --slots 8 --batch 64 --rounds 30 > tmp/average-ring.csv
python benchmarks/summarize_average.py tmp/average-hot.csv
python benchmarks/summarize_average.py tmp/average-ring.csv
```

After a one-second initial warmup, each block restores deterministic inputs,
performs two warmup passes over the frame ring, then times 64 calls together.
Only the calls are timed. All output is independently validated after each block.
Backends rotate through execution positions, with a seeded shuffle between groups;
30 rounds provide paired comparisons. Raw CSV retains every block. The summary
reports median, p10/p90, relative median absolute deviation, and paired ratios to
upstream. Bootstrap intervals are exploratory: successive rounds can share clock
or thermal drift and are not necessarily statistically independent.

One slot has a 7.91 MiB source/output working set; eight have 63.28 MiB at the
default dimensions. Base snapshots require additional memory outside timing.
These are warmed repeated-frame and larger rotating-frame workloads, respectively,
not guaranteed cold-cache tests. Averaging updates the destination in place and
converges toward the source during each block. This is appropriate for this
data-independent integer kernel, not a general recipe for value-dependent kernels.
Results include our public dispatch/validation but upstream's direct core wrapper;
neither includes complete filter/frame-management overhead. Pin to the same CPU,
run without concurrent builds/tests, and repeat in separate processes before
interpreting small differences. Do not compare these timings directly with the
single-call benchmark, whose resets and checks produce different cache conditions.

### Additional compositing comparisons

`overlay_mul` measures fused full-resolution YUV Multiply through
`CP_YUV_MULTIPLY`, equivalent to continuous `CP_GUIDED_MULTIPLY` on each plane
with source Y as the shared guide. It validates against the independent C path.
The optional upstream integer AVX2 core uses binary32 internally; its benchmark
comparison permits one output code difference and reports the observed maximum.
This tolerance applies to this bounded integer operation, not other kernels.

Integer `yuv_add` and `yuv_subtract` also compare actual upstream scalar templates,
with a minimal frame-view stand-in and no AviSynth SDK. Their outputs must match
exactly. The reference is enabled only when opacity is exactly representable as
binary32, avoiding a hidden parameter conversion. These are scalar templates
with compiler auto-vectorization enabled, not upstream handwritten SIMD.

## License and attribution

GPL version 2 or later, retaining the inherited AviSynth linking exception; see
[LICENSE](LICENSE). Algorithms and behavior were studied in AviSynth/AviSynth+ and
AviSynthMinus. Original AviSynth copyright includes Ben Rudiak-Gould and other
contributors; Overlay was originally written by Klaus Post (2003–2004). This
implementation also draws on the subsequent AviSynth+ contributors' fixes and
semantic clarifications. No upstream golden source is distributed in this tree.

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

F32 YUV Overlay Multiply with continuous planes with no mask or a shared mask and
0 < opacity <= .75 may use binary32 arithmetic when guide Y is in [0,1]
and the base colors are finite. Its error relative to scalar is bounded by
`8 * FLT_EPSILON * abs(reference) + 2 * FLT_TRUE_MIN` (about 0.0000954%
relative error, plus subnormal rounding). Zero mask preserves input bits.
Out-of-range guide values, nonfinite colors and other configurations retain
reference arithmetic. This does not restrict valid F32 color excursions.
Use `cp_process_yuv` / `CP_TARGET_C` for reference arithmetic.

F32 masked MIX on continuous planes may use binary32 arithmetic for 0 < opacity <= .75 and inputs in [-1,1], with absolute error <= 8*FLT_EPSILON. Zero mask copies exactly; other inputs retain reference arithmetic. This applies to both weight rules; integer CODE behavior is unchanged.
