# AviSynth — Composite

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

AviSynth — Composite is AviSynthMinus's independent image blending and compositing module. It builds without AviSynth and provides kernels for Merge, Overlay, Layer, and related channel and mask operations. It provides ordinary C implementations and cross-platform SIMD kernels using Google Highway.

The public interface uses C types and functions; the implementation uses C++17. It has no dependency on the AviSynth SDK, AvsCore, or AvsSimd.

## Why separate compositing?

Separating computational kernels from the frameserver allows their interfaces, numerical behavior, tests, and performance to be maintained independently. The module evolves alongside AviSynthMinus and can be included as a pinned Git submodule for static linking.

The host owns clips, script registration, frame allocation, properties, colorspace interpretation, and scheduling. The library operates on explicit channel views, masks, guides, and row ranges. It supplies building blocks for the host's filters; host integration is maintained separately.

## Supported operations

| Area | Capabilities |
|---|---|
| Plane blending | Mix, add, subtract, product, inverted-target mix, guided multiply, lighter/darker selection, and biased difference. |
| Coupled YUV | Full-resolution Add, Subtract, Soft Light, Hard Light, Difference, Exclusion, and Multiply; integer overshoot desaturation. F32 supports Add, Subtract, and Multiply. |
| Mask and guide sampling | 444, 422, 420, and 411; centered, MPEG2, and top-left placement with signed sampling phase. |
| Channel utilities | Copy/fill, affine transformation, clamp, RGB luma, color key, and rectangle intersection. |
| Compatibility | A separate historical Minus integer blend operation. |

Storage types are U8/8-bit, U16/9–16-bit, and F32/32-bit. Explicit channel views support planar and stepped packed channels. Alpha is an ordinary channel; weight masks are separate from blend targets. No implicit Porter–Duff equation is applied. Integer samples, masks, and guides must fit the declared bit depth. F32 colors support negative values and HDR; F32 masks must be finite and in [0,1]. See the [public headers](include/composite) for operation-specific restrictions.

Numerical behavior follows the reviewed C implementations and regression tests. Eligible integer SIMD paths permit at most **1 LSB per call**; selected F32 paths permit a magnitude-dependent rounding error. Repeated operations can accumulate error. Use ordinary C for reference arithmetic, or increase working bit depth to reduce the normalized size of integer rounding. Exact allowances, endpoints, and fallback conditions are in [Numerical behavior](NUMERICS.md). Matching a filter name does not promise identical output to every historical implementation.

## SIMD and CPU restrictions

`CP_TARGET_C` selects ordinary C. `CP_TARGET_NATIVE` selects an available native implementation, falling back to C when necessary. `cp_choose_target(allowed_bits)` selects from compiled targets supported by the CPU and permitted by the caller. `cp_get_kernels` returns an immutable function table; explicit unavailable or invalid targets return null. Targets use Highway bit values. Direct operation functions use ordinary C.

Selection is per consumer and does not change Highway's process-wide target restrictions. During AviSynth integration, AvsSimd stays in the host, where it interprets `SetMaxCPU` and supplies permitted targets. The host must map `SetMaxCPU("none")` to ordinary C and cache the selected table.

SIMD covers pixel operations, including stepped channels and bounded tails without requiring padding. Targets without FP64 retain ordinary C fallback for arithmetic requiring double precision. Rectangle intersection is scalar geometry. Wider SIMD targets do not guarantee higher speed.

## Building and integration

CMake 3.24 or later and a C++17 compiler are required. The C-compatible interface exposes no STL containers or Highway vector types. Use matching headers and libraries: the interface may evolve and does not promise binary interchangeability between releases.

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DCP_BUILD_TESTS=ON
cmake --build build/release --config Release --parallel
ctest --test-dir build/release -C Release --output-on-failure
```

Tests have no downloaded dependencies. Use `-DCP_BUILD_TESTS=OFF` for a library-only build; embedded builds default tests off. `-DCP_SCALAR_ONLY=ON` removes SIMD and the Highway dependency. Optional benchmarks use `-DCP_BUILD_BENCHMARKS=ON`. `-DCP_SANITIZERS=ON` enables ASan/UBSan on supported non-MSVC Clang/GCC toolchains.

The static library is `Composite`; its CMake alias is `AviSynth::Composite`. After adding the repository as a submodule, link the target directly:

```cmake
add_subdirectory(third_party/composite)
target_link_libraries(MyHost PRIVATE AviSynth::Composite)
```

Standalone builds use vendored Highway 1.4.0. Embedded builds reuse an existing compatible `hwy` target so Audio, Video, and the host can share one runtime. CMake propagates static link dependencies; consumers do not enumerate kernel sources. The supported integration is a joint CMake build, not an installed binary SDK package.

Public headers are under [include/composite](include/composite). Strides are signed byte counts; sample steps are positive byte counts. Follow each API's row-origin, overlap, and lifetime contracts. Pixel operations allocate no memory. Preserve original guides and masks until dependent channel operations finish. Independent calls into disjoint output regions can run concurrently.

## Testing and performance

Independent tests cover arithmetic, compatibility, sampling, channel utilities, C/SIMD comparisons, numerical boundaries, irregular sizes, signed strides, row bands, concurrency, and memory boundaries. A C consumer checks the public interface. The suite contains 15 CTest entries, with multiple cases and targets exercised internally.

CI covers Windows, Linux, and macOS with SIMD and ordinary C configurations, additional Windows Win32 builds, and Linux ASan/UBSan. Additional correctness and performance testing has covered Linux and FreeBSD on x86-64 and ARM64. Kernel validation does not replace host filter integration tests.

With tests enabled, the build disables implicit floating-point contraction for reproducible reference comparisons; explicit SIMD fused operations can still be used. Library-only builds permit contraction. Record this option when comparing results.

Optional benchmarks validate output against C before timing and report minimum, median, and maximum milliseconds. They cover 8-, 10-, 16-, and 32-bit inputs, contiguous and stepped channels, and packed layouts. Selected upstream kernels can be extracted from a local checkout. See [Kernel benchmarks](benchmarks/README.md) for commands, timing scope, and reference restrictions.

Compare equivalent inputs, layouts, build options, CPU restrictions, and timing scopes. Kernel-only timings and complete-filter timings are different measurements. Performance reports should include output comparisons as well as timings; small differences require proportionate verification.

## Development and contributions

The maintainer directs development, reviews changes, and is responsible for releases. Bug reports, suggestions, and contributions are welcome. Discuss numerical semantics, public interface changes, and substantial architectural changes before implementation.

This project uses AI-assisted implementation, tests, and review. Contributions should explain the problem, approach, validation, and how AI was involved. Reports should include the commit, OS, CPU, compiler, build options, input/output formats, and a minimal reproducer; performance reports should also describe dimensions, CPU targets, and the measurement method.

## Acknowledgments and license

This module builds on AviSynth, AviSynth+, AviSynthMinus, and their contributors, and uses Google Highway for SIMD. Thanks to the original authors and everyone contributing tests, reports, and improvements.

Thanks to [SB.SB](https://sb.sb) for sponsoring the LLM subscription used in this project's development.

The project uses GPL version 2 or later with the inherited AviSynth linking exception, retaining its original wording and scope. See [LICENSE](LICENSE). Source files retain their copyright notices; third-party components have their own licenses. Providing a new C interface does not expand the inherited exception.
