# Numerical test ranges

Finite sample values in the normal regression matrix represent image processing:

- Normal F32 luma/RGB: 0..1; signed chroma and intermediate colors are included.
- HDR and cancellation inputs: up to +/-100. This is a representative test envelope,
  not an API limit or a physical upper bound on scene-linear HDR.
- Small samples: approximately one 16-bit normalized code, plus exact signed zero.
- Opacity/mask: 0 and 1 exactly, common fractional settings, 1/65535 and 1-1/65535.
  Neighbors of integer rounding thresholds remain useful at ordinary magnitudes.
- Guided neutral: conventional normalized or declared-depth code values; a one-code
  excursion also checks the existing general path.
- Key/tolerance: normalized/HDR values or a few declared-depth code ranges.
- Clamp: ordinary bounds and one/two-code excursions, including fractional rounding.
- Integer pixels always fit their declared bit depth, including inversion tail samples.

Do not use FLT_MAX/DBL_MAX, subnormal sample grids, arbitrary IEEE exponent bit
patterns, or 1e-300 opacity as routine engineering requirements. Keep NaN/Inf
propagation, signed-zero/copy semantics and invalid-parameter rejection as separate
semantic checks. Memory-size/stride overflow and guard-page tests are unrelated to
brightness and remain unchanged.

These choices change test data only. They do not narrow public API validity or
remove production numerical fallbacks. Any future contract change requires a
separate decision, not an inference from the test envelope.
