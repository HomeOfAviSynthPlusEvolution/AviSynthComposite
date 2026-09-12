# Numeric reference fixtures

`overlay.csv` contains **data only**, produced from AviSynth+ upstream/master
`5c82777b374bdef16e13007a11e77d735ac1e4eb`, fetched on 2026-09-07.
The upstream source and extraction harness are deliberately not distributed.

Columns: bits, mode, masked, float opacity, base Y/U/V, overlay Y/U/V, mask Y/U/V,
expected Y/U/V. Mode 0..8 denotes Add, Subtract, SoftLight, HardLight, Difference,
Exclusion, Multiply, Lighten, Darken. Mask values are ignored when masked=0.
Opacity is serialized with enough digits to round-trip the original binary32.

Generation used scalar (no `INTEL_INTRINSICS`) upstream OF implementations and
upstream MagicDiv helpers, with a minimal one-row image adapter. Integer depths
8/10/12/14/16; opacity 0, .003, .5, .63, 1; with/without masks. `std::mt19937`
seed 4242+depth filled 4096 samples per Y/U/V channel in base/overlay/mask order,
with uniform modulo (max+1) code values. Each configuration records its first
eight pixels, producing 3600 rows. The full local differential experiment checked
4096 pixels per configuration (450 configurations, three channels).

Acceptance: exact for modes 0..5; at most one code for modes 6..8. Multiply has
float-intermediate versus double-intermediate rounding differences; masked,
full-opacity upstream Lighten/Darken truncates, while the new kernel rounds.
Upstream selection is inclusive only for the unmasked full-opacity route and
strict otherwise. The fixture test chooses that policy explicitly. This is not
a blanket decision to preserve that discontinuity in a future host filter.

These values supplement independent formula, endpoint and memory tests. They are
not a promise of historical bit identity for every input or compiler.

`layer.csv`: bits, mode (Add/Mul/Lighten/Darken), use_chroma, has/blend_alpha,
opacity, base G/B/R/A, overlay G/B/R/A, expected G/B/R/A. Same reference commit;
mt19937 seed 912+depth, 2009 pixels, first eight retained. The 300 profiles cover
8/10/12/14/16 bits, two alpha choices, five opacities, use_chroma both ways for
Add/Mul and true for Lighten/Darken (threshold=3 sample codes). All 2400 fixture
rows and the full local experiment match exactly. Original alpha supplies weight;
alpha target participates in Add/Mul/selection where enabled.

`subsample.csv`: bits, mode (Mul/mulovr/Lighten/Darken), plane (Y/U/V), has_alpha,
opacity, base, overlay, filtered base Y, filtered overlay Y, filtered alpha,
expected output. Reference uses 32x6 images, eight upstream MaskMode values,
mt19937 seed 823+depth+MaskMode, both alpha choices and five opacities; 1600
profiles. Two samples per plane retained (9600 rows). Full-image local comparisons
also include upstream spatial row preparation. Mul and selection match exactly;
mulovr differs by at most one code: upstream uses floor(max/2) as neutral chroma,
whereas the new reference configuration uses the correct 2^(bits-1). The API can
explicitly select the former center when needed. Regular CI independently tests
spatial taps, phase, odd edges and row-band equivalence.

`float-overlay.csv` adds 160 rows with the same column layout as overlay.csv,
using round-trippable binary32 decimal sample values and bits=32. It covers
upstream Float Add/Subtract; other artistic modes in the integer fixture are
not supported by the recorded upstream Float Overlay implementation. Source
and base UV are centered at zero. A temporary differential run checked 20
configurations of 8192 pixels: largest absolute sample error was 5.36442e-7
in the YUV overflow/desaturation transition. These fixtures allow 8e-7.

`float-layer.csv` adds 480 rows using the same layout as layer.csv, bits=32,
and threshold=3.0f/255. The corresponding run checked 60 configurations of
8193 pixels for Add, Mul and thresholded Lighten/Darken with RGB/RGBA and
monochrome targets. Largest absolute sample error was 1.19209e-7; these
fixtures allow 2e-7. Float guides retain upstream binary32 ordering, so the
small output tolerance does not permit different selection branches.

Both Float sets use the same recorded upstream commit and seeds as their
integer counterparts (4242+32 and 912+32), drawing samples on a 65536-point
unit interval. Each saved row retains the first 8 pixels of a configuration.
The Float tolerances are operation-specific absolute errors in normalized
sample units, not a blanket tolerance for arbitrary excursions or algorithms.
