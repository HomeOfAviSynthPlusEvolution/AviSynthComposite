#!/usr/bin/env python3
"""Extract scalar Layer references from a local Git ref into ignored tmp/.

Only the extraction script is versioned; generated files retain upstream notices.
Markers intentionally fail if upstream rearranges its implementation.
"""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repository", type=Path)
parser.add_argument("--ref", default="upstream/master")
parser.add_argument("--output", type=Path, default=Path("tmp/benchmark-upstream"))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output.resolve()
if not out.is_relative_to(root / "tmp"):
    parser.error("output must be within this project's tmp/ directory")

def git(*cmd):
    return subprocess.check_output(["git", "-C", str(args.repository), *cmd]).decode("utf-8")

commit = git("rev-parse", "--verify", args.ref + "^{commit}").strip()
def source(path):
    return git("show", commit + ":avs_core/filters/" + path)

blend = source("overlay/blend_common.h")
layer = source("layer.hpp")
header = "// Generated from upstream commit " + commit + "\n"
header += blend[:blend.index("#ifndef")]
header += layer[:layer.index("#ifndef")] if "#ifndef" in layer else ""
header += """
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <type_traits>
using BYTE = unsigned char;
using std::min; using std::max; using std::clamp;
#define AVS_FORCEINLINE inline
#define AVS_UNUSED(x) (void)(x)
#define LAYER_ROWPREP_FN prepare_effective_mask_for_row
constexpr int cyb=3736, cyg=19234, cyr=9798;
constexpr float cyb_f=.114f, cyg_f=.587f, cyr_f=.299f;
"""
start = blend.index("struct MagicDiv")
header += blend[start:blend.index("// ============================================================", start)]
header += blend[blend.index("enum MaskMode"):blend.index("// For MASK420/MASK420_MPEG2/MASK420_TOPLEFT")]
for name in ("layer_planarrgb_add_c", "layer_planarrgb_mul_c",
             "layer_planarrgb_add_f_c", "layer_planarrgb_mul_f_c"):
    start = layer.index("static void " + name + "(")
    start = layer.rindex("template<", 0, start)
    end = layer.index("\n}", start) + 2
    header += layer[start:end] + "\n"
out.mkdir(parents=True, exist_ok=True)
(out / "layer_kernels.h").write_text(header, encoding="utf-8")
(out / "upstream_commit.h").write_text('#define CP_UPSTREAM_COMMIT "' + commit + '"\n', encoding="utf-8")
average = source("intel/merge_avx2.cpp")
generated = average[:average.index("#if")] + "\n#include <immintrin.h>\n#include <cstdint>\n#include <cstddef>\nusing BYTE=unsigned char;\n"
start = average.index("template<typename pixel_t>\nvoid average_plane_avx2(")
end = average.index("\n}", start) + 2
generated += average[start:end] + "\n"
start = average.index("void average_plane_avx2_float(")
end = average.index("\n}", start) + 2
generated += average[start:end] + "\n"
generated += """
extern "C" void cp_bench_average_avx2(int bytes, BYTE* a, const BYTE* b, int pitch, int rowsize, int height) {
  if (bytes==1) average_plane_avx2<uint8_t>(a,b,pitch,pitch,rowsize,height);
  else if (bytes==2) average_plane_avx2<uint16_t>(a,b,pitch,pitch,rowsize,height);
  else average_plane_avx2_float(a,b,pitch,pitch,rowsize,height);
}
"""
(out / "average_avx2.cpp").write_text(generated, encoding="utf-8")
print(commit)

# Shared Merge/Overlay flat-weight AVX2 core. Keep source and notices in tmp.
weighted = source("overlay/intel/blend_common_avx2.cpp")
generated = weighted[:weighted.index("#include")] + "\n#include <immintrin.h>\n#include <cstdint>\nusing BYTE=unsigned char;\n"
start = weighted.index("static void weighted_merge_uint8_avx2_impl(")
end = weighted.index("\n}", weighted.index("void weighted_merge_float_avx2(")) + 2
generated += weighted[start:end]
generated += """
extern "C" void cp_bench_weighted_avx2(int bits, BYTE* a, const BYTE* b, int pitch, int width, int height, double weight) {
  if (bits == 32) weighted_merge_float_avx2(a,b,pitch,pitch,width,height,float(weight));
  else {
    const int w = int(weight * 32768);
    weighted_merge_avx2(a,b,pitch,pitch,width,height,w,32768-w,bits);
  }
}
"""
(out / "weighted_avx2.cpp").write_text(generated, encoding="utf-8")

# Full MASK444 reference, including upstream row preparation and scratch storage.
# Concatenate extracted files in dependency order; only include directives change.
def without_includes(text):
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("#include", "#pragma once"))) + "\n"
masked = '#include "layer_kernels.h"\n#include <immintrin.h>\n'
for name in ("masked_rowprep_avx2.h", "masked_rowprep_avx2_impl.h",
             "masked_rowprep_avx2.cpp", "masked_merge_avx2_impl.hpp"):
    masked += without_includes(source("overlay/intel/" + name))
masked += """
extern "C" void cp_bench_masked_avx2(int bits, BYTE* a, const BYTE* b, const BYTE* m,
                                    int pitch, int width, int height, double opacity) {
  if (bits == 32)
    masked_merge_float_avx2_impl<MASK444>(a,b,m,pitch,pitch,pitch,width,height,float(opacity));
  else
    masked_merge_avx2_impl<MASK444>(a,b,m,pitch,pitch,pitch,width,height,
                                  int(std::floor(opacity*((1u<<bits)-1)+.5)),bits);
}
"""
masked += """
extern "C" void cp_bench_sample420_avx2(int bits, BYTE* out, const BYTE* source, int pitch, int width, int height) {
  const MagicDiv magic = get_magic_div(bits == 32 ? 16 : bits);
  for (int y=0; y<height; ++y) {
    if (bits == 32)
      fill_mask420_float_avx2<true>(reinterpret_cast<float*>(out+y*pitch),
        reinterpret_cast<const float*>(source+2*y*pitch),pitch/4,width,1);
    else if (bits == 8)
      fill_mask420_avx2<uint8_t,true>(out+y*pitch,source+2*y*pitch,pitch,width,0,0,magic);
    else
      fill_mask420_avx2<uint16_t,true>(reinterpret_cast<uint16_t*>(out+y*pitch),
        reinterpret_cast<const uint16_t*>(source+2*y*pitch),pitch/2,width,0,0,magic);
  }
}
"""
(out / "masked_avx2.cpp").write_text(masked, encoding="utf-8")

# YUV Add/Subtract integer reference, with a minimal frame-view stand-in only.
add = source("overlay/OF_add.cpp")
yuv = add[:add.index("#include")] + '#include "layer_kernels.h"\n'
yuv += """
enum { PLANAR_Y, PLANAR_U, PLANAR_V };
struct ImageOverlayInternal {
  BYTE* planes[3]; int pitch, width, height;
  BYTE* GetPtr(int p) { return planes[p]; }
  int w() const { return width; } int h() const { return height; }
};
struct OL_AddImage {
  int bits_per_pixel; float opacity_f;
  template<class T, bool masked, bool add, bool full>
  void BlendImageMask(ImageOverlayInternal*,ImageOverlayInternal*,ImageOverlayInternal*);
};
"""
start = add.index("template<typename pixel_t, bool maskMode, bool of_add, bool fullOpacity>")
end = add.index("\n}", start) + 2
yuv += add[start:end] + "\n"
yuv += """
template<class T, bool Add, bool Mask>
void cp_bench_yuv_call(OL_AddImage& op, ImageOverlayInternal& a, ImageOverlayInternal& b, ImageOverlayInternal& m) {
  if (op.opacity_f == 1) op.BlendImageMask<T,Mask,Add,true>(&a,&b,&m);
  else op.BlendImageMask<T,Mask,Add,false>(&a,&b,&m);
}
template<class T>
void cp_bench_yuv_type(OL_AddImage& op, ImageOverlayInternal& a, ImageOverlayInternal& b, ImageOverlayInternal& m,
                       bool add, bool masked) {
  if (add) {
    if (masked) cp_bench_yuv_call<T,true,true>(op,a,b,m);
    else cp_bench_yuv_call<T,true,false>(op,a,b,m);
  } else {
    if (masked) cp_bench_yuv_call<T,false,true>(op,a,b,m);
    else cp_bench_yuv_call<T,false,false>(op,a,b,m);
  }
}
extern "C" void cp_bench_yuv_add(int bits, bool add, bool masked, BYTE** a, BYTE** b, BYTE* m,
                                int pitch, int width, int height, double opacity) {
  ImageOverlayInternal av{{a[0],a[1],a[2]},pitch,width,height};
  ImageOverlayInternal bv{{b[0],b[1],b[2]},pitch,width,height};
  ImageOverlayInternal mv{{m,m,m},pitch,width,height};
  OL_AddImage op{bits,float(opacity)};
  if (bits==8) cp_bench_yuv_type<uint8_t>(op,av,bv,mv,add,masked);
  else cp_bench_yuv_type<uint16_t>(op,av,bv,mv,add,masked);
}
"""
(out / "yuv_add.cpp").write_text(yuv, encoding="utf-8")

multiply = without_includes(source("overlay/intel/OF_multiply_avx2.cpp"))
multiply = '#include "layer_kernels.h"\n#include <immintrin.h>\n' + multiply
multiply += """
template<class T>
void cp_bench_overlay_mul_type(int bits, double opacity, bool masked, int width, int height,
                               const BYTE* source, BYTE** out, const BYTE* mask, int pitch) {
  const auto* sy = reinterpret_cast<const T*>(source);
  auto* y = reinterpret_cast<T*>(out[0]); auto* u = reinterpret_cast<T*>(out[1]); auto* v = reinterpret_cast<T*>(out[2]);
  const auto* m = reinterpret_cast<const T*>(mask);
  const int stride = pitch/sizeof(T);
  if (opacity == 1) {
    if (masked) of_multiply_avx2<T,true,true>(bits,float(opacity),256,width,height,sy,stride,y,u,v,stride,m,m,m,stride);
    else of_multiply_avx2<T,true,false>(bits,float(opacity),256,width,height,sy,stride,y,u,v,stride,m,m,m,stride);
  } else {
    if (masked) of_multiply_avx2<T,false,true>(bits,float(opacity),0,width,height,sy,stride,y,u,v,stride,m,m,m,stride);
    else of_multiply_avx2<T,false,false>(bits,float(opacity),0,width,height,sy,stride,y,u,v,stride,m,m,m,stride);
  }
}
extern "C" void cp_bench_overlay_mul(int bits, double opacity, bool masked, int width, int height,
                                     const BYTE* source, BYTE** out, const BYTE* mask, int pitch) {
  if (bits==8) cp_bench_overlay_mul_type<uint8_t>(bits,opacity,masked,width,height,source,out,mask,pitch);
  else cp_bench_overlay_mul_type<uint16_t>(bits,opacity,masked,width,height,source,out,mask,pitch);
}
"""
(out / "overlay_mul_avx2.cpp").write_text(multiply, encoding="utf-8")
