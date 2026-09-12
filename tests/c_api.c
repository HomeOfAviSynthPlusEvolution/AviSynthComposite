/* SPDX-License-Identifier: GPL-2.0-or-later */
#include <composite/composite.h>
int main(void) {
  unsigned char a[3] = {0, 100, 255}, b[3] = {255, 200, 0}, d[3] = {0};
  cp_format f = {CP_U8, 8};
  cp_const_plane pa = {a, 3, 1}, pb = {b, 3, 1};
  cp_plane pd = {d, 3, 1};
  cp_rows rows = {3, 1, 0, 1};
  cp_plane_config c = {0};
  c.format = f;
  c.opacity = 0.5;
  if (cp_sample_bytes(f) != 1 || cp_process_plane(&c, pa, pb, 0, 0, 0, pd, rows) != CP_OK)
    return 1;
  if (d[0] != 128 || d[1] != 150 || d[2] != 128)
    return 2;
  if (cp_fill(f, pd, rows, 42) != CP_OK || d[2] != 42)
    return 3;
  const cp_kernels* kernels = cp_get_kernels(CP_TARGET_NATIVE);
  if (!kernels || kernels->blend_compat(f, pa, pb, 0, pd, rows, 128) != CP_OK || d[1] != 150)
    return 4;
  if (kernels->affine(f, pa, pd, rows, -1, 255) != CP_OK || d[0] != 255 || d[2] != 0)
    return 5;
  cp_sampling sampling = {3, 1, 1, 1, CP_CENTER, 0, 0};
  if (kernels->resample_mask(f, pa, pd, &sampling, rows) != CP_OK || d[1] != 100)
    return 6;
  return 0;
}
