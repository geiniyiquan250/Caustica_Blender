/* SPDX-FileCopyrightText: 2026 CyclesPlus
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Photon caustics: kernel-side photon gather. Reads the photon grid from the
 * standard kernel data arrays, so it runs unchanged on CPU and GPU devices.
 * Grid parameters live in kernel_data.integrator (updated per progressive
 * batch). */

#pragma once

#include "kernel/integrator/photon_grid.h"

CCL_NAMESPACE_BEGIN

/* Epanechnikov-weighted flux sum from photons within the gather radius,
 * plus the raw photon count for the SPPM statistics update. `r2` is the
 * per-pixel SPPM radius; it starts at the grid cell size and only ever
 * shrinks. Normalization (disc area, generation average) happens in the
 * caller. */
/* Optionally the same sum split by light group. `r_group_sum` points at
 * `group_count` accumulators covering groups [group_base, group_base+count),
 * which the caller clears; every deposit carries its emitting light's group in
 * flux.w. The split rides along inside the ONE traversal that was going to
 * happen anyway - gathering per group separately would multiply the cost of
 * the most expensive per-pixel step in the renderer. Photons outside the
 * window (and ungrouped ones, group -1) still count towards the total, so the
 * caller can always compare the parts against the whole. */
ccl_device float3 photon_grid_gather(KernelGlobals kg,
                                     const float3 P,
                                     const float3 N,
                                     const float r2,
                                     ccl_private int *r_num_gathered,
                                     ccl_private float *r_centroid_offset,
                                     ccl_private float3 *r_group_sum = nullptr,
                                     const int group_base = 0,
                                     const int group_count = 0)
{
  *r_num_gathered = 0;
  *r_centroid_offset = 0.0f;

  if (kernel_data.integrator.photon_num == 0) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  const float inv_cell = kernel_data.integrator.photon_inv_cell;
  const uint table_size = (uint)kernel_data.integrator.photon_table_size;

  const int cx = (int)floorf(P.x * inv_cell);
  const int cy = (int)floorf(P.y * inv_cell);
  const int cz = (int)floorf(P.z * inv_cell);

  /* The per-pixel radius never exceeds the cell size, so the 3x3x3 cell
   * neighborhood always covers the gather disc. */
  const int range = 1;

  float3 sum = make_float3(0.0f, 0.0f, 0.0f);
  float3 weighted_offset = make_float3(0.0f, 0.0f, 0.0f);
  float weight_total = 0.0f;
  int num = 0;

  for (int dz = -range; dz <= range; dz++) {
    for (int dy = -range; dy <= range; dy++) {
      for (int dx = -range; dx <= range; dx++) {
        const uint b = photon_grid_hash(cx + dx, cy + dy, cz + dz, table_size);
        const int p0 = kernel_data_fetch(photon_cell_start, (int)b);
        const int p1 = kernel_data_fetch(photon_cell_start, (int)b + 1);
        for (int p = p0; p < p1; p++) {
          const float4 pp = kernel_data_fetch(photon_pos, p);
          const float3 d = make_float3(pp.x - P.x, pp.y - P.y, pp.z - P.z);
          const float d2 = dot(d, d);
          if (d2 >= r2) {
            continue;
          }
          /* Reject photons that are radially close but on another surface:
           * distance along the normal must stay well under the radius. */
          const float dn = dot(d, N);
          if (dn * dn > 0.1f * r2) {
            continue;
          }
          /* Reject photons deposited on an opposite-facing surface: the
           * plane test cannot separate the two millimeter-spaced walls of a
           * thin shell (inflatable toys!), which would make surfaces seen
           * through transmission glow with the sunlit outside's deposits. */
          if (dot(photon_unpack_normal(pp.w), N) <= 0.0f) {
            continue;
          }
          const float w = 1.0f - d2 / r2; /* Epanechnikov */
          const float4 flux = kernel_data_fetch(photon_flux, p);
          const float3 contribution = w * make_float3(flux.x, flux.y, flux.z);
          sum += contribution;
          if (r_group_sum != nullptr) {
            const int g = (int)flux.w - group_base;
            if (g >= 0 && g < group_count) {
              r_group_sum[g] += contribution;
            }
          }
          weighted_offset += w * d;
          weight_total += w;
          num++;
        }
      }
    }
  }

  *r_num_gathered = num;
  if (weight_total > 0.0f) {
    /* Weighted centroid offset relative to the radius: ~0 for a uniform
     * photon field, large when the disc straddles an illumination boundary
     * (waterline, shadow edge) and all photons sit on one side. */
    *r_centroid_offset = len(weighted_offset / weight_total) / sqrtf(r2);
  }
  return sum;
}

CCL_NAMESPACE_END
