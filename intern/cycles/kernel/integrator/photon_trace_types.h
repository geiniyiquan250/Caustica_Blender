/* SPDX-FileCopyrightText: 2026 CyclesPlus
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Photon caustics: PODs shared between the host batch scheduler
 * (integrator/photon_map.cpp) and the device photon tracing kernel
 * (kernel/integrator/photon_trace.h). Host-safe: no kernel includes. */

#pragma once

#include "util/math.h"
#include "util/types.h"

CCL_NAMESPACE_BEGIN

enum PhotonTraceMaterialKind {
  PHOTON_MAT_RECEIVER = 0,
  PHOTON_MAT_GLASS = 1,
  PHOTON_MAT_METAL = 2,
  PHOTON_MAT_SKIP = 3,
};

/* Per-shader photon material, extracted host-side from the shader graphs
 * (indexed by the intersection's shader index). 16-byte aligned PODs so the
 * layout matches between host and device. */
struct PhotonTraceMaterial {
  int kind;
  float ior;
  float rough;
  float transmission; /* receivers: thin-wall pass-through probability */

  float coat;
  float coat_rough;
  float coat_ior;
  /* != 0: the classifier could not capture this material exactly - the
   * photon walk runs the REAL surface shader at its hits (accurate mode). */
  float accurate;

  /* xyz: per-event tint. w: alpha-transparent pass probability (photons
   * pass straight through untouched, not a caustic event). */
  float4 color;
  /* xyz: Beer-Lambert absorption coefficient of the interior medium
   * (Volume Absorption node: density * (1 - color)); zero = clear. The
   * walk attenuates power over the distance travelled INSIDE the medium
   * (volume-tinted pool water, colored solid glass).
   * w: pass_mode - 1 = photon-owned (receiver pass-through arms deposits),
   * 0 = demoted (path tracing owns the caustics, no spec++), 2 = demoted
   * glass (Fresnel-share pass-through, no spec++). */
  float4 volume_sigma;
};

/* Emission source. Photon indices are partitioned into per-light ranges on
 * the host (budget from the pilot measurement); each light normalizes its
 * flux by its own range size, exactly like the host tracer. */
struct PhotonTraceLight {
  int type; /* 0 sun, 1 point, 2 spot, 3 area */
  int shape;
  float p0; /* sun: half angle | point/spot: radius */
  float spot_cos;

  int index_start; /* [start, end) photon indices of this light */
  int index_end;
  int n_total; /* flux normalization for this light's range */
  /* Object index of the emitter, for Blender's light linking (OBJECT_NONE
   * when unknown). The deposit is tested against the receiving object with
   * exactly the rule path tracing uses at its shading points. */
  int emitter_object;

  float4 color;
  float4 pos;
  float4 axis;  /* xyz = emission direction, w = sx (area) */
  /* x = sy (area)
   * y = light group index, or -1. Every deposit carries the group of the
   *     light that emitted it, so the gather can split the caustic the way
   *     Cycles splits Combined into Combined_<group> passes. */
  float4 extra;
};

/* Caster target (bounding sphere), cumulative weight for picking. */
struct PhotonTraceTarget {
  float4 c_r; /* xyz = center, w = radius */
  float wcum; /* cumulative pick weight in (0, 1] */
  /* Sun/world photons launch this far in front of the target so exterior
   * occluders block them exactly like they block the path tracer's rays.
   * Host-computed from the scene bounding sphere (|Tc - scene center| +
   * scene radius + margin); 0 means "unknown", the walks fall back to the
   * old ~50 m heuristic. */
  float start_dist;
  float pad1, pad2;
};

/* Batch seed, shared so host and kernel derive identical RNG streams. */
ccl_device_inline uint64_t photon_trace_batch_seed(const uint64_t k)
{
  return 0x9E3779B97F4A7C15ULL + k * 0x2545F4914F6CDD1DULL;
}

CCL_NAMESPACE_END
