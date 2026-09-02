/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/camera/camera.h"

#include "kernel/film/adaptive_sampling.h"
#include "kernel/film/light_passes.h"
#include "kernel/film/photon_passes.h"

#include "kernel/integrator/path_state.h"

#include "kernel/integrator/state_util.h"
#include "kernel/sample/pattern.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline Spectrum integrate_camera_sample(KernelGlobals kg,
                                                   const int sample,
                                                   const int x,
                                                   const int y,
                                                   const uint rng_pixel,
                                                   ccl_private Ray *ray,
                                                   ccl_private int &r_cache_miss)
{
  /* Filter sampling. */
  const float2 rand_filter = (sample == 0) ? make_float2(0.5f, 0.5f) :
                                             path_rng_2D(kg, rng_pixel, sample, PRNG_FILTER);

  /* Motion blur (time) and depth of field (lens) sampling. (time, lens_x, lens_y) */
  const bool use_motionblur = kernel_data.cam.shuttertime != -1.0f;
  const bool use_dof = kernel_data.cam.aperturesize > 0.0f;
  const bool use_custom_cam = kernel_data.cam.type == CAMERA_CUSTOM;
  const float3 rand_time_lens = (use_motionblur || use_dof || use_custom_cam) ?
                                    path_rng_3D(kg, rng_pixel, sample, PRNG_LENS_TIME) :
                                    zero_float3();

  /* We use x for time and y,z for lens because in practice with Sobol
   * sampling this seems to give better convergence when an object is
   * both motion blurred and out of focus, without significantly harming
   * convergence for focal blur alone.  This is a little surprising,
   * because one would expect using x,y for lens (the 2d part) would be
   * best, since x,y are the best stratified.  Since it's not entirely
   * clear why this is, this is probably worth revisiting at some point
   * to investigate further. */
  const float rand_time = rand_time_lens.x;
  const float2 rand_lens = make_float2(rand_time_lens.y, rand_time_lens.z);

  /* Generate camera ray. */
  return camera_sample(kg, x, y, rand_filter, rand_time, rand_lens, ray, r_cache_miss);
}

/* Return false to indicate that this pixel is finished.
 * Used by CPU implementation to not attempt to sample pixel for multiple samples once its known
 * that the pixel did converge. */
ccl_device bool integrator_init_from_camera(KernelGlobals kg,
                                            IntegratorState state,
                                            const ccl_global KernelWorkTile *ccl_restrict tile,
                                            ccl_global float *render_buffer,
                                            const int x_,
                                            const int y_,
                                            const int scheduled_sample)
{
  PROFILING_INIT(kg, PROFILING_RAY_SETUP);

  int x, y, sample;
  bool photon_writer = false;

  if (tile == nullptr) {
    /* Restart from miss. Reconstruct x, y, sample from state. */
    const uint pixel_index = INTEGRATOR_STATE(state, path, render_pixel_index);
    x = pixel_index % (int)kernel_data.cam.width;
    y = pixel_index / (int)kernel_data.cam.width;
    sample = INTEGRATOR_STATE(state, path, sample);
    /* Writer role survives the restart through the flag stashed below. */
    photon_writer = (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_PHOTON_HITPOINT_WRITER) != 0;
  }
  else {
    x = x_;
    y = y_;

    /* Initialize path state to give basic buffer access and allow early outputs. */
    path_state_init(state, tile, x, y);

    /* Check whether the pixel has converged and should not be sampled anymore. */
    if (!film_need_sample_pixel(kg, state, render_buffer)) {
      return false;
    }

    /* Count the sample and get an effective sample for this pixel. */
    sample = film_write_sample(kg, state, render_buffer, scheduled_sample, tile->sample_offset);

    /* Photon caustics (SPPM): a new sample invalidates the pixel's
     * measurement point; the first qualifying diffuse hit rewrites it.
     * Only the work's LAST scheduled sample touches the hitpoint passes:
     * on GPU all samples of a pixel are in flight concurrently, and racing
     * writers tear the P/normal/weight triplet, which the gather then
     * rejects through the normal test - measured as region-coherent
     * caustic fade at native pixel density (GPU-fade bug 2026-07-18).
     * A single deterministic writer preserves the sequential-CPU
     * semantics: the most recent sample's outcome wins.
     * Gated on the feature itself so that with photon caustics switched
     * off NOTHING here runs and no path flag bit is set: the product
     * promise is that unticking the box leaves plain Blender behind, and
     * a stray bit in PathRayFlag would become a silent corruption the day
     * Cycles claims bit 27 for itself. */
    if (kernel_data.integrator.use_photon_caustics) {
#ifdef __KERNEL_GPU__
      /* Key on the WORK's global last sample (kernel_data), not the tile's:
       * large images split one work into many tile launches, and a per-tile
       * last sample spawns several concurrent writers per pixel per work -
       * the hitpoint race returns in tile shape (equal-sized rectangular
       * patches in the caustic). */
      photon_writer = (scheduled_sample == kernel_data.integrator.photon_writer_sample);
#else
      /* CPU tiles are one sample each and run sequentially: every sample
       * writes, preserving the validated sequential semantics. */
      photon_writer = (scheduled_sample == tile->start_sample + tile->num_samples - 1);
#endif
      if (photon_writer) {
        film_clear_photon_hitpoint(kg, state, render_buffer);
      }
    }
  }

  /* Initialize random number seed for path. */
  const uint rng_pixel = path_rng_pixel_init(kg, sample, x, y);

  /* Generate camera ray. */
  Ray ray;
  int cache_miss = 0;
  Spectrum T = integrate_camera_sample(kg, sample, x, y, rng_pixel, &ray, cache_miss);
  if (cache_miss) {
    if (tile != nullptr) {
      integrator_path_init(state, DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA);
      INTEGRATOR_STATE_WRITE(state, path, sample) = sample;
      /* Stash the writer role for the restart (the restart branch above
       * reads it back before the flags are re-initialized). */
      INTEGRATOR_STATE_WRITE(state, path, flag) = photon_writer ?
                                                      PATH_RAY_PHOTON_HITPOINT_WRITER :
                                                      0;
    }
    integrator_path_cache_miss(state, DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA);
    return true;
  }

  if (is_zero(T)) {
    if (tile == nullptr) {
      integrator_path_terminate(
          kg, state, render_buffer, DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA);
    }
    return true;
  }

  /* Write camera ray to state. */
  integrator_state_write_ray(state, &ray);

  if (tile == nullptr) {
    /* Re-initialize path state for path integration. */
    path_state_init_integrator(kg, state, sample, rng_pixel, T);
    integrator_path_next(state,
                         DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA,
                         kernel_data.cam.is_inside_volume ?
                             DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK :
                             DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
  }
  else {
    /* Initialize path state for path integration. */
    path_state_init_integrator(kg, state, sample, rng_pixel, T);

    /* Continue with intersect_closest kernel, optionally initializing volume
     * stack before that if the camera may be inside a volume. */
    if (kernel_data.cam.is_inside_volume) {
      integrator_path_init(state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK);
    }
    else {
      integrator_path_init(state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
    }
  }

  /* Photon caustics: apply the writer role AFTER path_state_init_integrator -
   * it resets the path flags, which silently dropped the bit when it was set
   * earlier in this function (no caustics at all: the write never fired). */
  if (photon_writer) {
    INTEGRATOR_STATE_WRITE(state, path, flag) |= PATH_RAY_PHOTON_HITPOINT_WRITER;
  }

  return true;
}

CCL_NAMESPACE_END
