/* SPDX-FileCopyrightText: 2019, NVIDIA Corporation
 * SPDX-FileCopyrightText: 2019-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

// clang-format off
#include "kernel/device/optix/compat.h"
#include "kernel/device/optix/globals.h"

#include "kernel/device/gpu/image.h"  /* Texture lookup uses normal CUDA intrinsics. */

#include "kernel/tables.h"

#include "kernel/integrator/state.h"
#include "kernel/integrator/state_flow.h"
#include "kernel/integrator/state_util.h"

#include "kernel/integrator/intersect_closest.h"
#include "kernel/integrator/intersect_shadow.h"
#include "kernel/integrator/intersect_subsurface.h"
#include "kernel/integrator/intersect_volume_stack.h"
#include "kernel/integrator/intersect_dedicated_light.h"

#include "kernel/integrator/photon_trace.h"  /* CyclesPlus */
// clang-format on

/* Photon caustics tracing (CyclesPlus): one launch index = one photon.
 * All arguments arrive via kernel_params (staged in OptiXDeviceQueue). */
extern "C" __global__ void __raygen__kernel_optix_photon_trace()
{
  const int i = optixGetLaunchIndex().x;
  photon_trace_single(
      nullptr,
      kernel_params.photon_offset + i,
      photon_trace_batch_seed((uint64_t)kernel_params.photon_batch_k),
      kernel_params.photon_max_bounces,
      kernel_params.photon_debug_mode,
      (const ccl_global PhotonTraceLight *)kernel_params.photon_lights,
      kernel_params.photon_num_lights,
      (const ccl_global PhotonTraceTarget *)kernel_params.photon_targets,
      kernel_params.photon_num_targets,
      (const ccl_global PhotonTraceMaterial *)kernel_params.photon_materials,
      kernel_params.photon_out_pos,
      kernel_params.photon_out_flux,
      kernel_params.photon_out_counter,
      kernel_params.photon_out_capacity,
      kernel_params.photon_target_yield);
}

extern "C" __global__ void __raygen__kernel_optix_integrator_intersect_closest()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_intersect_closest(nullptr, path_index, kernel_params.render_buffer);
}

extern "C" __global__ void __raygen__kernel_optix_integrator_intersect_shadow()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_intersect_shadow(nullptr, path_index);
}

extern "C" __global__ void __raygen__kernel_optix_integrator_intersect_subsurface()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_intersect_subsurface(nullptr, path_index);
}

extern "C" __global__ void __raygen__kernel_optix_integrator_intersect_volume_stack()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_intersect_volume_stack(nullptr, path_index);
}

extern "C" __global__ void __raygen__kernel_optix_integrator_intersect_dedicated_light()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_intersect_dedicated_light(nullptr, path_index);
}
