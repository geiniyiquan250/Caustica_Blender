/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef KERNEL_STRUCT_BEGIN
#  define KERNEL_STRUCT_BEGIN(name, parent)
#endif
#ifndef KERNEL_STRUCT_END
#  define KERNEL_STRUCT_END(name)
#endif
#ifndef KERNEL_STRUCT_MEMBER
#  define KERNEL_STRUCT_MEMBER(parent, type, name)
#endif
#ifndef KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
#  define KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
#endif

/* Background. */

KERNEL_STRUCT_BEGIN(KernelBackground, background)
/* xyz store direction, w the angle. float4 instead of float3 is used
 * to ensure consistent padding/alignment across devices. */
KERNEL_STRUCT_MEMBER(background, float4, sun)
KERNEL_STRUCT_MEMBER(background, int, use_sun_guiding)
/* Only shader index. */
KERNEL_STRUCT_MEMBER(background, int, surface_shader)
KERNEL_STRUCT_MEMBER(background, int, volume_shader)
KERNEL_STRUCT_MEMBER(background, int, transparent)
KERNEL_STRUCT_MEMBER(background, float, transparent_roughness_squared_threshold)
/* Sun sampling. */
KERNEL_STRUCT_MEMBER(background, float, sun_weight)
/* Importance map sampling. */
KERNEL_STRUCT_MEMBER(background, float, map_weight)
KERNEL_STRUCT_MEMBER(background, float, portal_weight)
KERNEL_STRUCT_MEMBER(background, int, map_res_x)
KERNEL_STRUCT_MEMBER(background, int, map_res_y)
/* Ray differential used for generating the importance map. */
KERNEL_STRUCT_MEMBER(background, float, map_dD)
/* Multiple importance sampling. */
KERNEL_STRUCT_MEMBER(background, int, use_mis)
/* Light-group. */
KERNEL_STRUCT_MEMBER(background, int, lightgroup)
/* Object Index. */
KERNEL_STRUCT_MEMBER(background, int, object_index)
/* Padding. */
KERNEL_STRUCT_MEMBER(background, int, pad1)
KERNEL_STRUCT_END(KernelBackground)

/* BVH: own BVH2 if no native device acceleration struct used. */

KERNEL_STRUCT_BEGIN(KernelBVH, bvh)
KERNEL_STRUCT_MEMBER(bvh, int, root)
KERNEL_STRUCT_MEMBER(bvh, int, have_motion)
KERNEL_STRUCT_MEMBER(bvh, int, have_curves)
KERNEL_STRUCT_MEMBER(bvh, int, have_points)
KERNEL_STRUCT_MEMBER(bvh, int, have_volumes)
KERNEL_STRUCT_MEMBER(bvh, int, bvh_layout)
KERNEL_STRUCT_MEMBER(bvh, int, use_bvh_steps)
KERNEL_STRUCT_MEMBER(bvh, int, curve_subdivisions)
KERNEL_STRUCT_END(KernelBVH)

/* Film. */

KERNEL_STRUCT_BEGIN(KernelFilm, film)
/* XYZ to rendering color space transform. float4 instead of float3 to
 * ensure consistent padding/alignment across devices. */
KERNEL_STRUCT_MEMBER(film, float4, xyz_to_r)
KERNEL_STRUCT_MEMBER(film, float4, xyz_to_g)
KERNEL_STRUCT_MEMBER(film, float4, xyz_to_b)
KERNEL_STRUCT_MEMBER(film, float4, rgb_to_y)
KERNEL_STRUCT_MEMBER(film, float4, white_xyz)
/* Rec709 to rendering color space. */
KERNEL_STRUCT_MEMBER(film, float4, rec709_to_r)
KERNEL_STRUCT_MEMBER(film, float4, rec709_to_g)
KERNEL_STRUCT_MEMBER(film, float4, rec709_to_b)
KERNEL_STRUCT_MEMBER(film, int, is_rec709)
/* Exposure. */
KERNEL_STRUCT_MEMBER(film, float, exposure)
/* Passed used. */
KERNEL_STRUCT_MEMBER(film, int, pass_flag)
KERNEL_STRUCT_MEMBER(film, int, denoising_pass_flag)
KERNEL_STRUCT_MEMBER(film, int, light_pass_flag)
/* Pass offsets. */
KERNEL_STRUCT_MEMBER(film, int, pass_stride)
KERNEL_STRUCT_MEMBER(film, int, pass_combined)
KERNEL_STRUCT_MEMBER(film, int, pass_depth)
KERNEL_STRUCT_MEMBER(film, int, pass_position)
KERNEL_STRUCT_MEMBER(film, int, pass_normal)
KERNEL_STRUCT_MEMBER(film, int, pass_roughness)
KERNEL_STRUCT_MEMBER(film, int, pass_motion)
KERNEL_STRUCT_MEMBER(film, int, pass_motion_weight)
KERNEL_STRUCT_MEMBER(film, int, pass_uv)
KERNEL_STRUCT_MEMBER(film, int, pass_object_id)
KERNEL_STRUCT_MEMBER(film, int, pass_material_id)
KERNEL_STRUCT_MEMBER(film, int, pass_diffuse_color)
KERNEL_STRUCT_MEMBER(film, int, pass_glossy_color)
KERNEL_STRUCT_MEMBER(film, int, pass_transmission_color)
KERNEL_STRUCT_MEMBER(film, int, pass_diffuse_indirect)
KERNEL_STRUCT_MEMBER(film, int, pass_glossy_indirect)
KERNEL_STRUCT_MEMBER(film, int, pass_transmission_indirect)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_indirect)
KERNEL_STRUCT_MEMBER(film, int, pass_diffuse_direct)
KERNEL_STRUCT_MEMBER(film, int, pass_glossy_direct)
KERNEL_STRUCT_MEMBER(film, int, pass_transmission_direct)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_direct)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_scatter)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_scatter_denoised)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_transmit)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_transmit_denoised)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_majorant)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_majorant_sample_count)
KERNEL_STRUCT_MEMBER(film, int, pass_emission)
KERNEL_STRUCT_MEMBER(film, int, pass_background)
KERNEL_STRUCT_MEMBER(film, int, pass_ao)
KERNEL_STRUCT_MEMBER(film, float, pass_alpha_threshold)
KERNEL_STRUCT_MEMBER(film, int, pass_shadow_catcher)
KERNEL_STRUCT_MEMBER(film, int, pass_shadow_catcher_sample_count)
KERNEL_STRUCT_MEMBER(film, int, pass_shadow_catcher_matte)
KERNEL_STRUCT_MEMBER(film, int, pass_render_time)
/* Cryptomatte. */
KERNEL_STRUCT_MEMBER(film, int, cryptomatte_passes)
KERNEL_STRUCT_MEMBER(film, int, cryptomatte_depth)
KERNEL_STRUCT_MEMBER(film, int, pass_cryptomatte)
/* Adaptive sampling. */
KERNEL_STRUCT_MEMBER(film, int, pass_adaptive_aux_buffer)
KERNEL_STRUCT_MEMBER(film, int, pass_sample_count)
/* Photon caustics SPPM statistics (CyclesPlus). */
/* User-visible caustics pass (CyclesPlus): receives the same delta the
 * combined pass gets, so it holds exactly the caustic share of the image. */
KERNEL_STRUCT_MEMBER(film, int, pass_caustics)
KERNEL_STRUCT_MEMBER(film, int, pass_photon_hitpoint)
KERNEL_STRUCT_MEMBER(film, int, pass_photon_weight)
KERNEL_STRUCT_MEMBER(film, int, pass_photon_tau)
KERNEL_STRUCT_MEMBER(film, int, pass_photon_state)
/* Per-light-group caustics (CyclesPlus). Both blocks are contiguous and share
 * one index: group g writes its estimate to pass_caustics_lightgroup + 3*g and
 * keeps its running tau at pass_photon_tau_group + 3*g. The group count is the
 * kernel's loop bound, gathered in chunks of PHOTON_LIGHTGROUP_CHUNK. */
KERNEL_STRUCT_MEMBER(film, int, pass_caustics_lightgroup)
KERNEL_STRUCT_MEMBER(film, int, pass_photon_tau_group)
KERNEL_STRUCT_MEMBER(film, int, num_caustics_lightgroups)
/* Mist. */
KERNEL_STRUCT_MEMBER(film, int, pass_mist)
KERNEL_STRUCT_MEMBER(film, float, mist_start)
KERNEL_STRUCT_MEMBER(film, float, mist_inv_depth)
KERNEL_STRUCT_MEMBER(film, float, mist_falloff)
/* Denoising. */
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_albedo)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_specular_albedo)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_normal)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_roughness)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_depth)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_backward_motion)
KERNEL_STRUCT_MEMBER(film, int, denoising_pass_options_flag)
/* AOVs. */
KERNEL_STRUCT_MEMBER(film, int, pass_aov_color)
KERNEL_STRUCT_MEMBER(film, int, pass_aov_value)
/* Light groups. */
KERNEL_STRUCT_MEMBER(film, int, pass_lightgroup)
/* Baking. */
KERNEL_STRUCT_MEMBER(film, int, pass_bake_primitive)
KERNEL_STRUCT_MEMBER(film, int, pass_bake_seed)
KERNEL_STRUCT_MEMBER(film, int, pass_bake_differential)
/* Shadow catcher. */
KERNEL_STRUCT_MEMBER(film, int, use_approximate_shadow_catcher)
/* Path Guiding */
KERNEL_STRUCT_MEMBER(film, int, pass_guiding_color)
KERNEL_STRUCT_MEMBER(film, int, pass_guiding_probability)
KERNEL_STRUCT_MEMBER(film, int, pass_guiding_avg_roughness)
/* Padding. */
KERNEL_STRUCT_MEMBER(film, int, pad1)
KERNEL_STRUCT_END(KernelFilm)

/* Integrator. */

KERNEL_STRUCT_BEGIN(KernelIntegrator, integrator)
/* Emission. */
KERNEL_STRUCT_MEMBER(integrator, int, use_direct_light)
KERNEL_STRUCT_MEMBER(integrator, int, use_light_mis)
KERNEL_STRUCT_MEMBER(integrator, int, use_light_tree)
KERNEL_STRUCT_MEMBER(integrator, int, num_lights)
KERNEL_STRUCT_MEMBER(integrator, int, num_distant_lights)
KERNEL_STRUCT_MEMBER(integrator, int, num_background_lights)
/* Portal sampling. */
KERNEL_STRUCT_MEMBER(integrator, int, num_portals)
KERNEL_STRUCT_MEMBER(integrator, int, portal_offset)
/* Flat light distribution. */
KERNEL_STRUCT_MEMBER(integrator, int, num_distribution)
KERNEL_STRUCT_MEMBER(integrator, float, distribution_pdf_triangles)
KERNEL_STRUCT_MEMBER(integrator, float, distribution_pdf_lights)
KERNEL_STRUCT_MEMBER(integrator, float, light_inv_rr_threshold)
/* Bounces. */
KERNEL_STRUCT_MEMBER(integrator, int, min_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_diffuse_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_glossy_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_transmission_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_volume_bounce)
/* AO bounces. */
KERNEL_STRUCT_MEMBER(integrator, int, ao_bounces)
KERNEL_STRUCT_MEMBER(integrator, float, ao_bounces_distance)
KERNEL_STRUCT_MEMBER(integrator, float, ao_bounces_factor)
KERNEL_STRUCT_MEMBER(integrator, float, ao_additive_factor)
/* Transparency. */
KERNEL_STRUCT_MEMBER(integrator, int, transparent_min_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, transparent_max_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, transparent_shadows)
/* Caustics. */
KERNEL_STRUCT_MEMBER(integrator, int, caustics_reflective)
KERNEL_STRUCT_MEMBER(integrator, int, caustics_refractive)
KERNEL_STRUCT_MEMBER(integrator, float, filter_glossy)
/* Photon caustics (CyclesPlus). Grid parameters update per progressive
 * batch; the arrays live in data_arrays.h. photon_radius is the constant
 * cell size and the per-pixel SPPM start radius; the per-pixel radii live
 * in the PASS_PHOTON_* film passes and only ever shrink below it. */
KERNEL_STRUCT_MEMBER(integrator, int, use_photon_caustics)
/* Whether only flagged materials cast photons. */
KERNEL_STRUCT_MEMBER(integrator, int, photon_casters_selected)
/* Whether the photon map REPLACES the path-traced caustics of a material it
 * casts from, instead of adding to them. Off since 2026-08-25: replacing
 * costs more than it saves - see svm/photon_caustics.h. Kept as a support
 * hatch (CYCLESPLUS_PHOTON_REPLACE_PT=1), not as a user setting. */
KERNEL_STRUCT_MEMBER(integrator, int, photon_replace_pt)
/* Whether path tracing hands its caustic contributions to the photon map
 * instead of adding a second estimate of the same light. Narrow: only the
 * contribution is skipped, never a closure. CYCLESPLUS_PHOTON_PARTITION. */
KERNEL_STRUCT_MEMBER(integrator, int, photon_partition_pt)
/* Entries in the photon_shader_caster array; 0 disables the partition. */
KERNEL_STRUCT_MEMBER(integrator, int, photon_shader_count)
KERNEL_STRUCT_MEMBER(integrator, int, photon_num)
KERNEL_STRUCT_MEMBER(integrator, int, photon_table_size)
KERNEL_STRUCT_MEMBER(integrator, float, photon_radius)
KERNEL_STRUCT_MEMBER(integrator, float, photon_inv_cell)
KERNEL_STRUCT_MEMBER(integrator, int, photon_batch_steady)
/* Global last scheduled sample of the CURRENT render work. The photon
 * hitpoint writer must key on this, NOT on the KernelWorkTile's sample
 * range: large images split one work into many tile launches, and a
 * per-tile "last sample" spawns several concurrent writers per pixel per
 * work - the hitpoint race in tile shape (equal-sized rectangular patches
 * in the caustic, user 4MP car renders; invisible in the 200px test
 * battery where one launch covers the whole work). Updated per work by
 * PathTrace. */
KERNEL_STRUCT_MEMBER(integrator, int, photon_writer_sample)
/* Weight of the current photon generation in 1/16th of a full batch
 * (1..16). Interactive batch ramps publish small early generations; the
 * SPPM tau average weighs every gather by this, so full generations
 * dominate immediately instead of splitting the average evenly with the
 * grainy preview batches. */
KERNEL_STRUCT_MEMBER(integrator, int, photon_batch_weight)
/* Bitmask enabling the SPPM estimator heuristics (bit0 boundary-leak
 * suppression, bit1 per-gather clamp, bit2 batch weighting, bit3 radius
 * floor, bit4 maturity-faded display smoothing). Default all on (0x1F);
 * CYCLESPLUS_PHOTON_HEURISTICS overrides - mask 0 is textbook SPPM, the
 * ablation tool for estimator regressions. */
KERNEL_STRUCT_MEMBER(integrator, int, photon_heuristic_mask)
/* Artistic caustic multiplier (1 = physically correct). Scales the SPPM
 * estimate at the gather, so it never touches photon transport, clamping
 * or the radius statistics - and a change takes effect on the next gather
 * (the film delta write corrects itself against the last written value). */
KERNEL_STRUCT_MEMBER(integrator, float, photon_intensity)
/* Seed. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, seed)
/* Clamp. */
KERNEL_STRUCT_MEMBER(integrator, float, sample_clamp_direct)
KERNEL_STRUCT_MEMBER(integrator, float, sample_clamp_indirect)
/* Caustics. */
KERNEL_STRUCT_MEMBER(integrator, int, use_caustics)
/* Sampling pattern. */
KERNEL_STRUCT_MEMBER(integrator, int, sampling_pattern)
KERNEL_STRUCT_MEMBER(integrator, float, scrambling_distance)
/* Sobol pattern. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, tabulated_sobol_sequence_size)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, sobol_index_mask)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, blue_noise_sequence_length)
/* Volume render. */
KERNEL_STRUCT_MEMBER(integrator, int, use_volumes)
KERNEL_STRUCT_MEMBER(integrator, int, volume_ray_marching)
KERNEL_STRUCT_MEMBER(integrator, int, volume_max_steps)
/* Shadow catcher. */
KERNEL_STRUCT_MEMBER(integrator, int, has_shadow_catcher)
/* Closure filter. */
KERNEL_STRUCT_MEMBER(integrator, int, filter_closures)
/* MIS debugging. */
KERNEL_STRUCT_MEMBER(integrator, int, direct_light_sampling_type)
/* Path Guiding */
KERNEL_STRUCT_MEMBER(integrator, float, surface_guiding_probability)
KERNEL_STRUCT_MEMBER(integrator, float, volume_guiding_probability)
KERNEL_STRUCT_MEMBER(integrator, int, guiding_distribution_type)
KERNEL_STRUCT_MEMBER(integrator, int, guiding_directional_sampling_type)
KERNEL_STRUCT_MEMBER(integrator, float, guiding_roughness_threshold)
KERNEL_STRUCT_MEMBER(integrator, int, use_guiding)
KERNEL_STRUCT_MEMBER(integrator, int, train_guiding)
KERNEL_STRUCT_MEMBER(integrator, int, use_surface_guiding)
KERNEL_STRUCT_MEMBER(integrator, int, use_volume_guiding)
KERNEL_STRUCT_MEMBER(integrator, int, use_guiding_direct_light)
KERNEL_STRUCT_MEMBER(integrator, int, use_guiding_mis_weights)

KERNEL_STRUCT_MEMBER(integrator, float2, pixel_jitter)
KERNEL_STRUCT_END(KernelIntegrator)

/* Image. */

KERNEL_STRUCT_BEGIN(KernelImage, image)
KERNEL_STRUCT_MEMBER(image, float, mip_bias)

/* Padding. */
KERNEL_STRUCT_MEMBER(image, int, pad1)
KERNEL_STRUCT_MEMBER(image, int, pad2)
KERNEL_STRUCT_MEMBER(image, int, pad3)
KERNEL_STRUCT_END(KernelImage)

/* SVM. For shader specialization. */

KERNEL_STRUCT_BEGIN(KernelSVMUsage, svm_usage)
#define SHADER_NODE_TYPE(type) KERNEL_STRUCT_MEMBER(svm_usage, int, type)
#define SHADER_NODE_TYPE_DERIVATIVE(type) \
  SHADER_NODE_TYPE(type) \
  SHADER_NODE_TYPE(type##_DERIVATIVE)
#include "kernel/svm/node_types_template.h"
KERNEL_STRUCT_END(KernelSVMUsage)

#undef KERNEL_STRUCT_BEGIN
#undef KERNEL_STRUCT_MEMBER
#undef KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
#undef KERNEL_STRUCT_END
