/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/pass.h"

#include "util/log.h"
#include "util/time.h"

CCL_NAMESPACE_BEGIN

const char *pass_type_as_string(const PassType type)
{
  const int type_int = static_cast<int>(type);

  const NodeEnum *type_enum = Pass::get_type_enum();

  if (!type_enum->exists(type_int)) {
    LOG_DFATAL << "Unhandled pass type " << static_cast<int>(type) << ", not supposed to happen.";
    return "UNKNOWN";
  }

  return (*type_enum)[type_int].c_str();
}

const char *pass_mode_as_string(PassMode mode)
{
  switch (mode) {
    case PassMode::NOISY:
      return "NOISY";
    case PassMode::DENOISED:
      return "DENOISED";
  }

  LOG_DFATAL << "Unhandled pass mode " << static_cast<int>(mode) << ", should never happen.";
  return "UNKNOWN";
}

std::ostream &operator<<(std::ostream &os, PassMode mode)
{
  os << pass_mode_as_string(mode);
  return os;
}

const NodeEnum *Pass::get_type_enum()
{
  static NodeEnum pass_type_enum;

  if (pass_type_enum.empty()) {

    /* Light Passes. */
    pass_type_enum.insert("combined", PASS_COMBINED);
    pass_type_enum.insert("emission", PASS_EMISSION);
    pass_type_enum.insert("background", PASS_BACKGROUND);
    pass_type_enum.insert("ao", PASS_AO);
    pass_type_enum.insert("diffuse", PASS_DIFFUSE);
    pass_type_enum.insert("diffuse_direct", PASS_DIFFUSE_DIRECT);
    pass_type_enum.insert("diffuse_indirect", PASS_DIFFUSE_INDIRECT);
    pass_type_enum.insert("glossy", PASS_GLOSSY);
    pass_type_enum.insert("glossy_direct", PASS_GLOSSY_DIRECT);
    pass_type_enum.insert("glossy_indirect", PASS_GLOSSY_INDIRECT);
    pass_type_enum.insert("transmission", PASS_TRANSMISSION);
    pass_type_enum.insert("transmission_direct", PASS_TRANSMISSION_DIRECT);
    pass_type_enum.insert("transmission_indirect", PASS_TRANSMISSION_INDIRECT);
    pass_type_enum.insert("volume", PASS_VOLUME);
    pass_type_enum.insert("volume_direct", PASS_VOLUME_DIRECT);
    pass_type_enum.insert("volume_indirect", PASS_VOLUME_INDIRECT);
    pass_type_enum.insert("volume_scatter", PASS_VOLUME_SCATTER);
    pass_type_enum.insert("volume_transmit", PASS_VOLUME_TRANSMIT);

    /* Data passes. */
    pass_type_enum.insert("depth", PASS_DEPTH);
    pass_type_enum.insert("position", PASS_POSITION);
    pass_type_enum.insert("normal", PASS_NORMAL);
    pass_type_enum.insert("roughness", PASS_ROUGHNESS);
    pass_type_enum.insert("uv", PASS_UV);
    pass_type_enum.insert("object_id", PASS_OBJECT_ID);
    pass_type_enum.insert("material_id", PASS_MATERIAL_ID);
    pass_type_enum.insert("motion", PASS_MOTION);
    pass_type_enum.insert("motion_weight", PASS_MOTION_WEIGHT);
    pass_type_enum.insert("cryptomatte", PASS_CRYPTOMATTE);
    pass_type_enum.insert("aov_color", PASS_AOV_COLOR);
    pass_type_enum.insert("aov_value", PASS_AOV_VALUE);
    pass_type_enum.insert("adaptive_aux_buffer", PASS_ADAPTIVE_AUX_BUFFER);
    pass_type_enum.insert("sample_count", PASS_SAMPLE_COUNT);
    /* === CyclesPlus: Photon Pass Type Enum Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
    pass_type_enum.insert("caustics", PASS_CAUSTICS);
    pass_type_enum.insert("photon_hitpoint", PASS_PHOTON_HITPOINT);
    pass_type_enum.insert("photon_weight", PASS_PHOTON_WEIGHT);
    pass_type_enum.insert("photon_tau", PASS_PHOTON_TAU);
    pass_type_enum.insert("photon_state", PASS_PHOTON_STATE);
    pass_type_enum.insert("photon_tau_group", PASS_PHOTON_TAU_GROUP);
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
    /* === CyclesPlus: Photon Pass Type Enum End === */
    pass_type_enum.insert("diffuse_color", PASS_DIFFUSE_COLOR);
    pass_type_enum.insert("glossy_color", PASS_GLOSSY_COLOR);
    pass_type_enum.insert("transmission_color", PASS_TRANSMISSION_COLOR);
    pass_type_enum.insert("mist", PASS_MIST);
    pass_type_enum.insert("denoising_albedo", PASS_DENOISING_ALBEDO);
    pass_type_enum.insert("denoising_specular_albedo", PASS_DENOISING_SPECULAR_ALBEDO);
    pass_type_enum.insert("denoising_normal", PASS_DENOISING_NORMAL);
    pass_type_enum.insert("denoising_roughness", PASS_DENOISING_ROUGHNESS);
    pass_type_enum.insert("denoising_depth", PASS_DENOISING_DEPTH);
    pass_type_enum.insert("denoising_backward_motion", PASS_DENOISING_BACKWARD_MOTION);
    /* === CyclesPlus: DLSS Pass Type Enum Begin === */
#ifdef WITH_DLSS
    pass_type_enum.insert("denoising_specular_motion", PASS_DENOISING_SPECULAR_MOTION);
#endif  /* WITH_DLSS */
    /* === CyclesPlus: DLSS Pass Type Enum End === */
    pass_type_enum.insert("denoising_previous", PASS_DENOISING_PREVIOUS);
    pass_type_enum.insert("volume_majorant", PASS_VOLUME_MAJORANT);
    pass_type_enum.insert("volume_majorant_sample_count", PASS_VOLUME_MAJORANT_SAMPLE_COUNT);
    pass_type_enum.insert("render_time", PASS_RENDER_TIME);

    pass_type_enum.insert("shadow_catcher", PASS_SHADOW_CATCHER);
    pass_type_enum.insert("shadow_catcher_sample_count", PASS_SHADOW_CATCHER_SAMPLE_COUNT);
    pass_type_enum.insert("shadow_catcher_matte", PASS_SHADOW_CATCHER_MATTE);

    pass_type_enum.insert("bake_primitive", PASS_BAKE_PRIMITIVE);
    pass_type_enum.insert("bake_seed", PASS_BAKE_SEED);
    pass_type_enum.insert("bake_differential", PASS_BAKE_DIFFERENTIAL);

#ifdef WITH_CYCLES_DEBUG
    pass_type_enum.insert("guiding_color", PASS_GUIDING_COLOR);
    pass_type_enum.insert("guiding_probability", PASS_GUIDING_PROBABILITY);
    pass_type_enum.insert("guiding_avg_roughness", PASS_GUIDING_AVG_ROUGHNESS);
#endif
  }

  return &pass_type_enum;
}

const NodeEnum *Pass::get_mode_enum()
{
  static NodeEnum pass_mode_enum;

  if (pass_mode_enum.empty()) {
    pass_mode_enum.insert("noisy", static_cast<int>(PassMode::NOISY));
    pass_mode_enum.insert("denoised", static_cast<int>(PassMode::DENOISED));
  }

  return &pass_mode_enum;
}

NODE_DEFINE(Pass)
{
  NodeType *type = NodeType::add("pass", create);

  const NodeEnum *pass_type_enum = get_type_enum();
  const NodeEnum *pass_mode_enum = get_mode_enum();

  SOCKET_ENUM(type, "Type", *pass_type_enum, PASS_COMBINED);
  SOCKET_ENUM(mode, "Mode", *pass_mode_enum, static_cast<int>(PassMode::DENOISED));
  SOCKET_STRING(name, "Name", ustring());
  SOCKET_BOOLEAN(include_albedo, "Include Albedo", false);
  SOCKET_STRING(lightgroup, "Light Group", ustring());

  return type;
}

Pass::Pass() : Node(get_node_type()), is_auto_(false) {}

PassInfo Pass::get_info() const
{
  return get_info(type, mode, include_albedo, !lightgroup.empty());
}

bool Pass::is_written() const
{
  return get_info().is_written;
}

PassInfo Pass::get_info(const PassType type,
                        const PassMode mode,
                        const bool include_albedo,
                        const bool is_lightgroup)
{
  PassInfo pass_info;

  pass_info.use_filter = true;
  pass_info.use_exposure = false;
  pass_info.divide_type = PASS_NONE;
  pass_info.use_compositing = false;
  pass_info.use_denoising_albedo = true;

  switch (type) {
    case PASS_NONE:
      pass_info.num_components = 0;
      break;
    case PASS_COMBINED:
      pass_info.num_components = is_lightgroup ? 3 : 4;
      pass_info.use_exposure = true;
      pass_info.support_denoise = !is_lightgroup;
      break;
    case PASS_DEPTH:
      pass_info.num_components = 1;
      pass_info.use_filter = false;
      break;
    case PASS_MIST:
      pass_info.num_components = 1;
      break;
    case PASS_POSITION:
      pass_info.num_components = 3;
      pass_info.use_filter = false;
      break;
    case PASS_NORMAL:
      pass_info.num_components = 3;
      break;
    case PASS_ROUGHNESS:
      pass_info.num_components = 1;
      break;
    case PASS_UV:
      pass_info.num_components = 3;
      break;
    case PASS_MOTION:
      pass_info.num_components = 4;
      pass_info.divide_type = PASS_MOTION_WEIGHT;
      break;
    case PASS_MOTION_WEIGHT:
      pass_info.num_components = 1;
      break;
    case PASS_OBJECT_ID:
    case PASS_MATERIAL_ID:
      pass_info.num_components = 1;
      pass_info.use_filter = false;
      break;

    case PASS_EMISSION:
    case PASS_BACKGROUND:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      break;
    case PASS_AO:
      pass_info.num_components = 3;
      break;

    case PASS_DIFFUSE_COLOR:
    case PASS_GLOSSY_COLOR:
    case PASS_TRANSMISSION_COLOR:
      pass_info.num_components = 3;
      break;
    case PASS_DIFFUSE:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      pass_info.direct_type = PASS_DIFFUSE_DIRECT;
      pass_info.indirect_type = PASS_DIFFUSE_INDIRECT;
      pass_info.divide_type = (!include_albedo) ? PASS_DIFFUSE_COLOR : PASS_NONE;
      pass_info.use_compositing = true;
      pass_info.is_written = false;
      break;
    case PASS_DIFFUSE_DIRECT:
    case PASS_DIFFUSE_INDIRECT:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      pass_info.divide_type = (!include_albedo) ? PASS_DIFFUSE_COLOR : PASS_NONE;
      pass_info.use_compositing = true;
      break;
    case PASS_GLOSSY:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      pass_info.direct_type = PASS_GLOSSY_DIRECT;
      pass_info.indirect_type = PASS_GLOSSY_INDIRECT;
      pass_info.divide_type = (!include_albedo) ? PASS_GLOSSY_COLOR : PASS_NONE;
      pass_info.use_compositing = true;
      pass_info.is_written = false;
      break;
    case PASS_GLOSSY_DIRECT:
    case PASS_GLOSSY_INDIRECT:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      pass_info.divide_type = (!include_albedo) ? PASS_GLOSSY_COLOR : PASS_NONE;
      pass_info.use_compositing = true;
      break;
    case PASS_TRANSMISSION:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      pass_info.direct_type = PASS_TRANSMISSION_DIRECT;
      pass_info.indirect_type = PASS_TRANSMISSION_INDIRECT;
      pass_info.divide_type = (!include_albedo) ? PASS_TRANSMISSION_COLOR : PASS_NONE;
      pass_info.use_compositing = true;
      pass_info.is_written = false;
      break;
    case PASS_TRANSMISSION_DIRECT:
    case PASS_TRANSMISSION_INDIRECT:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      pass_info.divide_type = (!include_albedo) ? PASS_TRANSMISSION_COLOR : PASS_NONE;
      pass_info.use_compositing = true;
      break;
    case PASS_VOLUME:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      pass_info.direct_type = PASS_VOLUME_DIRECT;
      pass_info.indirect_type = PASS_VOLUME_INDIRECT;
      pass_info.use_compositing = true;
      pass_info.is_written = false;
      break;
    case PASS_VOLUME_DIRECT:
    case PASS_VOLUME_INDIRECT:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      break;
    case PASS_VOLUME_SCATTER:
    case PASS_VOLUME_TRANSMIT:
      /* Noisy buffer needs higher precision for accumulating the contribution, denoised buffer is
       * used directly and thus can have lower resolution. */
      pass_info.num_components = (mode == PassMode::NOISY) ? 3 : 1;
      pass_info.use_exposure = true;
      pass_info.use_filter = false;
      pass_info.support_denoise = true;
      break;
    /* === CyclesPlus: Photon Caustics Pass Info Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
    case PASS_CAUSTICS:
      /* Photon caustics (CyclesPlus). The gather kernel writes the caustic
       * scaled by the sample count, exactly like the combined pass - so this
       * must keep use_filter (which in PassInfo means "divide by the sample
       * count on read", not "apply the pixel filter"). Setting it false made
       * the pass read back num_samples times too bright, and Combined minus
       * Caustics no longer cancelled. */
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      /* Denoised, unlike Cycles' other light passes: an isolated caustic is
       * the grainiest thing in the image and compositing it is the whole
       * point of the pass, so a raw one would be half a feature.
       *
       * This flag ONLY makes finalize_passes keep a denoised slot - it does
       * not denoise anything by itself. It must stay in lockstep with the
       * denoiser pass lists (denoiser_gpu.cpp / denoiser_oidn.cpp): the flag
       * without the lists allocates a slot nobody fills, and the pass reads
       * back pure BLACK whenever denoising is on (exactly how this shipped
       * broken once). caustics_pass_test.py renders with the denoiser on to
       * keep the two ends honest.
       *
       * Per-light-group caustics are the exception and stay noisy, exactly
       * like Cycles' own light group passes: denoising is non-linear, so
       * denoised groups would no longer sum to the single Caustics pass, and
       * that identity is what makes the split verifiable. Staying noisy also
       * side-steps the whole noisy-twin problem for them. */
      pass_info.support_denoise = !is_lightgroup;
      break;
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
    /* === CyclesPlus: Photon Caustics Pass Info End === */
    case PASS_VOLUME_MAJORANT:
      pass_info.num_components = 1;
      pass_info.use_filter = false;
      pass_info.divide_type = PASS_VOLUME_MAJORANT_SAMPLE_COUNT;
      break;
    case PASS_VOLUME_MAJORANT_SAMPLE_COUNT:
      pass_info.num_components = 1;
      pass_info.use_filter = false;
      break;

    case PASS_CRYPTOMATTE:
      pass_info.num_components = 4;
      break;

    case PASS_DENOISING_ALBEDO:
    case PASS_DENOISING_SPECULAR_ALBEDO:
    case PASS_DENOISING_NORMAL:
    case PASS_DENOISING_BACKWARD_MOTION:
    /* === CyclesPlus: DLSS Specular Motion Pass Info Begin === */
#ifdef WITH_DLSS
    case PASS_DENOISING_SPECULAR_MOTION:
#endif  /* WITH_DLSS */
    /* === CyclesPlus: DLSS Specular Motion Pass Info End === */
      pass_info.num_components = 3;
      break;
    case PASS_DENOISING_ROUGHNESS:
    case PASS_DENOISING_DEPTH:
      pass_info.num_components = 1;
      break;
    case PASS_DENOISING_PREVIOUS:
      pass_info.num_components = 3;
      pass_info.use_exposure = true;
      break;

    case PASS_SHADOW_CATCHER:
      pass_info.num_components = 3;
      pass_info.use_exposure = false;
      pass_info.use_compositing = true;
      pass_info.use_denoising_albedo = false;
      pass_info.support_denoise = true;
      break;
    case PASS_SHADOW_CATCHER_SAMPLE_COUNT:
      pass_info.num_components = 1;
      break;
    case PASS_SHADOW_CATCHER_MATTE:
      pass_info.num_components = 4;
      pass_info.use_exposure = true;
      pass_info.support_denoise = true;
      /* Without shadow catcher approximation compositing is not needed.
       * Since we don't know here whether approximation is used or not, leave the decision up to
       * the caller which will know that. */
      break;

    case PASS_ADAPTIVE_AUX_BUFFER:
      pass_info.num_components = 4;
      break;
    case PASS_SAMPLE_COUNT:
      pass_info.num_components = 1;
      pass_info.use_exposure = false;
      break;

    /* === CyclesPlus: Photon Statistics Pass Info Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
    /* Photon caustics SPPM statistics (CyclesPlus): raw per-pixel state,
     * never filtered, exposed or accumulated like light passes. */
    case PASS_PHOTON_HITPOINT:
    case PASS_PHOTON_WEIGHT:
    case PASS_PHOTON_TAU:
    case PASS_PHOTON_STATE:
      pass_info.num_components = 4;
      pass_info.use_exposure = false;
      pass_info.use_filter = false;
      break;
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
    /* === CyclesPlus: Photon Statistics Pass Info End === */
    case PASS_PHOTON_TAU_GROUP:
      /* One per light group, holding that group's share of tau. Three
       * components - the photon count and the radius stay shared in
       * PASS_PHOTON_TAU / PASS_PHOTON_STATE, because every group is gathered
       * with the same disc. */
      pass_info.num_components = 3;
      pass_info.use_exposure = false;
      pass_info.use_filter = false;
      break;
    case PASS_RENDER_TIME:
      pass_info.num_components = 1;
      pass_info.use_exposure = false;
      pass_info.use_filter = false;
      pass_info.scale = 1000.0f / float(time_fast_frequency());
      break;

    case PASS_AOV_COLOR:
      pass_info.num_components = 4;
      break;
    case PASS_AOV_VALUE:
      pass_info.num_components = 1;
      break;

    case PASS_BAKE_PRIMITIVE:
      pass_info.num_components = 3;
      pass_info.use_exposure = false;
      pass_info.use_filter = false;
      break;
    case PASS_BAKE_SEED:
      pass_info.num_components = 1;
      pass_info.use_exposure = false;
      pass_info.use_filter = false;
      break;
    case PASS_BAKE_DIFFERENTIAL:
      pass_info.num_components = 4;
      pass_info.use_exposure = false;
      pass_info.use_filter = false;
      break;

    case PASS_GUIDING_COLOR:
      pass_info.num_components = 3;
      break;
    case PASS_GUIDING_PROBABILITY:
      pass_info.num_components = 1;
      break;
    case PASS_GUIDING_AVG_ROUGHNESS:
      pass_info.num_components = 1;
      break;

    default:
      LOG_DFATAL << "Unexpected pass type is used " << type;
      pass_info.num_components = 0;
      break;
  }

  return pass_info;
}

bool Pass::contains(const unique_ptr_vector<Pass> &passes, PassType type)
{
  for (const Pass *pass : passes) {
    if (pass->get_type() != type) {
      continue;
    }

    return true;
  }

  return false;
}

const Pass *Pass::find(const unique_ptr_vector<Pass> &passes, const string &name)
{
  for (const Pass *pass : passes) {
    if (pass->get_name() == name) {
      return pass;
    }
  }

  return nullptr;
}

const Pass *Pass::find(const unique_ptr_vector<Pass> &passes,
                       PassType type,
                       PassMode mode,
                       const ustring &lightgroup)
{
  for (const Pass *pass : passes) {
    if (pass->get_type() != type || pass->get_mode() != mode ||
        pass->get_lightgroup() != lightgroup)
    {
      continue;
    }
    return pass;
  }

  return nullptr;
}

int Pass::get_offset(const unique_ptr_vector<Pass> &passes, const Pass *pass)
{
  int pass_offset = 0;

  for (const Pass *current_pass : passes) {
    /* Note that pass name is allowed to be empty. This is why we check for type and mode. */
    if (current_pass->get_type() == pass->get_type() &&
        current_pass->get_mode() == pass->get_mode() &&
        current_pass->get_name() == pass->get_name())
    {
      if (current_pass->is_written()) {
        return pass_offset;
      }
      return PASS_UNUSED;
    }
    if (current_pass->is_written()) {
      pass_offset += current_pass->get_info().num_components;
    }
  }

  return PASS_UNUSED;
}

std::ostream &operator<<(std::ostream &os, const Pass &pass)
{
  os << "type: " << pass_type_as_string(pass.get_type());
  os << ", name: \"" << pass.get_name() << "\"";
  os << ", mode: " << pass.get_mode();
  os << ", is_written: " << string_from_bool(pass.is_written());

  return os;
}

bool is_volume_guiding_pass(const PassType pass_type)
{
  return (pass_type == PASS_VOLUME_SCATTER) || (pass_type == PASS_VOLUME_TRANSMIT);
}

CCL_NAMESPACE_END
