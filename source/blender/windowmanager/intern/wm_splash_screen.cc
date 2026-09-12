/* SPDX-FileCopyrightText: 2007 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup wm
 *
 * This file contains the splash screen logic (the `WM_OT_splash` operator).
 *
 * - Loads the splash image.
 * - Displaying version information.
 * - Lists New Files (application templates).
 * - Lists Recent files.
 * - Links to web sites.
 */

#include <cmath>
#include <cstring>

#include "DNA_screen_types.h"
#include "DNA_userdef_types.h"
#include "DNA_windowmanager_types.h"

#include "MEM_guardedalloc.h"

#include "BLI_listbase.h"
#include "BLI_math_color.h"
#include "BLI_math_base.h"
#include "BLI_path_utils.hh"
#include "BLI_utildefines.h"

#include "BKE_appdir.hh"
#include "BKE_blender_version.h"
#include "BKE_context.hh"
#include "BKE_preferences.h"

#include "BLT_translation.hh"

#include "IMB_imbuf.hh"
#include "IMB_imbuf_types.hh"

#include "ED_datafiles.h"
#include "ED_screen.hh"

#include "GPU_immediate.hh"
#include "GPU_state.hh"

#include "RNA_access.hh"

#include "UI_interface.hh"
#include "UI_interface_icons.hh"
#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "interface_intern.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "wm.hh"

namespace blender {

/* -------------------------------------------------------------------- */
/** \name Splash Screen
 * \{ */

struct SplashAnimationState {
  wmWindowManager *wm = nullptr;
  wmWindow *win = nullptr;
  ARegion *region = nullptr;
  wmTimer *timer = nullptr;
  float phase = 0.0f;
  float color_phase = 0.0f;
  float grid_phase = 0.0f;
};

static void wm_splash_animation_free(void *arg)
{
  SplashAnimationState *state = static_cast<SplashAnimationState *>(arg);
  if (!state) {
    return;
  }

  if (state->timer && state->wm) {
    WM_event_timer_remove(state->wm, state->win, state->timer);
  }
  MEM_delete(state);
}

static void wm_splash_animation_line(uint pos,
                                     uint col,
                                     const float color[4],
                                     float x1,
                                     float y1,
                                     float x2,
                                     float y2)
{
  immAttr4fv(col, color);
  immVertex2f(pos, x1, y1);
  immAttr4fv(col, color);
  immVertex2f(pos, x2, y2);
}

static void wm_splash_animation_draw(const SplashAnimationState *state, rcti *rect)
{
  if (!state || !rect || rect->xmin >= rect->xmax || rect->ymin >= rect->ymax) {
    return;
  }

  const float xmin = float(rect->xmin);
  const float ymin = float(rect->ymin);
  const float width = float(rect->xmax - rect->xmin);
  const float height = float(rect->ymax - rect->ymin);
  const float xmax = xmin + width;
  const float ymax = ymin + height;
  const float vanish_x = xmin + width * 0.5f;
  const float vanish_y = ymin + height * 0.60f;

  int scissor[4];
  GPU_scissor_get(scissor);
  const GPUBlend old_blend = GPU_blend_get();
  const bool old_line_smooth = GPU_line_smooth_get();
  GPU_scissor(rect->xmin, rect->ymin, rect->xmax - rect->xmin, rect->ymax - rect->ymin);
  GPU_blend(GPU_BLEND_ALPHA);
  GPU_line_smooth(true);
  GPU_line_width(1.0f);

  GPUVertFormat *format = immVertexFormat();
  const uint pos = GPU_vertformat_attr_add(format, "pos", gpu::VertAttrType::SFLOAT_32_32);
  const uint col = GPU_vertformat_attr_add(
      format, "color", gpu::VertAttrType::SFLOAT_32_32_32_32);
  immBindBuiltinProgram(GPU_SHADER_3D_FLAT_COLOR);

  const float background[4] = {0.003f, 0.004f, 0.006f, 1.0f};
  immBegin(GPU_PRIM_TRI_STRIP, 4);
  wm_splash_animation_line(pos, col, background, xmin, ymin, xmax, ymin);
  wm_splash_animation_line(pos, col, background, xmin, ymax, xmax, ymax);
  immEnd();

  constexpr int horizontal_count = 18;
  GPU_line_width(1.25f);
  immBegin(GPU_PRIM_LINES, horizontal_count * 2);
  for (int i = 0; i < horizontal_count; i++) {
    const float depth = fmodf(float(i) / float(horizontal_count) + state->grid_phase, 1.0f);
    const float perspective = depth * depth;
    const float y = vanish_y - (vanish_y - ymin) * perspective;
    const float brightness = 0.45f + depth * 0.95f;
    const float color[4] = {0.16f * brightness,
                            0.48f * brightness,
                            0.82f * brightness,
                            (0.025f + depth * 0.155f) * (0.35f + depth * 0.95f)};
    wm_splash_animation_line(pos, col, color, xmin, y, xmax, y);
  }
  immEnd();

  constexpr int lane_count = 12;
  constexpr int lane_segments = 8;
  immBegin(GPU_PRIM_LINES, lane_count * lane_segments * 2);
  for (int lane = 0; lane < lane_count; lane++) {
    const float normalized_lane = (float(lane) - float(lane_count - 1) * 0.5f) /
                                  (float(lane_count - 1) * 0.5f);
    const float near_vanish_x = vanish_x + normalized_lane * width * 0.018f;
    const float bottom_x = vanish_x + normalized_lane * width * 1.05f;
    const float color[4] = {0.20f, 0.34f, 0.72f, 0.075f};

    for (int segment = 0; segment < lane_segments; segment++) {
      const float t1 = float(segment) / float(lane_segments);
      const float t2 = float(segment + 1) / float(lane_segments);
      const float bend = normalized_lane * width * 0.035f;
      const float base_grid_brightness = 1.0f + 0.35f * t1;
      const float segment_color[4] = {color[0],
                                      color[1],
                                      color[2],
                                      color[3] * (0.25f + t1) * base_grid_brightness};
      const float x1 = interpf(bottom_x, near_vanish_x, t1) + bend * t1 * t1;
      const float y1 = interpf(ymin, vanish_y, t1);
      const float x2 = interpf(bottom_x, near_vanish_x, t2) + bend * t2 * t2;
      const float y2 = interpf(ymin, vanish_y, t2);
      wm_splash_animation_line(pos, col, segment_color, x1, y1, x2, y2);
    }
  }
  immEnd();

  constexpr int accent_count = 18;
  constexpr int accent_samples = 33;
  GPU_line_width(2.0f);
  immBindBuiltinProgram(GPU_SHADER_3D_SMOOTH_COLOR);
  const float pulse_phase = state->phase;
  const float color_phase = state->color_phase;
  for (int accent = 0; accent < accent_count; accent++) {
    const float normalized_lane = (float(accent) - float(accent_count - 1) * 0.5f) /
                                  (float(accent_count - 1) * 0.5f);
    const float drift_phase = pulse_phase * M_TAU;
    const float near_vanish_x = vanish_x + normalized_lane * width * 0.028f;
    const float bottom_x = vanish_x + normalized_lane * width * 0.82f;
    const float pulse_offset = 0.018f * sinf(float(accent) * 2.399f + drift_phase * 0.61f) +
                               0.008f * cosf(float(accent) * 1.731f + drift_phase * 1.19f);
    const float hue = fmodf(color_phase + float(accent) * 0.071f, 1.0f);
    const float next_hue_value = fmodf(hue + 0.09f, 1.0f);
    float color[3];
    float next_hue[3];
    hsv_to_rgb(hue, 0.82f, 1.0f, &color[0], &color[1], &color[2]);
    hsv_to_rgb(next_hue_value, 0.82f, 1.0f, &next_hue[0], &next_hue[1], &next_hue[2]);
    const float bend = normalized_lane * width * 0.035f;

    immBegin(GPU_PRIM_LINE_STRIP, accent_samples);
    for (int sample = 0; sample < accent_samples; sample++) {
      const float u = float(sample) / float(accent_samples - 1);
      const float wave = 0.5f + 0.5f * cosf((u - pulse_phase - pulse_offset) * M_TAU);
      const float pulse = powf(wave, 8.0f);
      const float edge = min_ff(u / 0.12f, (1.0f - u) / 0.12f);
      const float edge_fade = max_ff(0.0f, min_ff(edge, 1.0f));
      const float smooth_edge = edge_fade * edge_fade * (3.0f - 2.0f * edge_fade);
      const float hue_drift = 0.10f * sinf(drift_phase * 0.47f + float(accent) * 0.83f);
      const float hue_mix = max_ff(0.0f, min_ff(1.0f, 0.12f + 0.76f * u + hue_drift));
      const float sample_color[4] = {
          interpf(next_hue[0], color[0], hue_mix),
          interpf(next_hue[1], color[1], hue_mix),
          interpf(next_hue[2], color[2], hue_mix),
          (0.04f + 0.96f * pulse) * smooth_edge * (0.42f + 0.58f * pulse)};
      const float x = interpf(bottom_x, near_vanish_x, u) + bend * u * u;
      const float y = interpf(ymin, vanish_y, u);
      immAttr4fv(col, sample_color);
      immVertex2f(pos, x, y);
    }
    immEnd();
  }

  immUnbindProgram();
  GPU_line_smooth(old_line_smooth);
  GPU_blend(old_blend);
  GPU_scissor(scissor[0], scissor[1], scissor[2], scissor[3]);
}

static int wm_splash_animation_event(const bContext * /*C*/,
                                     ui::Block *block,
                                     const wmEvent *event)
{
  SplashAnimationState *state = static_cast<SplashAnimationState *>(
      block->handle->popup_create_vars.arg);
  if (state && state->timer && event->type == TIMER && event->customdata == state->timer) {
    state->grid_phase = fmodf(state->grid_phase + 0.0048f, 1.0f);
    state->phase += 0.0084f;
    state->color_phase = fmodf(state->color_phase + 0.0027f, 1.0f);
    if (state->region) {
      ED_region_tag_redraw(state->region);
    }
    return 1;
  }
  return 0;
}

static void wm_block_splash_close(bContext *C, ui::Block *block)
{
  wmWindow *win = CTX_wm_window(C);
  popup_block_close(C, win, block);
}

static void wm_block_splash_add_label(ui::Block *block, const char *label, int x, int y)
{
  if (!(label && label[0])) {
    return;
  }

  block_emboss_set(block, ui::EmbossType::None);

  ui::Button *but = uiDefBut(
      block, ui::ButtonType::Label, label, 0, y, x, UI_UNIT_Y, nullptr, 0, 0, std::nullopt);
  button_drawflag_disable(but, ui::BUT_TEXT_LEFT);
  button_drawflag_enable(but, ui::BUT_TEXT_RIGHT);

  /* Regardless of theme, this text should always be bright white. */
  uchar color[4] = {255, 255, 255, 255};
  button_color_set(but, color);

  block_emboss_set(block, ui::EmbossType::Emboss);
}

#ifndef WITH_HEADLESS
static void wm_block_splash_image_roundcorners_add(ImBuf *ibuf)
{
  uchar *rct = ibuf->byte_data_for_write();
  if (!rct) {
    return;
  }

  bTheme *btheme = ui::theme::theme_get();
  const float roundness = btheme->tui.wcol_menu_back.roundness * UI_SCALE_FAC;
  const int size = roundness * 20;

  if (size < ibuf->x && size < ibuf->y) {
    /* Y-axis initial offset. */
    rct += 4 * (ibuf->y - size) * ibuf->x;

    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++, rct += 4) {
        const float pixel = 1.0 / size;
        const float u = pixel * x;
        const float v = pixel * y;
        const float distance = sqrt(u * u + v * v);

        /* Pointer offset to the alpha value of pixel. */
        /* NOTE: the left corner is flipped in the X-axis. */
        const int offset_l = 4 * (size - x - x - 1) + 3;
        const int offset_r = 4 * (ibuf->x - size) + 3;

        if (distance > 1.0) {
          rct[offset_l] = 0;
          rct[offset_r] = 0;
        }
        else {
          /* Create a single pixel wide transition for anti-aliasing.
           * Invert the distance and map its range [0, 1] to [0, pixel]. */
          const float fac = (1.0 - distance) * size;

          if (fac > 1.0) {
            continue;
          }

          const uchar alpha = unit_float_to_uchar_clamp(fac);
          rct[offset_l] = alpha;
          rct[offset_r] = alpha;
        }
      }

      /* X-axis offset to the next row. */
      rct += 4 * (ibuf->x - size);
    }
  }
}
#endif /* !WITH_HEADLESS */

static ImBuf *wm_block_splash_image(int width, int *r_height)
{
  ImBuf *ibuf = nullptr;
  int height = 0;
#ifndef WITH_HEADLESS
  if (U.app_template[0] != '\0') {
    char splash_filepath[FILE_MAX];
    char template_directory[FILE_MAX];
    if (BKE_appdir_app_template_id_search(
            U.app_template, template_directory, sizeof(template_directory)))
    {
      BLI_path_join(splash_filepath, sizeof(splash_filepath), template_directory, "splash.png");
      ibuf = IMB_load_image_from_filepath(splash_filepath, ImBufFlags::ByteData);
    }
  }

  if (ibuf == nullptr) {
    const char *custom_splash_path = BLI_getenv("BLENDER_CUSTOM_SPLASH");
    if (custom_splash_path) {
      ibuf = IMB_load_image_from_filepath(custom_splash_path, ImBufFlags::ByteData);
    }
  }

  if (ibuf == nullptr) {
    const uchar *splash_data = reinterpret_cast<const uchar *>(datatoc_splash_png);
    size_t splash_data_size = datatoc_splash_png_size;
    ibuf = IMB_load_image_from_memory(
        splash_data, splash_data_size, ImBufFlags::ByteData, "<splash screen>");
  }

  if (ibuf) {
    ibuf->color_mode = ImColorMode::RGBA; /* The image might not have an alpha channel. */
    height = (width * ibuf->y) / ibuf->x;
    if (width != ibuf->x || height != ibuf->y) {
      IMB_scale(ibuf, width, height, IMBScaleFilter::Box, false);
    }

    wm_block_splash_image_roundcorners_add(ibuf);
    IMB_premultiply_alpha(ibuf);
  }

#else
  UNUSED_VARS(width);
#endif
  *r_height = height;
  return ibuf;
}

static ImBuf *wm_block_splash_banner_image(int *r_width,
                                           int *r_height,
                                           int max_width,
                                           int max_height)
{
  ImBuf *ibuf = nullptr;
  int height = 0;
  int width = max_width;
#ifndef WITH_HEADLESS

  const char *custom_splash_path = BLI_getenv("BLENDER_CUSTOM_SPLASH_BANNER");
  if (custom_splash_path) {
    ibuf = IMB_load_image_from_filepath(custom_splash_path, ImBufFlags::ByteData);
  }

  if (!ibuf) {
    return nullptr;
  }

  ibuf->color_mode = ImColorMode::RGBA; /* The image might not have an alpha channel. */

  width = ibuf->x;
  height = ibuf->y;
  if (width > 0 && height > 0 && (width > max_width || height > max_height)) {
    const float splash_ratio = max_width / float(max_height);
    const float banner_ratio = ibuf->x / float(ibuf->y);

    if (banner_ratio > splash_ratio) {
      /* The banner is wider than the splash image. */
      width = max_width;
      height = max_width / banner_ratio;
    }
    else if (banner_ratio < splash_ratio) {
      /* The banner is taller than the splash image. */
      height = max_height;
      width = max_height * banner_ratio;
    }
    else {
      width = max_width;
      height = max_height;
    }
    if (width != ibuf->x || height != ibuf->y) {
      IMB_scale(ibuf, width, height, IMBScaleFilter::Box, false);
    }
  }

  IMB_premultiply_alpha(ibuf);

#else
  UNUSED_VARS(max_height);
#endif
  *r_height = height;
  *r_width = width;
  return ibuf;
}

/**
 * Close the splash when opening a file-selector.
 */
static void wm_block_splash_close_on_fileselect(bContext *C, void *arg1, void * /*arg2*/)
{
  wmWindow *win = CTX_wm_window(C);
  if (!win) {
    return;
  }

  /* Check for the event as this will run before the new window/area has been created. */
  bool has_fileselect = false;
  for (const wmEvent &event : win->runtime->event_queue) {
    if (event.type == EVT_FILESELECT) {
      has_fileselect = true;
      break;
    }
  }

  if (has_fileselect) {
    wm_block_splash_close(C, static_cast<ui::Block *>(arg1));
  }
}

#if defined(__APPLE__)
/* Check if Blender is running under Rosetta for the purpose of displaying a splash screen warning.
 * From Apple's WWDC 2020 Session - Explore the new system architecture of Apple Silicon Macs.
 * Time code: 14:31 - https://developer.apple.com/videos/play/wwdc2020/10686/ */

#  include <sys/sysctl.h>

static int is_using_macos_rosetta()
{
  int ret = 0;
  size_t size = sizeof(ret);

  if (sysctlbyname("sysctl.proc_translated", &ret, &size, nullptr, 0) != -1) {
    return ret;
  }
  /* If "sysctl.proc_translated" is not present then must be native. */
  if (errno == ENOENT) {
    return 0;
  }
  return -1;
}
#endif /* __APPLE__ */

static ui::Block *wm_block_splash_create(bContext *C, ARegion *region, void *arg)
{
  SplashAnimationState *animation = static_cast<SplashAnimationState *>(arg);
  const uiStyle *style = ui::style_get_dpi();

  ui::Block *block = block_begin(C, region, "splash", ui::EmbossType::Emboss);

  /* Note on #BLOCK_NO_WIN_CLIP, the window size is not always synchronized
   * with the OS when the splash shows, window clipping in this case gives
   * ugly results and clipping the splash isn't useful anyway, just disable it #32938. */
  block_flag_enable(block, ui::BLOCK_LOOP | ui::BLOCK_KEEP_OPEN | ui::BLOCK_NO_WIN_CLIP);
  block_theme_style_set(block, ui::BLOCK_THEME_STYLE_POPUP);

  int splash_width = style->widget.points * 45 * UI_SCALE_FAC;
  CLAMP_MAX(splash_width, WM_window_native_pixel_x(CTX_wm_window(C)) * 0.7f);
  int splash_height;

  /* Would be nice to support caching this, so it only has to be re-read (and likely resized) on
   * first draw or if the image changed. */
  ImBuf *ibuf = wm_block_splash_image(splash_width, &splash_height);
  /* This should never happen, if it does - don't crash. */
  if (LIKELY(ibuf)) {
    ui::Button *but = uiDefButImage(
        block, ibuf, 0, 0.5f * U.widget_unit, splash_width, splash_height, nullptr);

    button_func_set(but, [block](bContext &C) { wm_block_splash_close(&C, block); });

    if (animation) {
      ui::Button *animation_but = uiDefBut(block,
                                           ui::ButtonType::Extra,
                                           "",
                                           0,
                                           0.5f * U.widget_unit,
                                           splash_width,
                                           splash_height,
                                           nullptr,
                                           0.0f,
                                           0.0f,
                                           "");
      button_func_set(animation_but,
                      [block](bContext &C) { wm_block_splash_close(&C, block); });
      button_func_drawextra_set(
          block,
          [animation](const bContext * /*C*/, rcti *rect) {
            wm_splash_animation_draw(animation, rect);
          });

      animation->wm = CTX_wm_manager(C);
      animation->win = CTX_wm_window(C);
      animation->region = region;
      if (!animation->timer) {
        animation->timer = WM_event_timer_add(
            animation->wm, animation->win, TIMER, 1.0f / 60.0f);
      }
      block->block_event_func = wm_splash_animation_event;
    }

    wm_block_splash_add_label(block,
                              BKE_blender_version_string(),
                              splash_width - 8.0 * UI_SCALE_FAC,
                              splash_height - 13.0 * UI_SCALE_FAC);
  }

  /* Banner image passed through the environment, to overlay on the splash and
   * indicate a custom Blender version. Transparency can be used. To replace the
   * full splash screen, see BLENDER_CUSTOM_SPLASH. */
  int banner_width = 0;
  int banner_height = 0;
  ImBuf *bannerbuf = wm_block_splash_banner_image(
      &banner_width, &banner_height, splash_width, splash_height);
  if (bannerbuf) {
    ui::Button *banner_but = uiDefButImage(
        block, bannerbuf, 0, 0.5f * U.widget_unit, banner_width, banner_height, nullptr);

    button_func_set(banner_but, [block](bContext &C) { wm_block_splash_close(&C, block); });
  }

  const int layout_margin_x = UI_SCALE_FAC * 26;
  ui::Layout &layout = ui::block_layout(block,
                                        ui::LayoutDirection::Vertical,
                                        ui::LayoutType::Panel,
                                        layout_margin_x,
                                        0,
                                        splash_width - (layout_margin_x * 2),
                                        UI_SCALE_FAC * 110,
                                        0,
                                        style);

  MenuType *mt;

  /* Draw setup screen if no preferences have been saved yet. */
  if (!bke::preferences::exists()) {
    mt = WM_menutype_find("WM_MT_splash_quick_setup", true);

    /* The #BLOCK_QUICK_SETUP flag prevents the button text from being left-aligned,
     * as it is for all menus due to the #BLOCK_LOOP flag, see in #ui_def_but. */
    block_flag_enable(block, ui::BLOCK_QUICK_SETUP);
  }
  else {
    mt = WM_menutype_find("WM_MT_splash", true);
  }

  block_func_set(block, wm_block_splash_close_on_fileselect, block, nullptr);

  if (mt) {
    ui::menutype_draw(C, mt, &layout);
  }

/* Displays a warning if blender is being emulated via Rosetta (macOS) or XTA (Windows) */
#if defined(__APPLE__) || defined(_M_X64)
#  if defined(__APPLE__)
  if (is_using_macos_rosetta() > 0)
#  elif defined(_M_X64)
  const char *proc_id = BLI_getenv("PROCESSOR_IDENTIFIER");
  if (proc_id && strncmp(proc_id, "ARM", 3) == 0)
#  endif
  {
    layout.separator(2.0f, ui::LayoutSeparatorType::Line);

    ui::Layout &split = layout.split(0.725, true);
    ui::Layout &row1 = split.row(true);
    ui::Layout &row2 = split.row(true);

    row1.label(RPT_("Intel binary detected. Expect reduced performance."), ICON_ERROR);

    PointerRNA op_ptr = row2.op("WM_OT_url_open",
                                CTX_IFACE_(BLT_I18NCONTEXT_OPERATOR_DEFAULT, "Learn More"),
                                ICON_URL,
                                wm::OpCallContext::InvokeDefault,
                                UI_ITEM_NONE);
#  if defined(__APPLE__)
    RNA_string_set(
        &op_ptr,
        "url",
        "https://docs.blender.org/manual/en/latest/getting_started/installing/macos.html");
#  elif defined(_M_X64)
    RNA_string_set(
        &op_ptr,
        "url",
        "https://docs.blender.org/manual/en/latest/getting_started/installing/windows.html");
#  endif

    layout.separator();
  }
#endif

  block_bounds_set_centered(block, 0);

  return block;
}

static wmOperatorStatus wm_splash_invoke(bContext *C,
                                         wmOperator * /*op*/,
                                         const wmEvent * /*event*/)
{
  SplashAnimationState *animation = MEM_new<SplashAnimationState>(__func__);
  ui::popup_block_invoke(C, wm_block_splash_create, animation, wm_splash_animation_free);

  return OPERATOR_FINISHED;
}

void WM_OT_splash(wmOperatorType *ot)
{
  ot->name = "Splash Screen";
  ot->idname = "WM_OT_splash";
  ot->description = "Open the splash screen with release info";

  ot->invoke = wm_splash_invoke;
  ot->poll = WM_operator_winactive;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Splash Screen: About
 * \{ */

static ui::Block *wm_block_about_create(bContext *C, ARegion *region, void * /*arg*/)
{
  const uiStyle *style = ui::style_get_dpi();
  const int dialog_width = style->widget.points * 42 * UI_SCALE_FAC;

  ui::Block *block = block_begin(C, region, "about", ui::EmbossType::Emboss);

  block_flag_enable(block, ui::BLOCK_KEEP_OPEN | ui::BLOCK_LOOP | ui::BLOCK_NO_WIN_CLIP);
  block_theme_style_set(block, ui::BLOCK_THEME_STYLE_POPUP);

  ui::Layout &layout = ui::block_layout(block,
                                        ui::LayoutDirection::Vertical,
                                        ui::LayoutType::Panel,
                                        0,
                                        0,
                                        dialog_width,
                                        0,
                                        0,
                                        style);

/* Blender logo. */
#ifndef WITH_HEADLESS
  constexpr bool show_color = false;
  const float size = 0.2f * dialog_width;

  ImBuf *ibuf = ui::svg_icon_bitmap(ICON_BLENDER_LOGO_LARGE, size, show_color);

  if (ibuf) {
    bTheme *btheme = ui::theme::theme_get();
    const uchar *color = btheme->tui.wcol_menu_back.text_sel;

    /* The top margin. */
    layout.row(false).separator(0.2f);

    /* The logo image. */
    layout.row(false).alignment_set(ui::LayoutAlign::Left);
    uiDefButImage(block, ibuf, 0, U.widget_unit, ibuf->x, ibuf->y, show_color ? nullptr : color);

    /* Padding below the logo. */
    layout.row(false).separator(2.7f);
  }
#endif /* !WITH_HEADLESS */

  ui::Layout &col = layout.column(true);

  uiItemL_ex(&col, IFACE_("Blender"), ICON_NONE, true, false);

  MenuType *mt = WM_menutype_find("WM_MT_splash_about", true);
  if (mt) {
    ui::menutype_draw(C, mt, &col);
  }

  block_bounds_set_centered(block, 22 * UI_SCALE_FAC);

  return block;
}

static wmOperatorStatus wm_splash_about_invoke(bContext *C,
                                               wmOperator * /*op*/,
                                               const wmEvent * /*event*/)
{
  ui::popup_block_invoke(C, wm_block_about_create, nullptr, nullptr);

  return OPERATOR_FINISHED;
}

void WM_OT_splash_about(wmOperatorType *ot)
{
  ot->name = "About Blender";
  ot->idname = "WM_OT_splash_about";
  ot->description = "Open a window with information about Blender";

  ot->invoke = wm_splash_about_invoke;
  ot->poll = WM_operator_winactive;
}

/** \} */

}  // namespace blender
