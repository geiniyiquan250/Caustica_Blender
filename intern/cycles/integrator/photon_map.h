/* SPDX-FileCopyrightText: 2026 CyclesPlus
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* === CyclesPlus: Personal Photon Map Header Begin === */
/* Photon caustics: host-side progressive photon map.
 *
 * Stochastic progressive photon mapping (SPPM): every render progression gets
 * an independent photon batch traced with a fresh seed at a constant gather
 * radius (= grid cell size). The per-pixel SPPM statistics in the film passes
 * consume each batch exactly once (kernel/film/photon_passes.h) and shrink
 * their own radii, so the caustics sharpen and de-noise in place while the
 * render resolves. */

#pragma once

#include <functional>

#include "device/memory.h"

#include "kernel/integrator/photon_grid.h"

#include "util/string.h"
#include "util/unique_ptr.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

class Scene;
class Progress;
struct PhotonMapData;

/* Material classification report ("fail loudly"): one entry per scene
 * material whose fast-mode photon profile is only approximate or silently
 * caustic-less. Refilled on every scene extraction; the addon UI polls it
 * via _cycles.photon_material_report() when drawing the caustics panel.
 * Materials that classify exactly are not listed. */
struct PhotonMaterialReport {
  string name;
  /* 1 = approximated, 2 = casts no caustics despite a caustic-like setup. */
  int level;
  string reason;
};

/* Thread-safe snapshot of the report from the latest extraction. */
vector<PhotonMaterialReport> photon_material_report_snapshot();

class PhotonMap {
 public:
  PhotonMap();
  ~PhotonMap();

  /* Re-extracts the scene, resets the progressive schedule and synchronously
   * traces the first photon batch, so caustics are visible from sample one.
   * Also starts tracing the second batch asynchronously.
   *
   * `interactive` shrinks the synchronous first batch to a small preview:
   * unbiased like every batch (flux is normalized per batch), just grainier,
   * so viewport edits unfreeze quickly and the caustics refine as the full
   * async batches arrive. Final renders keep the full-size first batch.
   *
   * `world_changed` invalidates the cached HDRI virtual suns; everything
   * else reuses them, since the background evaluation and median cut are
   * too expensive to redo for every object/light/material tweak. */
  void restart(Scene *scene,
               Progress *progress,
               const int photons_per_batch,
               const float detail,
               const bool interactive,
               const bool world_changed);

  /* Progressive refinement: publish the next photon generation. Only call
   * between render works - the published grid must not change while a work
   * is rendering. CPU tracing: if the async worker batch finished, publish
   * it and (if `allow_launch`) start the next one. GPU tracing: trace one
   * generation synchronously right here, but only when the previous one was
   * consumed (`generation_pending` == false). `allow_launch` == false stops
   * refinement (budget exhausted); a later call with true resumes it. */
  void advance(const bool allow_launch, const bool generation_pending);

  /* Abort any in-flight async photon generation and drain the photon
   * stream. MUST be called before a scene update swaps device buffers
   * (BVH, geometry, shader tables): a photon launch still in flight reads
   * the scene it was started against, and the swap under it is an illegal
   * address - observed 2026-08-05 as "Illegal address in CUDA queue
   * synchronize (photon_trace)" during rapid-fire material imports, with
   * the driver's OptiX context dying as collateral. */
  void abort_inflight();

  /* Process-wide "a final render is running" signal. The interactive
   * session parks its photon machinery while any background session
   * renders: the async photon stream is the only long-lived GPU actor
   * that crosses work boundaries, and coexisting with a full-resolution
   * F12 animation killed the driver (overnight repro 2026-08-09: headless
   * stable, solid-viewport stable, rendered-viewport parallel = device
   * lost in ~40s). Parking costs nothing - refinement traced during a
   * final render would be GPU time stolen from that render anyway. */
  static void background_render_begin();
  static void background_render_end();
  static bool background_render_active();

  /* Called from the tracing worker thread whenever a batch finishes, so the
   * session can wake up and consume it while the viewport is idle. Set once
   * before the first restart(). */
  void set_batch_done_callback(std::function<void()> callback);

  void clear();

  /* Null while empty/invalid. */
  const PhotonGrid *grid() const;

  /* Device-resident binning: scatter the front generation's VRAM deposits
   * into the render device's photon arrays (grid()->device_resident != 0).
   * Called by PathTrace::set_photon_grid between render works, after it
   * allocated the target arrays and uploaded the cell prefix table. */
  bool scatter_published(device_ptr out_pos,
                         device_ptr out_beam_start,
                         device_ptr out_flux,
                         device_ptr out_beam_sigma,
                         device_ptr cell_start_device);

 private:
  void launch_batch_async();

  unique_ptr<PhotonMapData> data_;
};

CCL_NAMESPACE_END
/* === CyclesPlus: Personal Photon Map Header End === */
