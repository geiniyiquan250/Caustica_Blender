/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <cstring>

#include "device/cpu/device.h"
#include "device/device.h"
#include "integrator/path_trace.h"
/* === CyclesPlus: Photon Caustics Includes Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
#include <cstdlib>
#include "integrator/photon_map.h"
#endif
/* === CyclesPlus: Photon Caustics Includes End === */
#include "util/caustics_profiler.h"
#include "scene/background.h"
#include "scene/camera.h"
#include "scene/image.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "session/buffers.h"
#include "session/display_driver.h"
#include "session/output_driver.h"
#include "session/session.h"

#include "util/log.h"
#include "util/math.h"
#include "util/task.h"
#include "util/time.h"

CCL_NAMESPACE_BEGIN

/* Photon caustics: extra viewport samples scheduled after the configured
 * sample count is reached, while the caustic estimate is still immature.
 * Real samples are what resolve the preview: every work gathers at freshly
 * jittered measurement points (the per-gather tau averaging), which smooths
 * the sparse photon discs the way continued sampling would, and consumes
 * newly traced photon generations at fresh points on the way.
 *
 * The first PHOTON_EXTRA_SAMPLES_SMOOTH run back to back right after the
 * last regular sample; beyond that, one more sample is scheduled per newly
 * published photon generation (the batch-done callback wakes the loop) up
 * to PHOTON_EXTRA_SAMPLES_MAX. Any scene change resets the budget. */
static constexpr int PHOTON_EXTRA_SAMPLES_SMOOTH = 16;
static constexpr int PHOTON_EXTRA_SAMPLES_MAX = PHOTON_EXTRA_SAMPLES_SMOOTH + 32;

Session::Session(const SessionParams &params_, const SceneParams &scene_params)
    : params(params_),
      eviction_manager_(params_.background),
      render_scheduler_(tile_manager_, params)
{
  TaskScheduler::init(params.threads);

  delayed_reset_.do_reset = false;

  pause_ = false;
  new_work_added_ = false;

  device = Device::create(params.device, stats, profiler, params_.headless);

  if (device->have_error()) {
    progress.set_error(device->error_message());
  }

  scene = make_unique<Scene>(scene_params, device.get());

  if (params.device == params.denoise_device) {
    /* Reuse render device. */
  }
  else {
    denoise_device_ = Device::create(params.denoise_device, stats, profiler, params_.headless);

    if (denoise_device_->have_error()) {
      progress.set_error(denoise_device_->error_message());
    }
  }

  /* Configure path tracer. */
  path_trace_ = make_unique<PathTrace>(device.get(),
                                       denoise_device(),
                                       scene->film,
                                       &scene->dscene,
                                       render_scheduler_,
                                       tile_manager_);
  path_trace_->set_progress(&progress);
  path_trace_->progress_update_cb = [&]() { update_status_time(); };

  tile_manager_.full_buffer_written_cb = [&](string_view filename) {
    if (!full_buffer_written_cb) {
      return;
    }
    full_buffer_written_cb(filename);
  };

  /* Create session thread. */
  session_thread_ = make_unique<thread>([this] { thread_run(); });
}

Session::~Session()
{
  /* Cancel any ongoing render operation. */
  cancel();

  /* Signal session thread to end. */
  {
    const thread_scoped_lock session_thread_lock(session_thread_mutex_);
    session_thread_state_ = SESSION_THREAD_END;
  }
  session_thread_cond_.notify_all();

  /* Destroy session thread. */
  session_thread_->join();
  session_thread_.reset();

  /* === CyclesPlus: Photon Map Destruction Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  /* Destroy the photon map before the device: it owns device buffers and
   * a queue for GPU photon tracing whose destruction frees device memory. */
  photon_map_.reset();
#endif
  /* === CyclesPlus: Photon Map Destruction End === */

  /* Destroy path tracer, before the device. This is needed because destruction might need to
   * access device for device memory free.
   * TODO(sergey): Convert device to be unique_ptr, and rely on C++ to destruct objects in the
   * pre-defined order. */
  path_trace_.reset();

  /* Destroy scene and device. */
  scene.reset();
  denoise_device_.reset();
  device.reset();
  /* === CyclesPlus: Photon Profile Flush Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  photon_profile_flush();
#endif
  /* === CyclesPlus: Photon Profile Flush End === */

  /* Stop task scheduler. */
  TaskScheduler::exit();
}

void Session::start()
{
  {
    /* Signal session thread to start rendering. */
    const thread_scoped_lock session_thread_lock(session_thread_mutex_);
    if (session_thread_state_ == SESSION_THREAD_RENDER) {
      /* Already rendering, nothing to do. */
      return;
    }
    session_thread_state_ = SESSION_THREAD_RENDER;
  }

  session_thread_cond_.notify_all();
}

void Session::cancel(bool quick)
{
  /* Cancel any long running device operations (e.g. shader compilations). */
  device->cancel();

  /* Check if session thread is rendering. */
  const bool rendering = is_session_thread_rendering();

  if (rendering) {
    /* Cancel path trace operations. */
    if (quick && path_trace_) {
      path_trace_->cancel();
    }

    /* Cancel other operations. */
    progress.set_cancel("Exiting");

    /* Signal unpause in case the render was paused. */
    {
      const thread_scoped_lock pause_lock(pause_mutex_);
      pause_ = false;
    }
    pause_cond_.notify_all();

    /* Wait for render thread to be cancelled or finished. */
    wait();
  }
}

bool Session::ready_to_reset()
{
  return path_trace_->ready_to_reset();
}

void Session::run_main_render_loop()
{
  /* === CyclesPlus: Photon Parking Signal Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  /* Signal interactive sessions to park their photon machinery for the
   * whole final render (see PhotonMap::background_render_begin). Scoped
   * to the loop so persistent-data sessions release it between renders. */
  const bool signal_background = params.background;
  if (signal_background) {
    PhotonMap::background_render_begin();
  }
#endif
  /* === CyclesPlus: Photon Parking Signal End === */

  path_trace_->zero_display();

  while (true) {
    RenderWork render_work = run_update_for_next_iteration();

    const bool did_cancel = progress.get_cancel();

    if (!render_work) {
      if (LOG_IS_ON(LOG_LEVEL_INFO)) {
        if (did_cancel) {
          LOG_INFO << "Rendering was canceled.";
        }
        else {
          double total_time;
          double render_time;
          progress.get_time(total_time, render_time);
          LOG_INFO << "Rendering in main loop is done in " << render_time << " seconds.";
          LOG_INFO << path_trace_->full_report();
        }
      }

      if (params.background) {
        /* if no work left and in background mode, we can stop immediately. */
        progress.set_status("Finished");
        break;
      }
    }

    if (did_cancel) {
      render_scheduler_.render_work_reschedule_on_cancel(render_work);
      if (!render_work) {
        break;
      }
    }
    else if (run_wait_for_work(render_work)) {
      continue;
    }

    /* Stop rendering if error happened during scene update or other step of preparing scene
     * for render. */
    if (device->have_error()) {
      progress.set_error(device->error_message());
      break;
    }

    {
      /* buffers mutex is locked entirely while rendering each
       * sample, and released/reacquired on each iteration to allow
       * reset and draw in between */
      const thread_scoped_lock buffers_lock(buffers_mutex_);

      /* update status and timing */
      update_status_time();

      /* render */
      path_trace_->render(render_work);

      /* update status and timing */
      update_status_time();

      /* Stop rendering if error happened during path tracing. */
      if (device->have_error()) {
        progress.set_error(device->error_message());
        break;
      }
    }

    progress.set_update();

    if (did_cancel) {
      break;
    }
  }

  /* === CyclesPlus: Photon Parking Release Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  if (params.background) {
    PhotonMap::background_render_end();
  }
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Photon Parking Release End === */
}

void Session::thread_run()
{
  while (true) {
    {
      thread_scoped_lock session_thread_lock(session_thread_mutex_);

      if (session_thread_state_ == SESSION_THREAD_WAIT) {
        /* Continue waiting for any signal from the main thread. */
        session_thread_cond_.wait(session_thread_lock);
        continue;
      }
      if (session_thread_state_ == SESSION_THREAD_END) {
        /* End thread immediately. */
        break;
      }
    }

    /* Execute a render. */
    thread_render();

    /* Go back from rendering to waiting. */
    {
      const thread_scoped_lock session_thread_lock(session_thread_mutex_);
      if (session_thread_state_ == SESSION_THREAD_RENDER) {
        session_thread_state_ = SESSION_THREAD_WAIT;
      }
    }
    session_thread_cond_.notify_all();
  }

  /* Flush any remaining operations and destroy display driver here. This ensure
   * graphics API resources are created and destroyed all in the session thread,
   * which can avoid problems contexts and multiple threads. */
  path_trace_->flush_display();
  path_trace_->set_display_driver(nullptr);
}

void Session::thread_render()
{
  if (params.use_profiling && (params.device.type == DEVICE_CPU)) {
    profiler.start();
  }

  /* session thread loop */
  progress.set_status("Waiting for render to start");

  /* run */
  if (!progress.get_cancel()) {
    /* reset number of rendered samples */
    progress.reset_sample();

    run_main_render_loop();
  }

  profiler.stop();

  /* progress update */
  if (progress.get_cancel()) {
    progress.set_status(progress.get_cancel_message());
  }
  else {
    progress.set_update();
  }

  /* === CyclesPlus: Render End Profile Flush Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  /* Persist partial captures before the session returns to waiting, including cancellation. */
  if (photon_profile_enabled()) {
    photon_profile_value("session.render_end_cancelled", this, progress.get_cancel());
    photon_profile_value("session.render_end_error", this, progress.get_error());
    photon_profile_flush();
  }
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Render End Profile Flush End === */
}

bool Session::is_session_thread_rendering()
{
  const thread_scoped_lock session_thread_lock(session_thread_mutex_);
  return (session_thread_state_ == SESSION_THREAD_RENDER);
}

RenderWork Session::run_update_for_next_iteration()
{
  /* === CyclesPlus: Photon Session Profiling Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  CCL_PHOTON_PROFILE_SCOPE("session.update_iteration", this);
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Photon Session Profiling End === */
  RenderWork render_work;

  /* === CyclesPlus: Photon Scene Lock Profiling Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  PhotonProfileScope scene_lock_wait("session.scene_lock_wait", this);
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  thread_scoped_lock scene_lock(scene->mutex);
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  scene_lock_wait.finish();
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Photon Scene Lock Profiling End === */

  /* === CyclesPlus: Photon Scene Update Drain Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  /* Photon caustics (CyclesPlus): only scene-data updates need to drain the
   * async photon stream. Camera, viewport-size, and local-view buffer resets
   * do not replace BVH/geometry/shader tables and must not stall navigation. */
  const bool scene_update_pending = scene->need_reset(false);
  if (photon_map_ && scene_update_pending) {
    photon_map_->abort_inflight();
  }
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Photon Scene Update Drain End === */

  /* Perform delayed reset if requested. */
  const bool reset_buffers = delayed_reset_buffer_params();

  /* Update scene */
  const bool reset_scene = update_scene(delayed_reset_.do_reset);
  /* === CyclesPlus: Photon Profile Recording Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  if (photon_profile_enabled()) {
    photon_profile_value("session.reset_buffers", this, reset_buffers);
    photon_profile_value("session.reset_scene", this, reset_scene);
    photon_profile_value("session.background", this, params.background);
    photon_profile_value("session.caustics_enabled", this,
                         scene->integrator->get_use_photon_caustics());
  }
#endif
  /* === CyclesPlus: Photon Profile Recording End === */

  /* Photon caustics: scene changes reset the extra-sample budget before the
   * sample target below is computed from it. */
  /* === CyclesPlus: Photon Extra Samples Reset Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  if (reset_scene) {
    photon_extra_samples_ = 0;
  }
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Photon Extra Samples Reset End === */

  /* Update buffers for new parameters. After scene update which influences the passes used. */
  bool have_tiles = true;
  bool switched_to_new_tile = false;

  if (reset_buffers) {
    update_buffers_for_params();

    /* After reset make sure the tile manager is at the first big tile. */
    have_tiles = tile_manager_.next();
    switched_to_new_tile = true;

    eviction_manager_.reset();
  }

  /* Update denoiser settings. */
  {
    DenoiseParams denoise_params = scene->integrator->get_denoise_params();
    /* === CyclesPlus: Photon OIDN GPU Routing Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
    /* CyclesPlus: OIDN's GPU backend brings a second CUDA context onto the
     * card; next to the OptiX photon raygen the repeated context traffic
     * hard-hangs the device after minutes of animation (bisected 2026-07-21:
     * OptiX+kernel+OIDN-GPU = 7x crash; denoise off, OIDN-CPU and the OptiX
     * denoiser each ran the same 18-frame gauntlet clean; CUDA devices are
     * unaffected; compute-sanitizer clean). Redirect OIDN to the CPU in
     * exactly that combination - identical image, the denoise step just
     * takes a second or two longer. Silent by design: the customer gets
     * one-click caustics, the routing is engine business. */
    /* Final renders: the hangs needed minutes of sustained animation load,
     * and CPU OIDN only costs a second or two per frame there. */
    if (params.background && denoise_params.use &&
        denoise_params.type == DENOISER_OPENIMAGEDENOISE && denoise_params.use_gpu &&
        device->info.type == DEVICE_OPTIX && scene->integrator->get_use_photon_caustics())
    {
      LOG_INFO << "CyclesPlus: OIDN denoising redirected to CPU "
                  "(GPU OIDN is unstable next to the OptiX photon kernel)";
      denoise_params.use_gpu = false;
    }
    /* Viewport: the same crash class arrived here once the ClangCL build
     * made the viewport truly fluid (sustained load like an animation).
     * First response was routing OIDN-GPU to the OptiX denoiser - but the
     * two denoise visibly differently (OptiX leaves camera-dependent
     * smooth/grainy patches on immature caustics; user spotted it within
     * one session), so OIDN-GPU stays. The real fix landed at the source:
     * the driver deaths were the in-flight photon stream racing scene
     * updates, killed by abort_inflight() before update_scene (plus the
     * shader-id hardening). The launch throttle built in the meantime is
     * opt-in only (CYCLESPLUS_PHOTON_OPTIX_THROTTLE=1 - full speed is the
     * product), and this redirect remains the second emergency hatch. */
    else if (!params.background && denoise_params.use &&
             denoise_params.type == DENOISER_OPENIMAGEDENOISE && denoise_params.use_gpu &&
             device->info.type == DEVICE_OPTIX && scene->integrator->get_use_photon_caustics())
    {
      static const bool viewport_safe = getenv("CYCLESPLUS_PHOTON_VIEWPORT_SAFE") != nullptr;
      if (viewport_safe) {
        LOG_INFO << "CyclesPlus: viewport OIDN-GPU denoising redirected to the OptiX "
                    "denoiser (CYCLESPLUS_PHOTON_VIEWPORT_SAFE)";
        denoise_params.type = DENOISER_OPTIX;
      }
    }
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
    /* === CyclesPlus: Photon OIDN GPU Routing End === */
    path_trace_->set_denoiser_params(denoise_params);
  }

  /* Update adaptive sampling. */
  {
    const AdaptiveSampling adaptive_sampling = scene->integrator->get_adaptive_sampling();
    path_trace_->set_adaptive_sampling(adaptive_sampling);
  }

  /* Update path guiding. */
  {
    const GuidingParams guiding_params = scene->integrator->get_guiding_params(device.get());
    const bool guiding_reset = (guiding_params.use) ? reset_scene : false;
    path_trace_->set_guiding_params(guiding_params, guiding_reset);
  }

  /* === CyclesPlus: Photon Sample Target Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  /* Photon caustics: the sample target includes the extra samples scheduled
   * for caustic refinement after the configured count (see below). */
  render_scheduler_.set_sample_params(params.samples + photon_extra_samples_,
                                      params.use_sample_subset,
                                      params.sample_subset_offset,
                                      params.sample_subset_length);
#else
  render_scheduler_.set_sample_params(
      params.samples, params.use_sample_subset, params.sample_subset_offset, params.sample_subset_length);
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Photon Sample Target End === */
  render_scheduler_.set_time_limit(params.time_limit);

  while (have_tiles) {
    render_work = render_scheduler_.get_render_work();
    if (render_work) {
      break;
    }

    progress.add_finished_tile(false);

    have_tiles = tile_manager_.next();
    if (have_tiles) {
      render_scheduler_.reset_for_next_tile();
      switched_to_new_tile = true;
    }
  }

  /* === CyclesPlus: Photon Caustics Update Logic Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  /* Update photon caustics map (CyclesPlus). Progressive: on scene changes
   * the schedule restarts (objects moved -> photons retrace automatically);
   * between progressions finished refinement batches are swapped in, so the
   * caustics sharpen while the render resolves. Once the configured sample
   * count is reached but the caustic is still immature, a bounded number of
   * extra samples keeps the refinement going (the batch-done callback wakes
   * the main loop when new generations arrive). */
  {
    const bool use_photon_caustics = scene->integrator->get_use_photon_caustics();
    if (use_photon_caustics) {
      const bool first_build = !photon_map_;
      if (first_build) {
        photon_map_ = make_unique<PhotonMap>();
        photon_map_->set_batch_done_callback([this]() {
          const thread_scoped_lock pause_lock(pause_mutex_);
          new_work_added_ = true;
          pause_cond_.notify_all();
        });
      }
      /* PARK while a final render runs (interactive sessions only): the
       * async photon stream is the only long-lived GPU actor crossing work
       * boundaries, and coexisting with a full-resolution F12 animation
       * killed the driver (2026-08-09 overnight repro: headless stable,
       * solid-viewport stable, rendered-viewport parallel = device lost in
       * ~40s - same black-screen class as the 08-05 hunt). abort_inflight
       * runs on OUR thread against OUR map, so there is no cross-session
       * call; the residual window is one session iteration (~ms). On
       * unpark the map restarts from scratch: costs one ~1s viewport ramp
       * and guarantees no wedged stream state survives the render (the
       * user-reported "viewport laggy after animation" heals the same
       * way a Blender restart did). Skipped scene resets stay recorded in
       * photon_world_changed_ / the forced restart. */
      const bool photon_parked = !params.background && PhotonMap::background_render_active();
      if (photon_parked) {
        photon_map_->abort_inflight();
        photon_parked_ = true;
      }
      else if (reset_scene || first_build || photon_parked_) {
        const int count = max(scene->integrator->get_photon_caustics_count(), 1) * 1000000;
        const float detail = scene->integrator->get_photon_caustics_detail();
        /* Interactive sessions get a small synchronous preview batch (fast
         * viewport feedback, full batches refine asynchronously); final
         * renders keep the full-size first batch. */
        photon_map_->restart(
            scene.get(), &progress, count, detail, !params.background, photon_world_changed_);
        photon_world_changed_ = false;
        photon_parked_ = false;
      }
      else {
        /* Keep the tracing pipeline running while render works arrive; once
         * the extra-sample budget is exhausted, stop it (a later scene
         * change restarts everything). GPU tracing paces itself on the
         * pending flag: a new generation is only traced after the previous
         * one was consumed by a gather. */
        const bool allow_launch = static_cast<bool>(render_work) ||
                                  photon_extra_samples_ < PHOTON_EXTRA_SAMPLES_MAX;
        photon_profile_value("session.navigation_with_tail", this, photon_navigating_);
        photon_profile_value("session.photon_launch_allowed", this, allow_launch);
        photon_map_->advance(allow_launch, path_trace_->photon_gather_pending());
      }
    }
    else if (photon_map_) {
      photon_map_->clear();
      photon_extra_samples_ = 0;
    }
    path_trace_->set_photon_grid(photon_map_ ? photon_map_->grid() : nullptr);

    /* Caustic refinement samples: the configured sample count is rendered,
     * but the photon estimate is still young. Schedule single extra samples
     * - real ones. Every sample re-gathers at a freshly jittered measurement
     * point (per-gather tau averaging), which is what visually resolves the
     * sparse preview dots into a smooth caustic; pending photon generations
     * get consumed at fresh points along the way. The first batch of extras
     * runs back to back, after that only a new photon generation justifies
     * another sample. */
    if (use_photon_caustics && !render_work && !params.background && photon_map_->grid() &&
        !progress.get_cancel())
    {
      const bool smoothing = photon_extra_samples_ < PHOTON_EXTRA_SAMPLES_SMOOTH;
      const bool new_generation = path_trace_->photon_gather_pending() &&
                                  photon_extra_samples_ < PHOTON_EXTRA_SAMPLES_MAX;
      if (smoothing || new_generation) {
        photon_extra_samples_++;
        photon_profile_value("session.extra_samples", this, photon_extra_samples_);
        render_scheduler_.set_sample_params(params.samples + photon_extra_samples_,
                                            params.use_sample_subset,
                                            params.sample_subset_offset,
                                            params.sample_subset_length);
        render_work = render_scheduler_.get_render_work();
        LOG_INFO << "CyclesPlus photon map: extra refinement sample "
                 << photon_extra_samples_ << "/" << PHOTON_EXTRA_SAMPLES_MAX;
      }
    }
  }
#endif
  /* === CyclesPlus: Photon Caustics Update Logic End === */

  /* Evict unused image tiles periodically. */
  if (eviction_manager_.need_eviction(!render_work, switched_to_new_tile)) {
    scene->image_manager->evict_unused(device.get(), scene.get());
  }

  if (render_work) {
    const scoped_timer update_timer;

    if (switched_to_new_tile) {
      BufferParams tile_params = buffer_params_;

      const Tile &tile = tile_manager_.get_current_tile();

      tile_params.width = tile.width;
      tile_params.height = tile.height;

      tile_params.window_x = tile.window_x;
      tile_params.window_y = tile.window_y;
      tile_params.window_width = tile.window_width;
      tile_params.window_height = tile.window_height;

      tile_params.full_x = tile.x + buffer_params_.full_x;
      tile_params.full_y = tile.y + buffer_params_.full_y;
      tile_params.full_width = buffer_params_.full_width;
      tile_params.full_height = buffer_params_.full_height;

      tile_params.update_offset_stride();

      path_trace_->reset(buffer_params_, tile_params, reset_buffers);
    }

    /* Update camera if dimensions changed for progressive render. the camera
     * knows nothing about progressive or cropped rendering, it just gets the
     * image dimensions passed in. */
    const float resolution = render_work.resolution_divider;
    const int width = max(1, int(buffer_params_.full_width / resolution));
    const int height = max(1, int(buffer_params_.full_height / resolution));

    scene->update_camera_resolution(progress, width, height);

    /* Unlock scene mutex before loading denoiser kernels, since that may attempt to activate
     * graphics interop, which can deadlock when the scene mutex is still being held. */
    scene_lock.unlock();

    path_trace_->load_kernels();
    path_trace_->alloc_work_memory();

    /* Wait for device to be ready (e.g. finish any background compilations). */
    string device_status;
    while (!device->is_ready(device_status)) {
      progress.set_status(device_status);
      if (progress.get_cancel()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    progress.add_skip_time(update_timer, params.background);
  }

  return render_work;
}

bool Session::run_wait_for_work(const RenderWork &render_work)
{
  /* In an offline rendering there is no pause, and no tiles will mean the job is fully done. */
  if (params.background) {
    return false;
  }

  thread_scoped_lock pause_lock(pause_mutex_);

  if (!pause_ && render_work) {
    /* Rendering is not paused and there is work to be done. No need to wait for anything. */
    return false;
  }

  const bool no_work = !render_work;
  update_status_time(pause_, no_work);

  /* Only leave the loop when rendering is not paused. But even if the current render is
   * un-paused but there is nothing to render keep waiting until new work is added. */
  while (!progress.get_cancel()) {
    const scoped_timer pause_timer;

    if (!pause_ && (render_work || new_work_added_ || delayed_reset_.do_reset)) {
      break;
    }

    const std::chrono::milliseconds wait_time = eviction_manager_.wait_time(!render_work);
    if (wait_time == std::chrono::milliseconds::zero()) {
      /* Break out of the loop for cache eviction. */
      break;
    }

    /* Wait for either pause state changed, extra samples added to render, or idle
     * timer before performing eviction. */
    if (wait_time == std::chrono::milliseconds::max()) {
      pause_cond_.wait(pause_lock);
    }
    else {
      pause_cond_.wait_for(pause_lock, wait_time);
    }

    if (pause_) {
      progress.add_skip_time(pause_timer, params.background);
    }

    update_status_time(pause_, no_work);
    progress.set_update();
  }

  new_work_added_ = false;

  return no_work;
}

void Session::draw()
{
  path_trace_->draw();
}

int2 Session::get_effective_tile_size() const
{
  const int image_width = buffer_params_.width;
  const int image_height = buffer_params_.height;

  if (!params.use_auto_tile) {
    return make_int2(image_width, image_height);
  }

  const int64_t image_area = static_cast<int64_t>(image_width) * image_height;

  /* TODO(sergey): Take available memory into account, and if there is enough memory do not
   * tile and prefer optimal performance. */

  const int tile_size = tile_manager_.compute_render_tile_size(params.tile_size);
  const int64_t actual_tile_area = static_cast<int64_t>(tile_size) * tile_size;

  if (actual_tile_area >= image_area && image_width <= TileManager::MAX_TILE_SIZE &&
      image_height <= TileManager::MAX_TILE_SIZE)
  {
    return make_int2(image_width, image_height);
  }

  return make_int2(tile_size, tile_size);
}

bool Session::delayed_reset_buffer_params()
{
  /* === CyclesPlus: Session Buffer Reset Profiling Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  CCL_PHOTON_PROFILE_SCOPE("session.buffer_reset", this);
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Session Buffer Reset Profiling End === */
  /* Reset buffer parameters, delayed from when we got the reset call so we can complete
   * rendering the sample. Otherwise e.g. viewport navigation might reset without ever
   * finishing anything. */
  const thread_scoped_lock reset_lock(delayed_reset_.mutex);
  if (!delayed_reset_.do_reset) {
    return false;
  }

  const thread_scoped_lock buffers_lock(buffers_mutex_);
  delayed_reset_.do_reset = false;

  params = delayed_reset_.session_params;
  buffer_params_ = delayed_reset_.buffer_params;

  /* Store parameters used for buffers access outside of scene graph. */
  buffer_params_.samples = min(params.samples, Integrator::MAX_SAMPLES);
  buffer_params_.exposure = scene->film->get_exposure();
  buffer_params_.use_approximate_shadow_catcher =
      scene->film->get_use_approximate_shadow_catcher();
  buffer_params_.use_transparent_background = scene->background->get_transparent();

  /* Tile and work scheduling. */
  tile_manager_.reset_scheduling(buffer_params_, get_effective_tile_size());

  return true;
}

void Session::update_buffers_for_params()
{
  render_scheduler_.set_sample_params(params.samples,
                                      params.use_sample_subset,
                                      params.sample_subset_offset,
                                      params.sample_subset_length);
  render_scheduler_.reset(buffer_params_);

  /* Update for new state of scene and passes. */
  buffer_params_.update_passes(scene->passes);
  tile_manager_.update(buffer_params_, scene.get());

  /* Update temp directory on reset.
   * This potentially allows to finish the existing rendering with a previously configure
   * temporary
   * directory in the host software and switch to a new temp directory when new render starts. */
  tile_manager_.set_temp_dir(params.temp_dir);

  /* Progress. */
  progress.reset_sample();
  progress.set_total_pixel_samples(static_cast<uint64_t>(buffer_params_.width) *
                                   buffer_params_.height * buffer_params_.samples);

  if (!params.background) {
    progress.set_start_time();
  }
  const double time_limit = params.time_limit * ((double)tile_manager_.get_num_tiles());
  progress.set_render_start_time();
  progress.set_time_limit(time_limit);
}

void Session::reset(const SessionParams &session_params, const BufferParams &buffer_params)
{
  /* === CyclesPlus: Session Reset Profiling Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  CCL_PHOTON_PROFILE_SCOPE("session.reset_request_and_cancel", this);
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Session Reset Profiling End === */
  {
    const thread_scoped_lock reset_lock(delayed_reset_.mutex);
    const thread_scoped_lock pause_lock(pause_mutex_);

    delayed_reset_.do_reset = true;
    delayed_reset_.session_params = session_params;
    delayed_reset_.buffer_params = buffer_params;

    scene->scene_updated_while_loading_kernels = true;

    path_trace_->cancel();
  }

  pause_cond_.notify_all();
}

void Session::set_samples(const int samples)
{
  if (samples == params.samples) {
    return;
  }

  params.samples = samples;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);
    new_work_added_ = true;
  }

  pause_cond_.notify_all();
}

void Session::set_time_limit(const double time_limit)
{
  if (time_limit == params.time_limit) {
    return;
  }

  params.time_limit = time_limit;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);
    new_work_added_ = true;
  }

  pause_cond_.notify_all();
}

void Session::set_pause(bool pause)
{
  bool notify = false;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);

    if (pause != pause_) {
      pause_ = pause;
      notify = true;
    }
  }

  if (is_session_thread_rendering()) {
    if (notify) {
      pause_cond_.notify_all();
    }
  }
  else if (pause_) {
    update_status_time(pause_);
  }
}

void Session::set_navigating(bool navigating)
{
  /* === CyclesPlus: Photon Navigation State Begin === */
  eviction_manager_.set_navigating(navigating);
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  if (photon_profile_enabled() && photon_navigating_ != navigating) {
    photon_profile_value("session.navigation", this, navigating);
  }
  /* Keep navigation state available for profiling while photon work remains
   * enabled during viewport interaction. */
  photon_navigating_ = navigating;
  path_trace_->set_photon_navigating(navigating);
#endif
  /* === CyclesPlus: Photon Navigation State End === */
}

void Session::set_output_driver(unique_ptr<OutputDriver> driver)
{
  path_trace_->set_output_driver(std::move(driver));
}

void Session::set_display_driver(unique_ptr<DisplayDriver> driver)
{
  path_trace_->set_display_driver(std::move(driver));
}

double Session::get_estimated_remaining_time() const
{
  const double completed = progress.get_progress();
  if (completed == 0.0) {
    return 0.0;
  }

  double total_time;
  double render_time;
  progress.get_time(total_time, render_time);
  double remaining = (1.0 - (double)completed) * (render_time / (double)completed);

  const double time_limit = render_scheduler_.get_time_limit() *
                            ((double)tile_manager_.get_num_tiles());
  if (time_limit != 0.0) {
    remaining = min(remaining, max(time_limit - render_time, 0.0));
  }

  return remaining;
}

void Session::wait()
{
  /* Wait until session thread either is waiting or ending. */
  while (true) {
    thread_scoped_lock session_thread_lock(session_thread_mutex_);
    if (session_thread_state_ != SESSION_THREAD_RENDER) {
      break;
    }
    session_thread_cond_.wait(session_thread_lock);
  }
}

bool Session::update_scene(const bool reset_samples)
{
  /* === CyclesPlus: Session Scene Update Profiling Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  CCL_PHOTON_PROFILE_SCOPE("session.scene_update", this);
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Session Scene Update Profiling End === */
  /* Update number of samples in the integrator.
   * Ideally this would need to happen once in `Session::set_samples()`, but the issue there is
   * the initial configuration when Session is created where the `set_samples()` is not used.
   *
   * NOTE: Unless reset was requested only allow increasing number of samples. */
  if (reset_samples || scene->integrator->get_aa_samples() < params.samples) {
    scene->integrator->set_aa_samples(params.samples);
  }

  scene->integrator->set_use_sample_subset(params.use_sample_subset);
  scene->integrator->set_sample_subset_offset(params.sample_subset_offset);
  scene->integrator->set_sample_subset_length(params.sample_subset_length);

  /* When multiple tiles are used SAMPLE_COUNT pass is used to keep track of possible partial
   * tile results. */
  scene->film->set_use_sample_count(tile_manager_.has_multiple_tiles());

  const bool reset = scene->need_reset(false);

  /* === CyclesPlus: Photon World Change Tracking Begin === */
#ifdef WITH_CYCLES_SPPM_CAUSTICS
  /* Photon caustics (CyclesPlus): remember world edits before scene->update
   * clears the modified flags - the photon map caches the expensive HDRI
   * virtual sun decomposition until the world actually changes. */
  {
    Shader *bg_shader = scene->background->get_shader(scene.get());
    if (scene->background->is_modified() || (bg_shader && bg_shader->is_modified())) {
      photon_world_changed_ = true;
    }
  }
#endif  /* WITH_CYCLES_SPPM_CAUSTICS */
  /* === CyclesPlus: Photon World Change Tracking End === */

  if (scene->update(progress)) {
    profiler.reset(scene->shaders.size(), scene->objects.size());
  }

  return reset;
}

static string status_append(const string &status, const string &suffix)
{
  string prefix = status;
  if (!prefix.empty()) {
    prefix += ", ";
  }
  return prefix + suffix;
}

void Session::update_status_time(bool show_pause, bool show_done)
{
  string status;
  string substatus;

  const int current_tile = progress.get_rendered_tiles();
  const int num_tiles = tile_manager_.get_num_tiles();

  const int current_sample = progress.get_current_sample();
  const int num_samples = render_scheduler_.get_num_samples();

  /* TIle. */
  if (tile_manager_.has_multiple_tiles()) {
    substatus = status_append(substatus,
                              string_printf("Rendered %d/%d Tiles", current_tile, num_tiles));
  }

  /* Sample. */
  if (!params.background && num_samples == Integrator::MAX_SAMPLES) {
    substatus = status_append(substatus, string_printf("Sample %d", current_sample));
  }
  else {
    substatus = status_append(substatus,
                              string_printf("Sample %d/%d", current_sample, num_samples));
  }

  /* Append any device-specific status (such as background kernel optimization) */
  string device_status;
  if (device->is_ready(device_status) && !device_status.empty()) {
    substatus += string_printf(" (%s)", device_status.c_str());
  }

  /* TODO(sergey): Denoising status from the path trace. */

  if (show_pause) {
    status = "Rendering Paused";
  }
  else if (show_done) {
    status = "Rendering Done";
    progress.set_end_time(); /* Save end time so that further calls to get_time are accurate. */
  }
  else {
    status = substatus;
    substatus.clear();
  }

  progress.set_status(status, substatus);
}

void Session::device_free()
{
  scene->device_free();
  path_trace_->device_free();
}

void Session::collect_statistics(RenderStats *render_stats)
{
  scene->collect_statistics(render_stats);
  if (params.use_profiling && (params.device.type == DEVICE_CPU)) {
    render_stats->collect_profiling(scene.get(), profiler);
  }
}

/* --------------------------------------------------------------------
 * Full-frame on-disk storage.
 */

void Session::process_full_buffer_from_disk(string_view filename)
{
  path_trace_->process_full_buffer_from_disk(filename);
}

CCL_NAMESPACE_END
