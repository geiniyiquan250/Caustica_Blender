/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_CUDA

#  include "device/cuda/queue.h"

#  include "device/cuda/device_impl.h"
#  include "device/cuda/graphics_interop.h"
#  include "device/cuda/kernel.h"
#  include "util/caustics_profiler.h"

CCL_NAMESPACE_BEGIN

/* CUDADeviceQueue */

CUDADeviceQueue::CUDADeviceQueue(CUDADevice *device)
    : DeviceQueue(device), cuda_device_(device), cuda_stream_(nullptr)
{
  const CUDAContextScope scope(cuda_device_);
  cuda_device_assert(cuda_device_, cuStreamCreate(&cuda_stream_, CU_STREAM_NON_BLOCKING));
}

CUDADeviceQueue::~CUDADeviceQueue()
{
  const CUDAContextScope scope(cuda_device_);
  caustics_profile_collect();
  for (CausticsProfileEvent &event : caustics_profile_events_) {
    if (event.begin) {
      cuEventDestroy(event.begin);
    }
    if (event.end) {
      cuEventDestroy(event.end);
    }
  }
  cuStreamDestroy(cuda_stream_);
}

bool CUDADeviceQueue::caustics_profile_read(CausticsProfileEvent &event)
{
  if (!event.pending) {
    return true;
  }
  const CUresult status = cuEventQuery(event.end);
  if (status == CUDA_ERROR_NOT_READY) {
    return false;
  }
  float milliseconds = 0.0f;
  const CUresult result = status == CUDA_SUCCESS ?
                              cuEventElapsedTime(&milliseconds, event.begin, event.end) : status;
  if (result == CUDA_SUCCESS) {
    photon_profile_record("gpu", device_kernel_as_string(event.kernel), this,
                           event.submitted, milliseconds, event.work_size);
  }
  else {
    photon_profile_value("gpu.event_error", this, result);
    caustics_profile_failed_ = true;
  }
  event.pending = false;
  return true;
}

void CUDADeviceQueue::caustics_profile_begin(DeviceKernel kernel, const int work_size)
{
  if (!photon_profile_enabled() || caustics_profile_failed_) {
    return;
  }
  if (caustics_profile_events_.empty()) {
    caustics_profile_events_.resize(128);
    photon_profile_value("gpu.device_type", this, cuda_device_->info.type);
    photon_profile_value("gpu.device_index", this, cuda_device_->cuDevId);
  }
  const size_t index = caustics_profile_cursor_++ % caustics_profile_events_.size();
  CausticsProfileEvent &event = caustics_profile_events_[index];
  if (!caustics_profile_read(event)) {
    photon_profile_value("gpu.timing_dropped_pool_busy", this, 1);
    return;
  }
  if (caustics_profile_failed_) {
    return;
  }
  CUresult result = CUDA_SUCCESS;
  if (!event.begin) {
    result = cuEventCreate(&event.begin, CU_EVENT_DEFAULT);
    if (result == CUDA_SUCCESS) {
      result = cuEventCreate(&event.end, CU_EVENT_DEFAULT);
    }
  }
  if (result == CUDA_SUCCESS) {
    result = cuEventRecord(event.begin, cuda_stream_);
  }
  if (result != CUDA_SUCCESS) {
    photon_profile_value("gpu.event_error", this, result);
    caustics_profile_failed_ = true;
    return;
  }
  event.kernel = kernel;
  event.work_size = work_size;
  event.submitted = time_dt();
  caustics_profile_active_ = int(index);
}

void CUDADeviceQueue::caustics_profile_end()
{
  if (caustics_profile_active_ < 0) {
    return;
  }
  CausticsProfileEvent &event = caustics_profile_events_[caustics_profile_active_];
  const CUresult result = cuEventRecord(event.end, cuda_stream_);
  if (result == CUDA_SUCCESS) {
    event.pending = true;
  }
  else {
    photon_profile_value("gpu.event_error", this, result);
    caustics_profile_failed_ = true;
  }
  caustics_profile_active_ = -1;
}

void CUDADeviceQueue::caustics_profile_collect()
{
  for (CausticsProfileEvent &event : caustics_profile_events_) {
    caustics_profile_read(event);
  }
}

int CUDADeviceQueue::num_concurrent_states(const size_t state_size) const
{
  const int max_num_threads = cuda_device_->get_num_multiprocessors() *
                              cuda_device_->get_max_num_threads_per_multiprocessor();
  int num_states = max(max_num_threads, 65536) * 16;

  const char *factor_str = getenv("CYCLES_CONCURRENT_STATES_FACTOR");
  if (factor_str) {
    const float factor = (float)atof(factor_str);
    if (factor != 0.0f) {
      num_states = max((int)(num_states * factor), 1024);
    }
    else {
      LOG_TRACE << "CYCLES_CONCURRENT_STATES_FACTOR evaluated to 0";
    }
  }

  LOG_TRACE << "GPU queue concurrent states: " << num_states << ", using up to "
            << string_human_readable_size(num_states * state_size);

  return num_states;
}

int CUDADeviceQueue::num_concurrent_busy_states(const size_t /*state_size*/) const
{
  const int max_num_threads = cuda_device_->get_num_multiprocessors() *
                              cuda_device_->get_max_num_threads_per_multiprocessor();

  if (max_num_threads == 0) {
    return 65536;
  }

  return 4 * max_num_threads;
}

void CUDADeviceQueue::init_execution()
{
  CCL_PHOTON_PROFILE_SCOPE("gpu.context_init_wait", this);
  /* Synchronize all textures and memory copies before executing task.
   * Use default stream (nullptr) since that's what we will synchronize
   * here to ensure all scene data is copied. */
  CUDAContextScope scope(cuda_device_);
  cuda_device_->load_image_info(nullptr);
  cuda_device_assert(cuda_device_, cuCtxSynchronize());

  debug_init_execution();
}

void CUDADeviceQueue::load_image_info()
{
  CUDAContextScope scope(cuda_device_);
  cuda_device_->load_image_info(this);
}

bool CUDADeviceQueue::enqueue(DeviceKernel kernel,
                              const int work_size,
                              const DeviceKernelArguments &args)
{
  if (cuda_device_->have_error()) {
    return false;
  }

  debug_enqueue_begin(kernel, work_size);

  const CUDAContextScope scope(cuda_device_);

  /* Update image info in case integrator memory alloc caused texture to move to host. */
  if (cuda_device_->load_image_info(nullptr)) {
    cuda_device_assert(cuda_device_, cuCtxSynchronize());
    if (cuda_device_->have_error()) {
      return false;
    }
  }

  /* Compute kernel launch parameters. */
  const CUDADeviceKernel &cuda_kernel = cuda_device_->kernels.get(kernel);
  const int num_threads_per_block = cuda_kernel.num_threads_per_block;
  const int num_blocks = divide_up(work_size, num_threads_per_block);

  int shared_mem_bytes = 0;

  switch (kernel) {
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY:
    case DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY:
      /* See parall_active_index.h for why this amount of shared memory is needed. */
      shared_mem_bytes = (num_threads_per_block + 1) * sizeof(int);
      break;

    default:
      break;
  }

  /* Launch kernel. */
  caustics_profile_begin(kernel, work_size);
  assert_success(cuLaunchKernel(cuda_kernel.function,
                                num_blocks,
                                1,
                                1,
                                num_threads_per_block,
                                1,
                                1,
                                shared_mem_bytes,
                                cuda_stream_,
                                const_cast<void **>(args.values),
                                nullptr),
                 "enqueue");

  caustics_profile_end();

  debug_enqueue_end();

  return !(cuda_device_->have_error());
}

bool CUDADeviceQueue::synchronize()
{
  if (cuda_device_->have_error()) {
    return false;
  }

  const CUDAContextScope scope(cuda_device_);
  PhotonProfileScope wait("gpu.queue_wait", this);
  assert_success(cuStreamSynchronize(cuda_stream_), "synchronize");
  wait.finish();
  caustics_profile_collect();

  debug_synchronize();

  return !(cuda_device_->have_error());
}

void CUDADeviceQueue::zero_to_device(device_memory &mem)
{
  assert(mem.type != MEM_IMAGE_TEXTURE);

  if (mem.memory_size() == 0) {
    return;
  }

  /* Allocate on demand. */
  if (mem.device_pointer == 0) {
    if (mem.type == MEM_GLOBAL) {
      cuda_device_->global_alloc(mem);
    }
    else {
      cuda_device_->mem_alloc(mem);
    }
  }

  /* Zero memory on device. */
  device_ptr d_ptr = mem.device->mem_device_ptr(mem, cuda_device_);
  assert(d_ptr != 0);

  const CUDAContextScope scope(cuda_device_);
  assert_success(cuMemsetD8Async((CUdeviceptr)d_ptr, 0, mem.memory_size(), cuda_stream_),
                 "zero_to_device");
}

void CUDADeviceQueue::copy_to_device(device_memory &mem)
{
  assert(mem.type != MEM_IMAGE_TEXTURE);

  if (mem.memory_size() == 0) {
    return;
  }

  /* Allocate on demand. */
  if (mem.device_pointer == 0) {
    if (mem.type == MEM_GLOBAL) {
      cuda_device_->global_alloc(mem);
    }
    else {
      cuda_device_->mem_alloc(mem);
    }
  }

  device_ptr d_ptr = mem.device->mem_device_ptr(mem, cuda_device_);
  assert(d_ptr != 0);
  assert(mem.host_pointer != nullptr);

  /* Copy memory to device. */
  const CUDAContextScope scope(cuda_device_);
  assert_success(
      cuMemcpyHtoDAsync((CUdeviceptr)d_ptr, mem.host_pointer, mem.memory_size(), cuda_stream_),
      "copy_to_device");
}

void CUDADeviceQueue::copy_from_device(device_memory &mem)
{
  assert(mem.type != MEM_GLOBAL && mem.type != MEM_IMAGE_TEXTURE);

  if (mem.memory_size() == 0) {
    return;
  }

  assert(mem.device_pointer != 0);
  assert(mem.host_pointer != nullptr);

  /* Copy memory from device. */
  const CUDAContextScope scope(cuda_device_);
  assert_success(
      cuMemcpyDtoHAsync(
          mem.host_pointer, (CUdeviceptr)mem.device_pointer, mem.memory_size(), cuda_stream_),
      "copy_from_device");
}

void *CUDADeviceQueue::copy_from_device_synchronized(device_memory &mem, vector<uint8_t> &storage)
{
  if (mem.memory_size() == 0) {
    return nullptr;
  }

  storage.resize(mem.memory_size());

  device_ptr d_ptr = mem.device->mem_device_ptr(mem, cuda_device_);
  assert(d_ptr != 0);

  const CUDAContextScope scope(cuda_device_);
  assert_success(
      cuMemcpyDtoHAsync(storage.data(), (CUdeviceptr)d_ptr, mem.memory_size(), cuda_stream_),
      "copy_from_device_synchronized");

  synchronize();
  return storage.data();
}

void CUDADeviceQueue::assert_success(CUresult result, const char *operation)
{
  if (result != CUDA_SUCCESS) {
    const char *name = cuewErrorString(result);
    cuda_device_->set_error(string_printf(
        "%s in CUDA queue %s (%s)", name, operation, debug_active_kernels().c_str()));
  }
}

unique_ptr<DeviceGraphicsInterop> CUDADeviceQueue::graphics_interop_create()
{
  return make_unique<CUDADeviceGraphicsInterop>(this);
}

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
