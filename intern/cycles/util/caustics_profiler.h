/* SPDX-FileCopyrightText: 2026 CyclesPlus
 *
 * SPDX-License-Identifier: Apache-2.0 */
/* === CyclesPlus: Personal Caustics Profiler Header Begin === */
#pragma once

#include <cstdlib>

#include "util/defines.h"
#include "util/time.h"

CCL_NAMESPACE_BEGIN

inline bool photon_profile_enabled()
{
  static const bool enabled = []() {
    const char *value = std::getenv("CYCLESPLUS_CAUSTICS_PROFILE");
    return value && value[0] == '1';
  }();
  return enabled;
}

/* Host durations include waiting. GPU durations are event elapsed times on one stream and can
 * include contention with other streams. Nested durations must not be added together. */
void photon_profile_record(const char *kind,
                           const char *name,
                           const void *context,
                           double start,
                           double duration_ms,
                           double value = 0.0);
void photon_profile_flush();

inline void photon_profile_value(const char *name, const void *context, double value)
{
  if (photon_profile_enabled()) {
    photon_profile_record("value", name, context, time_dt(), 0.0, value);
  }
}

class PhotonProfileScope {
 public:
  explicit PhotonProfileScope(const char *name, const void *context = nullptr, double value = 0.0)
      : name_(name), context_(context), value_(value),
        start_(photon_profile_enabled() ? time_dt() : 0.0)
  {
  }
  ~PhotonProfileScope()
  {
    finish();
  }
  void finish()
  {
    if (start_ == 0.0) {
      return;
    }
    photon_profile_record("host", name_, context_, start_, (time_dt() - start_) * 1000.0, value_);
    start_ = 0.0;
  }

  PhotonProfileScope(const PhotonProfileScope &) = delete;
  PhotonProfileScope &operator=(const PhotonProfileScope &) = delete;

 private:
  const char *name_;
  const void *context_;
  double value_;
  double start_;
};

CCL_NAMESPACE_END

#define CCL_PHOTON_PROFILE_JOIN_IMPL(a, b) a##b
#define CCL_PHOTON_PROFILE_JOIN(a, b) CCL_PHOTON_PROFILE_JOIN_IMPL(a, b)
#define CCL_PHOTON_PROFILE_SCOPE(...) \
  PhotonProfileScope CCL_PHOTON_PROFILE_JOIN(photon_profile_scope_, __LINE__)(__VA_ARGS__)
/* === CyclesPlus: Personal Caustics Profiler Header End === */
