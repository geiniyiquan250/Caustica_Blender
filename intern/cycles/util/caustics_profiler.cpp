/* SPDX-FileCopyrightText: 2026 CyclesPlus
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* === CyclesPlus: Personal Caustics Profiler Implementation Begin === */
#include "util/caustics_profiler.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "util/path.h"
#include "util/string.h"
#include "util/system.h"

CCL_NAMESPACE_BEGIN

namespace {

/* Buffered CSV avoids console output in hot paths. The fixed capture limit bounds disk use;
 * summaries are cumulative and keyed by stage, clock domain and session/queue address. */
class CausticsProfileLog {
 public:
  CausticsProfileLog()
  {
    const char *directory = std::getenv("CYCLESPLUS_CAUSTICS_PROFILE_DIR");
    if (!directory || !directory[0]) {
      directory = std::getenv("TEMP");
    }
    if (!directory || !directory[0]) {
      directory = ".";
    }
    const string filename = string_printf("caustics_profile_%llu_%llu.csv",
                                         (unsigned long long)std::time(nullptr),
                                         (unsigned long long)system_self_process_id());
    const string path = path_join(directory, filename);
    file_ = path_fopen(path, "wb");
    if (!file_) {
      fprintf(stderr, "Caustics profiler: cannot open %s\n", path.c_str());
      return;
    }
    setvbuf(file_, buffer_, _IOFBF, sizeof(buffer_));
    fputs("# caustics_profile_v1; times=ms; host=nested wall time; gpu=stream event elapsed; "
          "GPU timestamp=host submission; GPU elapsed includes contention; value units in stage; "
          "summary=cumulative; timestamps may be out of order; limit=128MiB\n", file_);
    fputs("kind,start_ms,thread,context,stage,duration_ms,value,count,total_ms,max_ms\n", file_);
    fflush(file_);
    fprintf(stderr, "Caustics profiler: %s\n", path.c_str());
  }

  ~CausticsProfileLog()
  {
    flush();
    if (file_) {
      fclose(file_);
    }
  }

  void record(const char *kind, const char *name, const void *context,
              double start, double duration_ms, double value)
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!file_ || capped_) {
      return;
    }
    const double now = time_dt();
    const auto thread_id = (unsigned long long)std::hash<std::thread::id>{}(
        std::this_thread::get_id());
    fprintf(file_, "%s,%.3f,%llu,%p,%s,%.6f,%.9g,1,%.6f,%.6f\n",
            kind, (start - origin_) * 1000.0, thread_id, context, name,
            duration_ms, value, duration_ms, duration_ms);
    if (kind[0] != 'v') {
      const string key = string_printf("%p,%s.%s", context, kind, name);
      Stats &stat = stats_[key];
      stat.count++;
      stat.total += duration_ms;
      stat.maximum = std::max(stat.maximum, duration_ms);
    }
    if (now - last_flush_ >= 1.0) {
      flush_locked(now);
    }
  }

  void flush()
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (file_ && !capped_) {
      flush_locked(time_dt());
    }
  }

 private:
  void flush_locked(double now)
  {
    for (const auto &[key, stat] : stats_) {
      fprintf(file_, "summary,%.3f,0,%s,summary,%.6f,0,%llu,%.6f,%.6f\n",
              (now - origin_) * 1000.0, key.c_str(), stat.total / stat.count,
              stat.count, stat.total, stat.maximum);
    }
    last_flush_ = now;
    const long position = ftell(file_);
    if (position >= 128 * 1024 * 1024 || position < 0 || ferror(file_)) {
      fputs("# capture stopped: size limit or file error\n", file_);
      capped_ = true;
      fprintf(stderr, "Caustics profiler: capture stopped (size limit or file error).\n");
    }
    if (fflush(file_) != 0) {
      capped_ = true;
      fprintf(stderr, "Caustics profiler: capture stopped (flush failed).\n");
    }
  }

  struct Stats {
    unsigned long long count = 0;
    double total = 0.0;
    double maximum = 0.0;
  };
  FILE *file_ = nullptr;
  char buffer_[64 * 1024];
  std::mutex mutex_;
  std::map<string, Stats> stats_;
  double origin_ = time_dt();
  double last_flush_ = origin_;
  bool capped_ = false;
};

CausticsProfileLog &profile_log()
{
  static CausticsProfileLog log;
  return log;
}

}  // namespace

void photon_profile_record(const char *kind, const char *name, const void *context,
                           double start, double duration_ms, double value)
{
  if (photon_profile_enabled()) {
    profile_log().record(kind, name, context, start, duration_ms, value);
  }
}

void photon_profile_flush()
{
  if (photon_profile_enabled()) {
    profile_log().flush();
  }
}

CCL_NAMESPACE_END
/* === CyclesPlus: Personal Caustics Profiler Implementation End === */
