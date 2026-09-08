/* SPDX-FileCopyrightText: 2026 CyclesPlus
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Photon caustics: host-side photon tracing.
 *
 * Ported from the FastCaustics native core (caustics_core.cpp): own SAH BVH
 * over the world-space triangles, specular photon walk (reflect/refract with
 * Fresnel), deposits only on diffuse receivers reached via at least one
 * specular bounce. Materials are classified from the Cycles shader graphs, so
 * unmodified scenes work without any material changes. */

#include "integrator/photon_map.h"
#include "util/caustics_profiler.h"
#include "integrator/shader_eval.h"

#include "device/device.h"
#include "device/memory.h"
#include "device/queue.h"

#include "kernel/integrator/photon_trace_types.h"
#include "kernel/types.h"

#include "scene/attribute.h"
#include "scene/background.h"
#include "util/colorspace.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/camera.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"

#include "util/image.h"
#include "util/log.h"
#include "util/progress.h"
#include "util/string.h"
#include "util/time.h"
#include "util/transform.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <thread>

CCL_NAMESPACE_BEGIN

/* Material report registry ("fail loudly"). Written at the end of every
 * scene extraction, read by the addon UI through the _cycles Python module -
 * possibly from another thread, hence the lock.
 *
 * Deliberately leaked (construct on first use, never destroyed): a static
 * ccl::vector's atexit destructor runs through the GuardedAllocator whose
 * global stats object can already be gone at that point (undefined static
 * destruction order across translation units) - crashed reproducibly at
 * process exit whenever the report was non-empty. */
static std::mutex photon_report_mutex;

static vector<PhotonMaterialReport> &photon_report_entries()
{
  static vector<PhotonMaterialReport> *entries = new vector<PhotonMaterialReport>();
  return *entries;
}

vector<PhotonMaterialReport> photon_material_report_snapshot()
{
  const std::lock_guard<std::mutex> lock(photon_report_mutex);
  return photon_report_entries();
}

namespace {

/* ------------------------------------------------------------------ rng */

struct PhotonRNG {
  uint64_t state, inc;
  void seed(const uint64_t s, const uint64_t seq)
  {
    state = 0;
    inc = (seq << 1u) | 1u;
    next();
    state += s;
    next();
  }
  uint32_t next()
  {
    const uint64_t old = state;
    state = old * 6364136223846793005ULL + inc;
    const uint32_t xs = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    const uint32_t rot = (uint32_t)(old >> 59u);
    return (xs >> rot) | (xs << ((-(int)rot) & 31));
  }
  float uniform()
  {
    return (next() >> 8) * (1.0f / 16777216.0f);
  }
  float range(const float a, const float b)
  {
    return a + (b - a) * uniform();
  }
  float3 unit()
  {
    for (;;) {
      const float3 v = make_float3(range(-1, 1), range(-1, 1), range(-1, 1));
      const float l2 = dot(v, v);
      if (l2 > 1e-8f && l2 <= 1.0f) {
        return v * (1.0f / sqrtf(l2));
      }
    }
  }
};

/* ------------------------------------------------------------------ scene */

enum MaterialKind {
  MAT_RECEIVER = 0,
  MAT_GLASS = 1,
  MAT_METAL = 2,
  MAT_SKIP = 3,
  MAT_VOLUME = 4,
};

struct PhotonMaterial {
  int kind = MAT_RECEIVER;
  float ior = 1.45f;
  float rough = 0.0f;
  /* Receivers with partial transmission (translucent shells like inflatable
   * toys): photons pass straight through with this probability, tinted by
   * `color`, instead of depositing. Matches how path tracing backlights
   * such materials. */
  float transmission = 0.0f;
  /* Clearcoat layer on any base (car paint: rough metal base + smooth
   * coat). Photons reflect specularly off the coat with the Fresnel share
   * before interacting with the base - the coat, not the rough base, casts
   * the sharp caustics of car paint. */
  float coat = 0.0f;
  float coat_rough = 0.0f;
  float coat_ior = 1.5f;
  float dispersion_inv_abbe = 0.0f;
  float3 color = make_float3(1.0f, 1.0f, 1.0f);
  /* Accurate mode (Weg B): the classifier could not capture this material
   * exactly - the KERNEL photon walk runs the real surface shader at its
   * hits. The host tracer (CPU fallback) ignores this and keeps the
   * approximate profile. */
  int accurate = 0;
  /* Alpha-derived transparency (Principled Alpha < 1): photons pass straight
   * through with this probability BEFORE any surface interaction, untinted -
   * Cycles mixes a WHITE Transparent BSDF for alpha. Not a caustic-forming
   * event. Without it an invisible pane is a photon wall: a caster behind
   * alpha-0 glass lost its caustic (user find 2026-08-01). */
  float transparent = 0.0f;
  /* Who owns this material's caustic light (per-material gate):
   *  1 = the photon map (a pass-through arms later deposits, spec++),
   *  0 = demoted - path tracing keeps this material's caustics, so a
   *      pass-through must NOT arm a deposit (it would double-count),
   *  2 = demoted GLASS: pass photons with the Fresnel-transmitted share so
   *      unflagged window panes stay windows for photons aimed at casters
   *      behind them (the glass table behind thin-wall windows). */
  int pass_mode = 1;
  /* Beer-Lambert absorption of the interior medium (Volume Absorption
   * node); zero = clear. */
  float3 volume_sigma = make_float3(0.0f, 0.0f, 0.0f);
  float3 volume_sigma_s = make_float3(0.0f, 0.0f, 0.0f);
  float volume_g = 0.0f;
};

struct PhotonTraceScene {
  vector<float3> verts;
  vector<float3> vnormals;
  vector<uint32_t> tri_v; /* 3 per tri */
  vector<int32_t> tri_mat;
  vector<uint8_t> tri_smooth;
  vector<PhotonMaterial> mats;
  /* Light linking: receiver set of the object each triangle belongs to.
   * Empty when the scene uses no light linking (the common case). */
  vector<uint32_t> tri_recv_set;

  /* Per kernel shader slot: does the photon map cast caustics from this
   * shader? The path tracer only hands a caustic contribution over for
   * shaders marked here - anything else it keeps, because we would have
   * nothing to put in its place. A glossy plastic table throws a specular
   * bounce in the kernel while the classifier files it as a receiver; the
   * old scene-wide suppression could not tell those apart, and that is the
   * mistake that blacked out interiors. */
  vector<uint8_t> shader_caster;
  bool volume_caustics = false;

  struct Node {
    float3 bmin, bmax;
    int32_t left; /* internal: left child (right stashed in start); leaf: -1 */
    int32_t start, count;
  };
  vector<Node> nodes;
  vector<uint32_t> order;
  vector<float3> centroids;
};

struct PhotonLight {
  int type; /* 0 sun, 1 point, 2 spot, 3 area */
  float3 color;
  float3 pos;
  float3 axis; /* emission direction (-Z of the light) */
  float p0;    /* sun: half angle | point/spot: radius | area: unused */
  float spot_cos;
  float sx, sy;
  int shape; /* 0 rect, 1 ellipse */
  /* Light linking. The kernel walk tests the emitter's object index against
   * the receiving object (same rule as path tracing); the host tracer has no
   * objects, so it carries the emitter's set membership and tests it against
   * the receiving triangle's receiver set. */
  int emitter_object = -1; /* OBJECT_NONE */
  uint64_t link_membership = ~uint64_t(0);
  /* Blender light group index (scene->lightgroups), -1 = none. Every deposit
   * carries it, so the gather can split the caustic into Caustics_<group>
   * passes the same way Cycles splits Combined. */
  int lightgroup = -1;
  /* IES profile slot in Cycles' own table, -1 = none. A lamp driven by an
   * IES texture emits into a measured distribution rather than uniformly,
   * and photons are emitted host-side, so the shape has to travel with the
   * light instead of falling out of the shader (tester report 2026-08-25:
   * IES lamps lit as if the profile were not there). */
  int ies_slot = -1;
  float3 initial_volume_sigma = make_float3(0.0f);
  float3 initial_volume_scatter = make_float3(0.0f);
  int initial_volume_object = OBJECT_NONE;
  int initial_volume_material = -1;
};

struct PhotonVolumeRegion {
  float3 bmin, bmax;
  float3 sigma, scatter;
  int object, material;
};

/* Light group index for a photon source, or -1 for none. Reads the very map
 * Cycles fills its own Combined_<group> passes from, so a caustic ends up in
 * the group the artist assigned in the light's normal panel - the feature
 * needs no UI of its own. Filled during extraction: Scene::device_update
 * refreshes scene->lightgroups before any of this runs. */
static int photon_lightgroup_index(const Scene *scene, const ustring &name)
{
  if (name.empty()) {
    return -1;
  }
  const auto it = scene->lightgroups.find(name);
  return (it != scene->lightgroups.end()) ? it->second : -1;
}

struct PhotonTarget {
  float3 c;
  float r, wcum;
  /* Photon launch distance for sun/world emission, from the scene bounding
   * sphere (set in a post-pass after extraction; 0 = fallback heuristic). */
  float start_dist = 0.0f;
};

struct PhotonDeposit {
  float3 pos, flux;
  float3 beam_start;
  float3 normal; /* surface normal, or volume direction toward the light */
  /* Extinction of the medium carrying a volume beam. Surface deposits leave
   * this at zero. */
  float3 volume_sigma = make_float3(0.0f, 0.0f, 0.0f);
  /* Light group of the emitting light, -1 = none. Packed into flux.w when the
   * deposits are flattened, the same slot the kernel tracer uses. */
  int lightgroup = -1;
  bool volume = false;
};

void tri_bounds(const PhotonTraceScene &s, const uint32_t t, float3 &bmin, float3 &bmax)
{
  const float3 a = s.verts[s.tri_v[3 * t]];
  const float3 b = s.verts[s.tri_v[3 * t + 1]];
  const float3 c = s.verts[s.tri_v[3 * t + 2]];
  bmin = min(a, min(b, c));
  bmax = max(a, max(b, c));
}

int32_t build_node(PhotonTraceScene &s, uint32_t *idx, const int64_t n)
{
  PhotonTraceScene::Node node;
  node.bmin = make_float3(1e30f, 1e30f, 1e30f);
  node.bmax = make_float3(-1e30f, -1e30f, -1e30f);
  float3 cmin = node.bmin, cmax = node.bmax;
  for (int64_t i = 0; i < n; i++) {
    float3 bmin, bmax;
    tri_bounds(s, idx[i], bmin, bmax);
    node.bmin = min(node.bmin, bmin);
    node.bmax = max(node.bmax, bmax);
    cmin = min(cmin, s.centroids[idx[i]]);
    cmax = max(cmax, s.centroids[idx[i]]);
  }
  const int32_t me = (int32_t)s.nodes.size();
  s.nodes.push_back(node);

  if (n <= 4) {
    s.nodes[me].left = -1;
    s.nodes[me].start = (int32_t)s.order.size();
    s.nodes[me].count = (int32_t)n;
    for (int64_t i = 0; i < n; i++) {
      s.order.push_back(idx[i]);
    }
    return me;
  }

  const float3 cext = cmax - cmin;
  int axis = 0;
  if (cext.y > cext.x) {
    axis = 1;
  }
  if (cext.z > (axis ? cext.y : cext.x)) {
    axis = 2;
  }
  const float ext = axis == 0 ? cext.x : axis == 1 ? cext.y : cext.z;

  int64_t ln = -1;
  if (ext >= 1e-12f) {
    const int NB = 16;
    int64_t cnt[NB] = {};
    float3 bmn[NB], bmx[NB];
    for (int b = 0; b < NB; b++) {
      bmn[b] = make_float3(1e30f, 1e30f, 1e30f);
      bmx[b] = make_float3(-1e30f, -1e30f, -1e30f);
    }
    const float base = axis == 0 ? cmin.x : axis == 1 ? cmin.y : cmin.z;
    const float scale = NB / ext;
    auto bin_of = [&](const uint32_t t) {
      const float3 c = s.centroids[t];
      const float v = axis == 0 ? c.x : axis == 1 ? c.y : c.z;
      return std::min(std::max((int)((v - base) * scale), 0), NB - 1);
    };
    for (int64_t i = 0; i < n; i++) {
      const int b = bin_of(idx[i]);
      cnt[b]++;
      float3 tmin, tmax;
      tri_bounds(s, idx[i], tmin, tmax);
      bmn[b] = min(bmn[b], tmin);
      bmx[b] = max(bmx[b], tmax);
    }
    auto area = [](const float3 mn, const float3 mx) {
      const float3 d = max(mx - mn, make_float3(0.0f, 0.0f, 0.0f));
      return d.x * d.y + d.y * d.z + d.z * d.x;
    };
    float best = 1e30f;
    int best_b = -1;
    for (int split = 1; split < NB; split++) {
      float3 lmn = make_float3(1e30f, 1e30f, 1e30f), lmx = -lmn, rmn = lmn, rmx = -lmn;
      int64_t nl = 0, nr = 0;
      for (int b = 0; b < split; b++) {
        if (cnt[b]) {
          lmn = min(lmn, bmn[b]);
          lmx = max(lmx, bmx[b]);
          nl += cnt[b];
        }
      }
      for (int b = split; b < NB; b++) {
        if (cnt[b]) {
          rmn = min(rmn, bmn[b]);
          rmx = max(rmx, bmx[b]);
          nr += cnt[b];
        }
      }
      if (!nl || !nr) {
        continue;
      }
      const float c = area(lmn, lmx) * nl + area(rmn, rmx) * nr;
      if (c < best) {
        best = c;
        best_b = split;
      }
    }
    if (best_b >= 0) {
      uint32_t *mid = std::partition(
          idx, idx + n, [&](const uint32_t t) { return bin_of(t) < best_b; });
      ln = mid - idx;
      if (ln == 0 || ln == n) {
        ln = n / 2;
      }
    }
  }
  if (ln < 0) {
    ln = n / 2;
  }

  const int32_t l = build_node(s, idx, ln);
  const int32_t r = build_node(s, idx + ln, n - ln);
  s.nodes[me].left = l;
  s.nodes[me].start = r;
  s.nodes[me].count = 0;
  return me;
}

struct PhotonHit {
  float t, u, v;
  uint32_t tri;
};

bool ray_box(
    const float3 o, const float3 rd_inv, const float tmax, const float3 bmin, const float3 bmax)
{
  float t0 = (bmin.x - o.x) * rd_inv.x, t1 = (bmax.x - o.x) * rd_inv.x;
  float tn = std::min(t0, t1), tf = std::max(t0, t1);
  t0 = (bmin.y - o.y) * rd_inv.y;
  t1 = (bmax.y - o.y) * rd_inv.y;
  tn = std::max(tn, std::min(t0, t1));
  tf = std::min(tf, std::max(t0, t1));
  t0 = (bmin.z - o.z) * rd_inv.z;
  t1 = (bmax.z - o.z) * rd_inv.z;
  tn = std::max(tn, std::min(t0, t1));
  tf = std::min(tf, std::max(t0, t1));
  return tf >= std::max(tn, 0.0f) && tn <= tmax;
}

bool scene_intersect(const PhotonTraceScene &s, const float3 o, const float3 d, PhotonHit &hit)
{
  const float3 rd_inv = make_float3(1.0f / (d.x == 0 ? 1e-20f : d.x),
                                    1.0f / (d.y == 0 ? 1e-20f : d.y),
                                    1.0f / (d.z == 0 ? 1e-20f : d.z));
  float tbest = 1e30f;
  bool found = false;
  int32_t stack[64];
  int sp = 0;
  stack[sp++] = 0;
  while (sp) {
    const PhotonTraceScene::Node &n = s.nodes[stack[--sp]];
    if (!ray_box(o, rd_inv, tbest, n.bmin, n.bmax)) {
      continue;
    }
    if (n.left < 0) {
      for (int32_t i = 0; i < n.count; i++) {
        const uint32_t t = s.order[n.start + i];
        const float3 v0 = s.verts[s.tri_v[3 * t]];
        const float3 e1 = s.verts[s.tri_v[3 * t + 1]] - v0;
        const float3 e2 = s.verts[s.tri_v[3 * t + 2]] - v0;
        const float3 p = cross(d, e2);
        const float det = dot(e1, p);
        if (fabsf(det) < 1e-12f) {
          continue;
        }
        const float inv = 1.0f / det;
        const float3 tv = o - v0;
        const float u = dot(tv, p) * inv;
        if (u < -1e-6f || u > 1.0f + 1e-6f) {
          continue;
        }
        const float3 q = cross(tv, e1);
        const float v = dot(d, q) * inv;
        if (v < -1e-6f || u + v > 1.0f + 1e-6f) {
          continue;
        }
        const float tt = dot(e2, q) * inv;
        if (tt > 1e-5f && tt < tbest) {
          tbest = tt;
          hit = {tt, u, v, t};
          found = true;
        }
      }
    }
    else {
      stack[sp++] = n.left;
      stack[sp++] = n.start;
    }
  }
  return found;
}

/* ------------------------------------------------------------------ tracing */

float3 reflect_dir(const float3 d, const float3 n)
{
  return d - n * (2.0f * dot(d, n));
}

/* Sample a GGX half-vector via the visible normal distribution (Heitz 2018)
 * and reflect `d` about it. VNDF matches Cycles' microfacet sampling: at
 * grazing incidence the lobe widens strongly, which is what spreads sphere
 * reflections across the floor instead of concentrating them near the
 * mirror direction. Roughness semantics like Cycles: alpha = rough^2. */
float3 ggx_scatter_reflect(PhotonRNG &rng, const float3 d, const float3 ns, const float rough)
{
  const float alpha = std::max(rough * rough, 1e-4f);

  /* Local frame with ns as +Z; view vector points away from the surface. */
  const float3 up = fabsf(ns.z) < 0.99f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
  const float3 t1 = normalize(cross(up, ns));
  const float3 t2 = cross(ns, t1);
  const float3 wi = -d;
  const float3 v_local = make_float3(dot(wi, t1), dot(wi, t2), dot(wi, ns));

  for (int attempt = 0; attempt < 4; attempt++) {
    /* Stretch view vector into the hemisphere configuration. */
    const float3 vh = normalize(make_float3(alpha * v_local.x, alpha * v_local.y, v_local.z));
    /* Orthonormal basis around vh. */
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const float3 T1 = lensq > 1e-9f ?
                          make_float3(-vh.y, vh.x, 0.0f) * (1.0f / sqrtf(lensq)) :
                          make_float3(1.0f, 0.0f, 0.0f);
    const float3 T2 = cross(vh, T1);
    /* Sample the visible hemisphere. */
    const float u1 = rng.uniform();
    const float u2 = rng.uniform();
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PI_F * u2;
    const float p1 = r * cosf(phi);
    float p2 = r * sinf(phi);
    const float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * sqrtf(std::max(0.0f, 1.0f - p1 * p1)) + s * p2;
    const float pz = sqrtf(std::max(0.0f, 1.0f - p1 * p1 - p2 * p2));
    const float3 nh = T1 * p1 + T2 * p2 + vh * pz;
    /* Unstretch. */
    const float3 h_local = normalize(
        make_float3(alpha * nh.x, alpha * nh.y, std::max(0.0f, nh.z)));
    const float3 h = t1 * h_local.x + t2 * h_local.y + ns * h_local.z;
    const float3 refl = reflect_dir(d, h);
    if (dot(refl, ns) > 0.0f) {
      return refl;
    }
  }
  return reflect_dir(d, ns);
}

bool refract_dir(const float3 d, const float3 n, const float eta, float3 &out)
{
  const float ci = -dot(d, n);
  const float s2 = eta * eta * (1.0f - ci * ci);
  if (s2 > 1.0f) {
    return false;
  }
  out = d * eta + n * (eta * ci - sqrtf(1.0f - s2));
  return true;
}

/* Host-tracer counterpart of the kernel's light-link test: the emitter
 * carries its set membership, the receiving triangle its object's receiver
 * set. Empty table (no light linking in the scene) matches everything. */
bool photon_link_match(const PhotonTraceScene &s,
                       const uint32_t tri,
                       const uint64_t link_membership)
{
  if (s.tri_recv_set.empty() || tri >= s.tri_recv_set.size()) {
    return true;
  }
  return ((uint64_t(1) << (uint64_t)s.tri_recv_set[tri]) & link_membership) != 0;
}

float fresnel_schlick(const float ci, const float n1, const float n2)
{
  float r0 = (n1 - n2) / (n1 + n2);
  r0 *= r0;
  const float m = 1.0f - fabsf(ci);
  const float m2 = m * m;
  return r0 + (1.0f - r0) * m2 * m2 * m;
}

void trace_photon(const PhotonTraceScene &s,
                  PhotonRNG &rng,
                  float3 o,
                  float3 d,
                  float3 power,
                  const int max_bounces,
                  vector<PhotonDeposit> &out,
                  const uint64_t link_membership = ~uint64_t(0),
                  const int lightgroup = -1,
                  const float beam_probability = 1.0f,
                  const float3 initial_volume_sigma = make_float3(0.0f),
                  const float3 initial_volume_scatter = make_float3(0.0f),
                  const int initial_volume_material = -1)
{
  PhotonRNG beam_rng;
  beam_rng.seed(rng.state ^ 0xBEA641ULL, rng.inc);
  const bool store_beams = beam_rng.uniform() < beam_probability;
  /* Kill photons whose power fades to a negligible FRACTION of what they
   * started with (dark tint chains). The threshold must be relative: photon
   * power scales with 1/n_total, so an absolute epsilon silently killed
   * every photon of large batches right after their first bounce (high
   * photon counts produced zero deposits). */
  const float power_cutoff = 1e-5f * std::max(power.x, std::max(power.y, power.z));
  int spec = 0;
  /* Beer-Lambert state of the medium the photon travels in (set on
   * refracting INTO a glass with volume absorption, cleared on exit).
   * vol_mat records which material put us there - see the vol_object note
   * in the kernel walk: a liquid modelled into the glass wall must not lose
   * its tint when the photon crosses that wall. */
  float3 vol_sigma = initial_volume_sigma;
  float3 vol_sigma_s = initial_volume_scatter;
  int vol_mat = initial_volume_material;
  for (int b = 0; b < max_bounces; b++) {
    /* Ray guard, mirroring the kernel walk: degenerate rays (rare RNG
     * corners in scatter normalizations) die instead of polluting
     * deposits - and instead of hanging RT cores on the GPU side. */
    const float d_len2 = dot(d, d);
    if (!std::isfinite(d_len2) || d_len2 < 0.25f || d_len2 > 4.0f ||
        !(std::isfinite(o.x) && std::isfinite(o.y) && std::isfinite(o.z))) {
      return;
    }
    PhotonHit h;
    if (!scene_intersect(s, o + d * 1e-4f, d, h)) {
      return;
    }
    const uint32_t t = h.tri;
    const float3 loc = o + d * (1e-4f + h.t);
    /* Seed a medium when a specular photon starts inside a closed volume.
     * Do this before the common segment record so inside and outside lights
     * use exactly the same beam weighting and attenuation path. */
    if (s.volume_caustics && spec > 0 && vol_sigma_s.x <= 0.0f && vol_sigma_s.y <= 0.0f &&
        vol_sigma_s.z <= 0.0f)
    {
      const PhotonMaterial &boundary = s.mats[s.tri_mat[t]];
      const float3 bv0 = s.verts[s.tri_v[3 * t]];
      const float3 bv1 = s.verts[s.tri_v[3 * t + 1]];
      const float3 bv2 = s.verts[s.tri_v[3 * t + 2]];
      const float3 bnormal = normalize(cross(bv1 - bv0, bv2 - bv0));
      if (boundary.kind == MAT_VOLUME && dot(d, bnormal) >= 0.0f) {
        vol_sigma = boundary.volume_sigma;
        vol_sigma_s = boundary.volume_sigma_s;
        vol_mat = (int)s.tri_mat[t];
      }
    }
    if (s.volume_caustics && (vol_sigma_s.x > 0.0f || vol_sigma_s.y > 0.0f ||
                              vol_sigma_s.z > 0.0f) &&
        store_beams && spec > 0) {
      /* A volume photon is a complete segment with incident power. The
       * camera-side beam estimator evaluates scattering and attenuation. */
      out.push_back({loc,
                     power / beam_probability,
                     o + d * 1e-4f,
                     -d,
                     vol_sigma,
                     lightgroup,
                     true});
    }
    if (vol_sigma.x > 0.0f || vol_sigma.y > 0.0f || vol_sigma.z > 0.0f) {
      const float seg = 1e-4f + h.t;
      power = power * make_float3(expf(-vol_sigma.x * seg),
                                  expf(-vol_sigma.y * seg),
                                  expf(-vol_sigma.z * seg));
    }
    const PhotonMaterial &m = s.mats[s.tri_mat[t]];
    const float3 v0 = s.verts[s.tri_v[3 * t]];
    const float3 v1 = s.verts[s.tri_v[3 * t + 1]];
    const float3 v2 = s.verts[s.tri_v[3 * t + 2]];
    const float3 ng = normalize(cross(v1 - v0, v2 - v0));
    float3 n = ng;
    if (s.tri_smooth[t]) {
      const float3 n0 = s.vnormals[s.tri_v[3 * t]];
      const float3 n1 = s.vnormals[s.tri_v[3 * t + 1]];
      const float3 n2 = s.vnormals[s.tri_v[3 * t + 2]];
      n = normalize(n0 * (1.0f - h.u - h.v) + n1 * h.u + n2 * h.v);
      if (dot(n, ng) < 0) {
        n = -n;
      }
    }
    const bool entering = dot(d, n) < 0.0f;
    const float3 ns = entering ? n : -n;

    /* Alpha transparency: pass through untouched (white Transparent BSDF),
     * NOT a caustic-forming event - no spec++, no tint. */
    if (m.transparent > 0.0f && rng.uniform() < m.transparent) {
      o = loc;
      continue;
    }

    /* Clearcoat: the smooth coat reflects its Fresnel share specularly
     * BEFORE the base layer interacts - on car paint (rough metal base +
     * coat) it is the coat that casts the sharp caustics. Untinted, like
     * Cycles' coat layer. */
    if (m.coat > 0.0f) {
      const float ci_coat = -dot(d, ns);
      const float fr_coat = std::min(m.coat, 1.0f) *
                            fresnel_schlick(ci_coat, 1.0f, m.coat_ior);
      if (rng.uniform() < fr_coat) {
        d = (m.coat_rough > 0.001f) ? ggx_scatter_reflect(rng, d, ns, m.coat_rough) :
                                      reflect_dir(d, ns);
        spec++;
        o = loc;
        continue;
      }
    }

    if (m.kind == MAT_VOLUME) {
      if (!s.volume_caustics) {
        o = loc;
        continue;
      }
      if (entering) {
        vol_sigma = m.volume_sigma;
        vol_sigma_s = m.volume_sigma_s;
        vol_mat = (int)s.tri_mat[t];
      }
      else if ((int)s.tri_mat[t] == vol_mat) {
        vol_sigma = make_float3(0.0f, 0.0f, 0.0f);
        vol_sigma_s = make_float3(0.0f, 0.0f, 0.0f);
        vol_mat = -1;
      }
      o = loc;
      continue;
    }

    if (m.kind == MAT_RECEIVER) {
      /* Translucent shell: pass through with the transmission probability
       * (thin-wall approximation: entry and exit refraction cancel), tinted
       * per event like Cycles' transmission. Rough transmission scatters the
       * direction - keeping it collimated concentrates the energy on the
       * shell's own shadow projection (measured +7.5% hot band vs the
       * converged path-traced reference on the flamingo).
       * pass_mode 2 = demoted glass: only the Fresnel-transmitted share
       * passes (real panes reflect). pass_mode != 1 = the pass-through does
       * not arm a deposit - path tracing owns that material's caustics. */
      float pass_p = m.transmission;
      if (m.pass_mode == 2) {
        pass_p *= 1.0f - fresnel_schlick(-dot(d, ns), 1.0f, m.ior);
      }
      if (pass_p > 0.0f && rng.uniform() < pass_p) {
        power = power * m.color;
        if (m.rough > 0.001f) {
          const float3 j = rng.unit() * (m.rough * 2.0f);
          d = normalize(d + j);
        }
        if (m.pass_mode == 1) {
          spec++;
        }
        if (std::max(power.x, std::max(power.y, power.z)) < power_cutoff) {
          return;
        }
        o = loc;
        continue;
      }
      if (spec > 0 && photon_link_match(s, t, link_membership)) {
        const float3 n_dep = (dot(d, ng) < 0.0f) ? ng : -ng;
                  out.push_back({loc,
                                 power,
                                 loc,
                                 n_dep,
                                 make_float3(0.0f),
                                 lightgroup,
                                 false});
      }
      return;
    }
    if (m.kind == MAT_METAL) {
      if (m.rough > 0.001f) {
        d = ggx_scatter_reflect(rng, d, ns, m.rough);
      }
      else {
        d = reflect_dir(d, ns);
      }
      power = power * m.color;
    }
    else {
      const float eta1 = entering ? 1.0f : m.ior;
      const float eta2 = entering ? m.ior : 1.0f;
      const float ci = -dot(d, ns);
      const float fr = fresnel_schlick(ci, eta1, eta2);
      float3 refr;
      if (!refract_dir(d, ns, eta1 / eta2, refr) || rng.uniform() < fr) {
        d = (m.rough > 0.001f) ? ggx_scatter_reflect(rng, d, ns, m.rough) : reflect_dir(d, ns);
      }
      else {
        /* Tint per refraction event, matching Cycles' transmission (Glass
         * tints with its raw color per event, Principled stores sqrt(base
         * color) in m.color at classification). Cycles' BSDFs also omit the
         * eta^2 radiance rescale at refractions, so photon power follows
         * that convention: scale by (eta_in/eta_out)^2 per event - cancels
         * for solid glass, factor 1/1.77 into standing water. The old
         * "raw color matched PT at the waterline" lesson was two canceling
         * errors: missing eta^2 times raw-instead-of-sqrt tint happens to
         * cancel exactly at the flamingo water color (0.318: 0.564*1.769
         * = 0.998). */
        d = refr;
        const float eta_scale = (eta1 / eta2) * (eta1 / eta2);
        power = power * (eta_scale * make_float3(std::max(m.color.x, 0.0f),
                                                 std::max(m.color.y, 0.0f),
                                                 std::max(m.color.z, 0.0f)));
        if (entering) {
          if (m.volume_sigma.x > 0.0f || m.volume_sigma.y > 0.0f || m.volume_sigma.z > 0.0f) {
            vol_sigma = m.volume_sigma;
            vol_sigma_s = m.volume_sigma_s;
            vol_mat = (int)s.tri_mat[t];
          }
        }
        else if ((int)s.tri_mat[t] == vol_mat) {
          vol_sigma = make_float3(0.0f, 0.0f, 0.0f);
          vol_sigma_s = make_float3(0.0f, 0.0f, 0.0f);
          vol_mat = -1;
        }
        if (m.rough > 0.001f) {
          /* Rough refraction: keep the previous mild scatter. */
          const float3 j = rng.unit() * (m.rough * m.rough * 2.0f);
          d = normalize(d + j);
        }
      }
    }
    spec++;
    if (std::max(power.x, std::max(power.y, power.z)) < power_cutoff) {
      return;
    }
    o = loc;
  }
}

const PhotonTarget *pick_target(const vector<PhotonTarget> &tg, PhotonRNG &rng, float &p_pick)
{
  const float r = rng.uniform();
  for (size_t i = 0; i < tg.size(); i++) {
    if (r <= tg[i].wcum || i == tg.size() - 1) {
      const float prev = i ? tg[i - 1].wcum : 0.0f;
      p_pick = std::max(tg[i].wcum - prev, 1e-9f);
      return &tg[i];
    }
  }
  p_pick = 1.0f;
  return &tg.back();
}

/* Same balance weight as the device tracer; keep single-target paths unchanged. */
float photon_target_balance_weight(const vector<PhotonTarget> &targets,
                                   const PhotonTarget *selected,
                                   const float selected_pdf,
                                   const float3 origin,
                                   const float3 direction,
                                   const bool directional)
{
  if (targets.size() == 1) {
    return 1.0f;
  }
  float mixture_pdf = selected_pdf;
  float previous = 0.0f;
  for (const PhotonTarget &target : targets) {
    const float probability = target.wcum - previous;
    previous = target.wcum;
    if (&target != selected) {
      mixture_pdf += probability *
                     photon_target_pdf(target.c, target.r, origin, direction, directional);
    }
  }
  return selected_pdf / mixture_pdf;
}

void emit_photons(const PhotonTraceScene &s,
                  const PhotonLight &L,
                  const vector<PhotonTarget> &tg,
                  const int64_t i0,
                  const int64_t i1,
                  const int64_t n_total,
                  const int max_bounces,
                  const uint64_t seed,
                  const uint64_t stream,
                  const std::atomic<bool> *cancel,
                  vector<PhotonDeposit> &out,
                  const float beam_probability = 1.0f)
{
  PhotonRNG rng;
  rng.seed(seed, stream);
  for (int64_t i = i0; i < i1; i++) {
    /* Scene changes abort the in-flight batch instead of blocking the
     * viewport until it finishes. */
    if ((i & 1023) == 0 && cancel && cancel->load(std::memory_order_relaxed)) {
      return;
    }
    float p_pick;
    const PhotonTarget *T = pick_target(tg, rng, p_pick);
    if (L.type == 0) { /* sun */
      float3 d;
      if (L.shape == 1) {
        /* World median-cut cell: uniform direction over the angular rect
         * (see eval_world_lights) - dissolves the square ghost patches of
         * the old centroid-cone on studio HDRIs. */
        const float phi = L.pos.x + rng.uniform() * L.pos.y;
        const float ct = L.pos.z + rng.uniform() * L.sx;
        const float st = sqrtf(std::max(0.0f, 1.0f - ct * ct));
        d = -make_float3(st * cosf(phi), st * sinf(phi), ct);
      }
      else {
        d = L.axis;
        if (L.p0 > 1e-5f) {
          d = normalize(d + rng.unit() * tanf(L.p0));
        }
      }
      const float3 u = rng.unit();
      const float3 t1 = normalize(
          cross(d, fabsf(dot(d, u)) < 0.99f ? u : make_float3(1, 0, 0)));
      const float3 t2 = cross(d, t1);
      const float rr = T->r * sqrtf(rng.uniform());
      const float ang = rng.uniform() * 2 * M_PI_F;
      /* Launch from outside the whole scene (per-target distance from the
       * scene bounding sphere) so exterior occluders block the photon
       * exactly like they block the path tracer's rays. Replaces both the
       * old ~50 m heuristic and the interim 10 km hack. */
      const float sd = (T->start_dist > 0.0f) ? T->start_dist : (T->r * 20.0f + 10.0f);
      const float3 start = T->c - d * sd + t1 * (rr * cosf(ang)) + t2 * (rr * sinf(ang));
      const float flux = M_PI_F * T->r * T->r / ((float)n_total * p_pick) *
                         photon_target_balance_weight(
                             tg, T, p_pick / (M_PI_F * T->r * T->r), start, d, true);
      trace_photon(s,
                   rng,
                   start,
                   d,
                   L.color * flux,
                   max_bounces,
                   out,
                   L.link_membership,
                   L.lightgroup,
                   beam_probability,
                   L.initial_volume_sigma,
                   L.initial_volume_scatter,
                   L.initial_volume_material);
    }
    else if (L.type == 1 || L.type == 2) { /* point / spot */
      const float inv4pi = 1.0f / (4.0f * M_PI_F);
      float3 o = L.pos;
      if (L.p0 > 1e-6f) {
        o = o + rng.unit() * (L.p0 * rng.uniform());
      }
      /* Uniform-cone sampling toward the target cap (matches the kernel;
       * the old point-in-sphere trick overestimated hits, see there). */
      float3 ca = T->c - o;
      const float dist = len(ca);
      if (dist < 1e-6f) {
        continue;
      }
      ca = ca * (1.0f / dist);
      const float st = std::min(1.0f, T->r / std::max(dist, T->r));
      const float cos_max = sqrtf(std::max(0.0f, 1.0f - st * st));
      const float ct = 1.0f - rng.uniform() * (1.0f - cos_max);
      const float sn = sqrtf(std::max(0.0f, 1.0f - ct * ct));
      const float ph = 2.0f * M_PI_F * rng.uniform();
      const float3 cup = fabsf(ca.z) < 0.99f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
      const float3 ct1 = normalize(cross(cup, ca));
      const float3 ct2 = cross(ca, ct1);
      const float3 d = ct1 * (sn * cosf(ph)) + ct2 * (sn * sinf(ph)) + ca * ct;
      if (L.type == 2 && dot(d, L.axis) < L.spot_cos) {
        continue;
      }
      const float omega = 2 * M_PI_F * (1.0f - cos_max);
      const float flux = inv4pi * omega / ((float)n_total * p_pick) *
                         photon_target_balance_weight(tg, T, p_pick / omega, o, d, false);
      trace_photon(s,
                   rng,
                   o,
                   d,
                   L.color * flux,
                   max_bounces,
                   out,
                   L.link_membership,
                   L.lightgroup,
                   beam_probability,
                   L.initial_volume_sigma,
                   L.initial_volume_scatter,
                   L.initial_volume_material);
    }
    else { /* area */
      const float area = L.sx * L.sy * (L.shape ? M_PI_F / 4.0f : 1.0f);
      const float inv_pi_area = 1.0f / (M_PI_F * std::max(area, 1e-8f));
      float lx, ly;
      if (L.shape) {
        const float rr = 0.5f * sqrtf(rng.uniform());
        const float ang = rng.uniform() * 2 * M_PI_F;
        lx = rr * cosf(ang) * L.sx;
        ly = rr * sinf(ang) * L.sy;
      }
      else {
        lx = (rng.uniform() - 0.5f) * L.sx;
        ly = (rng.uniform() - 0.5f) * L.sy;
      }
      const float3 zax = L.axis;
      const float3 up = fabsf(zax.z) < 0.99f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
      const float3 xax = normalize(cross(up, zax));
      const float3 yax = cross(zax, xax);
      const float3 o = L.pos + xax * lx + yax * ly;
      /* Uniform-cone sampling toward the target cap (matches the kernel). */
      float3 ca = T->c - o;
      const float dist = len(ca);
      if (dist < 1e-6f) {
        continue;
      }
      ca = ca * (1.0f / dist);
      const float st = std::min(1.0f, T->r / std::max(dist, T->r));
      const float cos_max = sqrtf(std::max(0.0f, 1.0f - st * st));
      const float ct = 1.0f - rng.uniform() * (1.0f - cos_max);
      const float sn = sqrtf(std::max(0.0f, 1.0f - ct * ct));
      const float ph = 2.0f * M_PI_F * rng.uniform();
      const float3 cup = fabsf(ca.z) < 0.99f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
      const float3 ct1 = normalize(cross(cup, ca));
      const float3 ct2 = cross(ca, ct1);
      const float3 d = ct1 * (sn * cosf(ph)) + ct2 * (sn * sinf(ph)) + ca * ct;
      const float cl = dot(d, zax);
      if (cl <= 0.0f) {
        continue;
      }
      const float omega = 2 * M_PI_F * (1.0f - cos_max);
      const float flux = inv_pi_area * area * cl * omega / ((float)n_total * p_pick) *
                         photon_target_balance_weight(tg, T, p_pick / omega, o, d, false);
      trace_photon(s,
                   rng,
                   o,
                   d,
                   L.color * flux,
                   max_bounces,
                   out,
                   L.link_membership,
                   L.lightgroup,
                   beam_probability,
                   L.initial_volume_sigma,
                   L.initial_volume_scatter,
                   L.initial_volume_material);
    }
  }
}

/* ------------------------------------------------------- scene extraction */

/* Match either the internal socket identifier ("transmission_weight") or the
 * UI name ("Transmission Weight") - find_input() only knows the former. */
const SocketType *find_input_socket(const ShaderNode *node, const char *name)
{
  const ustring uname(name);
  for (const SocketType &socket : node->type->inputs) {
    if (socket.name == uname || socket.ui_name == uname) {
      return &socket;
    }
  }
  return nullptr;
}

float find_float_input(const ShaderNode *node, const char *name, const float fallback)
{
  const SocketType *socket = find_input_socket(node, name);
  return (socket && socket->type == SocketType::FLOAT) ? node->get_float(*socket) : fallback;
}

float3 find_float3_input(const ShaderNode *node, const char *name, const float3 fallback)
{
  const SocketType *socket = find_input_socket(node, name);
  return (socket && (socket->type == SocketType::COLOR || socket->type == SocketType::VECTOR)) ?
             node->get_float3(*socket) :
             fallback;
}

const ShaderInput *find_input(const ShaderNode *node, const char *name)
{
  const SocketType *socket = find_input_socket(node, name);
  if (!socket) {
    return nullptr;
  }
  for (const ShaderInput *in : node->inputs) {
    if (in->socket_type.name == socket->name) {
      return in;
    }
  }
  return nullptr;
}

bool input_is_linked(const ShaderNode *node, const char *name)
{
  const ShaderInput *in = find_input(node, name);
  return in != nullptr && in->link != nullptr;
}

/* True when the node's Normal input is driven by a Bump or Normal Map node.
 * Such casters run the accurate path: the fast profile samples the smooth
 * surface normal, so mapped waves/detail would cast NO caustic structure
 * (the Octane/Corona habit "flat water plane + noise bump" silently lost
 * its wave caustics). The accurate walk evaluates the real shader, and with
 * the photon ray differentials set in photon_trace.h the map perturbs the
 * closure normal there. Direct-parent check on purpose: only these two node
 * types prove a deliberate normal map - anything else sync hangs onto the
 * socket must not reroute materials onto the slower path. */
bool input_has_normal_map(const ShaderNode *node)
{
  const ShaderInput *in = find_input(node, "Normal");
  return in != nullptr && in->link != nullptr &&
         (in->link->parent->is_a(BumpNode::get_node_type()) ||
          in->link->parent->is_a(NormalMapNode::get_node_type()));
}

/* Mean value of an image file (RGB average per pixel - matches Cycles'
 * implicit color->float conversion; raw file values, since the
 * caustic-deciding scalars come from Non-Color maps). ~256 rows are
 * sampled; cached per file for the process lifetime (leaked on purpose,
 * see the report registry). Returns -1 when the file cannot be read
 * (packed/UDIM/generated images) - the caller keeps the accurate path. */
float image_file_mean(const ustring &filename)
{
  static std::mutex mean_mutex;
  static auto *cache = new std::map<ustring, float>();

  if (filename.empty() || strstr(filename.c_str(), "<UDIM>") != nullptr) {
    return -1.0f;
  }
  {
    const std::lock_guard<std::mutex> lock(mean_mutex);
    const auto it = cache->find(filename);
    if (it != cache->end()) {
      return it->second;
    }
  }

  float mean = -1.0f;
  unique_ptr<ImageInput> in = ImageInput::open(filename.string());
  if (in) {
    const ImageSpec &spec = in->spec();
    const int64_t w = spec.width, h = spec.height, nch = spec.nchannels;
    if (w > 0 && h > 0 && nch > 0) {
      const int64_t step = std::max<int64_t>(1, h / 256);
      const int cch = (int)std::min<int64_t>(nch, 3);
      vector<float> row((size_t)(w * nch));
      double sum = 0.0;
      int64_t count = 0;
      for (int64_t y = 0; y < h; y += step) {
        if (!in->read_scanlines(
                0, 0, (int)y, (int)y + 1, 0, 0, (int)nch, OIIO::TypeDesc::FLOAT, row.data()))
        {
          count = 0;
          break;
        }
        for (int64_t x = 0; x < w; x++) {
          double v = 0.0;
          for (int c = 0; c < cch; c++) {
            v += row[(size_t)(x * nch + c)];
          }
          sum += v / (double)cch;
        }
        count += w;
      }
      if (count > 0) {
        mean = (float)(sum / (double)count);
      }
    }
    in->close();
  }

  const std::lock_guard<std::mutex> lock(mean_mutex);
  (*cache)[filename] = mean;
  return mean;
}

/* Mean of the image texture directly linked into `name` (hopping over the
 * implicit color<->float conversion nodes the sync inserts), -1 otherwise. */
float input_texture_mean(const ShaderNode *node, const char *name)
{
  const ShaderInput *in = find_input(node, name);
  if (in == nullptr || in->link == nullptr) {
    return -1.0f;
  }
  ShaderNode *src = in->link->parent;
  for (int hop = 0; hop < 4 && src->special_type == SHADER_SPECIAL_TYPE_AUTOCONVERT; hop++) {
    if (src->inputs.empty() || src->inputs[0]->link == nullptr) {
      return -1.0f;
    }
    src = src->inputs[0]->link->parent;
  }
  if (!src->is_a(ImageTextureNode::get_node_type())) {
    return -1.0f;
  }
  const ImageTextureNode *tex = static_cast<ImageTextureNode *>(src);
  /* Sampled during Blender sync - the only route that also works for the
   * packed images most real scenes carry (see blender/shader.cpp). Reading
   * the file stays as the fallback for graphs built without that sync, and
   * for images whose buffer could not be acquired. */
  if (tex->photon_texture_mean >= 0.0f) {
    return tex->photon_texture_mean;
  }
  return image_file_mean(tex->get_filename());
}

/* Average colour a linked colour input carries, or a negative w when the
 * chain is not worth guessing at.
 *
 * Constant chains are already folded by Cycles, so anything still linked
 * VARIES - over position, over UV, over a ramp. A single number cannot
 * represent that, but the average of what the chain mixes between is much
 * closer than falling back on a socket default nobody set. Drinks are the
 * case that forced this: the tester's liquid drives its absorption colour
 * through Geometry -> Separate XYZ -> Colour Ramp -> Mix, the standard way
 * to make a gradient from the surface down, and its socket default was a
 * leftover that tinted nothing (report 2026-08-25).
 *
 * Deliberately shallow: average the constant colour inputs of the source
 * node, one hop, no recursion into ramps or textures beyond the image mean
 * that already exists. Enough for mixes and gradients, and it fails to a
 * negative w rather than inventing a colour. */
float4 input_color_estimate(const ShaderNode *node, const char *name)
{
  const float m = input_texture_mean(node, name);
  if (m >= 0.0f) {
    return make_float4(m, m, m, 1.0f);
  }

  const ShaderInput *in = find_input(node, name);
  if (in == nullptr || in->link == nullptr) {
    return make_float4(0.0f, 0.0f, 0.0f, -1.0f);
  }
  ShaderNode *src = in->link->parent;
  for (int hop = 0; hop < 4 && src->special_type == SHADER_SPECIAL_TYPE_AUTOCONVERT; hop++) {
    if (src->inputs.empty() || src->inputs[0]->link == nullptr) {
      return make_float4(0.0f, 0.0f, 0.0f, -1.0f);
    }
    src = src->inputs[0]->link->parent;
  }

  float3 sum = make_float3(0.0f, 0.0f, 0.0f);
  int n = 0;
  for (const ShaderInput *si : src->inputs) {
    if (si->link != nullptr || si->type() != SocketType::COLOR) {
      continue;
    }
    const float3 c = src->get_float3(si->socket_type);
    sum = sum + c;
    n++;
  }
  if (n == 0) {
    return make_float4(0.0f, 0.0f, 0.0f, -1.0f);
  }
  return make_float4(sum.x / n, sum.y / n, sum.z / n, 1.0f);
}

/* ------------------------------------- weight-aware shader classification
 *
 * The classifier runs on FINALIZED Cycles graphs (ShaderManager's
 * device_update_pre always runs before PhotonMap::restart). That buys a lot:
 * node groups are already dissolved into plain nodes, constant input chains
 * are folded into socket values, nodes that do not feed the output are
 * pruned, and transform_multi_closure() rewired every closure's
 * "SurfaceMixWeight" into a chain of MixClosureWeightNode/MathNode that is
 * constant-evaluable whenever the mix factors are constant.
 *
 * So this code only adds what Cycles does not do for us: evaluate those
 * weight chains (mix/add shaders resolve to per-closure shares), pick the
 * dominant caustic layer, fold its share into the per-event tint, and
 * report every approximation loudly instead of silently dropping caustics. */

struct ClassifyNotes {
  int level = 0; /* 0 = exact, 1 = approximated, 2 = casts no caustics */
  /* Hidden-specular evidence (textured transmission/metallic): the material
   * MIGHT cast caustics even though its profile classified as receiver -
   * accurate mode then aims photons at it. Without this gate, force-accurate
   * would turn huge diffuse floors into aim targets whose bounding discs
   * soak up the whole photon budget. */
  bool maybe_caster = false;
  /* The material carries a medium that SCATTERS. The photon walk counts
   * scattering as loss only - what would scatter is removed and never comes
   * back - so our account of the light through it is incomplete by design.
   * The path tracer must keep its own caustic contribution there. */
  bool volume_scatters = false;
  string reason;

  void flag(const int lvl, const string &r)
  {
    level = std::max(level, lvl);
    if (reason.empty()) {
      reason = r;
    }
    else if (reason.find(r) == string::npos) {
      reason += "; " + r;
    }
  }
  void merge(const ClassifyNotes &o)
  {
    if (!o.reason.empty()) {
      flag(o.level, o.reason);
    }
  }
};

float eval_weight_socket(const ShaderInput *in, ClassifyNotes &notes, const int depth);

/* Constant value of one output in a transform_multi_closure() weight chain.
 * Textured (unfoldable) mix factors fall back to the stored socket value and
 * mark the material as approximated. */
float eval_weight_output(const ShaderOutput *out, ClassifyNotes &notes, const int depth)
{
  if (depth > 32) {
    notes.flag(1, "mix nesting too deep (full weight assumed)");
    return 1.0f;
  }
  ShaderNode *node = out->parent;
  if (node->is_a(MixClosureWeightNode::get_node_type())) {
    const ShaderInput *fac_in = find_input(node, "Fac");
    float fac = fac_in ? node->get_float(fac_in->socket_type) : 1.0f;
    if (fac_in && fac_in->link) {
      /* Textured mix factor. The original slider value is NOT preserved on
       * this internal node (transform_multi_closure only copies it when the
       * factor is unlinked; the node default of 1.0 would silently kill
       * closure 1). A directly linked image texture supplies its mean -
       * near-constant masks resolve exactly; everything else stays an
       * approximation and says so. */
      const float fm = input_texture_mean(node, "Fac");
      if (fm >= 0.0f) {
        fac = fm;
        if (fm > 0.05f && fm < 0.95f) {
          notes.flag(1, "textured mix factor (texture mean used)");
        }
      }
      else {
        notes.flag(1, "textured mix factor (50/50 assumed)");
        fac = 0.5f;
      }
    }
    fac = std::min(std::max(fac, 0.0f), 1.0f); /* the kernel saturates too */
    const float w = eval_weight_socket(find_input(node, "Weight"), notes, depth + 1);
    return (out->name() == "Weight1") ? w * (1.0f - fac) : w * fac;
  }
  if (node->is_a(MathNode::get_node_type())) {
    /* transform_multi_closure stacks pre-existing weights with default (add)
     * math nodes; multiply covers hand-built chains. */
    const NodeMathType math_type = static_cast<MathNode *>(node)->get_math_type();
    if (math_type == NODE_MATH_ADD || math_type == NODE_MATH_MULTIPLY) {
      const float v1 = eval_weight_socket(find_input(node, "Value1"), notes, depth + 1);
      const float v2 = eval_weight_socket(find_input(node, "Value2"), notes, depth + 1);
      return (math_type == NODE_MATH_ADD) ? v1 + v2 : v1 * v2;
    }
  }
  notes.flag(1, "non-constant mix weight (full weight assumed)");
  return 1.0f;
}

float eval_weight_socket(const ShaderInput *in, ClassifyNotes &notes, const int depth)
{
  if (in == nullptr) {
    return 1.0f;
  }
  if (in->link == nullptr) {
    return in->parent->get_float(in->socket_type);
  }
  return eval_weight_output(in->link, notes, depth);
}

/* Share of the surface output this closure receives. `has_weight` is false
 * when the graph has no evaluable weight (not finalized yet): the caller
 * falls back to the legacy full weight. */
float closure_weight(const ShaderNode *node,
                     ClassifyNotes &notes,
                     bool &has_weight,
                     const bool volume = false)
{
  const ShaderInput *win = find_input(node, volume ? "VolumeMixWeight" : "SurfaceMixWeight");
  if (win == nullptr) {
    has_weight = false;
    return 1.0f;
  }
  if (win->link != nullptr) {
    has_weight = true;
    return std::max(eval_weight_output(win->link, notes, 0), 0.0f);
  }
  const float w = node->get_float(win->socket_type);
  has_weight = (w != 0.0f);
  return w;
}

/* Constant emission radiance (closure weight x strength x color) of a
 * shader, black when it has none. Textured emission uses the node values
 * and flags a note. */
float3 shader_emission_radiance(Shader *shader, ClassifyNotes &notes)
{
  float3 total = make_float3(0.0f, 0.0f, 0.0f);
  if (!shader || !shader->graph) {
    return total;
  }
  for (ShaderNode *node : shader->graph->nodes) {
    const bool is_emission = node->is_a(EmissionNode::get_node_type());
    const bool is_principled = node->is_a(PrincipledBsdfNode::get_node_type());
    if (!is_emission && !is_principled) {
      continue;
    }
    ClassifyNotes wnotes;
    bool has_weight = false;
    float w = closure_weight(node, wnotes, has_weight);
    if (!has_weight) {
      w = 1.0f;
    }
    else if (w <= 1e-4f) {
      continue;
    }
    float strength;
    float3 color;
    if (is_emission) {
      strength = find_float_input(node, "Strength", 1.0f);
      color = find_float3_input(node, "Color", make_float3(1, 1, 1));
      if (input_is_linked(node, "Strength") || input_is_linked(node, "Color")) {
        notes.flag(1, "textured emission (node value used)");
      }
    }
    else {
      strength = find_float_input(node, "Emission Strength", 0.0f);
      color = find_float3_input(node, "Emission Color", make_float3(0, 0, 0));
      if (strength > 0.0f && (color.x > 0.0f || color.y > 0.0f || color.z > 0.0f) &&
          (input_is_linked(node, "Emission Strength") || input_is_linked(node, "Emission Color")))
      {
        notes.flag(1, "textured emission (node value used)");
      }
    }
    notes.merge(wnotes);
    total = total + color * (w * strength);
  }
  return make_float3(std::max(total.x, 0.0f), std::max(total.y, 0.0f), std::max(total.z, 0.0f));
}

struct VolumeProperties {
  float3 sigma_t = make_float3(0.0f, 0.0f, 0.0f);
  float3 sigma_s = make_float3(0.0f, 0.0f, 0.0f);
  float g = 0.0f;
};

/* Uniform volume coefficients used by the photon walk.
 *
 * Reading only a directly-connected Absorption node was too narrow for how
 * drinks are actually built. The tester's glass (2026-08-25) tints its
 * liquid entirely through the volume - the surface is colourless - and
 * combines absorption with scatter through an Add Shader, with the
 * absorption colour driven by another node. All three of those made us fall
 * through to "not simulated", so the drink cast a white caustic while the
 * render showed it orange.
 *
 * Walking every volume node in the graph and weighting it the same way the
 * surface candidates are weighted handles Add and Mix for free - Cycles
 * routes those through the same mix-weight mechanism.
 *
 * Same arithmetic Cycles uses (svm/closure.h, svm_node_closure_volume):
 * absorption contributes (1 - colour) * density, scatter contributes
 * colour * density, and the extinction is their sum.
 *
 * The photon walk uses these coefficients for one sampled scattering event;
 * heterogeneous textures and multiple scattering remain outside this fast
 * path. */
VolumeProperties volume_properties(Shader *shader, ClassifyNotes &notes)
{
  VolumeProperties total;
  if (!shader || !shader->graph) {
    return total;
  }

  for (ShaderNode *node : shader->graph->nodes) {
    const bool is_absorb = node->is_a(AbsorptionVolumeNode::get_node_type());
    const bool is_scatter = node->is_a(ScatterVolumeNode::get_node_type());
    const bool is_coefficients = node->is_a(VolumeCoefficientsNode::get_node_type());
    const bool is_principled = node->is_a(PrincipledVolumeNode::get_node_type());
    if (!(is_absorb || is_scatter || is_coefficients || is_principled)) {
      continue;
    }

    ClassifyNotes wnotes;
    bool has_weight = false;
    float w = closure_weight(node, wnotes, has_weight, true);
    if (!has_weight) {
      w = 1.0f;
    }
    else if (w <= 1e-4f) {
      continue; /* dead branch of a mix */
    }

    float density = find_float_input(node, "Density", 1.0f);
    const float dm = input_texture_mean(node, "Density");
    if (dm >= 0.0f) {
      density = dm;
    }
    else if (input_is_linked(node, "Density")) {
      wnotes.flag(1, "volume density is textured (node value used)");
    }
    if (density <= 0.0f) {
      continue;
    }

    float3 sigma_s = make_float3(0.0f, 0.0f, 0.0f);
    float3 sigma_a = make_float3(0.0f, 0.0f, 0.0f);
    float node_g = 0.0f;
    if (is_coefficients) {
      sigma_s = find_float3_input(node, "Scatter Coefficients", sigma_s);
      sigma_a = find_float3_input(node, "Absorption Coefficients", sigma_a);
      node_g = find_float_input(node, "Anisotropy", 0.0f);
    }
    else {
      float3 color = find_float3_input(node, "Color", make_float3(1, 1, 1));
      if (input_is_linked(node, "Color")) {
        const float4 est = input_color_estimate(node, "Color");
        if (est.w > 0.0f) {
          color = make_float3(est.x, est.y, est.z);
          wnotes.flag(1, "volume colour varies across the medium (average used)");
        }
        else {
          wnotes.flag(1, "volume colour comes from nodes (node value used)");
        }
      }
      color = make_float3(std::min(std::max(color.x, 0.0f), 1.0f),
                          std::min(std::max(color.y, 0.0f), 1.0f),
                          std::min(std::max(color.z, 0.0f), 1.0f));

      if (is_absorb) {
        sigma_a = (make_float3(1.0f, 1.0f, 1.0f) - color) * density;
      }
      else {
        sigma_s = color * density;
        if (is_principled) {
          const float3 absorption_color = make_float3(
              sqrtf(std::max(find_float3_input(node, "Absorption Color", make_float3(0, 0, 0)).x,
                             0.0f)),
              sqrtf(std::max(find_float3_input(node, "Absorption Color", make_float3(0, 0, 0)).y,
                             0.0f)),
              sqrtf(std::max(find_float3_input(node, "Absorption Color", make_float3(0, 0, 0)).z,
                             0.0f)));
          sigma_a = (make_float3(1, 1, 1) - color) *
                    (make_float3(1, 1, 1) - absorption_color) * density;
        }
        node_g = find_float_input(node, "Anisotropy", 0.0f);
        notes.volume_scatters = true;
        wnotes.flag(1,
                    is_principled ?
                        "principled volume uses uniform single-scattering coefficients" :
                        "volume scattering uses uniform single-scattering coefficients");
      }
    }

    sigma_s = max(sigma_s, make_float3(0.0f, 0.0f, 0.0f));
    sigma_a = max(sigma_a, make_float3(0.0f, 0.0f, 0.0f));
    const float3 sigma_t = sigma_s + sigma_a;
    const float scatter_weight = w * (sigma_s.x + sigma_s.y + sigma_s.z);
    total.sigma_s = total.sigma_s + sigma_s * w;
    total.sigma_t = total.sigma_t + sigma_t * w;
    if (scatter_weight > 0.0f) {
      total.g += node_g * scatter_weight;
      notes.volume_scatters = true;
    }
    notes.merge(wnotes);
  }

  const float scatter_weight = total.sigma_s.x + total.sigma_s.y + total.sigma_s.z;
  if (scatter_weight > 0.0f) {
    total.g = std::min(std::max(total.g / scatter_weight, -0.999f), 0.999f);
  }
  total.sigma_s = max(total.sigma_s, make_float3(0.0f, 0.0f, 0.0f));
  total.sigma_t = max(total.sigma_t, make_float3(0.0f, 0.0f, 0.0f));
  return total;
}

/* Classify a Cycles shader for the photon walk by inspecting its finalized
 * graph. `notes` collects why the profile is approximate. */
PhotonMaterial classify_shader_impl(Shader *shader, ClassifyNotes &notes)
{
  PhotonMaterial m;
  if (!shader || !shader->graph) {
    return m;
  }

  struct Candidate {
    int kind = MAT_RECEIVER;
    float w = 1.0f;
    float ior = 1.45f;
    float rough = 0.0f;
    float transmission = 0.0f;
    float dispersion_inv_abbe = 0.0f;
    float3 color = make_float3(1, 1, 1);
    ClassifyNotes notes;
  };
  vector<Candidate> cands;
  /* Pure transparent materials (helper planes, effect emitters) must not
   * block or receive photons - Cycles rays pass straight through them. */
  bool has_transparent = false;
  bool has_solid = false;
  float coat_sum = 0.0f, coat_best = 0.0f;
  float transparent_sum = 0.0f;
  float coat_rough = 0.03f, coat_ior = 1.5f;

  for (ShaderNode *node : shader->graph->nodes) {
    const bool is_glass = node->is_a(GlassBsdfNode::get_node_type()) ||
                          node->is_a(RefractionBsdfNode::get_node_type());
    const bool is_principled = node->is_a(PrincipledBsdfNode::get_node_type());
    const bool is_glossy = node->is_a(GlossyBsdfNode::get_node_type());
    /* Translucent counts as diffuse here. It scatters light rather than
     * focusing it, so it is no caster - but it must still RECEIVE caustics,
     * and a node type the loop does not recognise leaves the material with
     * no solid surface at all, which made lampshades and curtains invisible
     * to the gather (tester report 2026-08-25). */
    const bool is_diffuse = node->is_a(DiffuseBsdfNode::get_node_type()) ||
                            node->is_a(TranslucentBsdfNode::get_node_type());
    const bool is_transparent = node->is_a(TransparentBsdfNode::get_node_type());
    if (!(is_glass || is_principled || is_glossy || is_diffuse || is_transparent)) {
      continue;
    }

    ClassifyNotes wnotes;
    bool has_weight = false;
    float w = closure_weight(node, wnotes, has_weight);
    if (!has_weight) {
      w = 1.0f; /* unfinalized graph (defensive): legacy behaviour */
    }
    else if (w <= 1e-4f) {
      /* Dead branch of a mix. A weight chain that was only resolvable by
       * approximation still deserves its report entry - the branch might
       * well be live in the real render. */
      notes.merge(wnotes);
      continue;
    }

    if (is_transparent) {
      has_transparent = true;
      continue;
    }
    has_solid = true;
    if (is_diffuse) {
      continue; /* plain receiver contribution */
    }

    if (is_glass) {
      Candidate c;
      c.kind = MAT_GLASS;
      c.w = w;
      c.notes = wnotes;
      c.ior = find_float_input(node, "IOR", 1.45f);
      c.rough = find_float_input(node, "Roughness", 0.0f);
      c.color = find_float3_input(node, "Color", make_float3(1, 1, 1));
      if (input_is_linked(node, "Color")) {
        c.notes.flag(1, "textured glass color (node value used)");
      }
      if (input_is_linked(node, "Roughness")) {
        c.notes.flag(1, "textured roughness (node value used)");
      }
      if (input_is_linked(node, "IOR")) {
        c.notes.flag(1, "textured IOR (node value used)");
      }
      if (input_has_normal_map(node)) {
        c.notes.flag(1, "bump/normal map - handled by the accurate photon path");
      }
      cands.push_back(c);
      continue;
    }
    if (is_glossy) {
      Candidate c;
      c.kind = MAT_METAL;
      c.w = w;
      c.notes = wnotes;
      c.rough = find_float_input(node, "Roughness", 0.0f);
      c.color = find_float3_input(node, "Color", make_float3(1, 1, 1));
      if (input_is_linked(node, "Color")) {
        c.notes.flag(1, "textured metal color (node value used)");
      }
      if (input_is_linked(node, "Roughness")) {
        c.notes.flag(1, "textured roughness (node value used)");
      }
      if (input_has_normal_map(node)) {
        c.notes.flag(1, "bump/normal map - handled by the accurate photon path");
      }
      cands.push_back(c);
      continue;
    }

    /* Principled. Texture-driven deciding scalars: a DIRECTLY linked image
     * texture is read (mean) - near-constant masks classify exactly (the
     * white metallic mask on brass, the near-black mask on a prop), only
     * genuinely varying masks pay the accurate path. */
    float transmission = find_float_input(node, "Transmission Weight", 0.0f);
    float metallic = find_float_input(node, "Metallic", 0.0f);
    bool t_textured = input_is_linked(node, "Transmission Weight");
    bool metal_textured = input_is_linked(node, "Metallic");
    if (t_textured) {
      const float tm = input_texture_mean(node, "Transmission Weight");
      if (tm >= 0.0f && (tm <= 0.05f || tm >= 0.95f)) {
        transmission = tm;
        t_textured = false;
      }
    }
    if (metal_textured) {
      const float mm = input_texture_mean(node, "Metallic");
      if (mm >= 0.0f && (mm <= 0.05f || mm >= 0.95f)) {
        metallic = mm;
        metal_textured = false;
      }
    }

    /* Alpha (Principled mixes a white Transparent BSDF): the transparent
     * share passes photons untouched. Textured masks classify by their mean
     * when near-constant, like the other deciding scalars. */
    {
      float alpha = find_float_input(node, "Alpha", 1.0f);
      bool alpha_textured = input_is_linked(node, "Alpha");
      if (alpha_textured) {
        const float am = input_texture_mean(node, "Alpha");
        if (am >= 0.0f && (am <= 0.05f || am >= 0.95f)) {
          alpha = am;
          alpha_textured = false;
        }
        else {
          alpha = 1.0f; /* varying mask: stay opaque for photons */
          notes.flag(1, "textured alpha - photons treat it as opaque");
        }
      }
      transparent_sum += w * (1.0f - std::min(std::max(alpha, 0.0f), 1.0f));
    }

    /* Anisotropy (brushed metal, anodised aluminium, watch cases, turned
     * dials). The fast profile samples an ISOTROPIC GGX lobe - one
     * roughness, no tangent frame - so a brushed surface would throw the
     * round caustic of a polished one instead of a stretched streak. The
     * accurate walk runs the material's real BSDF and gets it right, so
     * flagging is all that is needed here. Cycles disconnects an unused
     * Anisotropic input during graph cleanup and the classifier sees the
     * finalized graph, so a linked socket really means it is in use. */
    const bool anisotropic = fabsf(find_float_input(node, "Anisotropic", 0.0f)) > 0.01f ||
                             input_is_linked(node, "Anisotropic");

    /* Clearcoat layer (car paint!). Newer Principled names first, the
     * pre-4.0 "Clearcoat" names as fallback. Coat layers of every mixed
     * Principled add up; the strongest contributor sets the coat params. */
    float coat = find_float_input(node, "Coat Weight", 0.0f);
    float this_coat_rough = find_float_input(node, "Coat Roughness", 0.03f);
    if (coat <= 0.0f) {
      coat = find_float_input(node, "Clearcoat", 0.0f);
      this_coat_rough = find_float_input(node, "Clearcoat Roughness", 0.03f);
    }
    coat_sum += w * coat;
    if (w * coat > coat_best) {
      coat_best = w * coat;
      coat_rough = this_coat_rough;
      coat_ior = find_float_input(node, "Coat IOR", 1.5f);
    }

    if (transmission > 0.5f) {
      Candidate c;
      c.kind = MAT_GLASS;
      c.w = w;
      c.notes = wnotes;
      c.ior = find_float_input(node, "IOR", 1.45f);
      const float dispersion_scale = std::clamp(
          find_float_input(node, "Transmission Dispersion Scale", 0.0f), 0.0f, 1.0f);
      const float abbe_number = std::max(
          find_float_input(node, "Transmission Dispersion Abbe Number", 20.0f), 0.0f);
      c.dispersion_inv_abbe = (abbe_number > 0.0f) ? dispersion_scale / abbe_number : 0.0f;
      if (input_is_linked(node, "Transmission Dispersion Scale") ||
          input_is_linked(node, "Transmission Dispersion Abbe Number"))
      {
        c.notes.flag(1, "textured dispersion - handled by the accurate photon path");
      }
      c.rough = find_float_input(node, "Roughness", 0.0f);
      /* Principled transmission tints refraction with sqrt(base color) per
       * interface, so the through-transmission of solid glass equals the
       * base color (svm/closure.h generalized_schlick_setup). The tracer
       * multiplies m.color per refraction event, so store the sqrt here;
       * the Glass BSDF keeps its raw color (tints fully on every event).
       * Measured: PT truth for base color 0.8 sits at ~sqrt semantics, the
       * raw-color profile lost ~2.6% region energy to the caustic. */
      const float3 bc = find_float3_input(node, "Base Color", make_float3(1, 1, 1));
      c.color = make_float3(sqrtf(std::min(std::max(bc.x, 0.0f), 1.0f)),
                            sqrtf(std::min(std::max(bc.y, 0.0f), 1.0f)),
                            sqrtf(std::min(std::max(bc.z, 0.0f), 1.0f)));
      if (t_textured) {
        c.notes.flag(1, "textured transmission (node value used)");
      }
      if (input_is_linked(node, "Base Color")) {
        c.notes.flag(1, "textured color (node value used)");
      }
      if (input_is_linked(node, "Roughness")) {
        c.notes.flag(1, "textured roughness (node value used)");
      }
      if (input_is_linked(node, "IOR")) {
        c.notes.flag(1, "textured IOR (node value used)");
      }
      if (anisotropic) {
        c.notes.flag(1, "anisotropic glass - handled by the accurate photon path");
      }
      if (input_has_normal_map(node)) {
        c.notes.flag(1, "bump/normal map - handled by the accurate photon path");
      }
      cands.push_back(c);
    }
    else if (metallic > 0.5f) {
      Candidate c;
      c.kind = MAT_METAL;
      c.w = w;
      c.notes = wnotes;
      c.rough = find_float_input(node, "Roughness", 0.0f);
      c.color = find_float3_input(node, "Base Color", make_float3(1, 1, 1));
      if (metal_textured) {
        c.notes.flag(1, "textured metallic (node value used)");
      }
      if (input_is_linked(node, "Base Color")) {
        c.notes.flag(1, "textured color (node value used)");
      }
      if (input_is_linked(node, "Roughness")) {
        c.notes.flag(1, "textured roughness (node value used)");
      }
      if (anisotropic) {
        c.notes.flag(1, "anisotropic metal - handled by the accurate photon path");
      }
      if (input_has_normal_map(node)) {
        c.notes.flag(1, "bump/normal map - handled by the accurate photon path");
      }
      cands.push_back(c);
    }
    else if (transmission > 0.0f) {
      /* Translucent receiver (inflatable toys, lampshades): deposits AND
       * lets photons through, so backlit caustics match path tracing. */
      Candidate c;
      c.kind = MAT_RECEIVER;
      c.w = w;
      c.notes = wnotes;
      c.transmission = transmission;
      c.rough = find_float_input(node, "Roughness", 0.0f);
      c.color = find_float3_input(node, "Base Color", make_float3(1, 1, 1));
      if (t_textured) {
        c.notes.flag(1, "textured transmission (node value used)");
      }
      cands.push_back(c);
    }
    else {
      /* Opaque Principled: plain receiver. A textured transmission whose
       * node value is zero is the silent killer - the material may well be
       * glass in places, but the fast mode cannot see it. Say so. */
      /* Texture-driven transmission/metallic with node value 0: the material
       * MIGHT be glass/metal across most of its surface (brass with a
       * metallic mask, glass with a painted transmission map - user scene:
       * vintage_microscope). The fast profile cannot know, so these run the
       * accurate path and become aim targets. TODO(Weg A Baustein 3 rest):
       * read the driving texture's mean so props with near-zero masks stay
       * silent fast-mode receivers and skip the target cost. */
      if (t_textured) {
        notes.flag(2, "textured transmission - invisible to the fast mode");
        notes.maybe_caster = true;
      }
      else if (metal_textured) {
        notes.flag(1, "textured metallic - invisible to the fast mode");
        notes.maybe_caster = true;
      }
    }
  }

  m.coat = std::min(coat_sum, 1.0f);
  if (coat_best > 0.0f) {
    m.coat_rough = coat_rough;
    m.coat_ior = coat_ior;
  }
  /* Set before the empty-candidates return: an opaque alpha-0 Principled has
   * no caustic candidate but must still pass photons. */
  m.transparent = std::min(transparent_sum, 1.0f);

  if (ShaderNode *out_node = shader->graph->output()) {
    const ShaderInput *vol_in = find_input(out_node, "Volume");
    if (vol_in != nullptr && vol_in->link != nullptr) {
      const VolumeProperties volume = volume_properties(shader, notes);
      m.volume_sigma = volume.sigma_t;
      m.volume_sigma_s = volume.sigma_s;
      m.volume_g = volume.g;
      if (cands.empty() &&
          max(volume.sigma_t.x, max(volume.sigma_t.y, volume.sigma_t.z)) > 0.0f) {
        m.kind = MAT_VOLUME;
      }
    }
  }

  /* A transparent surface mixed with a volume is the standard Cycles setup
   * for a volume object. Keep it on the dedicated volume walk so the boundary
   * enters and exits the medium instead of being discarded as a pure helper. */
  if (has_transparent && !has_solid) {
    const bool has_volume = max(m.volume_sigma.x, max(m.volume_sigma.y, m.volume_sigma.z)) > 0.0f;
    m.kind = has_volume ? MAT_VOLUME : MAT_SKIP;
    return m;
  }

  if (cands.empty()) {
    return m; /* receiver (diffuse floor etc.) */
  }

  /* The dominant caustic layer wins; single-closure materials behave exactly
   * like the legacy classifier (w == 1). */
  Candidate *best = &cands[0];
  float spec_total = 0.0f;
  for (Candidate &c : cands) {
    spec_total += c.w;
    if (c.w > best->w) {
      best = &c;
    }
  }
  if (cands.size() > 1 && spec_total - best->w > 0.05f) {
    best->notes.flag(1, "mixes several caustic layers (strongest kept)");
  }

  m.kind = best->kind;
  m.ior = best->ior;
  m.dispersion_inv_abbe = best->dispersion_inv_abbe;
  m.rough = best->rough;
  m.color = best->color;
  const float w = std::min(best->w, 1.0f);
  if (m.kind == MAT_RECEIVER) {
    m.transmission = std::min(best->transmission * w, 1.0f);
  }
  else if (w < 0.999f) {
    /* Fold the mix share into the per-event tint: refracting every photon
     * with 0.7x flux equals refracting 70% of them in expectation. */
    m.color = m.color * w;
  }
  notes.merge(best->notes);
  return m;
}

/* Classification + accurate-mode gate. CYCLESPLUS_PHOTON_ACCURATE:
 * 0 = off (pure fast mode, pre-Weg-B behaviour), 1/unset = auto (materials
 * the classifier flagged run the real shader), 2 = force ALL non-skip
 * materials through the accurate path (validation). */
/* Which materials are allowed to cast, read once per extraction from the
 * integrator. Everything here only ever REMOVES casters - the photon
 * transport itself is untouched. */
struct CasterPolicy {
  bool selected_only = false; /* only flagged materials cast */
  bool reflective = true;     /* Blender's "Reflective" caustics checkbox */
  bool refractive = true;     /* Blender's "Refractive" caustics checkbox */

  static CasterPolicy from_scene(Scene *scene)
  {
    CasterPolicy p;
    const Integrator *in = scene->integrator;
    p.selected_only = in->get_photon_casters_selected();
    p.reflective = in->get_caustics_reflective();
    p.refractive = in->get_caustics_refractive();
    return p;
  }
};

/* Strip every CASTING trait, keep the receiving ones. A plain receiver
 * absorbs direct photons (both walks only deposit after a specular event)
 * but still RECEIVES caustics from the remaining casters - the Corona/V-Ray
 * per-material caustics semantics. Thin-wall translucency survives because
 * it is a receiver trait. */
void demote_to_receiver(PhotonMaterial &m)
{
  const bool was_glass = (m.kind == MAT_GLASS);
  if (m.kind != MAT_RECEIVER) {
    /* Demoted glass stays a WINDOW for photons: they pass straight through
     * with the Fresnel-transmitted share (thin-pane approximation - entry
     * and exit bending cancel on flat panes), tinted per event like the
     * caster profile would. Without this a caster BEHIND unflagged glass
     * lost its caustic: photons died at the pane while path tracing passed
     * it (user find 2026-08-01, glass table behind thin-wall windows). */
    const float transmission = was_glass ? 1.0f : m.transmission;
    const float rough = m.rough;
    const float ior = m.ior;
    const float dispersion_inv_abbe = m.dispersion_inv_abbe;
    const float transparent = m.transparent;
    const float3 color = m.color;
    m = PhotonMaterial();
    m.transmission = transmission;
    m.rough = rough;
    m.color = color;
    m.transparent = transparent;
    if (was_glass) {
      m.ior = ior;
      m.dispersion_inv_abbe = dispersion_inv_abbe;
    }
  }
  /* Demoted material = path tracing keeps its caustics (per-material gate),
   * so a photon pass-through must NOT arm a later deposit - that light
   * would be counted twice. */
  m.pass_mode = was_glass ? 2 : 0;
  m.coat = 0.0f;
  m.accurate = 0;
}

PhotonMaterial classify_shader(Shader *shader,
                               const CasterPolicy &policy,
                               ClassifyNotes *out_notes = nullptr)
{
  ClassifyNotes local_notes;
  ClassifyNotes &notes = out_notes ? *out_notes : local_notes;
  PhotonMaterial m = classify_shader_impl(shader, notes);

  /* Pure transparent helpers keep passing photons through untouched. */
  if (m.kind == MAT_SKIP) {
    return m;
  }

  /* Volume-only materials have no surface BSDF for the accurate surface
   * walker to sample. Keep them on the dedicated volume transport path. */
  if (m.kind == MAT_VOLUME) {
    m.accurate = 0;
    return m;
  }

  /* Reflective/refractive split - Blender's own caustics checkboxes, which
   * the photon mode now honours instead of greying them out. Glass casts
   * refractive caustics, metal and clearcoat reflective ones. Accurate
   * materials run the real BSDF and cannot be split per event, so they
   * follow their classified kind (predictable: "glass stopped casting"). */
  const bool wants_refractive = (m.kind == MAT_GLASS);
  const bool wants_reflective = (m.kind == MAT_METAL);
  if ((wants_refractive && !policy.refractive) || (wants_reflective && !policy.reflective)) {
    demote_to_receiver(m);
    notes = ClassifyNotes();
    return m;
  }
  if (!policy.reflective) {
    m.coat = 0.0f; /* clearcoat is a reflective caustic */
  }

  /* Selected-casters mode: materials without the per-material "Cast Photon
   * Caustics" flag do not cast. Opted-out materials also drop out of the
   * panel report - there is nothing the user needs to fix about them. */
  if (policy.selected_only && shader && !shader->get_photon_cast()) {
    demote_to_receiver(m);
    notes = ClassifyNotes();
    return m;
  }

  static const int accurate_env = []() {
    const char *env = getenv("CYCLESPLUS_PHOTON_ACCURATE");
    return env ? atoi(env) : 1;
  }();
  if (accurate_env != 0) {
    m.accurate = (accurate_env == 2 || notes.level > 0 || m.kind == MAT_GLASS) ? 1 : 0;
  }
  return m;
}

/* Decompose the world background into virtual sun lights (Debevec median
 * cut), so HDRI/sky environments cast caustic photons. The background shader
 * is evaluated through Cycles' own ShaderEval, so any node setup works:
 * image environments, procedural skies, mapping rotations, packed images.
 * The caller checks for an enabled background light and caches the result:
 * this evaluation is the expensive part and only depends on the world. */
void eval_world_lights(Scene *scene, Progress *progress, vector<PhotonLight> &lights)
{
  /* Sky-texture sun (analytic): whenever Cycles' sun guiding is active, the
   * background evaluation below EXCLUDES the sky's sun disc
   * (kernel/svm/sky.h skips it for PATH_RAY_IMPORTANCE_BAKE under
   * use_sun_guiding) - so the archviz default lighting cast no photon sun,
   * and because photons suppress path-traced caustics, enabling the
   * checkbox DELETED the caustic path tracing had (measured, nishita_test).
   * Mirror Cycles' own analytic sun: direction/size exactly like
   * scene/light.cpp, irradiance from the sky model's disc radiance. When
   * sun guiding is OFF (several suns, transformed vector input), the disc
   * IS in the evaluation and the median cut below handles it - adding it
   * here too would double count, so this block mirrors the same condition. */
  if (scene->dscene.data.background.use_sun_guiding && scene->background) {
    Shader *bg_shader = scene->background->get_shader(scene);
    SkyTextureNode *sky = nullptr;
    BackgroundNode *bg_node = nullptr;
    if (bg_shader && bg_shader->graph) {
      for (ShaderNode *node : bg_shader->graph->nodes) {
        if (sky == nullptr && node->type == SkyTextureNode::get_node_type()) {
          SkyTextureNode *cand = (SkyTextureNode *)node;
          if (cand->get_sun_disc()) {
            sky = cand;
          }
        }
        else if (bg_node == nullptr && node->type == BackgroundNode::get_node_type()) {
          bg_node = (BackgroundNode *)node;
        }
      }
    }
    if (sky != nullptr) {
      /* Direction: same lat/long + mapping convention as scene/light.cpp. */
      const float latitude = sky->get_sun_elevation();
      const float longitude = sky->get_sun_rotation() + M_PI_2_F;
      float3 sun_dir = make_float3(
          cosf(latitude) * cosf(longitude), cosf(latitude) * sinf(longitude), sinf(latitude));
      const Transform sky_tfm = transform_inverse(sky->tex_mapping.compute_transform());
      sun_dir = normalize(transform_direction(&sky_tfm, sun_dir));
      const float half_angle = sky->get_sun_size() * 0.5f;
      /* Irradiance = limb-darkened disc radiance x solid angle, converted
       * XYZ -> scene linear RGB, scaled by the Background node's strength
       * (constant socket; a linked strength falls back to 1 - best effort,
       * an approximate sun beats a deleted caustic). */
      const float3 rad_xyz = sky->get_sun_disc_average_radiance_xyz();
      const Transform xyz_to_rgb = ColorSpaceManager::get_xyz_to_scene_linear_rgb();
      float3 rad_rgb = transform_direction(&xyz_to_rgb, rad_xyz);
      rad_rgb = make_float3(fmaxf(rad_rgb.x, 0.0f), fmaxf(rad_rgb.y, 0.0f), fmaxf(rad_rgb.z, 0.0f));
      const float omega = 2.0f * M_PI_F * (1.0f - cosf(half_angle));
      float strength = 1.0f;
      if (bg_node != nullptr) {
        const ShaderInput *strength_in = bg_node->input("Strength");
        if (strength_in == nullptr || strength_in->link == nullptr) {
          strength = bg_node->get_strength();
        }
      }
      const float3 irradiance = rad_rgb * (omega * strength);
      const float lum = 0.2126f * irradiance.x + 0.7152f * irradiance.y + 0.0722f * irradiance.z;
      if (lum > 1e-8f) {
        PhotonLight L;
        L.type = 0; /* sun */
        L.lightgroup = photon_lightgroup_index(scene, scene->background->get_lightgroup());
        L.color = irradiance;
        L.pos = make_float3(0.0f, 0.0f, 0.0f);
        L.axis = -sun_dir; /* photons travel away from the sun */
        L.p0 = half_angle;
        L.spot_cos = -1.0f;
        L.sx = L.sy = 0.0f;
        L.shape = 0; /* true cone sun, not a median-cut rect */
        lights.push_back(L);
        LOG_INFO << string_printf(
            "CyclesPlus photon map: sky-texture sun -> analytic photon sun, %.4f W/m2 "
            "from (%.2f %.2f %.2f), half angle %.4f rad",
            lum,
            -L.axis.x,
            -L.axis.y,
            -L.axis.z,
            half_angle);
      }
    }
  }

  /* Evaluate the background shader into an equirectangular pixel field
   * (same mechanism the light manager uses for importance sampling).
   * Convention (kernel/camera/projection.h): row y = 0 is the bottom.
   * Resolution must be high enough that a sharp sun disc (~0.009 rad, ~3 px
   * at 2048) reliably spans multiple samples - at 1024 a real HDRI sun can
   * still fall between pixel centers and lose most of its energy. */
  const int width = 2048;
  const int height = 1024;
  const int size = width * height;
  vector<float3> pixels(size);
  {
    Device *device = scene->device;
    DeviceScene *dscene = &scene->dscene;
    device->const_copy_to("data", &dscene->data, sizeof(dscene->data));

    ShaderEval shader_eval(device, *progress);
    shader_eval.eval(
        SHADER_EVAL_BACKGROUND,
        size,
        3,
        [&](device_vector<KernelShaderEvalInput> &d_input) {
          KernelShaderEvalInput *d_input_data = d_input.data();
          for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
              KernelShaderEvalInput in;
              /* Repurpose object and prim to pass resolution for ray differential. */
              in.object = width;
              in.prim = height;
              in.u = (x + 0.5f) / width;
              in.v = (y + 0.5f) / height;
              d_input_data[x + y * width] = in;
            }
          }
          return size;
        },
        [&](device_vector<float> &d_output) {
          const float *d_output_data = d_output.data();
          for (int i = 0; i < size; i++) {
            pixels[i] = make_float3(
                d_output_data[3 * i], d_output_data[3 * i + 1], d_output_data[3 * i + 2]);
          }
        });
  }

  /* Per-pixel energy: radiance * pixel solid angle (sin(theta) weighted). */
  vector<float> energy(size);
  vector<float> domega(height);
  double total = 0.0;
  for (int y = 0; y < height; y++) {
    const float v = (y + 0.5f) / height;
    const float theta = M_PI_F - M_PI_F * v;
    domega[y] = (2.0f * M_PI_F / width) * (M_PI_F / height) * sinf(theta);
    for (int x = 0; x < width; x++) {
      const float3 c = pixels[y * width + x];
      const float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
      energy[y * width + x] = std::max(lum, 0.0f) * domega[y];
      total += energy[y * width + x];
    }
  }
  if (total < 1e-8) {
    return;
  }

  /* Debevec median cut: 7 rounds -> up to 128 regions. */
  struct Region {
    int x0, y0, x1, y1; /* [x0, x1) x [y0, y1) */
  };
  auto region_energy = [&](const Region &r) {
    double e = 0.0;
    for (int y = r.y0; y < r.y1; y++) {
      for (int x = r.x0; x < r.x1; x++) {
        e += energy[y * width + x];
      }
    }
    return e;
  };
  vector<Region> regions;
  regions.push_back({0, 0, width, height});
  for (int round = 0; round < 7; round++) {
    vector<Region> next;
    for (const Region &r : regions) {
      const double e = region_energy(r);
      const int nx = r.x1 - r.x0, ny = r.y1 - r.y0;
      if (e <= 0.0 || (nx <= 1 && ny <= 1)) {
        next.push_back(r);
        continue;
      }
      /* Split along the angularly longer axis at the energy median. */
      const float theta_c = M_PI_F - M_PI_F * ((r.y0 + r.y1) * 0.5f / height);
      const float ext_x = nx * (2.0f * M_PI_F / width) * std::max(sinf(theta_c), 0.05f);
      const float ext_y = ny * (M_PI_F / height);
      const bool split_x = (nx > 1) && (ext_x >= ext_y || ny <= 1);
      double acc = 0.0;
      if (split_x) {
        int sx = r.x0 + 1;
        for (int x = r.x0; x < r.x1 - 1; x++) {
          for (int y = r.y0; y < r.y1; y++) {
            acc += energy[y * width + x];
          }
          if (acc >= e * 0.5) {
            sx = x + 1;
            break;
          }
        }
        next.push_back({r.x0, r.y0, sx, r.y1});
        next.push_back({sx, r.y0, r.x1, r.y1});
      }
      else {
        int sy = r.y0 + 1;
        for (int y = r.y0; y < r.y1 - 1; y++) {
          for (int x = r.x0; x < r.x1; x++) {
            acc += energy[y * width + x];
          }
          if (acc >= e * 0.5) {
            sy = y + 1;
            break;
          }
        }
        next.push_back({r.x0, r.y0, r.x1, sy});
        next.push_back({r.x0, sy, r.x1, r.y1});
      }
    }
    regions = next;
  }

  /* Each region becomes a virtual sun aimed at its energy centroid. */
  int num_world_suns = 0;
  for (const Region &r : regions) {
    double e = 0.0, cx = 0.0, cy = 0.0, solid = 0.0;
    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int y = r.y0; y < r.y1; y++) {
      for (int x = r.x0; x < r.x1; x++) {
        const float pe = energy[y * width + x];
        e += pe;
        cx += pe * (x + 0.5);
        cy += pe * (y + 0.5);
        solid += domega[y];
        color_sum += pixels[y * width + x] * domega[y];
      }
    }
    if (e < total * 1e-4) {
      continue; /* skip near-black regions */
    }
    const float u = (float)(cx / e) / width;
    const float v = (float)(cy / e) / height;
    const float phi = M_PI_F - 2.0f * M_PI_F * u;
    const float theta = M_PI_F - M_PI_F * v;
    const float3 dir = make_float3(
        sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta));

    PhotonLight L;
    L.type = 0; /* sun */
    /* Every median-cut region belongs to the world's own light group: the
     * environment is one light source to the artist, however many directions
     * we split it into internally. */
    L.lightgroup = photon_lightgroup_index(scene, scene->background->get_lightgroup());
    L.color = color_sum; /* integrated irradiance from this region, W/m^2 */
    /* Photon directions sample the region's ACTUAL angular rect (uniform
     * over its solid angle: phi uniform, cos(theta) uniform), not a small
     * cone around the centroid. The old centroid-cone concentrated a
     * studio HDRI's softbox panels into discrete blobs - visible as faint
     * square ghost patches on glossy floors/walls (user car scenes); the
     * cut partitions the sphere, so rect sampling tiles back into the
     * continuous environment light that path tracing integrates. shape=1
     * flags the rect; pos=(phi0, dphi, cos0), sx=dcos carry it (all unused
     * for suns; fill_kernel_light routes sx through axis.w). */
    const float u0 = (float)r.x0 / width, u1 = (float)r.x1 / width;
    const float v0 = (float)r.y0 / height, v1 = (float)r.y1 / height;
    const float phi0 = M_PI_F - 2.0f * M_PI_F * u1;
    const float dphi = 2.0f * M_PI_F * (u1 - u0);
    const float ct0 = cosf(M_PI_F - M_PI_F * v0);
    const float dct = cosf(M_PI_F - M_PI_F * v1) - ct0;
    L.pos = make_float3(phi0, dphi, ct0);
    L.axis = -dir; /* centroid direction: kept for logs/fallback */
    L.p0 = std::min(std::max(sqrtf((float)solid / M_PI_F) * 0.5f, 0.002f), 0.5f);
    L.spot_cos = -1.0f;
    L.sx = dct;
    L.sy = 0.0f;
    L.shape = 1; /* world median-cut cell */
    lights.push_back(L);
    num_world_suns++;
  }
  if (num_world_suns > 0) {
    /* Debug: strongest virtual sun tells us whether a sharp sun disc in the
     * environment was captured or lost to sampling. */
    float best = 0.0f;
    float3 best_axis = make_float3(0, 0, 0);
    for (size_t i = lights.size() - num_world_suns; i < lights.size(); i++) {
      const float3 c = lights[i].color;
      const float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
      if (lum > best) {
        best = lum;
        best_axis = lights[i].axis;
      }
    }
    LOG_INFO << string_printf(
        "CyclesPlus photon map: world/HDRI -> %d virtual suns, total %.4f W/m2, "
        "strongest %.4f W/m2 from (%.2f %.2f %.2f)",
        num_world_suns,
        total,
        (double)best,
        (double)-best_axis.x,
        (double)-best_axis.y,
        (double)-best_axis.z);
  }
}

/* `kernel_trace`: the render device runs the kernel photon tracer (CUDA or
 * OptiX). Then (a) accurate-handled materials stay out of the user-facing
 * report, and (b) the host tracer's world (verts/tris/BVH) is NOT built -
 * the kernel walks the real scene BVH, aim targets come from the objects'
 * world bounds, and the pilot runs on the GPU. Skipping the host world is
 * what removes the multi-second F12 startup on heavy scenes. */
bool extract_scene(Scene *scene,
                   Progress *progress,
                   PhotonTraceScene &ps,
                   vector<PhotonLight> &lights,
                   vector<PhotonTarget> &targets,
                   vector<PhotonLight> &world_cache,
                   bool &world_cache_valid,
                   const bool kernel_trace)
{
  ps.volume_caustics = scene->integrator->get_use_photon_volume_caustics();
  int stat_objects = 0, stat_meshes = 0, stat_meshes_empty = 0, stat_lights_seen = 0,
      stat_lights_disabled = 0, stat_shaders = 0;

  /* Material report ("fail loudly"): collected over all used shaders,
   * published at the end of the extraction, shown in the caustics panel. */
  vector<PhotonMaterialReport> report;
  std::set<ustring> report_seen;

  /* Which materials may cast (selected-only mode, reflective/refractive). */
  const CasterPolicy policy = CasterPolicy::from_scene(scene);
  /* Light linking is per-triangle state the host tracer only needs when the
   * scene actually uses it - dropped again below otherwise. */
  bool any_light_linking = false;

  /* Scene bounds over every photon-relevant object: sun/world photons must
   * launch from OUTSIDE the whole scene so exterior occluders block them
   * exactly like they block path-traced rays (the old fixed ~50 m start
   * skipped anything farther out; the interim host-only 10 km start is
   * replaced by this exact per-target distance). */
  float3 scene_bmin = make_float3(1e30f, 1e30f, 1e30f);
  float3 scene_bmax = make_float3(-1e30f, -1e30f, -1e30f);
  vector<PhotonVolumeRegion> volume_regions;

  /* Geometry + materials. */
  for (Object *object : scene->objects) {
    stat_objects++;
    Geometry *geom = object->get_geometry();
    if (!geom) {
      continue;
    }

    if (geom->is_light()) {
      stat_lights_seen++;
      Light *light = static_cast<Light *>(geom);
      if (!light->get_is_enabled()) {
        stat_lights_disabled++;
        continue;
      }
      /* Per-light opt-out: this light emits no photons, so it casts no
       * caustics and its share of the budget goes to the other lights
       * (the pilot measurement only sees the lights that remain). The
       * classic interior case - a few hero lights should throw caustics,
       * the dozens of fill lights should not. */
      if (!light->get_photon_cast()) {
        stat_lights_disabled++;
        continue;
      }
      const Transform tfm = object->get_tfm();
      PhotonLight L;
      L.emitter_object = object->index;
      L.link_membership = object->get_light_set_membership();
      L.lightgroup = photon_lightgroup_index(scene, object->get_lightgroup());
      L.color = light->get_strength();
      /* A lamp's strength only carries its Color field and energy. When the
       * artist drives the lamp with nodes instead - an Emission node with a
       * coloured input, the usual way to tint a light - that colour lives in
       * the lamp's shader and never reaches get_strength(), so our photons
       * stayed white while Cycles' own render was tinted (tester report
       * 2026-08-25: red emission node, white caustic). Cycles multiplies the
       * two, so we do the same. Lamps without nodes carry a default shader
       * that evaluates to white, which leaves this a no-op; a shader with no
       * emission at all returns zero and is skipped rather than blacking the
       * light out. */
      {
        ClassifyNotes lnotes;
        Shader *lshader = light->get_shader();
        const float3 emis = shader_emission_radiance(lshader, lnotes);
        if (emis.x > 0.0f || emis.y > 0.0f || emis.z > 0.0f) {
          L.color = L.color * emis;
        }
        /* An IES profile shapes emission per direction, so it cannot be
         * folded into a single colour - the slot travels to the kernel and
         * weights each photon by the profile, exactly as svm_node_ies
         * weights the shader. */
        if (lshader && lshader->graph) {
          for (ShaderNode *n : lshader->graph->nodes) {
            if (n->is_a(IESLightNode::get_node_type())) {
              L.ies_slot = static_cast<IESLightNode *>(n)->get_ies_slot();
              break;
            }
          }
        }
      }
      L.pos = transform_get_column(&tfm, 3);
      L.axis = -normalize(transform_get_column(&tfm, 2));
      L.p0 = 0.0f;
      L.spot_cos = -1.0f;
      L.sx = L.sy = 0.0f;
      L.shape = 0;
      if (light->is_sun_light()) {
        L.type = 0;
        L.p0 = 0.5f * static_cast<SunLight *>(light)->get_angle();
      }
      else if (light->is_spot_light()) {
        L.type = 2;
        SpotLight *spot = static_cast<SpotLight *>(light);
        L.p0 = spot->get_radius();
        L.spot_cos = cosf(0.5f * spot->get_angle());
      }
      else if (light->is_point_light()) {
        L.type = 1;
        L.p0 = static_cast<PointLight *>(light)->get_radius();
      }
      else if (light->is_area_light()) {
        L.type = 3;
        AreaLight *area = static_cast<AreaLight *>(light);
        const float su = len(transform_get_column(&tfm, 0));
        const float sv = len(transform_get_column(&tfm, 1));
        L.sx = area->get_sizeu() * su;
        L.sy = area->get_sizev() * sv;
        L.shape = area->get_ellipse() ? 1 : 0;
      }
      else {
        continue; /* background/portal: not supported in the spike */
      }
      lights.push_back(L);
      continue;
    }

    if (geom->geometry_type != Geometry::MESH) {
      continue;
    }
    /* Objects that cast no shadows must not block photons either
     * (invisible helper planes, effect emitters etc.). */
    if ((object->get_visibility() & PATH_RAY_VISIBILITY_SHADOW) == 0) {
      continue;
    }
    Mesh *mesh = static_cast<Mesh *>(geom);
    stat_meshes++;
    const Attribute *attr_p = mesh->attributes.find(ATTR_STD_POSITION);
    if (!attr_p || mesh->num_triangles() == 0) {
      stat_meshes_empty++;
      continue;
    }
    const packed_float3 *mesh_verts = attr_p->data<packed_float3>();
    const size_t num_verts = attr_p->size;

    /* One photon material per used shader of this mesh. */
    vector<int32_t> shader_to_mat;
    bool any_caster = false;
    bool volume_region_recorded = false;
    for (Node *snode : mesh->get_used_shaders()) {
      stat_shaders++;
      Shader *sh = static_cast<Shader *>(snode);
      ClassifyNotes cnotes;
      const PhotonMaterial m = classify_shader(sh, policy, &cnotes);
      if (cnotes.level > 0 && report_seen.insert(sh->name).second) {
        if (m.accurate && kernel_trace) {
          /* Handled exactly by the accurate path: there is NOTHING the user
           * needs to do or know - Octane/Redshift show nothing here either,
           * and neither do we. The console keeps the trace for debugging. */
          LOG_INFO << "CyclesPlus photon map: material '" << sh->name.string()
                   << "' runs the accurate photon path (" << cnotes.reason << ")";
        }
        else {
          /* The fast approximation really applies (CPU render or accurate
           * disabled): warn, because caustics may be missing or off. */
          report.push_back({sh->name.string(), cnotes.level, cnotes.reason});
        }
      }
      shader_to_mat.push_back((int32_t)ps.mats.size());
      ps.mats.push_back(m);
      if (!volume_region_recorded && object->bounds.valid() &&
          (m.volume_sigma_s.x > 0.0f || m.volume_sigma_s.y > 0.0f ||
           m.volume_sigma_s.z > 0.0f)) {
        volume_regions.push_back({object->bounds.min,
                                  object->bounds.max,
                                  m.volume_sigma,
                                  m.volume_sigma_s,
                                  object->index,
                                  shader_to_mat.back()});
        volume_region_recorded = true;
      }
      /* Accurate materials become aim targets ONLY on hidden-specular
       * evidence (textured transmission/metallic). Making every accurate
       * material a target turned large diffuse receivers into giant aim
       * discs that soaked up virtually the whole photon budget (measured:
       * 0.2% sphere hits in force-accurate) - and the target-guiding EMA
       * cannot recover, because photons aimed at the floor that happen to
       * clip the sphere credit their yield to the floor's target. */
      const bool shader_casts = (m.kind == MAT_GLASS || m.kind == MAT_METAL) || (m.coat > 0.0f) ||
                                (m.accurate != 0 && cnotes.maybe_caster);
      any_caster |= shader_casts;

      /* Same verdict, kept per shader slot for the path tracer's side of the
       * partition. Shader::id is garbage until ShaderManager hands out the
       * ids, so validate it the way the material upload does. */
      if (ps.shader_caster.size() != scene->shaders.size()) {
        ps.shader_caster.assign(scene->shaders.size(), 0);
      }
      const int shader_slot = (int)sh->id;
      if (shader_slot >= 0 && shader_slot < (int)ps.shader_caster.size()) {
        /* A scattering medium takes the material out of the partition even
         * though it still casts photons: measured on the ocean rebuild, the
         * path tracer's share is what carries the in-scattered light, and
         * handing it over lost 45% of the lit pixels at scatter density 1.
         * Clear media - pools, drinks, glass - are unaffected. */
        ps.shader_caster[shader_slot] = (shader_casts && !cnotes.volume_scatters) ? 1 : 0;
      }
    }
    if (shader_to_mat.empty()) {
      shader_to_mat.push_back((int32_t)ps.mats.size());
      ps.mats.push_back(PhotonMaterial());
    }

    /* Mesh emitters become photon sources. The photon mode suppresses the
     * path-traced caustics of every casting material (svm/photon_caustics.h)
     * - that is only correct if every light the suppression covers also casts
     * photons, and emissive meshes did not (measured: silent light loss).
     * K area patches sampled over
     * the emissive triangles emit like small one-sided area lights along
     * the face normal; power = pi * L_e * patch area. */
    {
      vector<float3> emit_radiance;
      emit_radiance.reserve(mesh->get_used_shaders().size());
      bool any_emissive = false;
      for (Node *snode : mesh->get_used_shaders()) {
        Shader *sh = static_cast<Shader *>(snode);
        ClassifyNotes enotes;
        /* Emissive meshes ARE lights, so selected-casters mode gates them
         * with the same per-material flag: in that mode nothing produces
         * caustics unless it was selected - emissive ceiling panels are a
         * classic interior firefly source. Lamps use their own per-light
         * checkbox. */
        const bool emits = !policy.selected_only || sh->get_photon_cast();
        const float3 e = emits ? shader_emission_radiance(sh, enotes) :
                                 make_float3(0.0f, 0.0f, 0.0f);
        emit_radiance.push_back(e);
        const bool nonzero = (e.x > 0.0f || e.y > 0.0f || e.z > 0.0f);
        any_emissive |= nonzero;
        if (nonzero && enotes.level > 0 && report_seen.insert(sh->name).second) {
          report.push_back({sh->name.string(), enotes.level, enotes.reason});
        }
      }
      if (any_emissive) {
        const Transform etfm = object->get_tfm();
        const bool needs_etfm = !mesh->transform_applied;
        const size_t num_tris = mesh->num_triangles();
        const array<int> &tris = mesh->get_triangles();
        const array<int> &tri_shader = mesh->get_shader();
        vector<float> cdf;
        vector<uint32_t> tidx;
        double area_sum = 0.0;
        for (size_t t = 0; t < num_tris; t++) {
          const int sh = (t < tri_shader.size()) ? std::max(tri_shader[t], 0) : 0;
          const float3 e = emit_radiance[std::min((size_t)sh, emit_radiance.size() - 1)];
          if (e.x <= 0.0f && e.y <= 0.0f && e.z <= 0.0f) {
            continue;
          }
          float3 v0 = make_float3(mesh_verts[tris[3 * t]].x, mesh_verts[tris[3 * t]].y,
                                  mesh_verts[tris[3 * t]].z);
          float3 v1 = make_float3(mesh_verts[tris[3 * t + 1]].x, mesh_verts[tris[3 * t + 1]].y,
                                  mesh_verts[tris[3 * t + 1]].z);
          float3 v2 = make_float3(mesh_verts[tris[3 * t + 2]].x, mesh_verts[tris[3 * t + 2]].y,
                                  mesh_verts[tris[3 * t + 2]].z);
          if (needs_etfm) {
            v0 = transform_point(&etfm, v0);
            v1 = transform_point(&etfm, v1);
            v2 = transform_point(&etfm, v2);
          }
          const float area = 0.5f * len(cross(v1 - v0, v2 - v0));
          if (area <= 1e-12f) {
            continue;
          }
          area_sum += area;
          tidx.push_back((uint32_t)t);
          cdf.push_back((float)area_sum);
        }
        if (area_sum > 0.0 && !tidx.empty()) {
          const int patches = std::min<int>(16, std::max<int>(4, (int)tidx.size()));
          PhotonRNG erng;
          erng.seed(0xE317ULL + (uint64_t)stat_objects, 0x51ULL);
          for (int kp = 0; kp < patches; kp++) {
            const float r = erng.uniform() * (float)area_sum;
            size_t lo_i = 0, hi_i = tidx.size() - 1;
            while (lo_i < hi_i) {
              const size_t mid = (lo_i + hi_i) / 2;
              if (r <= cdf[mid]) {
                hi_i = mid;
              }
              else {
                lo_i = mid + 1;
              }
            }
            const size_t t = tidx[lo_i];
            float3 v0 = make_float3(mesh_verts[tris[3 * t]].x, mesh_verts[tris[3 * t]].y,
                                    mesh_verts[tris[3 * t]].z);
            float3 v1 = make_float3(mesh_verts[tris[3 * t + 1]].x, mesh_verts[tris[3 * t + 1]].y,
                                    mesh_verts[tris[3 * t + 1]].z);
            float3 v2 = make_float3(mesh_verts[tris[3 * t + 2]].x, mesh_verts[tris[3 * t + 2]].y,
                                    mesh_verts[tris[3 * t + 2]].z);
            if (needs_etfm) {
              v0 = transform_point(&etfm, v0);
              v1 = transform_point(&etfm, v1);
              v2 = transform_point(&etfm, v2);
            }
            const float su = sqrtf(erng.uniform());
            const float sv = erng.uniform();
            const float3 p = v0 * (1.0f - su) + v1 * (su * (1.0f - sv)) + v2 * (su * sv);
            const float3 n = normalize(cross(v1 - v0, v2 - v0));
            const int sh = (t < tri_shader.size()) ? std::max(tri_shader[t], 0) : 0;
            const float3 e = emit_radiance[std::min((size_t)sh, emit_radiance.size() - 1)];
            const float patch_area = (float)(area_sum / patches);
            /* Blender mesh emission is DOUBLE-SIDED: both faces radiate the
             * full radiance, so each side is its own one-sided patch. */
            for (int side = 0; side < 2; side++) {
              const float3 ns = (side == 0) ? n : -n;
              PhotonLight L;
              L.emitter_object = object->index;
              L.link_membership = object->get_light_set_membership();
              L.lightgroup = photon_lightgroup_index(scene, object->get_lightgroup());
              L.type = 3; /* area */
              L.shape = 0;
              L.pos = p + ns * 1e-4f;
              L.axis = ns;
              L.sx = L.sy = sqrtf(patch_area);
              L.color = e * ((float)M_PI * patch_area);
              L.p0 = 0.0f;
              L.spot_cos = -1.0f;
              lights.push_back(L);
            }
          }
        }
      }
    }

    if (kernel_trace) {
      /* Kernel tracer: no host world needed - the aim target comes from the
       * object's world bounds (maintained by the scene update). */
      {
        const BoundBox b = object->bounds;
        if (b.valid()) {
          scene_bmin = min(scene_bmin, b.min);
          scene_bmax = max(scene_bmax, b.max);
        }
      }
      if (any_caster) {
        const BoundBox b = object->bounds;
        if (b.valid()) {
          const float3 c = (b.min + b.max) * 0.5f;
          const float r = 0.5f * len(b.max - b.min);
          if (r > 1e-6f) {
            targets.push_back({c, r * 1.1f, r * r});
          }
        }
      }
      continue;
    }

    /* Final renders bake object transforms into the mesh vertices (static
     * BVH); the viewport does not. Applying the transform again would shift
     * every deposit off the render surface. */
    const Transform tfm = object->get_tfm();
    const bool needs_tfm = !mesh->transform_applied;
    const uint32_t vert_offset = (uint32_t)ps.verts.size();
    float3 obj_min = make_float3(1e30f, 1e30f, 1e30f), obj_max = -obj_min;
    for (size_t v = 0; v < num_verts; v++) {
      const float3 lp = make_float3(mesh_verts[v].x, mesh_verts[v].y, mesh_verts[v].z);
      const float3 wp = needs_tfm ? transform_point(&tfm, lp) : lp;
      ps.verts.push_back(wp);
      obj_min = min(obj_min, wp);
      obj_max = max(obj_max, wp);
    }
    if (num_verts > 0) {
      scene_bmin = min(scene_bmin, obj_min);
      scene_bmax = max(scene_bmax, obj_max);
    }

    const size_t num_tris = mesh->num_triangles();
    const array<int> &tris = mesh->get_triangles();
    const array<int> &tri_shader = mesh->get_shader();
    const array<bool> &tri_smooth = mesh->get_smooth();
    for (size_t t = 0; t < num_tris; t++) {
      const int sh = (t < tri_shader.size()) ? tri_shader[t] : 0;
      const int32_t mat_idx =
          shader_to_mat[std::min((size_t)std::max(sh, 0), shader_to_mat.size() - 1)];
      if (ps.mats[mat_idx].kind == MAT_SKIP) {
        continue; /* fully transparent: photons pass through */
      }
      ps.tri_v.push_back(vert_offset + (uint32_t)tris[3 * t]);
      ps.tri_v.push_back(vert_offset + (uint32_t)tris[3 * t + 1]);
      ps.tri_v.push_back(vert_offset + (uint32_t)tris[3 * t + 2]);
      ps.tri_mat.push_back(mat_idx);
      ps.tri_smooth.push_back((t < tri_smooth.size() && tri_smooth[t]) ? 1 : 0);
      ps.tri_recv_set.push_back(object->get_receiver_light_set());
    }
    any_light_linking |= object->has_light_linking();

    if (any_caster) {
      const float3 c = (obj_min + obj_max) * 0.5f;
      const float r = 0.5f * len(obj_max - obj_min);
      if (r > 1e-6f) {
        targets.push_back({c, r * 1.1f, r * r});
      }
    }
  }

  if (!any_light_linking) {
    ps.tri_recv_set.clear();
    ps.tri_recv_set.shrink_to_fit();
  }

  /* Publish the material report (also on early-out paths below: the UI
   * should reflect the latest classification even for broken setups). */
  {
    const std::lock_guard<std::mutex> lock(photon_report_mutex);
    photon_report_entries() = std::move(report);
  }

  /* World/HDRI photons: decompose the background into virtual suns. The
   * enabled-check runs every restart (world visibility can toggle without a
   * shader edit); the evaluated suns are cached until the world changes. */
  bool has_background = false;
  Object *background_object = nullptr;
  for (Object *object : scene->objects) {
    Geometry *geom = object->get_geometry();
    if (geom && geom->is_light()) {
      Light *light = static_cast<Light *>(geom);
      if (light->is_background_light() && light->get_is_enabled() &&
          light->get_photon_cast()) {
        has_background = true;
        background_object = object;
        break;
      }
    }
  }
  if (has_background && progress != nullptr) {
    if (!world_cache_valid) {
      world_cache.clear();
      eval_world_lights(scene, progress, world_cache);
      world_cache_valid = true;
    }
    /* Stamped on insertion, not in the cache: the cache survives across
     * frames, object indices do not. */
    const size_t first_world = lights.size();
    lights.insert(lights.end(), world_cache.begin(), world_cache.end());
    if (background_object != nullptr) {
      for (size_t i = first_world; i < lights.size(); i++) {
        lights[i].emitter_object = background_object->index;
        lights[i].link_membership = background_object->get_light_set_membership();
      }
    }
  }

  /* A point/area emitter can be born inside a closed volume. In that case
   * there is no volume boundary hit before the first glass event, so carry
   * the medium into the photon walk from the emission point. */
  if (ps.volume_caustics && !volume_regions.empty()) {
    for (PhotonLight &light : lights) {
      if (light.type == 0) {
        continue; /* sun/world positions are angular or synthetic */
      }
      for (const PhotonVolumeRegion &region : volume_regions) {
        const float3 p = light.pos;
        const float eps = 1e-4f;
        if (p.x >= region.bmin.x - eps && p.x <= region.bmax.x + eps &&
            p.y >= region.bmin.y - eps && p.y <= region.bmax.y + eps &&
            p.z >= region.bmin.z - eps && p.z <= region.bmax.z + eps) {
          light.initial_volume_sigma = region.sigma;
          light.initial_volume_scatter = region.scatter;
          light.initial_volume_object = region.object;
          light.initial_volume_material = region.material;
          break;
        }
      }
    }
  }

  if ((!kernel_trace && ps.tri_v.empty()) || lights.empty() || targets.empty()) {
    LOG_WARNING << string_printf(
        "CyclesPlus photon map: setup incomplete: tris=%zu usable_lights=%zu casters=%zu "
        "(objects=%d meshes=%d empty_meshes=%d lights_seen=%d lights_disabled=%d "
        "shaders=%d)",
        ps.tri_v.size() / 3,
        lights.size(),
        targets.size(),
        stat_objects,
        stat_meshes,
        stat_meshes_empty,
        stat_lights_seen,
        stat_lights_disabled,
        stat_shaders);
    return false;
  }

  /* Cumulative target weights. */
  float wsum = 0.0f;
  for (const PhotonTarget &t : targets) {
    wsum += t.wcum;
  }
  float acc = 0.0f;
  for (PhotonTarget &t : targets) {
    acc += t.wcum / wsum;
    t.wcum = acc;
  }

  /* Per-target photon launch distance from the scene bounding sphere:
   * |target center - scene center| + scene radius + margin puts the start
   * outside EVERYTHING, so exterior occluders (a roof over a window room)
   * shadow the photons exactly like they shadow camera rays. */
  if (scene_bmax.x >= scene_bmin.x) {
    const float3 scene_c = (scene_bmin + scene_bmax) * 0.5f;
    const float scene_r = 0.5f * len(scene_bmax - scene_bmin);
    for (PhotonTarget &t : targets) {
      t.start_dist = len(t.c - scene_c) + scene_r * 1.05f + 1.0f;
    }
  }

  if (!kernel_trace) {
    /* Smooth vertex normals (area-weighted face normal accumulation). */
    ps.vnormals.resize(ps.verts.size(), make_float3(0, 0, 0));
    const size_t total_tris = ps.tri_v.size() / 3;
    for (size_t t = 0; t < total_tris; t++) {
      const float3 v0 = ps.verts[ps.tri_v[3 * t]];
      const float3 v1 = ps.verts[ps.tri_v[3 * t + 1]];
      const float3 v2 = ps.verts[ps.tri_v[3 * t + 2]];
      const float3 fn = cross(v1 - v0, v2 - v0); /* length ~ 2*area */
      ps.vnormals[ps.tri_v[3 * t]] += fn;
      ps.vnormals[ps.tri_v[3 * t + 1]] += fn;
      ps.vnormals[ps.tri_v[3 * t + 2]] += fn;
    }
    for (float3 &n : ps.vnormals) {
      n = normalize(n);
    }

    /* BVH. */
    ps.centroids.resize(total_tris);
    for (size_t t = 0; t < total_tris; t++) {
      ps.centroids[t] = (ps.verts[ps.tri_v[3 * t]] + ps.verts[ps.tri_v[3 * t + 1]] +
                         ps.verts[ps.tri_v[3 * t + 2]]) *
                        (1.0f / 3.0f);
    }
    vector<uint32_t> idx(total_tris);
    for (size_t i = 0; i < total_tris; i++) {
      idx[i] = (uint32_t)i;
    }
    ps.nodes.reserve(2 * total_tris / 3 + 16);
    ps.order.reserve(total_tris);
    build_node(ps, idx.data(), (int64_t)total_tris);
    ps.centroids.clear();
  }

  return true;
}

}  // namespace

/* ------------------------------------------------------------ PhotonMap */

/* Progressive state. `front` is the grid the renderer reads; the worker
 * thread only ever writes the back buffer. Swaps happen on the session
 * thread between render works. */
struct PhotonMapData {
  /* Scene snapshot, rebuilt on restart(). */
  PhotonTraceScene scene;
  vector<PhotonLight> lights;
  vector<PhotonTarget> targets;
  /* Share of the photon budget per light, measured by a pilot batch:
   * lights whose photons actually deposit caustic flux (unoccluded, aimed
   * at working caster paths) get proportionally more photons. */
  vector<float> light_share;
  bool scene_valid = false;

  /* Progressive schedule. The gather radius stays constant: it is the grid
   * cell size and the per-pixel SPPM start radius; the actual shrinking
   * happens per pixel in the film statistics. */
  int photons_per_batch = 2000000;
  uint64_t iteration = 0; /* batches traced so far (seeds the RNG streams) */
  float radius = 0.0f;

  /* Interactive sessions ramp the batch size up over time instead of
   * tracing full batches from the start: small early batches publish
   * within fractions of a second (caustics densify immediately after an
   * edit and survive further edits), large late batches carry the full
   * quality. `photons_per_sec` is measured on every finished batch. */
  bool interactive = false;
  double photons_per_sec = 0.0;

  struct Storage {
    vector<float4> pos;
    vector<float4> beam_start;
    vector<float4> flux;
    vector<float4> beam_sigma;
    vector<int> cell_start;
    vector<float4> volume_beam_start;
    vector<float4> volume_beam_end;
    vector<float4> volume_beam_flux;
    vector<float4> volume_beam_sigma;
    vector<KernelPhotonBeamNode> volume_beam_nodes;
    PhotonGrid grid;
    bool valid = false;
    /* Device-resident staging: this generation's unsorted deposits stay in
     * VRAM until PathTrace::set_photon_grid scatters them into the render
     * device's photon arrays. Per-Storage (double buffered), because the
     * NEXT generation is traced before the current one is scattered. */
    unique_ptr<device_vector<float4>> dep_pos_dev;
    unique_ptr<device_vector<float4>> dep_beam_start_dev;
    unique_ptr<device_vector<float4>> dep_flux_dev;
    unique_ptr<device_vector<float4>> dep_beam_sigma_dev;
  };
  Storage buf[2];
  int front = 0;
  uint64_t generation = 0;

  std::thread worker;
  std::atomic<bool> batch_done{false};
  bool worker_running = false;
  /* Runs on the worker thread after each finished batch (session wake-up). */
  std::function<void()> batch_done_callback;
  /* Aborts the in-flight batch, so scene changes never wait for a full
   * batch to finish tracing. */
  std::atomic<bool> cancel{false};

  /* Cached HDRI virtual suns (median cut result). Only depends on the
   * world shader, so object/light/material edits reuse it. */
  vector<PhotonLight> world_cache;
  bool world_cache_valid = false;

  /* GPU photon tracing: batches trace on the render device (CUDA or OptiX)
   * via DEVICE_KERNEL_PHOTON_TRACE and copy the deposits back for the
   * existing host binning. CPU devices keep the host tracer. */
  Device *trace_device = nullptr;
  bool gpu_trace = false;
  /* The render device wraps more than one card (a Cycles MultiDevice). The
   * photons are traced on ONE of them (see photon_pick_trace_device) and the
   * finished map reaches all of them through the MEM_GLOBAL upload in
   * PathTrace::set_photon_grid. Device-resident binning stays off in that
   * case: it hands the raw device pointers of the render arrays to our
   * scatter kernel, and on a MultiDevice those are bookkeeping keys, not
   * addresses any card could write to. */
  bool multi_device = false;
  vector<PhotonTraceMaterial> gpu_materials;
  unique_ptr<DeviceQueue> gpu_queue;
  unique_ptr<device_vector<PhotonTraceLight>> d_lights;
  unique_ptr<device_vector<PhotonTraceTarget>> d_targets;
  unique_ptr<device_vector<PhotonTraceMaterial>> d_materials;
  unique_ptr<device_vector<float4>> d_pos;
  unique_ptr<device_vector<float4>> d_beam_start;
  unique_ptr<device_vector<float4>> d_flux;
  unique_ptr<device_vector<float4>> d_beam_sigma;
  unique_ptr<device_vector<uint>> d_counter;
  /* Device-resident binning: per-cell counts (reused as the scatter cursors
   * after the host consumed them into the prefix table). */
  unique_ptr<device_vector<uint>> d_cell_count;
  /* Multi-device binning: the sort still runs on the trace card, but its
   * RESULT has to travel through the host to reach the other cards, so the
   * scatter needs targets of its own instead of writing into the render
   * arrays. The prefix table goes back up because cursor_init reads it on
   * the device. Only allocated on a multi device. */
  unique_ptr<device_vector<float4>> d_sorted_pos;
  unique_ptr<device_vector<float4>> d_sorted_beam_start;
  unique_ptr<device_vector<float4>> d_sorted_flux;
  unique_ptr<device_vector<float4>> d_sorted_beam_sigma;
  unique_ptr<device_vector<int>> d_cell_start_dev;
  /* Target guiding (Stufe 1a): per-target deposited luminance of the current
   * generation (kernel atomicAdd) and its exponential moving average. The
   * next generation's pick weights blend pilot and EMA - exact for any
   * weights, the kernel divides by p_pick from the same table. */
  unique_ptr<device_vector<float>> d_target_yield;
  vector<float> target_yield_ema;
  /* Back-pointer for PhotonGrid::owner (scatter runs through the map's own
   * queue and staging buffers when the grid is bound to the render). */
  PhotonMap *self = nullptr;

  /* Async GPU generation (interactive viewport): chunks are enqueued between
   * render works WITHOUT synchronizing - the photon queue has its own
   * non-blocking CUDA stream, so tracing overlaps rendering and fills the
   * viewport's idle gaps. All host calls stay on the render thread (the
   * v0.5 lesson: a worker thread poisons the context, streams do not).
   * Synchronize before ANY buffer reset while inflight. */
  bool gpu_inflight = false;
  int gpu_off = 0;
  int gpu_total = 0;
  int gpu_num_photons = 0;
  float gpu_radius = 0.0f;
  int gpu_batch_k = 0;
  double gpu_begin_time = 0.0;
  /* OptiX viewport throttle (2026-08-05, OPT-IN since 2026-08-07): smaller
   * launches plus a breather between generations, built while the driver
   * deaths were unexplained. The root cause is fixed at the source now, so
   * the throttle is off by default (full speed is the product) and only
   * CYCLESPLUS_PHOTON_OPTIX_THROTTLE=1 turns it back on. Viewport-only -
   * F12 pacing stays deterministic either way. */
  bool optix_device = false;
  double optix_next_launch = 0.0;
};

namespace {

/* Pick the card the photon tracer runs on, and report how many cards the
 * render device covers.
 *
 * When several devices are enabled Cycles hands us a MultiDevice wrapper.
 * That wrapper renders by splitting the image across its sub-devices, but it
 * owns no command queue itself - Device::gpu_queue_create() is a LOG_FATAL in
 * the base class, so asking the wrapper for one takes the whole process down
 * (user report 2026-08-18, a 4-5 card rig died the moment the checkbox was
 * ticked; enabling a single card was the workaround). Note that info.type is
 * NOT a usable test for this: Device::get_multi_device only stamps
 * DEVICE_MULTI when the selected cards have DIFFERENT types, so a rig of
 * identical OptiX cards still reports DEVICE_OPTIX while being a MultiDevice
 * underneath. foreach_device is the honest question - it visits the real
 * cards on a multi device and the device itself on a single one.
 *
 * Tracing on one card costs nothing: the tracer runs at 100M+ photons/s and
 * has never been the bottleneck, while the gather - the part that scales with
 * resolution and samples - keeps running on every card over its own slice of
 * the image. The finished map travels to all of them as a MEM_GLOBAL upload,
 * the same route geometry and textures take. Same "first sub-device" rule
 * ShaderEval already applies for background evaluation. */
Device *photon_pick_trace_device(Device *render_device, int &num_devices)
{
  num_devices = 0;
  if (render_device == nullptr) {
    return nullptr;
  }

  Device *pick = nullptr;
  render_device->foreach_device([&](Device *sub) {
    num_devices++;
    /* Prefer a GPU: on a mixed CPU+GPU rig the photons still belong on the
     * card. Sub-devices come in the multi device's own order otherwise. */
    if (pick == nullptr || (pick->info.type == DEVICE_CPU && sub->info.type != DEVICE_CPU)) {
      pick = sub;
    }
  });

  return pick;
}

uint64_t batch_seed(const uint64_t k)
{
  return 0x9E3779B97F4A7C15ULL + k * 0x2545F4914F6CDD1DULL;
}

/* Full speed is the product default (2026-08-07): the driver deaths the
 * throttle guarded against were root-caused to the in-flight photon stream
 * racing scene updates, fixed at the source (abort_inflight before
 * update_scene). CYCLESPLUS_PHOTON_OPTIX_THROTTLE=1 re-enables the gentle
 * pacing (chunk shrink + generation breather) as a support fallback for
 * machines that still misbehave. */
bool optix_fullbore()
{
  static const bool throttle = getenv("CYCLESPLUS_PHOTON_OPTIX_THROTTLE") != nullptr;
  return !throttle;
}

/* Photons for batch k (1-based). Sizes depend ONLY on the batch index -
 * never on measured timings (see the feedback trap described below and the
 * animation flicker it caused in the F12 path). The viewport doubles from a
 * small preview up to the steady size, final renders jump there directly.
 * Every batch is a complete unbiased estimate regardless of its size, so
 * the SPPM statistics can mix them freely. */
int batch_size(const PhotonMapData &d, const uint64_t k)
{
  if (!d.interactive && !d.gpu_trace) {
    return d.photons_per_batch;
  }
  const int preview = std::min(d.photons_per_batch, 262144);
  /* Where the VIEWPORT ramp starts. The k=1 batch is traced synchronously
   * and blocks the viewport start, so it has to be cheap - but "cheap"
   * means something entirely different per device. 262144 was sized for the
   * host tracer; on the GPU it is ~2ms of work and lands ~12k deposits
   * scene-wide, which at any real Detail setting is invisible. So the first
   * frame showed nothing and the caustic only appeared once the ramp had
   * climbed - "the first caustics take longer". 2M photons cost ~15ms on
   * the GPU (imperceptible at startup) and land ~91k deposits: a visible
   * caustic in the FIRST frame. Kept separate from `preview` so the F12
   * ladder below is untouched. */
  const int vp_first = std::min(d.photons_per_batch,
                                d.gpu_trace ? 2 * 1024 * 1024 : 262144);
  if (d.interactive && k <= 1) {
    return vp_first;
  }
  /* Final renders (F12) use a DETERMINISTIC size ladder: the timed budget
   * below sizes batches from the measured photons_per_sec, which varies
   * run-to-run (and carries across animation frames), shifting every
   * light's photon index range and re-rolling all paths - single frames in
   * a sequence render visibly brighter/darker at random (animation
   * flicker; proven by repeat-render A/B where fixed-size generation 1
   * was bit-identical and all timed generations diverged). Sizes must
   * depend only on the batch index. CYCLESPLUS_PHOTON_TIMED_BATCHES=1
   * restores the old behavior for bisections. */
  static const bool timed_batches = getenv("CYCLESPLUS_PHOTON_TIMED_BATCHES") != nullptr;
  if (!timed_batches) {
    /* ppb/8 matches the throughput the 0.25s budget settled at on the
     * reference GPU (~ppb/8.5 at high counts), so wall-clock stays level;
     * a full batch per generation would be 8x slower at high counts. */
    const int full = d.photons_per_batch;
    const int steady = std::max(preview, full / 8);
    if (d.interactive) {
      /* The viewport ramps by DOUBLING per generation, also index-only.
       * Sizing it from the measured rate was a feedback trap: an async
       * generation's elapsed time spans the render works it overlaps, so
       * the rate reads ~200x below the real one (measured: 690k/s against
       * an actual 130M/s), which shrinks the next batch, which reads lower
       * still. Measured on room.blend: the ramp peaked at 2.2M photons and
       * spiralled to the 262k floor within 8 generations, then stayed
       * there - 48x below F12's batches. At high Detail the gather discs
       * are then mostly empty, so the caustic collapses towards zero and
       * the viewport "stays dark until you toggle the checkbox" (the
       * toggle restarts with a synchronous batch, whose timing is pure
       * trace time and therefore correct). */
      /* Quadruple per step, not double: k=1 already published a synchronous
       * preview, so there IS an image on screen and the next steps should
       * chase full strength rather than creep. Doubling reached the steady
       * size at k=7 (~1.3s); quadrupling gets there at k=4 (~0.5s), and the
       * first refinement is 4x stronger - which is what "instant caustics"
       * actually looks like. */
      const int shift = (int)std::min(2 * (k - 1), (uint64_t)20);
      return (int)std::min((int64_t)vp_first << shift, (int64_t)steady);
    }
    const int first = std::max(preview, full / 16);
    if (k <= 1) {
      return first; /* fast visible start, still deterministic */
    }
    return steady;
  }
  /* GPU batches run synchronously between render works: cap them at a
   * fraction of a second so the viewport stays fluid (the SPPM average
   * makes many small generations equivalent to few large ones). */
  const double budget_cap = d.gpu_trace ? 0.25 : 4.0;
  const int exp = (int)std::min(k >= 2 ? k - 2 : 0, (uint64_t)4);
  const double budget = std::min(0.25 * (double)(1 << exp), budget_cap);
  const double pps = d.photons_per_sec > 0.0 ? d.photons_per_sec : 2.0e6;
  const int64_t n = (int64_t)(budget * pps);
  return (int)std::max((int64_t)std::min(preview, d.photons_per_batch),
                       std::min(n, (int64_t)d.photons_per_batch));
}

/* Weight of a batch relative to the full batch size, in 1/16th steps. */
int batch_weight16(const PhotonMapData &d, const int num_photons)
{
  const int w = (int)roundf(16.0f * (float)num_photons / (float)std::max(d.photons_per_batch, 1));
  return std::max(1, std::min(w, 16));
}

/* Photon budget per light. Uniform splitting starves the one important sun
 * in scenes with many weak environment lights, so the shares come from a
 * pilot measurement of the caustic flux each light actually delivers
 * (occlusion and caster geometry included). Here: floor every share so no
 * light starves completely, then normalize to one. */
void normalize_light_shares(vector<float> &share)
{
  double total = 0.0;
  for (const float s : share) {
    total += s;
  }
  const float floor_share = (float)(total > 0.0 ? total : 1.0) * 0.002f;
  float sum = 0.0f;
  for (float &s : share) {
    s += floor_share;
    sum += s;
  }
  for (float &s : share) {
    s /= sum;
  }
}

void measure_light_shares(const PhotonTraceScene &ps,
                          const vector<PhotonLight> &lights,
                          const vector<PhotonTarget> &targets,
                          vector<float> &share,
                          const std::atomic<bool> *cancel)
{
  const int pilot_photons = 20000;
  share.assign(lights.size(), 0.0f);

  /* Parallel over lights (HDRI worlds bring up to 128 virtual suns and the
   * pilot runs on every restart). Seeds are per light, so the shares are
   * identical to the serial measurement. */
  int n_threads = (int)std::thread::hardware_concurrency();
  n_threads = std::max(1, std::min({n_threads, (int)lights.size(), 64}));
  std::atomic<size_t> next_light{0};
  vector<std::thread> pool;
  for (int th = 0; th < n_threads; th++) {
    pool.emplace_back([&]() {
      for (;;) {
        const size_t li = next_light.fetch_add(1);
        if (li >= lights.size()) {
          return;
        }
        vector<PhotonDeposit> pilot;
        emit_photons(ps,
                     lights[li],
                     targets,
                     0,
                     pilot_photons,
                     pilot_photons,
                     12,
                     0xC0FFEEULL,
                     li,
                     cancel,
                     pilot);
        double flux = 0.0;
        for (const PhotonDeposit &d : pilot) {
          flux += 0.2126f * d.flux.x + 0.7152f * d.flux.y + 0.0722f * d.flux.z;
        }
        share[li] = (float)flux;
      }
    });
  }
  for (std::thread &t : pool) {
    t.join();
  }

  normalize_light_shares(share);
}

/* Target guiding (Stufe 1a): pick weights for the aim targets = pilot blend
 * with the yield EMA. CYCLESPLUS_PHOTON_GUIDE_DISABLE keeps the pilot only.
 * The uniform pilot share (alpha) keeps every target alive (defensive
 * sampling: new caustic paths after edits are still discovered). */
vector<PhotonTraceTarget> build_guided_targets(PhotonMapData &d)
{
  static const bool guide_disable = getenv("CYCLESPLUS_PHOTON_GUIDE_DISABLE") != nullptr;
  const size_t n = d.targets.size();
  vector<PhotonTraceTarget> targets(n);
  vector<float> w(n);
  float prev = 0.0f;
  for (size_t i = 0; i < n; i++) {
    w[i] = std::max(d.targets[i].wcum - prev, 1e-6f);
    prev = d.targets[i].wcum;
  }
  if (!guide_disable && d.target_yield_ema.size() == n) {
    float ysum = 0.0f;
    for (const float y : d.target_yield_ema) {
      ysum += y;
    }
    if (ysum > 0.0f) {
      /* Damped blend PLUS an exploration floor: without the floor the
       * yield adaptation can lock itself out of a small target (few
       * photons aimed -> low measured yield -> even fewer photons), which
       * showed as per-frame caustic brightness jumps in animations (chair
       * scene: single frames -24% while GUIDE_DISABLE capped the same
       * jump at -8%). No bias either way - the estimator divides by the
       * pick probability; weights only shape variance. */
      const float alpha = 0.5f;
      for (size_t i = 0; i < n; i++) {
        const float blended = alpha * w[i] + (1.0f - alpha) * (d.target_yield_ema[i] / ysum);
        w[i] = std::max(blended, 0.35f * w[i]);
      }
    }
  }
  float sum = 0.0f;
  for (const float x : w) {
    sum += x;
  }
  float cum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    cum += w[i] / std::max(sum, 1e-12f);
    targets[i].c_r = make_float4(d.targets[i].c.x, d.targets[i].c.y, d.targets[i].c.z,
                                 d.targets[i].r);
    targets[i].wcum = (i + 1 == n) ? 1.0f : cum;
    targets[i].start_dist = d.targets[i].start_dist;
    targets[i].pad1 = targets[i].pad2 = 0.0f;
  }
  return targets;
}

/* Fill one kernel light record covering photon index range [start, end). */
void fill_kernel_light(const PhotonLight &L,
                       PhotonTraceLight &kl,
                       const int index_start,
                       const int index_end)
{
  kl.type = L.type;
  kl.shape = L.shape;
  kl.p0 = L.p0;
  kl.spot_cos = L.spot_cos;
  kl.index_start = index_start;
  kl.index_end = index_end;
  kl.n_total = index_end - index_start;
  kl.emitter_object = L.emitter_object;
  kl.color = make_float4(L.color.x, L.color.y, L.color.z, 0.0f);
  kl.pos = make_float4(L.pos.x, L.pos.y, L.pos.z, 0.0f);
  kl.axis = make_float4(L.axis.x, L.axis.y, L.axis.z, L.sx);
  /* extra.y = light group index, -1 = none (see PhotonTraceLight). */
  kl.extra = make_float4(L.sy, (float)L.lightgroup, (float)L.ies_slot, 0.0f);
  kl.initial_volume_sigma = make_float4(L.initial_volume_sigma.x,
                                        L.initial_volume_sigma.y,
                                        L.initial_volume_sigma.z,
                                        __int_as_float(L.initial_volume_object));
  kl.initial_volume_scatter = make_float4(L.initial_volume_scatter.x,
                                          L.initial_volume_scatter.y,
                                          L.initial_volume_scatter.z,
                                          0.0f);
}

/* Drain the null stream before launching photon kernels. Pageable HtoD
 * copies (all our zero/param uploads) may return once the data reaches the
 * driver staging buffer while the DMA to the device is still in flight on
 * the NULL stream - and our CU_STREAM_NON_BLOCKING queue does not order
 * against the null stream, so a kernel can race the tail of an upload
 * (residual bin-count losses / misplaced scatters after the big memset-race
 * fix, 2026-07-24). A pageable DtoH copy is stream-ordered after every
 * prior null-stream op and only returns once complete: reading the 4-byte
 * counter is a microsecond-cheap full barrier. */
void drain_null_stream(PhotonMapData &d)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.drain_null_stream", &d);
  if (d.d_counter && d.d_counter->device_pointer) {
    d.d_counter->copy_from_device();
  }
}

/* Zero the per-generation target yield accumulator (tiny upload). */
void arm_target_yield(PhotonMapData &d)
{
  const size_t n = d.targets.size();
  if (n == 0) {
    return;
  }
  if (!d.d_target_yield) {
    d.d_target_yield = make_unique<device_vector<float>>(d.trace_device, "photon_target_yield",
                                                         MEM_READ_WRITE);
  }
  memset(d.d_target_yield->alloc(n), 0, sizeof(float) * n);
  d.d_target_yield->copy_to_device();
}

/* Fold the generation's yield into the EMA. */
void harvest_target_yield(PhotonMapData &d)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.target_yield_readback", &d);
  const size_t n = d.targets.size();
  if (n == 0 || !d.d_target_yield || d.d_target_yield->size() != n) {
    return;
  }
  d.d_target_yield->copy_from_device();
  if (d.target_yield_ema.size() != n) {
    d.target_yield_ema.assign(n, 0.0f);
  }
  double yield_sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    yield_sum += d.d_target_yield->data()[i];
    d.target_yield_ema[i] = 0.8f * d.target_yield_ema[i] + 0.2f * d.d_target_yield->data()[i];
  }
  /* Energy diagnostic: total deposited luminance this generation. Guiding
   * must keep this constant in expectation - a drop under a concentrated
   * histogram means the pdf compensation is off. */
  LOG_INFO << "CyclesPlus photon map: generation deposited lum " << yield_sum;
}

/* GPU pilot: per-light yield via one tiny kernel launch per light (see
 * pilot_photons below; deposit capacity 1 - overflow is counted, not
 * written, only the target_yield sum is read back). Replaces the host pilot
 * on kernel-traced
 * devices, so GPU renders no longer build the host tracer's world (the
 * multi-second extraction/BVH startup the profiling showed). Unbiased for
 * any resulting shares: every light normalizes flux by its own count. */
bool measure_light_shares_gpu(PhotonMapData &d)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.gpu_light_pilot", &d);
  const int pilot_photons = 20000;
  const size_t num_lights = d.lights.size();
  d.light_share.assign(num_lights, 0.0f);
  if (num_lights == 0 || d.targets.empty() || d.trace_device == nullptr) {
    return false;
  }
  Device *device = d.trace_device;
  if (!d.gpu_queue) {
    d.gpu_queue = device->gpu_queue_create();
    if (!d.gpu_queue) {
      return false;
    }
    d.gpu_queue->init_execution();
  }
  if (!d.d_lights) {
    d.d_lights = make_unique<device_vector<PhotonTraceLight>>(
        device, "photon_trace_lights", MEM_READ_ONLY);
    d.d_targets = make_unique<device_vector<PhotonTraceTarget>>(
        device, "photon_trace_targets", MEM_READ_ONLY);
    d.d_materials = make_unique<device_vector<PhotonTraceMaterial>>(
        device, "photon_trace_materials", MEM_READ_ONLY);
    d.d_counter = make_unique<device_vector<uint>>(device, "photon_trace_counter", MEM_READ_WRITE);
    d.d_counter->alloc(1);
    d.d_counter->data()[0] = 0;
    d.d_counter->copy_to_device();
  }

  /* Throwaway 1-slot deposit staging, freed at scope end. */
  device_vector<float4> pilot_pos(device, "photon_pilot_pos", MEM_READ_WRITE);
  device_vector<float4> pilot_beam_start(device, "photon_pilot_beam_start", MEM_READ_WRITE);
  device_vector<float4> pilot_flux(device, "photon_pilot_flux", MEM_READ_WRITE);
  device_vector<float4> pilot_beam_sigma(device, "photon_pilot_beam_sigma", MEM_READ_WRITE);
  memset(pilot_pos.alloc(1), 0, sizeof(float4));
  pilot_pos.copy_to_device();
  memset(pilot_beam_start.alloc(1), 0, sizeof(float4));
  pilot_beam_start.copy_to_device();
  memset(pilot_flux.alloc(1), 0, sizeof(float4));
  pilot_flux.copy_to_device();
  memset(pilot_beam_sigma.alloc(1), 0, sizeof(float4));
  pilot_beam_sigma.copy_to_device();

  vector<PhotonTraceTarget> targets = build_guided_targets(d);
  memcpy(d.d_targets->alloc(targets.size()), targets.data(),
         sizeof(PhotonTraceTarget) * targets.size());
  d.d_targets->copy_to_device();
  memcpy(d.d_materials->alloc(d.gpu_materials.size()), d.gpu_materials.data(),
         sizeof(PhotonTraceMaterial) * d.gpu_materials.size());
  d.d_materials->copy_to_device();

  const device_ptr pos_ptr = pilot_pos.device_pointer;
  const device_ptr beam_start_ptr = pilot_beam_start.device_pointer;
  const device_ptr flux_ptr = pilot_flux.device_pointer;
  const device_ptr beam_sigma_ptr = pilot_beam_sigma.device_pointer;
  const device_ptr counter_ptr = d.d_counter->device_pointer;
  const device_ptr targets_ptr = d.d_targets->device_pointer;
  const device_ptr mats_ptr = d.d_materials->device_pointer;
  const int one = 1;
  const int num_targets = (int)targets.size();
  const int zero_off = 0;
  const int max_bounces = 12;
  const int debug_mode = 0;
  const int capacity = 1;
  const int n = pilot_photons;

  for (size_t li = 0; li < num_lights; li++) {
    PhotonTraceLight kl;
    fill_kernel_light(d.lights[li], kl, 0, pilot_photons);
    memcpy(d.d_lights->alloc(1), &kl, sizeof(kl));
    d.d_lights->copy_to_device();
    d.d_counter->data()[0] = 0;
    d.d_counter->copy_to_device();
    arm_target_yield(d);
    drain_null_stream(d);
    const device_ptr lights_ptr = d.d_lights->device_pointer;
    const device_ptr yield_ptr = d.d_target_yield ? d.d_target_yield->device_pointer : 0;
    /* Distinct seed stream per light (the kernel seeds by in-launch light
     * index, which is always 0 here). */
    const int batch_k = 1000000 + (int)li;
    const DeviceKernelArguments args(&pos_ptr,
                                     &beam_start_ptr,
                                     &flux_ptr,
                                     &beam_sigma_ptr,
                                     &counter_ptr,
                                     &lights_ptr,
                                     &one,
                                     &targets_ptr,
                                     &num_targets,
                                     &mats_ptr,
                                     &batch_k,
                                     &zero_off,
                                     &max_bounces,
                                     &debug_mode,
                                     &capacity,
                                     &n,
                                     &yield_ptr);
    if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_TRACE, pilot_photons, args)) {
      return false;
    }
    if (!d.gpu_queue->synchronize()) {
      return false;
    }
    d.d_target_yield->copy_from_device();
    double flux = 0.0;
    for (int t = 0; t < num_targets; t++) {
      flux += d.d_target_yield->data()[t];
    }
    d.light_share[li] = (float)flux;
  }
  normalize_light_shares(d.light_share);
  return true;
}

void build_volume_beam_bvh(const float4 *dep_pos,
                           const float4 *dep_beam_start,
                           const float4 *dep_flux,
                           const float4 *dep_sigma,
                           const size_t total,
                           const float radius,
                           PhotonMapData::Storage &out)
{
  CCL_PHOTON_PROFILE_SCOPE("volume.beam_bvh_build", &out, total);
  out.volume_beam_start.clear();
  out.volume_beam_end.clear();
  out.volume_beam_flux.clear();
  out.volume_beam_sigma.clear();
  out.volume_beam_nodes.clear();
  for (size_t i = 0; i < total; i++) {
    if (photon_deposit_is_volume(dep_flux[i].w)) {
      const float3 a = make_float3(dep_beam_start[i].x, dep_beam_start[i].y, dep_beam_start[i].z);
      const float3 b = make_float3(dep_pos[i].x, dep_pos[i].y, dep_pos[i].z);
      const float3 extent = fabs(b - a);
      const float minor_extent = extent.x + extent.y + extent.z - reduce_max(extent) - reduce_min(extent);
      /* Diced beam bounds, as in Tungsten's photon mapper. The finite cylinder
       * pieces are disjoint; w retains optical distance from the original start. */
      const int pieces = (int)clamp(ceilf(minor_extent / (8.0f * photon_safe_radius(radius))), 1.0f, 16.0f);
      const float beam_length = len(b - a);
      for (int part = 0; part < pieces; part++) {
        const float t0 = (float)part / pieces;
        const float t1 = (float)(part + 1) / pieces;
        const float3 start = a + (b - a) * t0;
        const float3 end = a + (b - a) * t1;
        out.volume_beam_start.push_back(make_float4(start.x, start.y, start.z, t0 * beam_length));
        out.volume_beam_end.push_back(make_float4(end.x, end.y, end.z, beam_length / pieces));
        out.volume_beam_flux.push_back(dep_flux[i]);
        out.volume_beam_sigma.push_back(dep_sigma[i]);
      }
    }
  }
  if (out.volume_beam_start.empty()) {
    return;
  }

  /* Cache beam bounds once. Recursive BVH construction revisits each beam at
   * every ancestor node; recomputing these values there made construction
   * unnecessarily O(n log n) in geometry loads and min/max operations. */
  vector<float3> beam_bounds_min(out.volume_beam_start.size());
  vector<float3> beam_bounds_max(out.volume_beam_start.size());
  vector<float3> beam_centers(out.volume_beam_start.size());
  const float beam_radius = photon_safe_radius(radius);
  for (size_t i = 0; i < out.volume_beam_start.size(); i++) {
    const float3 a = make_float3(out.volume_beam_start[i].x,
                                 out.volume_beam_start[i].y,
                                 out.volume_beam_start[i].z);
    const float3 b = make_float3(
        out.volume_beam_end[i].x, out.volume_beam_end[i].y, out.volume_beam_end[i].z);
    beam_bounds_min[i] = min(a, b) - make_float3(beam_radius);
    beam_bounds_max[i] = max(a, b) + make_float3(beam_radius);
    beam_centers[i] = (beam_bounds_min[i] + beam_bounds_max[i]) * 0.5f;
  }

  vector<int> order(out.volume_beam_start.size());
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = (int)i;
  }
  auto bounds_for = [&](const int index, float3 &bmin, float3 &bmax) {
    bmin = beam_bounds_min[index];
    bmax = beam_bounds_max[index];
  };
  std::function<int(size_t, size_t)> build = [&](const size_t begin, const size_t end) {
    const int node_index = (int)out.volume_beam_nodes.size();
    out.volume_beam_nodes.push_back({});
    float3 bmin = make_float3(1e30f), bmax = make_float3(-1e30f);
    float3 cmin = make_float3(1e30f), cmax = make_float3(-1e30f);
    for (size_t i = begin; i < end; i++) {
      float3 beam_min, beam_max;
      bounds_for(order[i], beam_min, beam_max);
      bmin = min(bmin, beam_min);
      bmax = max(bmax, beam_max);
      const float3 center = beam_centers[order[i]];
      cmin = min(cmin, center);
      cmax = max(cmax, center);
    }
    out.volume_beam_nodes[node_index].bmin = bmin;
    out.volume_beam_nodes[node_index].bmax = bmax;
    const size_t count = end - begin;
    /* Balance traversal depth against leaf scans without changing beam geometry or flux. */
    if (count <= 8) {
      out.volume_beam_nodes[node_index].left = -((int)begin + 1);
      out.volume_beam_nodes[node_index].right = (int)count;
      return node_index;
    }
    const float3 extent = cmax - cmin;
    int axis = 0;
    if (extent.y > extent.x) {
      axis = 1;
    }
    if ((axis == 0 ? extent.x : extent.y) < extent.z) {
      axis = 2;
    }
    const size_t mid = begin + count / 2;
    std::nth_element(order.begin() + begin,
                     order.begin() + mid,
                     order.begin() + end,
                     [&](const int a, const int b) {
                       float3 amin, amax, bmin2, bmax2;
                       bounds_for(a, amin, amax);
                       bounds_for(b, bmin2, bmax2);
                       const float3 ca = (amin + amax) * 0.5f;
                       const float3 cb = (bmin2 + bmax2) * 0.5f;
                       return (axis == 0 ? ca.x : axis == 1 ? ca.y : ca.z) <
                              (axis == 0 ? cb.x : axis == 1 ? cb.y : cb.z);
                     });
    const int left = build(begin, mid);
    const int right = build(mid, end);
    out.volume_beam_nodes[node_index].left = left;
    out.volume_beam_nodes[node_index].right = right;
    return node_index;
  };
  build(0, order.size());

  vector<float4> sorted_start, sorted_end, sorted_flux, sorted_sigma;
  sorted_start.reserve(order.size());
  sorted_end.reserve(order.size());
  sorted_flux.reserve(order.size());
  sorted_sigma.reserve(order.size());
  for (const int index : order) {
    sorted_start.push_back(out.volume_beam_start[index]);
    sorted_end.push_back(out.volume_beam_end[index]);
    sorted_flux.push_back(out.volume_beam_flux[index]);
    sorted_sigma.push_back(out.volume_beam_sigma[index]);
  }
  out.volume_beam_start.swap(sorted_start);
  out.volume_beam_end.swap(sorted_end);
  out.volume_beam_flux.swap(sorted_flux);
  out.volume_beam_sigma.swap(sorted_sigma);
  LOG_INFO << "CyclesPlus volume beams: " << order.size() << " segments, "
           << out.volume_beam_nodes.size() << " BVH nodes";
  photon_profile_value("volume.built_segments", &out, order.size());
  photon_profile_value("volume.built_nodes", &out, out.volume_beam_nodes.size());
}

/* Bin packed deposits (pos.w = packed normal) into the spatial hash. Shared
 * by the CPU tracer and the GPU trace-and-copy-back path. */
size_t bin_packed_deposits(const float4 *dep_pos,
                           const float4 *dep_beam_start,
                           const float4 *dep_flux,
                           const float4 *dep_sigma,
                           const size_t total,
                           const float radius,
                           PhotonMapData::Storage &out)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.cpu_binning", &out, total);
  if (total == 0) {
    return 0;
  }
  const float safe_radius = photon_safe_radius(radius);
  uint table_size = 64;
  while (table_size < total / 2) {
    table_size <<= 1;
  }
  const float inv_cell = 1.0f / safe_radius;

  out.cell_start.clear();
  out.cell_start.resize(table_size + 1, 0);
  auto bucket_of = [&](const float4 p) {
    return photon_grid_hash((int)floorf(p.x * inv_cell),
                            (int)floorf(p.y * inv_cell),
                            (int)floorf(p.z * inv_cell),
                            table_size);
  };
  for (size_t i = 0; i < total; i++) {
    out.cell_start[bucket_of(dep_pos[i]) + 1]++;
  }
  for (uint b = 0; b < table_size; b++) {
    out.cell_start[b + 1] += out.cell_start[b];
  }
  out.pos.resize(total);
  out.beam_start.resize(total);
  out.flux.resize(total);
  out.beam_sigma.resize(total);
  vector<int> cursor(out.cell_start.begin(), out.cell_start.end() - 1);
  for (size_t i = 0; i < total; i++) {
    const int at = cursor[bucket_of(dep_pos[i])]++;
    out.pos[at] = dep_pos[i];
    out.beam_start[at] = dep_beam_start[i];
    out.flux[at] = dep_flux[i];
    out.beam_sigma[at] = dep_sigma[i];
  }

  build_volume_beam_bvh(dep_pos, dep_beam_start, dep_flux, dep_sigma, total, safe_radius, out);

  out.grid.pos = out.pos.data();
  out.grid.beam_start = out.beam_start.data();
  out.grid.flux = out.flux.data();
  out.grid.beam_sigma = out.beam_sigma.data();
  out.grid.cell_start = out.cell_start.data();
  out.grid.num_photons = (int)total;
  out.grid.table_size = table_size;
  out.grid.radius = safe_radius;
  out.grid.inv_cell = inv_cell;
  out.grid.volume_beam_start = out.volume_beam_start.data();
  out.grid.volume_beam_end = out.volume_beam_end.data();
  out.grid.volume_beam_flux = out.volume_beam_flux.data();
  out.grid.volume_beam_sigma = out.volume_beam_sigma.data();
  out.grid.volume_beam_nodes = out.volume_beam_nodes.data();
  out.grid.num_volume_beams = (int)out.volume_beam_start.size();
  out.grid.num_volume_beam_nodes = (int)out.volume_beam_nodes.size();
  /* Host-resident: clear any device-resident state a previous generation
   * left in this reused Storage. */
  out.grid.device_resident = 0;
  out.grid.dep_pos_device = 0;
  out.grid.dep_beam_start_device = 0;
  out.grid.dep_flux_device = 0;
  out.grid.dep_beam_sigma_device = 0;
  out.grid.owner = nullptr;
  out.valid = true;
  return total;
}

/* Multi-device binning: sort on the trace card, hand the host the result.
 *
 * A multi device cannot keep the map in VRAM - the gather runs on every card
 * and PathTrace can only reach them all through a MEM_GLOBAL upload from
 * host memory. The obvious answer, binning on the host, turned out to cost
 * 3.5x the photon time (1.42s -> 4.96s measured 2026-08-25): the copy is
 * cheap once it is bounded by the deposit count, but bin_packed_deposits
 * walks millions of deposits twice, single-threaded, the second pass with
 * random-access writes.
 *
 * So the sort stays where it belongs. This runs the same count/cursor/
 * scatter kernels the single-device path runs, just into buffers of our own
 * instead of the render arrays, and copies the SORTED result back. The host
 * then owns a plain array pair and uploads it the ordinary way. One PCIe
 * round trip per generation is unavoidable here - the data has to reach the
 * other cards somehow - but the sorting does not have to happen on the CPU.
 *
 * The prefix table travels back up because cursor_init reads it device-side;
 * it is table_size ints, negligible next to the deposits. */
bool scatter_device_to_host(PhotonMapData &d,
                             PhotonMapData::Storage &out,
                             const device_ptr dep_pos_ptr,
                             const device_ptr dep_beam_start_ptr,
                             const device_ptr dep_flux_ptr,
                             const device_ptr dep_beam_sigma_ptr,
                             const int total,
                            const uint table_size,
                            const float inv_cell)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.scatter_and_readback", &d, total);
  Device *device = d.trace_device;
  if (device == nullptr || !d.gpu_queue || !d.d_cell_count) {
    return false;
  }

  if (!d.d_sorted_pos) {
    d.d_sorted_pos = make_unique<device_vector<float4>>(
        device, "photon_sorted_pos", MEM_READ_WRITE);
    d.d_sorted_beam_start = make_unique<device_vector<float4>>(
        device, "photon_sorted_beam_start", MEM_READ_WRITE);
    d.d_sorted_flux = make_unique<device_vector<float4>>(
        device, "photon_sorted_flux", MEM_READ_WRITE);
    d.d_sorted_beam_sigma = make_unique<device_vector<float4>>(
        device, "photon_sorted_beam_sigma", MEM_READ_WRITE);
    d.d_cell_start_dev = make_unique<device_vector<int>>(
        device, "photon_cell_start_dev", MEM_READ_WRITE);
  }
  /* High-water capacity, and host-zero + copy_to_device rather than
   * zero_to_device - see the memset-race notes in trace_batch_gpu. Growing
   * only upwards keeps the allocation warm and skips the per-generation
   * cuMemFree, which synchronizes the whole context. */
  if (d.d_sorted_pos->size() < (size_t)total) {
    const size_t cap = (size_t)total + (size_t)total / 4;
    memset(d.d_sorted_pos->alloc(cap), 0, sizeof(float4) * cap);
    d.d_sorted_pos->copy_to_device();
    memset(d.d_sorted_beam_start->alloc(cap), 0, sizeof(float4) * cap);
    d.d_sorted_beam_start->copy_to_device();
    memset(d.d_sorted_flux->alloc(cap), 0, sizeof(float4) * cap);
    d.d_sorted_flux->copy_to_device();
    memset(d.d_sorted_beam_sigma->alloc(cap), 0, sizeof(float4) * cap);
    d.d_sorted_beam_sigma->copy_to_device();
  }
  const size_t start_cap = std::max((size_t)table_size + 1, d.d_cell_start_dev->size());
  int *start_dst = d.d_cell_start_dev->alloc(start_cap);
  memcpy(start_dst, out.cell_start.data(), sizeof(int) * (table_size + 1));
  d.d_cell_start_dev->copy_to_device();
  drain_null_stream(d); /* prefix fully landed before cursor_init reads it */

  const device_ptr start_ptr = d.d_cell_start_dev->device_pointer;
  const device_ptr cursor_ptr = d.d_cell_count->device_pointer;
  const device_ptr out_pos = d.d_sorted_pos->device_pointer;
  const device_ptr out_beam_start = d.d_sorted_beam_start->device_pointer;
  const device_ptr out_flux = d.d_sorted_flux->device_pointer;
  const device_ptr out_beam_sigma = d.d_sorted_beam_sigma->device_pointer;
  const int itable = (int)table_size;

  const DeviceKernelArguments iargs(&start_ptr, &cursor_ptr, &itable);
  if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_BIN_CURSOR_INIT, itable, iargs)) {
    LOG_WARNING << "CyclesPlus photon map: multi-device cursor init failed";
    return false;
  }
  const DeviceKernelArguments sargs(&dep_pos_ptr,
                                     &dep_beam_start_ptr,
                                     &dep_flux_ptr,
                                     &dep_beam_sigma_ptr,
                                     &cursor_ptr,
                                    &out_pos,
                                     &out_beam_start,
                                     &out_flux,
                                     &out_beam_sigma,
                                    &itable,
                                    &inv_cell,
                                    &total);
  if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_BIN_SCATTER, total, sargs) ||
      !d.gpu_queue->synchronize())
  {
    LOG_WARNING << "CyclesPlus photon map: multi-device scatter failed";
    return false;
  }

  /* Bounded by the deposit count, not the buffer capacity. */
  PhotonProfileScope readback("photon.deposit_readback_bytes", &d, double(total) * sizeof(float4) * 4);
  d.d_sorted_pos->copy_from_device(0, total, 1);
  d.d_sorted_beam_start->copy_from_device(0, total, 1);
  d.d_sorted_flux->copy_from_device(0, total, 1);
  d.d_sorted_beam_sigma->copy_from_device(0, total, 1);
  readback.finish();

  out.pos.resize(total);
  out.beam_start.resize(total);
  out.flux.resize(total);
  out.beam_sigma.resize(total);
  memcpy(out.pos.data(), d.d_sorted_pos->data(), sizeof(float4) * total);
  memcpy(out.beam_start.data(), d.d_sorted_beam_start->data(), sizeof(float4) * total);
  memcpy(out.flux.data(), d.d_sorted_flux->data(), sizeof(float4) * total);
  memcpy(out.beam_sigma.data(), d.d_sorted_beam_sigma->data(), sizeof(float4) * total);
  return true;
}

/* GPU photon tracing: dispatch DEVICE_KERNEL_PHOTON_TRACE in chunks, copy
 * the deposits back and reuse the host binning (device-resident binning is
 * the follow-up). Runs on CUDA and OptiX render devices. */
size_t trace_batch_gpu(PhotonMapData &d,
                       const int num_photons,
                       const float radius,
                       const uint64_t k,
                       const std::atomic<bool> *cancel,
                       PhotonMapData::Storage &out)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.sync_gpu_batch", &d, num_photons);
  photon_profile_value("photon.batch_index", &d, k);
  out.valid = false;
  const float safe_radius = photon_safe_radius(radius);
  const int max_bounces = 12;
  Device *device = d.trace_device;

  /* Per-light photon index ranges (same budget logic as the CPU tracer). */
  vector<PhotonTraceLight> lights(d.lights.size());
  int total_photons = 0;
  for (size_t li = 0; li < d.lights.size(); li++) {
    const PhotonLight &L = d.lights[li];
    PhotonTraceLight &kl = lights[li];
    const int n = (int)std::max((int64_t)((double)num_photons * d.light_share[li]),
                                (int64_t)256);
    fill_kernel_light(L, kl, total_photons, total_photons + n);
    total_photons += n;
  }
  vector<PhotonTraceTarget> targets = build_guided_targets(d);
  if (targets.empty()) {
    /* Kernel-side guard exists too; never launch against a null target
     * table (the 256-photon-per-light floor would launch regardless). */
    return 0;
  }

  /* Lazy device resources. */
  const int chunk = 8 * 1024 * 1024;
  /* OptiX final renders launch in 2M slices: the F12 photon path fired 8M
   * launches while the viewport was long since chunked small - and OptiX +
   * photon FINAL-render load killed the driver three times on 2026-08-09/11
   * (chair4 native, ~40s-4min in, no TDR event, nvidia-smi "No devices
   * found"; headless AND rendered-viewport both died, OIDN guard active,
   * so neither the denoiser nor the viewport session is the trigger).
   * Smaller launches with the existing per-slice synchronize give the
   * driver's display engine room to breathe - the same recipe that held a
   * 45min viewport marathon on 08-05. Costs microseconds per generation;
   * seeds depend only on the photon index, so slicing cannot change the
   * image. Staging capacity stays at `chunk` - a smaller capacity would
   * force grow-and-retrace loops. */
  const int launch_chunk = (d.optix_device && !d.interactive) ? 2 * 1024 * 1024 : chunk;
  if (!d.gpu_queue) {
    /* Only a real GPU sub-device gets here (photon_pick_trace_device), and a
     * device that cannot make a queue returns null rather than one we would
     * dereference below. */
    d.gpu_queue = device->gpu_queue_create();
    if (!d.gpu_queue) {
      LOG_WARNING << "CyclesPlus photon map: no device queue, generation dropped";
      return 0;
    }
    d.gpu_queue->init_execution();
  }
  if (!d.d_lights) {
    d.d_lights = make_unique<device_vector<PhotonTraceLight>>(
        device, "photon_trace_lights", MEM_READ_ONLY);
    d.d_targets = make_unique<device_vector<PhotonTraceTarget>>(
        device, "photon_trace_targets", MEM_READ_ONLY);
    d.d_materials = make_unique<device_vector<PhotonTraceMaterial>>(
        device, "photon_trace_materials", MEM_READ_ONLY);
    d.d_counter = make_unique<device_vector<uint>>(device, "photon_trace_counter", MEM_READ_WRITE);
    /* copy_to_device here so the device allocation exists before the
     * device_pointer is read below (alloc() is host-side only). */
    d.d_counter->alloc(1);
    d.d_counter->data()[0] = 0;
    d.d_counter->copy_to_device();
  }

  memcpy(d.d_lights->alloc(lights.size()), lights.data(),
         sizeof(PhotonTraceLight) * lights.size());
  d.d_lights->copy_to_device();
  memcpy(d.d_targets->alloc(targets.size()), targets.data(),
         sizeof(PhotonTraceTarget) * targets.size());
  d.d_targets->copy_to_device();
  memcpy(d.d_materials->alloc(d.gpu_materials.size()), d.gpu_materials.data(),
         sizeof(PhotonTraceMaterial) * d.gpu_materials.size());
  d.d_materials->copy_to_device();

  const device_ptr counter_ptr = d.d_counter->device_pointer;
  const device_ptr lights_ptr = d.d_lights->device_pointer;
  const device_ptr targets_ptr = d.d_targets->device_pointer;
  const device_ptr mats_ptr = d.d_materials->device_pointer;
  const int num_lights = (int)lights.size();
  const int num_targets = (int)targets.size();
  const int batch_k = (int)k;

  /* Runtime-switchable fault bisection (see photon_trace_single). */
  static const int debug_mode = []() {
    const char *env = getenv("CYCLESPLUS_PHOTON_GPU_DEBUG");
    return env ? atoi(env) : 0;
  }();
  static const int debug_max_photons = []() {
    const char *env = getenv("CYCLESPLUS_PHOTON_GPU_MAXN");
    return env ? atoi(env) : 0;
  }();
  if (debug_max_photons > 0) {
    total_photons = std::min(total_photons, debug_max_photons);
  }
  if (debug_mode || debug_max_photons) {
    LOG_INFO << "CyclesPlus photon map: GPU debug mode " << debug_mode << ", max photons "
             << debug_max_photons << ", counter=" << (void *)counter_ptr
             << " lights=" << (void *)lights_ptr << " targets=" << (void *)targets_ptr
             << " mats=" << (void *)mats_ptr;
  }

  /* Debug mode 9: allocate all device resources but never launch a kernel
   * (isolates resource lifecycle problems from kernel execution). */
  if (debug_mode == 9) {
    return 0;
  }

  /* CYCLESPLUS_PHOTON_GPU_BIN_DISABLE forces the legacy copy-back + host
   * binning path (validation escape hatch, like the tracer's own DISABLE).
   * Debug modes also use it: the histograms need the deposits on the host. */
  static const bool bin_disable = getenv("CYCLESPLUS_PHOTON_GPU_BIN_DISABLE") != nullptr;
  /* 2026-07-24: the device-binning energy loss (~13% in final renders, its
   * sporadic absence = the residual animation flicker) was the NULL-stream
   * memset/upload racing the bin kernels on the non-blocking queue - fixed
   * via host-zero + copy_to_device + drain_null_stream barriers. The
   * temporary host-binning detour for final renders is gone; device
   * binning matches host binning to 1e-7 again on every generation
   * (BIN_VERIFY clean over repeated runs).
   *
   * Multi-card rigs keep the host-binning path. Device-resident binning ends
   * in scatter_published(), which writes straight into the render arrays
   * through the device pointers PathTrace hands it - and on a MultiDevice
   * those pointers are the wrapper's bookkeeping keys, valid on no card at
   * all.
   *
   * A multi device therefore still BINS on the trace card (device_bin) but
   * cannot leave the result there (device_resident): scatter_device_to_host
   * sorts into buffers of our own and copies the sorted arrays back, so the
   * ordinary MEM_GLOBAL upload can replicate them to every card. Binning on
   * the CPU instead was measured at 3.5x the photon cost. */
  const bool device_bin = !bin_disable && debug_mode == 0;
  const bool device_resident = device_bin && !d.multi_device && !d.scene.volume_caustics;

  if (device_bin) {
    /* Device-resident binning: the deposits stay in VRAM for the entire
     * generation; the host only sees the 4-byte deposit counter and the
     * per-cell count table (MBs instead of hundreds of MBs over PCIe).
     * Staging is per Storage (double buffered: the next generation is
     * traced before this one is scattered) and grows on overflow with an
     * exact-size retrace, so the cost is a one-time hiccup per session. */
    if (!out.dep_pos_dev) {
      out.dep_pos_dev = make_unique<device_vector<float4>>(
          device, "photon_dep_pos", MEM_READ_WRITE);
      out.dep_beam_start_dev = make_unique<device_vector<float4>>(
          device, "photon_dep_beam_start", MEM_READ_WRITE);
      out.dep_flux_dev = make_unique<device_vector<float4>>(
          device, "photon_dep_flux", MEM_READ_WRITE);
      out.dep_beam_sigma_dev = make_unique<device_vector<float4>>(
          device, "photon_dep_beam_sigma", MEM_READ_WRITE);
      /* Host-zero + copy_to_device, NEVER zero_to_device before kernel use:
       * cuMemsetD8 runs on the NULL stream, our queue is NON_BLOCKING, so
       * the memset races the kernels and wipes their writes (the cold
       * first-allocation memset reliably LOST the race and erased the bin
       * counts of the first two generations of every final render - the
       * ~13% caustic energy loss + residual flicker, root-caused
       * 2026-07-24). A blocking HtoD copy is ordered after null-stream ops
       * and does not return until complete. */
      memset(out.dep_pos_dev->alloc(chunk), 0, sizeof(float4) * chunk);
      out.dep_pos_dev->copy_to_device();
      memset(out.dep_beam_start_dev->alloc(chunk), 0, sizeof(float4) * chunk);
      out.dep_beam_start_dev->copy_to_device();
      memset(out.dep_flux_dev->alloc(chunk), 0, sizeof(float4) * chunk);
      out.dep_flux_dev->copy_to_device();
      memset(out.dep_beam_sigma_dev->alloc(chunk), 0, sizeof(float4) * chunk);
      out.dep_beam_sigma_dev->copy_to_device();
    }
    for (int attempt = 0; attempt < 3; attempt++) {
      const int capacity = (int)out.dep_pos_dev->size();
      const device_ptr dep_pos_ptr = out.dep_pos_dev->device_pointer;
      const device_ptr dep_beam_start_ptr = out.dep_beam_start_dev->device_pointer;
      const device_ptr dep_flux_ptr = out.dep_flux_dev->device_pointer;
      const device_ptr dep_beam_sigma_ptr = out.dep_beam_sigma_dev->device_pointer;
      d.d_counter->data()[0] = 0;
      d.d_counter->copy_to_device();
      arm_target_yield(d);
      const device_ptr yield_ptr = d.d_target_yield ? d.d_target_yield->device_pointer : 0;
      drain_null_stream(d); /* uploads fully landed before the trace launches */

      for (int off = 0; off < total_photons; off += launch_chunk) {
        if (cancel && cancel->load(std::memory_order_relaxed)) {
          return 0;
        }
        const int n = std::min(launch_chunk, total_photons - off);
        const DeviceKernelArguments args(&dep_pos_ptr,
                                         &dep_beam_start_ptr,
                                         &dep_flux_ptr,
                                         &dep_beam_sigma_ptr,
                                         &counter_ptr,
                                         &lights_ptr,
                                         &num_lights,
                                         &targets_ptr,
                                         &num_targets,
                                         &mats_ptr,
                                         &batch_k,
                                         &off,
                                         &max_bounces,
                                         &debug_mode,
                                         &capacity,
                                         &n,
                                         &yield_ptr);
        if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_TRACE, n, args)) {
          LOG_WARNING << "CyclesPlus photon map: GPU trace enqueue failed";
          return 0;
        }
        if (!d.gpu_queue->synchronize()) {
          LOG_WARNING << "CyclesPlus photon map: GPU trace synchronize failed";
          return 0;
        }
      }

      d.d_counter->copy_from_device();
      const uint counted = d.d_counter->data()[0];
      if (counted > (uint)capacity) {
        photon_profile_value("photon.staging_overflow", &d, counted);
        const int new_cap = (int)(counted + counted / 4);
        /* Host-zero + copy: see the staging-creation comment (memset race). */
        memset(out.dep_pos_dev->alloc(new_cap), 0, sizeof(float4) * new_cap);
        out.dep_pos_dev->copy_to_device();
        memset(out.dep_beam_start_dev->alloc(new_cap), 0, sizeof(float4) * new_cap);
        out.dep_beam_start_dev->copy_to_device();
        memset(out.dep_flux_dev->alloc(new_cap), 0, sizeof(float4) * new_cap);
        out.dep_flux_dev->copy_to_device();
        memset(out.dep_beam_sigma_dev->alloc(new_cap), 0, sizeof(float4) * new_cap);
        out.dep_beam_sigma_dev->copy_to_device();
        LOG_INFO << "CyclesPlus photon map: deposit staging grown to " << new_cap
                 << ", retracing generation";
        continue;
      }
      const int total = (int)counted;
      harvest_target_yield(d);
      if (total == 0) {
        return 0;
      }

      /* Count per cell on the device, prefix-sum on the host (same table
       * sizing as bin_packed_deposits, so both binners are exchangeable). */
      uint table_size = 64;
      while (table_size < (uint)total / 2) {
        table_size <<= 1;
      }
      const float inv_cell = 1.0f / safe_radius;
      if (!d.d_cell_count) {
        d.d_cell_count = make_unique<device_vector<uint>>(
            device, "photon_cell_count", MEM_READ_WRITE);
      }
      /* THE root cause of the final-render caustic energy loss: this was
       * zero_to_device (cuMemsetD8, NULL stream) racing the count kernel on
       * our NON_BLOCKING queue - a cold (re)allocated table memsets slowly
       * and wiped the counts of the first two generations of every final
       * render (~13% loss); a warm table usually won, its occasional loss
       * was the residual one-frame flicker. Host-zero + blocking HtoD is
       * ordered and complete on return. */
      /* High-water capacity: alloc(n) with a different n frees + reallocs
       * on the device, and cuMemFree synchronizes the whole context - paid
       * every generation while the viewport ramp grows the table. Keeping
       * the high-water size also keeps the table WARM, which the memset
       * race note above relies on. The count kernel only touches the first
       * table_size entries; the tail is dead weight, not state. */
      const size_t count_cap = std::max((size_t)table_size, d.d_cell_count->size());
      memset(d.d_cell_count->alloc(count_cap), 0, sizeof(uint) * count_cap);
      d.d_cell_count->copy_to_device();
      drain_null_stream(d); /* zero-upload fully landed before the count */
      const device_ptr count_ptr = d.d_cell_count->device_pointer;
      const int itable = (int)table_size;
      const DeviceKernelArguments cargs(&dep_pos_ptr, &count_ptr, &itable, &inv_cell, &total);
      if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_BIN_COUNT, total, cargs) ||
          !d.gpu_queue->synchronize())
      {
        LOG_WARNING << "CyclesPlus photon map: GPU bin count failed";
        return 0;
      }
      d.d_cell_count->copy_from_device();

      out.cell_start.clear();
      out.cell_start.resize(table_size + 1, 0);
      for (uint b = 0; b < table_size; b++) {
        out.cell_start[b + 1] = out.cell_start[b] + (int)d.d_cell_count->data()[b];
      }

      /* CYCLESPLUS_PHOTON_BIN_VERIFY=1: copy the staging back and rebuild the
       * cell histogram on the host - a mismatch localizes the device-binning
       * energy loss to the count kernel (hash/floor divergence GPU vs host),
       * agreement pushes it to the scatter or the gather-side binding. */
      static const bool bin_verify = getenv("CYCLESPLUS_PHOTON_BIN_VERIFY") != nullptr;
      if (bin_verify) {
        out.dep_pos_dev->copy_from_device();
        const float4 *hp = out.dep_pos_dev->data();
        vector<uint> host_count(table_size, 0);
        const float vinv = 1.0f / safe_radius;
        for (int i = 0; i < total; i++) {
          const float4 p = hp[i];
          host_count[photon_grid_hash((int)floorf(p.x * vinv),
                                      (int)floorf(p.y * vinv),
                                      (int)floorf(p.z * vinv),
                                      table_size)]++;
        }
        uint64_t mismatch_cells = 0, mismatch_abs = 0;
        for (uint b = 0; b < table_size; b++) {
          const int64_t dl = (int64_t)host_count[b] - (int64_t)d.d_cell_count->data()[b];
          if (dl != 0) {
            mismatch_cells++;
            mismatch_abs += (uint64_t)std::abs(dl);
          }
        }
        uint64_t dev_sum = 0;
        for (uint b = 0; b < table_size; b++) {
          dev_sum += d.d_cell_count->data()[b];
        }
        /* Second count over the same pointer AFTER the host copy-back: a
         * higher sum means the first count raced the still-running trace
         * (sync hole); an identical low sum means the count kernel reads
         * different memory than the copy-back (pointer identity broken). */
        memset(d.d_cell_count->data(), 0, sizeof(uint) * table_size);
        d.d_cell_count->copy_to_device();
        drain_null_stream(d);
        const DeviceKernelArguments vargs(&dep_pos_ptr, &count_ptr, &itable, &inv_cell, &total);
        uint64_t dev_sum2 = 0;
        if (d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_BIN_COUNT, total, vargs) &&
            d.gpu_queue->synchronize())
        {
          d.d_cell_count->copy_from_device();
          for (uint b = 0; b < table_size; b++) {
            dev_sum2 += d.d_cell_count->data()[b];
          }
        }
        LOG_INFO << "CyclesPlus BIN_VERIFY: total " << total << " table " << table_size
                 << " device-count-sum " << dev_sum << " recount " << dev_sum2
                 << " mismatch cells " << mismatch_cells << " abs " << mismatch_abs
                 << " staging-ptr " << (void *)dep_pos_ptr << " now "
                 << (void *)out.dep_pos_dev->device_pointer;
        /* Rebuild the prefix from the recount so the grid the render consumes
         * reflects the recount, not the possibly-broken first count. NOTE:
         * diagnosis only - this block is env-gated. */
      }

      if (!device_resident) {
        /* Multi device: sort here and ship the result, since the gather runs
         * on cards this one cannot write to. */
        if (!scatter_device_to_host(d,
                                    out,
                                    dep_pos_ptr,
                                    dep_beam_start_ptr,
                                    dep_flux_ptr,
                                    dep_beam_sigma_ptr,
                                    total,
                                    table_size,
                                    inv_cell))
        {
          return 0;
        }
        build_volume_beam_bvh(out.pos.data(),
                              out.beam_start.data(),
                              out.flux.data(),
                              out.beam_sigma.data(),
                              total,
                              safe_radius,
                              out);
        out.grid.pos = out.pos.data();
        out.grid.beam_start = out.beam_start.data();
        out.grid.flux = out.flux.data();
        out.grid.cell_start = out.cell_start.data();
        out.grid.num_photons = total;
        out.grid.table_size = table_size;
        out.grid.radius = safe_radius;
        out.grid.inv_cell = inv_cell;
        out.grid.volume_beam_start = out.volume_beam_start.data();
        out.grid.volume_beam_end = out.volume_beam_end.data();
        out.grid.volume_beam_flux = out.volume_beam_flux.data();
        out.grid.volume_beam_sigma = out.volume_beam_sigma.data();
        out.grid.volume_beam_nodes = out.volume_beam_nodes.data();
        out.grid.num_volume_beams = (int)out.volume_beam_start.size();
        out.grid.num_volume_beam_nodes = (int)out.volume_beam_nodes.size();
        out.grid.device_resident = 0;
        out.grid.dep_pos_device = 0;
        out.grid.dep_beam_start_device = 0;
        out.grid.dep_flux_device = 0;
        out.grid.dep_beam_sigma_device = 0;
        out.grid.owner = nullptr;
        out.valid = true;
        return (size_t)total;
      }

      out.pos.clear();
      out.beam_start.clear();
      out.flux.clear();
      out.grid.pos = nullptr;
      out.grid.beam_start = nullptr;
      out.grid.flux = nullptr;
      out.grid.cell_start = out.cell_start.data();
      out.grid.num_photons = total;
      out.grid.table_size = table_size;
      out.grid.radius = safe_radius;
      out.grid.inv_cell = inv_cell;
      out.grid.device_resident = 1;
      out.grid.dep_pos_device = (uint64_t)dep_pos_ptr;
      out.grid.dep_beam_start_device = (uint64_t)dep_beam_start_ptr;
      out.grid.dep_flux_device = (uint64_t)dep_flux_ptr;
      out.grid.dep_beam_sigma_device = (uint64_t)dep_beam_sigma_ptr;
      out.grid.owner = d.self;
      out.valid = true;
      return (size_t)total;
    }
    LOG_WARNING << "CyclesPlus photon map: deposit staging kept overflowing, generation dropped";
    return 0;
  }

  /* Legacy copy-back path (debug modes / escape hatch). */
  if (!d.d_pos) {
    d.d_pos = make_unique<device_vector<float4>>(device, "photon_trace_pos", MEM_READ_WRITE);
    d.d_beam_start = make_unique<device_vector<float4>>(
        device, "photon_trace_beam_start", MEM_READ_WRITE);
    d.d_flux = make_unique<device_vector<float4>>(device, "photon_trace_flux", MEM_READ_WRITE);
    d.d_beam_sigma = make_unique<device_vector<float4>>(
        device, "photon_trace_beam_sigma", MEM_READ_WRITE);
    /* Host-zero + copy: see the memset-race comments (trace_batch_gpu). */
    memset(d.d_pos->alloc(chunk), 0, sizeof(float4) * chunk);
    d.d_pos->copy_to_device();
    memset(d.d_beam_start->alloc(chunk), 0, sizeof(float4) * chunk);
    d.d_beam_start->copy_to_device();
    memset(d.d_flux->alloc(chunk), 0, sizeof(float4) * chunk);
    d.d_flux->copy_to_device();
    memset(d.d_beam_sigma->alloc(chunk), 0, sizeof(float4) * chunk);
    d.d_beam_sigma->copy_to_device();
  }
  vector<float4> all_pos, all_beam_start, all_flux, all_beam_sigma;
  all_pos.reserve((size_t)num_photons / 2);
  all_beam_start.reserve((size_t)num_photons / 2);
  all_flux.reserve((size_t)num_photons / 2);
  all_beam_sigma.reserve((size_t)num_photons / 2);
  const device_ptr pos_ptr = d.d_pos->device_pointer;
  const device_ptr beam_start_ptr = d.d_beam_start->device_pointer;
  const device_ptr flux_ptr = d.d_flux->device_pointer;
  const device_ptr beam_sigma_ptr = d.d_beam_sigma->device_pointer;
  arm_target_yield(d);
  const device_ptr yield_ptr = d.d_target_yield ? d.d_target_yield->device_pointer : 0;
  drain_null_stream(d);

  for (int off = 0; off < total_photons; off += chunk) {
    if (cancel && cancel->load(std::memory_order_relaxed)) {
      return 0;
    }
    const int n = std::min(chunk, total_photons - off);
    d.d_counter->data()[0] = 0;
    d.d_counter->copy_to_device();

    const int cap = n;
    const DeviceKernelArguments args(&pos_ptr,
                                     &beam_start_ptr,
                                     &flux_ptr,
                                     &beam_sigma_ptr,
                                     &counter_ptr,
                                     &lights_ptr,
                                     &num_lights,
                                     &targets_ptr,
                                     &num_targets,
                                     &mats_ptr,
                                     &batch_k,
                                     &off,
                                     &max_bounces,
                                     &debug_mode,
                                     &cap,
                                     &n,
                                     &yield_ptr);
    if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_TRACE, n, args)) {
      LOG_WARNING << "CyclesPlus photon map: GPU trace enqueue failed";
      return 0;
    }
    if (!d.gpu_queue->synchronize()) {
      LOG_WARNING << "CyclesPlus photon map: GPU trace synchronize failed";
      return 0;
    }
    d.d_counter->copy_from_device();
    const uint count = std::min(d.d_counter->data()[0], (uint)n);
    if (count > 0) {
      /* Read back the deposits that exist, not the whole staging buffer.
       * These arrays are allocated at chunk capacity - 8M float4 = 128 MB
       * each - while a generation usually fills a fraction of it, and the
       * unqualified copy_from_device() moves the full capacity twice per
       * chunk regardless. That was the bulk of the host-binning cost:
       * measured 2026-08-25 on the 3-sphere scene, the photon overhead went
       * 1.42s (device binning) to 4.97s (host binning) before this, and the
       * multi-device path has no choice but to bin on the host. */
      d.d_pos->copy_from_device(0, count, 1);
      d.d_beam_start->copy_from_device(0, count, 1);
      d.d_flux->copy_from_device(0, count, 1);
      d.d_beam_sigma->copy_from_device(0, count, 1);
      all_pos.insert(all_pos.end(), d.d_pos->data(), d.d_pos->data() + count);
      all_beam_start.insert(
          all_beam_start.end(), d.d_beam_start->data(), d.d_beam_start->data() + count);
      all_flux.insert(all_flux.end(), d.d_flux->data(), d.d_flux->data() + count);
      all_beam_sigma.insert(
          all_beam_sigma.end(), d.d_beam_sigma->data(), d.d_beam_sigma->data() + count);
    }
  }

  harvest_target_yield(d);

  if (debug_mode == 5) {
    /* Bounce histogram over flux.w (see kernel debug mode 5). */
    int histogram[16] = {};
    for (const float4 &f : all_flux) {
      const int b = std::min(std::max((int)f.w, 0), 15);
      histogram[b]++;
    }
    string msg = "CyclesPlus photon map: bounce histogram:";
    for (int b = 0; b < 13; b++) {
      msg += string_printf(" b%d=%d", b, histogram[b]);
    }
    LOG_INFO << msg;
  }

  if (debug_mode == 6 && !all_flux.empty()) {
    /* Post-eval probe (see kernel debug mode 6): flux encodes
     * (cache_miss, num_closure, bsdf weight sum, shader index). */
    double w_sum = 0.0;
    int cache_miss = 0, zero_closures = 0, zero_weight = 0;
    std::map<int, int> nclosure_hist;
    std::map<int, int> shader_hist;
    for (const float4 &f : all_flux) {
      if (f.x > 0.5f) cache_miss++;
      if ((int)f.y == 0) zero_closures++;
      if (f.z <= 0.0f) zero_weight++;
      w_sum += f.z;
      nclosure_hist[(int)f.y]++;
      shader_hist[(int)f.w]++;
    }
    string msg = string_printf(
        "CyclesPlus photon map: accurate probe: n=%zu cache_miss=%d zero_closures=%d "
        "zero_weight=%d w_mean=%.4f nclosure:",
        all_flux.size(),
        cache_miss,
        zero_closures,
        zero_weight,
        w_sum / (double)all_flux.size());
    for (const auto &t : nclosure_hist) {
      msg += string_printf(" c%d=%d", t.first, t.second);
    }
    msg += " shaders:";
    for (const auto &t : shader_hist) {
      msg += string_printf(" s%d=%d", t.first, t.second);
    }
    LOG_INFO << msg;
  }

  return bin_packed_deposits(
      all_pos.data(),
      all_beam_start.data(),
      all_flux.data(),
      all_beam_sigma.data(),
      all_pos.size(),
      radius,
      out);
}

/* ------------------------------------------------------------------
 * Async GPU generation (interactive viewport). Same kernels, staging and
 * binning as trace_batch_gpu, but spread across render works: begin() sets
 * the generation up, step() enqueues ONE chunk without synchronizing (the
 * photon queue's own CUDA stream overlaps the render), finish() synchronizes
 * once the last chunk is enqueued and bins device-side. Debug modes, the
 * bin escape hatch and final renders keep the synchronous path.
 * NOTE: setup/finish mirror trace_batch_gpu - fold together in the planned
 * gather refactor. */

bool gpu_async_begin(PhotonMapData &d,
                     const int num_photons,
                     const float radius,
                     const uint64_t k,
                     PhotonMapData::Storage &out)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.async_setup", &d, num_photons);
  photon_profile_value("photon.batch_index", &d, k);
  const float safe_radius = photon_safe_radius(radius);
  static const bool bin_disable = getenv("CYCLESPLUS_PHOTON_GPU_BIN_DISABLE") != nullptr;
  static const int debug_mode = []() {
    const char *env = getenv("CYCLESPLUS_PHOTON_GPU_DEBUG");
    return env ? atoi(env) : 0;
  }();
  if (bin_disable || debug_mode != 0) {
    return false; /* synchronous fallback handles these */
  }
  /* The async path bins device-side unconditionally (it has no copy-back
   * branch), so a multi-card rig takes the synchronous path with its host
   * binning - see the device_resident note in trace_batch_gpu. */
  if (d.multi_device) {
    return false;
  }

  /* The 2026-07-21 animation hangs were bisected to OIDN's GPU denoiser
   * context (session.cpp redirects it on final renders) - the async stream
   * was contained on OptiX during the hunt but never shown harmful, and a
   * week of heavy viewport use ran async + OptiX without a single crash.
   * Async is back on by default; CYCLESPLUS_PHOTON_OPTIX_SYNC=1 forces the
   * synchronous fallback on OptiX devices for future bisections. */
  static const bool optix_sync_optout = getenv("CYCLESPLUS_PHOTON_OPTIX_SYNC") != nullptr;
  if (d.trace_device->info.type == DEVICE_OPTIX && optix_sync_optout) {
    return false;
  }

  out.valid = false;
  Device *device = d.trace_device;

  vector<PhotonTraceLight> lights(d.lights.size());
  int total_photons = 0;
  for (size_t li = 0; li < d.lights.size(); li++) {
    const PhotonLight &L = d.lights[li];
    PhotonTraceLight &kl = lights[li];
    const int n = (int)std::max((int64_t)((double)num_photons * d.light_share[li]),
                                (int64_t)256);
    fill_kernel_light(L, kl, total_photons, total_photons + n);
    total_photons += n;
  }
  vector<PhotonTraceTarget> targets = build_guided_targets(d);
  if (targets.empty()) {
    return false; /* see the sync-path guard: no launch against a null table */
  }
  static const int debug_max_photons = []() {
    const char *env = getenv("CYCLESPLUS_PHOTON_GPU_MAXN");
    return env ? atoi(env) : 0;
  }();
  if (debug_max_photons > 0) {
    total_photons = std::min(total_photons, debug_max_photons);
  }

  const int chunk = 8 * 1024 * 1024;
  if (!d.gpu_queue) {
    d.gpu_queue = device->gpu_queue_create();
    if (!d.gpu_queue) {
      LOG_WARNING << "CyclesPlus photon map: no device queue, falling back to the sync path";
      return false;
    }
    d.gpu_queue->init_execution();
  }
  if (!d.d_counter) {
    d.d_counter = make_unique<device_vector<uint>>(device, "photon_trace_counter",
                                                   MEM_READ_WRITE);
    d.d_counter->alloc(1);
    d.d_counter->data()[0] = 0;
    d.d_counter->copy_to_device();
  }
  if (!d.d_lights) {
    d.d_lights = make_unique<device_vector<PhotonTraceLight>>(
        device, "photon_trace_lights", MEM_READ_ONLY);
    d.d_targets = make_unique<device_vector<PhotonTraceTarget>>(
        device, "photon_trace_targets", MEM_READ_ONLY);
    d.d_materials = make_unique<device_vector<PhotonTraceMaterial>>(
        device, "photon_trace_materials", MEM_READ_ONLY);
  }
  memcpy(d.d_lights->alloc(lights.size()), lights.data(),
         sizeof(PhotonTraceLight) * lights.size());
  d.d_lights->copy_to_device();
  memcpy(d.d_targets->alloc(targets.size()), targets.data(),
         sizeof(PhotonTraceTarget) * targets.size());
  d.d_targets->copy_to_device();
  memcpy(d.d_materials->alloc(d.gpu_materials.size()), d.gpu_materials.data(),
         sizeof(PhotonTraceMaterial) * d.gpu_materials.size());
  d.d_materials->copy_to_device();

  if (!out.dep_pos_dev) {
    out.dep_pos_dev = make_unique<device_vector<float4>>(device, "photon_dep_pos",
                                                         MEM_READ_WRITE);
    out.dep_beam_start_dev = make_unique<device_vector<float4>>(device,
                                                                "photon_dep_beam_start",
                                                                MEM_READ_WRITE);
    out.dep_flux_dev = make_unique<device_vector<float4>>(device, "photon_dep_flux",
                                                          MEM_READ_WRITE);
    out.dep_beam_sigma_dev = make_unique<device_vector<float4>>(
        device, "photon_dep_beam_sigma", MEM_READ_WRITE);
    /* Host-zero + copy: see the memset-race comments (trace_batch_gpu). */
    memset(out.dep_pos_dev->alloc(chunk), 0, sizeof(float4) * chunk);
    out.dep_pos_dev->copy_to_device();
    memset(out.dep_beam_start_dev->alloc(chunk), 0, sizeof(float4) * chunk);
    out.dep_beam_start_dev->copy_to_device();
    memset(out.dep_flux_dev->alloc(chunk), 0, sizeof(float4) * chunk);
    out.dep_flux_dev->copy_to_device();
    memset(out.dep_beam_sigma_dev->alloc(chunk), 0, sizeof(float4) * chunk);
    out.dep_beam_sigma_dev->copy_to_device();
  }
  d.d_counter->data()[0] = 0;
  d.d_counter->copy_to_device();
  arm_target_yield(d);
  drain_null_stream(d);

  d.gpu_inflight = true;
  d.gpu_off = 0;
  d.gpu_total = total_photons;
  d.gpu_num_photons = num_photons;
  d.gpu_radius = safe_radius;
  d.gpu_batch_k = (int)k;
  d.gpu_begin_time = time_dt();
  photon_profile_value("photon.batch_launched_photons", &d, total_photons);
  return true;
}

/* Enqueue one chunk without synchronizing. Chunk size targets ~30ms of GPU
 * work, so the finish sync (one work later) never stalls noticeably. */
void gpu_async_step(PhotonMapData &d, PhotonMapData::Storage &out)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.async_submit", &d);
  photon_profile_value("photon.batch_index", &d, d.gpu_batch_k);
  /* OptiX throttle: target ~8ms launches capped at 2M photons instead of
   * ~30ms/8M - the display engine gets serviced between launches, which is
   * what keeps the driver's OptiX side alive under hour-long sessions.
   * BURST EXCEPTION: the first generations after a reset run untouched -
   * they are what the user sees as "caustics resolving" (~1-2s of work),
   * and the driver hangs needed 30-60min of sustained load, not moments.
   * Full speed in bursts, relief on the long haul. */
  /* OptiX final renders slice to 2M regardless of the (opt-in) viewport
   * throttle - same rationale and evidence as the sync path's launch_chunk
   * (three driver deaths under F12 photon load, 2026-08-09/11). The F12
   * async first batch runs through here. */
  const bool gentle = d.optix_device &&
                      ((!d.interactive) || (!optix_fullbore() && d.gpu_batch_k > 6));
  const double chunk_target_s = gentle ? 0.008 : 0.03;
  const double chunk_cap = gentle ? (double)(2 * 1024 * 1024) : (double)(8 * 1024 * 1024);
  int chunk_n = 2 * 1024 * 1024;
  if (d.photons_per_sec > 0.0) {
    chunk_n = (int)std::min(chunk_cap,
                            std::max((double)(256 * 1024), d.photons_per_sec * chunk_target_s));
  }
  /* Hard bound on how long a generation may take to appear: one chunk goes
   * out per render work, so the chunk COUNT is the publication latency, and
   * that latency is what "instant caustics" means in the viewport. The rate
   * estimate above can be a large under-estimate, which used to split a
   * generation into ~49 chunks (10s to publish for 0.1s of GPU work). A
   * generation is 0.1s of tracing on the reference GPU, so a handful of
   * chunks is imperceptible even when the estimate is wrong. */
  const int max_chunks = gentle ? 8 : 4;
  chunk_n = std::max(chunk_n, (d.gpu_total + max_chunks - 1) / max_chunks);
  const int n = std::min(chunk_n, d.gpu_total - d.gpu_off);
  if (n <= 0) {
    return;
  }
  const device_ptr dep_pos_ptr = out.dep_pos_dev->device_pointer;
  const device_ptr dep_beam_start_ptr = out.dep_beam_start_dev->device_pointer;
  const device_ptr dep_flux_ptr = out.dep_flux_dev->device_pointer;
  const device_ptr dep_beam_sigma_ptr = out.dep_beam_sigma_dev->device_pointer;
  const device_ptr counter_ptr = d.d_counter->device_pointer;
  const device_ptr lights_ptr = d.d_lights->device_pointer;
  const device_ptr targets_ptr = d.d_targets->device_pointer;
  const device_ptr mats_ptr = d.d_materials->device_pointer;
  const int num_lights = (int)d.d_lights->size();
  const int num_targets = (int)d.d_targets->size();
  const int max_bounces = 12;
  const int debug_mode = 0;
  const int capacity = (int)out.dep_pos_dev->size();
  const device_ptr yield_ptr = d.d_target_yield ? d.d_target_yield->device_pointer : 0;
  const DeviceKernelArguments args(&dep_pos_ptr,
                                   &dep_beam_start_ptr,
                                   &dep_flux_ptr,
                                   &dep_beam_sigma_ptr,
                                   &counter_ptr,
                                   &lights_ptr,
                                   &num_lights,
                                   &targets_ptr,
                                   &num_targets,
                                   &mats_ptr,
                                   &d.gpu_batch_k,
                                   &d.gpu_off,
                                   &max_bounces,
                                   &debug_mode,
                                   &capacity,
                                   &n,
                                   &yield_ptr);
  if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_TRACE, n, args)) {
    LOG_WARNING << "CyclesPlus photon map: async trace enqueue failed";
    d.gpu_total = d.gpu_off; /* force finish on the next advance */
    return;
  }
  d.gpu_off += n;
  photon_profile_value("photon.chunk_photons", &d, n);
}

/* All chunks enqueued: synchronize (the stream is typically already done),
 * then bin device-side. Returns the deposit count, 0 on failure, or
 * SIZE_MAX when the staging overflowed and the generation restarts (still
 * inflight, exact-size staging). */
size_t gpu_async_finish(PhotonMapData &d, PhotonMapData::Storage &out)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.async_finish", &d);
  PhotonProfileScope trace_wait("photon.trace_wait", &d);
  if (!d.gpu_queue->synchronize()) {
    LOG_WARNING << "CyclesPlus photon map: async trace synchronize failed";
    return 0;
  }
  trace_wait.finish();
  PhotonProfileScope counter_read("photon.counter_readback", &d);
  d.d_counter->copy_from_device();
  counter_read.finish();
  const uint counted = d.d_counter->data()[0];
  photon_profile_value("photon.batch_deposits", &d, counted);
  const int capacity = (int)out.dep_pos_dev->size();
  if (counted > (uint)capacity) {
    photon_profile_value("photon.staging_overflow", &d, counted);
    const int new_cap = (int)(counted + counted / 4);
    /* Host-zero + copy: see the memset-race comments (trace_batch_gpu). */
    memset(out.dep_pos_dev->alloc(new_cap), 0, sizeof(float4) * new_cap);
    out.dep_pos_dev->copy_to_device();
    memset(out.dep_beam_start_dev->alloc(new_cap), 0, sizeof(float4) * new_cap);
    out.dep_beam_start_dev->copy_to_device();
    memset(out.dep_flux_dev->alloc(new_cap), 0, sizeof(float4) * new_cap);
    out.dep_flux_dev->copy_to_device();
    memset(out.dep_beam_sigma_dev->alloc(new_cap), 0, sizeof(float4) * new_cap);
    out.dep_beam_sigma_dev->copy_to_device();
    d.d_counter->data()[0] = 0;
    d.d_counter->copy_to_device();
    arm_target_yield(d);
    drain_null_stream(d);
    d.gpu_off = 0;
    LOG_INFO << "CyclesPlus photon map: async staging grown to " << new_cap << ", retracing";
    return (size_t)-1;
  }
  const int total = (int)counted;
  harvest_target_yield(d);
  if (total == 0) {
    return 0;
  }
  uint table_size = 64;
  while (table_size < (uint)total / 2) {
    table_size <<= 1;
  }
  const float inv_cell = 1.0f / photon_safe_radius(d.gpu_radius);
  if (!d.d_cell_count) {
    d.d_cell_count = make_unique<device_vector<uint>>(d.trace_device, "photon_cell_count",
                                                      MEM_READ_WRITE);
  }
  /* Host-zero + blocking copy - zero_to_device's null-stream memset races
   * the count kernel on the non-blocking queue (see trace_batch_gpu, the
   * root cause of the final-render caustic energy loss). High-water
   * capacity as in the sync path: no per-generation device realloc. */
  const size_t count_cap = std::max((size_t)table_size, d.d_cell_count->size());
  memset(d.d_cell_count->alloc(count_cap), 0, sizeof(uint) * count_cap);
  d.d_cell_count->copy_to_device();
  drain_null_stream(d);
  const device_ptr dep_pos_ptr = out.dep_pos_dev->device_pointer;
  const device_ptr count_ptr = d.d_cell_count->device_pointer;
  const int itable = (int)table_size;
  const DeviceKernelArguments cargs(&dep_pos_ptr, &count_ptr, &itable, &inv_cell, &total);
  if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_BIN_COUNT, total, cargs) ||
      !d.gpu_queue->synchronize())
  {
    LOG_WARNING << "CyclesPlus photon map: async bin count failed";
    return 0;
  }
  d.d_cell_count->copy_from_device();

  out.cell_start.clear();
  out.cell_start.resize(table_size + 1, 0);
  for (uint b = 0; b < table_size; b++) {
    out.cell_start[b + 1] = out.cell_start[b] + (int)d.d_cell_count->data()[b];
  }
  if (d.scene.volume_caustics) {
    if (!scatter_device_to_host(d,
                                out,
                                dep_pos_ptr,
                                out.dep_beam_start_dev->device_pointer,
                                out.dep_flux_dev->device_pointer,
                                out.dep_beam_sigma_dev->device_pointer,
                                total,
                                table_size,
                                inv_cell))
    {
      return 0;
    }
    build_volume_beam_bvh(out.pos.data(),
                          out.beam_start.data(),
                          out.flux.data(),
                          out.beam_sigma.data(),
                          total,
                          d.gpu_radius,
                          out);
    out.grid.pos = out.pos.data();
    out.grid.beam_start = out.beam_start.data();
    out.grid.flux = out.flux.data();
    out.grid.cell_start = out.cell_start.data();
    out.grid.num_photons = total;
    out.grid.table_size = table_size;
    out.grid.radius = photon_safe_radius(d.gpu_radius);
    out.grid.inv_cell = inv_cell;
    out.grid.volume_beam_start = out.volume_beam_start.data();
    out.grid.volume_beam_end = out.volume_beam_end.data();
    out.grid.volume_beam_flux = out.volume_beam_flux.data();
    out.grid.volume_beam_sigma = out.volume_beam_sigma.data();
    out.grid.volume_beam_nodes = out.volume_beam_nodes.data();
    out.grid.num_volume_beams = (int)out.volume_beam_start.size();
    out.grid.num_volume_beam_nodes = (int)out.volume_beam_nodes.size();
    out.grid.device_resident = 0;
    out.grid.dep_pos_device = 0;
    out.grid.dep_beam_start_device = 0;
    out.grid.dep_flux_device = 0;
    out.grid.dep_beam_sigma_device = 0;
    out.grid.owner = nullptr;
    out.valid = true;
    return (size_t)total;
  }
  out.pos.clear();
  out.beam_start.clear();
  out.flux.clear();
  out.grid.pos = nullptr;
  out.grid.beam_start = nullptr;
  out.grid.flux = nullptr;
  out.grid.cell_start = out.cell_start.data();
  out.grid.num_photons = total;
  out.grid.table_size = table_size;
  out.grid.radius = photon_safe_radius(d.gpu_radius);
  out.grid.inv_cell = inv_cell;
  out.grid.device_resident = 1;
  out.grid.dep_pos_device = (uint64_t)dep_pos_ptr;
  out.grid.dep_beam_start_device = (uint64_t)out.dep_beam_start_dev->device_pointer;
  out.grid.dep_flux_device = (uint64_t)out.dep_flux_dev->device_pointer;
  out.grid.dep_beam_sigma_device = (uint64_t)out.dep_beam_sigma_dev->device_pointer;
  out.grid.owner = d.self;
  out.valid = true;
  return (size_t)total;
}

/* Abort an in-flight async generation: wait for the stream (in-flight
 * kernels cannot be cancelled but are small), then drop the state. MUST be
 * called before any buffer reset while gpu_inflight. */
void gpu_async_abort(PhotonMapData &d)
{
  if (!d.gpu_inflight) {
    return;
  }
  CCL_PHOTON_PROFILE_SCOPE("photon.abort_wait", &d, d.gpu_off);
  if (d.gpu_queue) {
    d.gpu_queue->synchronize();
  }
  d.gpu_inflight = false;
  d.gpu_off = 0;
  d.gpu_total = 0;
}

size_t trace_batch(const PhotonTraceScene &ps,
                   const vector<PhotonLight> &lights,
                   const vector<PhotonTarget> &targets,
                   const vector<float> &light_share,
                   const int num_photons,
                   const float radius,
                   const uint64_t seed,
                   const std::atomic<bool> *cancel,
                   PhotonMapData::Storage &out)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.cpu_batch", &ps, num_photons);
  out.valid = false;

  const int max_bounces = 12;
  int n_threads = (int)std::thread::hardware_concurrency();
  n_threads = std::max(1, std::min(n_threads, 64));

  /* Budget per light, proportional to its measured caustic contribution. */
  vector<int64_t> n_light(lights.size());
  int64_t total_launched = 0;
  for (size_t li = 0; li < lights.size(); li++) {
    n_light[li] = std::max((int64_t)((double)num_photons * light_share[li]), (int64_t)256);
    total_launched += n_light[li];
  }
  const float beam_probability = std::min(1.0f, 65536.0f / (float)total_launched);

  vector<vector<PhotonDeposit>> results(n_threads);
  vector<std::thread> pool;
  for (int th = 0; th < n_threads; th++) {
    pool.emplace_back([&, th]() {
      vector<PhotonDeposit> &deposits = results[th];
      deposits.reserve((size_t)(num_photons / n_threads / 2));
      for (size_t li = 0; li < lights.size(); li++) {
        const int64_t i0 = n_light[li] * th / n_threads;
        const int64_t i1 = n_light[li] * (th + 1) / n_threads;
        emit_photons(ps,
                     lights[li],
                     targets,
                     i0,
                     i1,
                     n_light[li],
                     max_bounces,
                     seed + 7919 * (uint64_t)li,
                     (uint64_t)th * 1000003ULL + li,
                     cancel,
                     deposits,
                     beam_probability);
      }
    });
  }
  for (std::thread &t : pool) {
    t.join();
  }

  /* A cancelled batch is incomplete (wrong flux normalization) - discard. */
  if (cancel && cancel->load(std::memory_order_relaxed)) {
    return 0;
  }

  size_t total = 0;
  for (const vector<PhotonDeposit> &r : results) {
    total += r.size();
  }
  if (total == 0) {
    return 0;
  }

  /* Flatten into packed arrays and bin (shared with the GPU path). */
  vector<float4> dep_pos, dep_beam_start, dep_flux, dep_sigma;
  dep_pos.reserve(total);
  dep_beam_start.reserve(total);
  dep_flux.reserve(total);
  dep_sigma.reserve(total);
  for (const vector<PhotonDeposit> &r : results) {
    for (const PhotonDeposit &dep : r) {
      dep_pos.push_back(
          make_float4(dep.pos.x, dep.pos.y, dep.pos.z, photon_pack_normal(dep.normal)));
      dep_beam_start.push_back(make_float4(dep.beam_start.x, dep.beam_start.y, dep.beam_start.z, 0.0f));
      dep_flux.push_back(
          make_float4(dep.flux.x,
                      dep.flux.y,
                      dep.flux.z,
                      (float)dep.lightgroup + (dep.volume ? 0.5f : 0.0f)));
      dep_sigma.push_back(make_float4(dep.volume_sigma.x, dep.volume_sigma.y, dep.volume_sigma.z, 0.0f));
    }
  }
  return bin_packed_deposits(
      dep_pos.data(), dep_beam_start.data(), dep_flux.data(), dep_sigma.data(), total, radius, out);
}

}  // namespace

PhotonMap::PhotonMap() : data_(make_unique<PhotonMapData>())
{
  data_->self = this;
}

bool PhotonMap::scatter_published(const device_ptr out_pos,
                                  const device_ptr out_beam_start,
                                  const device_ptr out_flux,
                                  const device_ptr out_beam_sigma,
                                  const device_ptr cell_start_device)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.scatter_published", this);
  /* Scatter the front generation's device-resident deposits into the render
   * device's photon arrays. Called from PathTrace::set_photon_grid between
   * render works; runs on the map's own queue (same pattern as the trace).
   * The cell-count buffer doubles as the scatter cursors: its counts were
   * already consumed into the host prefix table. */
  PhotonMapData &d = *data_;
  const PhotonMapData::Storage &front = d.buf[d.front];
  if (!front.valid || !front.grid.device_resident || !d.gpu_queue || !d.d_cell_count) {
    return false;
  }
  const int total = front.grid.num_photons;
  const int table_size = (int)front.grid.table_size;
  const float inv_cell = front.grid.inv_cell;
  const device_ptr dep_pos_ptr = (device_ptr)front.grid.dep_pos_device;
  const device_ptr dep_beam_start_ptr = (device_ptr)front.grid.dep_beam_start_device;
  const device_ptr dep_flux_ptr = (device_ptr)front.grid.dep_flux_device;
  const device_ptr dep_beam_sigma_ptr = (device_ptr)front.grid.dep_beam_sigma_device;
  const device_ptr cursor_ptr = d.d_cell_count->device_pointer;
  /* The caller just uploaded the prefix table (pageable HtoD): make sure the
   * DMA landed before cursor_init reads it (null-stream vs non-blocking
   * queue - misplaced scatters otherwise). */
  drain_null_stream(d);

  const DeviceKernelArguments iargs(&cell_start_device, &cursor_ptr, &table_size);
  if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_BIN_CURSOR_INIT, table_size, iargs)) {
    LOG_WARNING << "CyclesPlus photon map: GPU cursor init enqueue failed";
    return false;
  }
  const DeviceKernelArguments sargs(&dep_pos_ptr,
                                    &dep_beam_start_ptr,
                                    &dep_flux_ptr,
                                    &dep_beam_sigma_ptr,
                                    &cursor_ptr,
                                    &out_pos,
                                    &out_beam_start,
                                    &out_flux,
                                    &out_beam_sigma,
                                    &table_size,
                                    &inv_cell,
                                    &total);
  if (!d.gpu_queue->enqueue(DEVICE_KERNEL_PHOTON_BIN_SCATTER, total, sargs)) {
    LOG_WARNING << "CyclesPlus photon map: GPU scatter enqueue failed";
    return false;
  }
  if (!d.gpu_queue->synchronize()) {
    LOG_WARNING << "CyclesPlus photon map: GPU scatter synchronize failed";
    return false;
  }
  return true;
}

PhotonMap::~PhotonMap()
{
  gpu_async_abort(*data_);
  data_->cancel = true;
  if (data_->worker.joinable()) {
    data_->worker.join();
  }
}

const PhotonGrid *PhotonMap::grid() const
{
  const PhotonMapData::Storage &front = data_->buf[data_->front];
  return front.valid ? &front.grid : nullptr;
}

void PhotonMap::abort_inflight()
{
  gpu_async_abort(*data_);
}

/* See photon_map.h: interactive sessions park while a final render runs. */
static std::atomic<int> background_renders_active = 0;

void PhotonMap::background_render_begin()
{
  background_renders_active++;
}

void PhotonMap::background_render_end()
{
  background_renders_active--;
}

bool PhotonMap::background_render_active()
{
  return background_renders_active.load() > 0;
}

void PhotonMap::clear()
{
  PhotonMapData &d = *data_;
  gpu_async_abort(d);
  d.cancel = true;
  if (d.worker.joinable()) {
    d.worker.join();
  }
  d.cancel = false;
  d.worker_running = false;
  d.batch_done = false;
  d.scene_valid = false;
  d.buf[0].valid = false;
  d.buf[1].valid = false;
  d.iteration = 0;
}

void PhotonMap::launch_batch_async()
{
  PhotonMapData &d = *data_;
  if (!d.scene_valid || d.gpu_trace) {
    return; /* GPU batches trace synchronously in advance() */
  }
  d.batch_done = false;
  d.worker_running = true;
  const int back = 1 - d.front;
  const float radius = d.radius;
  const uint64_t k = ++d.iteration;

  const int num_photons = batch_size(d, k);
  LOG_INFO << "CyclesPlus photon map: launching batch " << k << " (" << num_photons
           << " photons, " << (d.gpu_trace ? "GPU" : "CPU") << " trace)";
  PhotonMapData *pd = data_.get();
  d.worker = std::thread([pd, back, radius, k, num_photons]() {
    const double start = time_dt();
    const size_t total = pd->gpu_trace ?
        trace_batch_gpu(*pd, num_photons, radius, k, &pd->cancel, pd->buf[back]) :
        trace_batch(pd->scene,
                    pd->lights,
                    pd->targets,
                    pd->light_share,
                    num_photons,
                    radius,
                    batch_seed(k),
                    &pd->cancel,
                    pd->buf[back]);
    const double elapsed = time_dt() - start;
    if (total > 0 && elapsed > 1e-4 && !pd->cancel) {
      /* Happens-before the session's read via batch_done + join. */
      pd->photons_per_sec = (double)num_photons / elapsed;
      pd->buf[back].grid.weight16 = batch_weight16(*pd, num_photons);
    }
    LOG_INFO << "CyclesPlus photon map: batch " << k << " done, " << total << " deposits in "
             << elapsed << "s (cancelled: " << (pd->cancel ? 1 : 0) << ")";
    pd->batch_done = true;
    if (pd->batch_done_callback) {
      pd->batch_done_callback();
    }
  });
}

void PhotonMap::restart(Scene *scene,
                        Progress *progress,
                        const int photons_per_batch,
                        const float detail,
                        const bool interactive,
                        const bool world_changed)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.restart", this, photons_per_batch);
  PhotonMapData &d = *data_;
  gpu_async_abort(d);
  /* Abort the in-flight refinement batch instead of waiting for it. */
  d.cancel = true;
  if (d.worker.joinable()) {
    CCL_PHOTON_PROFILE_SCOPE("photon.worker_join", &d);
    d.worker.join();
  }
  d.cancel = false;
  d.worker_running = false;
  d.batch_done = false;

  /* Session teardown (BlenderSession dtor -> Session::cancel) can land
   * while the render thread is entering a restart - a full rebuild is
   * pointless work on a session that is going away, and touching the
   * scene mid-teardown is exactly the window the 2026-08-05 crash lived
   * in. Worker and GPU stream are stopped above; bail out before any
   * extraction, the buffers stay invalid. */
  if (progress != nullptr && progress->get_cancel()) {
    d.buf[0].valid = false;
    d.buf[1].valid = false;
    return;
  }

  if (world_changed) {
    d.world_cache_valid = false;
  }

  /* Forget the measured trace rate: it sizes the async chunks, and a new
   * scene may trace at a very different speed. The synchronous first batch
   * below re-measures it from pure trace time (no render works in between),
   * which is the only place the rate can be measured honestly. Safe to
   * clear now that batch SIZES no longer depend on it. */
  d.photons_per_sec = 0.0;

  d.scene = PhotonTraceScene();
  d.lights.clear();
  d.targets.clear();
  static const bool gpu_disabled_env = getenv("CYCLESPLUS_PHOTON_GPU_DISABLE") != nullptr;
  /* GPU tracing: CUDA and OptiX devices trace batches on the GPU via the
   * real scene BVH (OptiX through its own raygen entry); CPU devices keep
   * the host tracer. Decided BEFORE the extraction: kernel-traced renders
   * skip the host world entirely (see extract_scene).
   *
   * With several cards enabled scene->device is a MultiDevice wrapper that
   * cannot hand out a queue; photon_pick_trace_device returns one real card
   * to trace on (and the device itself when only one is enabled, so a single
   * GPU takes exactly the same path it always did). */
  int num_render_devices = 0;
  Device *photon_device = photon_pick_trace_device(scene->device, num_render_devices);
  d.multi_device = num_render_devices > 1;
  if (d.trace_device != photon_device) {
    d.gpu_queue.reset();
    d.d_lights.reset();
    d.d_targets.reset();
    d.d_materials.reset();
    d.d_pos.reset();
    d.d_flux.reset();
    d.d_counter.reset();
    d.d_cell_count.reset();
    d.d_sorted_pos.reset();
    d.d_sorted_flux.reset();
    d.d_cell_start_dev.reset();
    d.d_target_yield.reset();
    for (PhotonMapData::Storage &s : d.buf) {
      s.dep_pos_dev.reset();
      s.dep_flux_dev.reset();
      s.valid = false;
    }
    d.trace_device = photon_device;
  }
  /* Validation escape hatch: CYCLESPLUS_PHOTON_GPU_DISABLE forces the host
   * tracer (and the full host world) on GPU devices. */
  /* The OptiX photon raygen is fully back in service: the animation-load
   * GPU hangs were bisected (2026-07-21) to OIDN's GPU denoiser context
   * living next to the OptiX pipeline - session.cpp now redirects that one
   * combination to CPU denoising. The raygen itself ran the 18-frame
   * crash gauntlet clean with denoise off, CPU OIDN and the OptiX
   * denoiser. CYCLESPLUS_PHOTON_OPTIX_HOST=1 remains as an escape hatch
   * forcing the host tracer on OptiX devices only. */
  static const bool optix_host_optout = getenv("CYCLESPLUS_PHOTON_OPTIX_HOST") != nullptr;
  /* Both flags describe the card the photons are traced on, not the wrapper:
   * a multi device reports the type of its sub-devices only when they all
   * match, and it is the sub-device that owns the queue, the BVH and the
   * OptiX photon pipeline we launch into. */
  d.gpu_trace = !gpu_disabled_env && photon_device != nullptr &&
                (photon_device->info.type == DEVICE_CUDA ||
                 (photon_device->info.type == DEVICE_OPTIX && !optix_host_optout));
  d.optix_device = photon_device != nullptr && photon_device->info.type == DEVICE_OPTIX;

  if (d.multi_device) {
    LOG_INFO << "CyclesPlus photon map: " << num_render_devices
             << " render devices - tracing photons on " << photon_device->info.description
             << ", host binning, map replicated to every device";
  }

  const double t_extract0 = time_dt();
  d.scene_valid = extract_scene(
      scene, progress, d.scene, d.lights, d.targets, d.world_cache, d.world_cache_valid,
      d.gpu_trace);
  const double t_extract = time_dt() - t_extract0;
  if (photon_profile_enabled()) {
    photon_profile_record("host", "photon.extract_scene", this, t_extract0, t_extract * 1000.0);
    photon_profile_value("photon.scene_lights", &d, d.lights.size());
    photon_profile_value("photon.scene_targets", &d, d.targets.size());
    photon_profile_value("photon.scene_shaders", &d, scene->shaders.size());
    photon_profile_value("photon.scene_objects", &d, scene->objects.size());
    photon_profile_value("photon.volume_enabled", &d, d.scene.volume_caustics);
    photon_profile_value("photon.gpu_trace", &d, d.gpu_trace);
    photon_profile_value("photon.detail", &d, detail);
  }
  d.photons_per_batch = std::max(photons_per_batch, 1000);
  d.iteration = 0;

  if (!d.scene_valid) {
    d.buf[0].valid = false;
    d.buf[1].valid = false;
    return; /* extract_scene printed the diagnostics */
  }

  if (d.gpu_trace) {
    /* Shader::id is assigned by ShaderManager::device_update_pre and is
     * uninitialized garbage on a shader that has not completed a sync yet
     * (rapid-fire asset imports can land here in that window - crashed
     * 2026-08-05 with a wild heap write). Valid ids are always < the
     * shader count; anything else is skipped and picks up its real slot
     * on the next restart after the sync finishes. */
    const uint num_shaders = (uint)scene->shaders.size();
    int max_id = 0;
    for (Shader *shader : scene->shaders) {
      if (shader->id < num_shaders) {
        max_id = std::max(max_id, (int)shader->id);
      }
    }
    d.gpu_materials.assign(max_id + 1, PhotonTraceMaterial{});
    const CasterPolicy policy = CasterPolicy::from_scene(scene);
    for (Shader *shader : scene->shaders) {
      if (shader->id >= num_shaders || (int)shader->id > max_id) {
        continue;
      }
      const PhotonMaterial m = classify_shader(shader, policy);
      PhotonTraceMaterial &km = d.gpu_materials[shader->id];
      km.kind = m.kind;
      km.ior = m.ior;
      km.dispersion_inv_abbe = m.dispersion_inv_abbe;
      km.rough = m.rough;
      km.transmission = m.transmission;
      km.coat = m.coat;
      km.coat_rough = m.coat_rough;
      km.coat_ior = m.coat_ior;
      km.accurate = (float)m.accurate;
      /* color.w carries the alpha-transparent pass probability,
       * volume_sigma.w the pass_mode (see PhotonMaterial) - both float4
       * padding slots, so the kernel struct does not grow. */
      km.color = make_float4(m.color.x, m.color.y, m.color.z, m.transparent);
      km.volume_sigma = make_float4(
          m.volume_sigma.x, m.volume_sigma.y, m.volume_sigma.z, (float)m.pass_mode);
      km.volume_scatter = make_float4(
          m.volume_sigma_s.x, m.volume_sigma_s.y, m.volume_sigma_s.z, m.volume_g);
    }
  }

  /* KNOWN LIMITATION, measured 2026-08-27 and deliberately left alone.
   *
   * One start radius serves the whole scene and it comes from the LARGEST
   * caster, so an oversized one blurs every caustic in the file. Adding a
   * 24 m glass pane out of frame to the tester's drinks scene lifts it from
   * 0.04 mm to 7.08 mm and smears the sharp filaments beside the glass into
   * a soft glow. The per-pixel radii cannot climb back down: they shrink by
   * about 0.85 per generation, so recovering a factor of 177 would take
   * fifty-odd generations, more than any render provides.
   *
   * Deriving it from a typical caster instead of the largest was tried and
   * made that same scene WORSE - the caustic all but vanished. The radius is
   * also the cell size of the spatial hash, so a small radius with photons
   * spread over tens of metres floods the table and the gather stops finding
   * its own deposits. Radius, grid and photon budget are one problem, not
   * three, and the real answer is a per-photon footprint (photon
   * differentials, branch photon-differentials) rather than a better single
   * number. Until then: an oversized caster in the same file costs sharpness,
   * and 'Selected Materials Only' is the way around it. */
  float max_target_r = 0.0f;
  for (const PhotonTarget &t : d.targets) {
    max_target_r = std::max(max_target_r, t.r);
  }
  /* SPPM wants a generous start radius: the per-pixel radii only ever
   * shrink from here, and a larger disc means 4x lower estimate variance
   * per doubling. The old memoryless schedule used factor 24. Always
   * computed from the FULL batch size - the radius is the grid cell size
   * and the per-pixel start radius, constant across all generations, so
   * the interactive preview batch must use the same one. */
  /* The lower limit exists only to keep a degenerate (zero-size) caster from
   * producing a zero radius - and it has to scale with the scene, like every
   * other term here. It used to be an absolute 1e-4, which is 0.1mm at
   * Blender's default 1 unit = 1m: invisible on a pool, but LARGER than the
   * radius a ring or a watch asks for. Small casters were pinned to a fixed
   * blur, and since the clamp swallowed the Detail divisor as well, Detail 1
   * and Detail 10 produced the identical image at jewelry scale - the
   * control silently stopped working exactly where its target audience
   * needs it. A ten-thousandth of the caster is never reached in practice
   * (Detail is capped at 10, so the formula bottoms out around 1/200th of
   * the caster) while still ruling out a zero. */
  /* The start radius is a number of PIXELS, not a length in metres.
   *
   * It used to come from the largest caster in the scene, and that is what
   * made a 24 m glass pane standing out of frame blur the caustic under a
   * wine glass: the radius jumped from 0.04 mm to 7.08 mm and the per-pixel
   * radii cannot climb back down inside a render (they shrink by about 0.85
   * per generation). Two other cures were built and measured first - a
   * deeper shrink floor did nothing, and deriving the radius from a typical
   * caster instead of the largest made that scene worse, because the radius
   * is also the cell size of the spatial hash and small cells with photons
   * spread over tens of metres flood the table.
   *
   * V-Ray states its caustic search distance in pixels for exactly this
   * reason ("Search dist - specifies the initial photon lookup radius in
   * pixels", default 4, reduced to a quarter over the progressive passes -
   * the same quarter our floor uses). A radius in pixels has no notion of
   * scene scale: a ring and a stadium both get the sharpness the image can
   * actually show, and an oversized object elsewhere in the file cannot
   * reach it.
   *
   * Detail keeps its meaning by scaling the pixel count. The caster-derived
   * formula stays as the fallback for anything without a perspective camera
   * (bakes, orthographic setups), where a pixel has no world size. */
  float r0 = 0.0f;
  const Camera *cam = scene->camera;
  if (cam != nullptr && cam->get_camera_type() == CAMERA_PERSPECTIVE &&
      cam->get_full_width() > 0 && cam->get_fov() > 1e-4f && !d.targets.empty())
  {
    const Transform cam_tfm = cam->get_matrix();
    const float3 cam_p = transform_get_column(&cam_tfm, 3);
    /* Representative depth: the nearest caster is the one whose caustics the
     * shot is usually about, and it is the conservative choice - a nearer
     * reference gives a smaller pixel and therefore a sharper start. */
    float dist = FLT_MAX;
    for (const PhotonTarget &t : d.targets) {
      dist = std::min(dist, std::max(len(t.c - cam_p) - t.r, t.r));
    }
    if (dist > 0.0f && dist < FLT_MAX) {
      const float px_world = 2.0f * dist * tanf(0.5f * cam->get_fov()) /
                             (float)cam->get_full_width();
      r0 = 16.0f * px_world / std::max(detail, 0.1f);
    }
  }
  if (!(r0 > 0.0f) || !isfinite_safe(r0)) {
    r0 = std::max(48.0f * max_target_r /
                      (sqrtf((float)d.photons_per_batch) * std::max(detail, 0.1f)),
                  std::max(max_target_r * 1e-4f, 1e-9f));
  }
  /* Never let the cell grow past the scene's largest caster: the gather only
   * probes 3x3x3 cells, so a cell far bigger than the geometry would just
   * waste the neighbourhood. */
  r0 = std::min(r0, std::max(max_target_r, 1e-9f));
  d.radius = r0;

  /* Diagnostic: the gather radius is derived from the LARGEST caster, so one
   * oversized target sets the blur for every caustic in the scene. Print what
   * actually drives it. */
  if (LOG_IS_ON(LOG_LEVEL_INFO)) {
    vector<float> radii;
    radii.reserve(d.targets.size());
    for (const PhotonTarget &t : d.targets) {
      radii.push_back(t.r);
    }
    std::sort(radii.begin(), radii.end());
    string top;
    for (size_t i = radii.size(); i-- > 0 && top.size() < 60;) {
      top += string_printf("%.3f ", (double)radii[i]);
    }
    const float median = radii.empty() ? 0.0f : radii[radii.size() / 2];
    LOG_INFO << string_printf(
        "CyclesPlus photon map: radius r0=%.5f from max_target_r=%.3f "
        "(%d targets, median %.3f, largest: %s)",
        (double)r0, (double)max_target_r, (int)d.targets.size(), (double)median, top.c_str());
  }

  /* Fresh extraction = fresh target list: the yield EMA indices would not
   * match anymore. MUST clear BEFORE the pilot - build_guided_targets reads
   * the EMA whenever the size matches, so on an animation frame with an
   * unchanged target count a late clear let frame N's accumulated yield
   * steer frame N+1's pilot aim and thereby its light shares (carried-over
   * run state re-rolling every photon path: the flicker failure shape). */
  d.target_yield_ema.clear();

  /* Photon budget per light from a quick pilot measurement - on the GPU
   * for kernel-traced renders (the host world does not exist there). A
   * failed GPU pilot degrades to floor-uniform shares: unbiased, just
   * suboptimal budgets. */
  const double t_pilot0 = time_dt();
  if (d.gpu_trace) {
    if (!measure_light_shares_gpu(d)) {
      normalize_light_shares(d.light_share);
    }
  }
  else {
    measure_light_shares(d.scene, d.lights, d.targets, d.light_share, &d.cancel);
  }
  LOG_INFO << string_printf("CyclesPlus photon map: startup timing: extract %.3fs pilot %.3fs",
                            t_extract,
                            time_dt() - t_pilot0);

  /* First batch synchronously, so caustics are visible right after the
   * first render work gathers it (asynchronous publication could miss a
   * short render entirely, and the gather may only run between works).
   * Interactive sessions trace a small preview batch instead: same radius,
   * same estimator - each batch normalizes its own flux - just grainier,
   * so viewport edits unfreeze quickly and the async batch ramp refines
   * the caustics from there. */
  d.interactive = interactive;
  const int first_batch = batch_size(d, 1);
  const uint64_t k = ++d.iteration;
  const int back = 1 - d.front;

  /* GPU final renders launch the first batch ASYNCHRONOUSLY: the first
   * samples render plain path tracing for a few hundred ms and the caustic
   * joins with the first gather after the batch lands. This removes the
   * multi-second synchronous startup the profiling showed on every F12;
   * the estimator is unaffected (each batch normalizes its own flux, and
   * the refinement samples keep the caustic maturing). Interactive
   * sessions keep the small SYNCHRONOUS preview batch: viewport edits
   * must never show a caustic-less frame. */
  /* Deterministic pacing (see batch_size): the async first batch lands at a
   * wall-clock-dependent point, so WHICH work consumes it (and whether it is
   * ever consumed) varied between runs of the same frame. Final renders now
   * trace the first batch synchronously - it is 1/16th of a full batch, so
   * the startup cost stays far below the old multi-second full-batch sync.
   * CYCLESPLUS_PHOTON_TIMED_BATCHES=1 restores the async start. */
  static const bool timed_batches_async = getenv("CYCLESPLUS_PHOTON_TIMED_BATCHES") != nullptr;
  if (d.gpu_trace && !interactive && timed_batches_async &&
      gpu_async_begin(d, first_batch, d.radius, k, d.buf[back])) {
    gpu_async_step(d, d.buf[back]);
    LOG_INFO << string_printf(
        "CyclesPlus photon map: progressive, first batch async (%d photons), "
        "start radius %.5f, %zu tris, %zu lights, %zu targets",
        first_batch,
        (double)r0,
        d.scene.tri_v.size() / 3,
        d.lights.size(),
        d.targets.size());
    return;
  }

  const double first_start = time_dt();
  const size_t total = d.gpu_trace ?
      trace_batch_gpu(d, first_batch, r0, k, &d.cancel, d.buf[back]) :
      trace_batch(d.scene, d.lights, d.targets, d.light_share, first_batch, r0, batch_seed(k), &d.cancel, d.buf[back]);
  const double first_elapsed = time_dt() - first_start;
  if (total > 0) {
    /* The batch's WEIGHT and steady flag follow from its SIZE alone - never
     * from how long it took. They used to sit inside the timing branch
     * below, so a first batch that traced in under 0.1ms would have kept
     * whatever the recycled Storage still held (weight16 16, steady 1): a
     * small preview batch weighted like a full one in the tau average, and
     * adaptive sampling free to retire pixels whose caustic had not arrived.
     * Same trap as the batch sizes themselves (see batch_size) - anything a
     * wall clock touches must be advisory only. */
    d.buf[back].grid.weight16 = batch_weight16(d, first_batch);
    d.buf[back].grid.steady = (first_batch >= d.photons_per_batch) ? 1 : 0;
  }
  if (total > 0 && first_elapsed > 1e-4) {
    /* Advisory only: sizes the async chunks. */
    d.photons_per_sec = (double)first_batch / first_elapsed;
  }
  if (total > 0) {
    d.front = back;
    d.buf[d.front].grid.shader_caster = d.scene.shader_caster.data();
    d.buf[d.front].grid.num_shader_caster = (int)d.scene.shader_caster.size();
    d.buf[d.front].grid.generation = ++d.generation;
  }
  else {
    d.buf[0].valid = false;
    d.buf[1].valid = false;
  }
  LOG_INFO << string_printf(
      "CyclesPlus photon map: progressive, %zu deposits from first batch (%d photons), "
      "start radius %.5f, %zu tris, %zu lights, %zu targets",
      total,
      first_batch,
      (double)r0,
      d.scene.tri_v.size() / 3,
      d.lights.size(),
      d.targets.size());

  /* Refinement batches run in the background from here on. */
  launch_batch_async();
}

void PhotonMap::advance(const bool allow_launch, const bool generation_pending)
{
  CCL_PHOTON_PROFILE_SCOPE("photon.advance", this);
  photon_profile_value("photon.generation_pending", this, generation_pending);
  PhotonMapData &d = *data_;

  if (d.gpu_trace) {
    /* Async generation in flight (interactive): push the next chunk or, once
     * all chunks are enqueued, finish + publish. No synchronization while
     * chunks remain - the photon stream overlaps the render. */
    if (d.gpu_inflight) {
      const int back = 1 - d.front;
      PhotonMapData::Storage &out = d.buf[back];
      if (d.cancel.load(std::memory_order_relaxed) || !d.scene_valid) {
        gpu_async_abort(d);
        return;
      }
      if (d.gpu_off < d.gpu_total) {
        gpu_async_step(d, out);
        return;
      }
      const size_t total = gpu_async_finish(d, out);
      if (total == (size_t)-1) {
        return; /* staging grown, generation restarts (still inflight) */
      }
      d.gpu_inflight = false;
      if (total > 0) {
        const double elapsed = time_dt() - d.gpu_begin_time;
        if (photon_profile_enabled()) {
          photon_profile_record("host", "photon.async_publication_latency", &d,
                                d.gpu_begin_time, elapsed * 1000.0, d.gpu_num_photons);
        }
        if (elapsed > 1e-4) {
          /* This wall clock spans every render work the generation
           * overlapped, so it is a LOWER BOUND on the trace rate, not a
           * measurement of it - typically ~100x low. Letting it lower the
           * estimate was a spiral: the rate feeds the async CHUNK size, a
           * low rate makes tiny chunks, tiny chunks stretch the generation
           * over more works, which reads even lower (measured: 12.5M photons
           * = 0.1s of GPU work dribbled out in 49 chunks over 10s of wall
           * clock). Only ever raise it; restart() clears it so a new scene
           * re-measures from the synchronous first batch. */
          d.photons_per_sec = std::max(d.photons_per_sec,
                                       (double)d.gpu_num_photons / elapsed);
        }
        out.grid.weight16 = batch_weight16(d, d.gpu_num_photons);
        out.grid.steady = (d.gpu_batch_k >= 6 || d.gpu_num_photons >= d.photons_per_batch) ? 1 :
                                                                                             0;
        d.front = back;
        d.buf[d.front].grid.shader_caster = d.scene.shader_caster.data();
        d.buf[d.front].grid.num_shader_caster = (int)d.scene.shader_caster.size();
        d.buf[d.front].grid.generation = ++d.generation;
        d.optix_next_launch = time_dt() + 0.1;
        LOG_INFO << "CyclesPlus photon map: published async generation " << d.generation
                 << " (" << total << " deposits, " << d.gpu_num_photons << " photons, "
                 << elapsed << "s overlapped)";
      }
      return;
    }

    /* GPU tracing between render works, all host calls on the render thread
     * (concurrent device use from a worker thread poisons the CUDA context).
     * Interactive sessions run the generation asynchronously across works;
     * final renders (and debug modes) keep the synchronous path. Trace a new
     * generation only when the previous one was consumed and refinement is
     * still wanted. */
    if (!allow_launch || generation_pending || !d.scene_valid) {
      return;
    }
    /* OptiX breather: refinement generations wait ~100ms after the previous
     * publish. Refinement is throwaway-cheap to delay, and the pause is the
     * second half of the driver-relief throttle (see optix_device above).
     * The burst exception mirrors the chunk throttle: the first generations
     * after a reset (the visible "caustics resolving") run back to back.
     * Viewport only - F12 pacing must stay deterministic. */
    if (d.interactive && d.optix_device && !optix_fullbore() && d.iteration >= 6 &&
        time_dt() < d.optix_next_launch)
    {
      return;
    }
    const uint64_t k = ++d.iteration;
    const int back = 1 - d.front;
    const int num_photons = batch_size(d, k);
    if (d.interactive && gpu_async_begin(d, num_photons, d.radius, k, d.buf[back])) {
      gpu_async_step(d, d.buf[back]);
      return;
    }
    const double start = time_dt();
    const size_t total = trace_batch_gpu(d, num_photons, d.radius, k, &d.cancel, d.buf[back]);
    /* elapsed BEFORE the breather: it feeds photons_per_sec, and a sleep in
     * the measurement is exactly the wall-clock trap this file documents. */
    const double elapsed = time_dt() - start;
    /* OptiX final renders: breathe ~100ms after every completed generation.
     * The 2M launch slices alone stretched the driver's survival from
     * 40s-4min to 8-28min under the chair4 native soak, but did not kill
     * the failure; slices PLUS a generation breather is the recipe that
     * demonstrably held 45+ minute viewport marathons (2026-08-05). The
     * sync path is blocking, so a sleep cannot reorder anything -
     * deterministic pacing and the image are untouched; cost is ~100ms x
     * ~32 generations on a minute-long native frame (~5%).
     * CYCLESPLUS_PHOTON_F12_NO_BREATHER=1 disables it (benchmarks/bisection). */
    /* OPT-IN since 2026-08-11: the breather never saved the one machine
     * that kept dying (hardware/system suspect there; every pacing measure
     * only shifted the death time), so healthy machines do not pay the ~5%
     * animation cost. First rung of the support ladder for "black screens
     * during final renders": CYCLESPLUS_PHOTON_F12_BREATHER=1. */
    if (d.optix_device && !d.interactive && total > 0) {
      static const bool breather = getenv("CYCLESPLUS_PHOTON_F12_BREATHER") != nullptr;
      if (breather) {
        time_sleep(0.1);
      }
    }
    if (total > 0) {
      if (elapsed > 1e-4) {
        d.photons_per_sec = (double)num_photons / elapsed;
      }
      d.buf[back].grid.weight16 = batch_weight16(d, num_photons);
      /* The exponential ramp caps its time budget from k=6 on; from there
       * (or at full batch size) generations are steady. */
      d.buf[back].grid.steady = (k >= 6 || num_photons >= d.photons_per_batch) ? 1 : 0;
      d.front = back;
      d.buf[d.front].grid.shader_caster = d.scene.shader_caster.data();
      d.buf[d.front].grid.num_shader_caster = (int)d.scene.shader_caster.size();
      d.buf[d.front].grid.generation = ++d.generation;
      LOG_INFO << "CyclesPlus photon map: published generation " << d.generation << " ("
               << total << " deposits, " << num_photons << " photons, " << elapsed << "s GPU)";
    }
    else if (!(d.cancel.load(std::memory_order_relaxed))) {
      LOG_WARNING << "CyclesPlus photon map: batch produced no deposits, stopping refinement";
      d.scene_valid = false;
    }
    return;
  }

  if (d.worker_running) {
    if (!d.batch_done) {
      return; /* batch still tracing - keep showing the current grid */
    }
    d.worker.join();
    d.worker_running = false;
    const int back = 1 - d.front;
    if (d.buf[back].valid) {
      d.front = back;
      d.buf[d.front].grid.shader_caster = d.scene.shader_caster.data();
      d.buf[d.front].grid.num_shader_caster = (int)d.scene.shader_caster.size();
      d.buf[d.front].grid.generation = ++d.generation;
      LOG_INFO << "CyclesPlus photon map: published generation " << d.generation;
    }
    else {
      /* An uncancelled batch with zero deposits means this scene simply
       * produces none - stop the pipeline instead of relaunching forever.
       * The next restart() tries again. */
      LOG_WARNING << "CyclesPlus photon map: batch produced no deposits, stopping refinement";
      d.scene_valid = false;
      return;
    }
  }
  /* Launch the next batch, or resume a pipeline that was stopped when the
   * idle budget ran out. */
  if (allow_launch) {
    launch_batch_async();
  }
}

void PhotonMap::set_batch_done_callback(std::function<void()> callback)
{
  data_->batch_done_callback = std::move(callback);
}

CCL_NAMESPACE_END
