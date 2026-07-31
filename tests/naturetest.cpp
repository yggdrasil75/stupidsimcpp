// naturetest.cpp - Test Scene 2: "plants in sunlight"
//
// Where materialtest.cpp stresses metals and dielectrics in a cornell-style
// box, this scene stresses the things the sim is actually about:
//   - dirt floor        (rough, noisy heightfield terrain)
//   - sunlight          (skybox celestial body, warm, hard-ish shadows)
//   - moonlight         (second sky body, cool, dim - rendered as a night pass)
//   - bark              (very rough, dark, noisy albedo, zero metallic)
//   - leaves            (translucent: transmission + chlorophyll absorption,
//                        so backlit leaves glow green against the sun)
//   - dust              (two-tier trick, see DUST section below - no trillions
//                        of voxels required)
//
// DUST without trillions of voxels:
//   v1 tried filling the air with big semi-transparent voxels: fails, because
//   every transmissive interface in this renderer rolls fresnel + a GGX lobe
//   + hero-wavelength dispersion, so stacked haze hits turn to purple noise.
//   v2 tried a CPU fog post pass: global-only and burned CPU time.
//   v3 (current): GPU fog volumes. Octree::addFogVolume(min, max, density,
//   scatterColor, absorption) places a world-space AABB of participating
//   medium. It is evaluated inside wf_extend.comp's existing medium machinery
//   (analytic ray/box clip + max-channel distance sampling, isotropic phase),
//   and wf_shadow.comp attenuates shadow rays through it, so light shafts are
//   actually path-traced with correct shadowed edges. Zero voxels, zero CPU,
//   and fullylocation-controlled: put boxes only where you want dust.
//   The "mote" tier stays: a few thousand tiny opaque voxels near the ground
//   and in the shaft cone sell the visible-particle look up close.
//
// Three lighting passes are rendered from the same geometry:
//   day    - sun high, blue sky
//   sunset - sun low + orange, warm sky (best pass for dust shafts)
//   night  - sun below horizon, moon up, dark blue sky
//
// Output: output/nature/{fast,day,sunset,night}_*.bmp

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <functional>
#include <iomanip>
#include <sstream>

#include "../eigen/Eigen/Dense"
#include "../util/grid/camera.hpp"
#include "../util/grid/grid3eigen.hpp"
#include "../util/grid/grid3render.cpp"
#include "../util/grid/grid3physics.cpp"
#include "../util/output/frame.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/timing_decorator.hpp"
#include "../util/timing_decorator.cpp"

using Vec3 = Eigen::Vector3f;

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------
namespace cfg {
    // world
    constexpr float worldHalf   = 16.0f;   // octree bounds +/- this
    constexpr float groundHalf  = 16.0f;   // dirt extends +/- this in x/y

    // voxel steps
    constexpr float dirtStep    = 0.1f;
    constexpr float barkStep    = 0.03f;
    constexpr float leafStep    = 0.03f;

    // dust motes (haze is handled by octree.setFog - see file header)
    constexpr int   moteCount   = 25000;
    constexpr float moteSize    = 0.01f;
    constexpr float moteTop     = 11.0f;   // motes fill ground..this height

    // render
    constexpr int   width       = 1920;
    constexpr int   height      = 1080;
    constexpr int   samples     = 600;
    constexpr int   bounces     = 16;

    // object ids
    constexpr int OID_DIRT  = 1;
    constexpr int OID_ROCK  = 2;
    constexpr int OID_BARK  = 3;
    constexpr int OID_LEAF  = 4;
    constexpr int OID_HAZE  = 5;
    constexpr int OID_MOTE  = 6;
    constexpr int OID_GRASS = 7;

    // sky body ids
    constexpr int SKY_SUN  = 1;
    constexpr int SKY_MOON = 2;

    // animation
    constexpr bool  renderAnimation    = true; // Toggle camera animation sequence
    constexpr int   animFps            = 24;
    constexpr float animTimePerSegment = 3.0f; // Seconds spent moving between each consecutive view
}

static std::mt19937 rng(20260704);
static float frand(float a, float b) {
    std::uniform_real_distribution<float> d(a, b);
    return d(rng);
}

// ---------------------------------------------------------------------------
// View & Spline Utilities
// ---------------------------------------------------------------------------
struct View {
    std::string name;
    Vec3 origin;
    Vec3 target;
    float fov;      // explicit - the Camera default of 80 deg is fisheye
};

// Catmull-Rom spline interpolation (works for Vec3 and float)
template <typename T>
T catmullRom(const T& p0, const T& p1, const T& p2, const T& p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return 0.5f * (
        (2.0f * p1) +
        (-p0 + p2) * t +
        (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
        (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3
    );
}

// Handles boundary logic to seamlessly expand the view list and prevent wild loops at edges
View getVirtualView(const std::vector<View>& views, int idx) {
    if (idx < 0) {
        View v = views[0];
        View v1 = views[1];
        // Linearly extrapolate backwards
        v.origin = v.origin - (v1.origin - v.origin);
        v.target = v.target - (v1.target - v.target);
        v.fov = v.fov - (v1.fov - v.fov);
        return v;
    }
    if (idx >= static_cast<int>(views.size())) {
        View vn = views.back();
        View vn1 = views[views.size()-2];
        // Linearly extrapolate forwards
        vn.origin = vn.origin + (vn.origin - vn1.origin);
        vn.target = vn.target + (vn.target - vn1.target);
        vn.fov = vn.fov + (vn.fov - vn1.fov);
        return vn;
    }
    return views[idx];
}


// ---------------------------------------------------------------------------
// Terrain: noisy dirt heightfield
// ---------------------------------------------------------------------------
static PNoise2 noise(1337);

// ground surface height at (x, y). z-up, roughly around z = 0.
static float groundHeight(float x, float y) {
    float h = noise.fractalNoise(Eigen::Vector2f(x * 0.08f, y * 0.08f), 4, 0.5f, 2.0f) * 1.1f;
    h += noise.fractalNoise(Eigen::Vector2f(x * 0.45f + 100.0f, y * 0.45f), 2, 0.5f, 2.0f) * 0.18f; // small clods
    return h;
}

static void buildDirtFloor(Grid::Octree<int>& octree) {
    const float s = cfg::dirtStep;
    const Vec3 dirtA(0.36f, 0.24f, 0.14f); // moist brown
    const Vec3 dirtB(0.52f, 0.38f, 0.24f); // dry tan

    size_t count = 0;
    for (float x = -cfg::groundHalf; x <= cfg::groundHalf; x += s) {
        for (float y = -cfg::groundHalf; y <= cfg::groundHalf; y += s) {
            float h = groundHeight(x, y);
            // only the top few layers - nobody can see the core of the earth
            for (int layer = 0; layer < 3; ++layer) {
                float z = h - layer * s;
                // per-voxel tonal variation so it reads as dirt, not plastic
                float t = noise.normalizedNoise(Vec3(x * 1.7f, y * 1.7f, z * 1.7f));
                Vec3 albedo = dirtA + (dirtB - dirtA) * t;
                albedo *= frand(0.92f, 1.05f);
                octree.insert(1, Vec3(x + frand(-0.01f, 0.01f), y + frand(-0.01f, 0.01f), z),
                              true, albedo, s, true, cfg::OID_DIRT,
                              /*emit*/0.0f, /*rough*/0.97f, /*metal*/0.0f,
                              /*trans*/0.0f, /*ior*/1.45f);
                ++count;
            }
        }
    }

    // a few half-buried rocks
    for (int i = 0; i < 14; ++i) {
        Vec3 c(frand(-cfg::groundHalf * 0.9f, cfg::groundHalf * 0.9f),
               frand(-cfg::groundHalf * 0.9f, cfg::groundHalf * 0.9f), 0.0f);
        if (c.head<2>().norm() < 2.5f) continue; // keep the tree base clear
        c.z() = groundHeight(c.x(), c.y());
        float r = frand(0.25f, 0.7f);
        float gray = frand(0.35f, 0.55f);
        for (float x = -r; x <= r; x += s)
            for (float y = -r; y <= r; y += s)
                for (float z = -r * 0.5f; z <= r * 0.8f; z += s) {
                    Vec3 p(x, y, z);
                    // squashed ellipsoid
                    if ((p.cwiseQuotient(Vec3(r, r, r * 0.8f))).squaredNorm() > 1.0f) continue;
                    float v = gray * frand(0.9f, 1.1f);
                    octree.insert(1, c + p, true, Vec3(v, v, v * 1.03f), s, true, cfg::OID_ROCK,
                                  0.0f, 0.85f, 0.0f, 0.0f, 1.5f);
                }
    }

    // sparse dry-grass tufts (cheap: short 2-3 voxel columns)
    for (int i = 0; i < 500; ++i) {
        float x = frand(-cfg::groundHalf, cfg::groundHalf);
        float y = frand(-cfg::groundHalf, cfg::groundHalf);
        float z = groundHeight(x, y);
        int blades = 2 + (rng() % 2);
        Vec3 col(0.45f + frand(-0.05f, 0.05f), 0.42f + frand(-0.05f, 0.05f), 0.15f);
        for (int b = 0; b < blades; ++b)
            octree.insert(1, Vec3(x + frand(-0.05f, 0.05f), y + frand(-0.05f, 0.05f), z + 0.08f + b * 0.09f),
                          true, col, 0.06f, true, cfg::OID_GRASS,
                          0.0f, 0.8f, 0.0f, 0.0f, 1.4f);
    }

    std::cout << "  dirt voxels: ~" << count << std::endl;
}

// ---------------------------------------------------------------------------
// Tree: recursive branches (bark) + leaf clusters at the tips
// ---------------------------------------------------------------------------
static void placeLeafCluster(Grid::Octree<int>& octree, const Vec3& center, float radius) {
    const Vec3 leafA(0.13f, 0.42f, 0.08f); // deep green
    const Vec3 leafB(0.35f, 0.60f, 0.12f); // sunlit yellow-green
    // chlorophyll: absorbs red/blue, passes green -> backlit leaves glow green
    const Vec3 leafAbsorp(0.9f, 0.15f, 0.85f);

    // fill ~35% of an ellipsoid with leaf voxels so light leaks through gaps
    int target = static_cast<int>((radius * radius * radius) * 4.19f * 0.35f
                                  / (cfg::leafStep * cfg::leafStep * cfg::leafStep));
    for (int i = 0; i < target; ++i) {
        Vec3 dir(frand(-1, 1), frand(-1, 1), frand(-1, 1));
        if (dir.squaredNorm() > 1.0f) { --i; continue; }
        Vec3 p = center + dir.cwiseProduct(Vec3(radius, radius, radius * 0.8f));
        float t = frand(0.0f, 1.0f);
        Vec3 albedo = leafA + (leafB - leafA) * t;
        octree.insert(1, p, true, albedo, cfg::leafStep, true, cfg::OID_LEAF,
                      /*emit*/0.0f, /*rough*/0.55f, /*metal*/0.0f,
                      /*trans*/0.35f, /*ior*/1.4f, leafAbsorp);
    }
}

static void growBranch(Grid::Octree<int>& octree, const Vec3& base, Vec3 dir,
                       float length, float radius, int depth) {
    dir.normalize();
    const float s = cfg::barkStep;
    const Vec3 barkA(0.23f, 0.15f, 0.09f);
    const Vec3 barkB(0.38f, 0.28f, 0.18f);

    // build an orthonormal frame around the branch axis
    Vec3 up = std::abs(dir.z()) < 0.9f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
    Vec3 t1 = dir.cross(up).normalized();
    Vec3 t2 = dir.cross(t1).normalized();

    int steps = std::max(1, static_cast<int>(length / s));
    Vec3 tip = base;
    for (int i = 0; i <= steps; ++i) {
        float f = static_cast<float>(i) / steps;
        Vec3 c = base + dir * (f * length);
        tip = c;
        float r = radius * (1.0f - 0.35f * f); // taper along the segment

        // disc of voxels perpendicular to the axis
        for (float a = -r; a <= r; a += s) {
            for (float b = -r; b <= r; b += s) {
                if (a * a + b * b > r * r) continue;
                // hollow out thick trunks: only a ~2-voxel shell is visible
                if (r > 3.0f * s && (a * a + b * b) < (r - 2.0f * s) * (r - 2.0f * s)) continue;
                Vec3 p = c + t1 * a + t2 * b;
                // bark ridges: high-frequency noise along the surface
                float n = noise.normalizedNoise(Vec3(p * 6.0f));
                Vec3 albedo = barkA + (barkB - barkA) * n;
                float rough = 0.9f + 0.08f * n;
                octree.insert(1, p, true, albedo, s, true, cfg::OID_BARK,
                              0.0f, rough, 0.0f, 0.0f, 1.5f);
            }
        }
    }

    if (depth >= 5 || radius < 0.05f) {
        placeLeafCluster(octree, tip + dir * 0.3f, frand(0.7f, 1.1f));
        return;
    }

    // some mid-depth branches also carry foliage so the canopy has interior
    if (depth >= 3 && frand(0, 1) < 0.4f)
        placeLeafCluster(octree, tip, frand(0.5f, 0.8f));

    int children = 2 + (rng() % 2); // 2-3 child branches
    for (int i = 0; i < children; ++i) {
        // pitch away from the parent axis, biased upward
        float spread = frand(0.35f, 0.75f);
        Vec3 lateral = (t1 * frand(-1, 1) + t2 * frand(-1, 1)).normalized();
        Vec3 childDir = (dir + lateral * spread + Vec3(0, 0, 0.25f)).normalized();
        growBranch(octree, tip, childDir,
                   length * frand(0.6f, 0.75f),
                   radius * frand(0.55f, 0.7f),
                   depth + 1);
    }
}

static void buildTree(Grid::Octree<int>& octree) {
    Vec3 base(0.0f, 0.0f, groundHeight(0.0f, 0.0f) - 0.3f);
    // slight lean makes it read as organic instead of a lamp post
    Vec3 trunkDir = Vec3(0.08f, -0.05f, 1.0f).normalized();
    growBranch(octree, base, trunkDir, /*length*/4.2f, /*radius*/0.55f, /*depth*/0);
}

// ---------------------------------------------------------------------------
// Dust (see file header for the trick)
// ---------------------------------------------------------------------------
static void buildDust(Grid::Octree<int>& octree, const Vec3& sunDir) {
    // Opaque motes only. Half scattered everywhere, half seeded inside the
    // cone of the sun shaft through the canopy so they catch the light.
    // The volumetric haze itself is analytic fog (octree.setFog), not voxels.
    Vec3 shaftEntry(0.0f, 0.0f, 8.0f); // roughly where light punches the canopy
    size_t placed = 0;
    for (int i = 0; i < cfg::moteCount; ++i) {
        Vec3 p;
        if (i % 2 == 0) {
            p = Vec3(frand(-cfg::groundHalf, cfg::groundHalf),
                     frand(-cfg::groundHalf, cfg::groundHalf),
                     frand(0.2f, cfg::moteTop));
            if (p.z() < groundHeight(p.x(), p.y()) + 0.2f) continue;
        } else {
            // walk down-sun from the canopy gap, with jitter
            float d = frand(0.0f, 9.0f);
            p = shaftEntry - sunDir * d + Vec3(frand(-1.2f, 1.2f), frand(-1.2f, 1.2f), 0.0f);
            if (p.z() < groundHeight(p.x(), p.y()) + 0.15f || p.z() > cfg::moteTop) continue;
        }
        float bright = frand(0.75f, 1.0f);
        octree.insert(1, p, true, Vec3(bright, bright * 0.97f, bright * 0.9f),
                      cfg::moteSize, true, cfg::OID_MOTE,
                      0.0f, 0.9f, 0.0f, 0.8f, 1.45f);
        ++placed;
    }
    std::cout << "  dust motes: " << placed << " (haze = GPU fog volumes, 0 voxels)" << std::endl;
}

// ---------------------------------------------------------------------------
// Lighting passes
// ---------------------------------------------------------------------------
struct LightingPass {
    std::string name;
    Vec3 sunDir;        // unit, +z = up. z < 0 means below the horizon
    Vec3 sunColor;      // 0..1
    uint8_t sunEmit;
    Vec3 moonDir;
    uint8_t moonEmit;
    Vec3 sky;           // background color
    Vec3 skylight;      // ambient
    // dust fog volumes (the haze tier) - density per box, see applyLighting
    float mistDensity;    // low ground-mist slab across the scene
    float shaftDensity;   // denser box around the canopy for light shafts
    Vec3 fogTint;         // scattering albedo of the dust
};

static void applyLighting(Grid::Octree<int>& octree, const LightingPass& p) {
    octree.setBackgroundColor(p.sky);
    octree.setSkylight(p.skylight);
    auto to255 = [](float v) { return static_cast<uint8_t>(std::clamp(v * 255.0f, 0.0f, 255.0f)); };

    // sun: real angular radius is ~0.267 deg; we use ~1.5 deg so voxel-scale
    // shadows get a little penumbra instead of pure stair-steps.
    octree.addSkyBody(cfg::SKY_SUN, p.sunDir, 1.5f * static_cast<float>(M_PI) / 180.0f,
                      to255(p.sunColor.x()), to255(p.sunColor.y()), to255(p.sunColor.z()), p.sunEmit);
    octree.addSkyBody(cfg::SKY_MOON, p.moonDir, 1.0f * static_cast<float>(M_PI) / 180.0f,
                      200, 205, 235, p.moonEmit);
    octree.bakeSkyBody(cfg::SKY_SUN);
    octree.bakeSkyBody(cfg::SKY_MOON);

    // dust: fog volume boxes, placed only where the scene wants haze.
    // A thin mist slab hugging the ground plus a denser box wrapping the
    // canopy so sun rays punching through the leaves scatter into shafts.
    octree.clearFogVolumes();
    if (p.mistDensity > 0.0f)
        octree.addFogVolume(Vec3(-cfg::groundHalf, -cfg::groundHalf, -1.0f),
                            Vec3( cfg::groundHalf,  cfg::groundHalf,  2.5f),
                            p.mistDensity, p.fogTint);
    if (p.shaftDensity > 0.0f)
        octree.addFogVolume(Vec3(-4.5f, -4.5f, 2.0f),
                            Vec3( 4.5f,  4.5f, 9.5f),
                            p.shaftDensity, p.fogTint);
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "Building nature test scene (test scene 2)..." << std::endl;

    Vec3 minB(-cfg::worldHalf, -cfg::worldHalf, -cfg::worldHalf);
    Vec3 maxB( cfg::worldHalf,  cfg::worldHalf,  cfg::worldHalf);
    Grid::Octree<int> octree(minB, maxB, "output/naturescene", 32);

    // no dynamic physics in this scene - everything is STATIC. Gravity setup
    // kept for parity in case someone flips a rock to RIGID to test knockdown.
    octree.setphys_gravityCenter(Vec3(0.0f, 0.0f, -1000.0f));

    // primary sun direction, shared by scene-building (dust shaft seeding) and
    // the "day" pass. Points FROM the scene TOWARD the sun.
    const Vec3 daySun   = Vec3( 0.45f,  0.30f, 0.85f).normalized();
    const Vec3 setSun   = Vec3( 0.90f,  0.35f, 0.16f).normalized();
    const Vec3 downDir  = Vec3( 0.0f,   0.0f, -1.0f);
    const Vec3 nightMoon = Vec3(-0.35f, 0.25f, 0.75f).normalized();

    std::cout << "Terrain..." << std::endl;   buildDirtFloor(octree);
    std::cout << "Tree..."    << std::endl;   buildTree(octree);
    std::cout << "Dust..."    << std::endl;   buildDust(octree, daySun);

    octree.setLODMinDistance(1024);
    octree.setLODFalloff(0.01f);
    octree.setMaxDistance(4096);
    octree.printStats();

    std::vector<LightingPass> passes = {
        { "day",
          daySun, Vec3(1.0f, 0.96f, 0.88f), 255,
          downDir /*moon below horizon*/, 0,
          Vec3(0.47f, 0.71f, 0.92f), Vec3(0.20f, 0.26f, 0.34f),
          /*mist*/0.015f, /*shaft*/0.035f, Vec3(0.90f, 0.87f, 0.80f) },
        { "sunset",
          setSun, Vec3(1.0f, 0.55f, 0.22f), 255,
          downDir, 0,
          Vec3(0.72f, 0.40f, 0.24f), Vec3(0.16f, 0.10f, 0.09f),
          /*mist*/0.045f, /*shaft*/0.070f, Vec3(0.92f, 0.80f, 0.65f) },
        { "night",
          downDir /*sun below horizon*/, Vec3(1.0f, 0.96f, 0.88f), 0,
          nightMoon, 140,
          Vec3(0.015f, 0.02f, 0.05f), Vec3(0.010f, 0.013f, 0.025f),
          /*mist*/0.020f, /*shaft*/0.0f, Vec3(0.70f, 0.75f, 0.88f) },
    };

    std::vector<View> views = {
        { "wide",     Vec3(11.5f, -10.0f, 3.5f), Vec3(0.0f, 0.0f, 3.5f), 55.0f }, // whole tree
        { "low",      Vec3( 6.0f,  -7.5f, 0.9f), Vec3(0.0f, 0.0f, 4.5f), 50.0f }, // looking up, backlit leaves
        { "shaft",    Vec3(-7.0f,   6.0f, 1.8f), Vec3(1.5f, -0.5f, 3.5f), 55.0f }, // down-sun, dust glow
        { "closeup",  Vec3( 3.0f,  -2.6f, 2.0f), Vec3(0.0f, 0.0f, 2.6f), 45.0f }, // bark + dirt detail
        { "top",      Vec3( 6.5f,   6.5f, 12.5f), Vec3(0.0f, 0.0f, 3.5f), 50.0f }, // canopy from above
    };

    // ---- fast preview of every view (day lighting) -------------------------
    applyLighting(octree, passes[0]);
    for (const auto& v : views) {
        ScopedFunctionTimer meh("Fast pass");
        Camera cam;
        cam.origin = v.origin;
        cam.direction = (v.target - v.origin).normalized();
        cam.up = Vec3(0, 0, 1);
        cam.fov = v.fov;
        std::cout << "Fast preview: " << v.name << std::endl;
        frame out = octree.fastRenderFrameVulkan(cam, cfg::height, cfg::width, frame::colormap::RGB);
        BMPWriter::saveBMP("output/nature/fast_" + v.name + ".bmp", out);
    }
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    // ---- quality passes: day / sunset / night ------------------------------
    for (const auto& pass : passes) {
        applyLighting(octree, pass);
        for (const auto& v : views) {
            ScopedFunctionTimer meh("Quality pass");
            Camera cam;
            cam.origin = v.origin;
            cam.direction = (v.target - v.origin).normalized();
            cam.up = Vec3(0, 0, 1);
            cam.fov = v.fov;
            std::cout << "Rendering " << pass.name << " / " << v.name
                      << " (" << cfg::samples << " spp)..." << std::endl;
            frame out = octree.renderFrameVulkan(cam, cfg::height, cfg::width,
                                                 frame::colormap::RGB,
                                                 cfg::samples, cfg::bounces, false, true);
            BMPWriter::saveBMP("output/nature/" + pass.name + "_" + v.name + ".bmp", out);
        }
        FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    }

    // ---- Animation Sequence ------------------------------------------------
    if (cfg::renderAnimation && views.size() > 1) {
        std::cout << "\nRendering Animation Sequence (Catmull-Rom Spline Interpolation)..." << std::endl;
        
        applyLighting(octree, passes[0]); // Using Day lighting pass for the animation
        
        int totalSegments = views.size() - 1;
        int framesPerSegment = static_cast<int>(cfg::animFps * cfg::animTimePerSegment);
        int totalFrames = totalSegments * framesPerSegment + 1;
        int frameCount = 0;
        
        for (int seg = 0; seg < totalSegments; ++seg) {
            View v0 = getVirtualView(views, seg - 1);
            View v1 = getVirtualView(views, seg);
            View v2 = getVirtualView(views, seg + 1);
            View v3 = getVirtualView(views, seg + 2);
            
            for (int f = 0; f < framesPerSegment; ++f) {
                float t = static_cast<float>(f) / framesPerSegment;
                
                Vec3 interpOrigin = catmullRom(v0.origin, v1.origin, v2.origin, v3.origin, t);
                Vec3 interpTarget = catmullRom(v0.target, v1.target, v2.target, v3.target, t);
                float interpFov   = catmullRom(v0.fov, v1.fov, v2.fov, v3.fov, t);
                
                Camera cam;
                cam.origin = interpOrigin;
                // Preserve orthogonal Z-up for rendering, matching exactly how the static views do it
                cam.direction = (interpTarget - interpOrigin).normalized();
                cam.up = Vec3(0, 0, 1);
                cam.fov = interpFov;
                
                std::stringstream ss;
                ss << "output/nature/anim_frame_" << std::setw(4) << std::setfill('0') << frameCount << ".bmp";
                
                std::cout << "\rRendering animation frame " << frameCount << " / " << (totalFrames - 1) << std::flush;
                
                // Falling back to fast preview for animation sequence (quality passes would take an eternity)
                frame out = octree.fastRenderFrameVulkan(cam, cfg::height, cfg::width, frame::colormap::RGB);
                BMPWriter::saveBMP(ss.str(), out);
                
                frameCount++;
            }
        }
        
        // Ensure final view is perfectly landed
        View vFinal = views.back();
        Camera camFinal;
        camFinal.origin = vFinal.origin;
        camFinal.direction = (vFinal.target - vFinal.origin).normalized();
        camFinal.up = Vec3(0, 0, 1);
        camFinal.fov = vFinal.fov;
        
        std::stringstream ss;
        ss << "output/nature/anim_frame_" << std::setw(4) << std::setfill('0') << frameCount << ".bmp";
        std::cout << "\rRendering animation frame " << frameCount << " / " << (totalFrames - 1) << std::flush;
        frame out = octree.fastRenderFrameVulkan(camFinal, cfg::height, cfg::width, frame::colormap::RGB);
        BMPWriter::saveBMP(ss.str(), out);
        
        std::cout << "\nAnimation rendering complete." << std::endl;
    }

    std::cout << "\nNature scene renders complete!" << std::endl;
    return 0;
}