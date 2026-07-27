#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>

#include "../eigen/Eigen/Dense"
#include "../util/grid/camera.hpp"
#include "../util/grid/grid3eigen.hpp"
#include "../util/grid/grid3render.cpp"
#include "../util/grid/grid3physics.cpp"
#include "../util/output/frame.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/framewriter.hpp"
#include "../util/output/aviwriter.hpp"
#include "../util/output/y4mwriter.hpp"
#include "../util/timing_decorator.hpp"
#include "../util/timing_decorator.cpp"

using Vec3 = Eigen::Vector3f;


static float smoothNoise(float x, float y, float z, float scale) {
    float nx = x * scale, ny = y * scale, nz = z * scale;
    float val = std::sin(nx + std::cos(ny)) + std::sin(ny + std::cos(nz)) + std::sin(nz + std::cos(nx));
    return (val + 3.0f) / 6.0f;
}
static float clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }
static float smoothstep01(float x) {
    float t = clamp01(x);
    return t * t * (3.0f - 2.0f * t);
}

static void createBox(Grid::Octree<int>& octree, const Vec3& center, const Vec3& size, const Vec3& albedo,
                      float emission = 0.0f, float roughness = 0.8f, float metallic = 0.0f, float transmission = 0.0f,
                      float ior = 1.45f, const Vec3& absorp = Vec3::Zero(), int oid = 0,
                      Grid::BodyType bType = Grid::BodyType::STATIC, float mass = 1.0f, float step = 0.1f,
                      const Vec3& sellB = Vec3::Zero(), const Vec3& sellC = Vec3::Zero(), bool useSell = false) {
    Vec3 half = size / 2.0f, minB = center - half, maxB = center + half;
    static std::mt19937 rng(1337);
    std::uniform_real_distribution<float> jitter(-0.002f, 0.002f);
    for (float x = minB.x(); x <= maxB.x(); x += step)
        for (float y = minB.y(); y <= maxB.y(); y += step)
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                Vec3 pos(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                octree.insert(1, pos, true, albedo, step, true, oid, emission, roughness, metallic, transmission, ior, absorp, bType, mass);
                if (useSell) octree.setSellmeier(pos, sellB.cast<Eigen::half>(), sellC.cast<Eigen::half>());
            }
}

static void createCheckerBox(Grid::Octree<int>& octree, const Vec3& center, const Vec3& size,
                             const Vec3& color1, const Vec3& color2, float checkerSize, int oid = 100,
                             float roughness = 0.7f, float metallic = 0.0f, float step = 0.1f) {
    Vec3 half = size / 2.0f, minB = center - half, maxB = center + half;
    for (float x = minB.x(); x <= maxB.x(); x += step)
        for (float y = minB.y(); y <= maxB.y(); y += step)
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                int cx = (int)std::floor(x / checkerSize);
                int cy = (int)std::floor(y / checkerSize);
                int cz = (int)std::floor(z / checkerSize);
                bool even = ((cx + cy + cz) & 1) == 0;
                Vec3 albedo = even ? color1 : color2;
                octree.insert(1, Vec3(x, y, z), true, albedo, step, true, oid, 0.0f, roughness, metallic, 0.0f, 1.46f);
            }
}

static void createMirror(Grid::Octree<int>& octree, const Vec3& center, const Vec3& size, int oid, float step = 0.1f) {
    Vec3 half = size / 2.0f, minB = center - half, maxB = center + half;
    Vec3 silver(0.95f, 0.96f, 0.97f);
    for (float x = minB.x(); x <= maxB.x(); x += step)
        for (float y = minB.y(); y <= maxB.y(); y += step)
            for (float z = minB.z(); z <= maxB.z(); z += step)
                octree.insert(1, Vec3(x, y, z), true, silver, step, true, oid, 0.0f, 0.02f, 1.0f, 0.0f, 0.15f);
}

enum class GemCut { OCTAHEDRON, HEXAGONAL_BIPYRAMID };

static void createGem(Grid::Octree<int>& octree, const Vec3& center, float radius, GemCut cut, const Vec3& albedo,
                      float step, int oid, Grid::BodyType bType, float mass, float transmission, float ior,
                      const Vec3& absorp, const Vec3& sellB = Vec3::Zero(), const Vec3& sellC = Vec3::Zero(), bool useSell = false) {
    static std::mt19937 rng(1337);
    std::uniform_real_distribution<float> jitter(-0.002f, 0.002f);
    for (float x = center.x() - radius; x <= center.x() + radius; x += step)
        for (float y = center.y() - radius; y <= center.y() + radius; y += step)
            for (float z = center.z() - radius; z <= center.z() + radius; z += step) {
                float dx = x - center.x(), dy = y - center.y(), dz = z - center.z();
                bool inside = false;
                if (cut == GemCut::OCTAHEDRON)
                    inside = (std::abs(dx) + std::abs(dy) + std::abs(dz)) <= radius;
                else {
                    float hexDist = std::max(std::abs(dx), (std::abs(dx) + std::abs(dy) * 1.732f) / 2.0f);
                    inside = (hexDist + std::abs(dz) * 0.6f) <= (radius * 0.85f) && std::abs(dz) <= radius;
                }
                if (inside) {
                    Vec3 pos(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                    octree.insert(1, pos, true, albedo, step, true, oid, 0.0f, 0.01f, 0.0f, transmission, ior, absorp, bType, mass);
                    if (useSell) octree.setSellmeier(pos, sellB.cast<Eigen::half>(), sellC.cast<Eigen::half>());
                }
            }
}

static void createBulb(Grid::Octree<int>& octree, const Vec3& center, float radius, const Vec3& color,
                       float emittance, int oid, float step = 0.1f) {
    for (float x = center.x() - radius; x <= center.x() + radius; x += step)
        for (float y = center.y() - radius; y <= center.y() + radius; y += step)
            for (float z = center.z() - radius; z <= center.z() + radius; z += step) {
                if ((Vec3(x, y, z) - center).norm() > radius) continue;
                Vec3 pos(x, y, z);
                octree.insert(1, pos, true, color, step, true, oid, emittance, 0.9f, 0.0f, 0.0f, 1.45f);
                octree.setEmittance(pos, color * emittance, 0.01f);
            }
}

static size_t createWaterDrop(Grid::Octree<int>& octree, const Vec3& center, float radius, int count, int oid, float totalMass = 0.1f) {
    static std::mt19937 rng(9001);
    std::uniform_real_distribution<float> jitter(-0.15f, 0.15f);
    auto rAt = [](float t) -> float {
        if (t < 0.0f || t > 1.0f) return 0.0f;
        return std::sqrt(std::max(0.0f, 1.0f - t * t)) * (1.0f - t * 0.55f);
    };
    const float hh = radius, mw = radius * 0.85f;
    float vol = 0.0f;
    {
        const int sl = 512;
        const float dz = (2.0f * hh) / sl;
        for (int i = 0; i < sl; ++i) {
            float t = (i + 0.5f) / sl;
            float r = rAt(t) * mw;
            vol += 3.14159265f * r * r * dz;
        }
    }
    const float vox = std::cbrt(vol / (float)count) * 0.93f, step = vox;
    struct Cand {
        Vec3 pos;
        float rank;
    };
    std::vector<Cand> cs;
    for (float z = -hh; z <= hh; z += step) {
        float t = (z + hh) / (2.0f * hh), rz = rAt(t) * mw;
        if (rz <= 0.0f) continue;
        for (float x = -rz; x <= rz; x += step)
            for (float y = -rz; y <= rz; y += step) {
                if (x * x + y * y > rz * rz) continue;
                cs.push_back({Vec3(x + jitter(rng) * step, y + jitter(rng) * step, z + jitter(rng) * step), std::sqrt(x * x + y * y) / std::max(rz, 1e-6f)});
            }
    }
    if ((int)cs.size() > count) {
        std::nth_element(cs.begin(), cs.begin() + count, cs.end(), [](const Cand& a, const Cand& b){ return a.rank < b.rank; });
        cs.resize(count);
    }
    const float per = totalMass / std::max<size_t>(1, cs.size());
    Vec3 cWater(0.85f, 0.92f, 1.0f), absorp(0.06f, 0.02f, 0.01f);
    for (const auto& c : cs) {
        Vec3 pos = center + c.pos;
        octree.insert(1, pos, true, cWater, vox, true, oid, 0.0f, 0.02f, 0.0f, 0.92f, 1.333f, absorp, Grid::BodyType::FLUID, per);
        octree.setSellmeier(pos,
            Vec3(5.684027565e-1f, 1.726177391e-1f, 2.086189578e-2f).cast<Eigen::half>(),
            Vec3(5.101829712e-3f, 1.821153936e-2f, 2.620722293e-2f).cast<Eigen::half>());
    }
    return cs.size();
}

static size_t createFluidBody(Grid::Octree<int>& octree, const Vec3& center, const Vec3& size, int oid,
                              const Vec3& color, float totalMass, float step = 0.18f) {
    Vec3 half = size / 2.0f, minB = center - half, maxB = center + half;
    std::vector<Vec3> pts;
    for (float x = minB.x(); x <= maxB.x(); x += step)
        for (float y = minB.y(); y <= maxB.y(); y += step)
            for (float z = minB.z(); z <= maxB.z(); z += step)
                pts.push_back(Vec3(x, y, z));
    float per = totalMass / std::max<size_t>(1, pts.size());
    Vec3 absorp(0.04f, 0.02f, 0.06f);
    for (auto& p : pts) {
        octree.insert(1, p, true, color, step, true, oid, 0.0f, 0.03f, 0.0f, 0.85f, 1.333f, absorp, Grid::BodyType::FLUID, per);
        octree.setSellmeier(p,
            Vec3(5.684027565e-1f, 1.726177391e-1f, 2.086189578e-2f).cast<Eigen::half>(),
            Vec3(5.101829712e-3f, 1.821153936e-2f, 2.620722293e-2f).cast<Eigen::half>());
    }
    return pts.size();
}

static constexpr float ROOM_SPAN   = 12.0f;
static constexpr float ROOM_HALF_W = 5.0f;
static constexpr float FLOOR_Z     = -0.6f;
static constexpr float CEIL_Z      = 7.4f;
static constexpr float DOOR_HALF_W = 1.6f;

static float roomCenterY(int i) { return i * ROOM_SPAN; }

static void buildRoomShell(Grid::Octree<int>& octree, int roomIdx, const Vec3& floorA, const Vec3& floorB,
                           bool emissiveCeiling, float ceilEmit, const Vec3& ceilColor) {
    float cy = roomCenterY(roomIdx);
    Vec3 wallGray(0.35f, 0.35f, 0.38f);
    createCheckerBox(octree, Vec3(0, cy, FLOOR_Z), Vec3(2 * ROOM_HALF_W, ROOM_SPAN, 0.2f), floorA, floorB, 1.0f, 100, 0.55f, 0.0f, 0.18f);
    if (emissiveCeiling) {
        float step = 0.2f;
        for (float x = -ROOM_HALF_W + 1.0f; x <= ROOM_HALF_W - 1.0f; x += step)
            for (float y = cy - ROOM_SPAN / 2 + 1.0f; y <= cy + ROOM_SPAN / 2 - 1.0f; y += step) {
                Vec3 p(x, y, CEIL_Z);
                octree.insert(1, p, true, ceilColor, step, true, 300 + roomIdx, ceilEmit, 0.9f, 0.0f, 0.0f, 1.45f);
                octree.setEmittance(p, ceilColor * ceilEmit, 0.01f);
            }
        createBox(octree, Vec3(0, cy, CEIL_Z + 0.02f), Vec3(2 * ROOM_HALF_W, ROOM_SPAN, 0.18f), Vec3(0.02f, 0.02f, 0.02f), 0.0f, 0.9f, 0.1f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.4f);
    } else {
        createBox(octree, Vec3(0, cy, CEIL_Z), Vec3(2 * ROOM_HALF_W, ROOM_SPAN, 0.2f), Vec3(0.05f, 0.05f, 0.06f), 0.0f, 0.9f, 0.05f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.3f);
    }
    createBox(octree, Vec3( ROOM_HALF_W, cy, (FLOOR_Z + CEIL_Z) / 2), Vec3(0.2f, ROOM_SPAN, CEIL_Z - FLOOR_Z), wallGray, 0.0f, 0.8f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.25f);
    createBox(octree, Vec3(-ROOM_HALF_W, cy, (FLOOR_Z + CEIL_Z) / 2), Vec3(0.2f, ROOM_SPAN, CEIL_Z - FLOOR_Z), wallGray, 0.0f, 0.8f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.25f);
}

static void buildPartition(Grid::Octree<int>& octree, float boundaryY) {
    float step = 0.2f;
    Vec3 wallGray(0.30f, 0.30f, 0.33f);
    for (float x = -ROOM_HALF_W; x <= ROOM_HALF_W; x += step)
        for (float z = FLOOR_Z; z <= CEIL_Z; z += step) {
            bool inDoor = (std::abs(x) < DOOR_HALF_W) && (z < CEIL_Z - 2.5f);
            if (inDoor) continue;
            octree.insert(1, Vec3(x, boundaryY, z), true, wallGray, step, true, 100, 0.0f, 0.8f, 0.0f, 0.0f, 1.45f);
        }
}

static void buildEndCap(Grid::Octree<int>& octree, float y) {
    createBox(octree, Vec3(0, y, (FLOOR_Z + CEIL_Z) / 2), Vec3(2 * ROOM_HALF_W, 0.2f, CEIL_Z - FLOOR_Z), Vec3(0.25f, 0.25f, 0.28f), 0.0f, 0.85f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.25f);
}

namespace OID {
    constexpr int DRIP_DROP  = 210;
    constexpr int FLUID_TANK = 211;
    constexpr int FLICKER_A  = 340;
    constexpr int FLICKER_B  = 341;
    constexpr int FLICKER_C  = 342;
    constexpr int FLASHLIGHT = 350;
}

int main() {
    std::cout << "Initializing multi-room walkthrough scene..." << std::endl;

    const int   NUM_ROOMS = 6;
    const float corridorStartY = roomCenterY(0) - ROOM_SPAN / 2;
    const float corridorEndY   = roomCenterY(NUM_ROOMS - 1) + ROOM_SPAN / 2;

    Vec3 minBound(-ROOM_HALF_W - 2, corridorStartY - 3, FLOOR_Z - 3);
    Vec3 maxBound( ROOM_HALF_W + 2, corridorEndY + 3,   CEIL_Z + 3);
    Grid::Octree<int> octree(minBound, maxBound, "output/renderscene", 4);

    octree.setBackgroundColor(Vec3(0.01f, 0.01f, 0.015f));
    octree.setSkylight(Vec3(0.015f, 0.015f, 0.02f));
    octree.setphys_gravityCenter(Vec3(0.0f, 0.0f, -1000.0f));
    octree.setPhysicsSmoothingRadius(0.2f);
    octree.setPhysicsGasConstant(100.0f);
    octree.setPhysicsVelocityDamping(1.0f);
    octree.setPhysicsViscosity(15.0f);
    octree.setPhysicsAirDensity(1.225f);

    std::cout << "Building rooms..." << std::endl;
    Vec3 checkA(0.72f, 0.72f, 0.74f), checkB(0.14f, 0.14f, 0.16f);

    for (int r = 0; r < NUM_ROOMS; ++r) {
        bool emissiveCeil;
        float ceilEmit;
        Vec3 ceilColor(1.0f, 1.0f, 1.0f);
        switch (r) {
            case 0: emissiveCeil = true;  ceilEmit = 1.4f; break;
            case 1: emissiveCeil = true;  ceilEmit = 1.2f; break;
            case 2: emissiveCeil = true;  ceilEmit = 1.0f; break;
            case 3: emissiveCeil = false; ceilEmit = 0.0f; break;
            case 4: emissiveCeil = true;  ceilEmit = 0.9f; break;
            default: emissiveCeil = false; ceilEmit = 0.0f; break;
        }
        buildRoomShell(octree, r, checkA, checkB, emissiveCeil, ceilEmit, ceilColor);
    }
    for (int r = 0; r < NUM_ROOMS - 1; ++r) buildPartition(octree, roomCenterY(r) + ROOM_SPAN / 2);
    buildEndCap(octree, corridorStartY);
    buildEndCap(octree, corridorEndY);

    {
        float cy = roomCenterY(0);
        createMirror(octree, Vec3(ROOM_HALF_W - 0.25f, cy, 3.2f), Vec3(0.15f, ROOM_SPAN - 2.0f, 5.5f), 10);
        Vec3 gold(1.00f, 0.80f, 0.30f), silver(0.90f, 0.90f, 0.95f), copper(0.95f, 0.55f, 0.35f);
        createBox(octree, Vec3(0, cy, FLOOR_Z + 0.4f), Vec3(6.0f, 3.0f, 0.6f), Vec3(0.08f, 0.08f, 0.09f), 0.0f, 0.9f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.2f);
        auto sphere = [&](const Vec3& c, const Vec3& col, float rough, float metal, int oid) {
            float R = 0.7f;
            for (float x = c.x() - R; x <= c.x() + R; x += 0.1f)
                for (float y = c.y() - R; y <= c.y() + R; y += 0.1f)
                    for (float z = c.z() - R; z <= c.z() + R; z += 0.1f) {
                        if ((Vec3(x, y, z) - c).norm() > R) continue;
                        octree.insert(1, Vec3(x, y, z), true, col, 0.1f, true, oid, 0.0f, rough, metal, 0.0f, 0.3f);
                    }
        };
        sphere(Vec3(-2.2f, cy, 1.1f), silver, 0.02f, 1.0f, 11);
        sphere(Vec3( 0.0f, cy, 1.1f), gold,   0.15f, 1.0f, 12);
        sphere(Vec3( 2.2f, cy, 1.1f), copper, 0.35f, 1.0f, 13);
    }

    {
        float cy = roomCenterY(1);
        Vec3 cRuby(0.878f, 0.066f, 0.3725f), cAmethyst(0.6f, 0.4f, 0.8f);
        createBox(octree, Vec3(0, cy, FLOOR_Z + 0.35f), Vec3(6.5f, 3.5f, 0.5f), Vec3(0.06f, 0.06f, 0.07f), 0.0f, 0.9f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.2f);
        createGem(octree, Vec3(-2.0f, cy, 1.2f), 0.7f, GemCut::OCTAHEDRON, cRuby, 0.1f, 20,
                  Grid::BodyType::STATIC, 1.0f, 0.95f, 1.757f, Vec3(0.05f, 0.8f, 0.8f),
                  Vec3(1.4360479f, 0.64583146f, 3.4556846f), Vec3(0.0052998009f, 0.014262926f, 210.80888f), true);
        createGem(octree, Vec3(2.0f, cy, 1.2f), 0.7f, GemCut::HEXAGONAL_BIPYRAMID, cAmethyst, 0.1f, 21,
                  Grid::BodyType::STATIC, 1.0f, 0.97f, 1.534f, Vec3(0.8f, 0.6f, 0.05f),
                  Vec3(0.696f, 0.407f, 0.897f), Vec3(0.0046f, 0.013f, 97.934f), true);
        createBox(octree, Vec3(0, cy, 1.2f), Vec3(1.1f, 1.1f, 1.1f), Vec3(0.9f, 0.95f, 1.0f), 0.0f, 0.01f, 0.0f, 0.95f, 1.62f,
                  Vec3(0.10f, 0.03f, 0.02f), 22, Grid::BodyType::STATIC, 1.0f, 0.1f,
                  Vec3(0.54727636f, 0.15459328f, 0.13445437f), Vec3(0.0053423668f, 0.019974298f, 10.596549f), true);
    }

    Vec3 dripSpawn;
    {
        float cy = roomCenterY(2);
        Vec3 tankColor(0.85f, 0.9f, 1.0f);
        float tw = 3.0f, tl = 3.0f, th = 2.2f, base = FLOOR_Z + 0.1f;
        createBox(octree, Vec3(0, cy, base), Vec3(tw + 0.4f, tl + 0.4f, 0.2f), Vec3(0.1f, 0.1f, 0.12f), 0.0f, 0.5f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.15f);
        auto glassWall = [&](const Vec3& c, const Vec3& s) {
            createBox(octree, c, s, tankColor, 0.0f, 0.02f, 0.0f, 0.9f, 1.5f, Vec3(0.02f, 0.02f, 0.03f), 100, Grid::BodyType::STATIC, 1.0f, 0.12f);
        };
        glassWall(Vec3( tw / 2, cy, base + th / 2), Vec3(0.15f, tl, th));
        glassWall(Vec3(-tw / 2, cy, base + th / 2), Vec3(0.15f, tl, th));
        glassWall(Vec3(0, cy + tl / 2, base + th / 2), Vec3(tw, 0.15f, th));
        glassWall(Vec3(0, cy - tl / 2, base + th / 2), Vec3(tw, 0.15f, th));
        createFluidBody(octree, Vec3(0, cy, base + 0.6f), Vec3(tw - 0.4f, tl - 0.4f, 0.9f), OID::FLUID_TANK, Vec3(0.2f, 0.45f, 0.85f), 6.0f, 0.18f);
        dripSpawn = Vec3(0.0f, cy, CEIL_Z - 0.6f);
        createWaterDrop(octree, dripSpawn, 0.45f, 90, OID::DRIP_DROP, 0.08f);
    }

    {
        float cy = roomCenterY(3);
        createBulb(octree, Vec3(0, cy + ROOM_SPAN / 2 - 1.2f, 4.5f), 0.5f, Vec3(1.0f, 0.97f, 0.9f), 6.0f, 330);
        Vec3 stone(0.55f, 0.52f, 0.48f);
        for (int i = -2; i <= 2; ++i) {
            if (i == 0) continue;
            float px = i * 1.7f;
            createBox(octree, Vec3(px, cy, (FLOOR_Z + CEIL_Z) / 2), Vec3(0.6f, 0.6f, CEIL_Z - FLOOR_Z - 0.4f), stone, 0.0f, 0.85f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.15f);
        }
        createBox(octree, Vec3(0, cy - 2.5f, FLOOR_Z + 0.5f), Vec3(3.0f, 1.5f, 1.0f), Vec3(0.8f, 0.8f, 0.82f), 0.0f, 0.7f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.15f);
    }

    {
        float cy = roomCenterY(4);
        auto tube = [&](float yoff, const Vec3& col, int oid) {
            for (float x = -3.0f; x <= 3.0f; x += 0.15f) {
                Vec3 p(x, cy + yoff, CEIL_Z - 0.4f);
                octree.insert(1, p, true, col, 0.15f, true, oid, 1.5f, 0.9f, 0.0f, 0.0f, 1.45f);
                octree.setEmittance(p, col * 1.5f, 0.01f);
            }
        };
        tube(-3.0f, Vec3(1.0f, 0.15f, 0.2f), OID::FLICKER_A);
        tube( 0.0f, Vec3(0.2f, 1.0f, 0.35f), OID::FLICKER_B);
        tube( 3.0f, Vec3(0.25f, 0.4f, 1.0f), OID::FLICKER_C);
        createBox(octree, Vec3(0, cy, 1.2f), Vec3(2.0f, 2.0f, 2.0f), Vec3(0.8f, 0.8f, 0.8f), 0.0f, 0.6f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.2f);
    }

    {
        float cy = roomCenterY(5);
        Vec3 crate(0.6f, 0.45f, 0.3f);
        createBox(octree, Vec3(-2.5f, cy + 1.5f, FLOOR_Z + 0.8f), Vec3(1.4f, 1.4f, 1.6f), crate, 0.0f, 0.8f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.15f);
        createBox(octree, Vec3( 2.5f, cy - 1.0f, FLOOR_Z + 0.5f), Vec3(1.2f, 1.2f, 1.0f), crate, 0.0f, 0.8f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.15f);
        createBox(octree, Vec3( 0.0f, cy + 2.5f, FLOOR_Z + 1.2f), Vec3(0.8f, 0.8f, 2.4f), crate, 0.0f, 0.8f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100, Grid::BodyType::STATIC, 1.0f, 0.15f);
        createBulb(octree, Vec3(0, cy - ROOM_SPAN / 2 + 1.0f, 2.0f), 0.25f, Vec3(1.0f, 0.95f, 0.85f), 8.0f, OID::FLASHLIGHT, 0.08f);
    }

    std::cout << "Finalizing octree..." << std::endl;
    octree.setLODMinDistance(1024);
    octree.setLODFalloff(0.01f);
    octree.setMaxDistance(4096);
    octree.markPhysicsCollidersDirty();
    octree.printStats();

    struct WayPoint {
        Vec3 pos;
        Vec3 look;
    };
    std::vector<WayPoint> path = {
        {Vec3( 0.0f, roomCenterY(0) - 4.5f, 2.2f), Vec3( 2.0f, roomCenterY(0),        1.5f)},
        {Vec3(-1.5f, roomCenterY(0),        2.0f), Vec3( ROOM_HALF_W, roomCenterY(0), 3.0f)},
        {Vec3( 0.0f, roomCenterY(0) + 3.5f, 2.0f), Vec3( 0.0f, roomCenterY(0),        1.1f)},
        {Vec3(-2.5f, roomCenterY(1) - 3.5f, 2.0f), Vec3(-2.0f, roomCenterY(1),        1.2f)},
        {Vec3( 0.0f, roomCenterY(1),        2.2f), Vec3( 0.0f, roomCenterY(1),        1.2f)},
        {Vec3( 2.5f, roomCenterY(1) + 3.0f, 2.0f), Vec3( 2.0f, roomCenterY(1),        1.2f)},
        {Vec3( 0.0f, roomCenterY(2) - 4.0f, 2.4f), Vec3( 0.0f, roomCenterY(2),        1.0f)},
        {Vec3( 2.0f, roomCenterY(2) - 1.0f, 2.0f), Vec3( 0.0f, roomCenterY(2),        0.6f)},
        {Vec3( 0.0f, roomCenterY(2) + 3.5f, 2.2f), Vec3( 0.0f, roomCenterY(2) - 1.0f, 1.5f)},
        {Vec3(-3.0f, roomCenterY(3) - 3.5f, 2.0f), Vec3( 0.0f, roomCenterY(3) + 3.0f, 3.5f)},
        {Vec3( 0.0f, roomCenterY(3) - 1.0f, 1.8f), Vec3( 0.0f, roomCenterY(3) + 4.0f, 4.0f)},
        {Vec3( 3.0f, roomCenterY(3) + 3.0f, 2.0f), Vec3(-2.0f, roomCenterY(3),        1.5f)},
        {Vec3( 0.0f, roomCenterY(4) - 4.0f, 2.2f), Vec3( 0.0f, roomCenterY(4),        2.0f)},
        {Vec3(-2.0f, roomCenterY(4),        2.4f), Vec3( 0.0f, roomCenterY(4),        3.0f)},
        {Vec3( 0.0f, roomCenterY(4) + 3.5f, 2.0f), Vec3( 0.0f, roomCenterY(4),        1.2f)},
        {Vec3( 0.0f, roomCenterY(5) - 4.0f, 2.0f), Vec3( 0.0f, roomCenterY(5),        2.0f)},
        {Vec3( 0.0f, roomCenterY(5) - 0.5f, 2.2f), Vec3(-2.5f, roomCenterY(5) + 1.5f, 1.5f)},
        {Vec3( 0.5f, roomCenterY(5) + 2.5f, 2.2f), Vec3( 2.5f, roomCenterY(5) - 1.0f, 1.0f)},
    };

    int width = 512, height = 512;
    const float fps = 60.0f;
    const int   framesPerLeg = 90;
    const int   videosamples = 7;
    const int   bounces = 4;
    const float blendedfactor = 0.65f;
    const int   physicsSubsteps = 8;
    const float subDt = (1.0f / fps) / physicsSubsteps;

    const int totalLegs   = (int)path.size() - 1;
    const int totalFrames = totalLegs * framesPerLeg;

    Grid::FrameWriter writer(2, 8);

    std::vector<std::weak_ptr<Grid::Octree<int>::NodeData>> dripVoxels  = octree.getWeakNodesByObjectId(OID::DRIP_DROP);
    std::vector<std::weak_ptr<Grid::Octree<int>::NodeData>> flashVoxels = octree.getWeakNodesByObjectId(OID::FLASHLIGHT);

    Vec3 flashHome(0, roomCenterY(5) - ROOM_SPAN / 2 + 1.0f, 2.0f);
    Vec3 flashCurrent = flashHome;

    auto legRoom = [&](int leg) -> int { return std::min(NUM_ROOMS - 1, leg / 3); };

    std::cout << "\nRendering walkthrough: " << totalFrames << " frames.\n" << std::endl;

    Grid::InFlightFrame inflight;
    std::string pending;
    bool havePending = false;
    int globalFrame = 0;
    const int fluidReleaseFrame = framesPerLeg * 6;

    for (int leg = 0; leg < totalLegs; ++leg) {
        const WayPoint& A = path[leg];
        const WayPoint& B = path[leg + 1];
        int room = legRoom(leg);
        std::cout << "Leg " << leg << " (room " << room << ")" << std::endl;

        for (int f = 0; f < framesPerLeg; ++f, ++globalFrame) {
            float t  = (float)f / (float)framesPerLeg;
            float ts = smoothstep01(t);

            Camera cam;
            cam.fov = 70.0f;
            cam.origin = A.pos * (1.0f - ts) + B.pos * ts;
            Vec3 look  = A.look * (1.0f - ts) + B.look * ts;
            cam.up = Vec3(0.0f, 0.0f, 1.0f);
            cam.direction = (look - cam.origin).normalized();

            if (globalFrame >= fluidReleaseFrame) {
                for (int s = 0; s < physicsSubsteps; ++s) octree.stepPhysics(subDt);
            } else {
                for (auto& wp : dripVoxels)
                    if (auto sp = wp.lock()) {
                        sp->physics.velocity = Vec3::Zero();
                        sp->physics.force = Vec3::Zero();
                    }
            }

            {
                float ph = globalFrame * 0.5f;
                auto flick = [&](int oid, const Vec3& col, float seed) {
                    float n = smoothNoise(seed, ph, seed * 1.7f, 1.0f);
                    float level = (n > 0.30f) ? (0.8f + 0.7f * n) : (0.05f + 0.1f * n);
                    for (auto& wp : octree.getWeakNodesByObjectId(oid))
                        if (auto sp = wp.lock()) octree.setEmittance(sp->position, col * level * 1.6f, 0.02f);
                };
                flick(OID::FLICKER_A, Vec3(1.0f, 0.15f, 0.2f), 1.3f);
                flick(OID::FLICKER_B, Vec3(0.2f, 1.0f, 0.35f), 4.1f);
                flick(OID::FLICKER_C, Vec3(0.25f, 0.4f, 1.0f), 7.9f);
            }

            {
                float cy = roomCenterY(5);
                float sweep = globalFrame * 0.06f;
                Vec3 target = flashHome + Vec3(2.6f * std::sin(sweep), 2.6f * std::cos(sweep * 0.7f), 0.6f * std::sin(sweep * 1.3f));
                target.y() = std::clamp(target.y(), cy - ROOM_SPAN / 2 + 1.0f, cy + ROOM_SPAN / 2 - 1.0f);
                Vec3 delta = target - flashCurrent;
                if (delta.norm() > 1e-4f) {
                    for (auto& wp : flashVoxels)
                        if (auto sp = wp.lock()) octree.move(sp->position, sp->position + delta);
                    flashCurrent = target;
                }
                float pulse = 7.0f + 2.0f * std::sin(globalFrame * 0.4f);
                for (auto& wp : flashVoxels)
                    if (auto sp = wp.lock()) octree.setEmittance(sp->position, Vec3(1.0f, 0.95f, 0.85f) * pulse, 0.02f);
            }

            Grid::InFlightFrame next = octree.beginSuperBlendedRenderFrameVulkan(
                cam, height * 2, width * 2, blendedfactor, frame::colormap::RGB, videosamples, bounces, true);
            if (havePending) {
                frame prev = octree.endSuperBlendedRenderFrameVulkan(inflight);
                writer.enqueue(std::move(prev), "output/walkthrough/frame_" + pending + ".bmp");
            }
            inflight = next;
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%05d", globalFrame);
            pending = buf;
            havePending = true;
        }
    }
    if (havePending) {
        frame prev = octree.endSuperBlendedRenderFrameVulkan(inflight);
        writer.enqueue(std::move(prev), "output/walkthrough/frame_" + pending + ".bmp");
    }

    writer.drain();
    writer.shutdown();
    std::cout << "\nWalkthrough complete. Frames written: " << writer.writtenCount() << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}