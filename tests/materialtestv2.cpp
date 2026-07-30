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
static float smoothstep01(float x) {
    float t = std::clamp(x, 0.0f, 1.0f);
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

static constexpr float ROOM_SPAN = 12.0f;
static constexpr float ROOM_HALF_W = 5.0f;
static constexpr float FLOOR_Z = -0.6f;
static constexpr float CEIL_Z = 7.4f;
static constexpr float DOOR_HALF_W = 1.6f;

static float roomCenterY(int i) { return i * ROOM_SPAN; }

namespace WallCol {
    static const Vec3 NORTH_A(0.75f, 0.10f, 0.12f), NORTH_B(0.95f, 0.55f, 0.60f); // red / pink
    static const Vec3 SOUTH_A(0.10f, 0.20f, 0.75f), SOUTH_B(0.55f, 0.70f, 0.95f); // blue / light blue
    static const Vec3 WEST_A (0.10f, 0.55f, 0.15f), WEST_B (0.55f, 0.90f, 0.55f); // green / light green
    static const Vec3 EAST_A (0.45f, 0.40f, 0.05f), EAST_B (0.95f, 0.85f, 0.15f); // dark / bright yellow
    static const Vec3 FLOOR_A(0.03f, 0.03f, 0.03f), FLOOR_B(0.95f, 0.95f, 0.95f); // black / white
}

static void createCheckerWall(Grid::Octree<int>& octree, const Vec3& center, const Vec3& size,
                              const Vec3& c1, const Vec3& c2, float checker, float rough, float metal,
                              int oid = 100, float step = 0.2f) {
    Vec3 half = size / 2.0f, minB = center - half, maxB = center + half;
    for (float x = minB.x(); x <= maxB.x(); x += step)
        for (float y = minB.y(); y <= maxB.y(); y += step)
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                int cx = (int)std::floor(x / checker);
                int cy = (int)std::floor(y / checker);
                int cz = (int)std::floor(z / checker);
                bool even = ((cx + cy + cz) & 1) == 0;
                octree.insert(1, Vec3(x, y, z), true, even ? c1 : c2, step, true, oid,
                              0.0f, rough, metal, 0.0f, 1.46f);
            }
}

enum class Finish { CONCRETE, METAL, GLOSSY };

static void finishParams(Finish f, float& rough, float& metal) {
    switch (f) {
        case Finish::METAL:
            rough = 0.12f;
            metal = 1.0f;
            break;
        case Finish::GLOSSY:
            rough = 0.08f;
            metal = 0.0f;
            break;
        case Finish::CONCRETE:
        default:
            rough = 0.85f;
            metal = 0.0f;
            break;
    }
}

static void buildRoomShell(Grid::Octree<int>& octree, int roomIdx, Finish wallFinish, Finish floorFinish,
                           float ceilEmit, const Vec3& ceilColor, int ceilOid, float floorZ = FLOOR_Z) {
    float cy = roomCenterY(roomIdx);
    float wr, wm, fr, fm;
    finishParams(wallFinish, wr, wm);
    finishParams(floorFinish, fr, fm);
    const float wallThick = 0.3f;
    const float midZ = (floorZ + CEIL_Z) / 2.0f;
    const float wallH = CEIL_Z - floorZ;

    createCheckerWall(octree, Vec3(0, cy, floorZ), Vec3(2 * ROOM_HALF_W, ROOM_SPAN, 0.2f),
                      WallCol::FLOOR_A, WallCol::FLOOR_B, 1.0f, fr, fm, 100);

    createCheckerWall(octree, Vec3(ROOM_HALF_W, cy, midZ), Vec3(wallThick, ROOM_SPAN, wallH),
                      WallCol::EAST_A, WallCol::EAST_B, 1.0f, wr, wm, 100);

    createCheckerWall(octree, Vec3(-ROOM_HALF_W, cy, midZ), Vec3(wallThick, ROOM_SPAN, wallH),
                      WallCol::WEST_A, WallCol::WEST_B, 1.0f, wr, wm, 100);

    {
        float step = 0.2f;
        float lightZ = CEIL_Z - 0.3f;
        for (float x = -ROOM_HALF_W + 1.0f; x <= ROOM_HALF_W - 1.0f; x += step)
            for (float y = cy - ROOM_SPAN / 2 + 1.0f; y <= cy + ROOM_SPAN / 2 - 1.0f; y += step) {
                Vec3 p(x, y, lightZ);
                octree.insert(1, p, true, ceilColor, step, true, ceilOid, ceilEmit, 0.9f, 0.0f, 0.0f, 1.45f);
                octree.setEmittance(p, ceilColor * ceilEmit, 0.01f);
            }
        createBox(octree, Vec3(0, cy, CEIL_Z), Vec3(2 * ROOM_HALF_W, ROOM_SPAN, 0.18f),
                  Vec3(0.02f, 0.02f, 0.02f), 0.0f, 0.9f, 0.1f, 0.0f, 1.45f, Vec3::Zero(), 100,
                  Grid::BodyType::STATIC, 1.0f, 0.4f);
    }
}

static void buildPartition(Grid::Octree<int>& octree, float boundaryY, bool northFace, bool southFace) {
    float step = 0.2f;
    float midZ = (FLOOR_Z + CEIL_Z) / 2.0f;
    if (northFace)
        for (float x = -ROOM_HALF_W; x <= ROOM_HALF_W; x += step)
            for (float z = FLOOR_Z; z <= CEIL_Z; z += step) {
                if ((std::abs(x) < DOOR_HALF_W) && (z < CEIL_Z - 2.5f)) continue;
                int cx = (int)std::floor(x / 1.0f), cz = (int)std::floor(z / 1.0f);
                bool even = ((cx + cz) & 1) == 0;
                octree.insert(1, Vec3(x, boundaryY - 0.12f, z), true,
                              even ? WallCol::NORTH_A : WallCol::NORTH_B, step, true, 100,
                              0.0f, 0.85f, 0.0f, 0.0f, 1.45f);
            }
    if (southFace)
        for (float x = -ROOM_HALF_W; x <= ROOM_HALF_W; x += step)
            for (float z = FLOOR_Z; z <= CEIL_Z; z += step) {
                if ((std::abs(x) < DOOR_HALF_W) && (z < CEIL_Z - 2.5f)) continue;
                int cx = (int)std::floor(x / 1.0f), cz = (int)std::floor(z / 1.0f);
                bool even = ((cx + cz) & 1) == 0;
                octree.insert(1, Vec3(x, boundaryY + 0.12f, z), true,
                              even ? WallCol::SOUTH_A : WallCol::SOUTH_B, step, true, 100,
                              0.0f, 0.85f, 0.0f, 0.0f, 1.45f);
            }
}

namespace OID {
    constexpr int FLICKER_CEIL = 360;
    constexpr int WATER_TANK = 370;
    constexpr int COLOR_CEIL = 380;
    constexpr int ROVE_RED = 390;
    constexpr int ROVE_BLUE = 391;
    constexpr int ICE_BLOCK = 400;
    constexpr int HEAT_CEIL = 401;
    constexpr int ROVE_GREEN = 402;
}

enum Room {
    R_GENERIC = 0,
    R_FLICKER = 1,
    R_GLASS = 2,
    R_WATER = 3,
    R_COLOR = 4,
    R_METAL = 5,
    R_GLOSSY = 6,
    R_ROVING = 7,
    R_COMBINED= 8,
    NUM_ROOMS = 9
};

static const Vec3 BRIGHT_LIGHT(1.0f, 0.98f, 0.94f);
static constexpr float BRIGHT_EMIT = 1.6f;

int main() {
    std::cout << "Initializing material test walkthrough..." << std::endl;

    const float corridorStartY = roomCenterY(0) - ROOM_SPAN / 2;
    const float corridorEndY = roomCenterY(NUM_ROOMS - 1) + ROOM_SPAN / 2;
    const float WATER_FLOOR_Z = FLOOR_Z - 1.0f;

    Vec3 minBound(-ROOM_HALF_W - 2, corridorStartY - 3, WATER_FLOOR_Z - 3);
    Vec3 maxBound( ROOM_HALF_W + 2, corridorEndY + 3,   CEIL_Z + 3);
    Grid::Octree<int> octree(minBound, maxBound, "output/renderscene", 4);

    octree.setBackgroundColor(Vec3(0.01f, 0.01f, 0.015f));
    octree.setSkylight(Vec3(0.02f, 0.02f, 0.025f));
    octree.setphys_gravityCenter(Vec3(0.0f, 0.0f, -1000.0f));
    octree.setPhysicsSmoothingRadius(0.2f);
    octree.setPhysicsGasConstant(100.0f);
    octree.setPhysicsVelocityDamping(1.0f);
    octree.setPhysicsViscosity(15.0f);
    octree.setPhysicsAirDensity(1.225f);

    std::cout << "Building rooms..." << std::endl;

    auto ceilFor = [&](int r, float& emit, Vec3& col, int& oid) {
        emit = BRIGHT_EMIT;
        col = BRIGHT_LIGHT;
        oid = 300 + r;
        switch (r) {
            case R_FLICKER:
                oid = OID::FLICKER_CEIL;
                break;
            case R_COLOR:
                oid = OID::COLOR_CEIL;
                col = Vec3(1.0f, 0.15f, 0.15f);
                break;
            case R_ROVING:
                emit = 0.0f;
                break;
            case R_COMBINED:
                oid = OID::HEAT_CEIL;
                col = Vec3(1.0f, 0.25f, 0.12f);
                emit = 0.45f;
                break;
            default: break;
        }
    };

    for (int r = 0; r < NUM_ROOMS; ++r) {
        Finish wall = Finish::CONCRETE;
        Finish floor = Finish::CONCRETE;
        float fz = FLOOR_Z;
        if (r == R_METAL) wall = Finish::METAL;
        if (r == R_GLOSSY) wall = Finish::GLOSSY;
        if (r == R_WATER) fz = WATER_FLOOR_Z;
        if (r == R_COMBINED){
            wall = Finish::METAL;
            floor = Finish::GLOSSY;
        }

        float emit;
        Vec3 col;
        int oid;
        ceilFor(r, emit, col, oid);
        buildRoomShell(octree, r, wall, floor, emit, col, oid, fz);
    }

    buildPartition(octree, corridorStartY, true,  false);
    for (int r = 0; r < NUM_ROOMS - 1; ++r)
        buildPartition(octree, roomCenterY(r) + ROOM_SPAN / 2, true, true);
    buildPartition(octree, corridorEndY, false, true);

    {
        float cy = roomCenterY(R_GLASS);
        createBox(octree, Vec3(0, cy, FLOOR_Z + 1.8f), Vec3(2.6f, 2.6f, 3.4f),
                  Vec3(0.92f, 0.96f, 1.0f), 0.0f, 0.02f, 0.0f, 0.95f, 1.5f,
                  Vec3(0.02f, 0.02f, 0.03f), 40, Grid::BodyType::STATIC, 1.0f, 0.12f);
    }

    {
        float cy = roomCenterY(R_WATER);
        Vec3 water(0.2f, 0.45f, 0.85f), absorp(0.04f, 0.02f, 0.06f);
        float step = 0.2f;
        Vec3 c(0, cy, WATER_FLOOR_Z + 1.4f), s(4.5f, 4.5f, 2.4f);
        Vec3 half = s / 2.0f, minB = c - half, maxB = c + half;
        int nx = (int)((s.x()) / step) + 1, ny = (int)((s.y()) / step) + 1, nz = (int)((s.z()) / step) + 1;
        float per = 8.0f / std::max(1, nx * ny * nz);
        for (float x = minB.x(); x <= maxB.x(); x += step)
            for (float y = minB.y(); y <= maxB.y(); y += step)
                for (float z = minB.z(); z <= maxB.z(); z += step) {
                    octree.insert(1, Vec3(x, y, z), true, water, step, true, OID::WATER_TANK,
                                  0.0f, 0.03f, 0.0f, 0.85f, 1.333f, absorp, Grid::BodyType::STATIC, per);
                    octree.setSellmeier(Vec3(x, y, z),
                        Vec3(5.684027565e-1f, 1.726177391e-1f, 2.086189578e-2f).cast<Eigen::half>(),
                        Vec3(5.101829712e-3f, 1.821153936e-2f, 2.620722293e-2f).cast<Eigen::half>());
                }
    }

    {
        float cy = roomCenterY(R_ROVING);
        createBulb(octree, Vec3(-2.0f, cy, 3.0f), 0.4f, Vec3(1.0f, 0.1f, 0.1f), 20.0f, OID::ROVE_RED, 0.12f);
        createBulb(octree, Vec3( 2.0f, cy, 3.0f), 0.4f, Vec3(0.1f, 0.2f, 1.0f), 20.0f, OID::ROVE_BLUE, 0.12f);
        createBox(octree, Vec3(0, cy, FLOOR_Z + 0.6f), Vec3(1.4f, 1.4f, 1.2f),
                  Vec3(0.8f, 0.8f, 0.8f), 0.0f, 0.7f, 0.0f, 0.0f, 1.45f, Vec3::Zero(), 100,
                  Grid::BodyType::STATIC, 1.0f, 0.15f);
    }

    {
        float cy = roomCenterY(R_COMBINED);
        createBox(octree, Vec3(0, cy, FLOOR_Z + 1.6f), Vec3(2.8f, 2.8f, 3.0f),
                  Vec3(0.80f, 0.90f, 1.0f), 0.0f, 0.05f, 0.0f, 0.85f, 1.31f,
                  Vec3(0.02f, 0.01f, 0.005f), OID::ICE_BLOCK, Grid::BodyType::STATIC, 1.0f, 0.2f);
        createBulb(octree, Vec3(2.5f, cy, 3.2f), 0.4f, Vec3(0.1f, 1.0f, 0.2f), 6.0f, OID::ROVE_GREEN, 0.12f);
    }

    std::cout << "Finalizing octree..." << std::endl;
    octree.setLODMinDistance(1024);
    octree.setLODFalloff(0.01f);
    octree.setMaxDistance(4096);
    octree.markPhysicsCollidersDirty();
    octree.printStats();

    struct WayPoint { Vec3 pos; Vec3 look; };
    const float EYE_Z = 2.0f;
    const float DOOR_IN = ROOM_SPAN / 2;
    const float STEP_IN = 1.5f;
    const int LEGS_PER_ROOM = 4;

    std::vector<WayPoint> path;
    path.push_back({Vec3(0.0f, roomCenterY(0) - DOOR_IN - 2.0f, EYE_Z),
                    Vec3(0.0f, roomCenterY(0), EYE_Z - 0.4f)});
    for (int r = 0; r < NUM_ROOMS; ++r) {
        float cy = roomCenterY(r);
        path.push_back({Vec3(0.0f, cy - DOOR_IN + 0.5f, EYE_Z), Vec3(0.0f, cy, EYE_Z - 0.3f)});
        path.push_back({Vec3(0.0f, cy + STEP_IN, EYE_Z), Vec3(0.0f, cy + STEP_IN + 2.0f, EYE_Z - 0.2f)});
        path.push_back({Vec3(0.0f, cy + STEP_IN, EYE_Z), Vec3(0.0f, cy - DOOR_IN, EYE_Z - 0.2f)});
        float nextY = (r + 1 < NUM_ROOMS) ? roomCenterY(r + 1) : cy + DOOR_IN + 2.0f;
        path.push_back({Vec3(0.0f, cy + DOOR_IN - 0.5f, EYE_Z), Vec3(0.0f, nextY, EYE_Z - 0.3f)});
    }

    int width = 512;
    int height = 512;
    const float fps = 60.0f;
    const int framesPerLeg = 90;
    const int videosamples = 7;
    const int bounces = 4;
    const float blendedfactor = 0.65f;
    const int physicsSubsteps = 8;
    const float subDt = (1.0f / fps);

    const int totalLegs = (int)path.size() - 1;
    const int totalFrames = totalLegs * framesPerLeg;

    Grid::FrameWriter writer(2, 8);

    auto legRoom = [&](int leg) -> int {
        int r = (leg <= 0) ? 0 : (leg - 1) / LEGS_PER_ROOM;
        return std::clamp(r, 0, NUM_ROOMS - 1);
    };
    auto roomEntryFrame = [&](int r) -> int { return framesPerLeg * (1 + r * LEGS_PER_ROOM); };

    auto roveRed = octree.getWeakNodesByObjectId(OID::ROVE_RED);
    auto roveBlue = octree.getWeakNodesByObjectId(OID::ROVE_BLUE);
    auto roveGreen = octree.getWeakNodesByObjectId(OID::ROVE_GREEN);

    Vec3 roveRedHome (-2.0f, roomCenterY(R_ROVING),   3.0f);
    Vec3 roveBlueHome( 2.0f, roomCenterY(R_ROVING),   3.0f);
    Vec3 roveGreenHome(2.5f, roomCenterY(R_COMBINED), 3.2f);
    Vec3 roveRedCur = roveRedHome, roveBlueCur = roveBlueHome, roveGreenCur = roveGreenHome;

    bool waterSpawned = false, iceMelted = false;

    std::cout << "\nRendering walkthrough: " << totalFrames << " frames.\n" << std::endl;

    Grid::InFlightFrame inflight;
    std::string pending;
    bool havePending = false;
    int globalFrame = 0;

    auto moveOrb = [&](std::vector<std::weak_ptr<Grid::Octree<int>::NodeData>>& orb, Vec3& cur, const Vec3& target) {
        Vec3 delta = target - cur;
        if (delta.norm() > 1e-4f) {
            for (auto& wp : orb) if (auto sp = wp.lock()) octree.move(sp->position, sp->position + delta);
            cur = target;
        }
    };

    for (int leg = 0; leg < totalLegs; ++leg) {
        const WayPoint& A = path[leg];
        const WayPoint& B = path[leg + 1];
        int room = legRoom(leg);
        std::cout << "Leg " << leg << " (room " << room << ")" << std::endl;

        for (int f = 0; f < framesPerLeg; ++f, ++globalFrame) {
            float t = (float)f / (float)framesPerLeg;
            float ts = smoothstep01(t);

            Camera cam;
            cam.fov = 70.0f;
            cam.origin = A.pos * (1.0f - ts) + B.pos * ts;
            Vec3 look = A.look * (1.0f - ts) + B.look * ts;
            cam.up = Vec3(0.0f, 0.0f, 1.0f);
            cam.direction = (look - cam.origin).normalized();

            if (!waterSpawned && globalFrame >= roomEntryFrame(R_WATER)) {
                octree.makeObjectFluid(OID::WATER_TANK, 0.02f, Grid::BodyType::FLUID);
                octree.markPhysicsCollidersDirty();
                waterSpawned = true;
                std::cout << "  water spawned" << std::endl;
            }
            if (!iceMelted && globalFrame >= roomEntryFrame(R_COMBINED)) {
                octree.makeObjectFluid(OID::ICE_BLOCK, 0.02f, Grid::BodyType::FLUID);
                octree.markPhysicsCollidersDirty();
                iceMelted = true;
                std::cout << "  ice melting" << std::endl;
            }
            if (waterSpawned || iceMelted)
                octree.multiStepPhysics(subDt, physicsSubsteps);

            {
                float ph = globalFrame * 0.5f;
                float n = smoothNoise(1.3f, ph, 2.2f, 1.0f);
                float level = (n > 0.30f) ? BRIGHT_EMIT : BRIGHT_EMIT * 0.06f;
                for (auto& wp : octree.getWeakNodesByObjectId(OID::FLICKER_CEIL))
                    if (auto sp = wp.lock()) octree.setEmittance(sp->position, BRIGHT_LIGHT * level, 0.01f);
            }

            {
                int entry = roomEntryFrame(R_COLOR);
                int dwell = LEGS_PER_ROOM * framesPerLeg;
                float u = std::clamp((globalFrame - entry) / (float)dwell, 0.0f, 1.0f);
                Vec3 red(1.0f, 0.12f, 0.12f), green(0.12f, 1.0f, 0.18f), blue(0.15f, 0.2f, 1.0f);
                Vec3 col;
                if (u < 0.5f) {
                    float k = u / 0.5f;
                    col = red * (1 - k) + green * k;
                }
                else {
                    float k = (u - 0.5f) / 0.5f;
                    col = green * (1 - k) + blue * k;
                }
                for (auto& wp : octree.getWeakNodesByObjectId(OID::COLOR_CEIL))
                    if (auto sp = wp.lock()) octree.setEmittance(sp->position, col * BRIGHT_EMIT, 0.01f);
            }

            {
                float cy = roomCenterY(R_ROVING);
                float a = globalFrame * 0.08f, R = 2.6f;
                moveOrb(roveRed, roveRedCur, Vec3(R * std::cos(a), cy + R * std::sin(a), 3.0f));
                moveOrb(roveBlue, roveBlueCur, Vec3(R * std::cos(a + 3.1416f), cy + R * std::sin(a + 3.1416f), 3.0f));
            }

            {
                float cy = roomCenterY(R_COMBINED);
                float a = globalFrame * 0.10f, R = 2.8f;
                moveOrb(roveGreen, roveGreenCur, Vec3(R * std::cos(a), cy + R * std::sin(a), 3.2f));
                for (auto& wp : roveGreen)
                    if (auto sp = wp.lock()) octree.setEmittance(sp->position, Vec3(0.1f, 1.0f, 0.2f) * 7.0f, 0.02f);
            }

            // Grid::InFlightFrame next = octree.beginFastRenderFrameVulkan(cam, height, width, frame::colormap::RGB);
            Grid::InFlightFrame next = octree.beginSuperBlendedRenderFrameVulkan(cam, height, width, blendedfactor, frame::colormap::RGB, videosamples, bounces, true);
            if (havePending) {
                // frame prev = octree.endFastRenderFrameVulkan(inflight);
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
        // frame prev = octree.endFastRenderFrameVulkan(inflight);
        frame prev = octree.endSuperBlendedRenderFrameVulkan(inflight);
        writer.enqueue(std::move(prev), "output/walkthrough/frame_" + pending + ".bmp");
    }

    writer.drain();
    writer.shutdown();
    std::cout << "\nWalkthrough complete. Frames written: " << writer.writtenCount() << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}