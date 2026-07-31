#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

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
using Octree = Grid::Octree<int>;
using NodePtr = std::shared_ptr<Grid::NodeData_<int>>;
using Grid::BodyType;

namespace OID {
    constexpr int ROOM       = 100;
    constexpr int CEIL_LIGHT = 10;
    constexpr int CLUTTER    = 3;
    constexpr int FLUID      = 200;
    constexpr int HARDBLOCK  = 201;
    constexpr int CLEAVER    = 202;
    constexpr int SPHERE     = 203;
    constexpr int SPIKE      = 204;
    constexpr int BONE_A     = 220;
    constexpr int BONE_B     = 221;
}

constexpr int MUSCLE_SUBOBJ = 1;

// Checkerboard volume, taken from materialtest.cpp so the room matches.
static void createCheckerBox(Octree& octree, const Vec3& center, const Vec3& size,
                             const Vec3& color1, const Vec3& color2, float checkerSize) {
    float step = 0.1f;
    Vec3 halfSize = size / 2.0f;
    Vec3 minB = center - halfSize;
    Vec3 maxB = center + halfSize;
    for (float x = minB.x(); x <= maxB.x(); x += step) {
        for (float y = minB.y(); y <= maxB.y(); y += step) {
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                Vec3 pos(x, y, z);
                int cx = (int)std::floor(x / checkerSize);
                int cy = (int)std::floor(y / checkerSize);
                int cz = (int)std::floor(z / checkerSize);
                bool isEven = ((cx + cy + cz) % 2 == 0);
                Vec3 albedo = isEven ? color1 : color2;
                octree.insert(1, pos, true, albedo, step, true, OID::ROOM, 0.0f, 0.01f, 0.0f, 0.0f, 1.486f);
            }
        }
    }
}

int main(int argc, char** argv) {
    int width = 1280, height = 720;
    int framesTotal = 320;
    float voxel = 0.08f;
    int physicsSubsteps = 8;
    const float fps = 60.0f;
    const float subDt = 1.0f / fps;

    if (argc > 1) framesTotal = std::stoi(argv[1]);
    if (argc > 2) voxel = std::stof(argv[2]);

    Vec3 minBound(-10.0f, -10.0f, -10.0f);
    Vec3 maxBound(10.0f, 10.0f, 10.0f);
    Octree oct(minBound, maxBound, "output/physroomstore", 4);
    oct.setSkylight(Vec3(0.01f, 0.01f, 0.01f));
    oct.setPhysicsGravity(Vec3(0.0f, 0.0f, -9.81f));
    oct.setPhysicsVelocityDamping(0.4f);

    Vec3 cLightGray(0.8f, 0.8f, 0.8f);
    Vec3 cDarkGray(0.35f, 0.35f, 0.35f);
    float chkSize = 0.5f;

    // Floor
    createCheckerBox(oct, Vec3(0.0f, 0.0f, -0.6f), Vec3(14.4f, 14.4f, 0.2f), cLightGray, cDarkGray, chkSize);
    // Walls
    createCheckerBox(oct, Vec3( 7.1f,  0.0f, 3.5f), Vec3(0.2f, 14.4f, 8.0f), cLightGray, cDarkGray, chkSize);
    createCheckerBox(oct, Vec3(-7.1f,  0.0f, 3.5f), Vec3(0.2f, 14.4f, 8.0f), cLightGray, cDarkGray, chkSize);
    createCheckerBox(oct, Vec3( 0.0f,  7.1f, 3.5f), Vec3(14.0f, 0.2f, 8.0f), cLightGray, cDarkGray, chkSize);
    createCheckerBox(oct, Vec3( 0.0f, -7.1f, 3.5f), Vec3(14.0f, 0.2f, 8.0f), cLightGray, cDarkGray, chkSize);

    // Ceiling with an 8x8 bright emissive light panel in the middle.
    {
        Vec3 ceilingCenter(0.0f, 0.0f, 7.4f);
        Vec3 ceilingSize(14.4f, 14.4f, 0.2f);
        Vec3 lightSize(8.0f, 8.0f, 0.2f);
        float step = 0.1f;
        Vec3 minCeiling = ceilingCenter - ceilingSize / 2.0f;
        Vec3 maxCeiling = ceilingCenter + ceilingSize / 2.0f;
        Vec3 minLight = ceilingCenter - lightSize / 2.0f;
        Vec3 maxLight = ceilingCenter + lightSize / 2.0f;

        for (float x = minCeiling.x(); x <= maxCeiling.x(); x += step) {
            for (float y = minCeiling.y(); y <= maxCeiling.y(); y += step) {
                for (float z = minCeiling.z(); z <= maxCeiling.z(); z += step) {
                    bool isLightArea = (x >= minLight.x() && x <= maxLight.x() &&
                                        y >= minLight.y() && y <= maxLight.y());
                    Vec3 pos(x, y, z);
                    if (!isLightArea) {
                        oct.insert(1, pos, true, cDarkGray, step, true, OID::ROOM, 0.0f, 0.01f, 0.0f, 0.0f, 1.45f,
                                   Vec3::Zero(), BodyType::STATIC, 1.0f);
                    } else {
                        Vec3 warm(1.0f, 0.98f, 0.92f);
                        oct.insert(1, pos, true, warm, step, true, OID::CEIL_LIGHT, 4.0f, 0.8f, 0.0f, 0.0f, 1.45f,
                                   Vec3::Zero(), BodyType::STATIC, 1.0f);
                        oct.setEmittance(pos, warm * 4.0f, 0.01f);
                    }
                }
            }
        }
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> ux(-5.5f, 5.5f);
    std::uniform_real_distribution<float> uy(-5.5f, 5.5f);
    std::uniform_real_distribution<float> us(0.4f, 0.9f);
    for (int i = 0; i < 12; ++i) {
        float s = us(rng);
        Vec3 c(ux(rng), uy(rng), s * 0.5f);
        Vec3 col(0.3f + 0.5f * (i % 3 == 0), 0.3f + 0.4f * (i % 3 == 1), 0.35f + 0.5f * (i % 3 == 2));
        oct.insertCube(c, Vec3(s, s, s), voxel, col, OID::CLUTTER, false, BodyType::STATIC, 1.0f);
    }

    oct.insertCube(Vec3(-3.5f, -3.5f, 5.0f), Vec3(1.4f, 1.4f, 1.6f), voxel,
                   Vec3(0.2f, 0.45f, 0.95f), OID::FLUID, false, BodyType::RIGID, 0.03f);

    oct.insertCube(Vec3(3.5f, 3.5f, 3.0f), Vec3(1.8f, 1.8f, 1.2f), voxel,
                   Vec3(0.8f, 0.5f, 0.25f), OID::HARDBLOCK, false, BodyType::RIGID, 1.0f);
    oct.setObjectSplitPolicy(OID::HARDBLOCK, Grid::SplitPolicy::NEW_OID);
    oct.insertCube(Vec3(3.5f, 3.5f, 6.0f), Vec3(2.2f, 0.12f, 0.9f), voxel,
                   Vec3(0.85f, 0.87f, 0.9f), OID::CLEAVER, false, BodyType::RIGID, 8.0f);

    oct.insertCylinder(Vec3(-3.5f, 3.5f, 0.9f), 0.14f, 1.8f, voxel,
                       Vec3(0.7f, 0.7f, 0.75f), OID::SPIKE, false, -1.0f,
                       BodyType::STATIC, 1.0f);
    oct.insertSphere(Vec3(-3.5f, 3.5f, 5.5f), 0.85f, voxel,
                     Vec3(0.9f, 0.2f, 0.5f), OID::SPHERE, false, -1.0f,
                     BodyType::RIGID, 0.4f);

    oct.insertCylinder(Vec3(2.0f, -3.0f, 2.2f), 0.22f, 1.8f, voxel,
                       Vec3(0.92f, 0.90f, 0.82f), OID::BONE_A, false, -1.0f,
                       BodyType::RIGID, 0.6f);
    oct.insertCylinder(Vec3(2.0f, -3.0f, 4.6f), 0.22f, 1.8f, voxel,
                       Vec3(0.92f, 0.90f, 0.82f), OID::BONE_B, false, -1.0f,
                       BodyType::RIGID, 0.6f);

    for (auto& wp : oct.getWeakNodesByObjectId(OID::BONE_A))
        if (auto sp = wp.lock()) { sp->physics.velocity.setZero(); sp->setStatic(true); }

    auto muscleObj = oct.getOrCreateObject(OID::BONE_B);
    std::vector<NodePtr> aNodes, bNodes;
    for (auto& wp : oct.getWeakNodesByObjectId(OID::BONE_A)) if (auto sp = wp.lock()) aNodes.push_back(sp);
    for (auto& wp : oct.getWeakNodesByObjectId(OID::BONE_B)) if (auto sp = wp.lock()) bNodes.push_back(sp);

    std::sort(aNodes.begin(), aNodes.end(), [](const NodePtr& x, const NodePtr& y){ return x->position.z() > y->position.z(); });
    std::sort(bNodes.begin(), bNodes.end(), [](const NodePtr& x, const NodePtr& y){ return x->position.z() < y->position.z(); });

    int fibers = 0;
    int pairCount = std::min<int>(16, (int)std::min(aNodes.size(), bNodes.size()));
    for (int i = 0; i < pairCount; ++i) {
        NodePtr a = aNodes[i];
        NodePtr best = nullptr; float bd = 1e9f;
        for (int j = 0; j < pairCount; ++j) {
            float d = (a->position - bNodes[j]->position).squaredNorm();
            if (d < bd) { bd = d; best = bNodes[j]; }
        }
        if (best && oct.addMuscleFiber(muscleObj, MUSCLE_SUBOBJ, a, best, 300.0f, 0.0f, 0.4f))
            ++fibers;
    }
    std::cout << "Muscle fibers wired: " << fibers << std::endl;

    oct.markPhysicsCollidersDirty();
    oct.optimize();

    const int F_FLUID   = 40;
    const int F_MUSCLE0 = 90;
    const int F_MUSCLE1 = 160;
    const int F_MUSCLE2 = 230;

    Vec3 target(0.0f, 0.0f, 1.0f);
    Camera cam;
    cam.fov = 72.0f;
    cam.origin = Vec3(6.4f, -6.4f, 6.6f);
    cam.direction = (target - cam.origin).normalized();
    cam.up = Vec3(0.0f, 0.0f, 1.0f);

    Grid::FrameWriter writer(2, 8);
    Grid::InFlightFrame inflight;
    std::string pending;
    bool havePending = false;

    std::cout << "\nRendering physics room: " << framesTotal << " frames.\n" << std::endl;

    bool fluidOn = false;
    for (int f = 0; f < framesTotal; ++f) {
        if (!fluidOn && f >= F_FLUID) {
            oct.makeObjectFluid(OID::FLUID, 0.02f, BodyType::FLUID);
            oct.markPhysicsCollidersDirty();
            fluidOn = true;
            std::cout << "  [f" << f << "] fluid released" << std::endl;
        }
        if (f == F_MUSCLE0 || f == F_MUSCLE2) {
            oct.applyMuscleCommand({ OID::BONE_B, MUSCLE_SUBOBJ, Grid::MuscleVerb::CONTRACT, 1.0f });
            std::cout << "  [f" << f << "] muscle contract" << std::endl;
        }
        if (f == F_MUSCLE1) {
            oct.applyMuscleCommand({ OID::BONE_B, MUSCLE_SUBOBJ, Grid::MuscleVerb::RELAX, 1.0f });
            std::cout << "  [f" << f << "] muscle release" << std::endl;
        }

        oct.multiStepPhysics(subDt, physicsSubsteps);

        Grid::InFlightFrame next = oct.beginSuperBlendedRenderFrameVulkan(cam, height, width, 0.65f, frame::colormap::RGB, 7, 3, true);
        if (havePending) {
            frame prev = oct.endSuperBlendedRenderFrameVulkan(inflight);
            writer.enqueue(std::move(prev), "output/physroom/frame_" + pending + ".bmp");
        }
        inflight = next;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%05d", f);
        pending = buf;
        havePending = true;

        if (f % 20 == 0) std::cout << "  frame " << f << "/" << framesTotal << std::endl;
    }
    if (havePending) {
        frame prev = oct.endSuperBlendedRenderFrameVulkan(inflight);
        writer.enqueue(std::move(prev), "output/physroom/frame_" + pending + ".bmp");
    }

    writer.drain();
    writer.shutdown();
    std::cout << "\nDone. Frames written: " << writer.writtenCount() << std::endl;
    return 0;
}