#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <random>

// Include Eigen and project headers
#include "../eigen/Eigen/Dense" 
#include "../util/grid/camera.hpp"
#include "../util/grid/grid3eigen.hpp"
#include "../util/grid/grid3render.cpp"
#include "../util/grid/grid3physics.cpp"
#include "../util/output/frame.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/aviwriter.hpp"
#include "../util/output/y4mwriter.hpp"
#include "../util/timing_decorator.hpp"
#include "../util/timing_decorator.cpp"

// Helper function to create a solid volume of voxels with material properties
void createBox(Grid::Octree<int>& octree, const Eigen::Vector3f& center, const Eigen::Vector3f& size, const Eigen::Vector3f& albedo, float emission = 0.0f,
               float roughness = 0.8f, float metallic = 0.0f, float transmission = 0.0f, float ior = 1.45f,const Eigen::Vector3f& absorp = Eigen::Vector3f::Zero(), 
               int oid = 0, Grid::BodyType bType = Grid::BodyType::STATIC, float mass = 1.0f, float step = 0.1) {
    Eigen::Vector3f halfSize = size / 2.0f;
    Eigen::Vector3f minB = center - halfSize;
    Eigen::Vector3f maxB = center + halfSize;
    static std::mt19937 rng(1337);
    std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);
    
    for (float x = minB.x(); x <= maxB.x(); x += step) {
        for (float y = minB.y(); y <= maxB.y(); y += step) {
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                Eigen::Vector3f pos(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                Eigen::Vector3f emittance = albedo * emission;
                
                octree.insert(1, pos, albedo, true, step, true, oid, emittance, roughness, metallic, transmission, ior, absorp, bType, mass);
            }
        }
    }
}

// Helper function to create a checkerboard pattern volume
void createCheckerBox(Grid::Octree<int>& octree, const Eigen::Vector3f& center, const Eigen::Vector3f& size, 
                      const Eigen::Vector3f& color1, const Eigen::Vector3f& color2, float checkerSize) {
    float step = 0.1f;
    Eigen::Vector3f halfSize = size / 2.0f;
    Eigen::Vector3f minB = center - halfSize;
    Eigen::Vector3f maxB = center + halfSize;
    
    for (float x = minB.x(); x <= maxB.x(); x += step) {
        for (float y = minB.y(); y <= maxB.y(); y += step) {
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                Eigen::Vector3f pos(x, y, z);
                
                // Use floor to correctly handle negative coordinates for the repeating pattern
                int cx = static_cast<int>(std::floor(x / checkerSize));
                int cy = static_cast<int>(std::floor(y / checkerSize));
                int cz = static_cast<int>(std::floor(z / checkerSize));
                
                // 3D Checkerboard logic
                bool isEven = ((cx + cy + cz) % 2 == 0);
                Eigen::Vector3f albedo = isEven ? color1 : color2;
                
                octree.insert(1, pos, albedo, true, step, true, 100, Eigen::Vector3f::Zero(), 0.8f, 0.2f, 0.2f, 1.45f);
            }
        }
    }
}

enum class TargetState {
    FLUID,
    // GAS,
    RIGID
};

struct StateEvent {
    int frameTrigger;
    int objectId;
    float mass;
    bool isMoltenMetal;
    TargetState targetState;
};

int main() {
    std::cout << "Initializing Grid::Octree..." << std::endl;

    // 1. Initialize Grid::Octree bounds
    Eigen::Vector3f minBound(-10.0f, -10.0f, -10.0f);
    Eigen::Vector3f maxBound(10.0f, 10.0f, 10.0f);
    Grid::Octree<int> octree(minBound, maxBound, 8, 16);
    
    // Set a dark background to emphasize the PBR light emission
    octree.setBackgroundColor(Eigen::Vector3f(0.02f, 0.02f, 0.02f));
    octree.setSkylight(Eigen::Vector3f(0.01f, 0.01f, 0.01f));
    octree.setphys_gravityCenter(Eigen::Vector3f(0.0f, 0.0f, -1000.0f));
    octree.setPhysicsSmoothingRadius(0.2f);
    // octree.setPhysicsGasConstant(100.0f); // Lowered significantly from 2000 for stability
    octree.setPhysicsVelocityDamping(1.0f); // Higher damping stops infinite scattering
    octree.setPhysicsViscosity(15.0f);
    // octree.setPhysicsAirDensity(1.225f);
    // octree.setPhysicsSurfaceTension(2000.0f);


    std::cout << "Building scene..." << std::endl;

    // 2a. Build Room (Floor and 4 Walls)
    Eigen::Vector3f cLightGray(0.8f, 0.8f, 0.8f);
    Eigen::Vector3f cDarkGray(0.2f, 0.2f, 0.2f);
    float chkSize = 1.0f;

    // Floor
    createCheckerBox(octree, Eigen::Vector3f(0.0f, 0.0f, -0.6f), Eigen::Vector3f(14.4f, 14.4f, 0.2f), cLightGray, cDarkGray, chkSize);
    
    // Walls
    createCheckerBox(octree, Eigen::Vector3f( 7.1f,  0.0f, 3.5f), Eigen::Vector3f(0.2f, 14.4f, 8.0f), cLightGray, cDarkGray, chkSize); // +X
    createCheckerBox(octree, Eigen::Vector3f(-7.1f,  0.0f, 3.5f), Eigen::Vector3f(0.2f, 14.4f, 8.0f), cLightGray, cDarkGray, chkSize); // -X
    createCheckerBox(octree, Eigen::Vector3f( 0.0f,  7.1f, 3.5f), Eigen::Vector3f(14.0f, 0.2f, 8.0f), cLightGray, cDarkGray, chkSize); // +Y
    createCheckerBox(octree, Eigen::Vector3f( 0.0f, -7.1f, 3.5f), Eigen::Vector3f(14.0f, 0.2f, 8.0f), cLightGray, cDarkGray, chkSize); // -Y
    {
        Eigen::Vector3f ceilingCenter(0.0f, 0.0f, 7.4f);
        Eigen::Vector3f ceilingSize(14.4f, 14.4f, 0.2f);
        Eigen::Vector3f lightSize(8.0f, 8.0f, 0.2f);
        float step = 0.5f;

        Eigen::Vector3f minCeiling = ceilingCenter - ceilingSize / 2.0f;
        Eigen::Vector3f maxCeiling = ceilingCenter + ceilingSize / 2.0f;
        Eigen::Vector3f minLight = ceilingCenter - lightSize / 2.0f;
        Eigen::Vector3f maxLight = ceilingCenter + lightSize / 2.0f;

        Eigen::Vector3f cBlack(0.01f, 0.01f, 0.01f);
        Eigen::Vector3f cWhite(1.0f, 1.0f, 1.0f);

        for (float x = minCeiling.x(); x <= maxCeiling.x(); x += step) {
            for (float y = minCeiling.y(); y <= maxCeiling.y(); y += step) {
                for (float z = minCeiling.z(); z <= maxCeiling.z(); z += step) {
                    Eigen::Vector3f pos(x, y, z);
                    
                    bool isLightArea = (x >= minLight.x() && x <= maxLight.x() &&
                                        y >= minLight.y() && y <= maxLight.y());

                    if (isLightArea) {
                        
                        octree.insert(1, pos, cWhite, true, step, true, 10, cWhite, 0.8f, 0.0f, 0.0f, 1.45f, Eigen::Vector3f::Zero(), Grid::BodyType::STATIC, 1.0f);
                    } else {
                        octree.insert(1, pos, cBlack, true, step, true, 100, Eigen::Vector3f::Zero(), 0.8f, 0.2f, 0.0f, 1.45f, Eigen::Vector3f::Zero(), Grid::BodyType::STATIC, 1.0f);
                    }
                }
            }
        }
    }

    // 2b. Create the 3x3 material sampler grid inside the room
    Eigen::Vector3f cRed(1.0f, 0.1f, 0.1f);
    Eigen::Vector3f cBlue(0.1f, 0.1f, 1.0f);
    Eigen::Vector3f cPurple(0.6f, 0.1f, 0.8f);
    Eigen::Vector3f size(1.0f, 1.0f, 1.0f);
    Eigen::Vector3f cGold(1.00f, 0.80f, 0.30f);
    Eigen::Vector3f cSilver(0.90f, 0.90f, 0.95f);
    Eigen::Vector3f cBrass(0.78f, 0.69f, 0.22f);

    float sp = 2.0f; // spacing between cubes
    Grid::BodyType initType = Grid::BodyType::STATIC;
    float mass = 1.0f;

    // LAYER 1: Metals
    createBox(octree, Eigen::Vector3f(-sp, -sp, 0.0f), size, cGold,   0.0f, 0.08f, 0.99f, 0.0f, 1.45f, Eigen::Vector3f(0,0,0), 1, initType, mass);
    createBox(octree, Eigen::Vector3f(  0, -sp, 0.0f), size, cSilver, 0.0f, 0.08f, 0.99f, 0.0f, 1.45f, Eigen::Vector3f(0,0,0), 2, initType, mass);
    createBox(octree, Eigen::Vector3f( sp, -sp, 0.0f), size, cBrass,  0.0f, 0.08f, 0.99f, 0.0f, 1.45f, Eigen::Vector3f(0,0,0), 3, initType, mass);

    // LAYER 2: Opaque
    createBox(octree, Eigen::Vector3f(-sp,  0,  0.0f), size, cRed,    0.0f, 0.05f, 0.0f, 0.0f, 2.4f, Eigen::Vector3f(0,0,0), 4, initType, mass, 0.5);
    createBox(octree, Eigen::Vector3f(  0,  0,  0.0f), size, cBlue,   0.0f, 0.05f, 0.0f, 0.0f, 2.4f, Eigen::Vector3f(0,0,0), 5, initType, mass, 0.1);
    createBox(octree, Eigen::Vector3f( sp,  0,  0.0f), size, cPurple, 0.0f, 0.05f, 0.0f, 0.0f, 2.4f, Eigen::Vector3f(0,0,0), 6, initType, mass, 0.5);

    // LAYER 3: Glass
    createBox(octree, Eigen::Vector3f(-sp,  sp, 0.0f), size, cRed,    0.0f, 0.01f, 0.0f, 0.99f, 1.5f, Eigen::Vector3f(0.05f, 0.8f, 0.8f), 7, initType, mass, 0.5);
    createBox(octree, Eigen::Vector3f(  0,  sp, 0.0f), size, cBlue,   0.0f, 0.01f, 0.0f, 0.99f, 1.5f, Eigen::Vector3f(0.8f, 0.8f, 0.05f), 8, initType, mass, 0.5);
    createBox(octree, Eigen::Vector3f( sp,  sp, 0.0f), size, cPurple, 0.0f, 0.01f, 0.0f, 0.99f, 1.5f, Eigen::Vector3f(0.4f, 1.2f, 0.4f), 9, initType, mass, 0.5);

    std::cout << "Optimizing and Generating LODs..." << std::endl;
    // octree.generateLODs();
    octree.setLODMinDistance(1024);
    octree.setLODFalloff(0.01);
    octree.printStats();
    octree.setMaxDistance(4096);

    // 3. Setup rendering loop
    int width = 512;
    int height = 512;
    
    const float fps = 30.0f;
    const float durationPerSegment = 10.0f;
    const int framesPerSegment = static_cast<int>(fps * durationPerSegment);
    const int samples = 10;
    const int blendedsamples = 10;
    const float blendedfactor = 0.5;
    const int videosamples = 25;
    const int bounces = 4;
    const int physicsSubsteps = 10;
    const float physicsDt = 1.0f / fps;
    const float subDt = physicsDt / physicsSubsteps;
    const float fluidDuration = 120.0f;
    const int totalFluidFrames = static_cast<int>(fps * fluidDuration);

    struct View {
        std::string name;
        Eigen::Vector3f origin;
        Eigen::Vector3f up;
    };

    std::vector<View> views = {
        {"-Y", Eigen::Vector3f( 0.0f, -6.8f,  2.0f), Eigen::Vector3f(0.0f, 0.0f, 0.5f)},
        {"+X", Eigen::Vector3f( 6.8f,  0.0f,  2.0f), Eigen::Vector3f(0.0f, 0.0f, 0.5f)},
        {"+Y", Eigen::Vector3f( 0.0f,  6.8f,  2.0f), Eigen::Vector3f(0.0f, 0.0f, 0.5f)},
        {"-X", Eigen::Vector3f(-6.8f,  0.0f,  2.0f), Eigen::Vector3f(0.0f, 0.0f, 0.5f)},
        {"+Z", Eigen::Vector3f( 0.0f,  0.0f,  7.3f), Eigen::Vector3f(0.0f, 1.0f, 0.0f)}
    };

    std::vector<StateEvent> timeline = {
        { 20,   5,  1.0f, false, TargetState::FLUID   }, // Center -> Water
        // { 100,  8,  0.00005f, false, TargetState::GAS }, // Second Blue -> Lighter than air Gas
        { 180,  6,  1.2f, false, TargetState::FLUID   }, // Purple 1 -> Heavy Water
        { 260,  9,  0.4f, false, TargetState::RIGID   }, // Purple 2 -> RIGID (Floats lightly in fluid)
        { 340,  3,  8.5f, true,  TargetState::FLUID   }, // Brass -> Molten Brass
        { 420,  2, 10.5f, true,  TargetState::FLUID   }, // Silver -> Molten Silver
        { 500,  1, 19.3f, true,  TargetState::FLUID   }, // Gold -> Molten Gold
        // { 580,  4,  0.001f, false, TargetState::GAS   }, // Red 1 -> heavier GAS
        { 660,  7,  0.1f, false, TargetState::FLUID   }  // Red 2 -> fluid
    };

    Eigen::Vector3f target(0.0f, 0.0f, 0.5f);
    frame out;
    std::string filename;

    for (const auto& view : views) {
        ScopedFunctionTimer meh("Fast section");
        std::cout << "\nRendering view from " << view.name << " direction (Fast Pass)..." << std::endl;
        
        Camera cam;
        cam.origin = view.origin;
        cam.direction = (target - view.origin).normalized();
        cam.up = view.up;
        
        // out = octree.fastRenderFrame(cam, height, width, frame::colormap::RGB);
        // filename = "output/fast_cpurender_" + view.name + ".bmp";
        // BMPWriter::saveBMP(filename, out);
        
        out = octree.fastRenderFrameVulkan(cam, height, width, frame::colormap::RGB);
        filename = "output/fast_vulkanrender_" + view.name + ".bmp";
        BMPWriter::saveBMP(filename, out);
    }
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    for (const auto& view : views) {
        ScopedFunctionTimer meh("Slow Section");
        std::cout << "\nRendering view from " << view.name << " direction (Slow "<< samples <<" Samples Pass)..." << std::endl;
        
        Camera cam;
        cam.origin = view.origin;
        cam.direction = (target - view.origin).normalized();
        cam.up = view.up;
        
        out = octree.renderFrameVulkan(cam, height, width, frame::colormap::RGB, samples, bounces, false, true);
        filename = "output/slow_vulkanrender_" + view.name + ".bmp";
        BMPWriter::saveBMP(filename, out);
        std::cout << "slow done" << std::endl;

        out = octree.blendedRenderFrameVulkan(cam, height, width, blendedfactor, frame::colormap::RGB, blendedsamples, bounces, false, true);
        filename = "output/slow_blendrender_" + view.name + ".bmp";
        BMPWriter::saveBMP(filename, out);
        std::cout << "blended done" << std::endl;
    }
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    std::vector<frame> videoFrames;
    const int totalFrames = framesPerSegment * views.size();
    videoFrames.reserve(totalFrames);
    int frameCounter = 0;

    std::cout << "\nStarting video render..." << std::endl;
    std::cout << "Total frames to render: " << totalFrames << std::endl;

    // for (size_t i = 0; i < views.size(); ++i) {
    //     ScopedFunctionTimer meh("Video");
    //     const View& startView = views[i];
    //     const View& endView = views[(i + 1) % views.size()]; // Loop back to the first view at the end

    //     std::cout << "\nAnimating segment: " << startView.name << " -> " << endView.name << std::endl;

    //     for (int j = 0; j < framesPerSegment; ++j) {
    //         if (frameCounter < -1) {
    //             frameCounter++;
    //             continue;
    //         }
    //         frameCounter++;
    //         float t = static_cast<float>(j) / static_cast<float>(framesPerSegment);

    //         Eigen::Vector3f currentOrigin = startView.origin * (1.0f - t) + endView.origin * t;
            
    //         Eigen::Vector3f currentUp = (startView.up * (1.0f - t) + endView.up * t).normalized();
            
    //         Camera cam;
    //         cam.origin = currentOrigin;
    //         cam.up = currentUp;
    //         cam.direction = (target - cam.origin).normalized();
            
    //         std::cout << "Rendering video frame " << frameCounter << "/" << totalFrames << "..." << std::endl;
    //         // frame out = octree.fastRenderFrameVulkan(cam, height, width, frame::colormap::RGB);
    //         frame out = octree.blendedRenderFrameVulkan(cam, height, width, blendedfactor, frame::colormap::RGB, videosamples, bounces, false);
    //         // frame out = octree.renderFrameVulkan(cam, height, width, frame::colormap::RGB, videosamples, bounces, false, true);
    //         // videoFrames.push_back(std::move(out));
    //         std::string debugFilename = "output/materialframes/debug_material_" + std::to_string(frameCounter) + ".bmp";
    //         BMPWriter::saveBMP(debugFilename, out);
    //     }
    // }
    // FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    // std::cout << "\nAll frames rendered. Saving video file..." << std::endl;
    // std::string videoFilename = "output/material_test_video.y4m";
    
    // y4mWriter::save(videoFilename, videoFrames, fps);
    // if (AVIWriter::saveAVIFromCompressedFrames(videoFilename, std::move(videoFrames), width, height, fps)) {
    //     std::cout << "Video saved successfully to " << videoFilename << std::endl;
    // } else {
    //     std::cerr << "Error: Failed to save video!" << std::endl;
    // }

    std::cout << "\nStarting DYNAMIC FLUID video render (Time flows!)..." << std::endl;
    
    std::vector<frame> fluidVideoFrames;

    fluidVideoFrames.reserve(totalFluidFrames);
    int fluidframeCounter = 0;
    int framesPerView = totalFluidFrames / views.size();

    // std::vector<Grid::Octree<int>::NodeData> trackedWater = octree.collectNodesByObjectId(5);

    Camera cam;
    cam.fov = 100;
    for (size_t i = 0; i < views.size(); ++i) {
        ScopedFunctionTimer meh("Fluid");
        const View& startView = views[i];
        const View& endView = views[(i + 1) % views.size()]; // Loop back to the first view at the end

        std::cout << "\nAnimating segment: " << startView.name << " -> " << endView.name << std::endl;

        for (int j = 0; j < framesPerView; ++j) {
            fluidframeCounter++;
            
            // Check if it's time to melt a block
            for (const auto& event : timeline) {
                if (fluidframeCounter == event.frameTrigger) {
                    std::cout << ">>> TRIGGERING STATE CHANGE for Object ID: " << event.objectId << std::endl;
                    // if (event.targetState == TargetState::GAS) {
                    //     // Vaporize the object into the Eulerian gas field
                    //     octree.vaporizeObject(event.objectId, 1.0f);
                    // } else {
                        Grid::BodyType targetType = (event.targetState == TargetState::FLUID) ? Grid::BodyType::FLUID : Grid::BodyType::RIGID;
                        octree.makeObjectFluid(event.objectId, event.mass, targetType);
                        
                        if (event.isMoltenMetal) {
                            // Make metals glow slightly when molten and adjust PBR values
                            octree.setMaterialByObjectId(event.objectId, Eigen::Vector3f::Constant(1.5f), 0.2f, 1.0f, 1.45f, Eigen::Vector3f::Zero());
                        }
                    // }
                }
            }

            // if (fluidframeCounter >= 10 && fluidframeCounter <= 200) {
            //     for (auto& wp : trackedWater) {
            //         if (auto sp = wp.lock()) {
            //             sp->size += 0.0005f;
            //             sp->physics.mass += 0.0005f;
            //         }
            //     }
            // }

            // Step physics
            for (int s = 0; s < physicsSubsteps; ++s) {
                octree.stepPhysics(subDt);
            }

            // Interpolate camera
            float t = static_cast<float>(j) / static_cast<float>(framesPerView);
            cam.origin = startView.origin * (1.0f - t) + endView.origin * t;
            cam.up = (startView.up * (1.0f - t) + endView.up * t).normalized();
            cam.direction = (target - cam.origin).normalized();
            
            std::cout << "Rendering video frame " << fluidframeCounter << "/" << totalFluidFrames << "..." << std::endl;

            // 3. Render
            // frame out = octree.fastRenderFrameVulkan(cam, height, width, frame::colormap::RGB);
            // frame out = octree.blendedRenderFrameVulkan(cam, height, width, blendedfactor, frame::colormap::RGB, videosamples, bounces, false, true);
            frame out = octree.renderFrameVulkan(cam, height, width, frame::colormap::RGB, videosamples, bounces, true, false);
            // fluidVideoFrames.push_back(out);

            // saving to video is dumb so just gonna export here and then convert.
            std::string debugFilename = "output/fluidframes/debug_fluid_" + std::to_string(fluidframeCounter) + ".bmp";
            BMPWriter::saveBMP(debugFilename, out);
        }
    }

    // std::string fluidVideoFilename = "output/material_fluid_video.y4m";
    // y4mWriter::save(fluidVideoFilename, fluidVideoFrames, fps);
    // if (AVIWriter::saveAVIFromCompressedFrames(fluidVideoFilename, std::move(fluidVideoFrames), width, height, fps)) {
    //     std::cout << "Fluid Simulation video saved to " << fluidVideoFilename << std::endl;
    // }
    
    std::cout << "\nAll renders complete!" << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}