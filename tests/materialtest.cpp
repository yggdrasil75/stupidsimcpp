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
               int oid = 0, Grid::BodyType bType = Grid::BodyType::STATIC, float mass = 1.0f) {
    float step = 0.05f; // Voxel spacing
    Eigen::Vector3f halfSize = size / 2.0f;
    Eigen::Vector3f minB = center - halfSize;
    Eigen::Vector3f maxB = center + halfSize;
    static std::mt19937 rng(1337);
    std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);
    
    for (float x = minB.x(); x <= maxB.x(); x += step) {
        for (float y = minB.y(); y <= maxB.y(); y += step) {
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                Eigen::Vector3f pos(x + jitter(rng), y + jitter(rng), z + jitter(rng));
                
                // .set(data, pos, visible, albedo, size, active, objectId, subId, emission, roughness, metallic, transmission, ior, absorption)
                octree.set(1, pos, true, albedo, step, true, oid, emission, roughness, metallic, transmission, ior, absorp, bType, mass);
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
                
                octree.set(1, pos, true, albedo, step, true, 100, 0.0f, 0.8f, 0.2f, 0.2f, 1.45f);
            }
        }
    }
}

struct MeltEvent {
    int frameTrigger;
    int objectId;
    float mass;
    bool isMoltenMetal;
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
    octree.setPhysicsGasConstant(100.0f); // Lowered significantly from 2000 for stability
    octree.setPhysicsVelocityDamping(1.0f); // Higher damping stops infinite scattering
    octree.setPhysicsViscosity(15.0f);

    std::cout << "Building scene..." << std::endl;

    // 2a. Build Room (Floor and 4 Walls)
    Eigen::Vector3f cLightGray(0.8f, 0.8f, 0.8f);
    Eigen::Vector3f cDarkGray(0.2f, 0.2f, 0.2f);
    float chkSize = 1.0f;

    // Floor (Bounds: Z from -0.7 to -0.5)
    // The boxes sit exactly on Z = -0.5
    createCheckerBox(octree, Eigen::Vector3f(0.0f, 0.0f, -0.6f), Eigen::Vector3f(14.4f, 14.4f, 0.2f), cLightGray, cDarkGray, chkSize);
    
    // Walls (Bounds: X/Y inner boundaries at +/- 7.0, rising from Z=-0.5 up to Z=7.5)
    createCheckerBox(octree, Eigen::Vector3f( 7.1f,  0.0f, 3.5f), Eigen::Vector3f(0.2f, 14.4f, 8.0f), cLightGray, cDarkGray, chkSize); // +X
    createCheckerBox(octree, Eigen::Vector3f(-7.1f,  0.0f, 3.5f), Eigen::Vector3f(0.2f, 14.4f, 8.0f), cLightGray, cDarkGray, chkSize); // -X
    createCheckerBox(octree, Eigen::Vector3f( 0.0f,  7.1f, 3.5f), Eigen::Vector3f(14.0f, 0.2f, 8.0f), cLightGray, cDarkGray, chkSize); // +Y
    createCheckerBox(octree, Eigen::Vector3f( 0.0f, -7.1f, 3.5f), Eigen::Vector3f(14.0f, 0.2f, 8.0f), cLightGray, cDarkGray, chkSize); // -Y

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
    createBox(octree, Eigen::Vector3f(-sp,  0,  0.0f), size, cRed,    0.0f, 0.05f, 0.0f, 0.0f, 2.4f, Eigen::Vector3f(0,0,0), 4, initType, mass);
    createBox(octree, Eigen::Vector3f(  0,  0,  0.0f), size, cBlue,   0.0f, 0.05f, 0.0f, 0.0f, 2.4f, Eigen::Vector3f(0,0,0), 5, initType, mass);
    createBox(octree, Eigen::Vector3f( sp,  0,  0.0f), size, cPurple, 0.0f, 0.05f, 0.0f, 0.0f, 2.4f, Eigen::Vector3f(0,0,0), 6, initType, mass);

    // LAYER 3: Glass
    createBox(octree, Eigen::Vector3f(-sp,  sp, 0.0f), size, cRed,    0.0f, 0.01f, 0.0f, 0.99f, 1.5f, Eigen::Vector3f(0.05f, 0.8f, 0.8f), 7, initType, mass);
    createBox(octree, Eigen::Vector3f(  0,  sp, 0.0f), size, cBlue,   0.0f, 0.01f, 0.0f, 0.99f, 1.5f, Eigen::Vector3f(0.8f, 0.8f, 0.05f), 8, initType, mass);
    createBox(octree, Eigen::Vector3f( sp,  sp, 0.0f), size, cPurple, 0.0f, 0.01f, 0.0f, 0.99f, 1.5f, Eigen::Vector3f(0.4f, 1.2f, 0.4f), 9, initType, mass);

    createBox(octree, Eigen::Vector3f(0.0f, 0.0f, 7.4f), Eigen::Vector3f(8.0f, 8.0f, 0.2f), Eigen::Vector3f(1.0f, 1.0f, 1.0f), 15.0f, 0.8f, 0.0f, 0.0f, 1.45f, Eigen::Vector3f::Zero(), 10);

    std::cout << "Optimizing and Generating LODs..." << std::endl;
    octree.generateLODs();
    octree.printStats();
    octree.setMaxDistance(4096);

    // 3. Setup rendering loop
    int width = 512;
    int height = 512;
    
    const float fps = 10.0f;
    const float durationPerSegment = 10.0f;
    const int framesPerSegment = static_cast<int>(fps * durationPerSegment);
    const int samples = 10;
    const int blendedsamples = 10;
    const float blendedfactor = 0.5;
    const int video_samples = 250;
    const int bounces = 32;
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
        {"-Y", Eigen::Vector3f( 0.0f, -6.8f,  1.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f)},
        {"+X", Eigen::Vector3f( 6.8f,  0.0f,  1.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f)},
        {"+Y", Eigen::Vector3f( 0.0f,  6.8f,  1.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f)},
        {"-X", Eigen::Vector3f(-6.8f,  0.0f,  1.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f)},
        {"+Z", Eigen::Vector3f( 0.0f,  0.0f,  7.3f), Eigen::Vector3f(0.0f, 1.0f, 0.0f)}
    };
    std::vector<MeltEvent> timeline = {
        { 20,   5,  1.0f, false }, // Center -> Water
        { 100,  8,  1.0f, false }, // Second Blue -> Water
        { 180,  6,  1.2f, false }, // Purple 1 -> Heavy Water
        { 260,  9,  1.2f, false }, // Purple 2 -> Heavy Water
        { 340,  3,  8.5f, true  }, // Brass -> Molten Brass
        { 420,  2, 10.5f, true  }, // Silver -> Molten Silver
        { 500,  1, 19.3f, true  }, // Gold -> Molten Gold
        { 580,  4,  0.8f, false }, // Red 1 -> Oil
        { 660,  7,  0.8f, false }  // Red 2 -> Oil
    };

    Eigen::Vector3f target(0.0f, 0.0f, 0.5f);

    for (const auto& view : views) {
        ScopedFunctionTimer meh("Fast section");
        std::cout << "\nRendering view from " << view.name << " direction (Fast Pass)..." << std::endl;
        
        Camera cam;
        cam.origin = view.origin;
        cam.direction = (target - view.origin).normalized();
        cam.up = view.up;
        
        frame out = octree.fastRenderFrame(cam, height, width, frame::colormap::RGB);
        std::string filename = "output/fast_cpurender_" + view.name + ".bmp";
        BMPWriter::saveBMP(filename, out);
        
        out = octree.fastRenderFrameVulkan(cam, height, width, frame::colormap::RGB);
        filename = "output/fast_vulkanrender_" + view.name + ".bmp";
        // out.compressFrameLZ78();
        // out.decompress();
        BMPWriter::saveBMP(filename, out);
    }
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    // for (const auto& view : views) {
    //     ScopedFunctionTimer meh("Slow Section");
    //     std::cout << "\nRendering view from " << view.name << " direction (Slow "<< samples <<" Samples Pass)..." << std::endl;
        
    //     Camera cam;
    //     cam.origin = view.origin;
    //     cam.direction = (target - view.origin).normalized();
    //     cam.up = view.up;
        
    //     frame out = octree.renderFrameVulkan(cam, height, width, frame::colormap::RGB, samples, bounces, false, true);
    //     std::string filename = "output/slow_vulkanrender_" + view.name + ".bmp";
    //     BMPWriter::saveBMP(filename, out);
    //     std::cout << "slow done" << std::endl;

    //     out = octree.blendedRenderFrameVulkan(cam, height, width, blendedfactor, frame::colormap::RGB, blendedsamples, bounces, false, true);
    //     filename = "output/slow_blendrender_" + view.name + ".bmp";
    //     BMPWriter::saveBMP(filename, out);
    //     std::cout << "blended done" << std::endl;
    // }
    // FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);

    std::vector<frame> videoFrames;
    const int totalFrames = framesPerSegment * views.size();
    videoFrames.reserve(totalFrames);
    int frameCounter = 0;

    std::cout << "\nStarting video render..." << std::endl;
    std::cout << "Total frames to render: " << totalFrames << std::endl;

    for (size_t i = 0; i < views.size(); ++i) {
        ScopedFunctionTimer meh("Video");
        const View& startView = views[i];
        const View& endView = views[(i + 1) % views.size()]; // Loop back to the first view at the end

        std::cout << "\nAnimating segment: " << startView.name << " -> " << endView.name << std::endl;

        for (int j = 0; j < framesPerSegment; ++j) {
            frameCounter++;
            float t = static_cast<float>(j) / static_cast<float>(framesPerSegment);

            Eigen::Vector3f currentOrigin = startView.origin * (1.0f - t) + endView.origin * t;
            
            Eigen::Vector3f currentUp = (startView.up * (1.0f - t) + endView.up * t).normalized();
            
            Camera cam;
            cam.origin = currentOrigin;
            cam.up = currentUp;
            cam.direction = (target - cam.origin).normalized();
            
            std::cout << "Rendering video frame " << frameCounter << "/" << totalFrames << "..." << std::endl;
            frame out = octree.fastRenderFrameVulkan(cam, height, width, frame::colormap::RGB);
            // frame out = octree.blendedRenderFrameVulkan(cam, height, width, blendedfactor, frame::colormap::RGB, video_samples, bounces, false);
            // frame out = octree.renderFramefast(cam, height, width, frame::colormap::RGB, false, true);
            videoFrames.push_back(std::move(out));
        }
    }

    std::cout << "\nAll frames rendered. Saving video file..." << std::endl;
    std::string videoFilename = "output/material_test_video.y4m";
    
    y4mWriter::save(videoFilename, videoFrames, fps);
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
                    std::cout << ">>> TRIGGERING MELT for Object ID: " << event.objectId << std::endl;
                    octree.makeObjectFluid(event.objectId, event.mass);
                    
                    if (event.isMoltenMetal) {
                        // Make metals glow slightly when molten and adjust PBR values
                        octree.setMaterialByObjectId(event.objectId, 1.5f, 0.2f, 1.0f);
                    }
                }
            }

            // Step physics
            for (int s = 0; s < physicsSubsteps; ++s) {
                octree.stepPhysics(subDt);
            }

            // Interpolate camera
            float t = static_cast<float>(j) / static_cast<float>(framesPerView);
            Camera cam;
            cam.origin = startView.origin * (1.0f - t) + endView.origin * t;
            cam.up = (startView.up * (1.0f - t) + endView.up * t).normalized();
            cam.direction = (target - cam.origin).normalized();
            
            std::cout << "Rendering video frame " << fluidframeCounter << "/" << totalFluidFrames << "..." << std::endl;

            // 3. Render
            frame out = octree.fastRenderFrameVulkan(cam, height, width, frame::colormap::RGB);
            fluidVideoFrames.push_back(out);

            // SAVE DEBUG FRAME EVERY 10 FRAMES
            if (fluidframeCounter % 10 == 0) {
                std::string debugFilename = "output/fluidframes/debug_fluid_" + std::to_string(fluidframeCounter) + ".bmp";
                BMPWriter::saveBMP(debugFilename, out);
            }
        }
    }

    std::string fluidVideoFilename = "output/material_fluid_video.y4m";
    y4mWriter::save(fluidVideoFilename, fluidVideoFrames, fps);
    // if (AVIWriter::saveAVIFromCompressedFrames(fluidVideoFilename, std::move(fluidVideoFrames), width, height, fps)) {
    //     std::cout << "Fluid Simulation video saved to " << fluidVideoFilename << std::endl;
    // }
    
    std::cout << "\nAll renders complete!" << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}