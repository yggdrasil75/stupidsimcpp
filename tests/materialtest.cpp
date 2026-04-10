#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

// Include Eigen and project headers
#include "../eigen/Eigen/Dense" 
#include "../util/grid/camera.hpp"
#include "../util/grid/grid3eigen.hpp"
#include "../util/output/frame.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/aviwriter.hpp"
#include "../util/timing_decorator.hpp"
#include "../util/timing_decorator.cpp"

// Helper function to create a solid volume of voxels with material properties
void createBox(Octree<int>& octree, const Eigen::Vector3f& center, const Eigen::Vector3f& size, 
               const Eigen::Vector3f& albedo, float emission = 0.0f,
               float roughness = 0.8f, float metallic = 0.0f, float transmission = 0.0f, float ior = 1.45f) {
    float step = 0.1f; // Voxel spacing
    Eigen::Vector3f halfSize = size / 2.0f;
    Eigen::Vector3f minB = center - halfSize;
    Eigen::Vector3f maxB = center + halfSize;
    
    for (float x = minB.x(); x <= maxB.x(); x += step) {
        for (float y = minB.y(); y <= maxB.y(); y += step) {
            for (float z = minB.z(); z <= maxB.z(); z += step) {
                Eigen::Vector3f pos(x, y, z);
                
                // .set(data, pos, visible, albedo, size, active, objectId, subId, emission, roughness, metallic, transmission, ior)
                octree.set(1, pos, true, albedo, step, true, -1, emission, roughness, metallic, transmission, ior);
            }
        }
    }
}

// Helper function to create a checkerboard pattern volume
void createCheckerBox(Octree<int>& octree, const Eigen::Vector3f& center, const Eigen::Vector3f& size, 
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
                
                octree.set(1, pos, true, albedo, step, true, -1, 0.0f, 0.8f, 0.1f, 0.0f, 1.0f);
            }
        }
    }
}

int main() {
    std::cout << "Initializing Octree..." << std::endl;

    // 1. Initialize Octree bounds
    Eigen::Vector3f minBound(-10.0f, -10.0f, -10.0f);
    Eigen::Vector3f maxBound(10.0f, 10.0f, 10.0f);
    Octree<int> octree(minBound, maxBound, 8, 16);
    
    // Set a dark background to emphasize the PBR light emission
    octree.setBackgroundColor(Eigen::Vector3f(0.02f, 0.02f, 0.02f));
    octree.setSkylight(Eigen::Vector3f(0.01f, 0.01f, 0.01f));

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

    float sp = 2.0f; // spacing between cubes

    // --- LAYER 1: Metals ---
    // (metallic = 1.0, slight roughness for blurry reflections, transmission = 0.0)
    createBox(octree, Eigen::Vector3f(-sp, -sp, 0.0f), size, cRed,    0.0f, 0.15f, 1.0f, 0.0f, 1.45f);
    createBox(octree, Eigen::Vector3f(  0, -sp, 0.0f), size, cBlue,   0.0f, 0.15f, 1.0f, 0.0f, 1.45f);
    createBox(octree, Eigen::Vector3f( sp, -sp, 0.0f), size, cPurple, 0.0f, 0.15f, 1.0f, 0.0f, 1.45f);

    // --- LAYER 2: Opaque & Highly Refractive ---
    // (metallic = 0.0, very low roughness. transmission = 0.0 for opacity, ior = 2.4 for extreme diamond-like reflection)
    createBox(octree, Eigen::Vector3f(-sp,  0,  0.0f), size, cRed,    0.0f, 0.05f, 0.0f, 0.0f, 2.4f);
    createBox(octree, Eigen::Vector3f(  0,  0,  0.0f), size, cBlue,   0.0f, 0.05f, 0.0f, 0.0f, 2.4f);
    createBox(octree, Eigen::Vector3f( sp,  0,  0.0f), size, cPurple, 0.0f, 0.05f, 0.0f, 0.0f, 2.4f);

    // --- LAYER 3: Clear Glass ---
    // (metallic = 0.0, near-zero roughness, transmission = 1.0 for full transparency, ior = 1.5 for glass)
    createBox(octree, Eigen::Vector3f(-sp,  sp, 0.0f), size, cRed,    0.0f, 0.01f, 0.0f, 1.0f, 1.5f);
    createBox(octree, Eigen::Vector3f(  0,  sp, 0.0f), size, cBlue,   0.0f, 0.01f, 0.0f, 1.0f, 1.5f);
    createBox(octree, Eigen::Vector3f( sp,  sp, 0.0f), size, cPurple, 0.0f, 0.01f, 0.0f, 1.0f, 1.5f);

    // White Light Box (Above)
    // Placed near the ceiling (Z=7.4), made large (8x8) to cast soft shadows evenly over the whole 3x3 grid
    createBox(octree, Eigen::Vector3f(0.0f, 0.0f, 7.4f), Eigen::Vector3f(8.0f, 8.0f, 0.2f), Eigen::Vector3f(1.0f, 1.0f, 1.0f), 15.0f);

    std::cout << "Optimizing and Generating LODs..." << std::endl;
    octree.generateLODs();
    octree.printStats();

    // 3. Setup rendering loop
    int width = 512;
    int height = 512;
    
    const float fps = 10.0f;
    const float durationPerSegment = 1.0f;
    const int framesPerSegment = static_cast<int>(fps * durationPerSegment);
    const int video_samples = 400;
    const int video_bounces = 5;

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

    Eigen::Vector3f target(0.0f, 0.0f, 0.5f);

    for (const auto& view : views) {
        std::cout << "\nRendering view from " << view.name << " direction (Fast Pass)..." << std::endl;
        
        Camera cam;
        cam.origin = view.origin;
        cam.direction = (target - view.origin).normalized();
        cam.up = view.up;
        
        frame out = octree.fastRenderFrame(cam, height, width, frame::colormap::RGB);
        
        std::string filename = "output/fast/render_" + view.name + ".bmp";
        BMPWriter::saveBMP(filename, out);
    }
    
    for (const auto& view : views) {
        std::cout << "\nRendering view from " << view.name << " direction (Medium 60s Pass)..." << std::endl;
        
        Camera cam;
        cam.origin = view.origin;
        cam.direction = (target - view.origin).normalized();
        cam.up = view.up;
        
        frame out = octree.renderFrameTimed(cam, height, width, frame::colormap::RGB, 60, bounces, false, true);
        
        std::string filename = "output/medium/render_" + view.name + ".bmp";
        BMPWriter::saveBMP(filename, out);
    }
    
    std::vector<frame> videoFrames;
    const int totalFrames = framesPerSegment * views.size();
    videoFrames.reserve(totalFrames);
    int frameCounter = 0;

    std::cout << "\nStarting video render..." << std::endl;
    std::cout << "Total frames to render: " << totalFrames << std::endl;

    for (size_t i = 0; i < views.size(); ++i) {
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
            
            // frame out = octree.renderFrame(cam, height, width, frame::colormap::RGB, video_samples, video_bounces, false, true);
            frame out = octree.renderFramefast(cam, height, width, frame::colormap::RGB, false, true);
            
            videoFrames.push_back(std::move(out));
        }
    }

    std::cout << "\nAll frames rendered. Saving video file..." << std::endl;
    std::string videoFilename = "output/material_test_video.avi";
    
    if (AVIWriter::saveAVIFromCompressedFrames(videoFilename, std::move(videoFrames), width, height, fps)) {
        std::cout << "Video saved successfully to " << videoFilename << std::endl;
    } else {
        std::cerr << "Error: Failed to save video!" << std::endl;
    }
    
    std::cout << "\nRender complete!" << std::endl;

    std::cout << "\nAll renders complete!" << std::endl;
    return 0;
}