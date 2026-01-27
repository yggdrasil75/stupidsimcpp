#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <random>

#include "../util/grid/grid3eigen.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/output/frame.hpp"
#include "../util/timing_decorator.cpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/output/aviwriter.hpp"

#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "../stb/stb_image.h"

int main() {
    // Define octree boundaries (world space)
    using PointType = Eigen::Matrix<float, 3, 1>;
    PointType minBound(-2.0f, -2.0f, -2.0f);
    PointType maxBound(2.0f, 2.0f, 2.0f);
    
    // Create octree
    Octree<int> octree(minBound, maxBound, 16, 8); // max 16 points per node, max depth 8
    
    // Create green sphere (center at origin, radius 1.0)
    float radius = 1.0f;
    int pointCount = 0;
    Eigen::Vector3f greenColor(0.0f, 1.0f, 0.0f); // green
    
    // Add points on sphere surface (simplified representation)
    for (int i = 0; i < 10000; ++i) {
        // Spherical coordinates
        float u = static_cast<float>(rand()) / RAND_MAX;
        float v = static_cast<float>(rand()) / RAND_MAX;
        
        // Convert to spherical coordinates
        float theta = 2.0f * M_PI * u;  // azimuth
        float phi = acos(2.0f * v - 1.0f);  // polar
        
        // Convert to Cartesian coordinates
        float x = radius * sin(phi) * cos(theta);
        float y = radius * sin(phi) * sin(theta);
        float z = radius * cos(phi);
        
        PointType pos(x, y, z);
        
        // Set point data with larger size for visibility
        // Note: The third parameter is size, which should be radius squared for intersection test
        octree.set(i, pos, true, greenColor, 1.0, true);
        pointCount++;
    }
    
    std::cout << "Added " << pointCount << " points to the green sphere." << std::endl;
    
    // Render parameters
    int width = 2048;
    int height = 2048;
    
    // Set up random number generator for camera positions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> elevationDist(0.1f, M_PI - 0.1f); // Avoid poles
    std::uniform_real_distribution<float> radiusDist(2.0f, 4.0f); // Distance from sphere
    
    // Generate and save 15 random views
    const int numViews = 15;
    
    for (int viewIndex = 0; viewIndex < numViews; ++viewIndex) {
        std::cout << "\nRendering view " << (viewIndex + 1) << " of " << numViews << "..." << std::endl;
        
        // Generate random spherical coordinates for camera position
        float azimuth = angleDist(gen);
        float elevation = elevationDist(gen);
        float camRadius = radiusDist(gen);
        
        // Convert to Cartesian coordinates for camera position
        float camX = camRadius * sin(elevation) * cos(azimuth);
        float camY = camRadius * sin(elevation) * sin(azimuth);
        float camZ = camRadius * cos(elevation);
        
        // Camera looks at the origin (center of sphere)
        Vector3f cameraPos(camX, camY, camZ);
        Vector3f lookAt(0.0f, 0.0f, 0.0f);
        
        // Calculate camera direction (from position to lookAt)
        Vector3f cameraDir = (lookAt - cameraPos).normalized();
        
        // Calculate up vector (avoid gimbal lock)
        Vector3f worldUp(0.0f, 1.0f, 0.0f);
        Vector3f right = cameraDir.cross(worldUp).normalized();
        Vector3f cameraUp = right.cross(cameraDir).normalized();
        
        // Create camera
        Camera cam(cameraPos, cameraDir, cameraUp, 80);
        
        // Render frame
        frame renderedFrame = octree.renderFrame(cam, height, width, frame::colormap::RGB);
        
        std::cout << "Frame rendered. Dimensions: " 
                  << renderedFrame.getWidth() << "x" 
                  << renderedFrame.getHeight() << std::endl;
        
        // Save as BMP
        std::string filename = "output/green_sphere_view_" + std::to_string(viewIndex + 1) + ".bmp";
        std::cout << "Saving to " << filename << "..." << std::endl;
        
        if (BMPWriter::saveBMP(filename, renderedFrame)) {
            std::cout << "Successfully saved view to " << filename << std::endl;
            
            // Print camera position for reference
            std::cout << "Camera position: (" << camX << ", " << camY << ", " << camZ << ")" << std::endl;
        } else {
            std::cerr << "Failed to save BMP file: " << filename << std::endl;
        }
        
        // Small delay to ensure unique random seeds
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::cout << "\nAll " << numViews << " views have been saved to the output directory." << std::endl;
    
    return 0;
}