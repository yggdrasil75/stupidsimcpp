#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>

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
    for (int i = 0; i < 1000; ++i) {
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
        octree.set(i, pos, greenColor, 1.f, true);
        pointCount++;
    }
    
    std::cout << "Added " << pointCount << " points to the green sphere." << std::endl;
    
    // Set camera parameters
    PointType cameraPos(0.0f, 0.0f, 5.0f); // camera position
    PointType lookDir(0.0f, 0.0f, -1.0f);  // looking at sphere
    PointType upDir(0.0f, 1.0f, 0.0f);     // up direction
    PointType rightDir(1.0f, 0.0f, 0.0f);  // right direction
    
    // Render parameters
    int width = 800;
    int height = 600;
    
    std::cout << "Rendering frame..." << std::endl;
    
    // Render frame
    frame renderedFrame = octree.renderFrame(cameraPos, lookDir, upDir, rightDir, height, width, frame::colormap::RGB);
    
    std::cout << "Frame rendered. Dimensions: " 
              << renderedFrame.getWidth() << "x" 
              << renderedFrame.getHeight() << std::endl;
    
    // Save as BMP
    std::string filename = "output/green_sphere.bmp";
    std::cout << "Saving to " << filename << "..." << std::endl;
    
    if (BMPWriter::saveBMP(filename, renderedFrame)) {
        std::cout << "Successfully saved green sphere to " << filename << std::endl;
    } else {
        std::cerr << "Failed to save BMP file!" << std::endl;
        return 1;
    }
    
    return 0;
}