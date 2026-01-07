// main.cpp - Noise Grid Generator and Renderer
#include <iostream>
#include <memory>
#include <cmath>
#include <vector>
#include "../util/grid/grid3.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/timing_decorator.cpp"

// Constants
const size_t GRID_SIZE = 128;
const size_t RENDER_WIDTH = 512;
const size_t RENDER_HEIGHT = 512;

// Function to generate grayscale noise grid
void generateNoiseGrid(VoxelGrid& grid, PNoise2& noise) {
    TIME_FUNCTION;
    std::cout << "Generating 1024x1024x1024 noise grid..." << std::endl;
    
    // Generate Perlin noise in the grid
    #pragma omp parallel for
    for (size_t z = 0; z < GRID_SIZE; ++z) {
        if (z % 64 == 0) {
            std::cout << "Processing layer " << z << " of " << GRID_SIZE << std::endl;
        }
        
        //#pragma omp parallel for
        for (size_t y = 0; y < GRID_SIZE; ++y) {
            //#pragma omp parallel for
            for (size_t x = 0; x < GRID_SIZE; ++x) {
                // Create 3D noise coordinates (scaled for better frequency)
                float scale = 0.05f; // Controls noise frequency
                float noiseVal = noise.permute(Vec3f(x * scale, y * scale, z * scale));
                
                // Convert from [-1, 1] to [0, 1] range
                //float normalizedNoise = (noiseVal + 1.0f) * 0.5f;
                
                // Apply threshold to make some voxels "active"
                // Higher threshold = sparser voxels
                float threshold = 0.3;
                float active = (noiseVal > threshold) ? noiseVal : 0.0f;
                
                // Create grayscale color based on noise value
                uint8_t grayValue = static_cast<uint8_t>(noiseVal * 255);
                Vec3ui8 color(grayValue, grayValue, 0);
                #pragma omp critical
                if (active > threshold) {
                    grid.set(x, y, z, true, color);
                }
            }
        }
    }
    
    std::cout << "Noise grid generation complete!" << std::endl;
}

// Function to render grid from a specific viewpoint
bool renderView(const std::string& filename, VoxelGrid& grid, const Vec3f& position, 
                const Vec3f& direction, const Vec3f& up = Vec3f(0, 1, 0)) {
    TIME_FUNCTION;
    Camera cam(position, direction, up, 40);
    
    std::vector<uint8_t> renderBuffer;
    size_t width = RENDER_WIDTH;
    size_t height = RENDER_HEIGHT;
    
    // Render the view
    frame output = grid.renderFrame(position, direction, up, 40, RENDER_WIDTH, RENDER_HEIGHT);
    //grid.renderOut(renderBuffer, width, height, cam);
    
    // Save to BMP
    bool success = BMPWriter::saveBMP(filename, output);
    //bool success = BMPWriter::saveBMP(filename, renderBuffer, width, height);
    
    // if (success) {
    //     std::cout << "Saved: " << filename << std::endl;
    // } else {
    //     std::cout << "Failed to save: " << filename << std::endl;
    // }
    
    return success;
}

Vec3f rotateX(const Vec3f& vec, float angle) {
    TIME_FUNCTION;
    float cosA = cos(angle);
    float sinA = sin(angle);
    return Vec3f(
        vec.x,
        vec.y * cosA - vec.z * sinA,
        vec.y * sinA + vec.z * cosA
    );
}

Vec3f rotateY(const Vec3f& vec, float angle) {
    TIME_FUNCTION;
    float cosA = cos(angle);
    float sinA = sin(angle);
    return Vec3f(
        vec.x * cosA + vec.z * sinA,
        vec.y,
        -vec.x * sinA + vec.z * cosA
    );
}

Vec3f rotateZ(const Vec3f& vec, float angle) {
    TIME_FUNCTION;
    float cosA = cos(angle);
    float sinA = sin(angle);
    return Vec3f(
        vec.x * cosA - vec.y * sinA,
        vec.x * sinA + vec.y * cosA,
        vec.z
    );
}

int main() {
    TIME_FUNCTION;
    std::cout << "=== Noise Grid Generator and Renderer ===" << std::endl;
    std::cout << "Grid Size: 1024x1024x1024 voxels" << std::endl;
    std::cout << "Render Size: 512x512 pixels" << std::endl;
    
    // Initialize Perlin noise generator
    PNoise2 noise;
    
    // Create voxel grid
    VoxelGrid grid(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    
    // Generate noise grid
    generateNoiseGrid(grid, noise);
    
    // Define center of the grid for camera positioning
    Vec3f gridCenter(GRID_SIZE / 2.0f, GRID_SIZE / 2.0f, GRID_SIZE / 2.0f);
    
    // Camera distance from center (outside the grid)
    float cameraDistance = GRID_SIZE * 2.0f;
    
    int numFrames = 360;
    
    // Base camera position (looking from front)
    Vec3f basePosition(0, 0, cameraDistance);
    Vec3f baseDirection(0, 0, -1);  // Looking towards negative Z (towards center)
    Vec3f up(0, 1, 0);
    
    // // Render frames around 180 degrees
    // for (int i = 0; i <= numFrames; i++) {
    //     float angle = (float)i / numFrames * M_PI;  // 0 to π (180 degrees)
        
    //     // Rotate camera position around Y axis
    //     Vec3f rotatedPos = rotateY(basePosition, angle);
    //     Vec3f finalPos = gridCenter + rotatedPos;
    //     //Vec3f rotatedDir = rotateY(baseDirection, angle);
    //     Vec3f rotatedDir = (gridCenter - finalPos).normalized();
        
    //     // Create filename with frame number
    //     char filename[256];
    //     snprintf(filename, sizeof(filename), "output/framey_%03d.bmp", i);
        
    //     // std::cout << "Rendering frame " << i << "/" << numFrames 
    //     //             << " (angle: " << (angle * 360.0f / M_PI) << " degrees)" << std::endl;
        
    //     renderView(filename, grid, finalPos, rotatedDir, up);
    // }
    
    // basePosition = Vec3f(0, 0, cameraDistance);
    // baseDirection = Vec3f(0, 0, -1);
    // up = Vec3f(0, 1, 0);

    // for (int i = 0; i <= numFrames; i++) {
    //     float angle = (float)i / numFrames * M_PI;  // 0 to π (180 degrees)
        
    //     // Rotate camera position around Y axis
    //     Vec3f rotatedPos = rotateZ(basePosition, angle);
    //     Vec3f finalPos = gridCenter + rotatedPos;
    //     //Vec3f rotatedDir = rotateY(baseDirection, angle);
    //     Vec3f rotatedDir = (gridCenter - finalPos).normalized();
        
    //     // Create filename with frame number
    //     char filename[256];
    //     snprintf(filename, sizeof(filename), "output/framez_%03d.bmp", i);
        
    //     // std::cout << "Rendering frame " << i << "/" << numFrames 
    //     //             << " (angle: " << (angle * 360.0f / M_PI) << " degrees)" << std::endl;
        
    //     renderView(filename, grid, finalPos, rotatedDir, up);
    // }
    
    // basePosition = Vec3f(0, 0, cameraDistance);
    // baseDirection = Vec3f(0, 0, -1);
    // up = Vec3f(0, 1, 0);

    for (int i = 0; i <= numFrames; i++) {
        float angle = (float)i / numFrames * M_PI;  // 0 to π (180 degrees)
        
        // Rotate camera position around Y axis
        Vec3f rotatedPos = rotateX(basePosition, angle);
        Vec3f finalPos = gridCenter + rotatedPos;
        //Vec3f rotatedDir = rotateY(baseDirection, angle);
        Vec3f rotatedDir = (gridCenter - finalPos).normalized();
        
        // Create filename with frame number
        char filename[256];
        snprintf(filename, sizeof(filename), "output/framex_%03d.bmp", i);
        
        std::cout << "Rendering frame " << i << "/" << numFrames 
                    << " (angle: " << (angle * 360.0f / M_PI) << " degrees)" << std::endl;
        
        renderView(filename, grid, finalPos, rotatedDir, up);
    }
    
    std::cout << "\nRendering zenith and nadir views..." << std::endl;
    renderView("output/zenith_view.bmp", grid,
                gridCenter + Vec3f(0, cameraDistance, 0),
                Vec3f(0, -1, 0),   // Look down
                Vec3f(0, 0, -1));  // Adjust up vector for proper orientation
    
    // Nadir view (looking from below)
    renderView("output/nadir_view.bmp", grid,
                gridCenter + Vec3f(0, -cameraDistance, 0),
                Vec3f(0, 1, 0),    // Look up
                Vec3f(0, 0, 1));   // Adjust up vector
    
    std::cout << "\n=== All renders complete! ===" << std::endl;
    
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}