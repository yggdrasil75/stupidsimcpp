// main.cpp - Noise Grid Generator and Renderer
#include <iostream>
#include <memory>
#include <cmath>
#include "../util/grid/grid3.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/noise/pnoise2.hpp"

// Constants
const size_t GRID_SIZE = 1024;
const size_t RENDER_WIDTH = 512;
const size_t RENDER_HEIGHT = 512;

// Function to generate grayscale noise grid
void generateNoiseGrid(VoxelGrid& grid, PNoise2& noise) {
    std::cout << "Generating 1024x1024x1024 noise grid..." << std::endl;
    
    // Generate Perlin noise in the grid
    for (size_t z = 0; z < GRID_SIZE; ++z) {
        if (z % 64 == 0) {
            std::cout << "Processing layer " << z << " of " << GRID_SIZE << std::endl;
        }
        
        for (size_t y = 0; y < GRID_SIZE; ++y) {
            for (size_t x = 0; x < GRID_SIZE; ++x) {
                // Create 3D noise coordinates (scaled for better frequency)
                float scale = 0.05f; // Controls noise frequency
                float noiseVal = noise.permute(Vec3f(x * scale, y * scale, z * scale));
                
                // Convert from [-1, 1] to [0, 1] range
                float normalizedNoise = (noiseVal + 1.0f) * 0.5f;
                
                // Apply threshold to make some voxels "active"
                // Higher threshold = sparser voxels
                float threshold = 0.3f;
                float active = (normalizedNoise > threshold) ? normalizedNoise : 0.0f;
                
                // Create grayscale color based on noise value
                uint8_t grayValue = static_cast<uint8_t>(normalizedNoise * 255);
                Vec3ui8 color(grayValue, grayValue, grayValue);
                
                grid.set(x, y, z, active, color);
            }
        }
    }
    
    std::cout << "Noise grid generation complete!" << std::endl;
}

// Function to render grid from a specific viewpoint
bool renderView(const std::string& filename, VoxelGrid& grid, const Vec3f& position, 
                const Vec3f& direction, const Vec3f& up = Vec3f(0, 1, 0)) {
    
    Camera cam(position, direction, up);
    
    std::vector<uint8_t> renderBuffer;
    size_t width = RENDER_WIDTH;
    size_t height = RENDER_HEIGHT;
    
    // Render the view
    grid.renderOut(renderBuffer, width, height, cam);
    
    // Save to BMP
    bool success = BMPWriter::saveBMP(filename, renderBuffer, width, height);
    
    if (success) {
        std::cout << "Saved: " << filename << std::endl;
    } else {
        std::cout << "Failed to save: " << filename << std::endl;
    }
    
    return success;
}

int main() {
    try {
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
        
        // Render from 4 cardinal directions (looking at center)
        std::cout << "\nRendering cardinal views..." << std::endl;
        
        // North view (looking from positive Z towards center)
        renderView("north_view.bmp", grid, 
                   gridCenter + Vec3f(0, 0, cameraDistance),
                   Vec3f(0, 0, -1));  // Look towards negative Z
        
        // South view (looking from negative Z towards center)
        renderView("south_view.bmp", grid,
                   gridCenter + Vec3f(0, 0, -cameraDistance),
                   Vec3f(0, 0, 1));   // Look towards positive Z
        
        // East view (looking from positive X towards center)
        renderView("east_view.bmp", grid,
                   gridCenter + Vec3f(cameraDistance, 0, 0),
                   Vec3f(-1, 0, 0));  // Look towards negative X
        
        // West view (looking from negative X towards center)
        renderView("west_view.bmp", grid,
                   gridCenter + Vec3f(-cameraDistance, 0, 0),
                   Vec3f(1, 0, 0));   // Look towards positive X
        
        // Zenith view (looking from above)
        std::cout << "\nRendering zenith and nadir views..." << std::endl;
        renderView("zenith_view.bmp", grid,
                   gridCenter + Vec3f(0, cameraDistance, 0),
                   Vec3f(0, -1, 0),   // Look down
                   Vec3f(0, 0, -1));  // Adjust up vector for proper orientation
        
        // Nadir view (looking from below)
        renderView("nadir_view.bmp", grid,
                   gridCenter + Vec3f(0, -cameraDistance, 0),
                   Vec3f(0, 1, 0),    // Look up
                   Vec3f(0, 0, 1));   // Adjust up vector
        
        // Optional: Create a front view (alternative to north)
        renderView("front_view.bmp", grid,
                   gridCenter + Vec3f(0, 0, cameraDistance),
                   Vec3f(0, 0, -1),
                   Vec3f(0, 1, 0));
        
        std::cout << "\n=== All renders complete! ===" << std::endl;
        std::cout << "Generated BMP files:" << std::endl;
        std::cout << "1. north_view.bmp - Looking from positive Z" << std::endl;
        std::cout << "2. south_view.bmp - Looking from negative Z" << std::endl;
        std::cout << "3. east_view.bmp  - Looking from positive X" << std::endl;
        std::cout << "4. west_view.bmp  - Looking from negative X" << std::endl;
        std::cout << "5. zenith_view.bmp - Looking from above" << std::endl;
        std::cout << "6. nadir_view.bmp  - Looking from below" << std::endl;
        std::cout << "7. front_view.bmp  - Alternative front view" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}