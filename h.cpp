#include "util/grid/gridtest.hpp"
#include <random>
#include <memory>
#include "util/output/bmpwriter.hpp"

// Function to create a randomized voxel grid
std::unique_ptr<VoxelGrid> createRandomizedGrid(int width, int height, int depth, float density = 0.2f) {
    auto grid = std::make_unique<VoxelGrid>(width, height, depth);
    
    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> colorDist(0.2f, 1.0f);
    
    // Fill grid with random active voxels
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int z = 0; z < depth; z++) {
                if (dist(gen) < density) {
                    // Random color
                    Vec3f color(colorDist(gen), colorDist(gen), colorDist(gen));
                    grid->set(x, y, z, true, color);
                }
            }
        }
    }
    
    return grid;
}

// Create a simple structure to hold camera position and orientation
struct CameraPose {
    Vec3f position;
    Vec3f target;
    Vec3f up;
};

int main() {
    // Create a randomized voxel grid (15x15x15 with 20% density)
    const int gridSize = 128;
    auto grid = createRandomizedGrid(gridSize, gridSize, gridSize, 0.2f);
    
    // Create a renderer with our custom grid
    VoxelRenderer renderer;
    renderer.getGrid() = *grid; // Replace the default test pattern
    
    // Define 6 camera positions (looking at the center from each axis direction)
    std::vector<CameraPose> cameraPoses = {
        // Front (looking along negative Z)
        {Vec3f(0, 0, -gridSize * 2), Vec3f(0, 0, 0), Vec3f(0, 1, 0)},
        
        // Back (looking along positive Z)
        {Vec3f(0, 0, gridSize * 2), Vec3f(0, 0, 0), Vec3f(0, 1, 0)},
        
        // Right (looking along positive X)
        {Vec3f(gridSize * 2, 0, 0), Vec3f(0, 0, 0), Vec3f(0, 1, 0)},
        
        // Left (looking along negative X)
        {Vec3f(-gridSize * 2, 0, 0), Vec3f(0, 0, 0), Vec3f(0, 1, 0)},
        
        // Top (looking along negative Y)
        {Vec3f(0, gridSize * 2, 0), Vec3f(0, 0, 0), Vec3f(0, 0, -1)},
        
        // Bottom (looking along positive Y)
        {Vec3f(0, -gridSize * 2, 0), Vec3f(0, 0, 0), Vec3f(0, 0, 1)}
    };
    
    // Render settings
    const int screenWidth = 400;
    const int screenHeight = 400;
    
    // Vector to store all frames
    std::vector<frame> frames;
    
    std::cout << "Rendering voxel grid from 6 directions..." << std::endl;
    
    // Render from each camera position
    for (size_t i = 0; i < cameraPoses.size(); i++) {
        auto& pose = cameraPoses[i];
        
        // Update camera position
        renderer.getCamera().position = pose.position;
        
        // Calculate forward direction (from position to target)
        Vec3f forward = (pose.target - pose.position).normalized();
        
        // Set camera orientation
        renderer.getCamera().forward = forward;
        renderer.getCamera().up = pose.up;
        
        std::cout << "Rendering view " << (i+1) << "/6 from position ("
                  << pose.position.x << ", " << pose.position.y << ", " << pose.position.z << ")" << std::endl;
        
        // Render and add to frames vector
        frames.push_back(renderer.renderToFrame(screenWidth, screenHeight));
    }
    
    // Print frame information
    std::cout << "\nRendering complete!" << std::endl;
    std::cout << "Number of frames created: " << frames.size() << std::endl;
    
    for (size_t i = 0; i < frames.size(); i++) {
        std::cout << "Frame " << (i+1) << ": " << frames[i] << std::endl;
    }

    // Optional: Save frames to files if your frame class supports it
    for (size_t i = 0; i < frames.size(); i++) {
        std::string filename = "output/voxel_view_" + std::to_string(i+1) + ".bmp";
        BMPWriter::saveBMP(filename, frames[i]);
    }
    
    return 0;
}