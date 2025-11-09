#include "sim2.hpp"
#include "../util/bmpwriter.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>

// Function to convert simulation grid to pixel data for BMP
std::vector<std::vector<Vec3>> gridToPixels(const Sim2& sim, int width, int height) {
    std::vector<std::vector<Vec3>> pixels(height, std::vector<Vec3>(width, Vec3(0, 0, 0))); // Black background
    
    // Get all pixel data from the simulation grid
    const auto& positions = sim.getPositions();
    const auto& colors = sim.getColors();
    
    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec2& pos = positions[i];
        const Vec4& color = colors[i];
        
        int x = static_cast<int>(pos.x);
        int y = static_cast<int>(pos.y);
        
        // Only draw if within bounds
        if (x >= 0 && x < width && y >= 0 && y < height) {
            // Convert Vec4 (RGBA) to Vec3 (RGB)
            pixels[y][x] = Vec3(color.x, color.y, color.z);
        }
    }
    
    return pixels;
}

int main() {
    const int FRAME_COUNT = 60;
    const float TOTAL_TIME = 1.0f; // 1 second
    const float TIME_STEP = TOTAL_TIME / FRAME_COUNT;
    
    // Create output directory
    std::filesystem::create_directories("output");
    
    // Create a simulation with 50x30 grid and gravity
    Sim2 sim(50, 30, Vec2(0, -9.8f));
    
    // Create some objects
    auto ball1 = sim.createBall(Vec2(10, 25), 3.0f, Vec4(1.0f, 0.5f, 0.0f, 1.0f), 1.0f); // Orange ball
    auto ball2 = sim.createBall(Vec2(30, 20), 2.0f, Vec4(0.5f, 0.8f, 1.0f, 1.0f), 0.5f); // Blue ball
    
    // Create ground
    auto ground = sim.createGround(Vec2(0, 0), 50, Vec4(0.0f, 1.0f, 0.0f, 1.0f)); // Green ground
    
    // Create walls
    auto leftWall = sim.createWall(Vec2(0, 1), 29, Vec4(0.3f, 0.3f, 0.7f, 1.0f)); // Blue walls
    auto rightWall = sim.createWall(Vec2(49, 1), 29, Vec4(0.3f, 0.3f, 0.7f, 1.0f));
    
    // Set world bounds
    sim.setWorldBounds(Vec2(0, 0), Vec2(49, 29));
    
    // Simulation parameters
    sim.setElasticity(0.8f); // Bouncy
    sim.setFriction(0.05f);  // Low friction
    
    std::cout << "Rendering " << FRAME_COUNT << " frames over " << TOTAL_TIME << " seconds..." << std::endl;
    
    // Run simulation and save frames
    for (int frame = 0; frame < FRAME_COUNT; ++frame) {
        // Update simulation
        sim.update(TIME_STEP);
        
        // Convert simulation state to pixel data
        auto pixels = gridToPixels(sim, 50, 30);
        
        // Create filename with zero-padded frame number
        std::ostringstream filename;
        filename << "output/bounce" << std::setw(3) << std::setfill('0') << frame << ".bmp";
        
        // Save as BMP
        if (BMPWriter::saveBMP(filename.str(), pixels)) {
            std::cout << "Saved frame " << frame << " to " << filename.str() << std::endl;
        } else {
            std::cerr << "Failed to save frame " << frame << std::endl;
        }
    }
    
    std::cout << "Animation complete! " << FRAME_COUNT << " frames saved to output/ directory." << std::endl;
    
    return 0;
}