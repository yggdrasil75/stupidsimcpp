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
        
        // Flip Y coordinate to make (0,0) bottom-left instead of top-left
        int flippedY = height - 1 - y;
        
        // Only draw if within bounds
        if (x >= 0 && x < width && flippedY >= 0 && flippedY < height) {
            // Convert Vec4 (RGBA) to Vec3 (RGB)
            pixels[flippedY][x] = Vec3(color.x, color.y, color.z);
        }
    }
    
    return pixels;
}

int main() {
    const int FRAME_COUNT = 120; // Increased frame count for longer animation
    const float TOTAL_TIME = 4.0f; // Increased to 4 seconds for more movement
    const float TIME_STEP = TOTAL_TIME / FRAME_COUNT;
    
    // Create output directory
    std::filesystem::create_directories("output");
    
    // Create a simulation with 512x512 grid and SCALED gravity
    // Scale gravity by ~17x (512/30 ≈ 17) to account for larger grid
    Sim2 sim(512, 512, Vec2(0, -9.8f * 17.0f));
    
    // Create some objects - adjust positions and sizes for larger grid
    auto ball1 = sim.createBall(Vec2(100, 400), 40.0f, Vec4(1.0f, 0.5f, 0.0f, 1.0f), 1.0f); // Orange ball
    auto ball2 = sim.createBall(Vec2(300, 450), 35.0f, Vec4(0.5f, 0.8f, 1.0f, 1.0f), 0.8f); // Blue ball
    
    // Give initial velocities to make movement more visible
    ball1->setVelocity(Vec2(50.0f, 0.0f)); // Rightward velocity
    ball2->setVelocity(Vec2(-30.0f, 0.0f)); // Leftward velocity
    
    // Create ground - higher up so we can see falling
    auto ground = sim.createGround(Vec2(0, 0), 512, Vec4(0.0f, 1.0f, 0.0f, 1.0f)); // Green ground

    // auto realground = sim.createGround(Vec2(0, 512), 512, Vec4(0.0f, 1.0f, 0.0f, 1.0f)); // Green ground
    
    // Create walls
    auto leftWall = sim.createWall(Vec2(0, 51), 461, Vec4(0.3f, 0.3f, 0.7f, 1.0f)); // Blue walls (511-50=461)
    auto rightWall = sim.createWall(Vec2(511, 51), 461, Vec4(0.3f, 0.3f, 0.7f, 1.0f));

    
    // Set world bounds
    sim.setWorldBounds(Vec2(0, 0), Vec2(511, 511));
    
    // Simulation parameters - adjust for more visible movement
    sim.setElasticity(0.7f); // Slightly less bouncy
    sim.setFriction(0.02f);  // Very low friction
    
    std::cout << "Rendering " << FRAME_COUNT << " frames over " << TOTAL_TIME << " seconds..." << std::endl;
    std::cout << "Gravity: " << sim.getGlobalGravity().y << " px/s²" << std::endl;
    
    // Run simulation and save frames
    for (int frame = 0; frame < FRAME_COUNT; ++frame) {
        // Update simulation
        sim.update(TIME_STEP);
        
        // Convert simulation state to pixel data
        auto pixels = gridToPixels(sim, 512, 512);
        
        // Create filename with zero-padded frame number
        std::ostringstream filename;
        filename << "output/bounce" << std::setw(3) << std::setfill('0') << frame << ".bmp";
        
        // Save as BMP
        if (BMPWriter::saveBMP(filename.str(), pixels)) {
            if (frame % 10 == 0) { // Only print every 10 frames to reduce spam
                std::cout << "Saved frame " << frame << " to " << filename.str() << std::endl;
            }
        } else {
            std::cerr << "Failed to save frame " << frame << std::endl;
        }
    }
    
    std::cout << "Animation complete! " << FRAME_COUNT << " frames saved to output/ directory." << std::endl;
    
    return 0;
}