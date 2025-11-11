#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "../util/grid/grid2.hpp"
#include "../util/output/aviwriter.hpp"

int main() {
    // Create a Grid2 instance
    Grid2 grid;
    
    // Grid dimensions
    const int width = 100;
    const int height = 100;
    const int totalFrames = 60; // 2 seconds at 30fps
    
    std::cout << "Creating chromatic transformation animation..." << std::endl;
    
    // Initialize with grayscale gradient
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float gradient = (x + y) / float(width + height - 2);
            Vec2 position(static_cast<float>(x), static_cast<float>(y));
            Vec4 color(gradient, gradient, gradient, 1.0f);
            grid.addObject(position, color, 1.0f);
        }
    }
    
    std::cout << "Initial grayscale grid created with " << width * height << " objects" << std::endl;
    
    // Random number generation for seed points
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> xDist(0, width - 1);
    std::uniform_int_distribution<> yDist(0, height - 1);
    std::uniform_real_distribution<> colorDist(0.2f, 0.8f);
    
    // Generate multiple seed points for more interesting patterns
    const int numSeeds = 1;
    std::vector<Vec2> seedPoints;
    std::vector<Vec4> seedColors;
    
    for (int i = 0; i < numSeeds; ++i) {
        seedPoints.emplace_back(xDist(gen), yDist(gen));
        seedColors.emplace_back(colorDist(gen), colorDist(gen), colorDist(gen), colorDist(gen));
    }
    
    std::cout << "Generated " << numSeeds << " seed points for color propagation" << std::endl;
    
    // Create frames for AVI
    std::vector<std::vector<uint8_t>> frames;
    
    for (int frame = 0; frame < totalFrames; ++frame) {
        std::cout << "Processing frame " << frame + 1 << "/" << totalFrames << std::endl;
        
        // Apply color propagation based on frame progress
        float progress = static_cast<float>(frame) / (totalFrames - 1);
        
        // Update colors based on seed propagation
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Vec2 currentPos(x, y);
                size_t id = grid.getIndicesAt(currentPos)[0]; // Assuming one object per position
                Vec4 originalColor = grid.getColor(id);
                
                Vec4 newColor = originalColor;
                
                // For each seed point, calculate influence
                for (int s = 0; s < numSeeds; ++s) {
                    float distance = currentPos.distance(seedPoints[s]);
                    float maxDistance = std::max(width, height) * 0.6f;
                    float influence = std::max(0.0f, 1.0f - (distance / maxDistance));
                    
                    // Apply influence based on relative position to seed
                    Vec2 direction = currentPos - seedPoints[s];
                    float angle = std::atan2(direction.y, direction.x);
                    
                    // Different color channels respond to different directions
                    if (std::abs(angle) < M_PI / 4.0f) { // Right - affect alpha
                        newColor.a = std::fmod(newColor.a + seedColors[s].a * influence * progress, 1.0f);
                    } else if (std::abs(angle) > 3.0f * M_PI / 4.0f) { // Left - affect blue
                        newColor.b = std::fmod(newColor.b + seedColors[s].b * influence * progress, 1.0f);
                    } else if (angle > 0) { // Below - affect green
                        newColor.g = std::fmod(newColor.g + seedColors[s].g * influence * progress, 1.0f);
                    } else { // Above - affect red
                        newColor.r = std::fmod(newColor.r + seedColors[s].r * influence * progress, 1.0f);
                    }
                }
                
                // Clamp colors to valid range
                newColor = newColor.clampColor();
                grid.setColor(id, newColor);
            }
        }
        
        // Get current frame as RGB data
        int frameWidth, frameHeight;
        std::vector<int> rgbData;
        grid.getGridAsRGB(frameWidth, frameHeight, rgbData);
        
        // Debug output to check frame dimensions
        std::cout << "Frame " << frame << ": " << frameWidth << "x" << frameHeight 
                  << ", RGB data size: " << rgbData.size() << std::endl;
        
        // Convert to BGR format for AVI and ensure proper 8-bit values
        std::vector<uint8_t> bgrFrame(frameWidth * frameHeight * 3);
        #pragma omp parallel for
        for (int i = 0; i < frameWidth * frameHeight; ++i) {
            // Ensure values are in 0-255 range and convert RGB to BGR
            int r = std::clamp(rgbData[i * 3], 0, 255);
            int g = std::clamp(rgbData[i * 3 + 1], 0, 255);
            int b = std::clamp(rgbData[i * 3 + 2], 0, 255);
            
            bgrFrame[i * 3] = static_cast<uint8_t>(b);     // B
            bgrFrame[i * 3 + 1] = static_cast<uint8_t>(g); // G
            bgrFrame[i * 3 + 2] = static_cast<uint8_t>(r); // R
        }
        
        // Verify frame size matches expected
        if (bgrFrame.size() != width * height * 3) {
            std::cerr << "ERROR: Frame size mismatch! Expected: " << width * height * 3 
                      << ", Got: " << bgrFrame.size() << std::endl;
            return 1;
        }
        
        frames.push_back(bgrFrame);
    }
    
    // Save as AVI
    std::string filename = "output/chromatic_transformation.avi";
    std::cout << "Attempting to save AVI file: " << filename << std::endl;
    std::cout << "Frames to save: " << frames.size() << std::endl;
    std::cout << "Frame dimensions: " << width << "x" << height << std::endl;
    
    bool success = AVIWriter::saveAVI(filename, frames, width, height, 30.0f);
    
    if (success) {
        std::cout << "\nSuccessfully saved chromatic transformation animation to: " << filename << std::endl;
        std::cout << "Video details:" << std::endl;
        std::cout << "  - Dimensions: " << width << " x " << height << std::endl;
        std::cout << "  - Frames: " << totalFrames << " (2 seconds at 30fps)" << std::endl;
        std::cout << "  - Seed points: " << numSeeds << std::endl;
        
        // Print seed point information
        std::cout << "\nSeed points used:" << std::endl;
        for (int i = 0; i < numSeeds; ++i) {
            std::cout << "  Seed " << i + 1 << ": Position " << seedPoints[i] 
                      << ", Color " << seedColors[i].toColorString() << std::endl;
        }
    } else {
        std::cerr << "Failed to save AVI file!" << std::endl;
        
        // Additional debugging information
        std::cerr << "Debug info:" << std::endl;
        std::cerr << "  - Frames count: " << frames.size() << std::endl;
        if (!frames.empty()) {
            std::cerr << "  - First frame size: " << frames[0].size() << std::endl;
            std::cerr << "  - Expected frame size: " << width * height * 3 << std::endl;
        }
        std::cerr << "  - Width: " << width << ", Height: " << height << std::endl;
        
        return 1;
    }
    
    return 0;
}