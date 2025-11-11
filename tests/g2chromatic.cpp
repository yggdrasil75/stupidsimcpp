#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
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
    const int numSeeds = 8;
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
        
        // Convert to BGR format for AVI (OpenCV uses BGR)
        std::vector<uint8_t> bgrFrame(frameWidth * frameHeight * 3);
        #pragma omp parallel for
        for (int i = 0; i < frameWidth * frameHeight; ++i) {
            bgrFrame[i * 3] = rgbData[i * 3 + 2];
            bgrFrame[i * 3 + 1] = rgbData[i * 3 + 1];
            bgrFrame[i * 3 + 2] = rgbData[i * 3];
        }
        // for (int i = 0; i < frameWidth * frameHeight; ++i) {
        //     bgrFrame[i * 3] = static_cast<uint8_t>(rgbData[i * 3 + 2]);     // B
        //     bgrFrame[i * 3 + 1] = static_cast<uint8_t>(rgbData[i * 3 + 1]); // G
        //     bgrFrame[i * 3 + 2] = static_cast<uint8_t>(rgbData[i * 3]);     // R
        // }
        
        frames.push_back(bgrFrame);
    }
    
    // Save as AVI
    std::string filename = "output/chromatic_transformation.avi";
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
        return 1;
    }
    
    return 0;
}