#ifndef GRID2_HPP
#define GRID2_HPP

#include "Vec2.hpp"
#include "Vec4.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

class Grid2 {
public:
    std::vector<Vec2> positions;
    std::vector<Vec4> colors;
    
    Grid2() = default;
    
    // Constructor with initial size
    Grid2(size_t size) {
        positions.resize(size);
        colors.resize(size);
    }
    
    // Add a point with position and color
    void addPoint(const Vec2& position, const Vec4& color) {
        positions.push_back(position);
        colors.push_back(color);
    }
    
    // Clear all points
    void clear() {
        positions.clear();
        colors.clear();
    }
    
    // Get number of points
    size_t size() const {
        return positions.size();
    }
    
    // Check if grid is empty
    bool empty() const {
        return positions.empty();
    }
    
    // Resize the grid
    void resize(size_t newSize) {
        positions.resize(newSize);
        colors.resize(newSize);
    }
    
    // Render to RGB image data
    std::vector<uint8_t> renderToRGB(int width, int height, const Vec4& backgroundColor = Vec4(0, 0, 0, 1)) const {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("Width and height must be positive");
        }
        
        std::vector<uint8_t> imageData(width * height * 3);
        
        // Initialize with background color
        uint8_t bgR, bgG, bgB;
        backgroundColor.toUint8(bgR, bgG, bgB);
        
        for (int i = 0; i < width * height * 3; i += 3) {
            imageData[i] = bgR;
            imageData[i + 1] = bgG;
            imageData[i + 2] = bgB;
        }
        
        // Find the bounding box of all points to map to pixel coordinates
        if (positions.empty()) {
            return imageData;
        }
        
        Vec2 minPos = positions[0];
        Vec2 maxPos = positions[0];
        
        for (const auto& pos : positions) {
            minPos = minPos.min(pos);
            maxPos = maxPos.max(pos);
        }
        
        // Add a small margin to avoid division by zero and edge issues
        Vec2 size = maxPos - minPos;
        if (size.x < 1e-10f) size.x = 1.0f;
        if (size.y < 1e-10f) size.y = 1.0f;
        
        float margin = 0.05f; // 5% margin
        minPos -= size * margin;
        maxPos += size * margin;
        size = maxPos - minPos;
        
        // Render each point
        for (size_t i = 0; i < positions.size(); i++) {
            const Vec2& pos = positions[i];
            const Vec4& color = colors[i];
            
            // Convert world coordinates to pixel coordinates
            float normalizedX = (pos.x - minPos.x) / size.x;
            float normalizedY = 1.0f - (pos.y - minPos.y) / size.y; // Flip Y for image coordinates
            
            int pixelX = static_cast<int>(normalizedX * width);
            int pixelY = static_cast<int>(normalizedY * height);
            
            // Clamp to image bounds
            pixelX = std::clamp(pixelX, 0, width - 1);
            pixelY = std::clamp(pixelY, 0, height - 1);
            
            // Convert color to RGB
            uint8_t r, g, b;
            color.toUint8(r, g, b);
            
            // Set pixel color
            int index = (pixelY * width + pixelX) * 3;
            imageData[index] = r;
            imageData[index + 1] = g;
            imageData[index + 2] = b;
        }
        
        return imageData;
    }
    
    // Render to RGBA image data (with alpha channel)
    std::vector<uint8_t> renderToRGBA(int width, int height, const Vec4& backgroundColor = Vec4(0, 0, 0, 1)) const {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("Width and height must be positive");
        }
        
        std::vector<uint8_t> imageData(width * height * 4);
        
        // Initialize with background color
        uint8_t bgR, bgG, bgB, bgA;
        backgroundColor.toUint8(bgR, bgG, bgB, bgA);
        
        for (int i = 0; i < width * height * 4; i += 4) {
            imageData[i] = bgR;
            imageData[i + 1] = bgG;
            imageData[i + 2] = bgB;
            imageData[i + 3] = bgA;
        }
        
        if (positions.empty()) {
            return imageData;
        }
        
        // Find the bounding box (same as RGB version)
        Vec2 minPos = positions[0];
        Vec2 maxPos = positions[0];
        
        for (const auto& pos : positions) {
            minPos = minPos.min(pos);
            maxPos = maxPos.max(pos);
        }
        
        Vec2 size = maxPos - minPos;
        if (size.x < 1e-10f) size.x = 1.0f;
        if (size.y < 1e-10f) size.y = 1.0f;
        
        float margin = 0.05f;
        minPos -= size * margin;
        maxPos += size * margin;
        size = maxPos - minPos;
        
        // Render each point
        for (size_t i = 0; i < positions.size(); i++) {
            const Vec2& pos = positions[i];
            const Vec4& color = colors[i];
            
            float normalizedX = (pos.x - minPos.x) / size.x;
            float normalizedY = 1.0f - (pos.y - minPos.y) / size.y;
            
            int pixelX = static_cast<int>(normalizedX * width);
            int pixelY = static_cast<int>(normalizedY * height);
            
            pixelX = std::clamp(pixelX, 0, width - 1);
            pixelY = std::clamp(pixelY, 0, height - 1);
            
            uint8_t r, g, b, a;
            color.toUint8(r, g, b, a);
            
            int index = (pixelY * width + pixelX) * 4;
            imageData[index] = r;
            imageData[index + 1] = g;
            imageData[index + 2] = b;
            imageData[index + 3] = a;
        }
        
        return imageData;
    }
    
    // Get the bounding box of all positions
    void getBoundingBox(Vec2& minPos, Vec2& maxPos) const {
        if (positions.empty()) {
            minPos = Vec2(0, 0);
            maxPos = Vec2(0, 0);
            return;
        }
        
        minPos = positions[0];
        maxPos = positions[0];
        
        for (const auto& pos : positions) {
            minPos = minPos.min(pos);
            maxPos = maxPos.max(pos);
        }
    }
    
    // Scale all positions to fit within a specified range
    void normalizePositions(const Vec2& targetMin = Vec2(-1, -1), const Vec2& targetMax = Vec2(1, 1)) {
        if (positions.empty()) return;
        
        Vec2 currentMin, currentMax;
        getBoundingBox(currentMin, currentMax);
        
        Vec2 currentSize = currentMax - currentMin;
        Vec2 targetSize = targetMax - targetMin;
        
        if (currentSize.x < 1e-10f) currentSize.x = 1.0f;
        if (currentSize.y < 1e-10f) currentSize.y = 1.0f;
        
        for (auto& pos : positions) {
            float normalizedX = (pos.x - currentMin.x) / currentSize.x;
            float normalizedY = (pos.y - currentMin.y) / currentSize.y;
            
            pos.x = targetMin.x + normalizedX * targetSize.x;
            pos.y = targetMin.y + normalizedY * targetSize.y;
        }
    }
};

#endif