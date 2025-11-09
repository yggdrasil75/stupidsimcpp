#ifndef GRID2_HPP
#define GRID2_HPP

#include "../vec2.hpp"
#include "../vec4.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>

class Grid2 {
public:
    int width, height;
    
    Grid2() : width(0), height(0) {}
    Grid2(int size) : width(size), height(size) {
        positions.reserve(size);
        colors.reserve(size);
        sizes.reserve(size);
    }
    Grid2(int width, int height) : width(width), height(height) {
        positions.reserve(width * height);
        colors.reserve(width * height);
        sizes.reserve(width * height);
    }
    
    // Add a pixel at specific position
    int addPixel(const Vec2& position, const Vec4& color, float size = 1.0f) {
        int index = positions.size();
        positions.push_back(position);
        colors.push_back(color);
        sizes.push_back(size);
        positionToIndex[position] = index;
        return index;
    }
    
    // Add a pixel with integer coordinates
    int addPixel(int x, int y, const Vec4& color, float size = 1.0f) {
        return addPixel(Vec2(static_cast<float>(x), static_cast<float>(y)), color, size);
    }
    
    // Check if position is occupied
    bool isOccupied(const Vec2& position) const {
        return positionToIndex.find(position) != positionToIndex.end();
    }
    
    bool isOccupied(int x, int y) const {
        return isOccupied(Vec2(static_cast<float>(x), static_cast<float>(y)));
    }
    
    // Get pixel index at position, returns -1 if not found
    int getPixelIndex(const Vec2& position) const {
        auto it = positionToIndex.find(position);
        return (it != positionToIndex.end()) ? it->second : -1;
    }
    
    int getPixelIndex(int x, int y) const {
        return getPixelIndex(Vec2(static_cast<float>(x), static_cast<float>(y)));
    }
    
    // Remove pixel at position
    bool removePixel(const Vec2& position) {
        int index = getPixelIndex(position);
        if (index == -1) return false;
        
        // Swap with last element and update map
        if (index != positions.size() - 1) {
            positions[index] = positions.back();
            colors[index] = colors.back();
            sizes[index] = sizes.back();
            
            // Update mapping for the moved element
            positionToIndex[positions[index]] = index;
        }
        
        // Remove last element
        positions.pop_back();
        colors.pop_back();
        sizes.pop_back();
        positionToIndex.erase(position);
        
        return true;
    }
    
    bool removePixel(int x, int y) {
        return removePixel(Vec2(static_cast<float>(x), static_cast<float>(y)));
    }
    
    // Clear all pixels
    void clear() {
        positions.clear();
        colors.clear();
        sizes.clear();
        positionToIndex.clear();
    }
    
    // Get pixel count
    size_t getPixelCount() const {
        return positions.size();
    }
    
    // Check if grid is empty
    bool isEmpty() const {
        return positions.empty();
    }
    
    // Get bounding box of occupied pixels
    void getBoundingBox(Vec2& minCorner, Vec2& maxCorner) const {
        if (positions.empty()) {
            minCorner = Vec2(0, 0);
            maxCorner = Vec2(0, 0);
            return;
        }
        
        minCorner = positions[0];
        maxCorner = positions[0];
        
        for (const auto& pos : positions) {
            minCorner = minCorner.min(pos);
            maxCorner = maxCorner.max(pos);
        }
    }
    
    // Fill a rectangular region
    void fillRectangle(const Vec2& start, const Vec2& end, const Vec4& color, float size = 1.0f) {
        int startX = static_cast<int>(std::min(start.x, end.x));
        int endX = static_cast<int>(std::max(start.x, end.x));
        int startY = static_cast<int>(std::min(start.y, end.y));
        int endY = static_cast<int>(std::max(start.y, end.y));
        
        for (int y = startY; y <= endY; ++y) {
            for (int x = startX; x <= endX; ++x) {
                if (!isOccupied(x, y)) {
                    addPixel(x, y, color, size);
                }
            }
        }
    }
    
    // Create a circle pattern
    void fillCircle(const Vec2& center, float radius, const Vec4& color, float size = 1.0f) {
        int centerX = static_cast<int>(center.x);
        int centerY = static_cast<int>(center.y);
        int radiusInt = static_cast<int>(radius);
        
        for (int y = centerY - radiusInt; y <= centerY + radiusInt; ++y) {
            for (int x = centerX - radiusInt; x <= centerX + radiusInt; ++x) {
                float dx = x - center.x;
                float dy = y - center.y;
                if (dx * dx + dy * dy <= radius * radius) {
                    if (!isOccupied(x, y)) {
                        addPixel(x, y, color, size);
                    }
                }
            }
        }
    }
    
    // Create a line between two points
    void drawLine(const Vec2& start, const Vec2& end, const Vec4& color, float size = 1.0f) {
        // Bresenham's line algorithm
        int x0 = static_cast<int>(start.x);
        int y0 = static_cast<int>(start.y);
        int x1 = static_cast<int>(end.x);
        int y1 = static_cast<int>(end.y);
        
        int dx = std::abs(x1 - x0);
        int dy = std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;
        
        while (true) {
            if (!isOccupied(x0, y0)) {
                addPixel(x0, y0, color, size);
            }
            
            if (x0 == x1 && y0 == y1) break;
            
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y0 += sy;
            }
        }
    }
    
    // Get neighbors of a pixel (4-connected)
    std::vector<int> getNeighbors4(const Vec2& position) const {
        std::vector<int> neighbors;
        Vec2 offsets[] = {Vec2(1, 0), Vec2(-1, 0), Vec2(0, 1), Vec2(0, -1)};
        
        for (const auto& offset : offsets) {
            Vec2 neighborPos = position + offset;
            int index = getPixelIndex(neighborPos);
            if (index != -1) {
                neighbors.push_back(index);
            }
        }
        
        return neighbors;
    }
    
    // Get neighbors of a pixel (8-connected)
    std::vector<int> getNeighbors8(const Vec2& position) const {
        std::vector<int> neighbors;
        
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                
                Vec2 neighborPos = position + Vec2(dx, dy);
                int index = getPixelIndex(neighborPos);
                if (index != -1) {
                    neighbors.push_back(index);
                }
            }
        }
        
        return neighbors;
    }
    
    // Find connected components
    std::vector<std::vector<int>> findConnectedComponents() const {
        std::vector<std::vector<int>> components;
        std::unordered_map<Vec2, bool, std::hash<Vec2>> visited;
        
        for (size_t i = 0; i < positions.size(); ++i) {
            const Vec2& pos = positions[i];
            if (visited.find(pos) == visited.end()) {
                std::vector<int> component;
                floodFill(pos, visited, component);
                components.push_back(component);
            }
        }
        
        return components;
    }
    
    // Getters
    const std::vector<Vec2>& getPositions() const { return positions; }
    const std::vector<Vec4>& getColors() const { return colors; }
    const std::vector<float>& getSizes() const { return sizes; }
    
    Vec2 getPosition(int index) const { return positions[index]; }
    Vec4 getColor(int index) const { return colors[index]; }
    float getSize(int index) const { return sizes[index]; }
    
    void setColor(int index, const Vec4& color) { colors[index] = color; }
    void setSize(int index, float size) { sizes[index] = size; }
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    std::vector<Vec2> positions;
    std::vector<Vec4> colors;
    std::vector<float> sizes;
    std::unordered_map<Vec2, int, std::hash<Vec2>> positionToIndex;
    
    void floodFill(const Vec2& start, std::unordered_map<Vec2, bool, std::hash<Vec2>>& visited, 
                   std::vector<int>& component) const {
        std::vector<Vec2> stack;
        stack.push_back(start);
        
        while (!stack.empty()) {
            Vec2 current = stack.back();
            stack.pop_back();
            
            if (visited.find(current) != visited.end()) continue;
            
            visited[current] = true;
            int index = getPixelIndex(current);
            if (index != -1) {
                component.push_back(index);
                
                // Add 4-connected neighbors
                Vec2 neighbors[] = {current + Vec2(1, 0), current + Vec2(-1, 0), 
                                   current + Vec2(0, 1), current + Vec2(0, -1)};
                
                for (const auto& neighbor : neighbors) {
                    if (isOccupied(neighbor) && visited.find(neighbor) == visited.end()) {
                        stack.push_back(neighbor);
                    }
                }
            }
        }
    }
};

#endif