#ifndef SPATIAL_CELL_16X16_HPP
#define SPATIAL_CELL_16X16_HPP

#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec4.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>

class SpatialCell16x16 {
private:
    static constexpr int CELL_SIZE = 16;
    static constexpr int TOTAL_CELLS = CELL_SIZE * CELL_SIZE;
    
    // Store objects in the cell
    std::unordered_map<size_t, Vec2> positions;
    std::unordered_map<size_t, Vec4> colors;
    std::unordered_map<size_t, float> sizes;
    
    // Bounds of this cell in world coordinates
    Vec2 worldMin, worldMax;
    float worldCellSize; // Size of each pixel in world coordinates
    
public:
    SpatialCell16x16(const Vec2& minCorner, const Vec2& maxCorner) 
        : worldMin(minCorner), worldMax(maxCorner) {
        
        // Calculate world size per cell pixel
        worldCellSize = std::max(
            (worldMax.x - worldMin.x) / CELL_SIZE,
            (worldMax.y - worldMin.y) / CELL_SIZE
        );
    }
    
    // Convert world position to cell coordinates [0,15]
    std::pair<int, int> worldToCell(const Vec2& worldPos) const {
        float localX = (worldPos.x - worldMin.x) / (worldMax.x - worldMin.x);
        float localY = (worldPos.y - worldMin.y) / (worldMax.y - worldMin.y);
        
        int cellX = static_cast<int>(localX * CELL_SIZE);
        int cellY = static_cast<int>(localY * CELL_SIZE);
        
        // Clamp to valid range
        cellX = std::clamp(cellX, 0, CELL_SIZE - 1);
        cellY = std::clamp(cellY, 0, CELL_SIZE - 1);
        
        return {cellX, cellY};
    }
    
    // Convert cell coordinates to linear index
    int cellToIndex(int x, int y) const {
        return y * CELL_SIZE + x;
    }
    
    // Convert linear index to cell coordinates
    std::pair<int, int> indexToCell(int index) const {
        return {index % CELL_SIZE, index / CELL_SIZE};
    }
    
    // Convert cell coordinates to world position (center of cell)
    Vec2 cellToWorld(int x, int y) const {
        float worldX = worldMin.x + (x + 0.5f) * worldCellSize;
        float worldY = worldMin.y + (y + 0.5f) * worldCellSize;
        return Vec2(worldX, worldY);
    }
    
    // Add object to the spatial cell
    bool addObject(size_t id, const Vec2& position, const Vec4& color, float size = 1.0f) {
        if (!contains(position)) {
            return false;
        }
        
        positions[id] = position;
        colors[id] = color;
        sizes[id] = size;
        
        return true;
    }
    
    // Check if world position is within this cell's bounds
    bool contains(const Vec2& worldPos) const {
        return worldPos.x >= worldMin.x && worldPos.x <= worldMax.x &&
               worldPos.y >= worldMin.y && worldPos.y <= worldMax.y;
    }
    
    // Update object position
    void updateObject(size_t id, const Vec2& oldPos, const Vec2& newPos) {
        if (!hasObject(id)) return;
        
        positions[id] = newPos;
    }
    
    // Remove object
    void removeObject(size_t id) {
        if (!hasObject(id)) return;
        
        positions.erase(id);
        colors.erase(id);
        sizes.erase(id);
    }
    
    // Check if object exists
    bool hasObject(size_t id) const {
        return positions.find(id) != positions.end();
    }
    
    // Get object data
    Vec2 getPosition(size_t id) const {
        auto it = positions.find(id);
        return it != positions.end() ? it->second : Vec2();
    }
    
    Vec4 getColor(size_t id) const {
        auto it = colors.find(id);
        return it != colors.end() ? it->second : Vec4();
    }
    
    float getSize(size_t id) const {
        auto it = sizes.find(id);
        return it != sizes.end() ? it->second : 1.0f;
    }
    
    // Set object data
    void setPosition(size_t id, const Vec2& position) {
        if (hasObject(id)) {
            positions[id] = position;
        }
    }
    
    void setColor(size_t id, const Vec4& color) {
        colors[id] = color;
    }
    
    void setSize(size_t id, float size) {
        if (hasObject(id)) {
            sizes[id] = size;
        }
    }
    
    // Spatial queries
    std::vector<size_t> getObjectsAt(const Vec2& position) const {
        std::vector<size_t> result;
        
        // Check all objects since we don't have spatial indexing
        for (const auto& pair : positions) {
            size_t id = pair.first;
            const Vec2& objPos = pair.second;
            float size = sizes.at(id);
            
            // Check if position is within object bounds
            if (position.x >= objPos.x - size * 0.5f && position.x <= objPos.x + size * 0.5f &&
                position.y >= objPos.y - size * 0.5f && position.y <= objPos.y + size * 0.5f) {
                result.push_back(id);
            }
        }
        
        return result;
    }
    
    std::vector<size_t> getObjectsInRadius(const Vec2& center, float radius) const {
        std::vector<size_t> result;
        float radius_sq = radius * radius;
        
        // Check all objects since we don't have spatial indexing
        for (const auto& pair : positions) {
            size_t id = pair.first;
            const Vec2& pos = pair.second;
            
            float dx = pos.x - center.x;
            float dy = pos.y - center.y;
            if (dx * dx + dy * dy <= radius_sq) {
                result.push_back(id);
            }
        }
        
        return result;
    }
    
    std::vector<size_t> getObjectsInRegion(const Vec2& minCorner, const Vec2& maxCorner) const {
        std::vector<size_t> result;
        
        // Check all objects since we don't have spatial indexing
        for (const auto& pair : positions) {
            size_t id = pair.first;
            const Vec2& pos = pair.second;
            
            if (pos.x >= minCorner.x && pos.x <= maxCorner.x &&
                pos.y >= minCorner.y && pos.y <= maxCorner.y) {
                result.push_back(id);
            }
        }
        
        return result;
    }
    
    // Get all object IDs
    std::vector<size_t> getAllObjectIds() const {
        std::vector<size_t> ids;
        ids.reserve(positions.size());
        for (const auto& pair : positions) {
            ids.push_back(pair.first);
        }
        return ids;
    }
    
    // Get cell statistics
    size_t getObjectCount() const { return positions.size(); }
    size_t getNonEmptyCellCount() const {
        // Since we removed cellBuckets, return 1 if there are objects, 0 otherwise
        return positions.empty() ? 0 : 1;
    }
    
    // Get bounds
    Vec2 getMinCorner() const { return worldMin; }
    Vec2 getMaxCorner() const { return worldMax; }
    
    // Clear all objects
    void clear() {
        positions.clear();
        colors.clear();
        sizes.clear();
    }

private:
    // Spatial indexing is no longer used
    void updateSpatialIndex(size_t id, const Vec2& oldPos, const Vec2& newPos) {
        // Empty implementation since we removed spatial indexing
    }
};

#endif