#ifndef FIXED_SPATIAL_GRID_2_HPP
#define FIXED_SPATIAL_GRID_2_HPP

#include "../vectorlogic/vec2.hpp"
#include <vector>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <cmath>

class Grid2Fast {
private:
    struct Cell {
        std::vector<size_t> objectIds;
        
        void add(size_t id) {
            objectIds.push_back(id);
        }
        
        void remove(size_t id) {
            auto it = std::find(objectIds.begin(), objectIds.end(), id);
            if (it != objectIds.end()) {
                objectIds.erase(it);
            }
        }
        
        bool contains(size_t id) const {
            return std::find(objectIds.begin(), objectIds.end(), id) != objectIds.end();
        }
        
        void clear() {
            objectIds.clear();
        }
        
        size_t size() const {
            return objectIds.size();
        }
        
        bool empty() const {
            return objectIds.empty();
        }
    };

    // Fixed grid dimensions
    int gridWidth, gridHeight;
    float cellSize;
    Vec2 worldMin, worldMax;
    
    // 2D grid storage
    std::vector<Cell> grid;
    std::unordered_map<size_t, std::pair<int, int>> objectToCell;
    
    // Helper methods
    inline int toIndex(int x, int y) const {
        return y * gridWidth + x;
    }
    
    inline bool isValidCell(int x, int y) const {
        return x >= 0 && x < gridWidth && y >= 0 && y < gridHeight;
    }

public:
    Grid2Fast(const Vec2& minCorner, const Vec2& maxCorner, float cellSize)
        : cellSize(cellSize), worldMin(minCorner), worldMax(maxCorner) {
        
        // Calculate grid dimensions
        float worldWidth = maxCorner.x - minCorner.x;
        float worldHeight = maxCorner.y - minCorner.y;
        
        gridWidth = static_cast<int>(std::ceil(worldWidth / cellSize));
        gridHeight = static_cast<int>(std::ceil(worldHeight / cellSize));
        
        // Initialize grid with empty cells
        grid.resize(gridWidth * gridHeight);
    }
    
    Grid2Fast(float minX, float minY, float maxX, float maxY, float cellSize)
        : Grid2Fast(Vec2(minX, minY), Vec2(maxX, maxY), cellSize) {}
    
    // Convert world position to grid coordinates
    std::pair<int, int> worldToGrid(const Vec2& pos) const {
        int x = static_cast<int>((pos.x - worldMin.x) / cellSize);
        int y = static_cast<int>((pos.y - worldMin.y) / cellSize);
        
        // Clamp to grid boundaries
        x = std::clamp(x, 0, gridWidth - 1);
        y = std::clamp(y, 0, gridHeight - 1);
        
        return {x, y};
    }
    
    // Convert grid coordinates to world position (center of cell)
    Vec2 gridToWorld(int gridX, int gridY) const {
        float x = worldMin.x + (gridX + 0.5f) * cellSize;
        float y = worldMin.y + (gridY + 0.5f) * cellSize;
        return Vec2(x, y);
    }
    
    // Add object to spatial grid
    bool addObject(size_t id, const Vec2& position) {
        auto [gridX, gridY] = worldToGrid(position);
        
        if (!isValidCell(gridX, gridY)) {
            return false; // Object outside grid bounds
        }
        
        int index = toIndex(gridX, gridY);
        grid[index].add(id);
        objectToCell[id] = {gridX, gridY};
        return true;
    }
    
    // Remove object from spatial grid
    bool removeObject(size_t id) {
        auto it = objectToCell.find(id);
        if (it == objectToCell.end()) {
            return false;
        }
        
        auto [gridX, gridY] = it->second;
        if (isValidCell(gridX, gridY)) {
            int index = toIndex(gridX, gridY);
            grid[index].remove(id);
        }
        
        objectToCell.erase(it);
        return true;
    }
    
    // Update object position
    bool updateObject(size_t id, const Vec2& oldPos, const Vec2& newPos) {
        auto oldCell = worldToGrid(oldPos);
        auto newCell = worldToGrid(newPos);
        
        if (oldCell == newCell) {
            // Same cell, no update needed
            objectToCell[id] = newCell;
            return true;
        }
        
        // Remove from old cell
        auto [oldX, oldY] = oldCell;
        if (isValidCell(oldX, oldY)) {
            int oldIndex = toIndex(oldX, oldY);
            grid[oldIndex].remove(id);
        }
        
        // Add to new cell
        auto [newX, newY] = newCell;
        if (!isValidCell(newX, newY)) {
            // Object moved outside grid, remove completely
            objectToCell.erase(id);
            return false;
        }
        
        int newIndex = toIndex(newX, newY);
        grid[newIndex].add(id);
        objectToCell[id] = newCell;
        return true;
    }
    
    // Get objects in radius (optimized using grid)
    std::vector<size_t> getObjectsInRadius(const Vec2& position, float radius) const {
        std::vector<size_t> result;
        
        if (radius <= 0.0f) {
            return getObjectsAt(position);
        }
        
        Vec2 minPos(position.x - radius, position.y - radius);
        Vec2 maxPos(position.x + radius, position.y + radius);
        
        auto minCell = worldToGrid(minPos);
        auto maxCell = worldToGrid(maxPos);
        
        float radiusSq = radius * radius;
        
        // Check only relevant cells
        for (int y = minCell.second; y <= maxCell.second; ++y) {
            for (int x = minCell.first; x <= maxCell.first; ++x) {
                if (!isValidCell(x, y)) continue;
                
                int index = toIndex(x, y);
                const Cell& cell = grid[index];
                
                for (size_t id : cell.objectIds) {
                    // We need external position data for distance check
                    // This assumes the caller will filter results based on actual positions
                    result.push_back(id);
                }
            }
        }
        
        return result;
    }
    
    // Get objects at exact position
    std::vector<size_t> getObjectsAt(const Vec2& position) const {
        auto [gridX, gridY] = worldToGrid(position);
        
        if (!isValidCell(gridX, gridY)) {
            return {};
        }
        
        int index = toIndex(gridX, gridY);
        return grid[index].objectIds; // Return copy
    }
    
    // Get objects in rectangular region
    std::vector<size_t> getObjectsInRegion(const Vec2& minCorner, const Vec2& maxCorner) const {
        std::vector<size_t> result;
        
        auto minCell = worldToGrid(minCorner);
        auto maxCell = worldToGrid(maxCorner);
        
        for (int y = minCell.second; y <= maxCell.second; ++y) {
            for (int x = minCell.first; x <= maxCell.first; ++x) {
                if (!isValidCell(x, y)) continue;
                
                int index = toIndex(x, y);
                const Cell& cell = grid[index];
                
                // Add all objects from these cells
                // Note: This may include objects outside the exact region due to cell granularity
                // Caller should filter based on actual positions if precise region is needed
                result.insert(result.end(), cell.objectIds.begin(), cell.objectIds.end());
            }
        }
        
        return result;
    }
    
    // Get all objects in the grid
    std::vector<size_t> getAllObjects() const {
        std::vector<size_t> result;
        
        for (const auto& pair : objectToCell) {
            result.push_back(pair.first);
        }
        
        return result;
    }
    
    // Get cell information
    const Cell& getCell(int x, int y) const {
        static Cell emptyCell;
        if (!isValidCell(x, y)) {
            return emptyCell;
        }
        return grid[toIndex(x, y)];
    }
    
    const Cell& getCellAtWorldPos(const Vec2& pos) const {
        auto [x, y] = worldToGrid(pos);
        return getCell(x, y);
    }
    
    // Statistics
    size_t getTotalObjectCount() const {
        return objectToCell.size();
    }
    
    size_t getNonEmptyCellCount() const {
        size_t count = 0;
        for (const auto& cell : grid) {
            if (!cell.empty()) {
                ++count;
            }
        }
        return count;
    }
    
    size_t getMaxObjectsPerCell() const {
        size_t maxCount = 0;
        for (const auto& cell : grid) {
            maxCount = std::max(maxCount, cell.size());
        }
        return maxCount;
    }
    
    float getAverageObjectsPerCell() const {
        if (grid.empty()) return 0.0f;
        return static_cast<float>(objectToCell.size()) / grid.size();
    }
    
    // Grid properties
    int getGridWidth() const { return gridWidth; }
    int getGridHeight() const { return gridHeight; }
    float getCellSize() const { return cellSize; }
    Vec2 getWorldMin() const { return worldMin; }
    Vec2 getWorldMax() const { return worldMax; }
    
    // Clear all objects
    void clear() {
        for (auto& cell : grid) {
            cell.clear();
        }
        objectToCell.clear();
    }
    
    // Check if object exists in grid
    bool contains(size_t id) const {
        return objectToCell.find(id) != objectToCell.end();
    }
    
    // Get cell coordinates for object
    std::pair<int, int> getObjectCell(size_t id) const {
        auto it = objectToCell.find(id);
        if (it != objectToCell.end()) {
            return it->second;
        }
        return {-1, -1};
    }
};

#endif