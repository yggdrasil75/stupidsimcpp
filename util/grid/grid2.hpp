#ifndef GRID2_HPP
#define GRID2_HPP

#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include "spatc16.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <map>
#include <unordered_set>
#include <cmath>
#include <memory>

struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class Grid2 {
private:
    // Objects that don't fit in any SpatialCell16x16
    std::unordered_map<size_t, Vec2> loosePositions;
    std::unordered_map<size_t, Vec4> looseColors;
    std::unordered_map<size_t, float> looseSizes;
    
    // Spatial partitioning using SpatialCell16x16 cells
    std::unordered_map<std::pair<int, int>, std::unique_ptr<SpatialCell16x16>, PairHash> spatialCells;
    Vec2 gridMin, gridMax;
    bool useSpatialPartitioning;
    
    size_t next_id;
    
    // Convert world position to cell coordinates for SpatialCell16x16 placement
    std::pair<int, int> worldToCellCoord(const Vec2& worldPos) const {
        if (!useSpatialPartitioning) return {0, 0};
        
        float cellWorldSize = getCellWorldSize();
        int cellX = static_cast<int>((worldPos.x - gridMin.x) / cellWorldSize);
        int cellY = static_cast<int>((worldPos.y - gridMin.y) / cellWorldSize);
        return {cellX, cellY};
    }
    
    // Get the world size of each SpatialCell16x16 cell
    float getCellWorldSize() const {
        // SpatialCell16x16 is 16x16 internally, but we need world size
        return (gridMax.x - gridMin.x) / getGridCellsX();
    }
    
    // Calculate how many SpatialCell16x16 cells fit in X direction
    int getGridCellsX() const {
        if (!useSpatialPartitioning) return 1;
        float worldWidth = gridMax.x - gridMin.x;
        // Aim for cells that are roughly square in world coordinates
        float targetCellSize = std::max(16.0f, worldWidth / 16.0f);
        return std::max(1, static_cast<int>(worldWidth / targetCellSize));
    }
    
    // Calculate how many SpatialCell16x16 cells fit in Y direction  
    int getGridCellsY() const {
        if (!useSpatialPartitioning) return 1;
        float worldHeight = gridMax.y - gridMin.y;
        float targetCellSize = std::max(16.0f, worldHeight / 16.0f);
        return std::max(1, static_cast<int>(worldHeight / targetCellSize));
    }
    
    // Get cell bounds from coordinates
    Vec2 getCellMinCorner(int cellX, int cellY) const {
        float cellSizeX = (gridMax.x - gridMin.x) / getGridCellsX();
        float cellSizeY = (gridMax.y - gridMin.y) / getGridCellsY();
        return Vec2(
            gridMin.x + cellX * cellSizeX,
            gridMin.y + cellY * cellSizeY
        );
    }
    
    Vec2 getCellMaxCorner(int cellX, int cellY) const {
        float cellSizeX = (gridMax.x - gridMin.x) / getGridCellsX();
        float cellSizeY = (gridMax.y - gridMin.y) / getGridCellsY();
        return Vec2(
            gridMin.x + (cellX + 1) * cellSizeX,
            gridMin.y + (cellY + 1) * cellSizeY
        );
    }
    
    // Get or create spatial cell
    SpatialCell16x16* getOrCreateCell(int cellX, int cellY) {
        auto key = std::make_pair(cellX, cellY);
        auto it = spatialCells.find(key);
        if (it != spatialCells.end()) {
            return it->second.get();
        }
        
        // Create new SpatialCell16x16
        Vec2 minCorner = getCellMinCorner(cellX, cellY);
        Vec2 maxCorner = getCellMaxCorner(cellX, cellY);
        spatialCells[key] = std::make_unique<SpatialCell16x16>(minCorner, maxCorner);
        return spatialCells[key].get();
    }
    
    // Check if an object can fit in SpatialCell16x16
    bool canFitInSpatialCell(const Vec2& position, float size) const {
        if (!useSpatialPartitioning) return false;
        
        auto cellCoord = worldToCellCoord(position);
        Vec2 cellMin = getCellMinCorner(cellCoord.first, cellCoord.second);
        Vec2 cellMax = getCellMaxCorner(cellCoord.first, cellCoord.second);
        
        float halfSize = size * 0.5f;
        return (position.x - halfSize >= cellMin.x && 
                position.x + halfSize <= cellMax.x &&
                position.y - halfSize >= cellMin.y && 
                position.y + halfSize <= cellMax.y);
    }
    
    // Add object to appropriate storage
    bool addToSpatialCell(size_t id, const Vec2& position, const Vec4& color, float size) {
        if (!useSpatialPartitioning || !canFitInSpatialCell(position, size)) {
            return false;
        }
        
        auto cellCoord = worldToCellCoord(position);
        SpatialCell16x16* cell = getOrCreateCell(cellCoord.first, cellCoord.second);
        return cell->addObject(id, position, color, size);
    }
    
    // Get all cells that overlap with a circle
    std::vector<SpatialCell16x16*> getCellsInRadius(const Vec2& center, float radius) const {
        std::vector<SpatialCell16x16*> result;
        if (!useSpatialPartitioning) return result;
        
        Vec2 searchMin(center.x - radius, center.y - radius);
        Vec2 searchMax(center.x + radius, center.y + radius);
        
        auto [minCellX, minCellY] = worldToCellCoord(searchMin);
        auto [maxCellX, maxCellY] = worldToCellCoord(searchMax);
        
        for (int cellY = minCellY; cellY <= maxCellY; ++cellY) {
            for (int cellX = minCellX; cellX <= maxCellX; ++cellX) {
                auto key = std::make_pair(cellX, cellY);
                auto it = spatialCells.find(key);
                if (it != spatialCells.end()) {
                    result.push_back(it->second.get());
                }
            }
        }
        
        return result;
    }
    
    // Get all cells that overlap with a region
    std::vector<SpatialCell16x16*> getCellsInRegion(const Vec2& minCorner, const Vec2& maxCorner) const {
        std::vector<SpatialCell16x16*> result;
        if (!useSpatialPartitioning) return result;
        
        auto [minCellX, minCellY] = worldToCellCoord(minCorner);
        auto [maxCellX, maxCellY] = worldToCellCoord(maxCorner);
        
        for (int cellY = minCellY; cellY <= maxCellY; ++cellY) {
            for (int cellX = minCellX; cellX <= maxCellX; ++cellX) {
                auto key = std::make_pair(cellX, cellY);
                auto it = spatialCells.find(key);
                if (it != spatialCells.end()) {
                    result.push_back(it->second.get());
                }
            }
        }
        
        return result;
    }
    
    // Remove object from spatial cells
    void removeFromSpatialCells(size_t id, const Vec2& position, float size) {
        if (!useSpatialPartitioning) return;
        
        // Calculate object bounds
        float halfSize = size * 0.5f;
        Vec2 objMin(position.x - halfSize, position.y - halfSize);
        Vec2 objMax(position.x + halfSize, position.y + halfSize);
        
        // Get affected cells
        auto [minCellX, minCellY] = worldToCellCoord(objMin);
        auto [maxCellX, maxCellY] = worldToCellCoord(objMax);
        
        // Remove from all overlapping cells
        for (int cellY = minCellY; cellY <= maxCellY; ++cellY) {
            for (int cellX = minCellX; cellX <= maxCellX; ++cellX) {
                auto key = std::make_pair(cellX, cellY);
                auto it = spatialCells.find(key);
                if (it != spatialCells.end()) {
                    it->second->removeObject(id);
                }
            }
        }
    }
    
    // Check if object is in spatial cells
    bool isInSpatialCells(size_t id) const {
        if (!useSpatialPartitioning) return false;
        
        for (const auto& cellPair : spatialCells) {
            if (cellPair.second->hasObject(id)) {
                return true;
            }
        }
        return false;
    }
    
    // Get object position (handles both storage types)
    Vec2 getObjectPosition(size_t id) const {
        // Check spatial cells first
        if (useSpatialPartitioning) {
            for (const auto& cellPair : spatialCells) {
                if (cellPair.second->hasObject(id)) {
                    return cellPair.second->getPosition(id);
                }
            }
        }
        
        // Check loose objects
        auto it = loosePositions.find(id);
        return it != loosePositions.end() ? it->second : Vec2();
    }
    
    // Get object color (handles both storage types)
    Vec4 getObjectColor(size_t id) const {
        // Check spatial cells first
        if (useSpatialPartitioning) {
            for (const auto& cellPair : spatialCells) {
                if (cellPair.second->hasObject(id)) {
                    return cellPair.second->getColor(id);
                }
            }
        }
        
        // Check loose objects
        auto it = looseColors.find(id);
        return it != looseColors.end() ? it->second : Vec4();
    }
    
    // Get object size (handles both storage types)
    float getObjectSize(size_t id) const {
        // Check spatial cells first
        if (useSpatialPartitioning) {
            for (const auto& cellPair : spatialCells) {
                if (cellPair.second->hasObject(id)) {
                    return cellPair.second->getSize(id);
                }
            }
        }
        
        // Check loose objects
        auto it = looseSizes.find(id);
        return it != looseSizes.end() ? it->second : 1.0f;
    }
    
public:
    Grid2() : next_id(0), useSpatialPartitioning(false) {}
    
    // Initialize with spatial partitioning
    void initializeSpatialPartitioning(const Vec2& minCorner, const Vec2& maxCorner) {
        gridMin = minCorner;
        gridMax = maxCorner;
        spatialCells.clear();
        useSpatialPartitioning = true;
        
        // Move existing objects to appropriate storage
        std::vector<size_t> toRemove;
        for (const std::pair<const size_t, Vec2>& pair : loosePositions) {
            size_t id = pair.first;
            const Vec2& pos = pair.second;
            const Vec4& color = looseColors[id];
            float size = looseSizes[id];
            
            if (addToSpatialCell(id, pos, color, size)) {
                toRemove.push_back(id);
            }
        }
        
        // Remove objects that were moved to spatial cells
        for (size_t id : toRemove) {
            loosePositions.erase(id);
            looseColors.erase(id);
            looseSizes.erase(id);
        }
    }
    
    void initializeSpatialPartitioning(float minX, float minY, float maxX, float maxY) {
        initializeSpatialPartitioning(Vec2(minX, minY), Vec2(maxX, maxY));
    }
    
    void addObjects(const std::vector<std::tuple<Vec2, Vec4, float>>& objects) {
        for (const auto& obj : objects) {
            addObject(std::get<0>(obj), std::get<1>(obj), std::get<2>(obj));
        }
    }

    size_t addObject(const Vec2& position, const Vec4& color, float size = 1.0f) {
        size_t id = next_id++;
        
        // Try to add to spatial cell first
        if (!addToSpatialCell(id, position, color, size)) {
            // Fall back to loose storage
            loosePositions[id] = position;
            looseColors[id] = color;
            looseSizes[id] = size;
        }
        
        return id;
    }

    // Simple getters
    Vec2 getPosition(size_t id) const {
        return getObjectPosition(id);
    }

    Vec4 getColor(size_t id) const {
        return getObjectColor(id);
    }

    float getSize(size_t id) const {
        return getObjectSize(id);
    }

    void setPosition(size_t id, const Vec2& position) {
        if (!hasObject(id)) return;
        
        Vec2 oldPos = getObjectPosition(id);
        float size = getObjectSize(id);
        
        // Remove from current storage
        if (isInSpatialCells(id)) {
            removeFromSpatialCells(id, oldPos, size);
        } else {
            loosePositions.erase(id);
        }
        
        // Add to appropriate storage
        if (!addToSpatialCell(id, position, getObjectColor(id), size)) {
            loosePositions[id] = position;
        }
    }

    void setColor(size_t id, const Vec4& color) {
        if (!hasObject(id)) return;
        
        if (isInSpatialCells(id)) {
            // Update in spatial cell
            Vec2 pos = getObjectPosition(id);
            float size = getObjectSize(id);
            removeFromSpatialCells(id, pos, size);
            addToSpatialCell(id, pos, color, size);
        } else {
            // Update in loose storage
            looseColors[id] = color;
        }
    }

    void setSize(size_t id, float size) {
        if (!hasObject(id)) return;
        
        Vec2 pos = getObjectPosition(id);
        Vec4 color = getObjectColor(id);
        float oldSize = getObjectSize(id);
        
        // Remove from current storage
        if (isInSpatialCells(id)) {
            removeFromSpatialCells(id, pos, oldSize);
        } else {
            loosePositions.erase(id);
            looseColors.erase(id);
            looseSizes.erase(id);
        }
        
        // Add to appropriate storage with new size
        if (!addToSpatialCell(id, pos, color, size)) {
            loosePositions[id] = pos;
            looseColors[id] = color;
            looseSizes[id] = size;
        }
    }
    
    bool hasObject(size_t id) const {
        if (isInSpatialCells(id)) return true;
        return loosePositions.find(id) != loosePositions.end();
    }

    
    void removeObjects(const std::vector<size_t>& ids) {
        for (size_t id : ids) {
            removeObject(id);
        }
    }
    
    void removeObject(size_t id) {
        if (!hasObject(id)) return;
        
        Vec2 pos = getObjectPosition(id);
        float size = getObjectSize(id);
        
        if (isInSpatialCells(id)) {
            removeFromSpatialCells(id, pos, size);
        } else {
            loosePositions.erase(id);
            looseColors.erase(id);
            looseSizes.erase(id);
        }
    }
    
    std::vector<size_t> getIndicesAt(const Vec2& position, float radius = 0.0f) const {
        TIME_FUNCTION;
        
        std::unordered_set<size_t> result;
        
        // Query spatial cells
        if (useSpatialPartitioning && radius > 0.0f) {
            auto cells = getCellsInRadius(position, radius);
            for (SpatialCell16x16* cell : cells) {
                auto objects = cell->getObjectsInRadius(position, radius);
                result.insert(objects.begin(), objects.end());
            }
        } else if (useSpatialPartitioning && radius == 0.0f) {
            auto cellCoord = worldToCellCoord(position);
            auto key = std::make_pair(cellCoord.first, cellCoord.second);
            auto it = spatialCells.find(key);
            if (it != spatialCells.end()) {
                auto objects = it->second->getObjectsAt(position);
                result.insert(objects.begin(), objects.end());
            }
        }
        
        // Query loose objects
        float radius_sq = radius * radius;
        for (const auto& pair : loosePositions) {
            float dx = pair.second.x - position.x;
            float dy = pair.second.y - position.y;
            if (dx * dx + dy * dy <= radius_sq) {
                result.insert(pair.first);
            }
        }
        
        return std::vector<size_t>(result.begin(), result.end());
    }

    void updatePositions(const std::unordered_map<size_t, Vec2>& newPositions) {
        TIME_FUNCTION;
        
        for (const auto& [id, newPos] : newPositions) {
            if (!hasObject(id)) continue;
            
            Vec2 oldPos = getObjectPosition(id);
            Vec4 color = getObjectColor(id);
            float size = getObjectSize(id);
            
            // Remove from current storage
            if (isInSpatialCells(id)) {
                removeFromSpatialCells(id, oldPos, size);
            } else {
                loosePositions.erase(id);
            }
            
            // Add to appropriate storage with new position
            if (!addToSpatialCell(id, newPos, color, size)) {
                loosePositions[id] = newPos;
            }
        }
    }

    // Get all object IDs in the grid
    std::vector<size_t> getAllObjectIds() const {
        TIME_FUNCTION;
        
        std::unordered_set<size_t> result;
        
        // Get IDs from spatial cells
        if (useSpatialPartitioning) {
            for (const auto& cellPair : spatialCells) {
                auto cellIds = cellPair.second->getAllObjectIds();
                result.insert(cellIds.begin(), cellIds.end());
            }
        }
        
        // Get IDs from loose objects
        for (const auto& pair : loosePositions) {
            result.insert(pair.first);
        }
        
        return std::vector<size_t>(result.begin(), result.end());
    }

    // Bulk update colors using a function
    void bulkUpdateColors(const std::function<Vec4(size_t id, const Vec2& pos, const Vec4& currentColor)>& colorFunc) {
        TIME_FUNCTION;
        
        // Update colors in spatial cells
        if (useSpatialPartitioning) {
            for (auto& cellPair : spatialCells) {
                auto cell = cellPair.second.get();
                auto ids = cell->getAllObjectIds();
                
                for (size_t id : ids) {
                    Vec2 pos = cell->getPosition(id);
                    Vec4 currentColor = cell->getColor(id);
                    Vec4 newColor = colorFunc(id, pos, currentColor);
                    
                    // Remove and re-add with new color
                    float size = cell->getSize(id);
                    cell->removeObject(id);
                    cell->addObject(id, pos, newColor, size);
                }
            }
        }
        
        // Update colors in loose objects
        for (auto& pair : looseColors) {
            size_t id = pair.first;
            Vec2 pos = loosePositions[id];
            Vec4 currentColor = pair.second;
            Vec4 newColor = colorFunc(id, pos, currentColor);
            pair.second = newColor;
        }
    }

    // Get grid region as RGB data (for visualization/export)
    void getGridRegionAsRGB(const Vec2& minCorner, const Vec2& maxCorner,
                           int& width, int& height, std::vector<int>& rgbData) const {
        TIME_FUNCTION;
        
        // Determine output dimensions
        width = static_cast<int>(maxCorner.x - minCorner.x);
        height = static_cast<int>(maxCorner.y - minCorner.y);
        
        if (width <= 0 || height <= 0) {
            width = height = 0;
            rgbData.clear();
            return;
        }
        
        // Initialize output data (3 channels per pixel)
        rgbData.resize(width * height * 3, 0);
        
        // Get objects in the region
        std::vector<size_t> objectIds;
        
        if (useSpatialPartitioning) {
            // Use spatial partitioning for efficient query
            auto cells = getCellsInRegion(minCorner, maxCorner);
            for (SpatialCell16x16* cell : cells) {
                auto cellObjects = cell->getObjectsInRegion(minCorner, maxCorner);
                objectIds.insert(objectIds.end(), cellObjects.begin(), cellObjects.end());
            }
        } else {
            // Fall back to checking all loose objects
            for (const auto& pair : loosePositions) {
                const Vec2& pos = pair.second;
                if (pos.x >= minCorner.x && pos.x <= maxCorner.x &&
                    pos.y >= minCorner.y && pos.y <= maxCorner.y) {
                    objectIds.push_back(pair.first);
                }
            }
        }
        
        // Rasterize objects to RGB
        for (size_t id : objectIds) {
            Vec2 pos = getObjectPosition(id);
            Vec4 color = getObjectColor(id);
            float size = getObjectSize(id);
            
            // Convert world coordinates to pixel coordinates
            int pixelX = static_cast<int>((pos.x - minCorner.x));
            int pixelY = static_cast<int>((pos.y - minCorner.y));
            
            // Clamp to image bounds
            pixelX = std::clamp(pixelX, 0, width - 1);
            pixelY = std::clamp(pixelY, 0, height - 1);
            
            // Convert normalized color to 8-bit RGB
            int r = static_cast<int>(std::clamp(color.x, 0.0f, 1.0f) * 255);
            int g = static_cast<int>(std::clamp(color.y, 0.0f, 1.0f) * 255);
            int b = static_cast<int>(std::clamp(color.z, 0.0f, 1.0f) * 255);
            
            // Set pixel color (RGB order)
            int index = (pixelY * width + pixelX) * 3;
            rgbData[index] = r;
            rgbData[index + 1] = g;
            rgbData[index + 2] = b;
        }
    }

    // Get grid region as BGR data (OpenCV compatible)
    void getGridRegionAsBGR(const Vec2& minCorner, const Vec2& maxCorner,
                           int& width, int& height, std::vector<int>& bgrData) const {
        TIME_FUNCTION;
        
        // First get as RGB
        std::vector<int> rgbData;
        getGridRegionAsRGB(minCorner, maxCorner, width, height, rgbData);
        
        if (width <= 0 || height <= 0) {
            bgrData.clear();
            return;
        }
        
        // Convert RGB to BGR by swapping channels
        bgrData.resize(rgbData.size());
        for (int i = 0; i < width * height; ++i) {
            int rgbIndex = i * 3;
            int bgrIndex = i * 3;
            bgrData[bgrIndex] = rgbData[rgbIndex + 2];     // B <- R
            bgrData[bgrIndex + 1] = rgbData[rgbIndex + 1]; // G <- G  
            bgrData[bgrIndex + 2] = rgbData[rgbIndex];     // R <- B
        }
    }

    void getBoundingBox(Vec2& minCorner, Vec2& maxCorner) const {
        TIME_FUNCTION;
        
        bool hasObjects = false;
        minCorner = Vec2(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        maxCorner = Vec2(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
        
        // Process spatial cells
        if (useSpatialPartitioning) {
            for (const auto& cellPair : spatialCells) {
                auto ids = cellPair.second->getAllObjectIds();
                for (size_t id : ids) {
                    const Vec2& pos = cellPair.second->getPosition(id);
                    float size = cellPair.second->getSize(id);
                    float halfSize = size * 0.5f;
                    
                    minCorner.x = std::min(minCorner.x, pos.x - halfSize);
                    minCorner.y = std::min(minCorner.y, pos.y - halfSize);
                    maxCorner.x = std::max(maxCorner.x, pos.x + halfSize);
                    maxCorner.y = std::max(maxCorner.y, pos.y + halfSize);
                    hasObjects = true;
                }
            }
        }
        
        // Process loose objects
        for (const auto& pair : loosePositions) {
            const Vec2& pos = pair.second;
            float size = looseSizes.at(pair.first);
            float halfSize = size * 0.5f;
            
            minCorner.x = std::min(minCorner.x, pos.x - halfSize);
            minCorner.y = std::min(minCorner.y, pos.y - halfSize);
            maxCorner.x = std::max(maxCorner.x, pos.x + halfSize);
            maxCorner.y = std::max(maxCorner.y, pos.y + halfSize);
            hasObjects = true;
        }
        
        if (!hasObjects) {
            minCorner = Vec2(0.0f, 0.0f);
            maxCorner = Vec2(0.0f, 0.0f);
        }
    }

    // Spatial grid statistics
    size_t getSpatialGridCellCount() const { 
        return useSpatialPartitioning ? spatialCells.size() : 0; 
    }
    
    size_t getSpatialGridObjectCount() const { 
        if (!useSpatialPartitioning) return 0;
        
        size_t total = 0;
        for (const auto& pair : spatialCells) {
            total += pair.second->getObjectCount();
        }
        return total;
    }
    
    size_t getLooseObjectCount() const {
        return loosePositions.size();
    }
    
    bool isUsingSpatialPartitioning() const { return useSpatialPartitioning; }
    
    void clear() {
        loosePositions.clear();
        looseColors.clear();
        looseSizes.clear();
        spatialCells.clear();
        next_id = 0;
    }
    
private:
    std::unordered_map<std::pair<int, int>, std::unique_ptr<Grid2>, PairHash> nestedGrids;
    int gridLevel; 
    int maxLevels;
    bool isLeafGrid;
    
    Grid2* getOrCreateNestedGrid(int cellX, int cellY) {
        auto key = std::make_pair(cellX, cellY);
        auto it = nestedGrids.find(key);
        if (it != nestedGrids.end()) {
            return it->second.get();
        }
        
        if (gridLevel >= maxLevels) {
            return nullptr; // Reached maximum depth
        }
        
        // Create new nested grid with subdivided bounds
        Vec2 nestedMin = getCellMinCorner(cellX, cellY);
        Vec2 nestedMax = getCellMaxCorner(cellX, cellY);
        
        auto nestedGrid = std::make_unique<Grid2>();
        nestedGrid->initializeSpatialPartitioning(nestedMin, nestedMax);
        nestedGrid->setGridLevel(gridLevel + 1);
        nestedGrid->setMaxLevels(maxLevels);
        
        nestedGrids[key] = std::move(nestedGrid);
        return nestedGrids[key].get();
    }
    
    // Check if object should be pushed to nested grid
    bool shouldPushToNestedGrid(const Vec2& position, float size) const {
        if (gridLevel >= maxLevels) return false;
        
        // Push to nested grid if object fits completely in a single cell
        // and we haven't reached maximum depth
        return canFitInSpatialCell(position, size);
    }

public:
    void setGridLevel(int level) { gridLevel = level; }
    void setMaxLevels(int levels) { maxLevels = levels; }
    int getGridLevel() const { return gridLevel; }
};

#endif