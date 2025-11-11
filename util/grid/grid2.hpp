#ifndef GRID2_HPP
#define GRID2_HPP

#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec4.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <map>
#include <unordered_set>
#include <cmath>

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
    //size_t is index.
    //vec2 is x,y position of the sparse value
    std::multimap<size_t, Vec2> positions;
    //vec4 is rgba color at the position
    std::multimap<size_t, Vec4> colors;
    //size is a floating size to assign to a "pixel" (or voxel for grid3) to allow larger or smaller assignments in this map
    std::multimap<size_t, float> sizes;
    //others will be added later
    size_t next_id;
    
    std::unordered_map<size_t, std::pair<int, int>> cellIndices; // object ID -> grid cell
    std::unordered_map<std::pair<int, int>, std::unordered_set<size_t>, PairHash> spatialGrid; // cell -> object IDs
    float cellSize;
    
public:
    Grid2() : next_id(0), cellSize(1.0f) {}
    Grid2(float cellSize) : next_id(0), cellSize(cellSize) {}
    
    size_t addObject(const Vec2& position, const Vec4& color, float size = 1.0f) {
        size_t id = next_id++;
        positions.insert({id, position});
        colors.insert({id, color});
        sizes.insert({id, size});
        std::pair<int,int> cell = worldToGrid(position);
        spatialGrid[cell].insert(id);
        cellIndices[id] = cell;
        return id;
    }

    //gets
    Vec2 getPosition(size_t id) const {
        std::multimap<size_t, Vec2>::const_iterator it = positions.find(id);
        if (it != positions.end()) return it->second;
        return Vec2();
    }

    Vec4 getColor(size_t id) const {
        std::multimap<size_t, Vec4>::const_iterator it = colors.find(id);
        if (it != colors.end()) return it->second;
        return Vec4();
    }

    float getSize(size_t id) const {
        std::multimap<size_t, float>::const_iterator it = sizes.find(id);
        if (it != sizes.end()) return it->second;
        return 1.0f;
    }

    //sets
    void setPosition(size_t id, const Vec2& position) {
        if (!hasObject(id)) return;
        
        Vec2 oldPos = getPosition(id);
        positions.erase(id);
        positions.insert({id, position});
        updateSpatialIndex(id, oldPos, position);
    }

    void setColor(size_t id, const Vec4& color) {
        colors.erase(id);
        colors.insert({id, color});
    }

    void setSize(size_t id, float size) {
        sizes.erase(id);
        sizes.insert({id, size});
    }
    
    // Batch add/remove operations
    void addObjects(const std::vector<std::tuple<Vec2, Vec4, float>>& objects) {
        for (const std::tuple<Vec2, Vec4, float>& obj : objects) {
            addObject(std::get<0>(obj), std::get<1>(obj), std::get<2>(obj));
        }
    }
    
    void removeObjects(const std::vector<size_t>& ids) {
        for (size_t id : ids) {
            removeObject(id);
        }
    }
    
    // Batch position updates
    void updatePositions(const std::unordered_map<size_t, Vec2>& newPositions) {
        // Bulk update spatial grid - collect all changes first
        std::vector<std::tuple<size_t, Vec2, Vec2>> spatialUpdates;
        
        for (const std::pair<const size_t, Vec2>& pair : newPositions) {
            if (hasObject(pair.first)) {
                Vec2 oldPos = getPosition(pair.first);
                positions.erase(pair.first);
                positions.insert({pair.first, pair.second});
                spatialUpdates.emplace_back(pair.first, oldPos, pair.second);
            }
        }
        
        // Apply all spatial updates at once
        for (const std::tuple<size_t, Vec2, Vec2>& update : spatialUpdates) {
            updateSpatialIndex(std::get<0>(update), std::get<1>(update), std::get<2>(update));
        }
    }
    
    //other
    bool hasObject(size_t id) const {
        return positions.find(id) != positions.end();
    }
    
    void removeObject(size_t id) {
        // Remove from spatial grid first
        auto cellIt = cellIndices.find(id);
        if (cellIt != cellIndices.end()) {
            auto& cellObjects = spatialGrid[cellIt->second];
            cellObjects.erase(id);
            if (cellObjects.empty()) {
                spatialGrid.erase(cellIt->second);
            }
            cellIndices.erase(id);
        }
        
        // Remove from data maps
        positions.erase(id);
        colors.erase(id);
        sizes.erase(id);
    }
    
    std::vector<size_t> getIndicesAt(float x, float y, float radius = 0.0f) const {
        return getIndicesAt(Vec2(x, y), radius);
    }
    
    std::vector<size_t> getIndicesAt(const Vec2& position, float radius = 0.0f) const {
        std::vector<size_t> result;
        
        if (radius <= 0.0f) {
            // Exact position match
            for (const auto& pair : positions) {
                if (pair.second == position) {
                    result.push_back(pair.first);
                }
            }
        } else {
            // Radius-based search
            float radius_sq = radius * radius;
            for (const auto& pair : positions) {
                float dx = pair.second.x - position.x;
                float dy = pair.second.y - position.y;
                if (dx * dx + dy * dy <= radius_sq) {
                    result.push_back(pair.first);
                }
            }
        }
        
        return result;
    }
    
    void getBoundingBox(Vec2& minCorner, Vec2& maxCorner) const {
        if (positions.empty()) {
            minCorner = Vec2(0.0f, 0.0f);
            maxCorner = Vec2(0.0f, 0.0f);
            return;
        }
        
        auto it = positions.begin();
        minCorner = it->second;
        maxCorner = it->second;
        
        for (const auto& pair : positions) {
            const Vec2& pos = pair.second;
            float size = getSize(pair.first);
            float halfSize = size * 0.5f;
            
            minCorner.x = std::min(minCorner.x, pos.x - halfSize);
            minCorner.y = std::min(minCorner.y, pos.y - halfSize);
            maxCorner.x = std::max(maxCorner.x, pos.x + halfSize);
            maxCorner.y = std::max(maxCorner.y, pos.y + halfSize);
        }
    }
    
    //to picture
    void getGridAsRGB(int& width, int& height, std::vector<int>& rgbData) const {
        Vec2 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        
        // Calculate grid dimensions (adding 1 to include both ends)
        width = static_cast<int>(std::ceil(maxCorner.x - minCorner.x)) ;
        height = static_cast<int>(std::ceil(maxCorner.y - minCorner.y)) ;
        
        // Initialize with black (0,0,0)
        rgbData.resize(width * height * 3, 0);
        
        // Fill the grid with object colors, accounting for sizes
        for (const auto& posPair : positions) {
            size_t id = posPair.first;
            const Vec2& pos = posPair.second;
            float size = getSize(id);
            const Vec4& color = getColor(id);
            
            // Calculate the bounding box of this object in grid coordinates
            float halfSize = size * 0.5f;
            int minGridX = static_cast<int>(std::floor((pos.x - halfSize - minCorner.x)));
            int minGridY = static_cast<int>(std::floor((pos.y - halfSize - minCorner.y)));
            int maxGridX = static_cast<int>(std::ceil((pos.x + halfSize - minCorner.x)));
            int maxGridY = static_cast<int>(std::ceil((pos.y + halfSize - minCorner.y)));
            
            // Clamp to grid boundaries
            minGridX = std::max(0, minGridX);
            minGridY = std::max(0, minGridY);
            maxGridX = std::min(width - 1, maxGridX);
            maxGridY = std::min(height - 1, maxGridY);
            
            // Fill all pixels within the object's size
            for (int y = minGridY; y <= maxGridY; ++y) {
                for (int x = minGridX; x <= maxGridX; ++x) {
                    int index = (y * width + x) * 3;
                    
                    // Convert float color [0,1] to int [0,255]
                    rgbData[index] = static_cast<int>(color.r * 255);
                    rgbData[index + 1] = static_cast<int>(color.g * 255);
                    rgbData[index + 2] = static_cast<int>(color.b * 255);
                }
            }
        }
    }
    
    void getRegionAsRGB(float minX, float minY, float maxX, float maxY, 
                       int& width, int& height, std::vector<int>& rgbData) const {
        // Ensure valid region
        if (minX >= maxX || minY >= maxY) {
            width = 0;
            height = 0;
            rgbData.clear();
            return;
        }
        
        // Calculate grid dimensions
        width = static_cast<int>(std::ceil(maxX - minX));
        height = static_cast<int>(std::ceil(maxY - minY));
        
        // Initialize with black (0,0,0)
        rgbData.resize(width * height * 3, 0);
        
        // Fill the grid with object colors in the region, accounting for sizes
        for (const auto& posPair : positions) {
            size_t id = posPair.first;
            const Vec2& pos = posPair.second;
            float size = getSize(id);
            const Vec4& color = getColor(id);
            
            // Calculate the bounding box of this object in world coordinates
            float halfSize = size * 0.5f;
            float objMinX = pos.x - halfSize;
            float objMinY = pos.y - halfSize;
            float objMaxX = pos.x + halfSize;
            float objMaxY = pos.y + halfSize;
            
            // Check if object overlaps with the region
            if (objMaxX >= minX && objMinX <= maxX && objMaxY >= minY && objMinY <= maxY) {
                // Calculate overlapping region in grid coordinates
                int minGridX = static_cast<int>(std::floor(std::max(objMinX, minX) - minX));
                int minGridY = static_cast<int>(std::floor(std::max(objMinY, minY) - minY));
                int maxGridX = static_cast<int>(std::ceil(std::min(objMaxX, maxX) - minX));
                int maxGridY = static_cast<int>(std::ceil(std::min(objMaxY, maxY) - minY));
                
                // Clamp to grid boundaries
                minGridX = std::max(0, minGridX);
                minGridY = std::max(0, minGridY);
                maxGridX = std::min(width - 1, maxGridX);
                maxGridY = std::min(height - 1, maxGridY);
                
                // Fill all pixels within the object's overlapping region
                for (int y = minGridY; y <= maxGridY; ++y) {
                    for (int x = minGridX; x <= maxGridX; ++x) {
                        int index = (y * width + x) * 3;
                        
                        // Convert float color [0,1] to int [0,255]
                        rgbData[index] = static_cast<int>(color.r * 255);
                        rgbData[index + 1] = static_cast<int>(color.g * 255);
                        rgbData[index + 2] = static_cast<int>(color.b * 255);
                    }
                }
            }
        }
    }
    
    void getRegionAsRGB(const Vec2& minCorner, const Vec2& maxCorner,
                       int& width, int& height, std::vector<int>& rgbData) const {
        getRegionAsRGB(minCorner.x, minCorner.y, maxCorner.x, maxCorner.y, 
                      width, height, rgbData);
    }

    //spatial map
    std::pair<int, int> worldToGrid(const Vec2& pos) const {
        return {
            static_cast<int>(std::floor(pos.x / cellSize)),
            static_cast<int>(std::floor(pos.y / cellSize))
        };
    }
    
    void updateSpatialIndex(size_t id, const Vec2& oldPos, const Vec2& newPos) {
        auto oldCell = worldToGrid(oldPos);
        auto newCell = worldToGrid(newPos);
        
        if (oldCell != newCell) {
            // Remove from old cell
            auto oldIt = spatialGrid.find(oldCell);
            if (oldIt != spatialGrid.end()) {
                oldIt->second.erase(id);
                if (oldIt->second.empty()) {
                    spatialGrid.erase(oldIt);
                }
            }
            
            // Add to new cell
            spatialGrid[newCell].insert(id);
            cellIndices[id] = newCell;
        }
    }
    
    std::vector<size_t> getIndicesInRadius(const Vec2& position, float radius) const {
        std::vector<size_t> result;
        
        Vec2 minPos(position.x - radius, position.y - radius);
        Vec2 maxPos(position.x + radius, position.y + radius);
        
        auto minCell = worldToGrid(minPos);
        auto maxCell = worldToGrid(maxPos);
        
        float radiusSq = radius * radius;
        
        // Only check relevant cells
        for (int x = minCell.first; x <= maxCell.first; ++x) {
            for (int y = minCell.second; y <= maxCell.second; ++y) {
                auto cell = std::make_pair(x, y);
                auto it = spatialGrid.find(cell);
                if (it != spatialGrid.end()) {
                    for (size_t id : it->second) {
                        const Vec2& objPos = getPosition(id);
                        float dx = objPos.x - position.x;
                        float dy = objPos.y - position.y;
                        if (dx * dx + dy * dy <= radiusSq) {
                            result.push_back(id);
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    std::vector<size_t> getIndicesInRegion(const Vec2& minCorner, const Vec2& maxCorner) const {
        std::vector<size_t> result;
        
        auto minCell = worldToGrid(minCorner);
        auto maxCell = worldToGrid(maxCorner);
        
        for (int x = minCell.first; x <= maxCell.first; ++x) {
            for (int y = minCell.second; y <= maxCell.second; ++y) {
                auto cell = std::make_pair(x, y);
                auto it = spatialGrid.find(cell);
                if (it != spatialGrid.end()) {
                    for (size_t id : it->second) {
                        const Vec2& pos = getPosition(id);
                        if (pos.x >= minCorner.x && pos.x <= maxCorner.x &&
                            pos.y >= minCorner.y && pos.y <= maxCorner.y) {
                            result.push_back(id);
                        }
                    }
                }
            }
        }
        
        return result;
    }

    size_t getSpatialGridCellCount() const { return spatialGrid.size(); }
    size_t getSpatialGridObjectCount() const { return cellIndices.size(); }
    float getCellSize() const { return cellSize; }
};

#endif