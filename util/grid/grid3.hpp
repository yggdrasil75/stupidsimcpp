#ifndef GRID3_HPP
#define GRID3_HPP

#include "vec3.hpp"
#include "vec4.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <map>
#include <unordered_set>

class Grid3 {
private:
    // size_t is index
    // Vec3 is x,y,z position of the sparse voxel
    std::multimap<size_t, Vec3> positions;
    // Vec4 is rgba color at the position
    std::multimap<size_t, Vec4> colors;
    // size is a floating size to assign to a voxel to allow larger or smaller assignments
    std::multimap<size_t, float> sizes;
    size_t next_id;
    
    std::unordered_map<size_t, std::tuple<int, int, int>> cellIndices; // object ID -> grid cell
    std::unordered_map<std::tuple<int, int, int>, std::unordered_set<size_t>> spatialGrid; // cell -> object IDs
    float cellSize;
    
public:
    Grid3() : next_id(0), cellSize(1.0f) {}
    Grid3(float cellSize) : next_id(0), cellSize(cellSize) {}
    
    size_t addObject(const Vec3& position, const Vec4& color, float size = 1.0f) {
        size_t id = next_id++;
        positions.insert({id, position});
        colors.insert({id, color});
        sizes.insert({id, size});
        auto cell = worldToGrid(position);
        spatialGrid[cell].insert(id);
        cellIndices[id] = cell;
        return id;
    }

    // Gets
    Vec3 getPosition(size_t id) const {
        auto it = positions.find(id);
        if (it != positions.end()) return it->second;
        return Vec3();
    }

    Vec4 getColor(size_t id) const {
        auto it = colors.find(id);
        if (it != colors.end()) return it->second;
        return Vec4();
    }

    float getSize(size_t id) const {
        auto it = sizes.find(id);
        if (it != sizes.end()) return it->second;
        return 1.0f;
    }

    // Sets
    void setPosition(size_t id, const Vec3& position) {
        if (!hasObject(id)) return;
        
        Vec3 oldPos = getPosition(id);
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
    void addObjects(const std::vector<std::tuple<Vec3, Vec4, float>>& objects) {
        for (const auto& obj : objects) {
            addObject(std::get<0>(obj), std::get<1>(obj), std::get<2>(obj));
        }
    }
    
    void removeObjects(const std::vector<size_t>& ids) {
        for (size_t id : ids) {
            removeObject(id);
        }
    }
    
    // Batch position updates
    void updatePositions(const std::unordered_map<size_t, Vec3>& newPositions) {
        // Bulk update spatial grid - collect all changes first
        std::vector<std::tuple<size_t, Vec3, Vec3>> spatialUpdates;
        
        for (const auto& pair : newPositions) {
            if (hasObject(pair.first)) {
                Vec3 oldPos = getPosition(pair.first);
                positions.erase(pair.first);
                positions.insert({pair.first, pair.second});
                spatialUpdates.emplace_back(pair.first, oldPos, pair.second);
            }
        }
        
        // Apply all spatial updates at once
        for (const auto& update : spatialUpdates) {
            updateSpatialIndex(std::get<0>(update), std::get<1>(update), std::get<2>(update));
        }
    }
    
    // Other
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
    
    std::vector<size_t> getIndicesAt(float x, float y, float z, float radius = 0.0f) const {
        return getIndicesAt(Vec3(x, y, z), radius);
    }
    
    std::vector<size_t> getIndicesAt(const Vec3& position, float radius = 0.0f) const {
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
                float dz = pair.second.z - position.z;
                if (dx * dx + dy * dy + dz * dz <= radius_sq) {
                    result.push_back(pair.first);
                }
            }
        }
        
        return result;
    }
    
    void getBoundingBox(Vec3& minCorner, Vec3& maxCorner) const {
        if (positions.empty()) {
            minCorner = Vec3(0.0f, 0.0f, 0.0f);
            maxCorner = Vec3(0.0f, 0.0f, 0.0f);
            return;
        }
        
        auto it = positions.begin();
        minCorner = it->second;
        maxCorner = it->second;
        
        for (const auto& pair : positions) {
            const Vec3& pos = pair.second;
            minCorner.x = std::min(minCorner.x, pos.x);
            minCorner.y = std::min(minCorner.y, pos.y);
            minCorner.z = std::min(minCorner.z, pos.z);
            maxCorner.x = std::max(maxCorner.x, pos.x);
            maxCorner.y = std::max(maxCorner.y, pos.y);
            maxCorner.z = std::max(maxCorner.z, pos.z);
        }
    }
    
    // Get 2D slice of the 3D grid (useful for visualization)
    void getSliceAsRGB(int axis, float slicePos, 
                      int& width, int& height, std::vector<int>& rgbData) const {
        Vec3 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        
        // Determine slice dimensions based on axis (0=x, 1=y, 2=z)
        if (axis == 0) { // X-slice
            width = static_cast<int>(std::ceil(maxCorner.z - minCorner.z)) + 1;
            height = static_cast<int>(std::ceil(maxCorner.y - minCorner.y)) + 1;
        } else if (axis == 1) { // Y-slice
            width = static_cast<int>(std::ceil(maxCorner.z - minCorner.z)) + 1;
            height = static_cast<int>(std::ceil(maxCorner.x - minCorner.x)) + 1;
        } else { // Z-slice
            width = static_cast<int>(std::ceil(maxCorner.x - minCorner.x)) + 1;
            height = static_cast<int>(std::ceil(maxCorner.y - minCorner.y)) + 1;
        }
        
        // Initialize with black (0,0,0)
        rgbData.resize(width * height * 3, 0);
        
        // Fill the slice with object colors
        for (const auto& posPair : positions) {
            size_t id = posPair.first;
            const Vec3& pos = posPair.second;
            
            // Check if position is within slice tolerance
            float tolerance = 0.5f; // Half voxel tolerance
            bool inSlice = false;
            int gridX = 0, gridY = 0;
            
            if (axis == 0 && std::abs(pos.x - slicePos) <= tolerance) { // X-slice
                gridX = static_cast<int>(pos.z - minCorner.z);
                gridY = static_cast<int>(pos.y - minCorner.y);
                inSlice = true;
            } else if (axis == 1 && std::abs(pos.y - slicePos) <= tolerance) { // Y-slice
                gridX = static_cast<int>(pos.z - minCorner.z);
                gridY = static_cast<int>(pos.x - minCorner.x);
                inSlice = true;
            } else if (axis == 2 && std::abs(pos.z - slicePos) <= tolerance) { // Z-slice
                gridX = static_cast<int>(pos.x - minCorner.x);
                gridY = static_cast<int>(pos.y - minCorner.y);
                inSlice = true;
            }
            
            if (inSlice && gridX >= 0 && gridX < width && gridY >= 0 && gridY < height) {
                const Vec4& color = getColor(id);
                int index = (gridY * width + gridX) * 3;
                
                // Convert float color [0,1] to int [0,255]
                rgbData[index] = static_cast<int>(color.r * 255);
                rgbData[index + 1] = static_cast<int>(color.g * 255);
                rgbData[index + 2] = static_cast<int>(color.b * 255);
            }
        }
    }
    
    void getRegionAsRGB(float minX, float minY, float minZ, float maxX, float maxY, float maxZ,
                       int& width, int& height, std::vector<int>& rgbData) const {
        // For 3D, this creates a 2D projection (XY plane at average Z)
        if (minX >= maxX || minY >= maxY || minZ >= maxZ) {
            width = 0;
            height = 0;
            rgbData.clear();
            return;
        }
        
        // Calculate grid dimensions for XY projection
        width = static_cast<int>(std::ceil(maxX - minX));
        height = static_cast<int>(std::ceil(maxY - minY));
        
        // Initialize with black (0,0,0)
        rgbData.resize(width * height * 3, 0);
        
        // Fill the grid with object colors in the region (XY projection)
        for (const auto& posPair : positions) {
            size_t id = posPair.first;
            const Vec3& pos = posPair.second;
            
            // Check if position is within the region
            if (pos.x >= minX && pos.x < maxX && 
                pos.y >= minY && pos.y < maxY &&
                pos.z >= minZ && pos.z < maxZ) {
                
                // Convert world position to grid coordinates (XY projection)
                int gridX = static_cast<int>(pos.x - minX);
                int gridY = static_cast<int>(pos.y - minY);
                
                if (gridX >= 0 && gridX < width && gridY >= 0 && gridY < height) {
                    const Vec4& color = getColor(id);
                    int index = (gridY * width + gridX) * 3;
                    
                    // Convert float color [0,1] to int [0,255]
                    rgbData[index] = static_cast<int>(color.r * 255);
                    rgbData[index + 1] = static_cast<int>(color.g * 255);
                    rgbData[index + 2] = static_cast<int>(color.b * 255);
                }
            }
        }
    }
    
    void getRegionAsRGB(const Vec3& minCorner, const Vec3& maxCorner,
                       int& width, int& height, std::vector<int>& rgbData) const {
        getRegionAsRGB(minCorner.x, minCorner.y, minCorner.z, 
                      maxCorner.x, maxCorner.y, maxCorner.z,
                      width, height, rgbData);
    }

    // Spatial grid methods for 3D
    std::tuple<int, int, int> worldToGrid(const Vec3& pos) const {
        return {
            static_cast<int>(std::floor(pos.x / cellSize)),
            static_cast<int>(std::floor(pos.y / cellSize)),
            static_cast<int>(std::floor(pos.z / cellSize))
        };
    }
    
    void updateSpatialIndex(size_t id, const Vec3& oldPos, const Vec3& newPos) {
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
    
    std::vector<size_t> getIndicesInRadius(const Vec3& position, float radius) const {
        std::vector<size_t> result;
        
        Vec3 minPos(position.x - radius, position.y - radius, position.z - radius);
        Vec3 maxPos(position.x + radius, position.y + radius, position.z + radius);
        
        auto minCell = worldToGrid(minPos);
        auto maxCell = worldToGrid(maxPos);
        
        float radiusSq = radius * radius;
        
        // Only check relevant cells
        for (int x = std::get<0>(minCell); x <= std::get<0>(maxCell); ++x) {
            for (int y = std::get<1>(minCell); y <= std::get<1>(maxCell); ++y) {
                for (int z = std::get<2>(minCell); z <= std::get<2>(maxCell); ++z) {
                    auto cell = std::make_tuple(x, y, z);
                    auto it = spatialGrid.find(cell);
                    if (it != spatialGrid.end()) {
                        for (size_t id : it->second) {
                            const Vec3& objPos = getPosition(id);
                            float dx = objPos.x - position.x;
                            float dy = objPos.y - position.y;
                            float dz = objPos.z - position.z;
                            if (dx * dx + dy * dy + dz * dz <= radiusSq) {
                                result.push_back(id);
                            }
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    std::vector<size_t> getIndicesInRegion(const Vec3& minCorner, const Vec3& maxCorner) const {
        std::vector<size_t> result;
        
        auto minCell = worldToGrid(minCorner);
        auto maxCell = worldToGrid(maxCorner);
        
        for (int x = std::get<0>(minCell); x <= std::get<0>(maxCell); ++x) {
            for (int y = std::get<1>(minCell); y <= std::get<1>(maxCell); ++y) {
                for (int z = std::get<2>(minCell); z <= std::get<2>(maxCell); ++z) {
                    auto cell = std::make_tuple(x, y, z);
                    auto it = spatialGrid.find(cell);
                    if (it != spatialGrid.end()) {
                        for (size_t id : it->second) {
                            const Vec3& pos = getPosition(id);
                            if (pos.x >= minCorner.x && pos.x <= maxCorner.x &&
                                pos.y >= minCorner.y && pos.y <= maxCorner.y &&
                                pos.z >= minCorner.z && pos.z <= maxCorner.z) {
                                result.push_back(id);
                            }
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
    
    // 3D-specific utility methods
    size_t getVoxelCount() const { return positions.size(); }
    
    // Get density information (useful for volume rendering)
    std::vector<float> getDensityGrid(int resX, int resY, int resZ) const {
        std::vector<float> density(resX * resY * resZ, 0.0f);
        
        Vec3 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        
        Vec3 gridSize = maxCorner - minCorner;
        if (gridSize.x <= 0 || gridSize.y <= 0 || gridSize.z <= 0) {
            return density;
        }
        
        Vec3 voxelSize(gridSize.x / resX, gridSize.y / resY, gridSize.z / resZ);
        
        for (const auto& posPair : positions) {
            const Vec3& pos = posPair.second;
            
            // Convert to grid coordinates
            int gx = static_cast<int>((pos.x - minCorner.x) / gridSize.x * resX);
            int gy = static_cast<int>((pos.y - minCorner.y) / gridSize.y * resY);
            int gz = static_cast<int>((pos.z - minCorner.z) / gridSize.z * resZ);
            
            if (gx >= 0 && gx < resX && gy >= 0 && gy < resY && gz >= 0 && gz < resZ) {
                density[gz * resX * resY + gy * resX + gx] += 1.0f;
            }
        }
        
        return density;
    }
};

#endif