#ifndef GRID3_HPP
#define GRID3_HPP

#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "grid2.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <map>
#include <unordered_set>
#include <cmath>

class Grid3 {
private:
    std::multimap<size_t, Vec3> positions;
    std::multimap<size_t, Vec4> colors;
    std::multimap<size_t, float> sizes;
    size_t next_id;
    
    std::unordered_map<size_t, std::tuple<int, int, int>> cellIndices;
    std::unordered_map<std::tuple<int, int, int>, std::unordered_set<size_t>> spatialGrid;
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

    // Get operations
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

    // Set operations
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
    
    // Batch operations
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
    
    void updatePositions(const std::unordered_map<size_t, Vec3>& newPositions) {
        std::vector<std::tuple<size_t, Vec3, Vec3>> spatialUpdates;
        
        for (const auto& pair : newPositions) {
            if (hasObject(pair.first)) {
                Vec3 oldPos = getPosition(pair.first);
                positions.erase(pair.first);
                positions.insert({pair.first, pair.second});
                spatialUpdates.emplace_back(pair.first, oldPos, pair.second);
            }
        }
        
        for (const auto& update : spatialUpdates) {
            updateSpatialIndex(std::get<0>(update), std::get<1>(update), std::get<2>(update));
        }
    }
    
    // Object management
    bool hasObject(size_t id) const {
        return positions.find(id) != positions.end();
    }
    
    void removeObject(size_t id) {
        // Remove from spatial grid
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
    
    // Spatial queries
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
    
    // Bounding box
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
            float size = getSize(pair.first);
            float halfSize = size * 0.5f;
            
            minCorner.x = std::min(minCorner.x, pos.x - halfSize);
            minCorner.y = std::min(minCorner.y, pos.y - halfSize);
            minCorner.z = std::min(minCorner.z, pos.z - halfSize);
            maxCorner.x = std::max(maxCorner.x, pos.x + halfSize);
            maxCorner.y = std::max(maxCorner.y, pos.y + halfSize);
            maxCorner.z = std::max(maxCorner.z, pos.z + halfSize);
        }
    }
    
    // Grid2 slice generation
    Grid2 getSliceXY(float z, float thickness = 0.1f) const {
        Grid2 slice;
        Vec3 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        
        float halfThickness = thickness * 0.5f;
        float minZ = z - halfThickness;
        float maxZ = z + halfThickness;
        
        for (const auto& posPair : positions) {
            size_t id = posPair.first;
            const Vec3& pos = posPair.second;
            
            if (pos.z >= minZ && pos.z <= maxZ) {
                // Project to XY plane
                Vec2 slicePos(pos.x, pos.y);
                slice.addObject(slicePos, getColor(id), getSize(id));
            }
        }
        
        return slice;
    }
    
    Grid2 getSliceXZ(float y, float thickness = 0.1f) const {
        Grid2 slice;
        Vec3 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        
        float halfThickness = thickness * 0.5f;
        float minY = y - halfThickness;
        float maxY = y + halfThickness;
        
        for (const auto& posPair : positions) {
            size_t id = posPair.first;
            const Vec3& pos = posPair.second;
            
            if (pos.y >= minY && pos.y <= maxY) {
                // Project to XZ plane
                Vec2 slicePos(pos.x, pos.z);
                slice.addObject(slicePos, getColor(id), getSize(id));
            }
        }
        
        return slice;
    }
    
    Grid2 getSliceYZ(float x, float thickness = 0.1f) const {
        Grid2 slice;
        Vec3 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        
        float halfThickness = thickness * 0.5f;
        float minX = x - halfThickness;
        float maxX = x + halfThickness;
        
        for (const auto& posPair : positions) {
            size_t id = posPair.first;
            const Vec3& pos = posPair.second;
            
            if (pos.x >= minX && pos.x <= maxX) {
                // Project to YZ plane
                Vec2 slicePos(pos.y, pos.z);
                slice.addObject(slicePos, getColor(id), getSize(id));
            }
        }
        
        return slice;
    }
    
    // Amanatides and Woo ray-grid intersection
    struct RayHit {
        size_t objectId;
        Vec3 position;
        Vec3 normal;
        float distance;
        Vec4 color;
        
        RayHit() : objectId(-1), distance(std::numeric_limits<float>::max()) {}
    };
    
    RayHit amanatidesWooRaycast(const Vec3& rayOrigin, const Vec3& rayDirection, float maxDistance = 1000.0f) const {
        RayHit hit;
        
        if (positions.empty()) return hit;
        
        // Normalize direction
        Vec3 dir = rayDirection.normalized();
        
        // Initialize grid traversal
        auto startCell = worldToGrid(rayOrigin);
        int cellX = std::get<0>(startCell);
        int cellY = std::get<1>(startCell);
        int cellZ = std::get<2>(startCell);
        
        // Step directions
        int stepX = (dir.x > 0) ? 1 : -1;
        int stepY = (dir.y > 0) ? 1 : -1;
        int stepZ = (dir.z > 0) ? 1 : -1;
        
        // Calculate cell boundaries
        float cellMinX = cellX * cellSize;
        float cellMinY = cellY * cellSize;
        float cellMinZ = cellZ * cellSize;
        float cellMaxX = cellMinX + cellSize;
        float cellMaxY = cellMinY + cellSize;
        float cellMaxZ = cellMinZ + cellSize;
        
        // Calculate t values for cell boundaries
        float tMaxX, tMaxY, tMaxZ;
        if (dir.x != 0) {
            tMaxX = ((dir.x > 0 ? cellMaxX : cellMinX) - rayOrigin.x) / dir.x;
        } else {
            tMaxX = std::numeric_limits<float>::max();
        }
        
        if (dir.y != 0) {
            tMaxY = ((dir.y > 0 ? cellMaxY : cellMinY) - rayOrigin.y) / dir.y;
        } else {
            tMaxY = std::numeric_limits<float>::max();
        }
        
        if (dir.z != 0) {
            tMaxZ = ((dir.z > 0 ? cellMaxZ : cellMinZ) - rayOrigin.z) / dir.z;
        } else {
            tMaxZ = std::numeric_limits<float>::max();
        }
        
        // Calculate t delta
        float tDeltaX = (cellSize / std::abs(dir.x)) * (dir.x != 0 ? 1 : 0);
        float tDeltaY = (cellSize / std::abs(dir.y)) * (dir.y != 0 ? 1 : 0);
        float tDeltaZ = (cellSize / std::abs(dir.z)) * (dir.z != 0 ? 1 : 0);
        
        // Traverse grid
        float t = 0.0f;
        while (t < maxDistance) {
            // Check current cell for intersections
            auto cell = std::make_tuple(cellX, cellY, cellZ);
            auto cellIt = spatialGrid.find(cell);
            if (cellIt != spatialGrid.end()) {
                // Check all objects in this cell
                for (size_t id : cellIt->second) {
                    const Vec3& objPos = getPosition(id);
                    float objSize = getSize(id);
                    
                    // Simple sphere intersection test
                    Vec3 toObj = objPos - rayOrigin;
                    float b = toObj.dot(dir);
                    float c = toObj.dot(toObj) - objSize * objSize;
                    float discriminant = b * b - c;
                    
                    if (discriminant >= 0) {
                        float sqrtDisc = std::sqrt(discriminant);
                        float t1 = b - sqrtDisc;
                        float t2 = b + sqrtDisc;
                        
                        if (t1 >= 0 && t1 < hit.distance) {
                            hit.objectId = id;
                            hit.position = rayOrigin + dir * t1;
                            hit.normal = (hit.position - objPos).normalized();
                            hit.distance = t1;
                            hit.color = getColor(id);
                        } else if (t2 >= 0 && t2 < hit.distance) {
                            hit.objectId = id;
                            hit.position = rayOrigin + dir * t2;
                            hit.normal = (hit.position - objPos).normalized();
                            hit.distance = t2;
                            hit.color = getColor(id);
                        }
                    }
                }
                
                // If we found a hit, return it
                if (hit.objectId != static_cast<size_t>(-1)) {
                    return hit;
                }
            }
            
            // Move to next cell
            if (tMaxX < tMaxY && tMaxX < tMaxZ) {
                cellX += stepX;
                t = tMaxX;
                tMaxX += tDeltaX;
            } else if (tMaxY < tMaxZ) {
                cellY += stepY;
                t = tMaxY;
                tMaxY += tDeltaY;
            } else {
                cellZ += stepZ;
                t = tMaxZ;
                tMaxZ += tDeltaZ;
            }
        }
        
        return hit;
    }
    
    // Spatial indexing
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
        
        // Check relevant cells
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

    // Statistics
    size_t getObjectCount() const { return positions.size(); }
    size_t getSpatialGridCellCount() const { return spatialGrid.size(); }
    size_t getSpatialGridObjectCount() const { return cellIndices.size(); }
    float getCellSize() const { return cellSize; }
};

#endif