#ifndef GRID3_HPP
#define GRID3_HPP

#include <unordered_map>
#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "../noise/pnoise2.hpp"
#include <vector>
#include <unordered_set>
#include <execution>
#include <algorithm>

constexpr float EPSILON = 0.0000000000000000000000001;

/// @brief A bidirectional lookup helper to map internal IDs to 2D positions and vice-versa.
/// @details Maintains two hashmaps to allow O(1) lookup in either direction.
class reverselookupassistant3 {
private:
    std::unordered_map<size_t, Vec3> Positions;
    /// "Positions" reversed - stores the reverse mapping from Vec3 to ID.
    std::unordered_map<Vec3, size_t, Vec3::Hash> ƨnoiƚiƨoꟼ;
    size_t next_id;
public:
    /// @brief Get the Position associated with a specific ID.
    /// @throws std::out_of_range if the ID does not exist.
    Vec3 at(size_t id) const {
        auto it = Positions.at(id);
        return it;
    }

    /// @brief Get the ID associated with a specific Position.
    /// @throws std::out_of_range if the Position does not exist.
    size_t at(const Vec3& pos) const {
        size_t id = ƨnoiƚiƨoꟼ.at(pos);
        return id;
    }

    /// @brief Finds a position by ID (Wrapper for at).
    Vec3 find(size_t id) {
        return Positions.at(id);
    }

    /// @brief Registers a new position and assigns it a unique ID.
    /// @return The newly generated ID.
    size_t set(const Vec3& pos) {
        size_t id = next_id++;
        Positions[id] = pos;
        ƨnoiƚiƨoꟼ[pos] = id;
        return id;
    }

    /// @brief Removes an entry by ID.
    size_t remove(size_t id) {
        Vec3& pos = Positions[id];
        Positions.erase(id);
        ƨnoiƚiƨoꟼ.erase(pos);
        return id;
    }

    /// @brief Removes an entry by Position.
    size_t remove(const Vec3& pos) {
        size_t id = ƨnoiƚiƨoꟼ[pos];
        Positions.erase(id);
        ƨnoiƚiƨoꟼ.erase(pos);
        return id;
    }

    void reserve(size_t size) {
        Positions.reserve(size);
        ƨnoiƚiƨoꟼ.reserve(size);
    }

    size_t size() const {
        return Positions.size();
    }

    size_t getNext_id() {
        return next_id + 1;
    }
    
    size_t bucket_count() {
        return Positions.bucket_count();
    }

    bool empty() const {
        return Positions.empty();
    }
    
    void clear() {
        Positions.clear();
        Positions.rehash(0);
        ƨnoiƚiƨoꟼ.clear();
        ƨnoiƚiƨoꟼ.rehash(0);
        next_id = 0;
    }
    
    using iterator = typename std::unordered_map<size_t, Vec3>::iterator;
    using const_iterator = typename std::unordered_map<size_t, Vec3>::const_iterator;

    iterator begin() { 
        return Positions.begin(); 
    }
    iterator end() { 
        return Positions.end(); 
    }
    const_iterator begin() const { 
        return Positions.begin(); 
    }
    const_iterator end() const { 
        return Positions.end(); 
    }
    const_iterator cbegin() const { 
        return Positions.cbegin(); 
    }
    const_iterator cend() const { 
        return Positions.cend(); 
    }

    bool contains(size_t id) const {
        return (Positions.find(id) != Positions.end());
    }

    bool contains(const Vec3& pos) const {
        return (ƨnoiƚiƨoꟼ.find(pos) != ƨnoiƚiƨoꟼ.end());
    }
};

/// @brief Accelerates spatial queries by bucketizing positions into a grid.
class SpatialGrid3 {
private:
    float cellSize;
public:
    std::unordered_map<Vec3, std::unordered_set<size_t>, Vec3::Hash> grid;
    
    /// @brief Initializes the spatial grid.
    /// @param cellSize The dimension of the spatial buckets. Larger cells mean more items per bucket but fewer buckets.
    SpatialGrid3(float cellSize = 2.0f) : cellSize(cellSize) {}
    
    /// @brief Converts world coordinates to spatial grid coordinates.
    Vec3 worldToGrid(const Vec3& worldPos) const {
        return (worldPos / cellSize).floor();
    }
    
    /// @brief Adds an object ID to the spatial index at the given position.
    void insert(size_t id, const Vec3& pos) {
        Vec3 gridPos = worldToGrid(pos);
        grid[gridPos].insert(id);
    }
    
    /// @brief Removes an object ID from the spatial index.
    void remove(size_t id, const Vec3& pos) {
        Vec3 gridPos = worldToGrid(pos);
        auto cellIt = grid.find(gridPos);
        if (cellIt != grid.end()) {
            cellIt->second.erase(id);
            if (cellIt->second.empty()) {
                grid.erase(cellIt);
            }
        }
    }
    
    /// @brief Moves an object within the spatial index (removes from old cell, adds to new if changed).
    void update(size_t id, const Vec3& oldPos, const Vec3& newPos) {
        Vec3 oldGridPos = worldToGrid(oldPos);
        Vec3 newGridPos = worldToGrid(newPos);
        
        if (oldGridPos != newGridPos) {
            remove(id, oldPos);
            insert(id, newPos);
        }
    }
    
    /// @brief Returns all IDs located in the specific grid cell containing 'center'.
    std::unordered_set<size_t> find(const Vec3& center) const {
        auto cellIt = grid.find(worldToGrid(center));
        if (cellIt != grid.end()) {
            return cellIt->second;
        }
        return std::unordered_set<size_t>();
    }

    /// @brief Finds all object IDs within a square area around the center.
    /// @param center The world position center.
    /// @param radius The search radius (defines the bounds of grid cells to check).
    /// @return A vector of candidate IDs (Note: this returns objects in valid grid cells, further distance checks may be required).
    std::vector<size_t> queryRange(const Vec3& center, float radius) const {
        std::vector<size_t> results;
        float radiusSq = radius * radius;
        
        // Calculate grid bounds for the query
        Vec3 minGrid = worldToGrid(center - Vec3(radius, radius, radius));
        Vec3 maxGrid = worldToGrid(center + Vec3(radius, radius, radius));
        
        size_t estimatedSize = (maxGrid.x - minGrid.x + 1) * (maxGrid.y - minGrid.y + 1) * (maxGrid.z - minGrid.z + 1) * 10;
        results.reserve(estimatedSize);

        // Check all relevant grid cells
        for (int x = minGrid.x; x <= maxGrid.x; ++x) {
            for (int y = minGrid.y; y <= maxGrid.y; ++y) {
                for (int z = minGrid.z; z <= minGrid.z; ++z) {
                    auto cellIt = grid.find(Vec3(x, y, z));
                    if (cellIt != grid.end()) {
                        results.insert(results.end(), cellIt->second.begin(), cellIt->second.end());
                    }
                }
            }
        }
        
        return results;
    }
    
    void clear() {
        grid.clear();
        grid.rehash(0);
    }
};

/// @brief Represents a single point in the grid with an ID, color, and position.
class GenericVoxel {
protected:
    size_t id;
    Vec4 color;
    Vec3 pos;
public:
    //constructors
    GenericVoxel(size_t id, Vec4 color, Vec3 pos) : id(id), color(color), pos(pos) {};

    //getters
    Vec4 getColor() const {
        return color;
    }

    //setters
    void setColor(Vec4 newColor) {
        color = newColor;
    }

    void move(Vec3 newPos) {
        pos = newPos;
    }
    
    void recolor(Vec4 newColor) {
        color.recolor(newColor);
    }
    
};

class Grid3 {
protected:
    //all positions
    reverselookupassistant3 Positions;
    std::unordered_map<size_t, GenericVoxel> Pixels;
    
    std::vector<size_t> unassignedIDs;
    
    float neighborRadius = 1.0f;
    
    //TODO: spatial map
    SpatialGrid3 spatialGrid;
    float spatialCellSize = neighborRadius * 1.5f;

    // Default background color for empty spaces
    Vec4 defaultBackgroundColor = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
    PNoise2 noisegen;

    bool regenpreventer = false;
public:

    Grid3 noiseGenGrid(Vec3 min, Vec3 max, float minChance = 0.1f
                        , float maxChance = 1.0f, bool color = true, int noisemod = 42) {
        TIME_FUNCTION;
        noisegen = PNoise2(noisemod);
        std::cout << "generating a noise grid with the following: "<< min << " by  " << max << "chance min: " << minChance 
                << " max: " << maxChance << " gen colors: " << color << std::endl;
        std::vector<Vec3> poses;
        std::vector<Vec4> colors;
        for (int x = min.x; x < max.x; x++) {
            for (int y = min.y; y < max.y; y++) {
                for (int z = min.z; z < max.z; z++) {
                    float nx = (x+noisemod)/(max.x+EPSILON)/0.1;
                    float ny = (y+noisemod)/(max.y+EPSILON)/0.1;
                    float nz = (z+noisemod)/(max.z+EPSILON)/0.1;
                    Vec3 pos = Vec3(nx,ny,nz);
                    float alpha = noisegen.permute(pos);
                    if (alpha > minChance && alpha < maxChance) {
                        if (color) {
                            float red = noisegen.permute(Vec3(nx*0.3,ny*0.3, nz*0.3));
                            float green = noisegen.permute(Vec3(nx*0.6,ny*0.6, nz*0.6));
                            float blue = noisegen.permute(Vec3(nx*0.9,ny*0.9, nz*0.9));
                            Vec4 newc = Vec4(red,green,blue,1.0);
                            colors.push_back(newc);
                            poses.push_back(Vec3(x,y,z));
                        } else {
                            Vec4 newc = Vec4(alpha,alpha,alpha,1.0);
                            colors.push_back(newc);
                            poses.push_back(Vec3(x,y,z));
                        }
                   }
                }
            }
        }
        std::cout << "noise generated" << std::endl;
        bulkAddObjects(poses,colors);
        return *this;
    }

    size_t addObject(const Vec3& pos, const Vec4& color, float size = 1.0f) {
        size_t id = Positions.set(pos);
        Pixels.emplace(id, GenericVoxel(id, color, pos));
        spatialGrid.insert(id, pos);
        return id;
    }

    /// @brief Sets the default background color.
    void setDefault(const Vec4& color) {
        defaultBackgroundColor = color;
    }
    
    /// @brief Moves an object to a new position and updates spatial indexing.
    void setPosition(size_t id, const Vec3& newPosition) {
        Vec3 oldPosition = Positions.at(id);
        Pixels.at(id).move(newPosition);
        spatialGrid.update(id, oldPosition, newPosition);
        Positions.at(id).move(newPosition);
    }
    
    void setColor(size_t id, const Vec4 color) {
        Pixels.at(id).recolor(color);
    }
    
    void setNeighborRadius(float radius) {
        neighborRadius = radius;
        optimizeSpatialGrid();
    }
    
    Vec4 getDefaultBackgroundColor() const {
        return defaultBackgroundColor;
    }

    Vec3 getPositionID(size_t id) const {
        Vec3 it = Positions.at(id);
        return it;
    }

    size_t getPositionVec(const Vec3& pos, float radius = 0.0f) const {
        TIME_FUNCTION;
        if (radius == 0.0f) {
            // Exact match - use spatial grid to find the cell
            Vec3 gridPos = spatialGrid.worldToGrid(pos);
            auto cellIt = spatialGrid.grid.find(gridPos);
            if (cellIt != spatialGrid.grid.end()) {
                for (size_t id : cellIt->second) {
                    if (Positions.at(id) == pos) {
                        return id;
                    }
                }
            }
            throw std::out_of_range("Position not found");
        } else {
            auto results = getPositionVecRegion(pos, radius);
            if (!results.empty()) {
                return results[0]; // Return first found
            }
            throw std::out_of_range("No positions found in radius");
        }
    }

    size_t getOrCreatePositionVec(const Vec3& pos, float radius = 0.0f, bool create = true) {
        //TIME_FUNCTION; //called too many times and average time is less than 0.0000001 so ignore it.
        if (radius == 0.0f) {
            Vec3 gridPos = spatialGrid.worldToGrid(pos);
            auto cellIt = spatialGrid.grid.find(gridPos);
            if (cellIt != spatialGrid.grid.end()) {
                for (size_t id : cellIt->second) {
                    if (Positions.at(id) == pos) {
                        return id;
                    }
                }
            }
            if (create) {
                return addObject(pos, defaultBackgroundColor, 1.0f);
            }
            throw std::out_of_range("Position not found");
        } else {
            auto results = getPositionVecRegion(pos, radius);
            if (!results.empty()) {
                return results[0];
            }
            if (create) {
                return addObject(pos, defaultBackgroundColor, 1.0f);
            }
            throw std::out_of_range("No positions found in radius");
        }
    }

    std::vector<size_t> getPositionVecRegion(const Vec3& pos, float radius = 1.0f) const {
        //TIME_FUNCTION;
        float searchRadius = (radius == 0.0f) ? std::numeric_limits<float>::epsilon() : radius;
        
        // Get candidates from spatial grid
        std::vector<size_t> candidates = spatialGrid.queryRange(pos, searchRadius);
        
        // Fine-filter by exact distance
        std::vector<size_t> results;
        float radiusSq = searchRadius * searchRadius;
        
        for (size_t id : candidates) {
            if (Positions.at(id).distanceSquared(pos) <= radiusSq) {
                results.push_back(id);
            }
        }
        
        return results;
    }

    Vec4 getColor(size_t id) {
        return Pixels.at(id).getColor();
    }

    void getBoundingBox(Vec3& minCorner, Vec3& maxCorner) const {
        TIME_FUNCTION;
        if (Positions.empty()) {
            minCorner = Vec3(0, 0, 0);
            maxCorner = Vec3(0, 0, 0);
            return;
        }
        
        // Initialize with first position
        auto it = Positions.begin();
        minCorner = it->second;
        maxCorner = it->second;
        
        // Find min and max coordinates
        //#pragma omp parallel for
        for (const auto& [id, pos] : Positions) {
            minCorner.x = std::min(minCorner.x, pos.x);
            minCorner.y = std::min(minCorner.y, pos.y);
            maxCorner.x = std::max(maxCorner.x, pos.x);
            maxCorner.y = std::max(maxCorner.y, pos.y);
        }
    }

    frame getGridRegionAsFrame(const Vec3& minCorner, const Vec3& maxCorner,
                       Vec3& res, frame::colormap outChannels = frame::colormap::RGB)  {
        TIME_FUNCTION;
        //TODO: need to implement this.
    }

    size_t removeID(size_t id) {
        Vec3 oldPosition = Positions.at(id);
        Positions.remove(id);
        Pixels.erase(id);
        unassignedIDs.push_back(id);
        spatialGrid.remove(id, oldPosition);
        return id;
    }

    void bulkUpdatePositions(const std::unordered_map<size_t, Vec3>& newPositions) {
        TIME_FUNCTION;
        for (const auto& [id, newPos] : newPositions) {
            Vec3 oldPosition = Positions.at(id);
            Positions.at(id).move(newPos);
            Pixels.at(id).move(newPos);
            spatialGrid.update(id, oldPosition, newPos);
        }
    }

    std::vector<size_t> bulkAddObjects(const std::vector<Vec3> poses, std::vector<Vec4> colors) {
        TIME_FUNCTION;
        std::vector<size_t> ids;
        ids.reserve(poses.size());
        
        // Reserve space in maps to avoid rehashing
        if (Positions.bucket_count() < Positions.size() + poses.size()) {
            Positions.reserve(Positions.size() + poses.size());
            Pixels.reserve(Positions.size() + poses.size());
        }
        
        // Batch insertion
        std::vector<size_t> newids;
        for (size_t i = 0; i < poses.size(); ++i) {
            size_t id = Positions.set(poses[i]);
            Pixels.emplace(id, GenericVoxel(id, colors[i], poses[i]));
            spatialGrid.insert(id,poses[i]);
            newids.push_back(id);
        }
        
        shrinkIfNeeded();
        
        return newids;
    }

    void shrinkIfNeeded() {
        //TODO: garbage collector
    }

    void clear() {
        Positions.clear();
        Pixels.clear();
        spatialGrid.clear();
        Pixels.rehash(0);
        defaultBackgroundColor = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    void optimizeSpatialGrid() {
        //std::cout << "optimizeSpatialGrid()" << std::endl;
        spatialCellSize = neighborRadius * neighborRadius;
        spatialGrid = SpatialGrid3(spatialCellSize);
        
        // Rebuild spatial grid
        spatialGrid.clear();
        for (const auto& [id, pos] : Positions) {
            spatialGrid.insert(id, pos);
        }
    }

    std::vector<size_t> getNeighbors(size_t id) const {
        Vec3 pos = Positions.at(id);
        std::vector<size_t> candidates = spatialGrid.queryRange(pos, neighborRadius);
        
        std::vector<size_t> neighbors;
        float radiusSq = neighborRadius * neighborRadius;
        
        for (size_t candidateId : candidates) {
            if (candidateId != id && pos.distanceSquared(Positions.at(candidateId)) <= radiusSq) {
                neighbors.push_back(candidateId);
            }
        }
        
        return neighbors;
    }

    std::vector<size_t> getNeighborsRange(size_t id, float dist) const {
        Vec3 pos = Positions.at(id);
        std::vector<size_t> candidates = spatialGrid.queryRange(pos, neighborRadius);
        
        std::vector<size_t> neighbors;
        float radiusSq = dist * dist;
        
        for (size_t candidateId : candidates) {
            if (candidateId != id && 
                pos.distanceSquared(Positions.at(candidateId)) <= radiusSq) {
                neighbors.push_back(candidateId);
            }
        }
        
        return neighbors;
    }

    Grid3 backfillGrid() {
        Vec3 Min;
        Vec3 Max;
        getBoundingBox(Min, Max);
        std::vector<Vec3> newPos;
        std::vector<Vec4> newColors;
        for (size_t x = Min.x; x < Max.x; x++) {
            for (size_t y = Min.y; y < Max.y; y++) {
                for (size_t z = Min.z; z < Max.z; z++) {
                    Vec3 pos = Vec3(x,y,z);
                    if (Positions.contains(pos)) continue;
                    Vec4 color = defaultBackgroundColor;
                    float size = 0.1;
                    newPos.push_back(pos);
                    newColors.push_back(color);
                }
            }
        }
        bulkAddObjects(newPos, newColors);
        return *this;
    }



};

#endif