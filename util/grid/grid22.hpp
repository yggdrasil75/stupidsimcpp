#include <unordered_map>
#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include <vector>
#ifndef GRID2_HPP
#define GRID2_HPP

class Grid2 { 
private:
    //all positions
    std::unordered_map<size_t, Vec2> Positions;
    //all colors
    std::unordered_map<size_t, Vec4> Colors;
    //all sizes
    std::unordered_map<size_t, float> Sizes;

    std::vector<size_t> unassignedIDs;
    
    //grid min
    Vec2 gridMin;
    //grid max
    Vec2 gridMax;
    //next id
    size_t next_id;

    //TODO: neighbor map
    std::unordered_map<size_t, std::vector<size_t>> neighborMap;
    float neighborRadius = 1.0f; // Default neighbor search radius
    //TODO: spatial map
public:
    //get position from id
    Vec2 getPositionID(size_t id) const {
        auto it = Positions.find(id);
        return it != Positions.end() ? it->second : Vec2();
    }
    //get id from position (optional radius, picks first found. radius of 0 becomes epsilon if none are found)
    size_t getPositionVec(Vec2 pos, float radius = 0.0f) {
        float searchRadius = (radius == 0.0f) ? std::numeric_limits<float>::epsilon() : radius;

        float radiusSq = searchRadius*searchRadius;
        
        for (const auto& pair : Positions) {
            if (pair.second.distanceSquared(pos) <= radiusSq) {
                return pair.first;
            }
        }
        return -1;
    }

    size_t getPositionVec(float x, float y, float radius = 0.0f) { 
        return getPositionVec(Vec2(x,y), radius);
    }
    //get all id in region
    std::vector<size_t> getPositionVecRegion(Vec2 pos, float radius = 1.0f) {
        float searchRadius = (radius == 0.0f) ? std::numeric_limits<float>::epsilon() : radius;

        float radiusSq = searchRadius*searchRadius;
        std::vector<size_t> posvec;
        for (const auto& pair : Positions) {
            if (pair.second.distanceSquared(pos) <= radiusSq) {
                posvec.push_back(pair.first);
            }
        }
        return posvec;
    }
    //get color from id
    Vec4 getColor(size_t id) {
        return Colors.at(id);
    }
    //get color from position (use get id from position and then get color from id)
    Vec4 getColor(float x, float y) {
        size_t id = getPositionVec(Vec2(x,y),0.0);
        return getColor(id);
    }
    //get size from id
    Vec4 getSize(size_t id) {
        return Colors.at(id);
    }
    //get size from position (use get id from position and then get size from id)
    Vec4 getSize(float x, float y) {
        size_t id = getPositionVec(Vec2(x,y),0.0);
        return getSize(id);
    }
    
    //add pixel (default color and default size provided)
    size_t addObject(const Vec2& pos, const Vec4& color, float size = 1.0f) {
        size_t id = next_id++;
        Positions[id] = pos;
        Colors[id] = color;
        Sizes[id] = size;
        return id;
        updateNeighborForID(id);
    }
    //set position by id
    void setPosition(size_t id, const Vec2& position) {
        Positions.at(id).move(position);
        updateNeighborForID(id);
    }
    void setPosition(size_t id, float x, float y) {
        Positions.at(id).move(Vec2(x,y));
        updateNeighborForID(id);
    }
    //set color by id (by pos same as get color)
    void setColor(size_t id, const Vec4 color) {
        Colors.at(id).recolor(color);
    }
    void setColor(size_t id, float r, float g, float b, float a=1.0f) {
        Colors.at(id).recolor(Vec4(r,g,b,a));
    }
    void setColor(float x, float y, const Vec4 color) {
        size_t id = getPositionVec(Vec2(x,y));
        Colors.at(id).recolor(color);
    }
    void setColor(float x, float y, float r, float g, float b, float a=1.0f) {
        size_t id = getPositionVec(Vec2(x,y));
        Colors.at(id).recolor(Vec4(r,g,b,a));
    }
    void setColor(const Vec2& pos, const Vec4 color) {
        size_t id = getPositionVec(pos);
        Colors.at(id).recolor(color);
    }
    void setColor(const Vec2& pos, float r, float g, float b, float a=1.0f) {
        size_t id = getPositionVec(pos);
        Colors.at(id).recolor(Vec4(r,g,b,a));
    }
    //set size by id (by pos same as get size)
    void setSize(size_t id, float size) {
        Sizes.at(id) = size;
    }
    void setSize(float x, float y, float size) {
        size_t id = getPositionVec(Vec2(x,y));
        Sizes.at(id) = size;
    }
    void setSize(const Vec2& pos, float size) {
        size_t id = getPositionVec(pos);
        Sizes.at(id) = size;
    }

    //remove object (should remove the id, the color, the position, and the size)
    size_t removeID(size_t id) {
        Positions.erase(id);
        Colors.erase(id);
        Sizes.erase(id);
        unassignedIDs.push_back(id);
        updateNeighborForID(id);
        return id;
    }
    size_t removeID(Vec2 pos) {
        size_t id = getPositionVec(pos);
        Positions.erase(id);
        Colors.erase(id);
        Sizes.erase(id);
        unassignedIDs.push_back(id);
        updateNeighborForID(id);
        return id;
    }

    //bulk update positions
    void bulkUpdatePositions(const std::unordered_map<size_t, Vec2>& newPositions) {
        TIME_FUNCTION;
        for (const auto& [id, newPos] : newPositions) {
            auto it = Positions.find(id);
            if (it != Positions.end()) {
                it->second = newPos;
            }
        }
        updateNeighborMap();
    }
    // Bulk update colors
    void bulkUpdateColors(const std::unordered_map<size_t, Vec4>& newColors) {
        TIME_FUNCTION;
        for (const auto& [id, newColor] : newColors) {
            auto it = Colors.find(id);
            if (it != Colors.end()) {
                it->second = newColor;
            }
        }
    }
    // Bulk update sizes
    void bulkUpdateSizes(const std::unordered_map<size_t, float>& newSizes) {
        TIME_FUNCTION;
        for (const auto& [id, newSize] : newSizes) {
            auto it = Sizes.find(id);
            if (it != Sizes.end()) {
                it->second = newSize;
            }
        }
    }
    //bulk add
    std::vector<size_t> bulkAddObjects(const std::vector<std::tuple<Vec2, Vec4, float>>& objects) {
        TIME_FUNCTION;
        std::vector<size_t> ids;
        ids.reserve(objects.size());
        
        // Reserve space in maps to avoid rehashing
        if (Positions.bucket_count() < Positions.size() + objects.size()) {
            Positions.reserve(Positions.size() + objects.size());
            Colors.reserve(Colors.size() + objects.size());
            Sizes.reserve(Sizes.size() + objects.size());
        }
        
        // Batch insertion
        #pragma omp parallel for
        for (size_t i = 0; i < objects.size(); ++i) {
            size_t id = next_id + i;
            const auto& [pos, color, size] = objects[i];
            
            Positions[id] = pos;
            Colors[id] = color;
            Sizes[id] = size;
        }
        
        // Update next_id atomically
        next_id += objects.size();
        return getAllIDs(); // Or generate ID range
    }
    
    std::vector<size_t> bulkAddObjects(const std::vector<Vec2> poses, std::vector<Vec4> colors, std::vector<float>& sizes) {
        TIME_FUNCTION;
        std::vector<size_t> ids;
        ids.reserve(poses.size());
        
        // Reserve space in maps to avoid rehashing
        if (Positions.bucket_count() < Positions.size() + poses.size()) {
            Positions.reserve(Positions.size() + poses.size());
            Colors.reserve(Colors.size() + colors.size());
            Sizes.reserve(Sizes.size() + sizes.size());
        }
        
        // Batch insertion
        #pragma omp parallel for
        for (size_t i = 0; i < poses.size(); ++i) {
            size_t id = next_id + i;
            
            Positions[id] = poses[i];
            Colors[id] = colors[i];
            Sizes[id] = sizes[i];
        }
        
        // Update next_id atomically
        next_id += poses.size();
        return getAllIDs(); // Or generate ID range
    }
    //get all ids
    std::vector<size_t> getAllIDs() const {
        TIME_FUNCTION;
        std::vector<size_t> ids;
        ids.reserve(Positions.size());
        
        for (const auto& pair : Positions) {
            ids.push_back(pair.first);
        }
        
        return ids;
    }

    // no return because it passes back a 1d vector of ints between 0 and 255 with a width and height
    //get region as rgb 
    void getGridRegionAsRGB(const Vec2& minCorner, const Vec2& maxCorner,
                           int& width, int& height, std::vector<uint8_t>& rgbData) const {
        TIME_FUNCTION;
        // Calculate dimensions
        width = static_cast<int>(maxCorner.x - minCorner.x);
        height = static_cast<int>(maxCorner.y - minCorner.y);
        
        if (width <= 0 || height <= 0) {
            width = height = 0;
            rgbData.clear();
            return;
        }
        
        // Initialize RGB data (3 bytes per pixel: R, G, B)
        rgbData.resize(width * height * 3, 0);
        
        // For each position in the grid, find the corresponding pixel
        for (const auto& [id, pos] : Positions) {
            if (pos.x >= minCorner.x && pos.x < maxCorner.x &&
                pos.y >= minCorner.y && pos.y < maxCorner.y) {
                
                // Calculate pixel coordinates
                int pixelX = static_cast<int>(pos.x - minCorner.x);
                int pixelY = static_cast<int>(pos.y - minCorner.y);
                
                // Ensure within bounds
                if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {
                    // Get color and convert to RGB
                    const Vec4& color = Colors.at(id);
                    int index = (pixelY * width + pixelX) * 3;
                    
                    // Convert from [0,1] to [0,255] and store as RGB
                    rgbData[index]     = static_cast<unsigned char>(color.r * 255);
                    rgbData[index + 1] = static_cast<unsigned char>(color.g * 255);
                    rgbData[index + 2] = static_cast<unsigned char>(color.b * 255);
                }
            }
        }
    }

    // Get region as BGR
    void getGridRegionAsBGR(const Vec2& minCorner, const Vec2& maxCorner,
                           int& width, int& height, std::vector<uint8_t>& bgrData) const {
        TIME_FUNCTION;
        // Calculate dimensions
        width = static_cast<int>(maxCorner.x - minCorner.x);
        height = static_cast<int>(maxCorner.y - minCorner.y);
        
        if (width <= 0 || height <= 0) {
            width = height = 0;
            bgrData.clear();
            return;
        }
        
        // Initialize BGR data (3 bytes per pixel: B, G, R)
        bgrData.resize(width * height * 3, 0);
        
        // For each position in the grid, find the corresponding pixel
        for (const auto& [id, pos] : Positions) {
            if (pos.x >= minCorner.x && pos.x < maxCorner.x &&
                pos.y >= minCorner.y && pos.y < maxCorner.y) {
                
                // Calculate pixel coordinates
                int pixelX = static_cast<int>(pos.x - minCorner.x);
                int pixelY = static_cast<int>(pos.y - minCorner.y);
                
                // Ensure within bounds
                if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {
                    // Get color and convert to BGR
                    const Vec4& color = Colors.at(id);
                    int index = (pixelY * width + pixelX) * 3;
                    
                    // Convert from [0,1] to [0,255] and store as BGR
                    bgrData[index]     = static_cast<unsigned char>(color.b * 255);  // Blue
                    bgrData[index + 1] = static_cast<unsigned char>(color.g * 255);  // Green
                    bgrData[index + 2] = static_cast<unsigned char>(color.r * 255);  // Red
                }
            }
        }
    }
    //get full as rgb/bgr
    void getGridAsRGB(int& width, int& height, std::vector<uint8_t>& rgbData) const {
        Vec2 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        getGridRegionAsRGB(minCorner, maxCorner, width, height, rgbData);
    }

    void getGridAsBGR(int& width, int& height, std::vector<uint8_t>& bgrData) const {
        Vec2 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        getGridRegionAsBGR(minCorner, maxCorner, width, height, bgrData);
    }

    //get bounding box
    void getBoundingBox(Vec2& minCorner, Vec2& maxCorner) const {
        TIME_FUNCTION;
        if (Positions.empty()) {
            minCorner = Vec2(0, 0);
            maxCorner = Vec2(0, 0);
            return;
        }
        
        // Initialize with first position
        auto it = Positions.begin();
        minCorner = it->second;
        maxCorner = it->second;
        
        // Find min and max coordinates
        for (const auto& [id, pos] : Positions) {
            minCorner.x = std::min(minCorner.x, pos.x);
            minCorner.y = std::min(minCorner.y, pos.y);
            maxCorner.x = std::max(maxCorner.x, pos.x);
            maxCorner.y = std::max(maxCorner.y, pos.y);
        }
        
        // Add a small margin to avoid edge cases
        float margin = 1.0f;
        minCorner.x -= margin;
        minCorner.y -= margin;
        maxCorner.x += margin;
        maxCorner.y += margin;
    }

    //clear
    void clear() {
        Positions.clear();
        Colors.clear();
        Sizes.clear();
        next_id = 0;
    }

    // neighbor map
    void updateNeighborMap() {
        TIME_FUNCTION;
        neighborMap.clear();
        
        // For each object, find nearby neighbors
        for (const auto& [id1, pos1] : Positions) {
            std::vector<size_t> neighbors;
            float radiusSq = neighborRadius * neighborRadius;
            
            for (const auto& [id2, pos2] : Positions) {
                if (id1 != id2 && pos1.distanceSquared(pos2) <= radiusSq) {
                    neighbors.push_back(id2);
                }
            }
            neighborMap[id1] = std::move(neighbors);
        }
    }
    
    // Update neighbor map for a single object (more efficient)
    void updateNeighborForID(size_t id) {
        TIME_FUNCTION;
        auto pos_it = Positions.find(id);
        if (pos_it == Positions.end()) return;
        
        Vec2 pos1 = pos_it->second;
        std::vector<size_t> neighbors;
        float radiusSq = neighborRadius * neighborRadius;
        
        for (const auto& [id2, pos2] : Positions) {
            if (id != id2 && pos1.distanceSquared(pos2) <= radiusSq) {
                neighbors.push_back(id2);
            }
        }
        neighborMap[id] = std::move(neighbors);
    }
    
    // Get neighbors for an ID
    const std::vector<size_t>& getNeighbors(size_t id) const {
        static const std::vector<size_t> empty;
        auto it = neighborMap.find(id);
        return it != neighborMap.end() ? it->second : empty;
    }
    
    // Set neighbor search radius
    void setNeighborRadius(float radius) {
        neighborRadius = radius;
        updateNeighborMap(); // Recompute all neighbors
    }
    // spatial map
};

#endif