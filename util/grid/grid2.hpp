#ifndef GRID2_HPP
#define GRID2_HPP

#include <unordered_map>
#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "../noise/pnoise2.hpp"
#include "../simblocks/water.hpp"
#include <vector>
#include <unordered_set>

const float EPSILON = 0.0000000000000000000000001;

class reverselookupassistantclasscausecppisdumb {
private:
    std::unordered_map<size_t, Vec2> Positions;
    std::unordered_map<Vec2, size_t, Vec2::Hash> ƨnoiƚiƨoꟼ;
    size_t next_id;
public:
    Vec2 at(size_t id) const {
        auto it = Positions.at(id);
        return it;
    }

    size_t at(const Vec2& pos) const {
        size_t id = ƨnoiƚiƨoꟼ.at(pos);
        return id;
    }

    Vec2 find(size_t id) {
        return Positions.at(id);
    }

    size_t set(const Vec2& pos) {
        size_t id = next_id++;
        Positions[id] = pos;
        ƨnoiƚiƨoꟼ[pos] = id;
        return id;
    }

    size_t remove(size_t id) {
        Vec2& pos = Positions[id];
        Positions.erase(id);
        ƨnoiƚiƨoꟼ.erase(pos);
        return id;
    }

    size_t remove(const Vec2& pos) {
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
    
    using iterator = typename std::unordered_map<size_t, Vec2>::iterator;
    using const_iterator = typename std::unordered_map<size_t, Vec2>::const_iterator;

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
    
};

class SpatialGrid {
private:
    float cellSize;
public:
    std::unordered_map<Vec2, std::unordered_set<size_t>, Vec2::Hash> grid;
    SpatialGrid(float cellSize = 2.0f) : cellSize(cellSize) {}
    
    Vec2 worldToGrid(const Vec2& worldPos) const {
        return (worldPos / cellSize).floor();
    }
    
    void insert(size_t id, const Vec2& pos) {
        Vec2 gridPos = worldToGrid(pos);
        grid[gridPos].insert(id);
    }
    
    void remove(size_t id, const Vec2& pos) {
        Vec2 gridPos = worldToGrid(pos);
        auto cellIt = grid.find(gridPos);
        if (cellIt != grid.end()) {
            cellIt->second.erase(id);
            if (cellIt->second.empty()) {
                grid.erase(cellIt);
            }
        }
    }
    
    void update(size_t id, const Vec2& oldPos, const Vec2& newPos) {
        Vec2 oldGridPos = worldToGrid(oldPos);
        Vec2 newGridPos = worldToGrid(newPos);
        
        if (oldGridPos != newGridPos) {
            remove(id, oldPos);
            insert(id, newPos);
        }
    }
    
    std::vector<size_t> queryRange(const Vec2& center, float radius) const {
        std::vector<size_t> results;
        float radiusSq = radius * radius;
        
        // Calculate grid bounds for the query
        Vec2 minGrid = worldToGrid(center - Vec2(radius, radius));
        Vec2 maxGrid = worldToGrid(center + Vec2(radius, radius));
        
        // Check all relevant grid cells
        //#pragma omp parallel for
        for (int x = minGrid.x; x <= maxGrid.x; ++x) {
            for (int y = minGrid.y; y <= maxGrid.y; ++y) {
                Vec2 gridPos(x, y);
                auto cellIt = grid.find(gridPos);
                if (cellIt != grid.end()) {
                    for (size_t id : cellIt->second) {
                        results.push_back(id);
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

class Grid2 { 
protected:
    //all positions
    reverselookupassistantclasscausecppisdumb Positions;
    //all colors
    std::unordered_map<size_t, Vec4> Colors;
    //all sizes
    std::unordered_map<size_t, float> Sizes;

    std::vector<size_t> unassignedIDs;
    
    //grid min
    Vec2 gridMin;
    //grid max
    Vec2 gridMax;

    //neighbor map
    std::unordered_map<size_t, std::vector<size_t>> neighborMap;
    float neighborRadius = 1.0f;
    
    //TODO: spatial map
    SpatialGrid spatialGrid;
    float spatialCellSize = 2.0f;

    // Default background color for empty spaces
    Vec4 defaultBackgroundColor = Vec4(0.0f, 0.0f, 0.0f, 0.0f);

    PNoise2 noisegen;


    //water
    std::unordered_map<size_t, WaterParticle> water;
public:
    bool usable = false;
    
    // Set default background color for empty spaces
    void setDefault(const Vec4& color) {
        defaultBackgroundColor = color;
    }
    
    void setDefault(float r, float g, float b, float a = 0.0f) {
        defaultBackgroundColor = Vec4(r, g, b, a);
    }
    
    // Get current default background color
    Vec4 getDefaultBackgroundColor() const {
        return defaultBackgroundColor;
    }

    //get position from id
    Vec2 getPositionID(size_t id) const {
        Vec2 it = Positions.at(id);
        return it;
    }

    Grid2 noiseGenGrid(size_t minx,size_t miny, size_t maxx, size_t maxy, float minChance = 0.1f
                        , float maxChance = 1.0f, bool color = true, int noisemod = 42) {
        TIME_FUNCTION;
        std::cout << "generating a noise grid with the following: (" << minx << ", " << miny 
                << ") by (" << maxx << ", " << maxy << ") " << "chance: " << minChance 
                << " max: " << maxChance << " gen colors: " << color << std::endl;
        std::vector<Vec2> poses;
        std::vector<Vec4> colors;
        std::vector<float> sizes;
        for (int x = minx; x < maxx; x++) {
            for (int y = miny; y < maxy; y++) {
                float nx = (x+noisemod)/(maxx+EPSILON)/0.1;
                float ny = (y+noisemod)/(maxy+EPSILON)/0.1;
                Vec2 pos = Vec2(nx,ny);
                float alpha = noisegen.permute(pos);
                if (alpha > minChance && alpha < maxChance) {
                    if (color) {
                        // float red = noisegen.noise(x,y,1000);
                        // float green = noisegen.noise(x,y,2000);
                        // float blue = noisegen.noise(x,y,3000);
                        float red = noisegen.permute(Vec2(nx*0.3,ny*0.3));
                        float green = noisegen.permute(Vec2(nx*0.6,ny*.06));
                        float blue = noisegen.permute(Vec2(nx*0.9,ny*0.9));
                        Vec4 newc = Vec4(red,green,blue,1.0);
                        colors.push_back(newc);
                        poses.push_back(Vec2(x,y));
                        sizes.push_back(1.0f);
                    } else {
                        Vec4 newc = Vec4(alpha,alpha,alpha,1.0);
                        colors.push_back(newc);
                        poses.push_back(pos);
                        sizes.push_back(1.0f);
                    }
                }
            }
        }
        std::cout << "noise generated" << std::endl;
        bulkAddObjects(poses,colors,sizes);
        return *this;
    }

    size_t NoiseGenPointB(const Vec2& pos) {
        float grayc = noisegen.permute(pos);
        Vec4 newc = Vec4(grayc,grayc,grayc,grayc);
        return addObject(pos,newc,1.0);
    }
    
    size_t NoiseGenPointRGB(const Vec2& pos) {
        float red = noisegen.permute(pos);
        float green = noisegen.permute(pos);
        float blue = noisegen.permute(pos);
        Vec4 newc = Vec4(red,green,blue,1);
        return addObject(pos,newc,1.0);
    }

    size_t NoiseGenPointRGBA(const Vec2& pos) {
        float red = noisegen.permute(pos);
        float green = noisegen.permute(pos);
        float blue = noisegen.permute(pos);
        float alpha = noisegen.permute(pos);
        Vec4 newc = Vec4(red,green,blue,alpha);
        return addObject(pos,newc,1.0);
    }

    //get id from position (optional radius, picks first found. radius of 0 becomes epsilon if none are found)
    size_t getPositionVec(const Vec2& pos, float radius = 0.0f) const {
        TIME_FUNCTION;
        if (radius == 0.0f) {
            // Exact match - use spatial grid to find the cell
            Vec2 gridPos = spatialGrid.worldToGrid(pos);
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

    size_t getOrCreatePositionVec(const Vec2& pos, float radius = 0.0f, bool create = false) {
        TIME_FUNCTION;
        if (radius == 0.0f) {
            // Exact match - use spatial grid to find the cell
            Vec2 gridPos = spatialGrid.worldToGrid(pos);
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
                return results[0]; // Return first found
            }
            if (create) {
                return addObject(pos, defaultBackgroundColor, 1.0f);
            }
            throw std::out_of_range("No positions found in radius");
        }
    }

    size_t getPositionVec(float x, float y, float radius = 0.0f) const { 
        return getPositionVec(Vec2(x,y), radius);
    }

    //get all id in region
    std::vector<size_t> getPositionVecRegion(const Vec2& pos, float radius = 1.0f) const {
        TIME_FUNCTION;
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
    size_t getSize(size_t id) {
        return Sizes.at(id);
    }
    
    //get size from position (use get id from position and then get size from id)
    size_t getSize(float x, float y) {
        size_t id = getPositionVec(Vec2(x,y),0.0);
        return getSize(id);
    }
    
    //add pixel (default color and default size provided)
    size_t addObject(const Vec2& pos, const Vec4& color, float size = 1.0f) {
        size_t id = Positions.set(pos);
        Colors[id] = color;
        Sizes[id] = size;
        
        // Add to spatial grid
        spatialGrid.insert(id, pos);
        updateNeighborForID(id);
        return id;
    }
    
    //set position by id
    void setPosition(size_t id, const Vec2& newPosition) {
        Vec2 oldPosition = Positions.at(id);
        spatialGrid.update(id, oldPosition, newPosition);
        Positions.at(id).move(newPosition);
        updateNeighborForID(id);
    }
    
    void setPosition(size_t id, float x, float y) {
        Vec2 newPos = Vec2(x,y);
        Vec2 oldPos = Positions.at(id);

        spatialGrid.update(id, oldPos, newPos);
        Positions.at(id).move(newPos);
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
        Vec2 oldPosition = Positions.at(id);
        Positions.remove(id);
        Colors.erase(id);
        Sizes.erase(id);
        unassignedIDs.push_back(id);
        spatialGrid.remove(id, oldPosition);
        updateNeighborForID(id);
        return id;
    }
    
    size_t removeID(Vec2 pos) {
        size_t id = getPositionVec(pos);
        Positions.remove(id);
        Colors.erase(id);
        Sizes.erase(id);
        unassignedIDs.push_back(id);
        spatialGrid.remove(id, pos);
        updateNeighborForID(id);
        return id;
    }

    //bulk update positions
    void bulkUpdatePositions(const std::unordered_map<size_t, Vec2>& newPositions) {
        TIME_FUNCTION;
        //#pragma omp parallel for
        for (const auto& [id, newPos] : newPositions) {
            Vec2 oldPosition = Positions.at(id);
            Positions.at(id).move(newPos);
            spatialGrid.update(id, oldPosition, newPos);
        }
        updateNeighborMap();
    }
    
    // Bulk update colors
    void bulkUpdateColors(const std::unordered_map<size_t, Vec4>& newColors) {
        TIME_FUNCTION;
        //#pragma omp parallel for
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
        //#pragma omp parallel for
        for (const auto& [id, newSize] : newSizes) {
            auto it = Sizes.find(id);
            if (it != Sizes.end()) {
                it->second = newSize;
            }
        }
    }

    void shrinkIfNeeded() {
        //TODO: cleanup all as needed.
    }
    
    //bulk add
    std::vector<size_t> bulkAddObjects(const std::vector<std::tuple<Vec2, Vec4, float>>& objects) {
        TIME_FUNCTION;
        std::vector<size_t> ids;
        ids.reserve(objects.size());
        
        // Reserve space in maps to avoid rehashing
        Positions.reserve(Positions.size() + objects.size());
        Colors.reserve(Colors.size() + objects.size());
        Sizes.reserve(Sizes.size() + objects.size());
        
        // Batch insertion
        //#pragma omp parallel for
        for (size_t i = 0; i < objects.size(); ++i) {
            const auto& [pos, color, size] = objects[i];
            size_t id = Positions.set(pos);
            
            Colors[id] = color;
            Sizes[id] = size;
            spatialGrid.insert(id,pos);
        }
        
        shrinkIfNeeded();
        updateNeighborMap();
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
        std::vector<size_t> newids;
        for (size_t i = 0; i < poses.size(); ++i) {
            size_t id = Positions.set(poses[i]);
            Colors[id] = colors[i];
            Sizes[id] = sizes[i];
            spatialGrid.insert(id,poses[i]);
            newids.push_back(id);
        }
        
        shrinkIfNeeded();
        std::cout << "shrunk. " << std::endl;
        updateNeighborMap();
        std::cout << "neighbormap updated. " << std::endl;
        
        usable = true;
        return newids;
    }

    //get all ids
    std::vector<size_t> getAllIDs() {
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
            rgbData.shrink_to_fit();
            return;
        }
        
        // Initialize RGB data with default background color
        std::vector<Vec4> rgbaBuffer(width * height, Vec4(0,0,0,0));
        
        // for (int x = minCorner.x; x < maxCorner.x; x++) {
        //     for (int y = minCorner.y; x < maxCorner.y; y++){
        //         Vec2 pos = Vec2(x,y);
        //         size_t posID = getPositionVec(pos, 1.0f, false);
        
        //     }
        // }
        // For each position in the grid, find the corresponding pixel
        for (const auto& [id, pos] : Positions) {
            size_t size = Sizes.at(id);
            
            // Calculate pixel coordinates
            int pixelXm = static_cast<int>(pos.x - size/2 - minCorner.x);
            int pixelXM = static_cast<int>(pos.x + size/2 - minCorner.x);
            int pixelYm = static_cast<int>(pos.y - size/2 - minCorner.y);
            int pixelYM = static_cast<int>(pos.y + size/2 - minCorner.y);
            
            pixelXm = std::max(0, pixelXm);
            pixelXM = std::min(width - 1, pixelXM);
            pixelYm = std::max(0, pixelYm);
            pixelYM = std::min(height - 1, pixelYM);
            
            // Ensure within bounds
            if (pixelXM >= minCorner.x && pixelXm < width && pixelYM >= minCorner.y && pixelYm < height) {
                const Vec4& color = Colors.at(id);
                float srcAlpha = color.a;
                float invSrcAlpha = 1.0f - srcAlpha;
                for (int py = pixelYm; py <= pixelYM; ++py){
                    for (int px = pixelXm; px <= pixelXM; ++px){
                        int index = (py * width + px);
                        Vec4 dest = rgbaBuffer[index];
                    
                        // Alpha blending: new_color = src * src_alpha + dest * (1 - src_alpha)
                        dest.r = color.r * srcAlpha + dest.r * invSrcAlpha;
                        dest.g = color.g * srcAlpha + dest.g * invSrcAlpha;
                        dest.b = color.b * srcAlpha + dest.b * invSrcAlpha;
                        dest.a = srcAlpha + dest.a * invSrcAlpha;
                        rgbaBuffer[index] = dest;
                    }
                }
            }
        }
        
        // Convert to RGB bytes
        rgbData.resize(rgbaBuffer.size() * 3);
        for (int i = 0; i < rgbaBuffer.size(); ++i) {
            Vec4& color = rgbaBuffer[i];
            int rgbIndex = i * 3;
            float alpha = color.a;

            if (alpha < 1.0) {
                float invalpha = 1.0 - alpha;
                color.r = defaultBackgroundColor.r * alpha + color.r * invalpha;
                color.g = defaultBackgroundColor.g * alpha + color.g * invalpha;
                color.b = defaultBackgroundColor.b * alpha + color.b * invalpha;
            }
            
            // Convert from [0,1] to [0,255] and store as RGB
            rgbData[rgbIndex + 0] = static_cast<unsigned char>(color.r * 255);
            rgbData[rgbIndex + 1] = static_cast<unsigned char>(color.g * 255);
            rgbData[rgbIndex + 2] = static_cast<unsigned char>(color.b * 255);
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
            bgrData.shrink_to_fit();
            return;
        }
        
        // Initialize RGB data with default background color
        std::vector<Vec4> rgbaBuffer(width * height, defaultBackgroundColor);
        
        // For each position in the grid, find the corresponding pixel
        for (const auto& [id, pos] : Positions) {
            size_t size = Sizes.at(id);
                
            // Calculate pixel coordinates
            int pixelXm = static_cast<int>(pos.x - size/2 - minCorner.x);
            int pixelXM = static_cast<int>(pos.x + size/2 - minCorner.x);
            int pixelYm = static_cast<int>(pos.y - size/2 - minCorner.y);
            int pixelYM = static_cast<int>(pos.y + size/2 - minCorner.y);
            
            pixelXm = std::max(0, pixelXm);
            pixelXM = std::min(width - 1, pixelXM);
            pixelYm = std::max(0, pixelYm);
            pixelYM = std::min(height - 1, pixelYM);

            // Ensure within bounds
            if (pixelXM >= minCorner.x && pixelXm < width && pixelYM >= minCorner.y && pixelYm < height) {
                const Vec4& color = Colors.at(id);
                float srcAlpha = color.a;
                float invSrcAlpha = 1.0f - srcAlpha;
                for (int py = pixelYm; py <= pixelYM; ++py){
                    for (int px = pixelXm; px <= pixelXM; ++px){
                        int index = (py * width + px);
                        Vec4 dest = rgbaBuffer[index];
                        
                        // Alpha blending: new_color = src * src_alpha + dest * (1 - src_alpha)
                        dest.r = color.r * srcAlpha + dest.r * invSrcAlpha;
                        dest.g = color.g * srcAlpha + dest.g * invSrcAlpha;
                        dest.b = color.b * srcAlpha + dest.b * invSrcAlpha;
                        dest.a = srcAlpha + dest.a * invSrcAlpha;
                        rgbaBuffer[index] = dest;
                    }
                }
            }
        }
        
        // Convert to BGR bytes
        bgrData.resize(rgbaBuffer.size() * 3);
        for (int i = 0; i < rgbaBuffer.size(); ++i) {
            const Vec4& color = rgbaBuffer[i];
            int bgrIndex = i * 3;
            
            // Convert from [0,1] to [0,255] and store as BGR
            bgrData[bgrIndex + 2] = static_cast<unsigned char>(color.r * 255);
            bgrData[bgrIndex + 1] = static_cast<unsigned char>(color.g * 255);
            bgrData[bgrIndex + 0] = static_cast<unsigned char>(color.b * 255);
        }
    }

    
    void getGridRegionAsRGBA(const Vec2& minCorner, const Vec2& maxCorner,
                           int& width, int& height, std::vector<uint8_t>& rgbaData) const {
        TIME_FUNCTION;
        // Calculate dimensions
        width = static_cast<int>(maxCorner.x - minCorner.x);
        height = static_cast<int>(maxCorner.y - minCorner.y);
        
        if (width <= 0 || height <= 0) {
            width = height = 0;
            rgbaData.clear();
            rgbaData.shrink_to_fit();
            return;
        }
        
        // Initialize RGBA data with default background color
        std::vector<Vec4> rgbaBuffer(width * height, defaultBackgroundColor);
        
        // For each position in the grid, find the corresponding pixel
        for (const auto& [id, pos] : Positions) {
            size_t size = Sizes.at(id);
                
            // Calculate pixel coordinates
            int pixelXm = static_cast<int>(pos.x - size/2 - minCorner.x);
            int pixelXM = static_cast<int>(pos.x + size/2 - minCorner.x);
            int pixelYm = static_cast<int>(pos.y - size/2 - minCorner.y);
            int pixelYM = static_cast<int>(pos.y + size/2 - minCorner.y);
            
            pixelXm = std::max(0, pixelXm);
            pixelXM = std::min(width - 1, pixelXM);
            pixelYm = std::max(0, pixelYm);
            pixelYM = std::min(height - 1, pixelYM);

            // Ensure within bounds
            if (pixelXM >= minCorner.x && pixelXm < width && pixelYM >= minCorner.y && pixelYm < height) {
                const Vec4& color = Colors.at(id);
                float srcAlpha = color.a;
                float invSrcAlpha = 1.0f - srcAlpha;
                for (int py = pixelYm; py <= pixelYM; ++py){
                    for (int px = pixelXm; px <= pixelXM; ++px){
                        int index = (py * width + px);
                        Vec4 dest = rgbaBuffer[index];
                        
                        // Alpha blending: new_color = src * src_alpha + dest * (1 - src_alpha)
                        dest.r = color.r * srcAlpha + dest.r * invSrcAlpha;
                        dest.g = color.g * srcAlpha + dest.g * invSrcAlpha;
                        dest.b = color.b * srcAlpha + dest.b * invSrcAlpha;
                        dest.a = srcAlpha + dest.a * invSrcAlpha;
                        rgbaBuffer[index] = dest;
                    }
                }
            }
        }
        
        // Convert to RGBA bytes
        rgbaData.resize(rgbaBuffer.size() * 4);
        for (int i = 0; i < rgbaBuffer.size(); ++i) {
            const Vec4& color = rgbaBuffer[i];
            int rgbaIndex = i * 4;
            
            // Convert from [0,1] to [0,255] and store as RGBA
            rgbaData[rgbaIndex + 0] = static_cast<unsigned char>(color.r * 255);
            rgbaData[rgbaIndex + 1] = static_cast<unsigned char>(color.g * 255);
            rgbaData[rgbaIndex + 2] = static_cast<unsigned char>(color.b * 255);
            rgbaData[rgbaIndex + 3] = static_cast<unsigned char>(color.a * 255);
        }
    }
    
    //get full as rgb/bgr
    void getGridAsRGB(int& width, int& height, std::vector<uint8_t>& rgbData)  {
        Vec2 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        getGridRegionAsRGB(minCorner, maxCorner, width, height, rgbData);
    }

    void getGridAsBGR(int& width, int& height, std::vector<uint8_t>& bgrData)  {
        Vec2 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        getGridRegionAsBGR(minCorner, maxCorner, width, height, bgrData);
    }
      
    //frame stuff
    frame getGridRegionAsFrameRGB(const Vec2& minCorner, const Vec2& maxCorner) const {
        TIME_FUNCTION;
        int width, height;
        std::vector<uint8_t> rgbData;
        getGridRegionAsRGB(minCorner, maxCorner, width, height, rgbData);
        
        frame resultFrame(width, height, frame::colormap::RGB);
        resultFrame.setData(rgbData);
        return resultFrame;
    }
    
    frame getGridRegionAsFrameRGBA(const Vec2& minCorner, const Vec2& maxCorner) const {
        TIME_FUNCTION;
        int width, height;
        std::vector<uint8_t> rgbaData;
        getGridRegionAsRGBA(minCorner, maxCorner, width, height, rgbaData);
        
        frame resultFrame(width, height, frame::colormap::RGBA);
        resultFrame.setData(rgbaData);
        return resultFrame;
    }

    // Get region as frame (BGR format)
    frame getGridRegionAsFrameBGR(const Vec2& minCorner, const Vec2& maxCorner) const {
        TIME_FUNCTION;
        int width, height;
        std::vector<uint8_t> bgrData;
        getGridRegionAsBGR(minCorner, maxCorner, width, height, bgrData);
        
        frame resultFrame(width, height, frame::colormap::BGR);
        resultFrame.setData(bgrData);
        return resultFrame;
    }

    // Get entire grid as frame with specified format
    frame getGridAsFrame(frame::colormap format = frame::colormap::RGB) const {
        TIME_FUNCTION;
        Vec2 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        frame Frame;

        switch (format) {
            case frame::colormap::RGB:
                Frame = std::move(getGridRegionAsFrameRGB(minCorner, maxCorner));
                break;
            case frame::colormap::RGBA:
                Frame = std::move(getGridRegionAsFrameRGBA(minCorner, maxCorner));
                break;
            case frame::colormap::BGR:
                Frame = std::move(getGridRegionAsFrameBGR(minCorner, maxCorner));
                break;
            default:
                Frame = std::move(getGridRegionAsFrameRGB(minCorner, maxCorner));
                break;
        }
        //Frame.compressFrameDiff();
        //Frame.compressFrameRLE();
        //Frame.compressFrameLZ78();
        return Frame;
    }

    // Get compressed frame with specified compression
    frame getGridAsCompressedFrame(frame::colormap format = frame::colormap::RGB,
                                frame::compresstype compression = frame::compresstype::RLE) {
        TIME_FUNCTION;
        frame gridFrame = getGridAsFrame(format);
        
        if (gridFrame.getData().empty()) {
            return gridFrame;
        }
        
        switch (compression) {
            case frame::compresstype::RLE:
                return gridFrame.compressFrameRLE();
            case frame::compresstype::DIFF:
                return gridFrame.compressFrameDiff();
            case frame::compresstype::DIFFRLE:
                return gridFrame.compressFrameDiffRLE();
            case frame::compresstype::HUFFMAN:
                return gridFrame.compressFrameHuffman();
            case frame::compresstype::RAW:
            default:
                return gridFrame;
        }
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
        //#pragma omp parallel for
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
        spatialGrid.clear();
        neighborMap.clear();
        Colors.rehash(0);
        Sizes.rehash(0);
        neighborMap.rehash(0);
        // Reset to default background color
        defaultBackgroundColor = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // neighbor map
    void updateNeighborMap() {
        TIME_FUNCTION;
        neighborMap.clear();
        
        // For each object, find nearby neighbors
        #pragma omp parallel for
        for (const auto& [id1, pos1] : Positions) {
            std::vector<size_t> neighbors;
            float radiusSq = neighborRadius * neighborRadius;
            auto candidate_ids = spatialGrid.queryRange(pos1, neighborRadius);
            
            for (size_t id2 : candidate_ids) {
                if (id1 != id2 && Positions.at(id1).distanceSquared(Positions.at(id2)) <= radiusSq) {
                    neighbors.push_back(id2);
                }
            }
            neighborMap[id1] = std::move(neighbors);
        }
    }
    
    // Update neighbor map for a single object
    void updateNeighborForID(size_t id) {
        TIME_FUNCTION;
        Vec2 pos_it = Positions.at(id);

        std::vector<size_t> neighbors;
        float radiusSq = neighborRadius * neighborRadius;
        
        for (const auto& [id2, pos2] : Positions) {
            if (id != id2 && pos_it.distanceSquared(pos2) <= radiusSq) {
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
    
};

#endif