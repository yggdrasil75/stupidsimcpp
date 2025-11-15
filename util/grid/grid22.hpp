#include <unordered_map>
#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include <vector>
#include <unordered_set>
#ifndef GRID2_HPP
#define GRID2_HPP

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

    bool empty() {
        return Positions.empty();
    }
    
    void clear() {
        Positions.clear();
        ƨnoiƚiƨoꟼ.clear();
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
        TIME_FUNCTION;
        std::vector<size_t> results;
        float radiusSq = radius * radius;
        
        // Calculate grid bounds for the query
        Vec2 minGrid = worldToGrid(center - Vec2(radius, radius));
        Vec2 maxGrid = worldToGrid(center + Vec2(radius, radius));
        
        // Check all relevant grid cells
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
    }
};

class Grid2 { 
private:
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
public:
    //get position from id
    Vec2 getPositionID(size_t id) const {
        Vec2 it = Positions.at(id);
        return it;
    }

    //get id from position (optional radius, picks first found. radius of 0 becomes epsilon if none are found)
    size_t getPositionVec(const Vec2& pos, float radius = 0.0f) {
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

    size_t getPositionVec(float x, float y, float radius = 0.0f) { 
        return getPositionVec(Vec2(x,y), radius);
    }

    //get all id in region
    std::vector<size_t> getPositionVecRegion(const Vec2& pos, float radius = 1.0f) {
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
        #pragma omp parallel for
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
        #pragma omp parallel for
        for (size_t i = 0; i < poses.size(); ++i) {
            size_t id = Positions.set(poses[i]);
            Colors[id] = colors[i];
            Sizes[id] = sizes[i];
            spatialGrid.insert(id,poses[i]);
        }
        
        shrinkIfNeeded();
        updateNeighborMap();
        
        return getAllIDs();
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

    // Get region as frame (RGBA format)
    frame getGridRegionAsFrameRGBA(const Vec2& minCorner, const Vec2& maxCorner) const {
        TIME_FUNCTION;
        int width, height;
        std::vector<uint8_t> rgbData;
        getGridRegionAsRGB(minCorner, maxCorner, width, height, rgbData);
        
        // Convert RGB to RGBA
        std::vector<uint8_t> rgbaData;
        rgbaData.reserve(width * height * 4);
        
        for (size_t i = 0; i < rgbData.size(); i += 3) {
            rgbaData.push_back(rgbData[i]);     // R
            rgbaData.push_back(rgbData[i + 1]); // G
            rgbaData.push_back(rgbData[i + 2]); // B
            rgbaData.push_back(255);            // A (fully opaque)
        }
        
        frame resultFrame(width, height, frame::colormap::RGBA);
        resultFrame.setData(rgbaData);
        return resultFrame;
    }

    // Get region as frame (BGRA format)
    frame getGridRegionAsFrameBGRA(const Vec2& minCorner, const Vec2& maxCorner) const {
        TIME_FUNCTION;
        int width, height;
        std::vector<uint8_t> bgrData;
        getGridRegionAsBGR(minCorner, maxCorner, width, height, bgrData);
        
        // Convert BGR to BGRA
        std::vector<uint8_t> bgraData;
        bgraData.reserve(width * height * 4);
        
        for (size_t i = 0; i < bgrData.size(); i += 3) {
            bgraData.push_back(bgrData[i]);     // B
            bgraData.push_back(bgrData[i + 1]); // G
            bgraData.push_back(bgrData[i + 2]); // R
            bgraData.push_back(255);            // A (fully opaque)
        }
        
        frame resultFrame(width, height, frame::colormap::BGRA);
        resultFrame.setData(bgraData);
        return resultFrame;
    }

    // Get region as frame (Grayscale format)
    frame getGridRegionAsFrameGrayscale(const Vec2& minCorner, const Vec2& maxCorner) const {
        TIME_FUNCTION;
        int width, height;
        std::vector<uint8_t> rgbData;
        getGridRegionAsRGB(minCorner, maxCorner, width, height, rgbData);
        
        // Convert RGB to grayscale
        std::vector<uint8_t> grayData;
        grayData.reserve(width * height);
        
        for (size_t i = 0; i < rgbData.size(); i += 3) {
            uint8_t r = rgbData[i];
            uint8_t g = rgbData[i + 1];
            uint8_t b = rgbData[i + 2];
            // Standard grayscale conversion formula
            uint8_t gray = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
            grayData.push_back(gray);
        }
        
        frame resultFrame(width, height, frame::colormap::B); // B for single channel/grayscale
        resultFrame.setData(grayData);
        return resultFrame;
    }

    // Get entire grid as frame with specified format
    frame getGridAsFrame(frame::colormap format = frame::colormap::RGB) {
        TIME_FUNCTION;
        Vec2 minCorner, maxCorner;
        getBoundingBox(minCorner, maxCorner);
        frame Frame;

        switch (format) {
            case frame::colormap::RGB:
                Frame = getGridRegionAsFrameRGB(minCorner, maxCorner);
            case frame::colormap::BGR:
                Frame = getGridRegionAsFrameBGR(minCorner, maxCorner);
            case frame::colormap::RGBA:
                Frame = getGridRegionAsFrameRGBA(minCorner, maxCorner);
            case frame::colormap::BGRA:
                Frame = getGridRegionAsFrameBGRA(minCorner, maxCorner);
            case frame::colormap::B:
                Frame = getGridRegionAsFrameGrayscale(minCorner, maxCorner);
            default:
                Frame = getGridRegionAsFrameRGB(minCorner, maxCorner);
        }
        Frame.compressFrameZigZagRLE();
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
            case frame::compresstype::ZIGZAG:
                return gridFrame.compressFrameZigZag();
            case frame::compresstype::DIFF:
                return gridFrame.compressFrameDiff();
            case frame::compresstype::ZIGZAGRLE:
                return gridFrame.compressFrameZigZagRLE();
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
    void getBoundingBox(Vec2& minCorner, Vec2& maxCorner) {
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
        spatialGrid.clear();
        neighborMap.clear();
    }

    // neighbor map
    void updateNeighborMap() {
        TIME_FUNCTION;
        neighborMap.clear();
        
        // For each object, find nearby neighbors
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