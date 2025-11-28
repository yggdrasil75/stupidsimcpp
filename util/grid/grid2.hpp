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
#include "../simblocks/temp.hpp"
#include <vector>
#include <unordered_set>

const float EPSILON = 0.0000000000000000000000001;

class reverselookupassistant {
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

    bool contains(size_t id) const {
        return (Positions.find(id) != Positions.end());
    }

    bool contains(const Vec2& pos) const {
        return (ƨnoiƚiƨoꟼ.find(pos) != ƨnoiƚiƨoꟼ.end());
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
    
    std::unordered_set<size_t> find(const Vec2& center) const {
        //Vec2 g2pos = worldToGrid(center);
        auto cellIt = grid.find(worldToGrid(center));
        if (cellIt != grid.end()) {
            return cellIt->second;
        }
        return std::unordered_set<size_t>();
    }

    std::vector<size_t> queryRange(const Vec2& center, float radius) const {
        std::vector<size_t> results;
        float radiusSq = radius * radius;
        
        // Calculate grid bounds for the query
        Vec2 minGrid = worldToGrid(center - Vec2(radius, radius));
        Vec2 maxGrid = worldToGrid(center + Vec2(radius, radius));
        
        size_t estimatedSize = (maxGrid.x - minGrid.x + 1) * (maxGrid.y - minGrid.y + 1) * 10;
        results.reserve(estimatedSize);

        // Check all relevant grid cells
        for (int x = minGrid.x; x <= maxGrid.x; ++x) {
            for (int y = minGrid.y; y <= maxGrid.y; ++y) {
                auto cellIt = grid.find(Vec2(x, y));
                if (cellIt != grid.end()) {
                    results.insert(results.end(), cellIt->second.begin(), cellIt->second.end());
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

class GenericPixel {
protected:
    size_t id;
    Vec4 color;
    Vec2 pos;
public:
    //constructors
    GenericPixel(size_t id, Vec4 color, Vec2 pos) : id(id), color(color), pos(pos) {};

    //getters
    Vec4 getColor() const {
        return color;
    }

    //setters
    void setColor(Vec4 newColor) {
        color = newColor;
    }

    void move(Vec2 newPos) {
        pos = newPos;
    }
    
    void recolor(Vec4 newColor) {
        color.recolor(newColor);
    }
    
};

class Grid2 { 
protected:
    //all positions
    reverselookupassistant Positions;
    std::unordered_map<size_t, GenericPixel> Pixels;
    
    std::vector<size_t> unassignedIDs;
    
    //grid min
    Vec2 gridMin;
    //grid max
    Vec2 gridMax;

    float neighborRadius = 1.0f;
    
    //TODO: spatial map
    SpatialGrid spatialGrid;
    float spatialCellSize = neighborRadius * 1.5f;

    // Default background color for empty spaces
    Vec4 defaultBackgroundColor = Vec4(0.0f, 0.0f, 0.0f, 0.0f);

    PNoise2 noisegen;


    //water
    std::unordered_map<size_t, WaterParticle> water;

    std::unordered_map<size_t, Temp> tempMap;
public:
    bool usable = false;
    
    Grid2 noiseGenGrid(size_t minx,size_t miny, size_t maxx, size_t maxy, float minChance = 0.1f
                        , float maxChance = 1.0f, bool color = true, int noisemod = 42) {
        TIME_FUNCTION;
        std::cout << "generating a noise grid with the following: (" << minx << ", " << miny 
                << ") by (" << maxx << ", " << maxy << ") " << "chance: " << minChance 
                << " max: " << maxChance << " gen colors: " << color << std::endl;
        std::vector<Vec2> poses;
        std::vector<Vec4> colors;
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
                    } else {
                        Vec4 newc = Vec4(alpha,alpha,alpha,1.0);
                        colors.push_back(newc);
                        poses.push_back(Vec2(x,y));
                    }
                }
            }
        }
        std::cout << "noise generated" << std::endl;
        bulkAddObjects(poses,colors);
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

    //add pixel (default color and default size provided)
    size_t addObject(const Vec2& pos, const Vec4& color, float size = 1.0f) {
        size_t id = Positions.set(pos);
        Pixels.insert({id, GenericPixel(id, color, pos)});
        spatialGrid.insert(id, pos);
        return id;
    }

    // Set default background color for empty spaces
    void setDefault(const Vec4& color) {
        defaultBackgroundColor = color;
    }
    
    void setDefault(float r, float g, float b, float a = 0.0f) {
        defaultBackgroundColor = Vec4(r, g, b, a);
    }
    
    void setMaterialProperties(size_t id, double conductivity, double specific_heat, double density = 1.0) {
        auto it = tempMap.at(id);
        it.conductivity = conductivity;
        it.specific_heat = specific_heat;
        it.diffusivity = conductivity / (density * specific_heat);
    }
    
    //set position by id
    void setPosition(size_t id, const Vec2& newPosition) {
        Vec2 oldPosition = Positions.at(id);
        Pixels.at(id).move(newPosition);
        spatialGrid.update(id, oldPosition, newPosition);
        Positions.at(id).move(newPosition);
    }
        
    //set color by id (by pos same as get color)
    void setColor(size_t id, const Vec4 color) {
        Pixels.at(id).recolor(color);
    }
    
    // Set neighbor search radius
    void setNeighborRadius(float radius) {
        neighborRadius = radius;
        // updateNeighborMap(); // Recompute all neighbors
        optimizeSpatialGrid();
    }
    
    void setTemp(const Vec2 pos, double temp) {
        size_t id = getOrCreatePositionVec(pos, 0.0, true);
        setTemp(id, temp);
    }

    void setTemp(size_t id, double temp) {
        Temp tval = Temp(temp);
        tempMap.emplace(id, tval);
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
    
    Vec4 getColor(size_t id) {
        return Pixels.at(id).getColor();
    }
    
    double getTemp(size_t id) {
        if (tempMap.find(id) != tempMap.end()) {
            Temp temp = Temp(getPositionID(id), getTemps());
            tempMap.emplace(id, temp);
        }
        else {
            std::cout << "found a temp: " << tempMap.at(id).temp << std::endl;
        }
        return tempMap.at(id).temp;
    }
    
    double getTemp(Vec2 pos) {
        size_t posid;
        if (Positions.contains(pos)) {
            posid = Positions.at(pos);
        }
        if (tempMap.find(posid) != tempMap.end()) {
            return Temp(getPositionID(posid), getTemps()).temp;
        }
        else {
            return tempMap.at(posid).temp;
        }
    }

    std::unordered_map<Vec2, Temp> getTemps() {
        std::unordered_map<Vec2, Temp> out;
        for (const auto& [id, temp] : tempMap) {
            out.emplace(getPositionID(id), temp);
        }
        return out;
    }

    std::unordered_map<Vec2, Temp> getTemps(size_t id) {
        std::unordered_map<Vec2, Temp> out;
        std::vector<size_t> tval = spatialGrid.queryRange(Positions.at(id), 10);
        for (size_t tempid : tval) {
            Vec2 pos = Positions.at(tempid);
            if (tempMap.find(id) != tempMap.end()) {
                Temp temp = tempMap.at(tempid);
                out.insert({pos, temp});
            }
        }
        return out;
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
        
    }

    frame getGridRegionAsFrame(const Vec2& minCorner, const Vec2& maxCorner,
                       Vec2& res, frame::colormap outChannels = frame::colormap::RGB) const {
        TIME_FUNCTION;
        size_t width = static_cast<int>(maxCorner.x - minCorner.x);
        size_t height = static_cast<int>(maxCorner.y - minCorner.y);
        size_t outputWidth = static_cast<int>(res.x);
        size_t outputHeight = static_cast<int>(res.y);
        float widthScale = outputWidth / width;
        float heightScale = outputHeight / height;
        
        frame outframe = frame();
        outframe.colorFormat = outChannels;

        if (width <= 0 || height <= 0) {
            width = height = 0;
            return outframe;
        }

        std::cout << "Rendering region: " << minCorner << " to " << maxCorner 
                << " at resolution: " << res << std::endl;
        std::cout << "Scale factors: " << widthScale << " x " << heightScale << std::endl;
        
        std::unordered_map<Vec2,Vec4> colorBuffer;
        colorBuffer.reserve(outputHeight*outputWidth);
        std::unordered_map<Vec2,Vec4> colorTempBuffer;
        colorTempBuffer.reserve(outputHeight * outputWidth);
        std::unordered_map<Vec2,int> countBuffer;
        countBuffer.reserve(outputHeight * outputWidth);
        std::cout << "built buffers" << std::endl;

        for (const auto& [id, pos] : Positions) {
            if (pos.x >= minCorner.x && pos.x <= maxCorner.x && 
                pos.y >= minCorner.y && pos.y <= maxCorner.y) {
                float relx = pos.x - minCorner.x;
                float rely = pos.y - minCorner.y;
                int pixx = static_cast<int>(relx * widthScale);
                int pixy = static_cast<int>(rely * heightScale);
                Vec2 pix = Vec2(pixx,pixy);
                
                colorTempBuffer[pix] += Pixels.at(id).getColor();
                countBuffer[pix]++;
                //std::cout << "pixel at " << pix << " is: " << Pixels.at(id).getColor();
            }
        }
        std::cout << std::endl << "built initial buffer" << std::endl;

        for (size_t y = 0; y < outputHeight; ++y) {
            for (size_t x = 0; x < outputWidth; ++x) {
                if (countBuffer[Vec2(x,y)] > 0) colorBuffer[Vec2(x,y)] = colorTempBuffer[Vec2(x,y)] / static_cast<float>(countBuffer[Vec2(x,y)]) * 255;
                else colorBuffer[Vec2(x,y)] = defaultBackgroundColor;
            }
        }
        std::cout << "blended second buffer" << std::endl;

        switch (outChannels) {
            case frame::colormap::RGBA: {
                std::vector<uint8_t> colorBuffer2(outputWidth*outputHeight*4, 0);
                std::cout << "outputting RGBA: " << std::endl;
                for (const auto& [v2,getColor] : colorBuffer) {
                    size_t index = (v2.y * outputWidth + v2.x) * 4;
                    // std::cout << "index: " << index << std::endl;
                    colorBuffer2[index+0] = getColor.r;
                    colorBuffer2[index+1] = getColor.g;
                    colorBuffer2[index+2] = getColor.b;
                    colorBuffer2[index+3] = getColor.a;
                }
                frame result = frame(res.x,res.y, frame::colormap::RGBA);
                result.setData(colorBuffer2);
                std::cout << "returning result" << std::endl;
                return result;
                break;
            }
            case frame::colormap::BGR: {
                std::vector<uint8_t> colorBuffer2(outputWidth*outputHeight*3, 0);
                std::cout << "outputting BGR: " << std::endl;
                for (const auto& [v2,getColor] : colorBuffer) {
                    size_t index = (v2.y * outputWidth + v2.x) * 3;
                    // std::cout << "index: " << index << std::endl;
                    colorBuffer2[index+2] = getColor.r;
                    colorBuffer2[index+1] = getColor.g;
                    colorBuffer2[index+0] = getColor.b;
                    //colorBuffer2[index+3] = getColor.a;
                }
                frame result = frame(res.x,res.y, frame::colormap::BGR);
                result.setData(colorBuffer2);
                std::cout << "returning result" << std::endl;
                return result;
                break;
            }
            case frame::colormap::RGB: 
            default: {
                std::vector<uint8_t> colorBuffer2(outputWidth*outputHeight*3, 0);
                std::cout << "outputting RGB: " << std::endl;
                for (const auto& [v2,getColor] : colorBuffer) {
                    size_t index = (v2.y * outputWidth + v2.x) * 3;
                    // std::cout << "index: " << index << std::endl;
                    colorBuffer2[index+0] = getColor.r;
                    colorBuffer2[index+1] = getColor.g;
                    colorBuffer2[index+2] = getColor.b;
                    //colorBuffer2[index+3] = getColor.a;
                }
                frame result = frame(res.x,res.y, frame::colormap::RGB);
                result.setData(colorBuffer2);
                std::cout << "returning result" << std::endl;
                return result;
                break;
            }
        }
    }

    frame getGridAsFrame(frame::colormap outchannel = frame::colormap::RGB) {
        Vec2 min;
        Vec2 max;
        getBoundingBox(min,max);
        Vec2 res = (max + 1) - min;
        std::cout << "getting grid as frame with the following: " << min << max << res << std::endl;
        return getGridRegionAsFrame(min, max, res, outchannel);
    }

    frame getTempAsFrame(Vec2 minCorner, Vec2 maxCorner, Vec2 res, frame::colormap outcolor = frame::colormap::RGB)  {
        TIME_FUNCTION;
        int pcount = 0;
        size_t sheight = maxCorner.x - minCorner.x;
        size_t swidth = maxCorner.y - minCorner.y;
        
        int width = static_cast<int>(res.x);
        int height = static_cast<int>(res.y);
        std::unordered_map<Vec2, double> tempBuffer;
        tempBuffer.reserve(res.x * res.y);
        double maxTemp = 0.0;
        double minTemp = 0.0;
        float xdiff = (maxCorner.x - minCorner.x);
        float ydiff = (maxCorner.y - minCorner.y);
        for (int x = 0; x < res.x; x++) {
            for (int y = 0; y < res.y; y++) {
                Vec2 cposout = Vec2(x,y);
                Vec2 cposin = Vec2(minCorner.x + (x * xdiff / res.x),minCorner.y + (y * ydiff / res.y));
                double ctemp = getTemp(cposin);
                
                tempBuffer[Vec2(x,y)] = ctemp;
                if (ctemp > maxTemp) maxTemp = ctemp;
                else if (ctemp < minTemp) minTemp = ctemp;
            }
        }
        std::cout << "max temp: " << maxTemp << " min temp: " << minTemp << std::endl;
        
        switch (outcolor) {
            case frame::colormap::RGBA: {
                std::vector<uint8_t> rgbaBuffer(width*height*4, 0);
                for (const auto& [v2, temp] : tempBuffer) {
                    size_t index = (v2.y * width + v2.x) * 4;
                    uint8_t atemp  = static_cast<unsigned char>((((temp-minTemp)) / (maxTemp-minTemp)) * 255);
                    rgbaBuffer[index+0] = atemp;
                    rgbaBuffer[index+1] = atemp;
                    rgbaBuffer[index+2] = atemp;
                    rgbaBuffer[index+3] = 255;
                }
                frame result = frame(res.x,res.y, frame::colormap::RGBA);
                result.setData(rgbaBuffer);
                return result;
                break;
            }
            case frame::colormap::BGR: {
                std::vector<uint8_t> rgbaBuffer(width*height*3, 0);
                for (const auto& [v2, temp] : tempBuffer) {
                    size_t index = (v2.y * width + v2.x) * 3;
                    uint8_t atemp  = static_cast<unsigned char>((((temp-minTemp)) / (maxTemp-minTemp)) * 255);
                    rgbaBuffer[index+2] = atemp;
                    rgbaBuffer[index+1] = atemp;
                    rgbaBuffer[index+0] = atemp;
                }
                frame result = frame(res.x,res.y, frame::colormap::BGR);
                result.setData(rgbaBuffer);
                return result;
                break;
            }
            case frame::colormap::RGB: 
            default: {
                std::vector<uint8_t> rgbaBuffer(width*height*3, 0);
                for (const auto& [v2, temp] : tempBuffer) {
                    size_t index = (v2.y * width + v2.x) * 3;
                    uint8_t atemp  = static_cast<unsigned char>((((temp-minTemp)) / (maxTemp-minTemp)) * 255);
                    rgbaBuffer[index+0] = atemp;
                    rgbaBuffer[index+1] = atemp;
                    rgbaBuffer[index+2] = atemp;
                }
                frame result = frame(res.x,res.y, frame::colormap::RGB);
                result.setData(rgbaBuffer);
                return result;
                break;
            }
        }
    }

    size_t removeID(size_t id) {
        Vec2 oldPosition = Positions.at(id);
        Positions.remove(id);
        Pixels.erase(id);
        unassignedIDs.push_back(id);
        spatialGrid.remove(id, oldPosition);
        return id;
    }
    
    //bulk update positions
    void bulkUpdatePositions(const std::unordered_map<size_t, Vec2>& newPositions) {
        TIME_FUNCTION;
        for (const auto& [id, newPos] : newPositions) {
            Vec2 oldPosition = Positions.at(id);
            Positions.at(id).move(newPos);
            Pixels.at(id).move(newPos);
            spatialGrid.update(id, oldPosition, newPos);
        }
    }
    
    std::vector<size_t> bulkAddObjects(const std::vector<Vec2> poses, std::vector<Vec4> colors) {
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
            Pixels.emplace(id, GenericPixel(id, colors[i], poses[i]));
            spatialGrid.insert(id,poses[i]);
            newids.push_back(id);
        }
        
        shrinkIfNeeded();
        
        usable = true;
        return newids;
    }

    std::vector<size_t> bulkAddObjects(const std::vector<Vec2> poses, std::vector<Vec4> colors, std::vector<float>& temps) {
        TIME_FUNCTION;
        std::vector<size_t> ids;
        ids.reserve(poses.size());
        
        // Reserve space in maps to avoid rehashing
        if (Positions.bucket_count() < Positions.size() + poses.size()) {
            Positions.reserve(Positions.size() + poses.size());
            Pixels.reserve(Positions.size() + poses.size());
            tempMap.reserve(tempMap.size() + temps.size());
        }
        
        // Batch insertion
        std::vector<size_t> newids;
        for (size_t i = 0; i < poses.size(); ++i) {
            size_t id = Positions.set(poses[i]);
            Pixels.emplace(id, GenericPixel(id, colors[i], poses[i]));
            Temp temptemp = Temp(temps[i]);
            tempMap.insert({id, temptemp});
            spatialGrid.insert(id,poses[i]);
            newids.push_back(id);
        }
        
        shrinkIfNeeded();
        
        usable = true;
        return newids;
    }

    void shrinkIfNeeded() {
        //TODO: garbage collector
    }
    
    //clear
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
        spatialGrid = SpatialGrid(spatialCellSize);
        
        // Rebuild spatial grid
        spatialGrid.clear();
        for (const auto& [id, pos] : Positions) {
            spatialGrid.insert(id, pos);
        }
    }

    std::vector<size_t> getNeighbors(size_t id) const {
        Vec2 pos = Positions.at(id);
        std::vector<size_t> candidates = spatialGrid.queryRange(pos, neighborRadius);
        
        // Filter out self and apply exact distance check
        std::vector<size_t> neighbors;
        float radiusSq = neighborRadius * neighborRadius;
        
        for (size_t candidateId : candidates) {
            if (candidateId != id && 
                pos.distanceSquared(Positions.at(candidateId)) <= radiusSq) {
                neighbors.push_back(candidateId);
            }
        }
        
        return neighbors;
    }
    
    std::vector<size_t> getNeighborsRange(size_t id, float dist) const {
        Vec2 pos = Positions.at(id);
        std::vector<size_t> candidates = spatialGrid.queryRange(pos, neighborRadius);
        
        // Filter out self and apply exact distance check
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
    
    Grid2 noiseGenGridTemps(size_t minx,size_t miny, size_t maxx, size_t maxy, float minChance = 0.1f
                        , float maxChance = 1.0f, bool color = true, int noisemod = 42) {
        TIME_FUNCTION;
        std::cout << "generating a noise grid with the following: (" << minx << ", " << miny 
                << ") by (" << maxx << ", " << maxy << ") " << "chance: " << minChance 
                << " max: " << maxChance << " gen colors: " << color << std::endl;
        std::vector<Vec2> poses;
        std::vector<Vec4> colors;
        std::vector<float> temps;
        int callnumber = 0;
        for (int x = minx; x < maxx; x++) {
            for (int y = miny; y < maxy; y++) {
                float nx = (x+noisemod)/(maxx+EPSILON)/0.1;
                float ny = (y+noisemod)/(maxy+EPSILON)/0.1;
                Vec2 pos = Vec2(nx,ny);
                float temp = noisegen.permute(Vec2(nx*0.2+1,ny*0.1+2));
                float alpha = noisegen.permute(pos);
                if (alpha > minChance && alpha < maxChance) {
                    if (color) {
                        float red = noisegen.permute(Vec2(nx*0.3,ny*0.3));
                        float green = noisegen.permute(Vec2(nx*0.6,ny*.06));
                        float blue = noisegen.permute(Vec2(nx*0.9,ny*0.9));
                        Vec4 newc = Vec4(red,green,blue,1.0);
                        colors.push_back(newc);
                        poses.push_back(Vec2(x,y));
                        temps.push_back(temp * 100);
                        //std::cout << "temp: " << temp << std::endl;
                    } else {
                        Vec4 newc = Vec4(alpha,alpha,alpha,1.0);
                        colors.push_back(newc);
                        poses.push_back(Vec2(x,y));
                        temps.push_back(temp * 100);
                    }
                }
            }
        }
        std::cout << "noise generated" << std::endl;
        bulkAddObjects(poses, colors, temps);
        return *this;
    }

    std::unordered_map<size_t, Temp*> findTempsInRegion(const Vec2& center, float radius) {
        std::unordered_map<size_t, Temp*> results;
        
        // Get all IDs in the region
        auto idsInRegion = spatialGrid.queryRange(center, radius);
        results.reserve(idsInRegion.size());
        
        // Filter for ones that have temperature data
        for (size_t id : idsInRegion) {
            auto tempIt = tempMap.find(id);
            if (tempIt != tempMap.end()) {
                results.emplace(id, &tempIt->second);
            }
        }
        
        return results;
    }

    Grid2 backfillGrid() {
        Vec2 Min;
        Vec2 Max;
        getBoundingBox(Min, Max);
        std::vector<Vec2> newPos;
        std::vector<Vec4> newColors;
        for (size_t x = Min.x; x < Max.x; x++) {
            for (size_t y = Min.y; y < Max.y; x++) {
                Vec2 pos = Vec2(x,y);
                if (Positions.contains(pos)) continue;
                Vec4 color = defaultBackgroundColor;
                float size = 0.1;
                newPos.push_back(pos);
                newColors.push_back(color);
            }
        }
        bulkAddObjects(newPos, newColors);
        gradTemps();
        return *this;
    }

    void gradTemps() {
        //run this at the start. it generates temps for the grid from a sampling
        std::vector<Vec2> toProcess;
        
        Vec2 Min, Max;
        getBoundingBox(Min, Max);
        
        std::cout << "min: " << Min << std::endl;
        std::cout << "max: " << Max << std::endl;
        for (size_t x = Min.x; x < Max.x; x++) {
            for (size_t y = Min.y; y < Max.y; y++) {
                Vec2 pasdfjlkasdfasdfjlkasdfjlk = Vec2(x,y);
                toProcess.emplace_back(pasdfjlkasdfasdfjlkasdfjlk);
            }
        }

        while (toProcess.size() > 0) {
            std::cout << "setting temp on " << toProcess.size() << " values" << std::endl;
            for (size_t iter = 0; iter < toProcess.size(); iter++) {
                Vec2 cpos = toProcess[iter];
                size_t id = getPositionVec(cpos);
                if (tempMap.find(id) != tempMap.end()) {
                    toProcess.erase(toProcess.begin()+iter);
                }
            }
            for (auto [id, temp] : tempMap) {
                std::vector<size_t> neighbors = spatialGrid.queryRange(getPositionID(id), 35);
                std::unordered_map<Vec2, Temp> neighbortemps;
                for (size_t id : neighbors) {
                    auto tempIt = tempMap.find(id);
                    if (tempIt != tempMap.end()) {
                        neighbortemps.insert({getPositionID(id), tempIt->second});
                    }
                }
                Vec2 pos = getPositionID(id);

                for (size_t neighbor : neighbors) {
                    // if (tempMap.find(neighbor) != tempMap.end()) {
                        Vec2 npos = getPositionID(neighbor);
                        float newtemp = Temp::calTempIDW(npos, neighbortemps);
                        Temp newTempT = Temp(newtemp);
                        tempMap.insert({neighbor, newTempT});
                    // }
                }
            }
        }
    }

    void diffuseTemps(int timestep) {
        TIME_FUNCTION;
        if (tempMap.empty() || timestep < 1) return;
        std::unordered_map<size_t, float> cTemps;
        cTemps.reserve(tempMap.size());
        for (const auto& [id, temp] : tempMap) {
            Vec2 pos = getPositionID(id);

        }
    }
};

#endif