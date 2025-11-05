#ifndef VOXEL_HPP
#define VOXEL_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <tuple>
#include "timing_decorator.hpp"
#include "vec3.hpp"
#include "vec4.hpp"

class VoxelGrid {
private:
    std::unordered_map<Vec3, size_t> positionToIndex;
    std::vector<Vec3> positions;
    std::vector<Vec4> colors;
    std::vector<int> layers;
    
    Vec3 gridSize;

public:
    Vec3 voxelSize;
    
    enum LayerType {
        ATMOSPHERE = 0,
        CRUST = 1,
        MANTLE = 2,
        OUTER_CORE = 3,
        INNER_CORE = 4,
        EMPTY = -1
    };
    
    VoxelGrid(const Vec3& size, const Vec3& voxelSize = Vec3(1, 1, 1)) : gridSize(size), voxelSize(voxelSize) {}
    
    void addVoxel(const Vec3& position, const Vec4& color) {
        Vec3 gridPos = worldToGrid(position);
        
        auto it = positionToIndex.find(gridPos);
        if (it == positionToIndex.end()) {
            size_t index = positions.size();
            positions.push_back(gridPos);
            colors.push_back(color);
            layers.push_back(EMPTY);
            positionToIndex[gridPos] = index;
        } else {
            colors[it->second] = color;
        }
    }
    
    void addVoxelWithLayer(const Vec3& position, const Vec4& color, int layer) {
        Vec3 gridPos = worldToGrid(position);
        
        auto it = positionToIndex.find(gridPos);
        if (it == positionToIndex.end()) {
            size_t index = positions.size();
            positions.push_back(gridPos);
            colors.push_back(color);
            layers.push_back(layer);
            positionToIndex[gridPos] = index;
        } else {
            colors[it->second] = color;
            layers[it->second] = layer;
        }
    }
    
    Vec4 getVoxel(const Vec3& position) const {
        Vec3 gridPos = worldToGrid(position);
        auto it = positionToIndex.find(gridPos);
        if (it != positionToIndex.end()) {
            return colors[it->second];
        }
        return Vec4(0, 0, 0, 0);
    }
    
    int getVoxelLayer(const Vec3& position) const {
        Vec3 gridPos = worldToGrid(position);
        auto it = positionToIndex.find(gridPos);
        if (it != positionToIndex.end()) {
            return layers[it->second];
        }
        return EMPTY;
    }
    
    bool isOccupied(const Vec3& position) const {
        Vec3 gridPos = worldToGrid(position);
        return positionToIndex.find(gridPos) != positionToIndex.end();
    }
    
    Vec3 worldToGrid(const Vec3& worldPos) const {
        return (worldPos / voxelSize).floor();
    }
    
    Vec3 gridToWorld(const Vec3& gridPos) const {
        return gridPos * voxelSize;
    }
    
    const std::vector<Vec3>& getOccupiedPositions() const {
        return positions;
    }
    
    const std::vector<Vec4>& getColors() const {
        return colors;
    }
    
    const std::vector<int>& getLayers() const {
        return layers;
    }
    
    const std::unordered_map<Vec3, size_t>& getPositionToIndexMap() const {
        return positionToIndex;
    }
    
    const Vec3& getGridSize() const {
        return gridSize;
    }
    
    const Vec3& getVoxelSize() const {
        return voxelSize;
    }
    
    void clear() {
        positions.clear();
        colors.clear();
        layers.clear();
        positionToIndex.clear();
    }
    
    void assignPlanetaryLayers(const Vec3& center = Vec3(0, 0, 0)) {
        TIME_FUNCTION;
        printf("Assigning planetary layers...\n");
        
        const float atmospherePercent = 0.05f;
        const float crustPercent = 0.01f;
        const float mantlePercent = 0.10f;
        const float outerCorePercent = 0.42f;
        const float innerCorePercent = 0.42f;
        
        float maxDistance = 0.0f;
        for (const auto& pos : positions) {
            Vec3 worldPos = gridToWorld(pos);
            float distance = (worldPos - center).length();
            maxDistance = std::max(maxDistance, distance);
        }
        
        printf("Maximum distance from center: %.2f\n", maxDistance);
        
        const float atmosphereStart = maxDistance * (1.0f - atmospherePercent);
        const float crustStart = maxDistance * (1.0f - atmospherePercent - crustPercent);
        const float mantleStart = maxDistance * (1.0f - atmospherePercent - crustPercent - mantlePercent);
        const float outerCoreStart = maxDistance * (1.0f - atmospherePercent - crustPercent - mantlePercent - outerCorePercent);
        
        printf("Layer boundaries:\n");
        printf("  Atmosphere: %.2f to %.2f\n", atmosphereStart, maxDistance);
        printf("  Crust: %.2f to %.2f\n", crustStart, atmosphereStart);
        printf("  Mantle: %.2f to %.2f\n", mantleStart, crustStart);
        printf("  Outer Core: %.2f to %.2f\n", outerCoreStart, mantleStart);
        printf("  Inner Core: 0.00 to %.2f\n", outerCoreStart);
        
        int atmosphereCount = 0, crustCount = 0, mantleCount = 0, outerCoreCount = 0, innerCoreCount = 0;
        
        for (size_t i = 0; i < positions.size(); ++i) {
            Vec3 worldPos = gridToWorld(positions[i]);
            float distance = (worldPos - center).length();
            
            Vec4 layerColor;
            int layerType;
            
            if (distance >= atmosphereStart) {
                // Atmosphere - transparent blue
                layerColor = Vec4(0.2f, 0.4f, 1.0f, 0.3f); // Semi-transparent blue
                layerType = ATMOSPHERE;
                atmosphereCount++;
            } else if (distance >= crustStart) {
                // Crust - light brown
                layerColor = Vec4(0.8f, 0.7f, 0.5f, 1.0f); // Light brown
                layerType = CRUST;
                crustCount++;
            } else if (distance >= mantleStart) {
                // Mantle - reddish brown
                layerColor = Vec4(0.7f, 0.3f, 0.2f, 1.0f); // Reddish brown
                layerType = MANTLE;
                mantleCount++;
            } else if (distance >= outerCoreStart) {
                // Outer Core - orange/yellow
                layerColor = Vec4(1.0f, 0.6f, 0.2f, 1.0f); // Orange
                layerType = OUTER_CORE;
                outerCoreCount++;
            } else {
                // Inner Core - bright yellow
                layerColor = Vec4(1.0f, 0.9f, 0.1f, 1.0f); // Bright yellow
                layerType = INNER_CORE;
                innerCoreCount++;
            }
            
            colors[i] = layerColor;
            layers[i] = layerType;
        }
        
        printf("Layer distribution:\n");
        printf("  Atmosphere: %d voxels (%.1f%%)\n", atmosphereCount, (atmosphereCount * 100.0f) / positions.size());
        printf("  Crust: %d voxels (%.1f%%)\n", crustCount, (crustCount * 100.0f) / positions.size());
        printf("  Mantle: %d voxels (%.1f%%)\n", mantleCount, (mantleCount * 100.0f) / positions.size());
        printf("  Outer Core: %d voxels (%.1f%%)\n", outerCoreCount, (outerCoreCount * 100.0f) / positions.size());
        printf("  Inner Core: %d voxels (%.1f%%)\n", innerCoreCount, (innerCoreCount * 100.0f) / positions.size());
    }
};

#endif