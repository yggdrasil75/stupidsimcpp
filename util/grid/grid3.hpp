#ifndef GRID3_HPP
#define GRID3_HPP

#include "../vec3.hpp"
#include "../vec4.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <functional>

class Grid3 {
public:
    // Constructors
    Grid3() : width(0), height(0), depth(0) {}
    Grid3(int size) : width(size), height(size), depth(size) {
        positions.reserve(size * size * size);
        colors.reserve(size * size * size);
        sizes.reserve(size * size * size);
    }
    Grid3(int width, int height, int depth) : width(width), height(height), depth(depth) {
        positions.reserve(width * height * depth);
        colors.reserve(width * height * depth);
        sizes.reserve(width * height * depth);
    }
    
    // Add a voxel at specific position
    int addVoxel(const Vec3& position, const Vec4& color, float size = 1.0f) {
        int index = positions.size();
        positions.push_back(position);
        colors.push_back(color);
        sizes.push_back(size);
        positionToIndex[position] = index;
        return index;
    }
    
    // Add a voxel with integer coordinates
    int addVoxel(int x, int y, int z, const Vec4& color, float size = 1.0f) {
        return addVoxel(Vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)), color, size);
    }
    
    // Check if position is occupied
    bool isOccupied(const Vec3& position) const {
        return positionToIndex.find(position) != positionToIndex.end();
    }
    
    bool isOccupied(int x, int y, int z) const {
        return isOccupied(Vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
    }
    
    // Get voxel index at position, returns -1 if not found
    int getVoxelIndex(const Vec3& position) const {
        auto it = positionToIndex.find(position);
        return (it != positionToIndex.end()) ? it->second : -1;
    }
    
    int getVoxelIndex(int x, int y, int z) const {
        return getVoxelIndex(Vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
    }
    
    // Remove voxel at position
    bool removeVoxel(const Vec3& position) {
        int index = getVoxelIndex(position);
        if (index == -1) return false;
        
        // Swap with last element and update map
        if (index != positions.size() - 1) {
            positions[index] = positions.back();
            colors[index] = colors.back();
            sizes[index] = sizes.back();
            
            // Update mapping for the moved element
            positionToIndex[positions[index]] = index;
        }
        
        // Remove last element
        positions.pop_back();
        colors.pop_back();
        sizes.pop_back();
        positionToIndex.erase(position);
        
        return true;
    }
    
    bool removeVoxel(int x, int y, int z) {
        return removeVoxel(Vec3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
    }
    
    // Clear all voxels
    void clear() {
        positions.clear();
        colors.clear();
        sizes.clear();
        positionToIndex.clear();
    }
    
    // Get voxel count
    size_t getVoxelCount() const {
        return positions.size();
    }
    
    // Check if grid is empty
    bool isEmpty() const {
        return positions.empty();
    }
    
    // Get bounding box of occupied voxels
    void getBoundingBox(Vec3& minCorner, Vec3& maxCorner) const {
        if (positions.empty()) {
            minCorner = Vec3(0, 0, 0);
            maxCorner = Vec3(0, 0, 0);
            return;
        }
        
        minCorner = positions[0];
        maxCorner = positions[0];
        
        for (const auto& pos : positions) {
            minCorner = minCorner.min(pos);
            maxCorner = maxCorner.max(pos);
        }
    }
    
    // Fill a rectangular prism region
    void fillCuboid(const Vec3& start, const Vec3& end, const Vec4& color, float size = 1.0f) {
        int startX = static_cast<int>(std::min(start.x, end.x));
        int endX = static_cast<int>(std::max(start.x, end.x));
        int startY = static_cast<int>(std::min(start.y, end.y));
        int endY = static_cast<int>(std::max(start.y, end.y));
        int startZ = static_cast<int>(std::min(start.z, end.z));
        int endZ = static_cast<int>(std::max(start.z, end.z));
        
        for (int z = startZ; z <= endZ; ++z) {
            for (int y = startY; y <= endY; ++y) {
                for (int x = startX; x <= endX; ++x) {
                    if (!isOccupied(x, y, z)) {
                        addVoxel(x, y, z, color, size);
                    }
                }
            }
        }
    }
    
    // Create a sphere
    void fillSphere(const Vec3& center, float radius, const Vec4& color, float size = 1.0f) {
        int centerX = static_cast<int>(center.x);
        int centerY = static_cast<int>(center.y);
        int centerZ = static_cast<int>(center.z);
        int radiusInt = static_cast<int>(radius);
        
        for (int z = centerZ - radiusInt; z <= centerZ + radiusInt; ++z) {
            for (int y = centerY - radiusInt; y <= centerY + radiusInt; ++y) {
                for (int x = centerX - radiusInt; x <= centerX + radiusInt; ++x) {
                    float dx = x - center.x;
                    float dy = y - center.y;
                    float dz = z - center.z;
                    if (dx * dx + dy * dy + dz * dz <= radius * radius) {
                        if (!isOccupied(x, y, z)) {
                            addVoxel(x, y, z, color, size);
                        }
                    }
                }
            }
        }
    }
    
    // Create a hollow sphere (just the surface)
    void fillHollowSphere(const Vec3& center, float radius, const Vec4& color, float thickness = 1.0f, float size = 1.0f) {
        int centerX = static_cast<int>(center.x);
        int centerY = static_cast<int>(center.y);
        int centerZ = static_cast<int>(center.z);
        int radiusInt = static_cast<int>(radius);
        
        for (int z = centerZ - radiusInt; z <= centerZ + radiusInt; ++z) {
            for (int y = centerY - radiusInt; y <= centerY + radiusInt; ++y) {
                for (int x = centerX - radiusInt; x <= centerX + radiusInt; ++x) {
                    float dx = x - center.x;
                    float dy = y - center.y;
                    float dz = z - center.z;
                    float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                    
                    if (std::abs(distance - radius) <= thickness) {
                        if (!isOccupied(x, y, z)) {
                            addVoxel(x, y, z, color, size);
                        }
                    }
                }
            }
        }
    }
    
    // Create a cylinder
    void fillCylinder(const Vec3& baseCenter, const Vec3& axis, float radius, float height, 
                     const Vec4& color, float size = 1.0f) {
        // Simplified cylinder aligned with Y-axis
        Vec3 normalizedAxis = axis.normalized();
        
        for (int h = 0; h < static_cast<int>(height); ++h) {
            Vec3 center = baseCenter + normalizedAxis * static_cast<float>(h);
            
            for (int y = -static_cast<int>(radius); y <= static_cast<int>(radius); ++y) {
                for (int x = -static_cast<int>(radius); x <= static_cast<int>(radius); ++x) {
                    if (x * x + y * y <= radius * radius) {
                        Vec3 pos = center + Vec3(x, y, 0);
                        if (!isOccupied(pos)) {
                            addVoxel(pos, color, size);
                        }
                    }
                }
            }
        }
    }
    
    // Get neighbors of a voxel (6-connected)
    std::vector<int> getNeighbors6(const Vec3& position) const {
        std::vector<int> neighbors;
        Vec3 offsets[] = {
            Vec3(1, 0, 0), Vec3(-1, 0, 0),
            Vec3(0, 1, 0), Vec3(0, -1, 0),
            Vec3(0, 0, 1), Vec3(0, 0, -1)
        };
        
        for (const auto& offset : offsets) {
            Vec3 neighborPos = position + offset;
            int index = getVoxelIndex(neighborPos);
            if (index != -1) {
                neighbors.push_back(index);
            }
        }
        
        return neighbors;
    }
    
    // Get neighbors of a voxel (26-connected)
    std::vector<int> getNeighbors26(const Vec3& position) const {
        std::vector<int> neighbors;
        
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    
                    Vec3 neighborPos = position + Vec3(dx, dy, dz);
                    int index = getVoxelIndex(neighborPos);
                    if (index != -1) {
                        neighbors.push_back(index);
                    }
                }
            }
        }
        
        return neighbors;
    }
    
    // Create a simple teapot model (simplified)
    void createTeapot(const Vec3& position, float scale, const Vec4& color, float size = 1.0f) {
        // This is a very simplified teapot representation
        // In practice, you'd load a proper voxel model
        
        // Teapot body (ellipsoid)
        fillEllipsoid(position + Vec3(0, scale * 0.3f, 0), 
                     Vec3(scale * 0.4f, scale * 0.3f, scale * 0.4f), color, size);
        
        // Teapot lid (smaller ellipsoid on top)
        fillEllipsoid(position + Vec3(0, scale * 0.6f, 0), 
                     Vec3(scale * 0.3f, scale * 0.1f, scale * 0.3f), color, size);
        
        // Teapot spout (cylinder)
        fillCylinder(position + Vec3(scale * 0.3f, scale * 0.2f, 0), 
                    Vec3(1, 0.2f, 0), scale * 0.05f, scale * 0.3f, color, size);
        
        // Teapot handle (torus segment)
        fillTorusSegment(position + Vec3(-scale * 0.3f, scale * 0.3f, 0), 
                        Vec3(0, 1, 0), scale * 0.1f, scale * 0.2f, color, size);
    }
    
    // Fill an ellipsoid
    void fillEllipsoid(const Vec3& center, const Vec3& radii, const Vec4& color, float size = 1.0f) {
        int radiusX = static_cast<int>(radii.x);
        int radiusY = static_cast<int>(radii.y);
        int radiusZ = static_cast<int>(radii.z);
        
        for (int z = -radiusZ; z <= radiusZ; ++z) {
            for (int y = -radiusY; y <= radiusY; ++y) {
                for (int x = -radiusX; x <= radiusX; ++x) {
                    float normalizedX = static_cast<float>(x) / radii.x;
                    float normalizedY = static_cast<float>(y) / radii.y;
                    float normalizedZ = static_cast<float>(z) / radii.z;
                    
                    if (normalizedX * normalizedX + normalizedY * normalizedY + normalizedZ * normalizedZ <= 1.0f) {
                        Vec3 pos = center + Vec3(x, y, z);
                        if (!isOccupied(pos)) {
                            addVoxel(pos, color, size);
                        }
                    }
                }
            }
        }
    }
    
    // Fill a torus segment
    void fillTorusSegment(const Vec3& center, const Vec3& axis, float majorRadius, float minorRadius, 
                         const Vec4& color, float size = 1.0f) {
        Vec3 normalizedAxis = axis.normalized();
        
        // Simplified torus - in practice you'd use proper torus equation
        for (float angle = 0; angle < 2 * M_PI; angle += 0.2f) {
            Vec3 circleCenter = center + Vec3(std::cos(angle) * majorRadius, 0, std::sin(angle) * majorRadius);
            fillSphere(circleCenter, minorRadius, color, size);
        }
    }
    
    // Find connected components in 3D
    std::vector<std::vector<int>> findConnectedComponents() const {
        std::vector<std::vector<int>> components;
        std::unordered_map<Vec3, bool, std::hash<Vec3>> visited;
        
        for (size_t i = 0; i < positions.size(); ++i) {
            const Vec3& pos = positions[i];
            if (visited.find(pos) == visited.end()) {
                std::vector<int> component;
                floodFill3D(pos, visited, component);
                components.push_back(component);
            }
        }
        
        return components;
    }
    
    // Getters
    const std::vector<Vec3>& getPositions() const { return positions; }
    const std::vector<Vec4>& getColors() const { return colors; }
    const std::vector<float>& getSizes() const { return sizes; }
    
    Vec3 getPosition(int index) const { return positions[index]; }
    Vec4 getColor(int index) const { return colors[index]; }
    float getSize(int index) const { return sizes[index]; }
    
    void setColor(int index, const Vec4& color) { colors[index] = color; }
    void setSize(int index, float size) { sizes[index] = size; }
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getDepth() const { return depth; }

private:
    std::vector<Vec3> positions;
    std::vector<Vec4> colors;
    std::vector<float> sizes;
    std::unordered_map<Vec3, int, std::hash<Vec3>> positionToIndex;
    int width, height, depth;
    
    void floodFill3D(const Vec3& start, std::unordered_map<Vec3, bool, std::hash<Vec3>>& visited, 
                     std::vector<int>& component) const {
        std::vector<Vec3> stack;
        stack.push_back(start);
        
        while (!stack.empty()) {
            Vec3 current = stack.back();
            stack.pop_back();
            
            if (visited.find(current) != visited.end()) continue;
            
            visited[current] = true;
            int index = getVoxelIndex(current);
            if (index != -1) {
                component.push_back(index);
                
                // Add 6-connected neighbors
                Vec3 neighbors[] = {
                    current + Vec3(1, 0, 0), current + Vec3(-1, 0, 0),
                    current + Vec3(0, 1, 0), current + Vec3(0, -1, 0),
                    current + Vec3(0, 0, 1), current + Vec3(0, 0, -1)
                };
                
                for (const auto& neighbor : neighbors) {
                    if (isOccupied(neighbor) && visited.find(neighbor) == visited.end()) {
                        stack.push_back(neighbor);
                    }
                }
            }
        }
    }
};

#endif