#ifndef VOXEL_GENERATORS_HPP
#define VOXEL_GENERATORS_HPP

#include "grid3.hpp"
#include <cmath>
#include <vector>
#include <functional>
#include "../noise/pnoise2.hpp"
#include "../vectorlogic/vec3.hpp"
#include <array>

class VoxelGenerators {
public:
    // Basic Primitive Generators
    static void createSphere(VoxelGrid& grid, const Vec3f& center, float radius, 
                            const Vec3ui8& color = Vec3ui8(255, 255, 255), 
                            bool filled = true) {
        TIME_FUNCTION;
        
        Vec3i gridCenter = (center / grid.binSize).floorToI();
        Vec3i radiusVoxels = Vec3i(static_cast<int>(radius / grid.binSize));
        
        Vec3i minBounds = gridCenter - radiusVoxels;
        Vec3i maxBounds = gridCenter + radiusVoxels;
        
        // Ensure bounds are within grid
        minBounds = minBounds.max(Vec3i(0, 0, 0));
        maxBounds = maxBounds.min(Vec3i(grid.getWidth() - 1, grid.getHeight() - 1, grid.getDepth() - 1));
        
        float radiusSq = radius * radius;
        
        for (int z = minBounds.z; z <= maxBounds.z; ++z) {
            for (int y = minBounds.y; y <= maxBounds.y; ++y) {
                for (int x = minBounds.x; x <= maxBounds.x; ++x) {
                    Vec3f voxelCenter(x * grid.binSize, y * grid.binSize, z * grid.binSize);
                    Vec3f delta = voxelCenter - center;
                    float distanceSq = delta.lengthSquared();
                    
                    if (filled) {
                        // Solid sphere
                        if (distanceSq <= radiusSq) {
                            grid.set(Vec3i(x, y, z), true, color);
                        }
                    } else {
                        // Hollow sphere (shell)
                        float shellThickness = grid.binSize;
                        if (distanceSq <= radiusSq && distanceSq >= (radius - shellThickness) * (radius - shellThickness)) {
                            grid.set(Vec3i(x, y, z), true, color);
                        }
                    }
                }
            }
        }
        
        grid.clearMeshCache();
    }
    
    static void createCube(VoxelGrid& grid, const Vec3f& center, const Vec3f& size,
                          const Vec3ui8& color = Vec3ui8(255, 255, 255),
                          bool filled = true) {
        TIME_FUNCTION;
        
        Vec3f halfSize = size * 0.5f;
        Vec3f minPos = center - halfSize;
        Vec3f maxPos = center + halfSize;
        
        Vec3i minVoxel = (minPos / grid.binSize).floorToI();
        Vec3i maxVoxel = (maxPos / grid.binSize).floorToI();
        
        // Clamp to grid bounds
        minVoxel = minVoxel.max(Vec3i(0, 0, 0));
        maxVoxel = maxVoxel.min(Vec3i(grid.getWidth() - 1, grid.getHeight() - 1, grid.getDepth() - 1));
        
        if (filled) {
            // Solid cube
            for (int z = minVoxel.z; z <= maxVoxel.z; ++z) {
                for (int y = minVoxel.y; y <= maxVoxel.y; ++y) {
                    for (int x = minVoxel.x; x <= maxVoxel.x; ++x) {
                        grid.set(Vec3i(x, y, z), true, color);
                    }
                }
            }
        } else {
            // Hollow cube (just the faces)
            for (int z = minVoxel.z; z <= maxVoxel.z; ++z) {
                for (int y = minVoxel.y; y <= maxVoxel.y; ++y) {
                    for (int x = minVoxel.x; x <= maxVoxel.x; ++x) {
                        // Check if on any face
                        bool onFace = (x == minVoxel.x || x == maxVoxel.x ||
                                      y == minVoxel.y || y == maxVoxel.y ||
                                      z == minVoxel.z || z == maxVoxel.z);
                        
                        if (onFace) {
                            grid.set(Vec3i(x, y, z), true, color);
                        }
                    }
                }
            }
        }
        
        grid.clearMeshCache();
    }
    
    static void createCylinder(VoxelGrid& grid, const Vec3f& center, float radius, float height,
                              const Vec3ui8& color = Vec3ui8(255, 255, 255),
                              bool filled = true, int axis = 1) { // 0=X, 1=Y, 2=Z
        TIME_FUNCTION;
        
        Vec3f halfHeight = Vec3f(0, 0, 0);
        halfHeight[axis] = height * 0.5f;
        
        Vec3f minPos = center - halfHeight;
        Vec3f maxPos = center + halfHeight;
        
        Vec3i minVoxel = (minPos / grid.binSize).floorToI();
        Vec3i maxVoxel = (maxPos / grid.binSize).floorToI();
        
        minVoxel = minVoxel.max(Vec3i(0, 0, 0));
        maxVoxel = maxVoxel.min(Vec3i(grid.getWidth() - 1, grid.getHeight() - 1, grid.getDepth() - 1));
        
        float radiusSq = radius * radius;
        
        for (int k = minVoxel[axis]; k <= maxVoxel[axis]; ++k) {
            // Iterate through the other two dimensions
            for (int j = minVoxel[(axis + 1) % 3]; j <= maxVoxel[(axis + 1) % 3]; ++j) {
                for (int i = minVoxel[(axis + 2) % 3]; i <= maxVoxel[(axis + 2) % 3]; ++i) {
                    Vec3i pos;
                    pos[axis] = k;
                    pos[(axis + 1) % 3] = j;
                    pos[(axis + 2) % 3] = i;
                    
                    Vec3f voxelCenter = pos.toFloat() * grid.binSize;
                    
                    // Calculate distance from axis
                    float dx = voxelCenter.x - center.x;
                    float dy = voxelCenter.y - center.y;
                    float dz = voxelCenter.z - center.z;
                    
                    // Zero out the axis component
                    if (axis == 0) dx = 0;
                    else if (axis == 1) dy = 0;
                    else dz = 0;
                    
                    float distanceSq = dx*dx + dy*dy + dz*dz;
                    
                    if (filled) {
                        if (distanceSq <= radiusSq) {
                            grid.set(pos, true, color);
                        }
                    } else {
                        float shellThickness = grid.binSize;
                        if (distanceSq <= radiusSq && 
                            distanceSq >= (radius - shellThickness) * (radius - shellThickness)) {
                            grid.set(pos, true, color);
                        }
                    }
                }
            }
        }
        
        grid.clearMeshCache();
    }
    
    static void createCone(VoxelGrid& grid, const Vec3f& baseCenter, float radius, float height,
                          const Vec3ui8& color = Vec3ui8(255, 255, 255),
                          bool filled = true, int axis = 1) { // 0=X, 1=Y, 2=Z
        TIME_FUNCTION;
        
        Vec3f tip = baseCenter;
        tip[axis] += height;
        
        Vec3f minPos = baseCenter.min(tip);
        Vec3f maxPos = baseCenter.max(tip);
        
        // Expand by radius in other dimensions
        for (int i = 0; i < 3; ++i) {
            if (i != axis) {
                minPos[i] -= radius;
                maxPos[i] += radius;
            }
        }
        
        Vec3i minVoxel = (minPos / grid.binSize).floorToI();
        Vec3i maxVoxel = (maxPos / grid.binSize).floorToI();
        
        minVoxel = minVoxel.max(Vec3i(0, 0, 0));
        maxVoxel = maxVoxel.min(Vec3i(grid.getWidth() - 1, grid.getHeight() - 1, grid.getDepth() - 1));
        
        for (int k = minVoxel[axis]; k <= maxVoxel[axis]; ++k) {
            // Current height from base
            float h = (k * grid.binSize) - baseCenter[axis];
            if (h < 0 || h > height) continue;
            
            // Current radius at this height
            float currentRadius = radius * (1.0f - h / height);
            
            for (int j = minVoxel[(axis + 1) % 3]; j <= maxVoxel[(axis + 1) % 3]; ++j) {
                for (int i = minVoxel[(axis + 2) % 3]; i <= maxVoxel[(axis + 2) % 3]; ++i) {
                    Vec3i pos;
                    pos[axis] = k;
                    pos[(axis + 1) % 3] = j;
                    pos[(axis + 2) % 3] = i;
                    
                    Vec3f voxelCenter = pos.toFloat() * grid.binSize;
                    
                    // Calculate distance from axis
                    float dx = voxelCenter.x - baseCenter.x;
                    float dy = voxelCenter.y - baseCenter.y;
                    float dz = voxelCenter.z - baseCenter.z;
                    
                    // Zero out the axis component
                    if (axis == 0) dx = 0;
                    else if (axis == 1) dy = 0;
                    else dz = 0;
                    
                    float distanceSq = dx*dx + dy*dy + dz*dz;
                    
                    if (filled) {
                        if (distanceSq <= currentRadius * currentRadius) {
                            grid.set(pos, true, color);
                        }
                    } else {
                        float shellThickness = grid.binSize;
                        if (distanceSq <= currentRadius * currentRadius && 
                            distanceSq >= (currentRadius - shellThickness) * (currentRadius - shellThickness)) {
                            grid.set(pos, true, color);
                        }
                    }
                }
            }
        }
        
        grid.clearMeshCache();
    }
    
    static void createTorus(VoxelGrid& grid, const Vec3f& center, float majorRadius, float minorRadius,
                           const Vec3ui8& color = Vec3ui8(255, 255, 255)) {
        TIME_FUNCTION;
        
        float outerRadius = majorRadius + minorRadius;
        Vec3f minPos = center - Vec3f(outerRadius, outerRadius, minorRadius);
        Vec3f maxPos = center + Vec3f(outerRadius, outerRadius, minorRadius);
        
        Vec3i minVoxel = (minPos / grid.binSize).floorToI();
        Vec3i maxVoxel = (maxPos / grid.binSize).floorToI();
        
        minVoxel = minVoxel.max(Vec3i(0, 0, 0));
        maxVoxel = maxVoxel.min(Vec3i(grid.getWidth() - 1, grid.getHeight() - 1, grid.getDepth() - 1));
        
        for (int z = minVoxel.z; z <= maxVoxel.z; ++z) {
            for (int y = minVoxel.y; y <= maxVoxel.y; ++y) {
                for (int x = minVoxel.x; x <= maxVoxel.x; ++x) {
                    Vec3f pos(x * grid.binSize, y * grid.binSize, z * grid.binSize);
                    Vec3f delta = pos - center;
                    
                    // Torus equation: (sqrt(x² + y²) - R)² + z² = r²
                    float xyDist = std::sqrt(delta.x * delta.x + delta.y * delta.y);
                    float distToCircle = xyDist - majorRadius;
                    float distanceSq = distToCircle * distToCircle + delta.z * delta.z;
                    
                    if (distanceSq <= minorRadius * minorRadius) {
                        grid.set(Vec3i(x, y, z), true, color);
                    }
                }
            }
        }
        
        grid.clearMeshCache();
    }
    
    // Procedural Generators
    static void createPerlinNoiseTerrain(VoxelGrid& grid, float frequency = 0.1f, float amplitude = 10.0f,
                                        int octaves = 4, float persistence = 0.5f,
                                        const Vec3ui8& baseColor = Vec3ui8(34, 139, 34)) {
        TIME_FUNCTION;
        
        if (grid.getHeight() < 1) return;
        
        PNoise2 noise;
        
        for (int z = 0; z < grid.getDepth(); ++z) {
            for (int x = 0; x < grid.getWidth(); ++x) {
                // Generate height value using Perlin noise
                float heightValue = 0.0f;
                float freq = frequency;
                float amp = amplitude;
                
                for (int octave = 0; octave < octaves; ++octave) {
                    float nx = x * freq / 100.0f;
                    float nz = z * freq / 100.0f;
                    heightValue += noise.permute(Vec2f(nx, nz)) * amp;
                    
                    freq *= 2.0f;
                    amp *= persistence;
                }
                
                // Normalize and scale to grid height
                int terrainHeight = static_cast<int>((heightValue + amplitude) / (2.0f * amplitude) * grid.getHeight());
                terrainHeight = std::max(0, std::min(grid.getHeight() - 1, terrainHeight));
                
                // Create column of voxels
                for (int y = 0; y <= terrainHeight; ++y) {
                    // Color gradient based on height
                    float t = static_cast<float>(y) / grid.getHeight();
                    Vec3ui8 color = baseColor;
                    
                    // Add some color variation
                    if (t < 0.3f) {
                        // Water level
                        color = Vec3ui8(30, 144, 255);
                    } else if (t < 0.5f) {
                        // Sand
                        color = Vec3ui8(238, 214, 175);
                    } else if (t < 0.8f) {
                        // Grass
                        color = baseColor;
                    } else {
                        // Snow
                        color = Vec3ui8(255, 250, 250);
                    }
                    
                    grid.set(Vec3i(x, y, z), true, color);
                }
            }
        }
        
        grid.clearMeshCache();
    }
    
    static void createMengerSponge(VoxelGrid& grid, int iterations = 3,
                                  const Vec3ui8& color = Vec3ui8(255, 255, 255)) {
        TIME_FUNCTION;
        
        // Start with a solid cube
        createCube(grid, 
                  Vec3f(grid.getWidth() * grid.binSize * 0.5f,
                        grid.getHeight() * grid.binSize * 0.5f,
                        grid.getDepth() * grid.binSize * 0.5f),
                  Vec3f(grid.getWidth() * grid.binSize,
                        grid.getHeight() * grid.binSize,
                        grid.getDepth() * grid.binSize),
                  color, true);
        
        // Apply Menger sponge iteration
        for (int iter = 0; iter < iterations; ++iter) {
            int divisor = static_cast<int>(std::pow(3, iter + 1));
            
            // Calculate the pattern
            for (int z = 0; z < grid.getDepth(); ++z) {
                for (int y = 0; y < grid.getHeight(); ++y) {
                    for (int x = 0; x < grid.getWidth(); ++x) {
                        // Check if this voxel should be removed in this iteration
                        int modX = x % divisor;
                        int modY = y % divisor;
                        int modZ = z % divisor;
                        
                        int third = divisor / 3;
                        
                        // Remove center cubes
                        if ((modX >= third && modX < 2 * third) &&
                            (modY >= third && modY < 2 * third)) {
                            grid.set(Vec3i(x, y, z), false, color);
                        }
                        if ((modX >= third && modX < 2 * third) &&
                            (modZ >= third && modZ < 2 * third)) {
                            grid.set(Vec3i(x, y, z), false, color);
                        }
                        if ((modY >= third && modY < 2 * third) &&
                            (modZ >= third && modZ < 2 * third)) {
                            grid.set(Vec3i(x, y, z), false, color);
                        }
                    }
                }
            }
        }
        
        grid.clearMeshCache();
    }
    
    // Helper function to check if a point is inside a polygon (for 2D shapes)
    static bool pointInPolygon(const Vec2f& point, const std::vector<Vec2f>& polygon) {
        bool inside = false;
        size_t n = polygon.size();
        
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            if (((polygon[i].y > point.y) != (polygon[j].y > point.y)) &&
                (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / 
                          (polygon[j].y - polygon[i].y) + polygon[i].x)) {
                inside = !inside;
            }
        }
        
        return inside;
    }
    
    // Utah Teapot (simplified voxel approximation)
    static void createTeapot(VoxelGrid& grid, const Vec3f& position, float scale = 1.0f,
                           const Vec3ui8& color = Vec3ui8(200, 200, 200)) {
        TIME_FUNCTION;
        
        // Simplified teapot using multiple primitive components
        Vec3f center = position;
        
        // Body (ellipsoid)
        createSphere(grid, center, 3.0f * scale, color, false);
        
        // Spout (rotated cylinder)
        Vec3f spoutStart = center + Vec3f(2.0f * scale, 0, 0);
        Vec3f spoutEnd = center + Vec3f(4.0f * scale, 1.5f * scale, 0);
        createCylinderBetween(grid, spoutStart, spoutEnd, 0.5f * scale, color, true);
        
        // Handle (semi-circle)
        Vec3f handleStart = center + Vec3f(-2.0f * scale, 0, 0);
        Vec3f handleEnd = center + Vec3f(-3.0f * scale, 2.0f * scale, 0);
        createCylinderBetween(grid, handleStart, handleEnd, 0.4f * scale, color, true);
        
        // Lid (small cylinder on top)
        Vec3f lidCenter = center + Vec3f(0, 3.0f * scale, 0);
        createCylinder(grid, lidCenter, 1.0f * scale, 0.5f * scale, color, true, 1);
        
        grid.clearMeshCache();
    }
    
    static void createCylinderBetween(VoxelGrid& grid, const Vec3f& start, const Vec3f& end, float radius,
                                     const Vec3ui8& color, bool filled = true) {
        TIME_FUNCTION;
        
        Vec3f direction = (end - start).normalized();
        float length = (end - start).length();
        
        // Create local coordinate system
        Vec3f up(0, 1, 0);
        if (std::abs(direction.dot(up)) > 0.99f) {
            up = Vec3f(1, 0, 0);
        }
        
        Vec3f right = direction.cross(up).normalized();
        Vec3f localUp = right.cross(direction).normalized();
        
        Vec3f minPos = start.min(end) - Vec3f(radius, radius, radius);
        Vec3f maxPos = start.max(end) + Vec3f(radius, radius, radius);
        
        Vec3i minVoxel = (minPos / grid.binSize).floorToI();
        Vec3i maxVoxel = (maxPos / grid.binSize).floorToI();
        
        minVoxel = minVoxel.max(Vec3i(0, 0, 0));
        maxVoxel = maxVoxel.min(Vec3i(grid.getWidth() - 1, grid.getHeight() - 1, grid.getDepth() - 1));
        
        float radiusSq = radius * radius;
        
        for (int z = minVoxel.z; z <= maxVoxel.z; ++z) {
            for (int y = minVoxel.y; y <= maxVoxel.y; ++y) {
                for (int x = minVoxel.x; x <= maxVoxel.x; ++x) {
                    Vec3f voxelPos(x * grid.binSize, y * grid.binSize, z * grid.binSize);
                    
                    // Project point onto cylinder axis
                    Vec3f toPoint = voxelPos - start;
                    float t = toPoint.dot(direction);
                    
                    // Check if within cylinder length
                    if (t < 0 || t > length) continue;
                    
                    Vec3f projected = start + direction * t;
                    Vec3f delta = voxelPos - projected;
                    float distanceSq = delta.lengthSquared();
                    
                    if (filled) {
                        if (distanceSq <= radiusSq) {
                            grid.set(Vec3i(x, y, z), true, color);
                        }
                    } else {
                        float shellThickness = grid.binSize;
                        if (distanceSq <= radiusSq && 
                            distanceSq >= (radius - shellThickness) * (radius - shellThickness)) {
                            grid.set(Vec3i(x, y, z), true, color);
                        }
                    }
                }
            }
        }
    }
    
    // Generate from mathematical function
    template<typename Func>
    static void createFromFunction(VoxelGrid& grid, Func func, 
                                  const Vec3f& minBounds, const Vec3f& maxBounds,
                                  float threshold = 0.5f,
                                  const Vec3ui8& color = Vec3ui8(255, 255, 255)) {
        TIME_FUNCTION;
        
        Vec3i minVoxel = (minBounds / grid.binSize).floorToI();
        Vec3i maxVoxel = (maxBounds / grid.binSize).floorToI();
        
        minVoxel = minVoxel.max(Vec3i(0, 0, 0));
        maxVoxel = maxVoxel.min(Vec3i(grid.getWidth() - 1, grid.getHeight() - 1, grid.getDepth() - 1));
        
        for (int z = minVoxel.z; z <= maxVoxel.z; ++z) {
            for (int y = minVoxel.y; y <= maxVoxel.y; ++y) {
                for (int x = minVoxel.x; x <= maxVoxel.x; ++x) {
                    Vec3f pos(x * grid.binSize, y * grid.binSize, z * grid.binSize);
                    
                    float value = func(pos.x, pos.y, pos.z);
                    
                    if (value > threshold) {
                        grid.set(Vec3i(x, y, z), true, color);
                    }
                }
            }
        }
        
        grid.clearMeshCache();
    }
    
    // Example mathematical functions
    static float sphereFunction(float x, float y, float z) {
        return 1.0f - (x*x + y*y + z*z);
    }
    
    static float torusFunction(float x, float y, float z, float R = 2.0f, float r = 1.0f) {
        float d = std::sqrt(x*x + y*y) - R;
        return r*r - d*d - z*z;
    }
    
    static float gyroidFunction(float x, float y, float z, float scale = 0.5f) {
        x *= scale; y *= scale; z *= scale;
        return std::sin(x) * std::cos(y) + std::sin(y) * std::cos(z) + std::sin(z) * std::cos(x);
    }
};

#endif 