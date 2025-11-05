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
#include "util/timing_decorator.hpp"
#include "util/vec3.hpp"
#include "util/vec4.hpp"
#include "util/bmpwriter.hpp"
#include "util/voxelgrid.hpp"

std::vector<Vec3> generateSphere(int numPoints, float radius = 1.0f, float wiggleAmount = 0.1f) {
    
    printf("Generating sphere with %d points using grid method...\n", numPoints);
    printf("Wiggle amount: %.3f\n", wiggleAmount);
    
    std::vector<Vec3> points;
    
    // Random number generator for wiggling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Calculate grid resolution based on desired number of points
    // For a sphere, we need to account that only about 52% of points in a cube will be inside the sphere
    int gridRes = static_cast<int>(std::cbrt(numPoints / 0.52f)) + 1;
    
    printf("Using grid resolution: %d x %d x %d\n", gridRes, gridRes, gridRes);
    
    // Calculate voxel size for wiggling (based on average distance between points)
    float voxelSize = 2.0f / gridRes;
    float maxWiggle = wiggleAmount * voxelSize;
    
    printf("Voxel size: %.4f, Max wiggle: %.4f\n", voxelSize, maxWiggle);
    
    // Generate points in a cube from -1 to 1 in all dimensions
    for (int x = 0; x < gridRes; ++x) {
        for (int y = 0; y < gridRes; ++y) {
            for (int z = 0; z < gridRes; ++z) {
                // Convert grid coordinates to normalized cube coordinates [-1, 1]
                Vec3 point(
                    (2.0f * x / (gridRes - 1)) - 1.0f,
                    (2.0f * y / (gridRes - 1)) - 1.0f,
                    (2.0f * z / (gridRes - 1)) - 1.0f
                );
                
                // Check if point is inside the unit sphere
                if (point.lengthSquared() <= 1.0f) {
                    // Apply randomized wiggling
                    Vec3 wiggle(
                        dist(gen) * maxWiggle,
                        dist(gen) * maxWiggle,
                        dist(gen) * maxWiggle
                    );
                    
                    Vec3 wiggledPoint = point + wiggle;
                    
                    // Re-normalize to maintain spherical shape while preserving the wiggle
                    // This ensures the point stays within the sphere while having natural variation
                    float currentLength = wiggledPoint.length();
                    if (currentLength > 1.0f) {
                        // Scale back to unit sphere surface, but preserve the wiggle direction
                        wiggledPoint = wiggledPoint * (1.0f / currentLength);
                    }
                    
                    points.push_back(wiggledPoint * radius); // Scale by radius
                }
            }
        }
    }
    
    printf("Generated %zu points inside sphere\n", points.size());
    
    // If we have too many points, randomly sample down to the desired number
    if (points.size() > static_cast<size_t>(numPoints)) {
        printf("Sampling down from %zu to %d points...\n", points.size(), numPoints);
        
        // Shuffle and resize
        std::shuffle(points.begin(), points.end(), gen);
        points.resize(numPoints);
    }
    // If we have too few points, we'll use what we have
    else if (points.size() < static_cast<size_t>(numPoints)) {
        printf("Warning: Only generated %zu points (requested %d)\n", points.size(), numPoints);
    }
    
    return points;
}

// Alternative sphere generation with perlin-like noise for more natural wiggling
std::vector<Vec3> generateSphereWithNaturalWiggle(int numPoints, float radius = 1.0f, float noiseStrength = 0.15f) {
    
    printf("Generating sphere with natural wiggling using %d points...\n", numPoints);
    printf("Noise strength: %.3f\n", noiseStrength);
    
    std::vector<Vec3> points;
    
    // Random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Calculate grid resolution
    int gridRes = static_cast<int>(std::cbrt(numPoints / 0.52f)) + 1;
    printf("Using grid resolution: %d x %d x %d\n", gridRes, gridRes, gridRes);
    
    float voxelSize = 2.0f / gridRes;
    float maxDisplacement = noiseStrength * voxelSize;
    
    // Pre-compute some random offsets for pseudo-perlin noise
    std::vector<float> randomOffsets(gridRes * gridRes * gridRes);
    for (size_t i = 0; i < randomOffsets.size(); ++i) {
        randomOffsets[i] = dist(gen);
    }
    
    auto getNoise = [&](int x, int y, int z) -> float {
        int idx = (x * gridRes * gridRes) + (y * gridRes) + z;
        return randomOffsets[idx % randomOffsets.size()];
    };
    
    for (int x = 0; x < gridRes; ++x) {
        for (int y = 0; y < gridRes; ++y) {
            for (int z = 0; z < gridRes; ++z) {
                Vec3 point(
                    (2.0f * x / (gridRes - 1)) - 1.0f,
                    (2.0f * y / (gridRes - 1)) - 1.0f,
                    (2.0f * z / (gridRes - 1)) - 1.0f
                );
                
                if (point.lengthSquared() <= 1.0f) {
                    // Use smoother noise for more natural displacement
                    float noiseX = getNoise(x, y, z);
                    float noiseY = getNoise(y, z, x);
                    float noiseZ = getNoise(z, x, y);
                    
                    // Apply displacement that preserves the overall spherical structure
                    Vec3 displacement(
                        noiseX * maxDisplacement,
                        noiseY * maxDisplacement,
                        noiseZ * maxDisplacement
                    );
                    
                    Vec3 displacedPoint = point + displacement;
                    
                    // Ensure we don't push points too far from sphere surface
                    float currentLength = displacedPoint.length();
                    if (currentLength > 1.0f) {
                        displacedPoint = displacedPoint * (0.95f + 0.05f * dist(gen)); // Small random scaling
                    }
                    
                    points.push_back(displacedPoint * radius);
                }
            }
        }
    }
    
    printf("Generated %zu points with natural wiggling\n", points.size());
    
    // Sample down if needed
    if (points.size() > static_cast<size_t>(numPoints)) {
        printf("Sampling down from %zu to %d points...\n", points.size(), numPoints);
        std::shuffle(points.begin(), points.end(), gen);
        points.resize(numPoints);
    } else if (points.size() < static_cast<size_t>(numPoints)) {
        printf("Warning: Only generated %zu points (requested %d)\n", points.size(), numPoints);
    }
    
    return points;
}


// Modified function to populate voxel grid with sphere points and assign layers
void populateVoxelGridWithLayeredSphere(VoxelGrid& grid, const std::vector<Vec3>& points) {
    printf("Populating voxel grid with %zu sphere points...\n", points.size());
    
    // First add all voxels with a default color
    Vec4 defaultColor(1.0f, 1.0f, 1.0f, 1.0f); // Temporary color
    for (const auto& point : points) {
        grid.addVoxel(point, defaultColor);
    }
    
    printf("Voxel grid populated with %zu voxels\n", grid.getOccupiedPositions().size());
    
    // Now assign the planetary layers
    grid.assignPlanetaryLayers();
}

Vec3 rotateX(const Vec3& point, float angle) {
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);
    return Vec3(
        point.x,
        point.y * cosA - point.z * sinA,
        point.y * sinA + point.z * cosA
    );
}

Vec3 rotateY(const Vec3& point, float angle) {
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);
    return Vec3(
        point.x * cosA + point.z * sinA,
        point.y,
        -point.x * sinA + point.z * cosA
    );
}

Vec3 rotateZ(const Vec3& point, float angle) {
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);
    return Vec3(
        point.x * cosA - point.y * sinA,
        point.x * sinA + point.y * cosA,
        point.z
    );
}

void visualizePointCloud(const std::vector<Vec3>& points, const std::vector<Vec4>& colors, 
                                 const std::string& filename, int width = 1000, int height = 1000,
                                 float angleX = 0.0f, float angleY = 0.0f, float angleZ = 0.0f) {
    TIME_FUNCTION;
    std::vector<uint8_t> pixels(width * height * 3, 0);
    
    // Background color (dark blue)
    for (int i = 0; i < width * height * 3; i += 3) {
        pixels[i] = 30;     // B
        pixels[i + 1] = 30; // G
        pixels[i + 2] = 50; // R
    }
    
    // Find bounds of point cloud
    Vec3 minPoint( std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max());
    Vec3 maxPoint(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    
    // Apply rotation to all points and find new bounds
    std::vector<Vec3> rotatedPoints;
    rotatedPoints.reserve(points.size());
    
    for (const auto& point : points) {
        Vec3 rotated = point;
        rotated = rotateX(rotated, angleX);
        rotated = rotateY(rotated, angleY);
        rotated = rotateZ(rotated, angleZ);
        rotatedPoints.push_back(rotated);
        
        minPoint.x = std::min(minPoint.x, rotated.x);
        minPoint.y = std::min(minPoint.y, rotated.y);
        minPoint.z = std::min(minPoint.z, rotated.z);
        maxPoint.x = std::max(maxPoint.x, rotated.x);
        maxPoint.y = std::max(maxPoint.y, rotated.y);
        maxPoint.z = std::max(maxPoint.z, rotated.z);
    }
    
    Vec3 cloudSize = maxPoint - minPoint;
    float maxDim = std::max({cloudSize.x, cloudSize.y, cloudSize.z});
    
    // Draw points
    for (size_t i = 0; i < rotatedPoints.size(); i++) {
        const auto& point = rotatedPoints[i];
        const auto& color = colors[i];
        
        // Map 3D point to 2D screen coordinates (orthographic projection)
        int screenX = static_cast<int>(((point.x - minPoint.x) / maxDim) * (width - 20)) + 10;
        int screenY = static_cast<int>(((point.y - minPoint.y) / maxDim) * (height - 20)) + 10;
        
        if (screenX >= 0 && screenX < width && screenY >= 0 && screenY < height) {
            uint8_t r, g, b;
            color.toUint8(r, g, b);
            
            // Draw a 2x2 pixel for each point
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int px = screenX + dx;
                    int py = screenY + dy;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        int index = (py * width + px) * 3;
                        pixels[index] = b;
                        pixels[index + 1] = g;
                        pixels[index + 2] = r;
                    }
                }
            }
        }
    }
    
    BMPWriter::saveBMP(filename, pixels, width, height);
}

int main() {
    printf("=== Layered Sphere Generation and Visualization ===\n\n");
    
    const int numPoints = 100000000;
    const float radius = 2.0f;
    
    printf("Generating layered spheres with %d points each, radius %.1f\n\n", numPoints, radius);
    
    // Create voxel grid
    VoxelGrid grid(Vec3(10, 10, 10), Vec3(0.1f, 0.1f, 0.1f));
    
    // Generate sphere with different wiggle options:
    
    // Option 1: Simple random wiggling (small amount - 10% of voxel size)
    printf("1. Generating sphere with simple wiggling...\n");
    auto sphere1 = generateSphereWithNaturalWiggle(numPoints, radius, 0.1f);
    
    populateVoxelGridWithLayeredSphere(grid, sphere1);
    
    // Extract positions and colors for visualization
    std::vector<Vec3> occupiedPositions = grid.getOccupiedPositions();
    std::vector<Vec4> layerColors = grid.getColors();
    
    // Convert to world coordinates for visualization
    std::vector<Vec3> worldPositions;
    for (const auto& gridPos : occupiedPositions) {
        worldPositions.push_back(grid.gridToWorld(gridPos));
    }
    
    // Create multiple visualizations from different angles
    printf("\nGenerating views from different angles...\n");
    
    // Front view (0, 0, 0)
    visualizePointCloud(worldPositions, layerColors, "output/sphere_front.bmp", 1000, 1000, 0.0f, 0.0f, 0.0f);
    printf("  - sphere_front.bmp (front view)\n");
    
    // 45 degree rotation around Y axis
    visualizePointCloud(worldPositions, layerColors, "output/sphere_45y.bmp", 1000, 1000, 0.0f, M_PI/4, 0.0f);
    printf("  - sphere_45y.bmp (45° Y rotation)\n");
    
    // 90 degree rotation around Y axis (side view)
    visualizePointCloud(worldPositions, layerColors, "output/sphere_side.bmp", 1000, 1000, 0.0f, M_PI/2, 0.0f);
    printf("  - sphere_side.bmp (side view)\n");
    
    // 45 degree rotation around X axis (top-down perspective)
    visualizePointCloud(worldPositions, layerColors, "output/sphere_45x.bmp", 1000, 1000, M_PI/4, 0.0f, 0.0f);
    printf("  - sphere_45x.bmp (45° X rotation)\n");
    
    // Combined rotation (30° X, 30° Y)
    visualizePointCloud(worldPositions, layerColors, "output/sphere_30x_30y.bmp", 1000, 1000, M_PI/6, M_PI/6, 0.0f);
    printf("  - sphere_30x_30y.bmp (30° X, 30° Y rotation)\n");
    
    // Top view (90° X rotation)
    visualizePointCloud(worldPositions, layerColors, "output/sphere_top.bmp", 1000, 1000, M_PI/2, 0.0f, 0.0f);
    printf("  - sphere_top.bmp (top view)\n");
    
    printf("\n=== Sphere generated successfully ===\n");
    printf("Files created in output/ directory:\n");
    printf("  - sphere_front.bmp     (Front view)\n");
    printf("  - sphere_45y.bmp       (45° Y rotation)\n");
    printf("  - sphere_side.bmp      (Side view)\n");
    printf("  - sphere_45x.bmp       (45° X rotation)\n");
    printf("  - sphere_30x_30y.bmp   (30° X, 30° Y rotation)\n");
    printf("  - sphere_top.bmp       (Top view)\n");
    
    return 0;
}