
#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../vecmat/mat4.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <corecrt_math_defines.h>

struct Voxel {
    bool active;
    Vec3f color;
};

class VoxelGrid {
private:
    int width, height, depth;
    std::vector<Voxel> voxels;
    
public:
    VoxelGrid(int w, int h, int d) : width(w), height(h), depth(d) {
        voxels.resize(w * h * d);
        // Initialize all voxels as inactive
        for (auto& v : voxels) {
            v.active = false;
            v.color = Vec3f(0.5f, 0.5f, 0.5f);
        }
    }
    
    Voxel& get(int x, int y, int z) {
        return voxels[z * width * height + y * width + x];
    }
    
    const Voxel& get(int x, int y, int z) const {
        return voxels[z * width * height + y * width + x];
    }
    
    void set(int x, int y, int z, bool active, Vec3f color = Vec3f(0.8f, 0.3f, 0.2f)) {
        if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
            Voxel& v = get(x, y, z);
            v.active = active;
            v.color = color;
        }
    }
    
    // Create a simple test pattern
    void createTestPattern() {
        // Create a hollow cube
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                for (int z = 0; z < depth; z++) {
                    if (x == 0 || x == width-1 || y == 0 || y == height-1 || z == 0 || z == depth-1) {
                        float r = float(x) / width;
                        float g = float(y) / height;
                        float b = float(z) / depth;
                        set(x, y, z, true, Vec3f(r, g, b));
                    }
                }
            }
        }
        
        // Add some interior voxels
        for (int i = 5; i < 10; i++) {
            set(i, i, i, true, Vec3f(1.0f, 0.0f, 0.0f));
        }
    }
    
    // Amanatides & Woo ray-grid traversal algorithm
    bool rayCast(const Vec3f& rayOrigin, const Vec3f& rayDirection, float maxDistance, 
                 Vec3f& hitPos, Vec3f& hitNormal, Vec3f& hitColor) const {
        
        // Initialize step directions
        Vec3i step;
        Vec3f tMax, tDelta;
        
        // Current voxel coordinates
        Vec3f voxel = rayOrigin.floor();
        
        // Check if starting outside grid
        if (voxel.x < 0 || voxel.x >= width || 
            voxel.y < 0 || voxel.y >= height || 
            voxel.z < 0 || voxel.z >= depth) {
            
            // Calculate distance to grid bounds
            Vec3f t0, t1;
            for (int i = 0; i < 3; i++) {
                if (rayDirection[i] >= 0) {
                    t0[i] = (0 - rayOrigin[i]) / rayDirection[i];
                    t1[i] = (width - rayOrigin[i]) / rayDirection[i];
                } else {
                    t0[i] = (width - rayOrigin[i]) / rayDirection[i];
                    t1[i] = (0 - rayOrigin[i]) / rayDirection[i];
                }
            }
            
            float tEnter = t0.maxComp();
            float tExit = t1.minComp();
            
            if (tEnter > tExit || tExit < 0) {
                return false;
            }
            
            if (tEnter > 0) {
                voxel = Vec3f((rayOrigin + rayDirection * tEnter).floor());
            }
        }
        
        // Initialize step and tMax based on ray direction
        for (int i = 0; i < 3; i++) {
            if (rayDirection[i] < 0) {
                step[i] = -1;
                tMax[i] = ((float)voxel[i] - rayOrigin[i]) / rayDirection[i];
                tDelta[i] = -1.0f / rayDirection[i];
            } else {
                step[i] = 1;
                tMax[i] = ((float)(voxel[i] + 1) - rayOrigin[i]) / rayDirection[i];
                tDelta[i] = 1.0f / rayDirection[i];
            }
        }
        
        // Main traversal loop
        float distance = 0;
        while (distance < maxDistance) {
            // Check current voxel
            if (voxel.x >= 0 && voxel.x < width && 
                voxel.y >= 0 && voxel.y < height && 
                voxel.z >= 0 && voxel.z < depth) {
                
                const Voxel& current = get(voxel.x, voxel.y, voxel.z);
                if (current.active) {
                    // Hit found
                    hitPos = rayOrigin + rayDirection * distance;
                    hitColor = current.color;
                    
                    // Determine hit normal (which plane we hit)
                    if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
                        hitNormal = Vec3f(-step.x, 0, 0);
                    } else if (tMax.y <= tMax.z) {
                        hitNormal = Vec3f(0, -step.y, 0);
                    } else {
                        hitNormal = Vec3f(0, 0, -step.z);
                    }
                    
                    return true;
                }
            }
            
            // Move to next voxel
            if (tMax.x < tMax.y && tMax.x < tMax.z) {
                distance = tMax.x;
                tMax.x += tDelta.x;
                voxel.x += step.x;
            } else if (tMax.y < tMax.z) {
                distance = tMax.y;
                tMax.y += tDelta.y;
                voxel.y += step.y;
            } else {
                distance = tMax.z;
                tMax.z += tDelta.z;
                voxel.z += step.z;
            }
        }
        
        return false;
    }
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getDepth() const { return depth; }
};

float radians(float deg) {
    return deg * (M_PI / 180);
}

// Simple camera class
class Camera {
public:
    Vec3f position;
    Vec3f forward;
    Vec3f up;
    float fov;
    
    Camera() : position(0, 0, -10), forward(0, 0, 1), up(0, 1, 0), fov(45.0f) {}
    
    Mat4f getViewMatrix() const {
        return lookAt(position, position + forward, up);
    }
    
    Mat4f getProjectionMatrix(float aspectRatio) const {
        return perspective(radians(fov), aspectRatio, 0.1f, 100.0f);
    }
    
    void rotate(float yaw, float pitch) {
        forward = (Vec3f(
            cos(yaw) * cos(pitch),
            sin(pitch),
            sin(yaw) * cos(pitch)
        ).normalized());
    }
};

// Simple renderer using ray casting
class VoxelRenderer {
private:
    VoxelGrid grid;
    Camera camera;
    
public:
    VoxelRenderer() : grid(20, 20, 20) {
        grid.createTestPattern();
    }
    
    void render(int screenWidth, int screenHeight) {
        float aspectRatio = float(screenWidth) / float(screenHeight);
        
        // Get matrices
        Mat4f projection = camera.getProjectionMatrix(aspectRatio);
        Mat4f view = camera.getViewMatrix();
        Mat4f invViewProj = (projection * view).inverse();
        
        for (int y = 0; y < screenHeight; y++) {
            for (int x = 0; x < screenWidth; x++) {
                // Convert screen coordinates to normalized device coordinates
                float ndcX = (2.0f * x) / screenWidth - 1.0f;
                float ndcY = 1.0f - (2.0f * y) / screenHeight;
                
                // Create ray in world space using inverse view-projection
                Vec4f rayStartNDC = Vec4f(ndcX, ndcY, -1.0f, 1.0f);
                Vec4f rayEndNDC = Vec4f(ndcX, ndcY, 1.0f, 1.0f);
                
                // Transform to world space
                Vec4f rayStartWorld = invViewProj * rayStartNDC;
                Vec4f rayEndWorld = invViewProj * rayEndNDC;
                
                // Perspective divide
                rayStartWorld /= rayStartWorld.w;
                rayEndWorld /= rayEndWorld.w;
                
                // Calculate ray direction
                Vec3f rayStart = Vec3f(rayStartWorld.x, rayStartWorld.y, rayStartWorld.z);
                Vec3f rayEnd = Vec3f(rayEndWorld.x, rayEndWorld.y, rayEndWorld.z);
                Vec3f rayDir = (rayEnd - rayStart).normalized();
                
                // Perform ray casting (use camera position as ray origin)
                Vec3f hitPos, hitNormal, hitColor;
                if (grid.rayCast(camera.position, rayDir, 100.0f, hitPos, hitNormal, hitColor)) {
                    // Simple lighting
                    Vec3f lightDir = Vec3f(1, 1, -1).normalized();
                    float diffuse = std::max(static_cast<float>(hitNormal.dot(lightDir)), 0.2f);
                    
                    // Output pixel (simplified - in real OpenGL, set pixel color)
                    if (x == screenWidth/2 && y == screenHeight/2) {
                        std::cout << "Center ray hit at: " << hitPos.x << ", " 
                                << hitPos.y << ", " << hitPos.z << std::endl;
                    }
                }
            }
        }
    }
    
    void rotateCamera(float yaw, float pitch) {
        camera.rotate(yaw, pitch);
    }
    
    void moveCamera(const Vec3f& movement) {
        camera.position += movement;
    }
    
    Camera& getCamera() { return camera; }
};
