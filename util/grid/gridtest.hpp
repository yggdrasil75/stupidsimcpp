#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../vecmat/mat4.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "../output/frame.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct Voxel {
    bool active;
    Vec3f color;
};

class VoxelGrid {
private:
    
public:
    int width, height, depth;
    std::vector<Voxel> voxels;
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
    
    // Amanatides & Woo ray-grid traversal algorithm
    bool rayCast(const Vec3f& rayOrigin, const Vec3f& rayDirection, float maxDistance, 
                 Vec3f& hitPos, Vec3f& hitNormal, Vec3f& hitColor) const {
        
        // Initialize step directions
        Vec3f step;
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
        int maxSteps = width + height + depth;
        int steps = 0;
        while (distance < maxDistance && steps < maxSteps) {
            steps++;
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
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) {
                    distance = tMax.x;
                    tMax.x += tDelta.x;
                    voxel.x += step.x;
                } else {
                    distance = tMax.z;
                    tMax.z += tDelta.z;
                    voxel.z += step.z;
                }
            } else {
                if (tMax.y < tMax.z) {
                    distance = tMax.y;
                    tMax.y += tDelta.y;
                    voxel.y += step.y;
                } else {
                    distance = tMax.z;
                    tMax.z += tDelta.z;
                    voxel.z += step.z;
                }
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
        // Clamp pitch to avoid gimbal lock
        pitch = std::clamp(pitch, -89.0f * (static_cast<float>(M_PI) / 180.0f), 89.0f * (static_cast<float>(M_PI) / 180.0f));
        
        forward = Vec3f(
            cos(yaw) * cos(pitch),
            sin(pitch),
            sin(yaw) * cos(pitch)
        ).normalized();
    }
};

// Simple renderer using ray casting
class VoxelRenderer {
private:
    
public:
    VoxelGrid grid;
    Camera camera;
    VoxelRenderer() : grid(20, 20, 20) { }
    
    // Render to a frame object
    frame renderToFrame(int screenWidth, int screenHeight) {
        
        // Create a frame with RGBA format for color + potential transparency
        frame renderedFrame(screenWidth, screenHeight, frame::colormap::RGBA);
        
        float aspectRatio = float(screenWidth) / float(screenHeight);
        
        // Get matrices
        Mat4f projection = camera.getProjectionMatrix(aspectRatio);
        Mat4f view = camera.getViewMatrix();
        Mat4f invViewProj = (projection * view).inverse();
        
        // Get reference to frame data for direct pixel writing
        std::vector<uint8_t>& frameData = const_cast<std::vector<uint8_t>&>(renderedFrame.getData());
        
        // Background color (dark gray)
        Vec3f backgroundColor(0.1f, 0.1f, 0.1f);
        
        // Simple light direction
        Vec3f lightDir = Vec3f(1, 1, -1).normalized();
        
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
                
                // Perform ray casting
                Vec3f hitPos, hitNormal, hitColor;
                bool hit = grid.rayCast(camera.position, rayDir, 100.0f, hitPos, hitNormal, hitColor);
                
                // Calculate pixel index in frame data
                int pixelIndex = (y * screenWidth + x) * 4; // 4 channels for RGBA
                
                if (hit) {
                    // Simple lighting
                    float diffuse = std::max(hitNormal.dot(lightDir), 0.2f);
                    
                    // Apply lighting to color
                    Vec3f finalColor = hitColor * diffuse;
                    
                    // Clamp color values to [0, 1]
                    finalColor.x = std::max(0.0f, std::min(1.0f, finalColor.x));
                    finalColor.y = std::max(0.0f, std::min(1.0f, finalColor.y));
                    finalColor.z = std::max(0.0f, std::min(1.0f, finalColor.z));
                    
                    // Convert to 8-bit RGB and set pixel
                    frameData[pixelIndex]     = static_cast<uint8_t>(finalColor.x * 255); // R
                    frameData[pixelIndex + 1] = static_cast<uint8_t>(finalColor.y * 255); // G
                    frameData[pixelIndex + 2] = static_cast<uint8_t>(finalColor.z * 255); // B
                    frameData[pixelIndex + 3] = 255; // A (fully opaque)
                    
                    // Debug: mark center pixel with a crosshair
                    if (x == screenWidth/2 && y == screenHeight/2) {
                        // White crosshair for center
                        frameData[pixelIndex]     = 255;
                        frameData[pixelIndex + 1] = 255;
                        frameData[pixelIndex + 2] = 255;
                        std::cout << "Center ray hit at: " << hitPos.x << ", " 
                                << hitPos.y << ", " << hitPos.z << std::endl;
                    }
                } else {
                    // Background color
                    frameData[pixelIndex]     = static_cast<uint8_t>(backgroundColor.x * 255);
                    frameData[pixelIndex + 1] = static_cast<uint8_t>(backgroundColor.y * 255);
                    frameData[pixelIndex + 2] = static_cast<uint8_t>(backgroundColor.z * 255);
                    frameData[pixelIndex + 3] = 255; // A
                }
            }
        }
        
        return renderedFrame;
    }
    
    // Overload for backward compatibility (calls the new method and discards frame)
    void render(int screenWidth, int screenHeight) {
        frame tempFrame = renderToFrame(screenWidth, screenHeight);
        // Optional: print info about the rendered frame
        std::cout << "Rendered frame: " << tempFrame << std::endl;
    }
    
    void rotateCamera(float yaw, float pitch) {
        camera.rotate(yaw, pitch);
    }
    
    void moveCamera(const Vec3f& movement) {
        camera.position += movement;
    }
    
    Camera& getCamera() { return camera; }
    
    // Get reference to the voxel grid (for testing/debugging)
    VoxelGrid& getGrid() { return grid; }
};