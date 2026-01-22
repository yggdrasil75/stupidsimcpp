#ifndef GRID3_HPP
#define GRID3_HPP

#include <unordered_map>
#include <fstream>
#include <cstring>
#include <memory>
#include <array>
#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "../noise/pnoise2.hpp"
#include "../vecmat/mat4.hpp"
//#include "../vecmat/mat3.hpp"
#include <vector>
#include <algorithm>
#include "../basicdefines.hpp"
#include <cfloat> 

//constexpr char magic[4] = {'Y', 'G', 'G', '3'};
static constexpr int CHUNK_THRESHOLD = 16; //at this size, subdivide.

Mat4f lookAt(const Vec3f& eye, const Vec3f& center, const Vec3f& up) {
    Vec3f const f = (center - eye).normalized();
    Vec3f const s = f.cross(up).normalized();
    Vec3f const u = s.cross(f);

    Mat4f Result = Mat4f::identity();
    Result(0, 0) = s.x;
    Result(1, 0) = s.y;
    Result(2, 0) = s.z;
    Result(3, 0) = -s.dot(eye);
    Result(0, 1) = u.x;
    Result(1, 1) = u.y;
    Result(2, 1) = u.z;
    Result(3, 1) = -u.dot(eye);
    Result(0, 2) = -f.x;
    Result(1, 2) = -f.y;
    Result(2, 2) = -f.z;
    Result(3, 2) = f.dot(eye);
    return Result;
}

Mat4f perspective(float fovy, float aspect, float zNear, float zfar) {
    float const tanhalfF = tan(fovy / 2);
    Mat4f Result = 0;
    Result(0,0) = 1 / (aspect * tanhalfF);
    Result(1,1) = 1 / tanhalfF;
    Result(2,2) = zfar / (zNear - zfar);
    Result(2,3) = -1;
    Result(3,2) = -(zfar * zNear) / (zfar - zNear);
    return Result;
}

struct Voxel {
    float weight = 1.0;
    bool active = false;
    float alpha = 0.0;
    Vec3ui8 color = Vec3ui8(0,0,0);
    Voxel() = default;
    Voxel(float weight, bool active, float alpha, Vec3ui8 color) : weight(weight), active(active), alpha(alpha), color(color) {}
    // TODO: add curving and similar for water and glass and so on.
    auto members() const -> std::tuple<const float&, const bool&, const float&, const Vec3ui8&> {
        return std::tie(weight, active, alpha, color);
    }
    
    auto members() -> std::tuple<float&, bool&, float&, Vec3ui8&> {
        return std::tie(weight, active, alpha, color);
    }
};

struct Camera {
    Ray3f posfor;
    Vec3f up;
    float fov;
    Camera(Vec3f pos, Vec3f viewdir, Vec3f up, float fov = 80) : posfor(Ray3f(pos, viewdir)), up(up), fov(fov) {}
    
    void rotateYaw(float angle) {
        float cosA = cos(angle);
        float sinA = sin(angle);
        
        Vec3f right = posfor.direction.cross(up).normalized();
        posfor.direction = posfor.direction * cosA + right * sinA;
        posfor.direction = posfor.direction.normalized();
    }
    
    void rotatePitch(float angle) {
        float cosA = cos(angle);
        float sinA = sin(angle);
        
        Vec3f right = posfor.direction.cross(up).normalized();
        posfor.direction = posfor.direction * cosA + up * sinA;
        posfor.direction = posfor.direction.normalized();
        
        up = right.cross(posfor.direction).normalized();
    }

    Vec3f forward() const {
        return (posfor.direction - posfor.origin).normalized();
    }

    Vec3f right() const {
        return forward().cross(up).normalized();
    }

    float fovRad() const {
        return fov * (M_PI / 180);
    }
};

struct Chunk {
    Voxel reprVoxel; //average of all voxels in chunk for LOD rendering
    //std::vector<Voxel> voxels; //list of all voxels in chunk.
    std::vector<Chunk> children; //list of all chunks in chunk
    bool active; //active if any child chunk or child voxel is active. used to efficiently find active voxels by only going down when an active chunk is found.
    int chunkSize; //should be (CHUNK_THRESHOLD/2) * 2 ^ depth I think. (ie: 1 depth will be (16/2)*(2^1) or 16, second will be (16/2)*(2^2) or 8*4=32)
    Vec3i minCorner; //position of chunk in world space.
    int depth; //number of parent/child traversals to get here. 
};

class VoxelGrid {
private:
    Vec3i gridSize;
    std::vector<Voxel> voxels;
    std::unordered_map<Vec3i, Chunk, Vec3i::Hash> chunkList;
    std::unordered_map<Vec3i, bool, Vec3i::Hash> activeChunks;
    int xyPlane;
    
    float radians(float rads) {
        return rads * (M_PI / 180);
    }
    
    Vec3i getChunkCoord(const Vec3i& voxelPos) const {
        return voxelPos / CHUNK_THRESHOLD;
    }

    void updateChunkStatus(const Vec3i& pos, bool isActive) {
        Vec3i chunkCoord = getChunkCoord(pos);

        if (isActive) {
            chunkList[chunkCoord].active = true;
            activeChunks[chunkCoord] = true;
        }
    }

    // Slab method for AABB intersection
    bool intersectRayAABB(const Vec3f& origin, const Vec3f& dir, const Vec3f& boxMin, const Vec3f& boxMax, float& tNear, float& tFar) const {
        Vec3f invDir(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
        
        float t1 = (boxMin.x - origin.x) * invDir.x;
        float t2 = (boxMax.x - origin.x) * invDir.x;
        
        float tMin = std::min(t1, t2);
        float tMax = std::max(t1, t2);
        
        t1 = (boxMin.y - origin.y) * invDir.y;
        t2 = (boxMax.y - origin.y) * invDir.y;
        
        tMin = std::max(tMin, std::min(t1, t2));
        tMax = std::min(tMax, std::max(t1, t2));
        
        t1 = (boxMin.z - origin.z) * invDir.z;
        t2 = (boxMax.z - origin.z) * invDir.z;
        
        tMin = std::max(tMin, std::min(t1, t2));
        tMax = std::min(tMax, std::max(t1, t2));
        
        tNear = tMin;
        tFar = tMax;
        
        return tMax >= tMin && tMax >= 0.0f;
    }

public:
    VoxelGrid() : gridSize(0,0,0) {
        std::cout << "creating empty grid." << std::endl;
    }

    VoxelGrid(int w, int h, int d) : gridSize(w,h,d) {
        voxels.resize(w * h * d);
    }
    
    bool serializeToFile(const std::string& filename);
    
    static std::unique_ptr<VoxelGrid> deserializeFromFile(const std::string& filename);
    
    Voxel& get(int x, int y, int z) {
        return voxels[z * xyPlane + y * gridSize.x + x];
    }

    const Voxel& get(int x, int y, int z) const {
        return voxels[z * xyPlane + y * gridSize.x + x];
    }

    Voxel& get(const Vec3i& xyz) {
        return get(xyz.x, xyz.y, xyz.z);
    }

    const Voxel& get(const Vec3i& xyz) const {
        return get(xyz.x, xyz.y, xyz.z);
    }

    void resize(int newW, int newH, int newD) {
        std::vector<Voxel> newVoxels(newW * newH * newD);
        
        std::unordered_map<Vec3i, Chunk, Vec3i::Hash> chunklist;
        std::unordered_map<Vec3i, bool, Vec3i::Hash> newActiveChunks;

        int copyW = std::min(static_cast<int>(gridSize.x), newW);
        int copyH = std::min(static_cast<int>(gridSize.y), newH);
        int copyD = std::min(static_cast<int>(gridSize.z), newD);
        for (int z = 0; z < copyD; ++z) {
            for (int y = 0; y < copyH; ++y) {
                int oldRowStart = z * gridSize.x * gridSize.y + y * gridSize.x;
                int newRowStart = z * newW * newH + y * newW;
                
                std::copy(
                    voxels.begin() + oldRowStart, 
                    voxels.begin() + oldRowStart + copyW, 
                    newVoxels.begin() + newRowStart
                );

                for (int x = 0; x < copyW; ++x) {
                    if (voxels[oldRowStart + x].active) {
                        Vec3i cc(x / CHUNK_THRESHOLD, y / CHUNK_THRESHOLD, z / CHUNK_THRESHOLD);
                        newActiveChunks[cc] = true;
                    }
                }
            }
        }
        voxels = std::move(newVoxels);
        activeChunks = std::move(newActiveChunks);
        gridSize = Vec3i(newW, newH, newD);
        xyPlane = gridSize.x * gridSize.y;
    }

    void resize(Vec3i newsize) {
        resize(newsize.x, newsize.y, newsize.z);
    }

    void set(int x, int y, int z, bool active, Vec3ui8 color, float alpha = 1) {
        set(Vec3i(x,y,z), active, color, alpha);
    }
    
    void set(Vec3i pos, bool active, Vec3ui8 color, float alpha = 1.f) {
        if (pos.AllGTE(0)) {
            if (pos.AnyGTE(gridSize)) {
                resize(gridSize.max(pos));
            }
            
            Voxel& v = get(pos);
            v.active = active;
            v.color = color;
            v.alpha = alpha;
            updateChunkStatus(pos, active);
        }
    }
    
    void setBatch(const std::vector<Vec3i>& positions, bool active, Vec3ui8 color, float alpha = 1.0f) {
        // First, ensure grid is large enough
        Vec3i maxPos(0,0,0);
        for (const auto& pos : positions) {
            maxPos = maxPos.max(pos);
        }
        
        if (maxPos.AnyGTE(gridSize)) {
            resize(maxPos);
        }
        
        // Set all positions
        for (const auto& pos : positions) {
            Voxel& v = get(pos);
            v.active = active;
            v.color = color;
            v.alpha = alpha;
            updateChunkStatus(pos, active);
        }
    }

    bool inGrid(Vec3i voxl) const {
        return voxl.AllGTE(0) && voxl.AllLT(gridSize);
    }

    void voxelTraverse(const Vec3f& origin, const Vec3f& end, Voxel& outVoxel, Vec3i& step, const Vec3f& ray, Vec3f& tMax, int maxDist = 10000000) const {
        Vec3i cv = origin.floorToI();
        Vec3i lv = end.floorToI();
        step = Vec3i(ray.x >= 0 ? 1 : -1, ray.y >= 0 ? 1 : -1, ray.z >= 0 ? 1 : -1);
        Vec3f tDelta = Vec3f(ray.x != 0 ? std::abs(1.0f / ray.x) : INF,
                             ray.y != 0 ? std::abs(1.0f / ray.y) : INF,
                             ray.z != 0 ? std::abs(1.0f / ray.z) : INF);

        float dist = 0.0f; 
        outVoxel.alpha = 0.0;
        
        while (lv != cv && inGrid(cv) && outVoxel.alpha < 1.f) { 
            
            const Voxel& curv = get(cv);
            if (curv.active) {
                outVoxel.active = true;
                float remainingOpacity = 1.f - outVoxel.alpha;
                float contribution = curv.alpha * remainingOpacity;
                //Vec3f curC = curv.color.toFloat();
                if (outVoxel.alpha < EPSILON) {
                    outVoxel.color = curv.color;
                } else {
                    outVoxel.color = outVoxel.color + (curv.color * remainingOpacity);
                }
                outVoxel.alpha += contribution;
            }
            
            // Step Logic
            int axis = (tMax.x < tMax.y) ? 
                    ((tMax.x < tMax.z) ? 0 : 2) :
                    ((tMax.y < tMax.z) ? 1 : 2);
            
            switch(axis) {
                case 0: 
                    tMax.x += tDelta.x; 
                    cv.x += step.x; 
                    break;
                case 1: 
                    tMax.y += tDelta.y; 
                    cv.y += step.y; 
                    break;
                case 2: 
                    tMax.z += tDelta.z; 
                    cv.z += step.z; 
                    break;
                }
        }
        //outVoxel.color = newC 
        return;
    }

    int getWidth() const {
        return gridSize.x;
    }
    int getHeight() const {
        return gridSize.y;
    }
    int getDepth() const {
        return gridSize.z;
    }
    
    frame renderFrame(const Camera& cam, Vec2i resolution, frame::colormap colorformat = frame::colormap::RGB) const {
        TIME_FUNCTION;
        Vec3f forward = cam.forward();
        Vec3f right = cam.right();
        Vec3f up = cam.up;
        
        float aspect = resolution.aspect();
        float viewH = tan(cam.fov * 0.5f);
        float viewW = viewH * aspect;
        float maxDist = std::sqrt(gridSize.lengthSquared());
        Vec3i step;
        
        // Defines the bounds of the grid for AABB checking
        Vec3f gridMin(0, 0, 0);
        std::array<Vec3i, 8> precomputedSteps;
        int baseQuadrant = 0;
        
        baseQuadrant = forward.calculateInvOctantMask();
        
        precomputedSteps[0] = Vec3i(1, 1, 1);   // +++
        precomputedSteps[1] = Vec3i(-1, 1, 1);  // -++
        precomputedSteps[2] = Vec3i(1, -1, 1);  // +-+
        precomputedSteps[3] = Vec3i(-1, -1, 1); // --+
        precomputedSteps[4] = Vec3i(1, 1, -1);  // ++-
        precomputedSteps[5] = Vec3i(-1, 1, -1); // -+-
        precomputedSteps[6] = Vec3i(1, -1, -1); // +--
        precomputedSteps[7] = Vec3i(-1, -1, -1);// ---
        std::array<Vec3f, 8> precomputedTMax;
                
        Vec3f floored = cam.posfor.origin.floor();
        Vec3f dNext = floored + 1.f - cam.posfor.origin;
        Vec3f dPrev = cam.posfor.origin - floored;
        
        precomputedTMax[0] = Vec3f(dNext.x, dNext.y, dNext.z);
        precomputedTMax[1] = Vec3f(dPrev.x, dNext.y, dNext.z);
        precomputedTMax[2] = Vec3f(dNext.x, dPrev.y, dNext.z);
        precomputedTMax[3] = Vec3f(dPrev.x, dPrev.y, dNext.z);
        precomputedTMax[4] = Vec3f(dNext.x, dNext.y, dPrev.z);
        precomputedTMax[5] = Vec3f(dPrev.x, dNext.y, dPrev.z);
        precomputedTMax[6] = Vec3f(dNext.x, dPrev.y, dPrev.z);
        precomputedTMax[7] = Vec3f(dPrev.x, dPrev.y, dPrev.z);

        frame outFrame(resolution.x, resolution.y, colorformat);
        std::vector<uint8_t> colorBuffer;
        if (colorformat == frame::colormap::RGB) {
            colorBuffer.resize(resolution.x * resolution.y * 3);
        } else {
            colorBuffer.resize(resolution.x * resolution.y * 4);
        }

        #pragma omp parallel for
        for (int y = 0; y < resolution.y; y++) {
            float v = (1.f - 2.f * (y+0.5f) / resolution.y) * viewH;
            Vec3f vup = up * v;
            int yQuad = baseQuadrant;
            if (v < 0) yQuad ^= 2;
            for (int x = 0; x < resolution.x; x++) {
                Voxel outVoxel(0, false, 0.f, Vec3ui8(10, 10, 255));
                float u = (2.f * (x+0.5f)/resolution.x - 1.f) * viewW;
                Vec3f rayDirWorld = (forward + right * u + vup).normalized();
                float tNear = 0.0f;
                float tFar = maxDist;
                bool hit = intersectRayAABB(cam.posfor.origin, rayDirWorld, gridMin, gridSize, tNear, tFar);
                if (!hit) {
                    Vec3ui8 hitColor = outVoxel.color;
                    
                    // Set pixel color in buffer
                    switch (colorformat) {
                        case frame::colormap::BGRA: {
                            int idx = (y * resolution.y + x) * 4;
                            colorBuffer[idx + 3] = hitColor.x;
                            colorBuffer[idx + 2] = hitColor.y;
                            colorBuffer[idx + 1] = hitColor.z;
                            colorBuffer[idx + 0] = 255;
                            break;
                        }
                        case frame::colormap::RGB:
                        default: {
                            int idx = (y * resolution.y + x) * 3;
                            colorBuffer[idx] = hitColor.x;
                            colorBuffer[idx + 1] = hitColor.y;
                            colorBuffer[idx + 2] = hitColor.z;
                            break;
                        }
                    }
                    continue;
                }

                Vec3f rayStartGrid = cam.posfor.origin; //  + rayDirWorld * tNear;
                Vec3f rayEnd = rayStartGrid + rayDirWorld * tFar;
                int xQuad = yQuad;
                if (u < 0) xQuad ^= 1;
                step = precomputedSteps[xQuad];
                Vec3f tMaxBase = precomputedTMax[xQuad];
                Vec3f ray = rayEnd - rayStartGrid;
                Vec3f tMax(
                    ray.x != 0 ? tMaxBase.x / std::abs(ray.x) : INF,
                    ray.y != 0 ? tMaxBase.y / std::abs(ray.y) : INF,
                    ray.z != 0 ? tMaxBase.z / std::abs(ray.z) : INF
                );

                voxelTraverse(rayStartGrid, rayEnd, outVoxel, step, ray, tMax, maxDist);
                Vec3ui8 hitColor = outVoxel.color;
                // Set pixel color in buffer
                switch (colorformat) {
                    case frame::colormap::BGRA: {
                        int idx = (y * resolution.y + x) * 4;
                        colorBuffer[idx + 3] = hitColor.x;
                        colorBuffer[idx + 2] = hitColor.y;
                        colorBuffer[idx + 1] = hitColor.z;
                        colorBuffer[idx + 0] = 255;
                        break;
                    }
                    case frame::colormap::RGB:
                    default: {
                        int idx = (y * resolution.y + x) * 3;
                        colorBuffer[idx] = hitColor.x;
                        colorBuffer[idx + 1] = hitColor.y;
                        colorBuffer[idx + 2] = hitColor.z;
                        break;
                    }
                }
            }
        }
        
        outFrame.setData(colorBuffer);
        return outFrame;
    }

    void printStats() const {
        int totalVoxels = gridSize.x * gridSize.y * gridSize.z;
        int activeVoxels = 0;
        
        // Count active voxels
        for (const Voxel& voxel : voxels) {
            if (voxel.active) {
                activeVoxels++;
            }
        }
        
        float activePercentage = (totalVoxels > 0) ? 
            (static_cast<float>(activeVoxels) / static_cast<float>(totalVoxels)) * 100.0f : 0.0f;
        
        std::cout << "=== Voxel Grid Statistics ===" << std::endl;
        std::cout << "Grid dimensions: " << gridSize.x << " x " << gridSize.y << " x " << gridSize.z << std::endl;
        std::cout << "Total voxels: " << totalVoxels << std::endl;
        std::cout << "Active voxels: " << activeVoxels << std::endl;
        std::cout << "Inactive voxels: " << (totalVoxels - activeVoxels) << std::endl;
        std::cout << "Active chunks (map size): " << activeChunks.size() << std::endl;
        std::cout << "Active percentage: " << activePercentage << "%" << std::endl;
        std::cout << "Memory usage (approx): " << (voxels.size() * sizeof(Voxel)) / 1024 << " KB" << std::endl;
        std::cout << "============================" << std::endl;
    }
    
    std::vector<frame> genSlices(frame::colormap colorFormat = frame::colormap::RGB) const {
        TIME_FUNCTION;
        int colors;
        std::vector<frame> outframes;
        switch (colorFormat) {
            case frame::colormap::RGBA:
            case frame::colormap::BGRA: {
                colors = 4;
                break;
            }
            case frame::colormap::B: {
                colors = 1;
                break;
            }
            case frame::colormap::RGB:
            case frame::colormap::BGR:
            default: {
                colors = 3;
                break;
            }
        }
        int cbsize = gridSize.x * gridSize.y * colors;
        for (int layer = 0; layer < getDepth(); layer++) {
            int layerMult = layer * gridSize.x * gridSize.y;
            frame layerFrame(gridSize.x, gridSize.y, colorFormat);
            std::vector<uint8_t> colorBuffer(cbsize);
            for (int y = 0; y < gridSize.y; y++) {
                int yMult = layerMult + (y * gridSize.x);
                for (int x = 0; x < gridSize.x; x++)  {
                    int vidx = yMult + x;
                    int pidx = (y * gridSize.x + x) * colors;
                    Voxel cv = voxels[vidx];
                    Vec3ui8 cvColor;
                    float cvAlpha;
                    if (cv.active) {
                        cvColor = cv.color;
                        cvAlpha = cv.alpha;
                    } else {
                        cvColor = Vec3ui8(255,255,255);
                        cvAlpha = 255;
                    }
                    switch (colorFormat) {
                        case frame::colormap::RGBA: {
                            colorBuffer[pidx + 0] = cvColor.x;
                            colorBuffer[pidx + 1] = cvColor.y;
                            colorBuffer[pidx + 2] = cvColor.z;
                            colorBuffer[pidx + 3] = cvAlpha;
                            break;
                        }
                        case frame::colormap::BGRA: {
                            colorBuffer[pidx + 3] = cvColor.x;
                            colorBuffer[pidx + 2] = cvColor.y;
                            colorBuffer[pidx + 1] = cvColor.z;
                            colorBuffer[pidx + 0] = cvAlpha;
                            break;
                        }
                        case frame::colormap::RGB: {
                            colorBuffer[pidx + 0] = cvColor.x;
                            colorBuffer[pidx + 1] = cvColor.y;
                            colorBuffer[pidx + 2] = cvColor.z;
                            break;
                        }
                        case frame::colormap::BGR: {
                            colorBuffer[pidx + 2] = cvColor.x;
                            colorBuffer[pidx + 1] = cvColor.y;
                            colorBuffer[pidx + 0] = cvColor.z;
                            break;
                        }
                        case frame::colormap::B: {
                            colorBuffer[pidx] = static_cast<uint8_t>((cvColor.x * 0.299) + (cvColor.y * 0.587) + (cvColor.z * 0.114));
                            break;
                        }
                    }
                }
            }
            layerFrame.setData(colorBuffer);
            //layerFrame.compressFrameLZ78();
            outframes.emplace_back(layerFrame);   
        }
        return outframes;
    }
};
//#include "g3_serialization.hpp" needed to be usable

#endif
