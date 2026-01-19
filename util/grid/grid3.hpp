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
#include <vector>
#include <algorithm>
#include "../basicdefines.hpp"

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

struct Vertex {
    Vec3f position;
    Vec3f normal;
    Vec3ui8 color;
    Vec2f texCoord;
    
    Vertex() = default;
    Vertex(Vec3f pos, Vec3f norm, Vec3ui8 colr, Vec2f tex = Vec2f(0,0)) : position(pos), normal(norm), color(colr), texCoord(tex) {}
};

struct Chunk {
    float weight;
    bool active;
    float alpha;
    Vec3ui8 color;
    std::vector<Voxel> voxels;
    std::vector<Chunk> children;
    Vec3i chunkSize;
    Vec3i minCorner;
    Vec3i maxCorner;
    int depth;
    bool dirty = false;

    Chunk(Vec3i minCorner, Vec3i maxCorner, int depth = 0) 
      : minCorner(minCorner), maxCorner(maxCorner), depth(depth), active(false) {
        chunkSize = maxCorner - minCorner;
        if (chunkSize.x > CHUNK_THRESHOLD || chunkSize.y > CHUNK_THRESHOLD || chunkSize.z > CHUNK_THRESHOLD) {
            subdivide();
        } else {
            voxels.resize(chunkSize.x * chunkSize.y * chunkSize.z);
        }
    }

    bool inChunk(Vec3i pos) const {
        if (pos.AllGT(minCorner) && pos.AllLT(maxCorner)) return true;
        return false;
    }

    bool set(Vec3i pos, Voxel newVox) {
        if (!inChunk(pos)) return false;

        if (children.empty()) {
            int index = getVoxelIndex(pos);
            if (index >= 0 && index < static_cast<int>(voxels.size())) {
                voxels[index] = newVox;
                if (newVox.active) {
                    active = true;
                }
                dirty = true;
                return true;
            }
            return false;
        }

        for (Chunk& child : children) {
            if (child.set(pos, newVox)) {
                if (newVox.active) {
                    active = true;
                }
                dirty = true;
                return true;
            }
        }
        return false;
    }

    Voxel* getPtr(Vec3i pos, int maxDepth = 0) {
        if (!inChunk(pos)) {
            return nullptr;
        }

        if (children.empty() || (maxDepth > 0 && depth >= maxDepth)) {
            int index = getVoxelIndex(pos);
            if (index >= 0 && index < static_cast<int>(voxels.size())) {
                return &voxels[index];
            }
            return nullptr;
        }
        
        for (Chunk& child : children) {
            if (child.inChunk(pos)) {
                return child.getPtr(pos, maxDepth);
            }
        }
        return nullptr;
    }
    
    Voxel get(Vec3i pos, int maxDepth = 0)  {
        if (!inChunk(pos)) {
            return Voxel();
        }

        if (children.empty() || (maxDepth > 0 && depth >= maxDepth)) {
            int index = getVoxelIndex(pos);
            if (index >= 0 && index < static_cast<int>(voxels.size())) {
                if (dirty) {
                    updateAveragesRecursive();
                }
                return voxels[index];
            }
            return Voxel();
        } else if (maxDepth > 0 && depth >= maxDepth) {
            return Voxel(weight, active, alpha, color);
        }
        
        for (Chunk& child : children) {
            if (child.inChunk(pos)) {
                if (dirty) {
                    updateAveragesRecursive();
                }
                return child.get(pos, maxDepth);
            }
        }
        return Voxel();
    }

    Voxel get(const Vec3i& pos, int maxDepth = 0) const {
        if (!inChunk(pos)) {
            return Voxel();
        }

        if (children.empty() || (maxDepth > 0 && depth >= maxDepth)) {
            int index = getVoxelIndex(pos);
            if (index >= 0 && index < static_cast<int>(voxels.size())) {
                if (dirty) {
                    std::cout << "need to clean chunk" << std::endl;
                }
                return voxels[index];
            }
            return Voxel();
        } else if (maxDepth > 0 && depth >= maxDepth) {
            return Voxel(weight, active, alpha, color);
        }
        
        for (const Chunk& child : children) {
            if (child.inChunk(pos)) {
                return child.get(pos, maxDepth);
            }
        }
        return Voxel();
    }
    
    int getVoxelIndex(Vec3i pos) const {
        Vec3i localPos = pos - minCorner;
        return localPos.z * chunkSize.x * chunkSize.y + localPos.y * chunkSize.x + localPos.x;
    }

    void updateAverages() {
        TIME_FUNCTION;
        if (voxels.empty()) {
            weight = 0.0f;
            alpha = 0.0f;
            color = Vec3ui8(0, 0, 0);
            active = false;
            return;
        }
        
        float totalWeight = 0.0f;
        float totalAlpha = 0.0f;
        float totalR = 0.0f;
        float totalG = 0.0f;
        float totalB = 0.0f;
        int activeCount = 0;
        
        for (const Voxel& voxel : voxels) {
            totalWeight += voxel.weight;
            totalAlpha += voxel.alpha;
            
            if (voxel.active) {
                totalR += voxel.color.x;
                totalG += voxel.color.y;
                totalB += voxel.color.z;
                activeCount++;
            }
        }
        
        int voxelCount = voxels.size();
        weight = totalWeight / voxelCount;
        alpha = totalAlpha / voxelCount;
        
        if (activeCount > 0) {
            color = Vec3ui8(
                static_cast<uint8_t>(totalR / activeCount),
                static_cast<uint8_t>(totalG / activeCount),
                static_cast<uint8_t>(totalB / activeCount)
            );
            active = true;
        } else {
            color = Vec3ui8(0, 0, 0);
            active = false;
        }
        dirty = false;
    }
    
    void updateAveragesRecursive() {
        TIME_FUNCTION;
        if (children.empty()) {
            updateAverages();
        } else {
            // Update all children first
            for (Chunk& child : children) {
                child.updateAveragesRecursive();
            }
            
            // Then update this chunk based on children
            if (children.empty()) return;
            
            float totalWeight = 0.0f;
            float totalAlpha = 0.0f;
            float totalR = 0.0f;
            float totalG = 0.0f;
            float totalB = 0.0f;
            int activeChildren = 0;
            
            for (const Chunk& child : children) {
                totalWeight += child.weight;
                totalAlpha += child.alpha;
                
                if (child.active) {
                    totalR += child.color.x;
                    totalG += child.color.y;
                    totalB += child.color.z;
                    activeChildren++;
                }
            }
            
            int childCount = children.size();
            weight = totalWeight / childCount;
            alpha = totalAlpha / childCount;
            
            if (activeChildren > 0) {
                color = Vec3ui8(
                    static_cast<uint8_t>(totalR / activeChildren),
                    static_cast<uint8_t>(totalG / activeChildren),
                    static_cast<uint8_t>(totalB / activeChildren)
                );
                active = true;
            } else {
                color = Vec3ui8(0, 0, 0);
                active = false;
            }
        }
        dirty = false;
    }

    void subdivide(int numChildrenPerAxis = 2) {
        TIME_FUNCTION;
        if (!children.empty()) return;  // Already subdivided
        
        Vec3i childSize = (maxCorner - minCorner) / numChildrenPerAxis;
        
        for (int z = 0; z < numChildrenPerAxis; z++) {
            for (int y = 0; y < numChildrenPerAxis; y++) {
                for (int x = 0; x < numChildrenPerAxis; x++) {
                    Vec3i childMin = minCorner + Vec3i(x, y, z) * childSize;
                    Vec3i childMax = childMin + childSize;
                    
                    Chunk child(childMin, childMax, depth + 1);
                    
                    // Copy voxel data from parent to child
                    for (int cz = childMin.z; cz < childMax.z; cz++) {
                        for (int cy = childMin.y; cy < childMax.y; cy++) {
                            for (int cx = childMin.x; cx < childMax.x; cx++) {
                                Vec3i pos(cx, cy, cz);
                                int parentIndex = getVoxelIndex(pos);
                                if (parentIndex >= 0 && parentIndex < static_cast<int>(voxels.size())) {
                                    int childLocalIndex = (cz - childMin.z) * childSize.x * childSize.y +
                                                         (cy - childMin.y) * childSize.x +
                                                         (cx - childMin.x);
                                    if (childLocalIndex >= 0 && childLocalIndex < static_cast<int>(child.voxels.size())) {
                                        child.voxels[childLocalIndex] = voxels[parentIndex];
                                    }
                                }
                            }
                        }
                    }
                    
                    children.push_back(child);
                }
            }
        }
    }

    void merge() {
        TIME_FUNCTION;
        if (children.empty()) return;
        
        // Calculate total size from children
        Vec3i totalSize(0, 0, 0);
        for (const Chunk& child : children) {
            totalSize = totalSize.max(child.maxCorner);
        }
        chunkSize = totalSize - minCorner;
        
        // Resize voxels vector
        voxels.resize(chunkSize.x * chunkSize.y * chunkSize.z);
        
        // Copy data from children
        for (const Chunk& child : children) {
            for (int z = child.minCorner.z; z < child.maxCorner.z; z++) {
                for (int y = child.minCorner.y; y < child.maxCorner.y; y++) {
                    for (int x = child.minCorner.x; x < child.maxCorner.x; x++) {
                        Vec3i pos(x, y, z);
                        Voxel voxel = child.get(pos);
                        
                        int parentIndex = getVoxelIndex(pos);
                        if (parentIndex >= 0 && parentIndex < static_cast<int>(voxels.size())) {
                            voxels[parentIndex] = voxel;
                        }
                    }
                }
            }
        }
        
        // Clear children
        children.clear();
        children.shrink_to_fit();
    }
};

class VoxelGrid {
private:
    Vec3i gridSize;
    std::vector<Chunk> chunks;
    std::vector<Voxel> voxels;
    bool useChunks = true;
    bool meshDirty = true;

    float radians(float rads) {
        return rads * (M_PI / 180);
    }

    // Helper to align a dimension to the next multiple of CHUNK_THRESHOLD
    static int alignToChunkSize(int size) {
        if (size <= 0) return CHUNK_THRESHOLD;
        int remainder = size % CHUNK_THRESHOLD;
        if (remainder == 0) return size;
        return size + (CHUNK_THRESHOLD - remainder);
    }

    void createChunksFromVoxels() {
        TIME_FUNCTION;
        chunks.clear();
        
        // Create chunks based on CHUNK_THRESHOLD
        int chunksX = std::max(1, (gridSize.x + CHUNK_THRESHOLD - 1) / CHUNK_THRESHOLD);
        int chunksY = std::max(1, (gridSize.y + CHUNK_THRESHOLD - 1) / CHUNK_THRESHOLD);
        int chunksZ = std::max(1, (gridSize.z + CHUNK_THRESHOLD - 1) / CHUNK_THRESHOLD);
        
        Vec3i chunkSize = Vec3i(
            (gridSize.x + chunksX - 1) / chunksX,
            (gridSize.y + chunksY - 1) / chunksY,
            (gridSize.z + chunksZ - 1) / chunksZ
        );
        
        for (int z = 0; z < chunksZ; z++) {
            for (int y = 0; y < chunksY; y++) {
                for (int x = 0; x < chunksX; x++) {
                    Vec3i minCorner = Vec3i(x * chunkSize.x, y * chunkSize.y, z * chunkSize.z);
                    Vec3i maxCorner = Vec3i(
                        std::min(minCorner.x + chunkSize.x, gridSize.x),
                        std::min(minCorner.y + chunkSize.y, gridSize.y),
                        std::min(minCorner.z + chunkSize.z, gridSize.z)
                    );
                    
                    Chunk chunk(minCorner, maxCorner);
                    
                    // Copy voxel data to chunk
                    for (int cz = minCorner.z; cz < maxCorner.z; cz++) {
                        for (int cy = minCorner.y; cy < maxCorner.y; cy++) {
                            for (int cx = minCorner.x; cx < maxCorner.x; cx++) {
                                Vec3i pos(cx, cy, cz);
                                int voxelIndex = cz * gridSize.x * gridSize.y + cy * gridSize.x + cx;
                                int chunkIndex = chunk.getVoxelIndex(pos);
                                
                                if (voxelIndex >= 0 && voxelIndex < static_cast<int>(voxels.size()) &&
                                    chunkIndex >= 0 && chunkIndex < static_cast<int>(chunk.voxels.size())) {
                                    chunk.voxels[chunkIndex] = voxels[voxelIndex];
                                }
                            }
                        }
                    }
                    
                    chunks.push_back(chunk);
                }
            }
        }
    }

public:
    double binSize = 1;
    VoxelGrid() : gridSize(0,0,0) {
        std::cout << "creating empty grid." << std::endl;
        resize(CHUNK_THRESHOLD, CHUNK_THRESHOLD, CHUNK_THRESHOLD);
    }

    VoxelGrid(int w, int h, int d) : gridSize(w,h,d) {
        voxels.resize(w * h * d);
        
    }
    
    bool serializeToFile(const std::string& filename);
    
    static std::unique_ptr<VoxelGrid> deserializeFromFile(const std::string& filename);
    
    Voxel& get(int x, int y, int z) {
        if (useChunks) {
            for (Chunk& chunk : chunks) {
                if (chunk.inChunk(Vec3i(x, y, z))) {
                    Voxel* voxelPtr = chunk.getPtr(Vec3i(x, y, z));
                    if (voxelPtr) {
                        return *voxelPtr;
                    }
                }
            }
        }
        return voxels[z * gridSize.x * gridSize.y + y * gridSize.x + x];
    }

    Voxel get(int x, int y, int z) const {
        if (useChunks) {
            for (const Chunk& chunk : chunks) {
                if (chunk.inChunk(Vec3i(x, y, z))) {
                    return chunk.get(Vec3i(x, y, z)); 
                }
            }
        }
        return voxels[z * gridSize.x * gridSize.y + y * gridSize.x + x];
    }

    Voxel& get(const Vec3i& xyz) {
        return get(xyz.x, xyz.y, xyz.z);
    }

    Voxel get(const Vec3i& xyz) const {
        return get(xyz.x, xyz.y, xyz.z);
    }

    void resize(int newW, int newH, int newD) {
        TIME_FUNCTION;
        
        int alignedW = alignToChunkSize(newW);
        int alignedH = alignToChunkSize(newH);
        int alignedD = alignToChunkSize(newD);

        if (alignedW == gridSize.x && alignedH == gridSize.y && alignedD == gridSize.z) {
            return;
        }

        std::vector<Voxel> newVoxels(alignedW * alignedH * alignedD);
        int copyW = std::min(static_cast<int>(gridSize.x), alignedW);
        int copyH = std::min(static_cast<int>(gridSize.y), alignedH);
        int copyD = std::min(static_cast<int>(gridSize.z), alignedD);
        
        for (int z = 0; z < copyD; ++z) {
            for (int y = 0; y < copyH; ++y) {
                int oldRowStart = z * gridSize.x * gridSize.y + y * gridSize.x;
                int newRowStart = z * alignedW * alignedH + y * alignedW;
                std::copy(
                    voxels.begin() + oldRowStart, 
                    voxels.begin() + oldRowStart + copyW, 
                    newVoxels.begin() + newRowStart
                );
            }
        }
        
        voxels = std::move(newVoxels);
        gridSize = Vec3i(alignedW, alignedH, alignedD);
        
        // Rebuild chunks structure (which will now be perfectly aligned cubes)
        useChunks = true;
        createChunksFromVoxels();
    }

    void resize(Vec3i newsize) {
        resize(newsize.x, newsize.y, newsize.z);
    }

    void set(int x, int y, int z, bool active, Vec3ui8 color) {
        set(Vec3i(x,y,z), active, color);
    }

    void set(Vec3i pos, bool active, Vec3ui8 color) {
        if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0) {
            // Check against current aligned gridSize
            if (!(pos.x < gridSize.x)) {
                resize(pos.x, gridSize.y, gridSize.z);
            }
            else if (!(pos.y < gridSize.y)) {
                resize(gridSize.x, pos.y, gridSize.z);
            }
            else if (!(pos.z < gridSize.z)) {
                resize(gridSize.x, gridSize.y, pos.z);
            }

            Voxel& v = get(pos);
            v.active = active;
            v.color = color;
            
            // // Also update in chunks if using chunks
            // if (useChunks) {
            //     for (Chunk& chunk : chunks) {
            //         if (chunk.inChunk(pos)) {
            //             Voxel newVoxel;
            //             newVoxel.active = active;
            //             newVoxel.color = color;
            //             chunk.set(pos, newVoxel);
            //             break;
            //         }
            //     }
            // }
        }
    }

    void set(Vec3i pos, Vec4ui8 rgbaval) {
        set(pos, static_cast<float>(rgbaval.a / 255), rgbaval.toVec3());
    }

    template<typename T>
    bool inGrid(Vec3<T> voxl) const {
        return (voxl >= 0 && voxl.x < gridSize.x && voxl.y < gridSize.y && voxl.z < gridSize.z);
    }

    void chunkTraverse(const Vec3d& origin, const Vec3d& end, std::vector<Vec3i>& visitedVoxel) const {
        
    }

    void voxelTraverse(const Vec3d& origin, const Vec3d& end, std::vector<Vec3i>& visitedVoxel, int maxDist = 10000000) const {
        Vec3i cv = (origin / binSize).floorToI();
        Vec3i lv = (end / binSize).floorToI();
        Vec3d ray = end - origin;
        
        Vec3f step = Vec3f(ray.x >= 0 ? 1 : -1, ray.y >= 0 ? 1 : -1, ray.z >= 0 ? 1 : -1);
        Vec3d nextVox = cv.toDouble() + step * binSize;
        Vec3d tMax = Vec3d(ray.x != 0 ? (nextVox.x - origin.x) / ray.x : INF,
                           ray.y != 0 ? (nextVox.y - origin.y) / ray.y : INF,
                           ray.z != 0 ? (nextVox.z - origin.z) / ray.z : INF);
        Vec3d tDelta = Vec3d(ray.x != 0 ? binSize / ray.x * step.x : INF,
                             ray.y != 0 ? binSize / ray.y * step.y : INF,
                             ray.z != 0 ? binSize / ray.z * step.z : INF);
        float dist = 0;
        Vec3i diff(0,0,0);
        bool negRay = false;
        if (cv.x != lv.x && ray.x < 0) {
            diff.x = diff.x--;
            negRay = true;
        }
        if (cv.y != lv.y && ray.y < 0) {
            diff.y = diff.y--;
            negRay = true;
        }
        if (cv.z != lv.z && ray.z < 0) {
            diff.z = diff.z--;
            negRay = true;
        }

        if (negRay) {
            cv += diff;
            visitedVoxel.push_back(cv);
        }

        while (lv != cv && inGrid(cv) && visitedVoxel.size() < 10 && dist < maxDist) {
            Voxel cvv = get(cv);

            if (cvv.active) {
                visitedVoxel.push_back(cv);
            }
            
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) {
                    dist += tDelta.x;
                    cv.x += step.x;
                    tMax.x += tDelta.x;
                } else {
                    dist += tDelta.y;
                    cv.z += step.z;
                    tMax.z += tDelta.z;
                }
            } else {
                if (tMax.y < tMax.z) {
                    dist += tDelta.y;
                    cv.y += step.y;
                    tMax.y += tDelta.y;
                } else {
                    dist += tDelta.z;
                    cv.z += step.z;
                    tMax.z += tDelta.z;
                }
            }
        }
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
        Vec3f upCor = right.cross(forward).normalized();
        float aspect = resolution.aspect();
        float fovRad = cam.fovRad();
        float viewH = 2 * tan(fovRad / 2);
        float viewW = viewH * aspect;
        float maxDist = std::sqrt(gridSize.lengthSquared()) * binSize;
        frame outFrame(resolution.x, resolution.y, frame::colormap::RGB);
        std::vector<uint8_t> colorBuffer(resolution.x * resolution.y * 3);
        //#pragma omp parallel for
        for (int y = 0; y < resolution.x; y++) {
            float v = (static_cast<float>(y) / static_cast<float>(resolution.x - 1)) - 0.5f;    
            for (int x = 0; x < resolution.y; x++) {
                std::vector<Vec3i> hitVoxels;
                float u = (static_cast<float>(x) / static_cast<float>(resolution.y - 1)) - 0.5f;
                Vec3f rayDirWorld = (forward + right * (u * viewW) + upCor * (v * viewH)).normalized();
                Vec3f rayEnd = cam.posfor.origin + rayDirWorld * maxDist;
                Vec3d rayStartGrid = cam.posfor.origin.toDouble() / binSize;
                Vec3d rayEndGrid = rayEnd.toDouble() / binSize;
                if (useChunks) {
                    chunkTraverse(rayStartGrid, rayEndGrid, hitVoxels);
                } else voxelTraverse(rayStartGrid, rayEndGrid, hitVoxels);
                
                //std::cout << "traversed";
                Vec3ui8 hitColor(10, 10, 255);
                for (const Vec3i& voxelPos : hitVoxels) {
                    if (inGrid(voxelPos)) {
                        const Voxel voxel = get(voxelPos);
                        if (voxel.active) {
                            hitColor = voxel.color;
                            
                            break;
                        }
                    }
                }
                //std::cout << "hit done" << std::endl;
                hitVoxels.clear();
                hitVoxels.shrink_to_fit();
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
                        colorBuffer[idx + 0] = hitColor.x;
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
        std::cout << "Active percentage: " << activePercentage << "%" << std::endl;
        std::cout << "Memory usage (approx): " << (voxels.size() * sizeof(Voxel)) / 1024 << " KB" << std::endl; //needs to be updated to include chunks
        std::cout << "Using chunks: " << (useChunks ? "Yes" : "No") << std::endl;
        if (useChunks) {
            std::cout << "Number of chunks: " << chunks.size() << std::endl;
        }
        std::cout << "============================" << std::endl;
    }

private:
    // Helper function to check if a voxel is on the surface
    bool isSurfaceVoxel(int x, int y, int z) const {
        if (!inGrid(Vec3i(x, y, z))) return false;
        if (!get(x, y, z).active) return false;
        
        // Check all 6 neighbors
        static const std::array<Vec3i, 6> neighbors = {{
            Vec3i(1, 0, 0), Vec3i(-1, 0, 0),
            Vec3i(0, 1, 0), Vec3i(0, -1, 0),
            Vec3i(0, 0, 1), Vec3i(0, 0, -1)
        }};
        
        for (const auto& n : neighbors) {
            Vec3i neighborPos(x + n.x, y + n.y, z + n.z);
            if (!inGrid(neighborPos) || !get(neighborPos).active) {
                return true; // At least one empty neighbor means this is a surface voxel
            }
        }
        
        return false;
    }
    
    // Get normal for a surface voxel
    Vec3f calculateVoxelNormal(int x, int y, int z) const {
        Vec3f normal(0, 0, 0);
        
        // Simple gradient-based normal calculation
        if (inGrid(Vec3i(x+1, y, z)) && !get(x+1, y, z).active) normal.x += 1;
        if (inGrid(Vec3i(x-1, y, z)) && !get(x-1, y, z).active) normal.x -= 1;
        if (inGrid(Vec3i(x, y+1, z)) && !get(x, y+1, z).active) normal.y += 1;
        if (inGrid(Vec3i(x, y-1, z)) && !get(x, y-1, z).active) normal.y -= 1;
        if (inGrid(Vec3i(x, y, z+1)) && !get(x, y, z+1).active) normal.z += 1;
        if (inGrid(Vec3i(x, y, z-1)) && !get(x, y, z-1).active) normal.z -= 1;
        
        if (normal.lengthSquared() > 0) {
            return normal.normalized();
        }
        return Vec3f(0, 1, 0); // Default up normal
    }

public:
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
        size_t cbsize = gridSize.x * gridSize.y * colors;
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