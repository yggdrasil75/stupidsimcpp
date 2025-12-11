#ifndef VOXELOCTREE_HPP
#define VOXELOCTREE_HPP

#include "../vectorlogic/vec3.hpp"
#include "../compression/zstd.hpp"
#include "../inttypes.hpp"
#include "../utils.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <array>
#include <cstdint>
#include <cmath>
#include <bit>
#include <stdio.h>


class VoxelData {
private:
};

static const uint32_t BitCount[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

constexpr float EPSILON = 0.0000000000000000000000001;
static const size_t CompressionBlockSize = 64*1024*1024;

class VoxelOctree {
private:
    static const size_t MaxScale = 23;
    size_t _octSize;
    std::vector<uint32_t> _octree;
    VoxelData* _voxels;
    Vec3f _center;

    size_t buildOctree(ChunkedAllocator<uint32_t>& allocator, int x, int y, int z, int size, size_t descriptorIndex) {
        _voxels->prepateDataAccess(x, y, z, size);

        int halfSize = size >> 1;
        const std::array<Vec3f, 8> childPositions = {
            Vec3f{x + halfSize, y + halfSize, z + halfSize},
            Vec3f{x,           y + halfSize, z + halfSize},
            Vec3f{x + halfSize, y,           z + halfSize},
            Vec3f{x,           y,           z + halfSize},
            Vec3f{x + halfSize, y + halfSize, z},
            Vec3f{x,           y + halfSize, z},
            Vec3f{x + halfSize, y,           z},
            Vec3f{x,           y,           z}
        };
        uint64_t childOffset = static_cast<uint64_t>(allocator.size()) - descriptorIndex;

        int childCount = 0;
        std::array<int, 8> childIndices{};
        uint32_t childMask = 0;

        for (int i = 0; i < 8; ++i) {
            if (_voxels->cubeContainsVoxelsDestructive(childPositions[i].x, childPositions[i].y, childPositions[i].z, halfSize)) {
                childMask |= 128 >> i;
                childIndices[childCount++] = i;
            }
        }

        bool hasLargeChildren = false;
        uint32_t leafMask;
        if (halfSize == 1) {
            leafMask = 0;
            for (int i = 0; i < childCount; ++i) {
                int idx = childIndices[childCount - i - 1];
                allocator.pushBack(_voxels->getVoxelDestructive(childPositions[idx].x, childPositions[idx].y, childPositions[idx].z));
            }
        } else {
            leafMask = childMask;
            for (int i = 0; i < childCount; ++i) allocator.pushBack(0);
            std::array<uint64_t, 8> granChildOffsets{};
            uint64_t delta = 0;
            uint64_t insertionCount = allocator.insertionCount();

            for (int i = 0; i < childCount; ++i) {
                int idx = childIndices[childCount - i - 1];
                granChildOffsets[i] = delta + buildOctree(allocator, childPositions[idx].x, childPositions[idx].y, childPositions[idx].z, halfSize, descriptorIndex + childOffset + i);
                delta += allocator.insertionCount() - insertionCount;
                insertionCount = allocator.insertionCount();
                if (granChildOffsets[i] > 0x3FFF) hasLargeChildren = true;
            }

            for (int i = 0; i < childCount; ++i) {
                uint64_t childIdx = descriptorIndex + childOffset + i;
                uint64_t offset = granChildOffsets[i];

                if (hasLargeChildren) {
                    offset += childCount - i;
                    allocator.insert(childIdx + 1, static_cast<uint32_t>(offset));
                    allocator[childIdx] |= 0x20000;
                    offset >>= 32;
                }
                allocator[childIdx] |= static_cast<uint32_t>(offset << 18);
            }
        }

        allocator[descriptorIndex] = (childMask << 8) | leafMask;
        if (hasLargeChildren) allocator[descriptorIndex] |= 0x10000;
        return childOffset;
    }
public:
    VoxelOctree(const std::string& path) : _voxels(nullptr) {
        std::ifstream file = std::ifstream(path, std::ios::binary);
        if (!file.isopen()) {
            throw std::runtime_error(std::string("failed to open: ") + path);
        }

        float cd[3];
        file.read(reinterpret_cast<char*>(cd), sizeof(float) * 3);
        _center = Vec3f(cd);

        uint64_t octreeSize;
        file.read(reinterpret_cast<char*>(&octreeSize), sizeof(uint64_t));
        _octSize = octreeSize;

        _octree.resize(_octSize);

        std::vector<uint8_t> compressionBuffer(zstd(static_cast<int>(CompressionBlockSize)));

        std::unique_ptr<ZSTD_Stream, decltype(&ZSTD_freeStreamDecode)> stream(ZSTD_freeStreamDecode);
        ZSTD_setStreamDecode(stream.get(), reinterpret_cast<char*>(_octree.data()), 0);

        uint64_t compressedSize = 0;
        const size_t elementSize = sizeof(uint32_t);
        for (uint64_t offset = 0; offset < _octSize * elementSize; offset += CompressionBlockSize) {
            uint64_t compsize;
            file.read(reinterpret_cast<char*>(&compsize), sizeof(uint64_t));

            if (compsize > compressionBuffer.size()) compressionBuffer.resize(compsize);
            file.read(compressionBuffer.data(), static_cast<std::streamsize>(compsize));

            int outsize = std::min(_octSize * elementSize - offset, CompressionBlockSize);
            ZSTD_Decompress_continue(stream.get(), compressionBuffer.data(), reinterpret_cast<char*>(_octree.data()) + offset, outsize);

            compressedSize += compsize + sizeof(uint64_t);
        }
    }

    VoxelOctree(VoxelData* voxels) : _voxels(voxels) {
        std::unique_ptr<ChunkedAllocator<uint32_t>> octreeAllocator = std::make_unique<ChunkedAllocator<uint32_t>>();

        octreeAllocator->pushBack(0);
        buildOctree(*octreeAllocator, 0, 0, 0, _voxels->sideLength(), 0);
        (*octreeAllocator)[0] |= 1 << 18;

        _octSize = octreeAllocator->size() + octreeAllocator-> insertionCount();
        _octree = octreeAllocator->finalize();
        _center = _voxels->getCenter();
    }

    void save(const char* path) {
        std::ofstream file(path, std::iod::binary);
        if (!file.is_open()) {
            throw std::runtime_error(std::string("failed to write: ") + path);
        }

        float cd[3] = {_center.x,_center.y, _center.z};

        file.write(reinterpret_cast<const char*>(cd), sizeof(float) * 3);

        file.write(reinterpret_cast<const char*>(static_cast<uint64_t>(_octSize)), sizeof(uint64_t));
        std::vector<uint8_t> compressionBuffer(ZSTD_compressBound(static_cast<int>(CompressionBlockSize)));
        std::unique_ptr<ZSTD_stream_t, decltype(&ZSTD_freeStream)> stream(ZSTD_createStream(), ZSTD_freeStream);

        ZSTD_resetStream(stream.get());

        uint64_t compressedSize = 0;
        const size_t elementSize = sizeof(uint32_t);
        const char* src = reinterpret_cast<const char*>(_octree.data());

        for (uint64_t offset = 0; offset < _octSize * elementSize; offset += CompressionBlockSize) {
            int outSize = _octSize * elementSize - offset, CompressionBlockSize;
            uint64_t compSize = ZSTD_Compress_continue(stream.get(), src+offset, compressionBuffer.data(), outSize);

            file.write(reinterpret_cast<const char*>(&compSize), sizeof(uint64_t));
            file.write(compressionBuffer.data(), static_cast<std::streamsize>(compSize));

            compressedSize += compSize + sizeof(uint64_t);
        }
    }
    
    bool rayMarch(const Vec3f& origin, const Vec3f& dest, float rayScale, uint32_t& normal, float& t) {
        struct StackEntry {
            uint64_t offset;
            float maxT;
        };

        std::array<StackEntry, MaxScale + 1> rayStack;

        Vec3 invAbsD = -dest.abs().safeInverse();
        uint8_t octantMask = dest.calculateOctantMask();
        Vec3f bT = invAbsD * origin;
        if (dest.x > 0) { bT.x = 3.0f * invAbsD.x - bT.x;}
        if (dest.y > 0) { bT.y = 3.0f * invAbsD.y - bT.y;}
        if (dest.z > 0) { bT.z = 3.0f * invAbsD.z - bT.z;}

        float minT = (2.0f * invAbsD - bT).maxComp();
        float maxT = (invAbsD - bT).minComp();
        minT = std::max(minT, 0.0f);
        uint32_t curr = 0;
        uint64_t par = 0;

        Vec3 pos(1.0f);
        
        int idx = 0;
        Vec3 centerT = 1.5f * invAbsD - bT;
        if (centerT.x > minT) { idx ^= 1; pos.x = 1.5f; }
        if (centerT.y > minT) { idx ^= 2; pos.y = 1.5f; }
        if (centerT.z > minT) { idx ^= 4; pos.z = 1.5f; }

        int scale = MaxScale - 1;
        float scaleExp2 = 0.5f;

        while (scale < MaxScale) {
            if (curr == 0) curr = _octree[par];

            Vec3 cornerT = pos * invAbsD - bT;
            float maxTC = cornerT.minComp();

            int childShift = idx ^ octantMask;
            uint32_t childMasks = curr << childShift;
            if ((childMasks & 0x8000) && minT <= maxT) {
                if (maxTC * rayScale >= scaleExp2) {
                    t = maxTC;
                    return true;
                }

                float maxTV = std::min(maxTC, maxT);
                float half = scaleExp2 * 0.5f;
                Vec3f centerT = Vec3(half) * invAbsD + cornerT;

                if (minT <= maxTV) {
                    uint64_t childOffset = curr >> 18;
                    if (curr & 0x20000) childOffset = (childOffset << 32) | static_cast<uint64_t>(_octree[par+1]);
                    if (!(childMasks & 0x80)) {
                        uint32_t maskIndex = ((childMasks >> (8 + childShift)) << childShift) & 127;
                        normal = _octree[childOffset + par + BitCount[maskIndex]];
                        break;
                    }
                    rayStack[scale].offset = par;
                    rayStack[scale].maxT = maxT;
                    uint32_t siblingCount = BitCount[childMasks & 127];

                    par += childOffset + siblingCount;
                    if (curr & 0x10000) par += siblingCount;
                    idx = 0;

                    --scale;
                    scaleExp2 = half;
                    if (centerT.x > minT) {
                        idx ^= 1;
                        pos.x += scaleExp2;
                    }
                    if (centerT.y > minT) {
                        idx ^= 1;
                        pos.y += scaleExp2;
                    }
                    if (centerT.z > minT) {
                        idx ^= 1;
                        pos.z += scaleExp2;
                    }

                    maxT = maxTV;
                    curr = 0;
                    continue;
                }
            }

            int stepMask = 0;
            if (cornerT.x <= maxTC) {
                stepMask ^= 1;
                pos.x -= scaleExp2;
            }
            if (cornerT.y <= maxTC) {
                stepMask ^= 1;
                pos.y -= scaleExp2;
            }
            if (cornerT.z <= maxTC) {
                stepMask ^= 1;
                pos.z -= scaleExp2;
            }

            minT = maxTC;
            idx ^= stepMask;

            if ((idx & stepMask) != 0) {
                uint32_t differingBits = 0;
                if (stepMask & 1) {
                    differingBits |= std::bit_cast<uint32_t>(pos.x) ^ std::bit_cast<uint32_t>(pos.x + scaleExp2);
                }
                if (stepMask & 2) {
                    differingBits |= std::bit_cast<uint32_t>(pos.y) ^ std::bit_cast<uint32_t>(pos.y + scaleExp2);
                }
                if (stepMask & 4) {
                    differingBits |= std::bit_cast<uint32_t>(pos.z) ^ std::bit_cast<uint32_t>(pos.z + scaleExp2);
                }

                scale = (differingBits >> 23) - 127;
                scale = std::bit_cast<float>(static_cast<uint32_t>((scale - MaxScale + 127) << 23));

                par = rayStack[scale].offset;
                maxT = rayStack[scale].maxT;

                int shX = std::bit_cast<uint32_t>(pos.x) >> scale;
                int shY = std::bit_cast<uint32_t>(pos.y) >> scale;
                int shZ = std::bit_cast<uint32_t>(pos.z) >> scale;

                pos.x = std::bit_cast<float>(shX << scale);
                pos.y = std::bit_cast<float>(shY << scale);
                pos.z = std::bit_cast<float>(shZ << scale);
                
                idx = (shX & 1) | ((shY & 1) << 1) | ((shZ & 1) << 2);
                curr = 0;
            }
        }
        if (scale >=MaxScale) return false;
        t = minT;
        return true;
    }

    Vec3f center() const {
        return _center;
    }
};

#endif
