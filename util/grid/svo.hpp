#ifndef VOXELOCTREE_HPP
#define VOXELOCTREE_HPP

#include "../vectorlogic/vec3.hpp"
#include "../compression/zstd.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <array>
#include <cstdint>
#include <cmath>
#include <bit>

class VoxelData;

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
        const std::array<int, 8> posX = { x + halfSize, x, x+halfSize, x, x+halfSize, x, x+halfSize, x};
        const std::array<int, 8> posY = { y + halfSize, y+halfSize, y, y, y+halfSize, y+halfSize, y, y};
        const std::array<int, 8> posZ = { z + halfSize, z+halfSize, z+halfSize, z+halfSize, z, z, z, z};
        
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

        float cd[3] = _center;

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
    
    bool rayMarch(const Vec3f& origin, const Vec3f& dest, float rayScale, uint32_t& normal, float& t);

    Vec3f center() const {
        return _center;
    }
};

#endif