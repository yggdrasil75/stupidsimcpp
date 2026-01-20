#ifndef GRID3_Serialization
#define GRID3_Serialization

#include <fstream>
#include <cstring>
#include "grid3.hpp"

constexpr char magic[4] = {'Y', 'G', 'G', '3'};

inline bool VoxelGrid::serializeToFile(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "failed to open file (serializeToFile): " << filename << std::endl;
        return false;
    }

    file.write(magic, 4);
    int dims[3] = {gridSize.x, gridSize.y, gridSize.z};
    file.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    size_t voxelCount = voxels.size();
    file.write(reinterpret_cast<const char*>(&voxelCount), sizeof(voxelCount));
    for (const Voxel& voxel : voxels) {
        auto write_member = [&file](const auto& member) {
            file.write(reinterpret_cast<const char*>(&member), sizeof(member));
        };
        
        std::apply([&write_member](const auto&... members) {
            (write_member(members), ...);
        }, voxel.members());
    }

    file.close();
    return !file.fail();
}

std::unique_ptr<VoxelGrid> VoxelGrid::deserializeFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "failed to open file (deserializeFromFile): " << filename << std::endl;
        return nullptr;
    }
    
    // Read and verify magic number
    char filemagic[4];
    file.read(filemagic, 4);
    if (std::strncmp(filemagic, magic, 4) != 0) {
        std::cerr << "Error: Invalid file format or corrupted file (expected " 
                  << magic[0] << magic[1] << magic[2] << magic[3]
                  << ", got " << filemagic[0] << filemagic[1] << filemagic[2] << filemagic[3]
                  << ")" << std::endl;
        return nullptr;
    }
    
    // Create output grid
    auto outgrid = std::make_unique<VoxelGrid>();
    
    // Read grid dimensions
    int dims[3];
    file.read(reinterpret_cast<char*>(dims), sizeof(dims));
    
    // Resize grid
    outgrid->gridSize = Vec3i(dims[0], dims[1], dims[2]);
    outgrid->voxels.resize(dims[0] * dims[1] * dims[2]);
    
    // Read voxel count
    size_t voxelCount;
    file.read(reinterpret_cast<char*>(&voxelCount), sizeof(voxelCount));
    
    // Verify voxel count matches grid dimensions
    size_t expectedCount = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
    if (voxelCount != expectedCount) {
        std::cerr << "Error: Voxel count mismatch. Expected " << expectedCount 
                 << ", found " << voxelCount << std::endl;
        return nullptr;
    }
    
    // Read all voxels
    for (size_t i = 0; i < voxelCount; ++i) {
        auto members = outgrid->voxels[i].members();
        
        std::apply([&file](auto&... member) {
            ((file.read(reinterpret_cast<char*>(&member), sizeof(member))), ...);
        }, members);
    }
    
    file.close();
    
    if (file.fail() && !file.eof()) {
        std::cerr << "Error: Failed to read from file: " << filename << std::endl;
        return nullptr;
    }
    
    std::cout << "Successfully loaded grid: " << dims[0] << " x " 
              << dims[1] << " x " << dims[2] << std::endl;
    
    return outgrid;
}

#endif