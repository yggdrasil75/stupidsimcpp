#ifndef GRID3_Serialization
#define GRID3_Serialization

#include <fstream>
#include <cstring>
#include "grid3.hpp"

constexpr char magic[4] = {'Y', 'G', 'G', '3'};

inline bool serializeToFile(const VoxelGrid& grid, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "failed to open file (serializeToFile): " << filename << std::endl;
        return false;
    }

    file.write(magic, 4);
    //int dims[3] = {grid.gridSize.x, grid.gridSize.y, grid.gridSize.z};

}

#endif