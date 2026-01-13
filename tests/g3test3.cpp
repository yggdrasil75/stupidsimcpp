#include "../util/grid/grid33.hpp"
//#include "../util/grid/treexy/treexy_serialization.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/timing_decorator.cpp"
#include <random>
#include <iostream>

struct configuration {
    float threshold = 0.1;
    int gridWidth = 128;
    int gridHeight = 128;
    int gridDepth = 128;
    PNoise2 noise = PNoise2(42);
};

void setup(configuration& config, VoxelGrid<Vec3ui8, 2, 3>& grid) {
    TIME_FUNCTION;
    uint8_t thresh = config.threshold * 255;
    for (int z = 0; z < config.gridDepth; ++z) {
        if (z % 64 == 0) {
            std::cout << "Processing layer " << z << " of " << config.gridDepth << std::endl;
        }

        for (int y = 0; y < config.gridHeight; ++y) {
            for (int x = 0; x < config.gridWidth; ++x) {
                uint8_t r = std::clamp(config.noise.permute(Vec3f(static_cast<float>(x) / config.gridWidth / 64, static_cast<float>(y) / config.gridHeight / 64, static_cast<float>(z) / config.gridDepth / 64)), 0.f, 1.f) * 255;
                uint8_t g = std::clamp(config.noise.permute(Vec3f(static_cast<float>(x) / config.gridWidth / 32, static_cast<float>(y) / config.gridHeight / 32, static_cast<float>(z) / config.gridDepth / 32)), 0.f, 1.f) * 255;
                uint8_t b = std::clamp(config.noise.permute(Vec3f(static_cast<float>(x) / config.gridWidth / 16, static_cast<float>(y) / config.gridHeight / 16, static_cast<float>(z) / config.gridDepth / 16)), 0.f, 1.f) * 255;
                uint8_t a = std::clamp(config.noise.permute(Vec3f(static_cast<float>(x) / config.gridWidth / 8 , static_cast<float>(y) / config.gridHeight / 8 , static_cast<float>(z) / config.gridDepth / 8 )), 0.f, 1.f) * 255;
                if (a > thresh) {
                    bool wasOn = grid.setVoxelColor(Vec3d(x,y,z), Vec3ui8(r,g,b));
                }
            }
        }
    }
}

int main() {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> pos_dist(-1.0, 1.0);
    std::uniform_int_distribution<> color_dist(0, 255);
    
    // Create a voxel grid with 0.1 unit resolution
    VoxelGrid<Vec3ui8, 2, 3> voxelGrid(0.1);
        
    configuration config;
    setup(config, voxelGrid);
    
    std::cout << "\nMemory usage: " << voxelGrid.getMemoryUsage() << " bytes" << std::endl;
    
    // Render to BMP
    std::cout << "\nRendering orthographic projection to BMP..." << std::endl;
    
    int width = 512;
    int height = 512;
    
    // Create a buffer for the rendered image
    std::vector<uint8_t> imageBuffer;
    
    // Render with orthographic projection (view along Z axis)
    Vec3d camPos = Vec3d(config.gridDepth, config.gridHeight, config.gridWidth * 2);
    Vec3d lookAt = Vec3d(config.gridDepth / 2, config.gridHeight / 2, config.gridWidth / 2);
    Vec3d viewDir = (lookAt-camPos).normalized();
    voxelGrid.renderToRGB(imageBuffer, width, height, camPos, viewDir, Vec3d(0, 1, 0), 80.f);
    
    std::cout << "Image buffer size: " << imageBuffer.size() << " bytes" << std::endl;
    std::cout << "Expected size: " << (width * height * 3) << " bytes" << std::endl;
    
    // Save to BMP using BMPWriter
    std::string filename = "output/voxel_render.bmp";
    
    // Create a frame object from the buffer
    frame renderFrame(width, height, frame::colormap::RGB);
    renderFrame.setData(imageBuffer);
    
    // Save as BMP
    if (BMPWriter::saveBMP(filename, renderFrame)) {
        std::cout << "Successfully saved to: " << filename << std::endl;
        
    } else {
        std::cout << "Failed to save BMP!" << std::endl;
    }
    
    // Test accessor functionality
    std::cout << "\nTesting accessor functionality..." << std::endl;
    auto accessor = voxelGrid.createAccessor();
    
    // Try to retrieve one of the voxels
    Vec3d testPos(0.0, 0.0, 0.0);  // Center point
    Vec3i testCoord = voxelGrid.posToCoord(testPos);
    
    Vec3ui8* retrievedColor = voxelGrid.getVoxelColor(testPos);
    if (retrievedColor) {
        std::cout << "Found voxel at center: Color(RGB:" 
                  << static_cast<int>(retrievedColor->x) << ","
                  << static_cast<int>(retrievedColor->y) << ","
                  << static_cast<int>(retrievedColor->z) << ")" << std::endl;
    } else {
        std::cout << "No voxel found at center position" << std::endl;
    }
    
    // Iterate through all voxels using forEachCell
    std::cout << "\nIterating through all voxels:" << std::endl;
    int voxelCount = 0;
    voxelGrid.forEachCell([&](const Vec3ui8& color, const Vec3i& coord) {
        Vec3d pos = voxelGrid.Vec3iToPos(coord);
        std::cout << "Voxel " << ++voxelCount << ": "
                  << "Coord(" << coord.x << ", " << coord.y << ", " << coord.z << ") "
                  << "WorldPos(" << pos.x << ", " << pos.y << ", " << pos.z << ") "
                  << "Color(RGB:" << static_cast<int>(color.x) << ","
                  << static_cast<int>(color.y) << "," << static_cast<int>(color.z) << ")"
                  << std::endl;
    });
    
    std::cout << "\nTotal voxels in grid: " << voxelCount << std::endl;
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    
    return 0;
}