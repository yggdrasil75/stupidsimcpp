// test_voxel_render.cpp
#include "../util/grid/grid33.hpp"
#include "../util/output/bmpwriter.hpp"
#include <random>
#include <iostream>

int main() {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> pos_dist(-1.0, 1.0);
    std::uniform_int_distribution<> color_dist(0, 255);
    
    // Create a voxel grid with 0.1 unit resolution
    VoxelGrid<Vec3ui8, 2, 3> voxelGrid(0.1);
    
    std::cout << "Placing 10 random colored voxels..." << std::endl;
    
    // Place 10 random colored voxels
    for (int i = 0; i < 10; ++i) {
        // Generate random position
        double x = pos_dist(gen);
        double y = pos_dist(gen);
        double z = pos_dist(gen);
        
        // Generate random color
        uint8_t r = static_cast<uint8_t>(color_dist(gen));
        uint8_t g = static_cast<uint8_t>(color_dist(gen));
        uint8_t b = static_cast<uint8_t>(color_dist(gen));
        
        Vec3d worldPos(x, y, z);
        Vec3ui8 color(r, g, b);
        
        // Set voxel color
        bool wasOn = voxelGrid.setVoxelColor(worldPos, color);
        
        std::cout << "Voxel " << i + 1 << ": "
                  << "Pos(" << x << ", " << y << ", " << z << ") "
                  << "Color(RGB:" << static_cast<int>(r) << ","
                  << static_cast<int>(g) << "," << static_cast<int>(b) << ") "
                  << (wasOn ? "(overwritten)" : "(new)") << std::endl;
    }
    
    std::cout << "\nMemory usage: " << voxelGrid.getMemoryUsage() << " bytes" << std::endl;
    
    // Render to BMP
    std::cout << "\nRendering orthographic projection to BMP..." << std::endl;
    
    int width = 512;
    int height = 512;
    
    // Create a buffer for the rendered image
    std::vector<uint8_t> imageBuffer;
    
    // Render with orthographic projection (view along Z axis)
    voxelGrid.renderProjectedToRGBBuffer(imageBuffer, width, height, 
                                        Vec3d(0, 0, 1),  // View direction (looking along Z)
                                        Vec3d(0, 1, 0)); // Up direction
    
    std::cout << "Image buffer size: " << imageBuffer.size() << " bytes" << std::endl;
    std::cout << "Expected size: " << (width * height * 3) << " bytes" << std::endl;
    
    // Save to BMP using BMPWriter
    std::string filename = "voxel_render.bmp";
    
    // Create a frame object from the buffer
    frame renderFrame(width, height, frame::colormap::RGB);
    renderFrame.setData(imageBuffer);
    
    // Save as BMP
    if (BMPWriter::saveBMP(filename, renderFrame)) {
        std::cout << "Successfully saved to: " << filename << std::endl;
        
        // Also save using the direct vector interface as backup
        if (BMPWriter::saveBMP("voxel_render_direct.bmp", imageBuffer, width, height)) {
            std::cout << "Also saved direct version: voxel_render_direct.bmp" << std::endl;
        }
    } else {
        std::cout << "Failed to save BMP!" << std::endl;
        
        // Try alternative save method
        std::cout << "Trying alternative save method..." << std::endl;
        
        // Convert to Vec3ui8 format
        std::vector<Vec3ui8> pixelsVec3;
        pixelsVec3.reserve(width * height);
        
        for (size_t i = 0; i < imageBuffer.size(); i += 3) {
            pixelsVec3.push_back(Vec3ui8(
                imageBuffer[i],
                imageBuffer[i + 1],
                imageBuffer[i + 2]
            ));
        }
        
        if (BMPWriter::saveBMP("voxel_render_vec3.bmp", pixelsVec3, width, height)) {
            std::cout << "Saved Vec3ui8 version: voxel_render_vec3.bmp" << std::endl;
        } else {
            std::cout << "All save methods failed!" << std::endl;
        }
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
    
    return 0;
}