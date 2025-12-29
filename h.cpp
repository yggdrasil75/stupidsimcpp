#include "util/grid/gridtest.hpp"

int main() {
    VoxelRenderer renderer;
    
    // Simple test
    // std::cout << "Voxel Grid: " << renderer.getGrid().getWidth() << "x" 
    //           << renderer.getGrid().getHeight() << "x" 
    //           << renderer.getGrid().getDepth() << std::endl;
    
    // Test ray from center
    Vec3f rayOrigin(5, 5, -5);
    Vec3f rayDir(0, 0, 1);
    Vec3f hitPos, hitNormal, hitColor;
    
    if (renderer.getGrid().rayCast(rayOrigin, rayDir, 20.0f, hitPos, hitNormal, hitColor)) {
        std::cout << "Test ray hit at: " << hitPos.x << ", " 
                  << hitPos.y << ", " << hitPos.z << std::endl;
        std::cout << "Hit color: " << hitColor.r << ", " 
                  << hitColor.g << ", " << hitColor.b << std::endl;
    } else {
        std::cout << "Test ray missed" << std::endl;
    }
    
    // Render a simple 100x100 "image"
    renderer.render(100, 100);
    
    return 0;
}