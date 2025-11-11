#include <iostream>
#include <vector>
#include "../util/grid/grid2.hpp"
#include "../util/output/bmpwriter.hpp"

int main() {
    // Create a Grid2 instance
    Grid2 grid;
    
    // Grid dimensions
    const int width = 100;
    const int height = 100;
    
    std::cout << "Creating grayscale gradient..." << std::endl;
    
    // Add objects to create a grayscale gradient
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Calculate gradient value (0.0 at top-left to 1.0 at bottom-right)
            float gradient = (x + y) / float(width + height - 2);
            
            // Create position
            Vec2 position(static_cast<float>(x), static_cast<float>(y));
            
            // Create grayscale color (r=g=b=gradient, a=1.0)
            Vec4 color(gradient, gradient, gradient, 1.0f);
            
            // Add to grid with size 1.0 (single pixel)
            grid.addObject(position, color, 1.0f);
        }
    }
    
    std::cout << "Added " << width * height << " objects to grid" << std::endl;
    
    // Get the entire grid as RGB data
    int outputWidth, outputHeight;
    std::vector<int> rgbData;
    grid.getGridAsRGB(outputWidth, outputHeight, rgbData);
    
    std::cout << "Output dimensions: " << outputWidth << " x " << outputHeight << std::endl;
    std::cout << "RGB data size: " << rgbData.size() << " elements" << std::endl;
    
    // Convert RGB data to format suitable for BMPWriter
    std::vector<Vec3> pixels;
    pixels.reserve(outputWidth * outputHeight);
    
    for (size_t i = 0; i < rgbData.size(); i += 3) {
        float r = rgbData[i] / 255.0f;
        float g = rgbData[i + 1] / 255.0f;
        float b = rgbData[i + 2] / 255.0f;
        pixels.emplace_back(r, g, b);
    }
    
    // Save as BMP
    std::string filename = "output/grayscale_gradient.bmp";
    bool success = BMPWriter::saveBMP(filename, pixels, outputWidth, outputHeight);
    
    if (success) {
        std::cout << "Successfully saved grayscale gradient to: " << filename << std::endl;
        
        // Print some gradient values for verification
        std::cout << "\nGradient values at key positions:" << std::endl;
        std::cout << "Top-left (0,0): " << grid.getColor(grid.getIndicesAt(0, 0)[0]).r << std::endl;
        std::cout << "Center (" << width/2 << "," << height/2 << "): " 
                  << grid.getColor(grid.getIndicesAt(width/2, height/2)[0]).r << std::endl;
        std::cout << "Bottom-right (" << width-1 << "," << height-1 << "): " 
                  << grid.getColor(grid.getIndicesAt(width-1, height-1)[0]).r << std::endl;
    } else {
        std::cerr << "Failed to save BMP file!" << std::endl;
        return 1;
    }
    
    return 0;
}