#include <iostream>
#include <string>
#include <vector>
#include "util/grid2.hpp"
#include "util/bmpwriter.hpp"

// Function to convert hex color string to Vec4
Vec4 hexToVec4(const std::string& hex) {
    if (hex.length() != 6) {
        return Vec4(0, 0, 0, 1); // Default to black if invalid
    }
    
    int r, g, b;
    sscanf(hex.c_str(), "%02x%02x%02x", &r, &g, &b);
    
    return Vec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
}

int main(int argc, char* argv[]) {
    // Check for gradient flag
    bool createGradient = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gradient" || arg == "-g") {
            createGradient = true;
            break;
        }
    }
    
    if (!createGradient) {
        std::cout << "Usage: " << argv[0] << " --gradient (-g)" << std::endl;
        std::cout << "Creates a gradient image with red, green, and blue corners" << std::endl;
        return 1;
    }
    
    // Create a grid with points arranged in a gradient pattern
    const int WIDTH = 512;
    const int HEIGHT = 512;
    const int POINTS_PER_DIM = 256; // Resolution of the gradient
    
    Grid2 grid;
    
    // Define our target colors at specific positions
    Vec4 white = hexToVec4("ffffff");    // Top-left corner (1,1)
    Vec4 red = hexToVec4("ff0000");      // Top-right corner (1,-1)
    Vec4 green = hexToVec4("00ff00");    // Center (0,0)
    Vec4 blue = hexToVec4("0000ff");     // Bottom-left corner (-1,-1)
    Vec4 black = hexToVec4("000000");    // Bottom-right corner (-1,1)
    
    // Create gradient points
    for (int y = 0; y < POINTS_PER_DIM; ++y) {
        for (int x = 0; x < POINTS_PER_DIM; ++x) {
            // Normalize coordinates to [-1, 1]
            float nx = (static_cast<float>(x) / (POINTS_PER_DIM - 1)) * 2.0f - 1.0f;
            float ny = (static_cast<float>(y) / (POINTS_PER_DIM - 1)) * 2.0f - 1.0f;
            
            // Create position
            Vec2 pos(nx, ny);
            
            // Calculate weights for each corner based on distance
            // We'll use bilinear interpolation
            
            // Convert to [0,1] range for interpolation
            float u = (nx + 1.0f) / 2.0f;  // maps -1..1 to 0..1
            float v = (ny + 1.0f) / 2.0f;  // maps -1..1 to 0..1
            
            // For a more natural gradient, we'll interpolate between the four corners
            // and blend with the center color based on distance from center
            
            // Bilinear interpolation between corners
            Vec4 top = white * (1.0f - u) + red * u;
            Vec4 bottom = blue * (1.0f - u) + black * u;
            Vec4 cornerColor = top * (1.0f - v) + bottom * v;
            
            // Calculate distance from center (0,0)
            float distFromCenter = std::sqrt(nx * nx + ny * ny) / std::sqrt(2.0f); // normalize to [0,1]
            
            Vec4 color = green * (1.0f - distFromCenter) + cornerColor * distFromCenter;
            
            grid.addPoint(pos, color);
        }
    }
    
    // Render to RGB image
    std::vector<uint8_t> imageData = grid.renderToRGB(WIDTH, HEIGHT);
    
    // Save as BMP
    if (BMPWriter::saveBMP("output/gradient.bmp", imageData, WIDTH, HEIGHT)) {
        std::cout << "Gradient image saved as 'gradient.bmp'" << std::endl;
        std::cout << "Color positions: " << std::endl;
        std::cout << "  Top-left: ffffff (white)" << std::endl;
        std::cout << "  Top-right: ff0000 (red)" << std::endl;
        std::cout << "  Center: 00ff00 (green)" << std::endl;
        std::cout << "  Bottom-left: 0000ff (blue)" << std::endl;
        std::cout << "  Bottom-right: 000000 (black)" << std::endl;
    } else {
        std::cerr << "Failed to save gradient image" << std::endl;
        return 1;
    }
    
    return 0;
}