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
    Vec4 red = hexToVec4("ff0000");      // Top-left corner
    Vec4 green = hexToVec4("00ff00");    // Center
    Vec4 blue = hexToVec4("0000ff");     // Bottom-right corner
    Vec4 white = hexToVec4("ffffff");    // Top-right corner
    Vec4 black = hexToVec4("000000");    // Bottom-left corner
    
    // Create gradient points
    for (int y = 0; y < POINTS_PER_DIM; ++y) {
        for (int x = 0; x < POINTS_PER_DIM; ++x) {
            // Normalize coordinates to [0, 1]
            float nx = static_cast<float>(x) / (POINTS_PER_DIM - 1);
            float ny = static_cast<float>(y) / (POINTS_PER_DIM - 1);
            
            // Create position in [-1, 1] range
            Vec2 pos(nx * 2.0f - 1.0f, ny * 2.0f - 1.0f);
            
            // Calculate interpolated color based on position
            Vec4 color;
            
            if (nx + ny <= 1.0f) {
                // Lower triangle: interpolate between red, green, and black
                if (nx <= 0.5f && ny <= 0.5f) {
                    // Bottom-left quadrant: red to black to green
                    float t1 = nx * 2.0f; // Horizontal interpolation
                    float t2 = ny * 2.0f; // Vertical interpolation
                    
                    if (t1 + t2 <= 1.0f) {
                        // Interpolate between red and black
                        color = red * (1.0f - t1 - t2) + black * (t1 + t2);
                    } else {
                        // Interpolate between black and green
                        color = black * (2.0f - t1 - t2) + green * (t1 + t2 - 1.0f);
                    }
                } else {
                    // Use bilinear interpolation for other areas
                    Vec4 topLeft = red;
                    Vec4 topRight = white;
                    Vec4 bottomLeft = black;
                    Vec4 bottomRight = green;
                    
                    Vec4 top = topLeft * (1.0f - nx) + topRight * nx;
                    Vec4 bottom = bottomLeft * (1.0f - nx) + bottomRight * nx;
                    color = bottom * (1.0f - ny) + top * ny;
                }
            } else {
                // Upper triangle: interpolate between green, blue, and white
                if (nx >= 0.5f && ny >= 0.5f) {
                    // Top-right quadrant: green to white to blue
                    float t1 = (nx - 0.5f) * 2.0f; // Horizontal interpolation
                    float t2 = (ny - 0.5f) * 2.0f; // Vertical interpolation
                    
                    if (t1 + t2 <= 1.0f) {
                        // Interpolate between green and white
                        color = green * (1.0f - t1 - t2) + white * (t1 + t2);
                    } else {
                        // Interpolate between white and blue
                        color = white * (2.0f - t1 - t2) + blue * (t1 + t2 - 1.0f);
                    }
                } else {
                    // Use bilinear interpolation for other areas
                    Vec4 topLeft = red;
                    Vec4 topRight = white;
                    Vec4 bottomLeft = black;
                    Vec4 bottomRight = blue;
                    
                    Vec4 top = topLeft * (1.0f - nx) + topRight * nx;
                    Vec4 bottom = bottomLeft * (1.0f - nx) + bottomRight * nx;
                    color = bottom * (1.0f - ny) + top * ny;
                }
            }
            
            grid.addPoint(pos, color);
        }
    }
    
    // Render to RGB image
    std::vector<uint8_t> imageData = grid.renderToRGB(WIDTH, HEIGHT);
    
    // Save as BMP
    if (BMPWriter::saveBMP("output/gradient.bmp", imageData, WIDTH, HEIGHT)) {
        std::cout << "Gradient image saved as 'gradient.bmp'" << std::endl;
        std::cout << "Colors: " << std::endl;
        std::cout << "  Top-left: ff0000 (red)" << std::endl;
        std::cout << "  Center: 00ff00 (green)" << std::endl;
        std::cout << "  Bottom-right: 0000ff (blue)" << std::endl;
        std::cout << "  Gradient between ffffff and 000000 throughout" << std::endl;
    } else {
        std::cerr << "Failed to save gradient image" << std::endl;
        return 1;
    }
    
    return 0;
}