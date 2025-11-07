#include <iostream>
#include <string>
#include "util/simple_httpserver.hpp"
#include "util/grid2.hpp"
#include "util/bmpwriter.hpp"
#include "util/jxlwriter.hpp"
#include "util/timing_decorator.hpp"
#include "simtools/sim2.hpp"

// Function to convert hex color string to Vec4
Vec4 hexToVec4(const std::string& hex) {
    TIME_FUNCTION;
    
    if (hex.length() != 6) {
        return Vec4(0, 0, 0, 1); // Default to black if invalid
    }
    
    int r, g, b;
    sscanf(hex.c_str(), "%02x%02x%02x", &r, &g, &b);
    
    return Vec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
}

// Generate gradient image
bool generateGradientImage(const std::string& filename, int width = 512, int height = 512) {
    TIME_FUNCTION;
    
    const int POINTS_PER_DIM = 256;
    
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
            
            // Convert to [0,1] range for interpolation
            float u = (nx + 1.0f) / 2.0f;
            float v = (ny + 1.0f) / 2.0f;
            
            // Bilinear interpolation between corners
            Vec4 top = white * (1.0f - u) + red * u;
            Vec4 bottom = blue * (1.0f - u) + black * u;
            Vec4 cornerColor = top * (1.0f - v) + bottom * v;
            
            // Calculate distance from center (0,0)
            float distFromCenter = std::sqrt(nx * nx + ny * ny) / std::sqrt(2.0f);
            
            Vec4 color = green * (1.0f - distFromCenter) + cornerColor * distFromCenter;
            
            grid.addPoint(pos, color);
        }
    }
    
    // Render to RGB image
    std::vector<uint8_t> imageData = grid.renderToRGB(width, height);
    
    // Save as JXL
    return JXLWriter::saveJXL(filename, imageData, width, height);
}

// Generate terrain simulation image
bool generateTerrainImage(const std::string& filename, int width = 512, int height = 512) {
    TIME_FUNCTION;
    
    static Sim2 sim(width, height);
    
    // Randomize seed for variety
    sim.randomizeSeed();
    
    // Render to RGB image
    std::vector<uint8_t> imageData = sim.renderToRGB(width, height);
    
    // Save as JXL
    return JXLWriter::saveJXL(filename, imageData, width, height);
}

// Add this function to get timing stats as JSON
std::string getTimingStatsJSON() {
    
    auto stats = FunctionTimer::getStats();
    std::stringstream json;
    
    json << "[";
    bool first = true;
    
    for (const auto& [func_name, data] : stats) {
        if (!first) json << ",";
        first = false;
        
        auto percentiles = FunctionTimer::calculatePercentiles(data.timings);
        
        json << "{"
             << "\"function\":\"" << func_name << "\","
             << "\"call_count\":" << data.call_count << ","
             << "\"total_time\":" << std::fixed << std::setprecision(6) << data.total_time << ","
             << "\"avg_time\":" << std::fixed << std::setprecision(6) << data.avg_time() << ","
             << "\"min_time\":" << std::fixed << std::setprecision(6) << percentiles.min << ","
             << "\"max_time\":" << std::fixed << std::setprecision(6) << percentiles.max << ","
             << "\"median_time\":" << std::fixed << std::setprecision(6) << percentiles.median << ","
             << "\"p90_time\":" << std::fixed << std::setprecision(6) << percentiles.p90 << ","
             << "\"p95_time\":" << std::fixed << std::setprecision(6) << percentiles.p95 << ","
             << "\"p99_time\":" << std::fixed << std::setprecision(6) << percentiles.p99
             << "}";
    }
    
    json << "]";
    return json.str();
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    int port = 8080;
    std::string webRoot = "web";
    std::string mode = "gradient"; // Default mode
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" || arg == "-p") {
            if (i + 1 < argc) {
                port = std::stoi(argv[++i]);
            }
        } else if (arg == "--webroot" || arg == "-w") {
            if (i + 1 < argc) {
                webRoot = argv[++i];
            }
        } else if (arg == "-2d") {
            mode = "terrain";
        } else if (arg == "-all") {
            mode = "all";
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -p, --port PORT    Set server port (default: 8080)" << std::endl;
            std::cout << "  -w, --webroot DIR  Set web root directory (default: web)" << std::endl;
            std::cout << "  -2d                Display 2D terrain simulation" << std::endl;
            std::cout << "  -all               Allow switching between gradient and terrain" << std::endl;
            std::cout << "  -h, --help         Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Generate initial image based on mode
    std::cout << "Generating " << mode << " image..." << std::endl;
    bool success = false;
    
    if (mode == "terrain") {
        success = generateTerrainImage(webRoot + "/output/display.jxl");
    } else {
        success = generateGradientImage(webRoot + "/output/display.jxl");
    }
    
    if (success) {
        std::cout << mode << " image generated successfully" << std::endl;
    } else {
        std::cerr << "Failed to generate " << mode << " image" << std::endl;
        return 1;
    }
    
    SimpleHTTPServer server(port, webRoot);
    
    // Add timing stats endpoint
    server.addRoute("/api/timing-stats", [](const std::string& method, const std::string& body) {
        if (method == "GET") {
            return std::make_pair(200, getTimingStatsJSON());
        }
        return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
    });
    
    // Add clear stats endpoint
    server.addRoute("/api/clear-stats", [](const std::string& method, const std::string& body) {
        if (method == "POST") {
            FunctionTimer::clearStats();
            return std::make_pair(200, std::basic_string("{\"status\":\"success\"}"));
        }
        return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
    });
    
    // Add mode switching endpoint for -all mode
    if (mode == "all") {
        server.addRoute("/api/switch-mode", [webRoot](const std::string& method, const std::string& body) {
            if (method == "POST") {
                static bool currentModeGradient = true;
                
                bool success = false;
                if (currentModeGradient) {
                    success = generateTerrainImage(webRoot + "/output/display.jxl");
                } else {
                    success = generateGradientImage(webRoot + "/output/display.jxl");
                }
                
                if (success) {
                    currentModeGradient = !currentModeGradient;
                    std::string newMode = currentModeGradient ? "gradient" : "terrain";
                    return std::make_pair(200, std::basic_string("{\"status\":\"success\", \"mode\":\"" + newMode + "\"}"));
                } else {
                    return std::make_pair(500, std::basic_string("{\"error\":\"Failed to generate image\"}"));
                }
            }
            return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
        });
        
        server.addRoute("/api/current-mode", [](const std::string& method, const std::string& body) {
            if (method == "GET") {
                static bool currentModeGradient = true;
                std::string mode = currentModeGradient ? "gradient" : "terrain";
                return std::make_pair(200, std::basic_string("{\"mode\":\"" + mode + "\"}"));
            }
            return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
        });
    }
    
    // Add refresh endpoint for terrain mode (fast regeneration)
    if (mode == "terrain" || mode == "all") {
        server.addRoute("/api/refresh-terrain", [webRoot](const std::string& method, const std::string& body) {
            if (method == "POST") {
                bool success = generateTerrainImage(webRoot + "/output/display.jxl");
                if (success) {
                    return std::make_pair(200, std::basic_string("{\"status\":\"success\"}"));
                } else {
                    return std::make_pair(500, std::basic_string("{\"error\":\"Failed to generate terrain\"}"));
                }
            }
            return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
        });
    }
    
    if (!server.start()) {
        std::cerr << "Failed to start server on port " << port << std::endl;
        return 1;
    }
    
    std::cout << "Server running on http://localhost:" << port << std::endl;
    std::cout << "Web root: " << webRoot << std::endl;
    std::cout << "Mode: " << mode << std::endl;
    std::cout << "Timing stats available at /api/timing-stats" << std::endl;
    
    if (mode == "all") {
        std::cout << "Mode switching available at /api/switch-mode" << std::endl;
    }
    
    if (mode == "terrain") {
        std::cout << "Fast terrain refresh available at /api/refresh-terrain" << std::endl;
    }
    
    std::cout << "Press Ctrl+C to stop the server" << std::endl;
    
    server.handleRequests();
    
    return 0;
}