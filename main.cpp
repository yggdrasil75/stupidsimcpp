#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <queue>
#include <condition_variable>
#include "util/simple_httpserver.hpp"
#include "util/grid2.hpp"
#include "util/bmpwriter.hpp"
#include "util/jxlwriter.hpp"
#include "util/timing_decorator.hpp"
#include "simtools/sim2.hpp"

// Global variables for live streaming
std::queue<std::vector<uint8_t>> frameQueue;
std::mutex queueMutex;
std::condition_variable frameCondition;
std::atomic<bool> streaming{false};
std::atomic<int> activeClients{0};
std::atomic<uint32_t> frameCounter{0};

// Current simulation parameters
struct SimulationParams {
    std::string mode = "gradient";
    uint32_t seed = 42;
    float scale = 4.0f;
    int octaves = 4;
    float persistence = 0.5f;
    float lacunarity = 2.0f;
    float elevation = 1.0f;
    float waterLevel = 0.3f;
};

SimulationParams currentParams;
std::mutex paramsMutex;

// Function to convert hex color string to Vec4
Vec4 hexToVec4(const std::string& hex) {
    TIME_FUNCTION;
    
    if (hex.length() != 6) {
        return Vec4(0, 0, 0, 1);
    }
    
    int r, g, b;
    sscanf(hex.c_str(), "%02x%02x%02x", &r, &g, &b);
    
    return Vec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
}

// Generate gradient frame data
std::vector<uint8_t> generateGradientFrame(int width = 512, int height = 512) {
    TIME_FUNCTION;
    
    const int POINTS_PER_DIM = 256;
    Grid2 grid;
    
    // Define colors
    Vec4 white = hexToVec4("ffffff");
    Vec4 red = hexToVec4("ff0000");
    Vec4 green = hexToVec4("00ff00");
    Vec4 blue = hexToVec4("0000ff");
    Vec4 black = hexToVec4("000000");
    
    // Animate based on frame counter for live effect
    float time = frameCounter * 0.01f;
    
    for (int y = 0; y < POINTS_PER_DIM; ++y) {
        for (int x = 0; x < POINTS_PER_DIM; ++x) {
            float nx = (static_cast<float>(x) / (POINTS_PER_DIM - 1)) * 2.0f - 1.0f;
            float ny = (static_cast<float>(y) / (POINTS_PER_DIM - 1)) * 2.0f - 1.0f;
            
            Vec2 pos(nx, ny);
            float u = (nx + 1.0f) / 2.0f;
            float v = (ny + 1.0f) / 2.0f;
            
            Vec4 top = white * (1.0f - u) + red * u;
            Vec4 bottom = blue * (1.0f - u) + black * u;
            Vec4 cornerColor = top * (1.0f - v) + bottom * v;
            
            // Calculate distance from center (0,0)
            float distFromCenter = std::sqrt(nx * nx + ny * ny) / std::sqrt(2.0f);
            
            Vec4 color = green * (1.0f - distFromCenter) + cornerColor * distFromCenter;
            
            grid.addPoint(pos, color);
        }
    }
    
    return grid.renderToRGB(width, height);
}

// Generate terrain simulation frame data
std::vector<uint8_t> generateTerrainFrame(int width = 512, int height = 512) {
    TIME_FUNCTION;
    
    SimulationParams params;
    {
        std::lock_guard<std::mutex> lock(paramsMutex);
        params = currentParams;
    }
    
    // Use frame counter to animate terrain
    uint32_t animatedSeed = params.seed + frameCounter;
    
    Sim2 sim(width, height, animatedSeed, params.scale, params.octaves, 
             params.persistence, params.lacunarity, params.waterLevel, params.elevation);
    sim.generateTerrain();
    
    return sim.renderToRGB(width, height);
}

// Streaming thread function
void streamingThread() {
    auto lastFrameTime = std::chrono::steady_clock::now();
    
    while (streaming) {
        auto startTime = std::chrono::steady_clock::now();
        
        // Only generate frames if there are active clients
        if (activeClients > 0) {
            std::vector<uint8_t> frame;
            
            {
                std::lock_guard<std::mutex> lock(paramsMutex);
                if (currentParams.mode == "terrain") {
                    frame = generateTerrainFrame();
                } else {
                    frame = generateGradientFrame();
                }
            }
            
            // Add frame to queue
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                // Limit queue size to prevent memory issues
                while (frameQueue.size() > 10) {
                    frameQueue.pop();
                }
                frameQueue.push(std::move(frame));
                frameCounter++;
            }
            
            // Notify waiting clients
            frameCondition.notify_all();
        }
        
        // Control frame rate (30 FPS max)
        auto endTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        auto targetFrameTime = std::chrono::milliseconds(33); // ~30 FPS
        
        if (elapsed < targetFrameTime) {
            std::this_thread::sleep_for(targetFrameTime - elapsed);
        }
        
        lastFrameTime = endTime;
    }
}

// Get the latest frame (blocks until frame is available)
std::vector<uint8_t> getLatestFrame(int timeoutMs = 1000) {
    std::unique_lock<std::mutex> lock(queueMutex);
    
    if (frameCondition.wait_for(lock, std::chrono::milliseconds(timeoutMs), 
                               []{ return !frameQueue.empty(); })) {
        auto frame = std::move(frameQueue.front());
        // Keep only the latest frame for new clients
        std::queue<std::vector<uint8_t>> empty;
        std::swap(frameQueue, empty);
        frameQueue.push(frame); // Keep one frame for the next client
        return frame;
    }
    
    return {}; // Return empty frame on timeout
}

// Convert RGB data to JPEG (simplified - in real implementation use libjpeg)
std::vector<uint8_t> rgbToJpeg(const std::vector<uint8_t>& rgbData, int width, int height) {
    TIME_FUNCTION;
    
    // TODO: return mjpeg
    
    return rgbData; // Return uncompressed RGB for now
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
    std::string mode = "gradient";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" || arg == "-p") {
            if (i + 1 < argc) port = std::stoi(argv[++i]);
        } else if (arg == "--webroot" || arg == "-w") {
            if (i + 1 < argc) webRoot = argv[++i];
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
    
    // Set initial mode
    {
        std::lock_guard<std::mutex> lock(paramsMutex);
        currentParams.mode = mode;
    }
    
    // Start streaming thread
    streaming = true;
    std::thread streamThread(streamingThread);
    
    SimpleHTTPServer server(port, webRoot);
    
    // MJPEG stream endpoint
    server.addRoute("/stream.mjpg", [](const std::string& method, const std::string& body) {
        if (method == "GET") {
            activeClients++;
            
            // Set up MJPEG stream headers
            std::string response = "HTTP/1.1 200 OK\r\n";
            response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n";
            response += "Cache-Control: no-cache\r\n";
            response += "Connection: close\r\n";
            response += "\r\n";
            
            return std::make_pair(200, response);
        }
        return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
    });
    
    // Single frame endpoint
    server.addRoute("/api/frame", [](const std::string& method, const std::string& body) {
        if (method == "GET") {
            activeClients++;
            
            auto frame = getLatestFrame();
            if (!frame.empty()) {
                // Convert to JPEG (in real implementation)
                auto jpegData = rgbToJpeg(frame, 512, 512);
                
                // For now, send as raw RGB with custom content type
                std::string response(jpegData.begin(), jpegData.end());
                activeClients--;
                return std::make_pair(200, response);
            }
            
            activeClients--;
            return std::make_pair(503, std::basic_string("No frame available"));
        }
        return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
    });
    
    // Frame info endpoint
    server.addRoute("/api/frame-info", [](const std::string& method, const std::string& body) {
        if (method == "GET") {
            std::stringstream json;
            json << "{"
                 << "\"frame_count\":" << frameCounter << ","
                 << "\"active_clients\":" << activeClients << ","
                 << "\"width\":512,"
                 << "\"height\":512,"
                 << "\"channels\":3"
                 << "}";
            return std::make_pair(200, json.str());
        }
        return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
    });

    // Parameter setting endpoint
    server.addRoute("/api/set-parameters", [](const std::string& method, const std::string& body) {
        if (method == "POST") {
            try {
                SimulationParams newParams;
                {
                    std::lock_guard<std::mutex> lock(paramsMutex);
                    newParams = currentParams;
                }
                
                // Parse parameters from JSON body
                if (body.find("\"scale\"") != std::string::npos) {
                    size_t pos = body.find("\"scale\":") + 8;
                    newParams.scale = std::stof(body.substr(pos));
                }
                if (body.find("\"octaves\"") != std::string::npos) {
                    size_t pos = body.find("\"octaves\":") + 10;
                    newParams.octaves = std::stoi(body.substr(pos));
                }
                if (body.find("\"persistence\"") != std::string::npos) {
                    size_t pos = body.find("\"persistence\":") + 14;
                    newParams.persistence = std::stof(body.substr(pos));
                }
                if (body.find("\"lacunarity\"") != std::string::npos) {
                    size_t pos = body.find("\"lacunarity\":") + 13;
                    newParams.lacunarity = std::stof(body.substr(pos));
                }
                if (body.find("\"elevation\"") != std::string::npos) {
                    size_t pos = body.find("\"elevation\":") + 12;
                    newParams.elevation = std::stof(body.substr(pos));
                }
                if (body.find("\"waterLevel\"") != std::string::npos) {
                    size_t pos = body.find("\"waterLevel\":") + 13;
                    newParams.waterLevel = std::stof(body.substr(pos));
                }
                if (body.find("\"seed\"") != std::string::npos) {
                    size_t pos = body.find("\"seed\":") + 7;
                    newParams.seed = std::stoul(body.substr(pos));
                }
                
                {
                    std::lock_guard<std::mutex> lock(paramsMutex);
                    currentParams = newParams;
                }
                
                return std::make_pair(200, std::basic_string("{\"status\":\"success\"}"));
            } catch (const std::exception& e) {
                return std::make_pair(400, std::basic_string("{\"error\":\"Invalid parameters\"}"));
            }
        }
        return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
    });

    // Mode switching endpoint
    server.addRoute("/api/switch-mode", [](const std::string& method, const std::string& body) {
        if (method == "POST") {
            std::lock_guard<std::mutex> lock(paramsMutex);
            if (currentParams.mode == "gradient") {
                currentParams.mode = "terrain";
            } else {
                currentParams.mode = "gradient";
            }
            
            std::string response = "{\"status\":\"success\", \"mode\":\"" + currentParams.mode + "\"}";
            return std::make_pair(200, response);
        }
        return std::make_pair(405, std::basic_string("Method Not Allowed"));
    });
    
    // Current mode endpoint
    server.addRoute("/api/current-mode", [](const std::string& method, const std::string& body) {
        if (method == "GET") {
            std::lock_guard<std::mutex> lock(paramsMutex);
            std::string response = "{\"mode\":\"" + currentParams.mode + "\"}";
            return std::make_pair(200, response);
        }
        return std::make_pair(405, std::basic_string("Method Not Allowed"));
    });

    // Add timing stats endpoint
    server.addRoute("/api/timing-stats", [](const std::string& method, const std::string& body) {
        if (method == "GET") {
            return std::make_pair(200, getTimingStatsJSON());
        }
        return std::make_pair(405, std::basic_string("Method Not Allowed"));
    });
    
    // Add clear stats endpoint
    server.addRoute("/api/clear-stats", [](const std::string& method, const std::string& body) {
        if (method == "POST") {
            FunctionTimer::clearStats();
            return std::make_pair(200, std::basic_string("{\"status\":\"success\"}"));
        }
        return std::make_pair(405, std::basic_string("{\"error\":\"Method Not Allowed\"}"));
    });
    
    if (!server.start()) {
        std::cerr << "Failed to start server on port " << port << std::endl;
        streaming = false;
        if (streamThread.joinable()) streamThread.join();
        return 1;
    }
    
    std::cout << "Server running on http://localhost:" << port << std::endl;
    std::cout << "Web root: " << webRoot << std::endl;
    std::cout << "Mode: " << mode << " (Live Streaming)" << std::endl;
    std::cout << "Live stream available at /stream.mjpg" << std::endl;
    std::cout << "Single frames available at /api/frame" << std::endl;
    std::cout << "Timing stats available at /api/timing-stats" << std::endl;
    
    server.handleRequests();
    
    // Cleanup
    streaming = false;
    if (streamThread.joinable()) {
        streamThread.join();
    }
    
    return 0;
}