#ifndef SIMPLE_HTTPSERVER_HPP
#define SIMPLE_HTTPSERVER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <sstream>
#include <fstream>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <unistd.h>
    #include <arpa/inet.h>
#endif

#include "grid2.hpp"
#include "bmpwriter.hpp"
#include "jxlwriter.hpp"

class SimpleHTTPServer {
private:
    int serverSocket;
    int port;
    bool running;
    std::string webRoot;
    
    // Function to convert hex color string to Vec4
    Vec4 hexToVec4(const std::string& hex) {
        if (hex.length() != 6) {
            return Vec4(0, 0, 0, 1); // Default to black if invalid
        }
        
        int r, g, b;
        sscanf(hex.c_str(), "%02x%02x%02x", &r, &g, &b);
        
        return Vec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
    }
    
    // Generate gradient image
    bool generateGradientImage(const std::string& filename, int width = 512, int height = 512) {
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
    
    // Read file content
    std::string readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            return "";
        }
        
        std::ostringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }
    
    // Get content type based on file extension
    std::string getContentType(const std::string& filename) {
        if (filename.find(".html") != std::string::npos) return "text/html";
        if (filename.find(".css") != std::string::npos) return "text/css";
        if (filename.find(".js") != std::string::npos) return "application/javascript";
        if (filename.find(".jxl") != std::string::npos) return "image/jxl";
        if (filename.find(".png") != std::string::npos) return "image/png";
        if (filename.find(".jpg") != std::string::npos || filename.find(".jpeg") != std::string::npos) return "image/jpeg";
        if (filename.find(".json") != std::string::npos) return "application/json";
        return "text/plain";
    }
    
    // Send HTTP response
    void sendResponse(int clientSocket, const std::string& content, const std::string& contentType = "text/html", int statusCode = 200) {
        std::string statusText = "OK";
        if (statusCode == 404) statusText = "Not Found";
        if (statusCode == 500) statusText = "Internal Server Error";
        
        std::ostringstream response;
        response << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n";
        response << "Content-Type: " << contentType << "\r\n";
        response << "Content-Length: " << content.length() << "\r\n";
        response << "Connection: close\r\n";
        response << "\r\n";
        response << content;
        
        std::string responseStr = response.str();
        send(clientSocket, responseStr.c_str(), responseStr.length(), 0);
    }
    
    // Serve static file
    void serveStaticFile(int clientSocket, const std::string& filepath) {
        std::string fullPath = webRoot + "/" + filepath;
        std::string content = readFile(fullPath);
        
        if (!content.empty()) {
            sendResponse(clientSocket, content, getContentType(filepath));
        } else {
            sendResponse(clientSocket, "404 Not Found: " + filepath, "text/plain", 404);
        }
    }

public:
    SimpleHTTPServer(int port = 8080, const std::string& webRoot = "web") 
        : port(port), serverSocket(-1), running(false), webRoot(webRoot) {}
    
    ~SimpleHTTPServer() {
        stop();
    }
    
    bool start() {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed" << std::endl;
            return false;
        }
#endif
        
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket < 0) {
            std::cerr << "Socket creation failed" << std::endl;
            return false;
        }
        
        int opt = 1;
#ifdef _WIN32
        if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)) < 0) {
#else
        if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
#endif
            std::cerr << "Setsockopt failed" << std::endl;
            return false;
        }
        
        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(port);
        
        if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "Bind failed on port " << port << std::endl;
            return false;
        }
        
        if (listen(serverSocket, 10) < 0) {
            std::cerr << "Listen failed" << std::endl;
            return false;
        }
        
        running = true;
        std::cout << "Server started on port " << port << std::endl;
        std::cout << "Web root: " << webRoot << std::endl;
        return true;
    }
    
    void stop() {
        running = false;
        if (serverSocket >= 0) {
#ifdef _WIN32
            closesocket(serverSocket);
            WSACleanup();
#else
            close(serverSocket);
#endif
            serverSocket = -1;
        }
    }
    
    void handleRequests() {
        while (running) {
            sockaddr_in clientAddr;
#ifdef _WIN32
            int clientAddrLen = sizeof(clientAddr);
#else
            socklen_t clientAddrLen = sizeof(clientAddr);
#endif
            int clientSocket = accept(serverSocket, (sockaddr*)&clientAddr, &clientAddrLen);
            
            if (clientSocket < 0) {
                if (running) {
                    std::cerr << "Accept failed" << std::endl;
                }
                continue;
            }
            
            char buffer[4096] = {0};
            int bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0);
            
            if (bytesReceived > 0) {
                std::string request(buffer);
                std::cout << "Received request: " << request.substr(0, request.find('\n')) << std::endl;
                
                // Handle different routes
                if (request.find("GET / ") != std::string::npos || request.find("GET /index.html") != std::string::npos) {
                    serveStaticFile(clientSocket, "index.html");
                } else if (request.find("GET /style.css") != std::string::npos) {
                    serveStaticFile(clientSocket, "style.css");
                } else if (request.find("GET /script.js") != std::string::npos) {
                    serveStaticFile(clientSocket, "script.js");
                } else if (request.find("GET /gradient.jxl") != std::string::npos) {
                    // Generate and serve the gradient image
                    if (generateGradientImage("output/gradient.jxl")) {
                        std::string imageContent = readFile("output/gradient.jxl");
                        if (!imageContent.empty()) {
                            sendResponse(clientSocket, imageContent, "image/jxl");
                        } else {
                            sendResponse(clientSocket, "Error generating image", "text/plain", 500);
                        }
                    } else {
                        sendResponse(clientSocket, "Error generating image", "text/plain", 500);
                    }
                } else if (request.find("GET /generate") != std::string::npos) {
                    // API endpoint to generate new gradient
                    std::string filename = "output/dynamic_gradient.jxl";
                    if (generateGradientImage(filename)) {
                        sendResponse(clientSocket, "{\"status\":\"success\",\"file\":\"" + filename + "\"}", "application/json");
                    } else {
                        sendResponse(clientSocket, "{\"status\":\"error\"}", "application/json", 500);
                    }
                } else {
                    sendResponse(clientSocket, "404 Not Found", "text/plain", 404);
                }
            }
            
#ifdef _WIN32
            closesocket(clientSocket);
#else
            close(clientSocket);
#endif
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};

#endif