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
#include <functional>
#include <unordered_map>
#include "timing_decorator.hpp"

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

class SimpleHTTPServer {
private:
    int serverSocket;
    int port;
    bool running;
    std::string webRoot;
    
    // Route handler type
    using RouteHandler = std::function<std::pair<int, std::string>(const std::string&, const std::string&)>;
    std::unordered_map<std::string, RouteHandler> routes;
    
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
        if (filename.find(".ico") != std::string::npos) return "image/x-icon";
        return "text/plain";
    }
    
    // Send HTTP response
    void sendResponse(int clientSocket, const std::string& content, const std::string& contentType = "text/html", int statusCode = 200) {
        TIME_FUNCTION;
        std::string statusText = "OK";
        if (statusCode == 404) statusText = "Not Found";
        if (statusCode == 500) statusText = "Internal Server Error";
        if (statusCode == 405) statusText = "Method Not Allowed";
        
        std::ostringstream response;
        response << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n";
        response << "Content-Type: " << contentType << "\r\n";
        response << "Content-Length: " << content.length() << "\r\n";
        response << "Access-Control-Allow-Origin: *\r\n";
        response << "Connection: close\r\n";
        response << "\r\n";
        response << content;
        
        std::string responseStr = response.str();
        send(clientSocket, responseStr.c_str(), responseStr.length(), 0);
    }
    
    // Extract method and path from HTTP request
    std::pair<std::string, std::string> parseRequest(const std::string& request) {
        TIME_FUNCTION;
        std::istringstream iss(request);
        std::string method, path;
        iss >> method >> path;
        return {method, path};
    }
    
    // Extract file path from HTTP request
    std::string getFilePath(const std::string& request) {
        auto [method, path] = parseRequest(request);
        
        // Only handle GET requests for files
        if (method != "GET") {
            return "";
        }
        
        // Default to index.html for root path
        if (path == "/") {
            return "index.html";
        }
        
        // Remove leading slash
        if (path.length() > 0 && path[0] == '/') {
            path = path.substr(1);
        }
        
        return path;
    }
    
    // Check if path is a registered route
    bool isRoute(const std::string& path) {
        return routes.find(path) != routes.end();
    }
    
    // Handle route request
    void handleRoute(int clientSocket, const std::string& request) {
        TIME_FUNCTION;
        auto [method, path] = parseRequest(request);
        
        // Extract request body if present
        std::string body;
        size_t bodyPos = request.find("\r\n\r\n");
        if (bodyPos != std::string::npos) {
            body = request.substr(bodyPos + 4);
        }
        
        if (routes.find(path) != routes.end()) {
            auto [statusCode, response] = routes[path](method, body);
            sendResponse(clientSocket, response, "application/json", statusCode);
        } else {
            sendResponse(clientSocket, "404 Not Found", "text/plain", 404);
        }
    }

public:
    SimpleHTTPServer(int port = 8080, const std::string& webRoot = "web") 
        : port(port), serverSocket(-1), running(false), webRoot(webRoot) {}
    
    ~SimpleHTTPServer() {
        stop();
    }
    
    // Add a route handler
    void addRoute(const std::string& path, RouteHandler handler) {
        routes[path] = handler;
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
                auto [method, path] = parseRequest(request);
                
                // Check if this is a registered route
                if (isRoute(path)) {
                    handleRoute(clientSocket, request);
                } else {
                    // Handle file serving for GET requests
                    std::string filePath = getFilePath(request);
                    
                    if (!filePath.empty()) {
                        std::cout << "Serving: " << filePath << std::endl;
                        
                        std::string fullPath = webRoot + "/" + filePath;
                        std::string content = readFile(fullPath);
                        
                        if (!content.empty()) {
                            sendResponse(clientSocket, content, getContentType(filePath));
                        } else {
                            sendResponse(clientSocket, "404 Not Found: " + filePath, "text/plain", 404);
                        }
                    } else {
                        sendResponse(clientSocket, "400 Bad Request", "text/plain", 400);
                    }
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