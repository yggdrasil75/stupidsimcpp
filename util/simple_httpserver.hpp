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

class SimpleHTTPServer {
private:
    int serverSocket;
    int port;
    bool running;
    std::string webRoot;
    
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
    
    // Extract file path from HTTP request
    std::string getFilePath(const std::string& request) {
        // Find the start of the path after "GET "
        size_t start = request.find("GET ") + 4;
        if (start == std::string::npos) return "";
        
        // Find the end of the path (space or ?)
        size_t end = request.find(" ", start);
        if (end == std::string::npos) end = request.find("?", start);
        if (end == std::string::npos) return "";
        
        std::string path = request.substr(start, end - start);
        
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