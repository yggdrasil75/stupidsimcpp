#ifndef FRAME_HPP
#define FRAME_HPP

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <queue>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <iostream>

class frame {
private:
    std::vector<uint8_t> _data;
    std::unordered_map<size_t, uint8_t> overheadmap;
    size_t ratio = 1;
    size_t sourceSize = 0;
    size_t width = 0;
    size_t height = 0;
    
public:
    enum class colormap {
        RGB,
        RGBA,
        BGR,
        BGRA,
        B
    };
    
    enum class compresstype {
        RLE,
        DIFF,
        DIFFRLE,
        LZ78,
        HUFFMAN,
        RAW
    };

    colormap colorFormat = colormap::RGB;
    compresstype cformat = compresstype::RAW;

    size_t getWidth() {
        return width;
    }
    size_t getHeight() {
        return height;
    }
    frame() {};
    frame(size_t w, size_t h, colormap format = colormap::RGB) 
        : width(w), height(h), colorFormat(format), cformat(compresstype::RAW) {
        size_t channels = 3;
        switch (format) {
            case colormap::RGBA: channels = 4; break;
            case colormap::BGR: channels = 3; break;
            case colormap::BGRA: channels = 4; break;
            case colormap::B: channels = 1; break;
            default: channels = 3; break;
        }
        _data.resize(width * height * channels);
    }

    void setData(const std::vector<uint8_t>& data) {
        _data = data;
        cformat = compresstype::RAW;
    }

    const std::vector<uint8_t>& getData() const {
        return _data;
    }

    // Run-Length Encoding (RLE) compression
    frame& compressFrameRLE() {
        TIME_FUNCTION;
        if (_data.empty()) {
            return *this;
        }
        if (cformat == compresstype::DIFF) {
            cformat = compresstype::DIFFRLE;
        } else if (cformat == compresstype::RLE) {
            return *this;
        } else {
            cformat = compresstype::RLE;
        }
        
        std::vector<uint8_t> compressedData;
        compressedData.reserve(_data.size() * 2);
        
        size_t width = 1;
        for (size_t i = 0; i < _data.size(); i++) {
            if (_data[i] == _data[i+1] && width < 255) {
                width++;
            } else {
                compressedData.push_back(width);
                compressedData.push_back(_data[i]);
                width = 1;
            }

        }
        ratio = compressedData.size() - _data.size();
        sourceSize = _data.size();
        _data.clear();
        _data = compressedData;
        return *this;
    }

    frame& decompressFrameRLE() {
        TIME_FUNCTION;
        std::vector<uint8_t> decompressed;
        decompressed.reserve(sourceSize);
        
        if (_data.size() % 2 != 0) {
            throw std::runtime_error("something broke (decompressFrameRLE)");
        }
        for (size_t i = 0; i < _data.size(); i+=2) {
            uint8_t width = _data[i];
            uint8_t value = _data[i+1];
            decompressed.insert(decompressed.end(),width, value);
        }
        
        _data = std::move(decompressed);
        cformat = compresstype::RAW;
        
        return *this;
    }

    // Differential compression
    frame& compressFrameDiff() {
        // TODO
        std::logic_error("Function not yet implemented");
    }

    frame& decompressFrameDiff() {
        // TODO
        std::logic_error("Function not yet implemented");
    }
    // Huffman compression
    frame& compressFrameHuffman() {
        // TODO
        std::logic_error("Function not yet implemented");
    }

    // Combined compression methods
    frame& compressFrameZigZagRLE() {
        // TODO
        std::logic_error("Function not yet implemented");
    }

    frame& compressFrameDiffRLE() {
        // TODO
        std::logic_error("Function not yet implemented");
    }

    // Generic decompression that detects compression type
    frame& decompress() {
        switch (cformat) {
            case compresstype::RLE:
                return decompressFrameRLE();
                break;
            case compresstype::DIFF:
                return decompressFrameDiff();
                break;
            case compresstype::DIFFRLE:
                // For combined methods, first decompress RLE then the base method
                decompressFrameRLE();
                cformat = compresstype::DIFF;
                return decompressFrameDiff();
                break;
            case compresstype::HUFFMAN:
                // Huffman decompression would be implemented here
                throw std::runtime_error("Huffman decompression not fully implemented");
                break;
            case compresstype::RAW:
            default:
                return *this; // Already decompressed
        }
    }

    double getCompressionRatio() const {
        if (_data.empty() || sourceSize == 0) return 0.0;
        return static_cast<double>(sourceSize) / _data.size();
    }

    // Get source size (uncompressed size)
    size_t getSourceSize() const {
        return sourceSize;
    }

    // Get compressed size
    size_t getCompressedSize() const {
        return _data.size();
    }

    // Print compression information
    void printCompressionInfo() const {
        std::cout << "Compression Type: ";
        switch (cformat) {
            case compresstype::RLE: std::cout << "RLE"; break;
            case compresstype::DIFF: std::cout << "DIFF"; break;
            case compresstype::DIFFRLE: std::cout << "DIFF + RLE"; break;
            case compresstype::HUFFMAN: std::cout << "HUFFMAN"; break;
            case compresstype::RAW: std::cout << "RAW (uncompressed)"; break;
            default: std::cout << "UNKNOWN"; break;
        }
        std::cout << std::endl;
        
        std::cout << "Source Size: " << getSourceSize() << " bytes" << std::endl;
        std::cout << "Compressed Size: " << getCompressedSize() << " bytes" << std::endl;
        std::cout << "Compression Ratio: " << getCompressionRatio() << ":1" << std::endl;
        
        if (getCompressionRatio() > 1.0) {
            double savings = (1.0 - (1.0 / getCompressionRatio())) * 100.0;
            std::cout << "Space Savings: " << savings << "%" << std::endl;
        }
    }

    // Print compression information in a compact format
    void printCompressionStats() const {
        std::cout << "[" << getCompressionTypeString() << "] "
                  << getSourceSize() << "B -> " << getCompressedSize() << "B "
                  << "(ratio: " << getCompressionRatio() << ":1)" << std::endl;
    }

    // Get compression type as string
    std::string getCompressionTypeString() const {
        switch (cformat) {
            case compresstype::RLE: return "RLE";
            case compresstype::DIFF: return "DIFF";
            case compresstype::DIFFRLE: return "DIFF+RLE";
            case compresstype::HUFFMAN: return "HUFFMAN";
            case compresstype::RAW: return "RAW";
            default: return "UNKNOWN";
        }
    }

    compresstype getCompressionType() const {
        return cformat;
    }

    const std::unordered_map<size_t, uint8_t>& getOverheadMap() const {
        return overheadmap;
    }

    bool isCompressed() const {
        return cformat != compresstype::RAW;
    }
};

#endif