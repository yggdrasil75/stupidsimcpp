#ifndef FRAME_HPP
#define FRAME_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <queue>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <iostream>
#include <future>
#include <mutex>
#include <atomic>
#include <cmath>
#include <iomanip>
#include "../timing_decorator.hpp"

#if !defined(FRAME_ENABLE_OPENGL)
    #if defined(__gl_h_) || defined(__GL_H__) || defined(__glew_h__) || \
        defined(__glad_h_) || defined(GLAD_GL_H) || defined(__gl3_h_)
        #define FRAME_ENABLE_OPENGL 1
    #endif
#endif

class frame {
private:
    std::vector<uint8_t> _data;
    std::vector<uint16_t> _compressedData;
    std::unordered_map<uint16_t, std::vector<uint8_t>> overheadmap;
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

    enum class interpolation {
        NEAREST,
        BILINEAR,
        AREA,
        LANCZOS4
    };
private:
    size_t getChannels(colormap fmt) const {
        switch (fmt) {
            case colormap::RGBA: return 4;
            case colormap::BGR: return 3;
            case colormap::BGRA: return 4;
            case colormap::B: return 1;
            
            case colormap::RGB: 
            default:
                return 3;
        }
    }

    void resetState(size_t newSize) {
        cformat = compresstype::RAW;
        _compressedData.clear();
        _compressedData.shrink_to_fit();
        overheadmap.clear();
        sourceSize = newSize;
    }
    
    float rgbToGrayscale(float r, float g, float b) const {
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
public:
    colormap colorFormat = colormap::RGB;
    compresstype cformat = compresstype::RAW;

    const size_t& getWidth() const {
        return width;
    }
    
    const size_t& getHeight() const {
        return height;
    }
    frame() {};
    frame(size_t w, size_t h, colormap format = colormap::RGB) 
        : width(w), height(h), colorFormat(format), cformat(compresstype::RAW) {
        _data.resize(width * height * getChannels(format));
    }

    void setData(const std::vector<uint8_t>& data) {
        _data = data;
        resetState(data.size());
    }

    void setData(const std::vector<uint8_t>& inputData, colormap inputFormat) {
        if (inputFormat == colorFormat) {
            setData(inputData);
            return;
        }

        size_t srcChannels = getChannels(inputFormat);
        size_t dstChannels = getChannels(colorFormat);
        size_t numPixels = width * height;

        if (inputData.size() != numPixels * srcChannels) {
            throw std::runtime_error("Input data size does not match frame dimensions for the specified format.");
        }

        std::vector<uint8_t> newData;
        newData.reserve(numPixels * dstChannels);

        for (size_t i = 0; i < numPixels; ++i) {
            size_t px = i * srcChannels;
            uint8_t r = 0, g = 0, b = 0, a = 255;

            switch (inputFormat) {
                case colormap::RGB: {
                    r = inputData[px];
                    g = inputData[px+1];
                    b = inputData[px+2];
                    break;
                }  
                case colormap::RGBA:
                    r = inputData[px];
                    g = inputData[px+1];
                    b = inputData[px+2];
                    a = inputData[px+3];
                    break;
                case colormap::BGR:
                    b = inputData[px];
                    g = inputData[px+1];
                    r = inputData[px+2];
                    break;
                case colormap::BGRA:
                    b = inputData[px];
                    g = inputData[px+1];
                    r = inputData[px+2];
                    a = inputData[px+3];
                    break;
                case colormap::B:
                    r = g = b = inputData[px];
                    break;
            }

            switch (colorFormat) {
                case colormap::RGB:
                    newData.push_back(r);
                    newData.push_back(g);
                    newData.push_back(b);
                    break;
                case colormap::RGBA:
                    newData.push_back(r);
                    newData.push_back(g);
                    newData.push_back(b);
                    newData.push_back(a);
                    break;
                case colormap::BGR:
                    newData.push_back(b);
                    newData.push_back(g);
                    newData.push_back(r);
                    break;
                case colormap::BGRA:
                    newData.push_back(b);
                    newData.push_back(g);
                    newData.push_back(r);
                    newData.push_back(a);
                    break;
                case colormap::B:    
                    newData.push_back(rgbToGrayscale(r, g, b));
                    break;
            }
        }

        _data = std::move(newData);
        resetState(_data.size());
    }

    void setData(const std::vector<float>& inputData) {
        size_t channels = getChannels(colorFormat);
        
        if (inputData.size() != width * height * channels) {
            throw std::runtime_error("Input float data size does not match frame dimensions.");
        }

        std::vector<uint8_t> newData;
        newData.reserve(inputData.size());

        for (float val : inputData) {
            // Clamp between 0.0 and 1.0, scale to 255
            float v = std::max(0.0f, std::min(1.0f, val));
            newData.push_back(static_cast<uint8_t>(v * 255.0f));
        }

        _data = std::move(newData);
        resetState(_data.size());
    }

    void setData(const std::vector<float>& inputData, colormap inputFormat) {
        size_t srcChannels = getChannels(inputFormat);
        size_t dstChannels = getChannels(colorFormat);
        size_t numPixels = width * height;

        if (inputData.size() != numPixels * srcChannels) {
            throw std::runtime_error("Input float data size does not match frame dimensions.");
        }

        std::vector<uint8_t> newData;
        newData.reserve(numPixels * dstChannels);

        auto floatToByte = [](float f) -> uint8_t {
            return static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, f)) * 255.0f);
        };

        for (size_t i = 0; i < numPixels; ++i) {
            size_t px = i * srcChannels;
            uint8_t r = 0, g = 0, b = 0, a = 255;

            // Extract and convert floats to bytes
            switch (inputFormat) {
                case colormap::RGB:  
                    r = floatToByte(inputData[px]);
                    g = floatToByte(inputData[px+1]);
                    b = floatToByte(inputData[px+2]);
                    break;
                case colormap::RGBA: 
                    r = floatToByte(inputData[px]);
                    g = floatToByte(inputData[px+1]);
                    b = floatToByte(inputData[px+2]);
                    a = floatToByte(inputData[px+3]);
                    break;
                case colormap::BGR:  
                    b = floatToByte(inputData[px]);
                    g = floatToByte(inputData[px+1]);
                    r = floatToByte(inputData[px+2]);
                    break;
                case colormap::BGRA: 
                    b = floatToByte(inputData[px]);
                    g = floatToByte(inputData[px+1]);
                    r = floatToByte(inputData[px+2]);
                    a = floatToByte(inputData[px+3]);
                    break;
                case colormap::B:    
                    r = g = b = floatToByte(inputData[px]);
                    break;
            }

            switch (colorFormat) {
                case colormap::RGB:
                    newData.push_back(r);
                    newData.push_back(g);
                    newData.push_back(b);
                    break;
                case colormap::RGBA:
                    newData.push_back(r);
                    newData.push_back(g);
                    newData.push_back(b);
                    newData.push_back(a);
                    break;
                case colormap::BGR:
                    newData.push_back(b);
                    newData.push_back(g);
                    newData.push_back(r);
                    break;
                case colormap::BGRA:
                    newData.push_back(b);
                    newData.push_back(g);
                    newData.push_back(r);
                    newData.push_back(a);
                    break;
                case colormap::B:    
                    newData.push_back(rgbToGrayscale(r, g, b));
                    break;
            }
        }

        _data = std::move(newData);
        resetState(_data.size());
    }

    const std::vector<uint8_t>& getData() const {
        return _data;
    }
    
    std::vector<uint8_t> getPixel(size_t x, size_t y) const {
        if (cformat != compresstype::RAW) {
            throw std::runtime_error("Cannot get pixel data from a compressed frame. Decompress first.");
        }
        if (x >= width || y >= height) {
            throw std::out_of_range("Pixel coordinates out of bounds.");
        }

        size_t channels = getChannels(colorFormat);
        size_t index = (y * width + x) * channels;
        
        std::vector<uint8_t> pixel;
        pixel.reserve(channels);

        for (size_t i = 0; i < channels; ++i) {
            pixel.push_back(_data[index + i]);
        }
        return pixel;
    }

    void setPixel(size_t x, size_t y, const std::vector<uint8_t>& values) {
        if (cformat != compresstype::RAW) {
            throw std::runtime_error("Cannot set pixel data on a compressed frame. Decompress first.");
        }
        if (x >= width || y >= height) {
            throw std::out_of_range("Pixel coordinates out of bounds.");
        }

        size_t channels = getChannels(colorFormat);
        if (values.size() != channels) {
            throw std::invalid_argument("Input value count does not match frame channel count.");
        }

        size_t index = (y * width + x) * channels;
        for (size_t i = 0; i < channels; ++i) {
            _data[index + i] = values[i];
        }
        
        // Since data changed, previous compression stats are invalid
        resetState(_data.size());
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
        } else if (cformat == compresstype::RAW) {
            cformat = compresstype::RLE;
        } else {
            throw std::runtime_error("compressFrameRLE: frame is already compressed with an incompatible codec; decompress first");
        }
        
        std::vector<uint16_t> compressedData;
        compressedData.reserve(_data.size() * 2);
        
        size_t width = 1;
        for (size_t i = 0; i < _data.size(); i++) {
            if (i + 1 < _data.size() && _data[i] == _data[i+1] && width < 65535) {
                width++;
            } else {
                compressedData.push_back(width);
                compressedData.push_back(_data[i]);
                width = 1;
            }
        }
        sourceSize = _data.size();
        _compressedData = std::move(compressedData);
        _data.clear();
        _data.shrink_to_fit();
        return *this;
    }

    // LZ78 compression
    frame& compressFrameLZ78() {
        TIME_FUNCTION;
        if (_data.empty()) {
            return *this;
        }
        if (cformat != compresstype::RAW) {
            throw std::runtime_error("LZ78 compression can only be applied to raw data");
        }
        
        std::unordered_map<uint32_t, uint16_t> dict;
        dict.reserve(65536);

        uint16_t nextDict = 1;
        uint16_t cpos = 0;
        
        for (uint8_t byte : _data) {
            uint32_t newval = (static_cast<uint32_t>(cpos) << 8) | byte;
            auto it = dict.find(newval);
            if (it != dict.end()) {
                cpos = it->second;
            } else {
                _compressedData.push_back(cpos);
                _compressedData.push_back(byte);
                if (nextDict < 65535) {
                    dict[newval] = nextDict++;
                } else {
                    dict.clear();
                    nextDict = 1;
                }
                cpos = 0;
            }
        }
        if (cpos != 0) {
            _compressedData.push_back(cpos);
            _compressedData.push_back(0);
        }

        sourceSize = _data.size();
        _data.clear();
        _data.shrink_to_fit();
        
        cformat = compresstype::LZ78;

        return *this;
    }

    // Differential compression
    frame& compressFrameDiff() {
        TIME_FUNCTION;
        if (_data.empty()) return *this;
        if (cformat != compresstype::RAW) {
            throw std::runtime_error("compressFrameDiff: frame is already compressed; decompress first");
        }
        size_t ch = getChannels(colorFormat);
        if (_data.size() >= ch) {
            for (size_t i = _data.size() - 1; i >= ch; --i) {
                _data[i] = static_cast<uint8_t>(_data[i] - _data[i - ch]);
            }
        }
        sourceSize = _data.size();
        cformat = compresstype::DIFF;
        return *this;
    }

    // Huffman compression
    frame& compressFrameHuffman() {
        TIME_FUNCTION;
        if (_data.empty()) return *this;
        if (cformat != compresstype::RAW) {
            throw std::runtime_error("compressFrameHuffman: frame is already compressed; decompress first");
        }

        std::array<uint64_t, 256> hfreq{};
        for (uint8_t b : _data) ++hfreq[b];

        struct HNode {
            uint64_t f;
            int sym;
            int left, right;
        };
        std::vector<HNode> hnodes;
        auto hcmp = [&hnodes](int a, int b) { return hnodes[a].f > hnodes[b].f; };
        std::priority_queue<int, std::vector<int>, decltype(hcmp)> hpq(hcmp);
        for (int s = 0; s < 256; ++s) {
            if (hfreq[s] > 0) {
                hnodes.push_back({hfreq[s], s, -1, -1});
                hpq.push(static_cast<int>(hnodes.size()) - 1);
            }
        }
        if (hpq.size() == 1) {
            hnodes.push_back({0, -1, -1, -1});
            hpq.push(static_cast<int>(hnodes.size()) - 1);
        }
        while (hpq.size() > 1) {
            int a = hpq.top();
            hpq.pop();
            int b = hpq.top();
            hpq.pop();
            hnodes.push_back({hnodes[a].f + hnodes[b].f, -1, a, b});
            hpq.push(static_cast<int>(hnodes.size()) - 1);
        }
        std::array<uint8_t, 256> codeLen{};
        std::function<void(int, int)> walk = [&](int n, int depth) {
            if (hnodes[n].sym >= 0) {
                codeLen[hnodes[n].sym] = static_cast<uint8_t>(std::max(depth, 1));
                return;
            }
            if (hnodes[n].left < 0) return;
            walk(hnodes[n].left, depth + 1);
            walk(hnodes[n].right, depth + 1);
        };
        walk(hpq.top(), 0);

        std::array<uint32_t, 256> code{};
        huffmanCanonicalCodes(codeLen, code);

        _compressedData.clear();
        _compressedData.reserve(257 + _data.size() / 3);
        for (int s = 0; s < 256; ++s) _compressedData.push_back(codeLen[s]);
        _compressedData.push_back(0);

        uint32_t acc = 0;
        int accBits = 0;
        for (uint8_t b : _data) {
            acc = (acc << codeLen[b]) | code[b];
            accBits += codeLen[b];
            while (accBits >= 16) {
                _compressedData.push_back(static_cast<uint16_t>(acc >> (accBits - 16)));
                accBits -= 16;
                acc &= (1u << accBits) - 1;
            }
        }
        uint16_t lastBits = 16;
        if (accBits > 0) {
            _compressedData.push_back(static_cast<uint16_t>(acc << (16 - accBits)));
            lastBits = static_cast<uint16_t>(accBits);
        }
        _compressedData[256] = lastBits;

        sourceSize = _data.size();
        _data.clear();
        _data.shrink_to_fit();
        cformat = compresstype::HUFFMAN;
        return *this;
    }

    frame& compressFrameZigZagRLE() {
        // TODO
        throw std::logic_error("Function not yet implemented");
    }

    frame& compressFrameDiffRLE() {
        TIME_FUNCTION;
        if (_data.empty()) return *this;
        size_t original = _data.size();
        compressFrameDiff();
        compressFrameRLE();
        sourceSize = original;
        return *this;
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
                decompressFrameRLE();
                cformat = compresstype::DIFF;
                return decompressFrameDiff();
                break;
            case compresstype::LZ78:
                return decompressFrameLZ78();
                break;
            case compresstype::HUFFMAN:
                return decompressFrameHuffman();
                break;
            case compresstype::RAW:
            default:
                return *this; // Already decompressed
        }
    }

    // Calculate the size of the dictionary in bytes
    size_t getDictionarySize() const {
        size_t dictSize = 0;
        dictSize = sizeof(overheadmap);
        return dictSize;
    }

    // Get compressed size including dictionary overhead
    size_t getTotalCompressedSize() const {
        size_t baseSize = getCompressedDataSize();
        return baseSize;
    }

    double getCompressionRatio() const {
        if (_compressedData.empty() || sourceSize == 0) return 0.0;
        return static_cast<double>(sourceSize) / getTotalCompressedSize();
    }

    size_t getSourceSize() const {
        return sourceSize;
    }

    size_t getCompressedDataSize() const {
        return _compressedData.size() * 2;
    }

    void printCompressionInfo() const {
        std::cout << "Compression Type: ";
        switch (cformat) {
            case compresstype::RLE: 
                std::cout << "RLE";
                break;
            case compresstype::DIFF:
                std::cout << "DIFF";
                break;
            case compresstype::DIFFRLE:
                std::cout << "DIFF + RLE";
                break;
            case compresstype::LZ78:
                std::cout << "LZ78 (kinda)";
                break;
            case compresstype::HUFFMAN:
                std::cout << "HUFFMAN";
                break;
            case compresstype::RAW:
                std::cout << "RAW (uncompressed)";
                break;
            default:
                std::cout << "UNKNOWN";
                break;
        }
        std::cout << std::endl;
        
        std::cout << "Source Size: " << getSourceSize() << " bytes" << std::endl;
        std::cout << "Compressed data Size: " << getCompressedDataSize() << " 16-bit words" << std::endl;
        std::cout << "Compressed Size: " << getCompressedDataSize() * 2 << " bytes" << std::endl;
        
        if (cformat == compresstype::LZ78) {
            std::cout << "Dictionary Size: " << getDictionarySize() << " bytes" << std::endl;
            std::cout << "Dictionary Entries: " << overheadmap.size() << std::endl;
            std::cout << "Total Compressed Size: " << getTotalCompressedSize() << " bytes" << std::endl;
        } else {
            std::cout << "Total Compressed Size: " << getTotalCompressedSize() << " bytes" << std::endl;
        }
        
        std::cout << "Compression Ratio: " << getCompressionRatio() << ":1" << std::endl;
        
        if (getCompressionRatio() > 1.0) {
            double savings = (1.0 - (1.0 / getCompressionRatio())) * 100.0;
            std::cout << "Space Savings: " << savings << "%" << std::endl;
        }
    }

    void printCompressionStats() const {
        std::cout << "[" << getCompressionTypeString() << "] "
                    << getSourceSize() << "B -> " << getTotalCompressedSize() << "B "
                    << "(ratio: " << getCompressionRatio() << ":1)" << std::endl;
    }

    // Get compression type as string
    std::string getCompressionTypeString() const {
        switch (cformat) {
            case compresstype::RLE: return "RLE";
            case compresstype::DIFF: return "DIFF";
            case compresstype::DIFFRLE: return "DIFF+RLE";
            case compresstype::LZ78: return "LZ78";
            case compresstype::HUFFMAN: return "HUFFMAN";
            case compresstype::RAW: return "RAW";
            default: return "UNKNOWN";
        }
    }

    compresstype getCompressionType() const {
        return cformat;
    }

    bool isCompressed() const {
        return cformat != compresstype::RAW;
    }

    //does this actually work? am I overthinking memory management?
    void free() {
        overheadmap.clear();
        overheadmap.rehash(0);
        _compressedData.clear();
        _data.clear();
        _compressedData.shrink_to_fit();
        _data.shrink_to_fit();
    }

    void resize(size_t newWidth, size_t newHeight, interpolation method = interpolation::NEAREST) {
        TIME_FUNCTION;
        if (cformat != compresstype::RAW) {
            throw std::runtime_error("Cannot resize a compressed frame. Decompress first.");
        }
        if (newWidth == 0 || newHeight == 0) {
            throw std::invalid_argument("Target dimensions must be non-zero.");
        }
        if (width == newWidth && height == newHeight) {
            return;
        }

        size_t channels = getChannels(colorFormat);
        std::vector<uint8_t> newData;
        newData.resize(newWidth * newHeight * channels);

        if (method == interpolation::NEAREST) {
            resizeNearest(newData, newWidth, newHeight, channels);
        } else if (method == interpolation::BILINEAR) {
            resizeBilinear(newData, newWidth, newHeight, channels);
        }

        _data = std::move(newData);
        width = newWidth;
        height = newHeight;
        
        resetState(_data.size());
    }

private:
    std::vector<std::vector<uint8_t>> sortvecs(std::vector<std::vector<uint8_t>> source) {
        std::sort(source.begin(), source.end(), [](const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {return a.size() > b.size();});
        return source;
    }
    
    frame& decompressFrameLZ78() {
        TIME_FUNCTION;
        if (cformat != compresstype::LZ78) {
            throw std::runtime_error("Data is not LZ78 compressed");
        }

        if (_compressedData.size() % 2 != 0) {
            throw std::runtime_error("decompressFrameLZ78: corrupt stream (odd word count)");
        }

        std::vector<std::vector<uint8_t>> dict;
        dict.reserve(65536);
        dict.emplace_back();
        uint16_t nextdict = 1;

        for (size_t i = 0; i < _compressedData.size(); i+=2) {
            uint16_t cpos = _compressedData[i];
            uint8_t byte = static_cast<uint8_t>(_compressedData[i+1]);
            if (cpos >= dict.size()) {
                throw std::runtime_error("decompressFrameLZ78: invalid dictionary reference");
            }
            std::vector<uint8_t> seq = dict[cpos];
            seq.push_back(byte);
            _data.insert(_data.end(), seq.begin(), seq.end());
            if (nextdict < 65535) {
                dict.push_back(std::move(seq));
                nextdict++;
            } else {
                dict.resize(1);
                nextdict = 1;
            }
        }

        if (sourceSize != 0) {
            if (_data.size() == sourceSize + 1) _data.pop_back();
            if (_data.size() != sourceSize) {
                throw std::runtime_error("decompressFrameLZ78: output size mismatch");
            }
        }

        _compressedData.clear();
        _compressedData.shrink_to_fit();
        cformat = compresstype::RAW;

        return *this;
    }

    frame& decompressFrameRLE() {
        TIME_FUNCTION;
        std::vector<uint8_t> decompressed;
        decompressed.reserve(sourceSize);
        
        if (_compressedData.size() % 2 != 0) {
            throw std::runtime_error("something broke (decompressFrameRLE)");
        }
        
        for (size_t i = 0; i < _compressedData.size(); i += 2) {
            uint16_t width = _compressedData[i];
            uint8_t value = static_cast<uint8_t>(_compressedData[i+1]);
            decompressed.insert(decompressed.end(), width, value);
        }
        
        _data = std::move(decompressed);
        _compressedData.clear();
        cformat = compresstype::RAW;
        
        return *this;
    }

    std::vector<std::vector<uint8_t>> getRepeats() {
        TIME_FUNCTION;
        std::vector<std::vector<uint8_t>> result;
        size_t pos = 0;
        const size_t chunksize = 65535;
        size_t dsize = _data.size();
        
        // Thread-safe storage with mutex protection
        struct ThreadSafeMatches {
            std::mutex mutex;
            std::vector<std::vector<uint8_t>> matches128plus;
            std::vector<std::vector<uint8_t>> matches64plus;
            //std::vector<std::vector<uint8_t>> matches32plus;
            //std::vector<std::vector<uint8_t>> matchesAll;
            
            void addMatch(std::vector<uint8_t>&& match, size_t length) {
                std::lock_guard<std::mutex> lock(mutex);
                if (length >= 128) {
                    if (matches128plus.size() < 65534) matches128plus.push_back(std::move(match));
                } else if (length >= 64) {
                    if (matches64plus.size() < 65534) matches64plus.push_back(std::move(match));
                } 
                // else if (length >= 32) {
                //     if (matches32plus.size() < 65534) matches32plus.push_back(std::move(match));
                // }
                // else {
                //     if (matchesAll.size() < 65534) matchesAll.push_back(std::move(match));
                // }
            }
        };
        
        ThreadSafeMatches threadMatches;
        
        while (pos < dsize && result.size() < 65534) {
            size_t chunk_end = std::min(pos + chunksize, dsize);
            std::vector<uint8_t> chunk(_data.begin() + pos, _data.begin() + chunk_end);
            
            if (chunk.size() <= 4) {
                pos = chunk_end;
                continue;
            }

            if (result.size() < 65534) {
                result.push_back(chunk);
            }

            std::vector<uint8_t> ffour(chunk.begin(), chunk.begin() + 4);
            
            // Split the search space across multiple threads
            const size_t num_threads = std::thread::hardware_concurrency();
            const size_t search_range = dsize - chunk_end - 3;
            const size_t block_size = (search_range + num_threads - 1) / num_threads;
            
            std::vector<std::future<void>> futures;
            
            for (size_t t = 0; t < num_threads; ++t) {
                size_t start = chunk_end + t * block_size;
                size_t end = std::min(start + block_size, dsize - 3);
                
                if (start >= end) continue;
                
                futures.push_back(std::async(std::launch::async, 
                    [&, start, end, chunk, ffour]() {
                        size_t searchpos = start;
                        while (searchpos <= end) {
                            // Check first 4 bytes
                            if (_data[searchpos] == ffour[0] &&
                                _data[searchpos + 1] == ffour[1] &&
                                _data[searchpos + 2] == ffour[2] &&
                                _data[searchpos + 3] == ffour[3]) {
                                
                                // Found match, calculate length
                                size_t matchlength = 4;
                                size_t chunk_compare_pos = 4;
                                size_t input_compare_pos = searchpos + 4;

                                while (chunk_compare_pos < chunk.size() && 
                                    input_compare_pos < dsize && 
                                    _data[input_compare_pos] == chunk[chunk_compare_pos]) {
                                    matchlength++;
                                    chunk_compare_pos++;
                                    input_compare_pos++;
                                }

                                std::vector<uint8_t> matchsequence(
                                    _data.begin() + searchpos, 
                                    _data.begin() + searchpos + matchlength
                                );
                                
                                threadMatches.addMatch(std::move(matchsequence), matchlength);
                                searchpos += matchlength;
                            } else {
                                searchpos++;
                            }
                        }
                    }
                ));
            }
            
            // Wait for all threads to complete
            for (auto& future : futures) {
                future.get();
            }
            
            pos = chunk_end;
        }
        
        // Merge results to main
        for (const auto& match : threadMatches.matches128plus) {
            result.push_back(match);
        }
        
        for (const auto& match : threadMatches.matches64plus) {
            if (result.size() < 65534) result.push_back(match);
            else break;
        }
        
        // for (const auto& match : threadMatches.matches32plus) {
        //     if (result.size() < 65534) result.push_back(match);
        //     else break;
        // }
        
        // for (const auto& match : threadMatches.matchesAll) {
        //     if (result.size() < 65534) result.push_back(match);
        //     else break;
        // }
        
        return result;
    }

    frame& decompressFrameDiff() {
        TIME_FUNCTION;
        size_t ch = getChannels(colorFormat);
        for (size_t i = ch; i < _data.size(); ++i) {
            _data[i] = static_cast<uint8_t>(_data[i] + _data[i - ch]);
        }
        cformat = compresstype::RAW;
        return *this;
    }

    static void huffmanCanonicalCodes(const std::array<uint8_t, 256>& codeLen,
                                      std::array<uint32_t, 256>& code) {
        // symbols sorted by (length, value) get consecutive codes
        std::vector<int> syms;
        for (int s = 0; s < 256; ++s)
            if (codeLen[s] > 0) syms.push_back(s);
        std::sort(syms.begin(), syms.end(), [&](int a, int b) {
            if (codeLen[a] != codeLen[b]) return codeLen[a] < codeLen[b];
            return a < b;
        });
        uint32_t c = 0;
        uint8_t prevLen = 0;
        for (int s : syms) {
            c <<= (codeLen[s] - prevLen);
            code[s] = c++;
            prevLen = codeLen[s];
        }
    }

    frame& decompressFrameHuffman() {
        TIME_FUNCTION;
        if (cformat != compresstype::HUFFMAN) {
            throw std::runtime_error("Data is not Huffman compressed");
        }
        if (_compressedData.size() < 257) {
            throw std::runtime_error("decompressFrameHuffman: stream too short");
        }

        std::array<uint8_t, 256> codeLen{};
        uint8_t maxLen = 0;
        for (int s = 0; s < 256; ++s) {
            codeLen[s] = static_cast<uint8_t>(_compressedData[s]);
            maxLen = std::max(maxLen, codeLen[s]);
        }
        if (maxLen == 0 || maxLen > 32) {
            throw std::runtime_error("decompressFrameHuffman: corrupt length table");
        }

        std::array<uint32_t, 256> code{};
        huffmanCanonicalCodes(codeLen, code);

        // (code, length) -> symbol
        std::unordered_map<uint64_t, uint8_t> decode;
        for (int s = 0; s < 256; ++s) {
            if (codeLen[s] > 0) {
                decode[(static_cast<uint64_t>(codeLen[s]) << 32) | code[s]] = static_cast<uint8_t>(s);
            }
        }

        uint16_t lastBits = _compressedData[256];
        _data.clear();
        _data.reserve(sourceSize);

        uint64_t acc = 0;
        int accLen = 0;
        for (size_t w = 257; w < _compressedData.size() && _data.size() < sourceSize; ++w) {
            int wordBits = (w + 1 == _compressedData.size()) ? lastBits : 16;
            acc = (acc << wordBits) | (_compressedData[w] >> (16 - wordBits));
            accLen += wordBits;

            bool progressed = true;
            while (progressed && _data.size() < sourceSize) {
                progressed = false;
                for (int len = 1; len <= maxLen && len <= accLen; ++len) {
                    uint32_t candidate = static_cast<uint32_t>(acc >> (accLen - len));
                    auto it = decode.find((static_cast<uint64_t>(len) << 32) | candidate);
                    if (it != decode.end()) {
                        _data.push_back(it->second);
                        accLen -= len;
                        acc &= (accLen > 0) ? ((1ull << accLen) - 1) : 0;
                        progressed = true;
                        break;
                    }
                }
            }
        }

        if (sourceSize != 0 && _data.size() != sourceSize) {
            throw std::runtime_error("decompressFrameHuffman: output size mismatch");
        }

        _compressedData.clear();
        _compressedData.shrink_to_fit();
        cformat = compresstype::RAW;
        return *this;
    }

    void resizeNearest(std::vector<uint8_t>& dst, size_t newW, size_t newH, size_t channels) {
        const double x_ratio = (double)width / newW;
        const double y_ratio = (double)height / newH;

        #pragma omp parallel for
        for (size_t y = 0; y < newH; ++y) {
            size_t srcY = static_cast<size_t>(std::floor(y * y_ratio));
            if (srcY >= height) srcY = height - 1;
            
            size_t destOffsetBase = y * newW * channels;
            size_t srcRowOffset = srcY * width * channels;

            for (size_t x = 0; x < newW; ++x) {
                size_t srcX = static_cast<size_t>(std::floor(x * x_ratio));
                if (srcX >= width) srcX = width - 1;

                size_t srcIndex = srcRowOffset + (srcX * channels);
                size_t destIndex = destOffsetBase + (x * channels);

                for (size_t c = 0; c < channels; ++c) {
                    dst[destIndex + c] = _data[srcIndex + c];
                }
            }
        }
    }

    void resizeBilinear(std::vector<uint8_t>& dst, size_t newW, size_t newH, size_t channels) {
        const float x_ratio = (width > 1) ? static_cast<float>(width - 1) / (newW - 1) : 0;
        const float y_ratio = (height > 1) ? static_cast<float>(height - 1) / (newH - 1) : 0;

        for (size_t y = 0; y < newH; ++y) {
            float srcY_f = y * y_ratio;
            size_t y_l = static_cast<size_t>(srcY_f);
            size_t y_h = (y_l + 1 < height) ? y_l + 1 : y_l;
            float y_weight = srcY_f - y_l;
            float y_inv = 1.0f - y_weight;

            size_t destOffsetBase = y * newW * channels;

            for (size_t x = 0; x < newW; ++x) {
                float srcX_f = x * x_ratio;
                size_t x_l = static_cast<size_t>(srcX_f);
                size_t x_h = (x_l + 1 < width) ? x_l + 1 : x_l;
                float x_weight = srcX_f - x_l;
                float x_inv = 1.0f - x_weight;

                size_t idx_TL = (y_l * width + x_l) * channels;
                size_t idx_TR = (y_l * width + x_h) * channels;
                size_t idx_BL = (y_h * width + x_l) * channels;
                size_t idx_BR = (y_h * width + x_h) * channels;
                size_t destIndex = destOffsetBase + (x * channels);

                for (size_t c = 0; c < channels; ++c) {
                    float val_TL = _data[idx_TL + c];
                    float val_TR = _data[idx_TR + c];
                    float val_BL = _data[idx_BL + c];
                    float val_BR = _data[idx_BR + c];

                    float top = val_TL * x_inv + val_TR * x_weight;
                    float bottom = val_BL * x_inv + val_BR * x_weight;
                    float result = top * y_inv + bottom * y_weight;
                    dst[destIndex + c] = static_cast<uint8_t>(result);
                }
            }
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, frame& f) {
    os << "Frame[" << f.getWidth() << "x" << f.getHeight() << "] ";
    
    // Color format
    os << "Format: ";
    switch (f.colorFormat) {
        case frame::colormap::RGB:
            os << "RGB";
            break;
        case frame::colormap::RGBA:
            os << "RGBA";
            break;
        case frame::colormap::BGR:
            os << "BGR";
            break;
        case frame::colormap::BGRA:
            os << "BGRA";
            break;
        case frame::colormap::B: 
            os << "Grayscale";
            break;
        default:
            os << "Unknown";
            break;
    }
    
    // Compression info
    os << " | Compression: " << f.getCompressionTypeString();
    
    // Size info
    if (f.isCompressed()) {
        os << " | " << f.getSourceSize() << "B -> " << f.getTotalCompressedSize() 
        << "B (ratio: " << std::fixed << std::setprecision(2) << f.getCompressionRatio() << ":1)";
    } else {
        os << " | Size: " << f.getData().size() << "B";
    }
    
    // Data status
    os << " | Data: ";
    if (!f.getData().empty()) {
        os << "raw(" << f.getData().size() << " bytes)";
    } else if (f.getCompressedDataSize() > 0) {
        os << "compressed(" << f.getCompressedDataSize() << " words)";
    } else {
        os << "empty";
    }
    
    return os;
}

#endif