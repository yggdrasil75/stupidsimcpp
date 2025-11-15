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

class frame {
private:
    std::vector<uint8_t> _data;
    std::vector<uint8_t> _compressedData;
    std::unordered_map<size_t, uint8_t> overheadmap;
    size_t width;
    size_t height;
    
    // Huffman coding structures
    struct HuffmanNode {
        uint8_t value;
        int freq;
        std::shared_ptr<HuffmanNode> left, right;
        
        HuffmanNode(uint8_t val, int f) : value(val), freq(f), left(nullptr), right(nullptr) {}
        HuffmanNode(int f, std::shared_ptr<HuffmanNode> l, std::shared_ptr<HuffmanNode> r) 
            : value(0), freq(f), left(l), right(r) {}
        
        bool isLeaf() const { return !left && !right; }
    };

    struct HuffmanCompare {
        bool operator()(const std::shared_ptr<HuffmanNode>& a, const std::shared_ptr<HuffmanNode>& b) {
            return a->freq > b->freq;
        }
    };

    void buildHuffmanCodes(const std::shared_ptr<HuffmanNode>& node, const std::string& code, 
                          std::unordered_map<uint8_t, std::string>& codes) {
        if (!node) return;
        
        if (node->isLeaf()) {
            codes[node->value] = code;
            return;
        }
        
        buildHuffmanCodes(node->left, code + "0", codes);
        buildHuffmanCodes(node->right, code + "1", codes);
    }

    std::vector<uint8_t> zigzagScan() {
        if (width == 0 || height == 0) return _data;
        
        std::vector<uint8_t> result;
        result.reserve(_data.size());
        
        for (size_t i = 0; i < width + height - 1; ++i) {
            if (i % 2 == 0) {
                // Even diagonal - go up
                for (size_t row = std::min(i, height - 1); row != (size_t)-1 && i - row < width; --row) {
                    size_t col = i - row;
                    result.push_back(_data[row * width + col]);
                }
            } else {
                // Odd diagonal - go down
                for (size_t col = std::min(i, width - 1); col != (size_t)-1 && i - col < height; --col) {
                    size_t row = i - col;
                    result.push_back(_data[row * width + col]);
                }
            }
        }
        
        return result;
    }

    std::vector<uint8_t> inverseZigzagScan(const std::vector<uint8_t>& zigzagData) {
        if (width == 0 || height == 0) return zigzagData;
        
        std::vector<uint8_t> result(_data.size(), 0);
        size_t idx = 0;
        
        for (size_t i = 0; i < width + height - 1; ++i) {
            if (i % 2 == 0) {
                // Even diagonal - go up
                for (size_t row = std::min(i, height - 1); row != (size_t)-1 && i - row < width; --row) {
                    size_t col = i - row;
                    result[row * width + col] = zigzagData[idx++];
                }
            } else {
                // Odd diagonal - go down
                for (size_t col = std::min(i, width - 1); col != (size_t)-1 && i - col < height; --col) {
                    size_t row = i - col;
                    result[row * width + col] = zigzagData[idx++];
                }
            }
        }
        
        return result;
    }

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
        ZIGZAG,
        DIFF,
        DIFFRLE,
        ZIGZAGRLE,
        LZ77,
        LZSS,
        HUFFMAN,
        RAW
    };

    colormap colorFormat;
    compresstype cformat;

    size_t getWidth() {
        return width;
    }
    size_t getHeight() {
        return height;
    }
    frame() {};
    frame(size_t w, size_t h, colormap format = colormap::RGB) 
        : width(w), height(h), colorFormat(format), cformat(compresstype::RAW) {
        size_t channels = 3; // Default for RGB
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
        _compressedData.clear();
        cformat = compresstype::RAW;
    }

    const std::vector<uint8_t>& getData() const {
        return _data;
    }

    const std::vector<uint8_t>& getCompressedData() const {
        return _compressedData;
    }

    // Run-Length Encoding (RLE) compression
    frame& compressFrameRLE() {
        if (_data.empty()) {
            _compressedData.clear();
            return *this;
        }
        
        if (cformat == compresstype::ZIGZAG) {
            cformat = compresstype::ZIGZAGRLE;
        } else if (cformat == compresstype::DIFF) {
            cformat = compresstype::DIFFRLE;
        } else {
            cformat = compresstype::RLE;
        }
        
        _compressedData.clear();
        _compressedData.reserve(_data.size() * 2);
        
        size_t i = 0;
        while (i < _data.size()) {
            uint8_t current = _data[i];
            size_t count = 1;
            
            // Count consecutive identical bytes
            while (i + count < _data.size() && _data[i + count] == current && count < 255) {
                count++;
            }
            
            if (count > 1) {
                // Encode run: 0xFF marker, count, value
                _compressedData.push_back(0xFF);
                _compressedData.push_back(static_cast<uint8_t>(count));
                _compressedData.push_back(current);
                i += count;
            } else {
                // Encode literal sequence
                size_t literal_start = i;
                while (i < _data.size() && 
                       (i + 1 >= _data.size() || _data[i] != _data[i + 1]) && 
                       (i - literal_start) < 127) {
                    i++;
                }
                
                size_t literal_length = i - literal_start;
                _compressedData.push_back(static_cast<uint8_t>(literal_length));
                
                for (size_t j = literal_start; j < i; ++j) {
                    _compressedData.push_back(_data[j]);
                }
            }
        }
        
        // Store compression metadata in overheadmap
        overheadmap[0] = static_cast<uint8_t>(cformat);
        overheadmap[1] = static_cast<uint8_t>(_compressedData.size() > 0 ? 1 : 0);
        
        return *this;
    }

    frame& decompressFrameRLE() {
        if (_compressedData.empty()) {
            return *this;
        }
        
        std::vector<uint8_t> decompressed;
        decompressed.reserve(_data.size());
        
        size_t i = 0;
        while (i < _compressedData.size()) {
            uint8_t marker = _compressedData[i++];
            
            if (marker == 0xFF) {
                // Run sequence
                if (i + 1 >= _compressedData.size()) {
                    throw std::runtime_error("Invalid RLE data");
                }
                
                uint8_t count = _compressedData[i++];
                uint8_t value = _compressedData[i++];
                
                for (int j = 0; j < count; ++j) {
                    decompressed.push_back(value);
                }
            } else {
                // Literal sequence
                uint8_t length = marker;
                if (i + length > _compressedData.size()) {
                    throw std::runtime_error("Invalid RLE data");
                }
                
                for (int j = 0; j < length; ++j) {
                    decompressed.push_back(_compressedData[i++]);
                }
            }
        }
        
        _data = std::move(decompressed);
        cformat = compresstype::RAW;
        overheadmap.clear();
        
        return *this;
    }

    // Zigzag compression
    frame& compressFrameZigZag() {
        if (cformat != compresstype::RAW) {
            throw std::runtime_error("Cannot apply zigzag to already compressed data");
        }
        
        cformat = compresstype::ZIGZAG;
        _compressedData = zigzagScan();
        
        // Store metadata
        overheadmap[0] = static_cast<uint8_t>(cformat);
        overheadmap[1] = static_cast<uint8_t>(width);
        overheadmap[2] = static_cast<uint8_t>(height);
        
        return *this;
    }

    frame& decompressFrameZigZag() {
        if (_compressedData.empty()) {
            return *this;
        }
        
        _data = inverseZigzagScan(_compressedData);
        cformat = compresstype::RAW;
        overheadmap.clear();
        
        return *this;
    }

    // Differential compression
    frame& compressFrameDiff() {
        if (cformat != compresstype::RAW) {
            throw std::runtime_error("Cannot apply diff to already compressed data");
        }
        
        cformat = compresstype::DIFF;
        _compressedData.clear();
        _compressedData.reserve(_data.size());
        
        if (_data.empty()) {
            return *this;
        }
        
        // First value remains the same
        _compressedData.push_back(_data[0]);
        
        // Subsequent values are differences
        for (size_t i = 1; i < _data.size(); ++i) {
            int16_t diff = static_cast<int16_t>(_data[i]) - static_cast<int16_t>(_data[i - 1]);
            // Convert to unsigned with bias of 128
            _compressedData.push_back(static_cast<uint8_t>((diff + 128) & 0xFF));
        }
        
        // Store metadata
        overheadmap[0] = static_cast<uint8_t>(cformat);
        overheadmap[1] = static_cast<uint8_t>(_data.size() > 0 ? 1 : 0);
        
        return *this;
    }

    frame& decompressFrameDiff() {
        if (_compressedData.empty()) {
            return *this;
        }
        
        std::vector<uint8_t> original;
        original.reserve(_compressedData.size());
        
        // First value is original
        original.push_back(_compressedData[0]);
        
        // Reconstruct subsequent values
        for (size_t i = 1; i < _compressedData.size(); ++i) {
            int16_t reconstructed = static_cast<int16_t>(original[i - 1]) + 
                                   (static_cast<int16_t>(_compressedData[i]) - 128);
            // Clamp to 0-255
            reconstructed = std::max(0, std::min(255, static_cast<int>(reconstructed)));
            original.push_back(static_cast<uint8_t>(reconstructed));
        }
        
        _data = std::move(original);
        cformat = compresstype::RAW;
        overheadmap.clear();
        
        return *this;
    }

    // Huffman compression
    frame& compressFrameHuffman() {
        cformat = compresstype::HUFFMAN;
        _compressedData.clear();
        
        if (_data.empty()) {
            return *this;
        }
        
        // Calculate frequency of each byte value
        std::unordered_map<uint8_t, int> freq;
        for (uint8_t byte : _data) {
            freq[byte]++;
        }
        
        // Build Huffman tree
        std::priority_queue<std::shared_ptr<HuffmanNode>, 
                           std::vector<std::shared_ptr<HuffmanNode>>, 
                           HuffmanCompare> pq;
        
        for (const auto& pair : freq) {
            pq.push(std::make_shared<HuffmanNode>(pair.first, pair.second));
        }
        
        while (pq.size() > 1) {
            auto left = pq.top(); pq.pop();
            auto right = pq.top(); pq.pop();
            
            auto parent = std::make_shared<HuffmanNode>(left->freq + right->freq, left, right);
            pq.push(parent);
        }
        
        auto root = pq.top();
        
        // Build codes
        std::unordered_map<uint8_t, std::string> codes;
        buildHuffmanCodes(root, "", codes);
        
        // Encode data
        std::string bitString;
        for (uint8_t byte : _data) {
            bitString += codes[byte];
        }
        
        // Convert bit string to bytes
        // Store frequency table size
        _compressedData.push_back(static_cast<uint8_t>(freq.size()));
        
        // Store frequency table
        for (const auto& pair : freq) {
            _compressedData.push_back(pair.first);
            // Store frequency as 4 bytes
            for (int i = 0; i < 4; ++i) {
                _compressedData.push_back(static_cast<uint8_t>((pair.second >> (i * 8)) & 0xFF));
            }
        }
        
        // Store encoded data
        uint8_t currentByte = 0;
        int bitCount = 0;
        
        for (char bit : bitString) {
            currentByte = (currentByte << 1) | (bit == '1' ? 1 : 0);
            bitCount++;
            
            if (bitCount == 8) {
                _compressedData.push_back(currentByte);
                currentByte = 0;
                bitCount = 0;
            }
        }
        
        // Pad last byte if necessary
        if (bitCount > 0) {
            currentByte <<= (8 - bitCount);
            _compressedData.push_back(currentByte);
            // Store number of padding bits
            _compressedData.push_back(static_cast<uint8_t>(8 - bitCount));
        } else {
            _compressedData.push_back(0); // No padding
        }
        
        // Store metadata
        overheadmap[0] = static_cast<uint8_t>(cformat);
        overheadmap[1] = static_cast<uint8_t>(freq.size());
        
        return *this;
    }

    // Combined compression methods
    frame& compressFrameZigZagRLE() {
        compressFrameZigZag();
        // Store intermediate zigzag data temporarily
        auto zigzagData = _compressedData;
        _data = std::move(zigzagData);
        cformat = compresstype::ZIGZAG;
        return compressFrameRLE();
    }

    frame& compressFrameDiffRLE() {
        compressFrameDiff();
        // Store intermediate diff data temporarily
        auto diffData = _compressedData;
        _data = std::move(diffData);
        cformat = compresstype::DIFF;
        return compressFrameRLE();
    }

    // Generic decompression that detects compression type
    frame& decompress() {
        switch (cformat) {
            case compresstype::RLE:
                return decompressFrameRLE();
            case compresstype::ZIGZAG:
                return decompressFrameZigZag();
            case compresstype::DIFF:
                return decompressFrameDiff();
            case compresstype::ZIGZAGRLE:
            case compresstype::DIFFRLE:
                // For combined methods, first decompress RLE then the base method
                decompressFrameRLE();
                // Now _data contains the intermediate compressed form
                if (cformat == compresstype::ZIGZAGRLE) {
                    cformat = compresstype::ZIGZAG;
                    return decompressFrameZigZag();
                } else {
                    cformat = compresstype::DIFF;
                    return decompressFrameDiff();
                }
            case compresstype::HUFFMAN:
                // Huffman decompression would be implemented here
                throw std::runtime_error("Huffman decompression not fully implemented");
            case compresstype::RAW:
            default:
                return *this; // Already decompressed
        }
    }

    // Get compression ratio
    double getCompressionRatio() const {
        if (_data.empty() || _compressedData.empty()) return 0.0;
        return static_cast<double>(_data.size()) / _compressedData.size();
    }

    compresstype getCompressionType() const {
        return cformat;
    }

    const std::unordered_map<size_t, uint8_t>& getOverheadMap() const {
        return overheadmap;
    }

    bool isCompressed() const {
        return cformat != compresstype::RAW && !_compressedData.empty();
    }
};

#endif