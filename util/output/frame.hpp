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
        _compressedData.clear();
        _compressedData.shrink_to_fit();
    }

    const std::vector<uint8_t>& getData() const {
        return _data;
    }

    const std::vector<uint16_t>& getCompressedData() const {
        return _compressedData;
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
        ratio = compressedData.size() / _data.size();
        sourceSize = _data.size();
        _compressedData = std::move(compressedData);
        _data.clear();
        _data.shrink_to_fit();
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
        std::vector<uint8_t>::iterator dbegin = _data.begin();
        
        //try to optimize space usage without losing speed
        std::vector<std::vector<uint8_t>> matches128plus;
        std::vector<std::vector<uint8_t>> matches64plus;
        std::vector<std::vector<uint8_t>> matches32plus;
        std::vector<std::vector<uint8_t>> matchesAll;
        
        while (pos < dsize && matches128plus.size() < 65534) {
            size_t chunk_end = std::min(pos + chunksize, dsize);
            std::vector<uint8_t> chunk(dbegin + pos, dbegin + chunk_end);
            if (chunk.size() <= 4) { 
                pos = chunk_end;
                continue;
            }

            if (result.size() < 65534) {
                result.push_back(chunk);
            }

            std::vector<uint8_t> ffour;
            ffour.assign(chunk.begin(), chunk.begin() + 4);
            size_t searchpos = chunk_end;            
            while (searchpos + 4 <= dsize) {
                bool match_found = true;
                for (int i = 0; i < 4; ++i) {
                    if (_data[searchpos + i] != ffour[i]) {
                        match_found = false;
                        break;
                    }
                }
                
                if (match_found) {
                    size_t matchlength = 4;
                    size_t chunk_compare_pos = 4;
                    size_t input_compare_pos = searchpos + 4;

                    while (chunk_compare_pos < chunk.size() && input_compare_pos < dsize && _data[input_compare_pos] == chunk[chunk_compare_pos]) {
                        matchlength++;
                        chunk_compare_pos++;
                        input_compare_pos++;
                    }

                    std::vector<uint8_t> matchsequence(dbegin + searchpos, dbegin + searchpos + matchlength);
                    
                    // Categorize matches by length
                    if (matchlength >= 128) {
                        if (matches128plus.size() < 65534) {
                            matches128plus.push_back(matchsequence);
                        }
                    } else if (matchlength >= 64) {
                        if (matches64plus.size() < 65534) {
                            matches64plus.push_back(matchsequence);
                        }
                    } else if (matchlength >= 32) {
                        if (matches32plus.size() < 65534) {
                            matches32plus.push_back(matchsequence);
                        }
                    } else {
                        if (matchesAll.size() < 65534) {
                            matchesAll.push_back(matchsequence);
                        }
                    }
                    
                    searchpos += matchlength;
                } else {
                    searchpos++;
                }
            }
            
            pos = chunk_end;
        }
        for (const auto& match : matches128plus) {
            result.push_back(match);
        }
        
        // Then add 64+ matches if we still have space
        for (const auto& match : matches64plus) {
            if (result.size() < 65534) {
                result.push_back(match);
            } else {
                break;
            }
        }
        
        // Then add 32+ matches if we still have space
        for (const auto& match : matches32plus) {
            if (result.size() < 65534) {
                result.push_back(match);
            } else {
                break;
            }
        }
        
        // Finally add all other matches if we still have space
        for (const auto& match : matchesAll) {
            if (result.size() < 65534) {
                result.push_back(match);
            } else {
                break;
            }
        }
        
        return result;
    }

    std::vector<std::vector<uint8_t>> sortvecs(std::vector<std::vector<uint8_t>> source) {
        std::sort(source.begin(), source.end(), [](const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {return a.size() > b.size();});
        return source;
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
        
        std::vector<std::vector<uint8_t>> repeats = getRepeats();
        repeats = sortvecs(repeats);
        uint16_t nextDict = 1;

        std::vector<uint16_t> compressed;
        size_t cpos = 0;
        
        for (const auto& rseq : repeats) {
            if (!rseq.empty() && rseq.size() > 1 && overheadmap.size() < 65535) {
                overheadmap[nextDict] = rseq;
                nextDict++;
            }
        }

        while (cpos < _data.size()) {
            bool found_match = false;
            uint16_t best_dict_index = 0;
            size_t best_match_length = 0;
            
            // Iterate through dictionary in priority order (longest patterns first)
            for (uint16_t dict_idx = 1; dict_idx <= overheadmap.size(); dict_idx++) {
                const auto& dict_seq = overheadmap[dict_idx];
                
                // Quick length check - if remaining data is shorter than pattern, skip
                if (dict_seq.size() > (_data.size() - cpos)) {
                    continue;
                }
                
                // Check if this pattern matches at current position
                bool match = true;
                for (size_t i = 0; i < dict_seq.size(); ++i) {
                    if (_data[cpos + i] != dict_seq[i]) {
                        match = false;
                        break;
                    }
                }
                
                if (match) {
                    // Found a match - use it immediately (first match is best due to sorting)
                    best_dict_index = dict_idx;
                    best_match_length = dict_seq.size();
                    found_match = true;
                    break; // Stop searching - we found our match
                }
            }
            
            if (found_match && best_match_length > 1) {
                // Write dictionary reference
                compressed.push_back(best_dict_index);
                cpos += best_match_length;
            } else {
                // Write literal: 0 followed by the literal byte
                compressed.push_back(0);
                compressed.push_back(_data[cpos]);
                cpos++;
            }
        }

        ratio = compressed.size() / _data.size();
        sourceSize = _data.size();
        
        _compressedData = std::move(compressed);
        _compressedData.shrink_to_fit();
        
        // Clear uncompressed data
        _data.clear();
        _data.shrink_to_fit();
        
        cformat = compresstype::LZ78;

        return *this;
    }

    frame& decompressFrameLZ78() {
        TIME_FUNCTION;
        if (cformat != compresstype::LZ78) {
            throw std::runtime_error("Data is not LZ78 compressed");
        }
        
        std::vector<uint8_t> decompressedData;
        decompressedData.reserve(sourceSize);
        
        size_t cpos = 0;
        
        while (cpos < _compressedData.size()) {
            uint16_t token = _compressedData[cpos++];
            
            if (token == 0) {
                // Literal byte
                if (cpos < _compressedData.size()) {
                    decompressedData.push_back(static_cast<uint8_t>(_compressedData[cpos++]));
                }
            } else {
                // Dictionary reference
                auto it = overheadmap.find(token);
                if (it != overheadmap.end()) {
                    const std::vector<uint8_t>& dict_entry = it->second;
                    decompressedData.insert(decompressedData.end(), dict_entry.begin(), dict_entry.end());
                } else {
                    throw std::runtime_error("Invalid dictionary reference in compressed data");
                }
            }
        }
        
        _data = std::move(decompressedData);
        _compressedData.clear();
        _compressedData.shrink_to_fit();
        cformat = compresstype::RAW;
        
        return *this;
    }

    // Differential compression
    frame& compressFrameDiff() {
        // TODO
        throw std::logic_error("Function not yet implemented");
    }

    frame& decompressFrameDiff() {
        // TODO
        throw std::logic_error("Function not yet implemented");
    }
    // Huffman compression
    frame& compressFrameHuffman() {
        // TODO
        throw std::logic_error("Function not yet implemented");
    }

    // Combined compression methods
    frame& compressFrameZigZagRLE() {
        // TODO
        throw std::logic_error("Function not yet implemented");
    }

    frame& compressFrameDiffRLE() {
        // TODO
        throw std::logic_error("Function not yet implemented");
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
            case compresstype::LZ78:
                return decompressFrameLZ78();
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
        if (_compressedData.empty() || sourceSize == 0) return 0.0;
        return static_cast<double>(sourceSize) / _compressedData.size();
    }

    // Get source size (uncompressed size)
    size_t getSourceSize() const {
        return sourceSize;
    }

    // Get compressed size
    size_t getCompressedSize() const {
        return _compressedData.size();
    }

    // Print compression information
    void printCompressionInfo() const {
        std::cout << "Compression Type: ";
        switch (cformat) {
            case compresstype::RLE: std::cout << "RLE"; break;
            case compresstype::DIFF: std::cout << "DIFF"; break;
            case compresstype::DIFFRLE: std::cout << "DIFF + RLE"; break;
            case compresstype::LZ78: std::cout << "LZ78 (kinda)"; break;
            case compresstype::HUFFMAN: std::cout << "HUFFMAN"; break;
            case compresstype::RAW: std::cout << "RAW (uncompressed)"; break;
            default: std::cout << "UNKNOWN"; break;
        }
        std::cout << std::endl;
        
        std::cout << "Source Size: " << getSourceSize() << " bytes" << std::endl;
        std::cout << "Compressed Size: " << getCompressedSize() << " 16-bit words" << std::endl;
        std::cout << "Compressed Size: " << getCompressedSize() * 2 << " bytes" << std::endl;
        std::cout << "Compression Ratio: " << getCompressionRatio() << ":1" << std::endl;
        
        if (getCompressionRatio() > 1.0) {
            double savings = (1.0 - (1.0 / getCompressionRatio())) * 100.0;
            std::cout << "Space Savings: " << savings << "%" << std::endl;
        }
        
        // Show dictionary size for LZ78
        if (cformat == compresstype::LZ78) {
            std::cout << "Dictionary Size: " << overheadmap.size() << " entries" << std::endl;
        }
    }

    // Print compression information in a compact format
    void printCompressionStats() const {
        std::cout << "[" << getCompressionTypeString() << "] "
                  << getSourceSize() << "B -> " << getCompressedSize() * 2 << "B "
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

    const std::unordered_map<uint16_t, std::vector<uint8_t>>& getOverheadMap() const {
        return overheadmap;
    }

    bool isCompressed() const {
        return cformat != compresstype::RAW;
    }

    // Check if compressed data is available
    bool hasCompressedData() const {
        return !_compressedData.empty();
    }

    // Check if uncompressed data is available
    bool hasUncompressedData() const {
        return !_data.empty();
    }
};

#endif