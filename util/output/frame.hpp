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
#include <future>
#include <mutex>
#include <atomic>

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
        overheadmap.clear();
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

    // Differential compression
    frame& compressFrameDiff() {
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

    // Calculate the size of the dictionary in bytes
    size_t getDictionarySize() const {
        size_t dictSize = 0;
        dictSize = sizeof(overheadmap);
        return dictSize;
    }

    // Get compressed size including dictionary overhead
    size_t getTotalCompressedSize() const {
        size_t baseSize = getCompressedDataSize();
        if (cformat == compresstype::LZ78) {
            baseSize += getDictionarySize();
        }
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
        return _compressedData.size();
    }

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
        if (cformat == compresstype::LZ78) {
            std::cout << "[" << getCompressionTypeString() << "] "
                      << "Source Size: " << getSourceSize() << " bytes"
                      << getTotalCompressedSize() << "B "
                      << "(ratio: " << getCompressionRatio() << ":1)" << std::endl;
        } else {
            std::cout << "[" << getCompressionTypeString() << "] "
                      << getSourceSize() << "B -> " << getTotalCompressedSize() << "B "
                      << "(ratio: " << getCompressionRatio() << ":1)" << std::endl;
        }
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

private:
    //moving decompression to private to prevent breaking stuff from external calls

    std::vector<std::vector<uint8_t>> sortvecs(std::vector<std::vector<uint8_t>> source) {
        std::sort(source.begin(), source.end(), [](const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {return a.size() > b.size();});
        return source;
    }
    
    frame& decompressFrameLZ78() {
        TIME_FUNCTION;
        if (cformat != compresstype::LZ78) {
            throw std::runtime_error("Data is not LZ78 compressed");
        }
        //std::cout << "why is this breaking? breakpoint f366" << std::endl;
        std::vector<uint8_t> decompressedData;
        decompressedData.reserve(sourceSize);
        
        size_t cpos = 0;
        
        while (cpos < _compressedData.size()) {
            uint16_t token = _compressedData[cpos++];
            //std::cout << "why is this breaking? breakpoint f374." << cpos << std::endl;
            if (token != 0) {
                // Dictionary reference
                auto it = overheadmap.find(token);
                if (it != overheadmap.end()) {
                    const std::vector<uint8_t>& dict_entry = it->second;
                    decompressedData.insert(decompressedData.end(), dict_entry.begin(), dict_entry.end());
                } else {
                    throw std::runtime_error("Invalid dictionary reference in compressed data");
                }
            } else {
                // Literal byte
                if (cpos < _compressedData.size()) {
                    decompressedData.push_back(static_cast<uint8_t>(_compressedData[cpos++]));
                }
            }
        }
        
        _data = std::move(decompressedData);
        _compressedData.clear();
        _compressedData.shrink_to_fit();
        overheadmap.clear();
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
        // TODO
        throw std::logic_error("Function not yet implemented");
    }

};

#endif