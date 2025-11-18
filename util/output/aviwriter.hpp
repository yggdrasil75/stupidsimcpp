#ifndef AVI_WRITER_HPP
#define AVI_WRITER_HPP

#include <vector>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <tuple>
#include <iostream>
#include "frame.hpp"

class AVIWriter {
private:
    #pragma pack(push, 1)
    struct RIFFChunk {
        uint32_t chunkId;
        uint32_t chunkSize;
        uint32_t format;
    };

    struct AVIListHeader {
        uint32_t listId;
        uint32_t listSize;
        uint32_t listType;
    };

    struct AVIMainHeader {
        uint32_t microSecPerFrame;
        uint32_t maxBytesPerSec;
        uint32_t paddingGranularity;
        uint32_t flags;
        uint32_t totalFrames;
        uint32_t initialFrames;
        uint32_t streams;
        uint32_t suggestedBufferSize;
        uint32_t width;
        uint32_t height;
        uint32_t reserved[4];
    };

    struct AVIStreamHeader {
        uint32_t type;
        uint32_t handler;
        uint32_t flags;
        uint16_t priority;
        uint16_t language;
        uint32_t initialFrames;
        uint32_t scale;
        uint32_t rate;
        uint32_t start;
        uint32_t length;
        uint32_t suggestedBufferSize;
        uint32_t quality;
        uint32_t sampleSize;
        struct {
            int16_t left;
            int16_t top;
            int16_t right;
            int16_t bottom;
        } rcFrame;
    };

    struct BITMAPINFOHEADER {
        uint32_t size;
        int32_t width;
        int32_t height;
        uint16_t planes;
        uint16_t bitCount;
        uint32_t compression;
        uint32_t sizeImage;
        int32_t xPelsPerMeter;
        int32_t yPelsPerMeter;
        uint32_t clrUsed;
        uint32_t clrImportant;
    };

    struct AVIIndexEntry {
        uint32_t chunkId;
        uint32_t flags;
        uint32_t offset;
        uint32_t size;
    };
    #pragma pack(pop)

    static bool createDirectoryIfNeeded(const std::string& filename) {
        std::filesystem::path filePath(filename);
        std::filesystem::path directory = filePath.parent_path();
        
        if (!directory.empty() && !std::filesystem::exists(directory)) {
            return std::filesystem::create_directories(directory);
        }
        return true;
    }

    static void writeChunk(std::ofstream& file, uint32_t chunkId, const void* data, uint32_t size) {
        file.write(reinterpret_cast<const char*>(&chunkId), 4);
        file.write(reinterpret_cast<const char*>(&size), 4);
        if (data && size > 0) {
            file.write(reinterpret_cast<const char*>(data), size);
        }
    }

    static void writeList(std::ofstream& file, uint32_t listType, const void* data, uint32_t size) {
        uint32_t listId = 0x5453494C; // 'LIST'
        file.write(reinterpret_cast<const char*>(&listId), 4);
        file.write(reinterpret_cast<const char*>(&size), 4);
        file.write(reinterpret_cast<const char*>(&listType), 4);
        if (data && size > 4) {
            file.write(reinterpret_cast<const char*>(data), size - 4);
        }
    }

    static std::vector<uint8_t> prepareFrameData(const frame& frm, uint32_t width, uint32_t height, uint32_t rowSize) {
        std::vector<uint8_t> paddedFrame(rowSize * height, 0);
        
        // Get the frame data (decompress if necessary)
        std::vector<uint8_t> frameData;
        if (frm.isCompressed()) {
            // Create a copy and decompress
            frame tempFrame = frm;
            tempFrame.decompress();
            frameData = tempFrame.getData();
        } else {
            frameData = frm.getData();
        }
        
        if (frameData.empty()) {
            return paddedFrame;
        }
        
        // Determine source format and convert to RGB
        size_t srcChannels = 3; // Default
        switch (frm.colorFormat) {
            case frame::colormap::RGBA: srcChannels = 4; break;
            case frame::colormap::BGR: srcChannels = 3; break;
            case frame::colormap::BGRA: srcChannels = 4; break;
            case frame::colormap::B: srcChannels = 1; break;
            default: srcChannels = 3; break;
        }
        
        uint32_t srcRowSize = width * srcChannels;
        uint32_t dstRowSize = width * 3; // RGB
        
        // Convert and flip vertically for BMP format
        for (uint32_t y = 0; y < height; ++y) {
            uint32_t srcY = height - 1 - y; // Flip vertically
            const uint8_t* srcRow = frameData.data() + (srcY * srcRowSize);
            uint8_t* dstRow = paddedFrame.data() + (y * rowSize);
            
            // Convert to RGB format
            switch (frm.colorFormat) {
                case frame::colormap::RGB:
                    memcpy(dstRow, srcRow, dstRowSize);
                    break;
                case frame::colormap::RGBA:
                    for (uint32_t x = 0; x < width; ++x) {
                        dstRow[x * 3 + 2] = srcRow[x * 4 + 0];     // R
                        dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
                        dstRow[x * 3 + 0] = srcRow[x * 4 + 2]; // B
                    }
                    break;
                case frame::colormap::BGR:
                    for (uint32_t x = 0; x < width; ++x) {
                        dstRow[x * 3 + 2] = srcRow[x * 3 + 2];     // R
                        dstRow[x * 3 + 1] = srcRow[x * 3 + 1]; // G
                        dstRow[x * 3 + 0] = srcRow[x * 3 + 0];     // B
                    }
                    break;
                case frame::colormap::BGRA:
                    for (uint32_t x = 0; x < width; ++x) {
                        dstRow[x * 3 + 2] = srcRow[x * 4 + 2];     // R
                        dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
                        dstRow[x * 3 + 0] = srcRow[x * 4 + 0];     // B
                    }
                    break;
                case frame::colormap::B:
                    for (uint32_t x = 0; x < width; ++x) {
                        uint8_t gray = srcRow[x];
                        dstRow[x * 3 + 0] = gray; // R
                        dstRow[x * 3 + 1] = gray; // G
                        dstRow[x * 3 + 2] = gray; // B
                    }
                    break;
            }
        }
        
        return paddedFrame;
    }

    static std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> writeheader(int width, int height, float fps, std::ofstream& file, uint32_t frameCount, uint32_t microSecPerFrame) {
        // Calculate padding for each frame (BMP-style row padding)
        uint32_t rowSize = (width * 3 + 3) & ~3;
        uint32_t frameSize = rowSize * height;
        uint32_t totalDataSize = frameCount * frameSize;

        // RIFF AVI header
        RIFFChunk riffHeader;
        riffHeader.chunkId = 0x46464952; // 'RIFF'
        riffHeader.format = 0x20495641;  // 'AVI '
        
        // We'll come back and write the size at the end
        uint32_t riffStartPos = static_cast<uint32_t>(file.tellp());
        file.write(reinterpret_cast<const char*>(&riffHeader), sizeof(riffHeader));

        // hdrl list
        uint32_t hdrlListStart = static_cast<uint32_t>(file.tellp());
        writeList(file, 0x6C726468, nullptr, 0); // 'hdrl' - we'll fill size later

        // avih chunk
        AVIMainHeader mainHeader;
        mainHeader.microSecPerFrame = microSecPerFrame;
        mainHeader.maxBytesPerSec = frameSize * static_cast<uint32_t>(fps);
        mainHeader.paddingGranularity = 0;
        mainHeader.flags = 0x000010; // HASINDEX flag
        mainHeader.totalFrames = frameCount;
        mainHeader.initialFrames = 0;
        mainHeader.streams = 1;
        mainHeader.suggestedBufferSize = frameSize;
        mainHeader.width = width;
        mainHeader.height = height;
        mainHeader.reserved[0] = 0;
        mainHeader.reserved[1] = 0;
        mainHeader.reserved[2] = 0;
        mainHeader.reserved[3] = 0;
        
        writeChunk(file, 0x68697661, &mainHeader, sizeof(mainHeader)); // 'avih'

        // strl list
        uint32_t strlListStart = static_cast<uint32_t>(file.tellp());
        writeList(file, 0x6C727473, nullptr, 0); // 'strl' - we'll fill size later

        // strh chunk
        AVIStreamHeader streamHeader;
        streamHeader.type = 0x73646976; // 'vids'
        streamHeader.handler = 0x00000000; // Uncompressed
        streamHeader.flags = 0;
        streamHeader.priority = 0;
        streamHeader.language = 0;
        streamHeader.initialFrames = 0;
        streamHeader.scale = 1;
        streamHeader.rate = static_cast<uint32_t>(fps);
        streamHeader.start = 0;
        streamHeader.length = frameCount;
        streamHeader.suggestedBufferSize = frameSize;
        streamHeader.quality = 0xFFFFFFFF; // Default quality
        streamHeader.sampleSize = 0;
        streamHeader.rcFrame.left = 0;
        streamHeader.rcFrame.top = 0;
        streamHeader.rcFrame.right = width;
        streamHeader.rcFrame.bottom = height;
        
        writeChunk(file, 0x68727473, &streamHeader, sizeof(streamHeader)); // 'strh'

        // strf chunk
        BITMAPINFOHEADER bitmapInfo;
        bitmapInfo.size = sizeof(BITMAPINFOHEADER);
        bitmapInfo.width = width;
        bitmapInfo.height = height;
        bitmapInfo.planes = 1;
        bitmapInfo.bitCount = 24;
        bitmapInfo.compression = 0; // BI_RGB - uncompressed
        bitmapInfo.sizeImage = frameSize;
        bitmapInfo.xPelsPerMeter = 0;
        bitmapInfo.yPelsPerMeter = 0;
        bitmapInfo.clrUsed = 0;
        bitmapInfo.clrImportant = 0;
        
        writeChunk(file, 0x66727473, &bitmapInfo, sizeof(bitmapInfo)); // 'strf'

        // Update strl list size
        uint32_t strlListEnd = static_cast<uint32_t>(file.tellp());
        file.seekp(strlListStart + 4);
        uint32_t strlListSize = strlListEnd - strlListStart - 8;
        file.write(reinterpret_cast<const char*>(&strlListSize), 4);
        file.seekp(strlListEnd);

        // Update hdrl list size
        uint32_t hdrlListEnd = static_cast<uint32_t>(file.tellp());
        file.seekp(hdrlListStart + 4);
        uint32_t hdrlListSize = hdrlListEnd - hdrlListStart - 8;
        file.write(reinterpret_cast<const char*>(&hdrlListSize), 4);
        file.seekp(hdrlListEnd);

        // movi list
        uint32_t moviListStart = static_cast<uint32_t>(file.tellp());
        writeList(file, 0x69766F6D, nullptr, 0); // 'movi' - we'll fill size later

        return {moviListStart, frameSize, rowSize, riffStartPos};
    }

    static void writeFooter(std::ofstream& file, uint32_t moviListStart, uint32_t riffStartPos, std::vector<AVIWriter::AVIIndexEntry>& indexEntries) {
        // Update movi list size
        uint32_t moviListEnd = static_cast<uint32_t>(file.tellp());
        file.seekp(moviListStart + 4);
        uint32_t moviListSize = moviListEnd - moviListStart - 8;
        file.write(reinterpret_cast<const char*>(&moviListSize), 4);
        file.seekp(moviListEnd);

        // idx1 chunk - index
        uint32_t idx1Size = static_cast<uint32_t>(indexEntries.size() * sizeof(AVIIndexEntry));
        writeChunk(file, 0x31786469, indexEntries.data(), idx1Size); // 'idx1'

        // Update RIFF chunk size
        uint32_t fileEnd = static_cast<uint32_t>(file.tellp());
        file.seekp(riffStartPos + 4);
        uint32_t riffSize = fileEnd - riffStartPos - 8;
        file.write(reinterpret_cast<const char*>(&riffSize), 4);

    }

public:
    // Original method for vector of raw frame data
    static bool saveAVI(const std::string& filename, 
                       const std::vector<std::vector<uint8_t>>& frames, 
                       int width, int height, float fps = 30.0f) {
        TIME_FUNCTION;
        if (frames.empty() || width <= 0 || height <= 0 || fps <= 0) {
            return false;
        }
        
        // Validate frame sizes
        size_t expectedFrameSize = width * height * 3;
        for (const auto& frame : frames) {
            if (frame.size() != expectedFrameSize) {
                return false;
            }
        }

        // Create directory if needed
        createDirectoryIfNeeded(filename);

        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            return false;
        }

        uint32_t frameCount = static_cast<uint32_t>(frames.size());
        uint32_t microSecPerFrame = static_cast<uint32_t>(1000000.0f / fps);
        
        auto [moviListStart, frameSize, rowSize, riffStartPos] = writeheader(width, height, fps, file, frameCount, microSecPerFrame);

        std::vector<AVIIndexEntry> indexEntries;
        indexEntries.reserve(frameCount);
        // Write frames
        for (uint32_t i = 0; i < frameCount; ++i) {
            uint32_t frameStart = static_cast<uint32_t>(file.tellp()) - moviListStart - 4;
            
            // Create padded frame data (BMP-style bottom-to-top with padding)
            std::vector<uint8_t> paddedFrame(frameSize, 0);
            const auto& frame = frames[i];
            uint32_t srcRowSize = width * 3;
            
            for (int y = 0; y < height; ++y) {
                int srcY = height - 1 - y; // Flip vertically for BMP format
                const uint8_t* srcRow = frame.data() + (srcY * srcRowSize);
                uint8_t* dstRow = paddedFrame.data() + (y * rowSize);
                memcpy(dstRow, srcRow, srcRowSize);
                // Padding bytes remain zeros
            }
            
            // Write frame as '00db' chunk
            writeChunk(file, 0x62643030, paddedFrame.data(), frameSize); // '00db'
            
            // Add to index
            AVIIndexEntry entry;
            entry.chunkId = 0x62643030; // '00db'
            entry.flags = 0x00000010;   // AVIIF_KEYFRAME
            entry.offset = frameStart;
            entry.size = frameSize;
            indexEntries.push_back(entry);
        }

        writeFooter(file, moviListStart, riffStartPos, indexEntries);

        return true;
    }

    // New method for streaming decompression of frame objects
    static bool saveAVIFromCompressedFrames(const std::string& filename,
                                          std::vector<frame> frames,
                                          int width, int height, 
                                          float fps = 30.0f) {
        TIME_FUNCTION;
        if (frames.empty() || width <= 0 || height <= 0 || fps <= 0) {
            return false;
        }

        // Create directory if needed
        if (!createDirectoryIfNeeded(filename)) {
            return false;
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            return false;
        }

        uint32_t frameCount = static_cast<uint32_t>(frames.size());
        uint32_t microSecPerFrame = static_cast<uint32_t>(1000000.0f / fps);

        auto [moviListStart, frameSize, rowSize, riffStartPos] = writeheader(width, height, fps, file, frameCount, microSecPerFrame);

        std::vector<AVIIndexEntry> indexEntries;
        indexEntries.reserve(frameCount);

        // Write frames with streaming decompression
        while (frameCount > 0) {
            uint32_t frameStart = static_cast<uint32_t>(file.tellp()) - moviListStart - 4;
            
            // Prepare frame data (decompresses if necessary and converts to RGB)
            std::vector<uint8_t> paddedFrame = prepareFrameData(frames[0], width, height, rowSize);
            //frames[i].free();
            // Write frame as '00db' chunk
            writeChunk(file, 0x62643030, paddedFrame.data(), frameSize); // '00db'
            
            // Add to index
            AVIIndexEntry entry;
            entry.chunkId = 0x62643030; // '00db'
            entry.flags = 0x00000010;   // AVIIF_KEYFRAME
            entry.offset = frameStart;
            entry.size = frameSize;
            indexEntries.push_back(entry);
            paddedFrame.clear();
            frames.erase(frames.begin());
            paddedFrame.shrink_to_fit();
            frameCount = frames.size();
        }

        writeFooter(file, moviListStart, riffStartPos, indexEntries);

        return true;
    }

};

#endif