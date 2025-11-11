#ifndef AVI_WRITER_HPP
#define AVI_WRITER_HPP

#include <vector>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>
#include <filesystem>
#include <chrono>

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

public:
    static bool saveAVI(const std::string& filename, 
                       const std::vector<std::vector<uint8_t>>& frames, 
                       int width, int height, float fps = 30.0f) {
        if (frames.empty() || width <= 0 || height <= 0 || fps <= 0) {
            return false;
        }
        
        std::cout << "1" << "width: " << width <<
            "height: " << height << "frame count: " << fps << std::endl;

        // Validate frame sizes
        size_t expectedFrameSize = width * height * 3;
        for (const auto& frame : frames) {
            if (frame.size() != expectedFrameSize) {
                return false;
            }
        }

        std::cout << "2" << std::endl;
        // Create directory if needed
        if (!createDirectoryIfNeeded(filename)) {
            return false;
        }

        std::cout << "3" << std::endl;
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            return false;
        }

        uint32_t frameCount = static_cast<uint32_t>(frames.size());
        uint32_t microSecPerFrame = static_cast<uint32_t>(1000000.0f / fps);
        
        // Calculate padding for each frame (BMP-style row padding)
        uint32_t rowSize = (width * 3 + 3) & ~3;
        uint32_t frameSize = rowSize * height;
        uint32_t totalDataSize = frameCount * frameSize;

        std::cout << "4" << std::endl;
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

        std::cout << "5" << std::endl;
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

        std::cout << "6" << std::endl;
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

        std::cout << "7" << std::endl;
        // Update strl list size
        uint32_t strlListEnd = static_cast<uint32_t>(file.tellp());
        file.seekp(strlListStart + 4);
        uint32_t strlListSize = strlListEnd - strlListStart - 8;
        file.write(reinterpret_cast<const char*>(&strlListSize), 4);
        file.seekp(strlListEnd);

        std::cout << "8" << std::endl;
        // Update hdrl list size
        uint32_t hdrlListEnd = static_cast<uint32_t>(file.tellp());
        file.seekp(hdrlListStart + 4);
        uint32_t hdrlListSize = hdrlListEnd - hdrlListStart - 8;
        file.write(reinterpret_cast<const char*>(&hdrlListSize), 4);
        file.seekp(hdrlListEnd);

        std::cout << "9" << std::endl;
        // movi list
        uint32_t moviListStart = static_cast<uint32_t>(file.tellp());
        writeList(file, 0x69766F6D, nullptr, 0); // 'movi' - we'll fill size later

        std::vector<AVIIndexEntry> indexEntries;
        indexEntries.reserve(frameCount);

        // Write frames
        for (uint32_t i = 0; i < frameCount; ++i) {
            uint32_t frameStart = static_cast<uint32_t>(file.tellp()) - moviListStart - 4;
            
            std::cout << "10-" << i << std::endl;
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
            
            std::cout << "11-" << i << std::endl;
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

        std::cout << "12" << std::endl;
        // Update movi list size
        uint32_t moviListEnd = static_cast<uint32_t>(file.tellp());
        file.seekp(moviListStart + 4);
        uint32_t moviListSize = moviListEnd - moviListStart - 8;
        file.write(reinterpret_cast<const char*>(&moviListSize), 4);
        file.seekp(moviListEnd);

        std::cout << "13" << std::endl;
        // idx1 chunk - index
        uint32_t idx1Size = static_cast<uint32_t>(indexEntries.size() * sizeof(AVIIndexEntry));
        writeChunk(file, 0x31786469, indexEntries.data(), idx1Size); // 'idx1'

        // Update RIFF chunk size
        uint32_t fileEnd = static_cast<uint32_t>(file.tellp());
        file.seekp(riffStartPos + 4);
        uint32_t riffSize = fileEnd - riffStartPos - 8;
        file.write(reinterpret_cast<const char*>(&riffSize), 4);

        std::cout << "14" << std::endl;
        return true;
    }

    // Convenience function to save from individual frame files
    static bool saveAVIFromFrames(const std::string& filename,
                                 const std::vector<std::string>& frameFiles,
                                 int width, int height,
                                 float fps = 30.0f) {
        std::vector<std::vector<uint8_t>> frames;
        frames.reserve(frameFiles.size());

        for (const auto& frameFile : frameFiles) {
            std::ifstream file(frameFile, std::ios::binary);
            if (!file) {
                return false;
            }

            // Read BMP file and extract pixel data
            file.seekg(0, std::ios::end);
            size_t fileSize = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> buffer(fileSize);
            file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

            // Simple BMP parsing - assumes 24-bit uncompressed BMP
            if (fileSize < 54 || buffer[0] != 'B' || buffer[1] != 'M') {
                return false;
            }

            // Extract pixel data offset from BMP header
            uint32_t dataOffset = *reinterpret_cast<uint32_t*>(&buffer[10]);
            if (dataOffset >= fileSize) {
                return false;
            }

            // Read pixel data (BGR format)
            std::vector<uint8_t> pixelData(buffer.begin() + dataOffset, buffer.end());
            frames.push_back(pixelData);
        }

        return saveAVI(filename, frames, width, height, fps);
    }
};

#endif