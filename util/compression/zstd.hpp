#ifndef ZSTD_HPP
#define ZSTD_HPP

#include <cstdint>
#include <vector>
#include <cstring>
#include <memory>

// Simple ZSTD compression implementation (simplified version)
class ZSTD_Stream {
public:
    virtual ~ZSTD_Stream() = default;
};

class ZSTD_CompressStream : public ZSTD_Stream {
private:
    std::vector<uint8_t> buffer;
    size_t position;
    
public:
    ZSTD_CompressStream() : position(0) {
        buffer.reserve(1024 * 1024); // 1MB initial capacity
    }
    
    void reset() {
        buffer.clear();
        position = 0;
    }
    
    size_t compress(const void* src, size_t srcSize, void* dst, size_t dstCapacity) {
        // Simplified compression - in reality this would use actual ZSTD algorithm
        // For this example, we'll just copy with simple RLE-like compression
        
        if (dstCapacity < srcSize) {
            return 0; // Not enough space
        }
        
        const uint8_t* srcBytes = static_cast<const uint8_t*>(src);
        uint8_t* dstBytes = static_cast<uint8_t*>(dst);
        
        size_t dstPos = 0;
        size_t srcPos = 0;
        
        while (srcPos < srcSize && dstPos + 2 < dstCapacity) {
            // Simple RLE compression for repeated bytes
            uint8_t current = srcBytes[srcPos];
            size_t runLength = 1;
            
            while (srcPos + runLength < srcSize && 
                   srcBytes[srcPos + runLength] == current && 
                   runLength < 127) {
                runLength++;
            }
            
            if (runLength > 3) {
                // Encode as RLE
                dstBytes[dstPos++] = 0x80 | (runLength & 0x7F);
                dstBytes[dstPos++] = current;
                srcPos += runLength;
            } else {
                // Encode as literal run
                size_t literalStart = srcPos;
                while (srcPos < srcSize && 
                       (srcPos - literalStart < 127) &&
                       (srcPos + 1 >= srcSize || 
                        srcBytes[srcPos] != srcBytes[srcPos + 1] ||
                        srcPos + 2 >= srcSize || 
                        srcBytes[srcPos] != srcBytes[srcPos + 2])) {
                    srcPos++;
                }
                
                size_t literalLength = srcPos - literalStart;
                if (dstPos + literalLength + 1 > dstCapacity) {
                    break;
                }
                
                dstBytes[dstPos++] = literalLength & 0x7F;
                memcpy(dstBytes + dstPos, srcBytes + literalStart, literalLength);
                dstPos += literalLength;
            }
        }
        
        return dstPos;
    }
    
    size_t compress_continue(const void* src, size_t srcSize, void* dst, size_t dstCapacity) {
        return compress(src, srcSize, dst, dstCapacity);
    }
    
    const std::vector<uint8_t>& getBuffer() const { return buffer; }
};

class ZSTD_DecompressStream : public ZSTD_Stream {
private:
    size_t position;
    
public:
    ZSTD_DecompressStream() : position(0) {}
    
    void reset() {
        position = 0;
    }
    
    size_t decompress(const void* src, size_t srcSize, void* dst, size_t dstCapacity) {
        const uint8_t* srcBytes = static_cast<const uint8_t*>(src);
        uint8_t* dstBytes = static_cast<uint8_t*>(dst);
        
        size_t srcPos = 0;
        size_t dstPos = 0;
        
        while (srcPos < srcSize && dstPos < dstCapacity) {
            uint8_t header = srcBytes[srcPos++];
            
            if (header & 0x80) {
                // RLE encoded
                size_t runLength = header & 0x7F;
                if (srcPos >= srcSize) break;
                
                uint8_t value = srcBytes[srcPos++];
                
                size_t toCopy = std::min(runLength, dstCapacity - dstPos);
                memset(dstBytes + dstPos, value, toCopy);
                dstPos += toCopy;
                
                if (toCopy < runLength) {
                    break; // Output buffer full
                }
            } else {
                // Literal data
                size_t literalLength = header;
                if (srcPos + literalLength > srcSize) break;
                
                size_t toCopy = std::min(literalLength, dstCapacity - dstPos);
                memcpy(dstBytes + dstPos, srcBytes + srcPos, toCopy);
                srcPos += toCopy;
                dstPos += toCopy;
                
                if (toCopy < literalLength) {
                    break; // Output buffer full
                }
            }
        }
        
        return dstPos;
    }
    
    size_t decompress_continue(const void* src, size_t srcSize, void* dst, size_t dstCapacity) {
        return decompress(src, srcSize, dst, dstCapacity);
    }
};

// Type definitions for compatibility with original code
using ZSTD_stream_t = ZSTD_CompressStream;

// Stream creation functions
inline ZSTD_CompressStream* ZSTD_createStream() {
    return new ZSTD_CompressStream();
}

inline ZSTD_DecompressStream* ZSTD_createStreamDecode() {
    return new ZSTD_DecompressStream();
}

// Stream free functions
inline void ZSTD_freeStream(ZSTD_Stream* stream) {
    delete stream;
}

inline void ZSTD_freeStreamDecode(ZSTD_Stream* stream) {
    delete stream;
}

// Stream reset functions
inline void ZSTD_resetStream(ZSTD_CompressStream* stream) {
    if (stream) stream->reset();
}

inline void ZSTD_setStreamDecode(ZSTD_DecompressStream* stream, const void* dict, size_t dictSize) {
    if (stream) stream->reset();
    // Note: dict parameter is ignored in this simplified version
    (void)dict;
    (void)dictSize;
}

// Compression functions
inline size_t ZSTD_compressBound(size_t srcSize) {
    // Worst case: no compression + 1 byte header per 127 bytes
    return srcSize + (srcSize / 127) + 1;
}

inline size_t ZSTD_Compress_continue(ZSTD_CompressStream* stream, 
                                     const void* src, void* dst, 
                                     size_t srcSize) {
    if (!stream) return 0;
    return stream->compress_continue(src, srcSize, dst, ZSTD_compressBound(srcSize));
}

inline size_t ZSTD_Decompress_continue(ZSTD_DecompressStream* stream,
                                       const void* src, void* dst,
                                       size_t srcSize) {
    if (!stream) return 0;
    // Note: srcSize is actually the destination size in the original code
    // This is confusing but matches the usage in VoxelOctree
    return stream->decompress_continue(src, srcSize, dst, srcSize);
}

#endif // ZSTD_HPP