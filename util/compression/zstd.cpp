#include "zstd.hpp"
#include <cstdint>
#include <vector>
#include <cstring>
#include <memory>
#include <algorithm>

// Implementation of ZSTD_CompressStream methods
size_t ZSTD_CompressStream::compress_continue(const void* src, void* dst, size_t srcSize) {
    // For compatibility with the original interface where dst size is inferred
    size_t dstCapacity = ZSTD_compressBound(srcSize);
    return this->compress(src, srcSize, dst, dstCapacity);
}

// Implementation of ZSTD_DecompressStream methods  
size_t ZSTD_DecompressStream::decompress_continue(const void* src, void* dst, size_t srcSize) {
    // Note: srcSize parameter is actually the destination buffer size
    // This matches the confusing usage in the original VoxelOctree code
    return this->decompress(src, srcSize, dst, srcSize);
}

// Helper functions for the compression system
namespace {
    // Simple hash function for LZ77-style matching
    uint32_t computeHash(const uint8_t* data) {
        return (data[0] << 16) | (data[1] << 8) | data[2];
    }
}

// More advanced compression implementation
size_t ZSTD_CompressStream::compress(const void* src, size_t srcSize, void* dst, size_t dstCapacity) {
    if (srcSize == 0 || dstCapacity == 0) return 0;
    
    const uint8_t* srcBytes = static_cast<const uint8_t*>(src);
    uint8_t* dstBytes = static_cast<uint8_t*>(dst);
    
    size_t dstPos = 0;
    size_t srcPos = 0;
    
    // Simple LZ77 compression
    while (srcPos < srcSize && dstPos + 3 < dstCapacity) {
        // Try to find a match in previous data
        size_t bestMatchPos = 0;
        size_t bestMatchLen = 0;
        
        // Look for matches in recent data (simplified search window)
        size_t searchStart = (srcPos > 1024) ? srcPos - 1024 : 0;
        for (size_t i = searchStart; i < srcPos && i + 3 <= srcSize; i++) {
            if (srcBytes[i] == srcBytes[srcPos]) {
                size_t matchLen = 1;
                while (srcPos + matchLen < srcSize && 
                       i + matchLen < srcPos &&
                       srcBytes[i + matchLen] == srcBytes[srcPos + matchLen] &&
                       matchLen < 255) {
                    matchLen++;
                }
                
                if (matchLen > bestMatchLen && matchLen >= 4) {
                    bestMatchLen = matchLen;
                    bestMatchPos = srcPos - i;
                }
            }
        }
        
        if (bestMatchLen >= 4) {
            // Encode match
            dstBytes[dstPos++] = 0x80 | ((bestMatchLen - 4) & 0x7F);
            dstBytes[dstPos++] = (bestMatchPos >> 8) & 0xFF;
            dstBytes[dstPos++] = bestMatchPos & 0xFF;
            srcPos += bestMatchLen;
        } else {
            // Encode literals
            size_t literalStart = srcPos;
            size_t maxLiteral = std::min(srcSize - srcPos, size_t(127));
            
            // Find run of non-compressible data
            while (srcPos - literalStart < maxLiteral) {
                // Check if next few bytes could be compressed
                bool canCompress = false;
                if (srcPos + 3 < srcSize) {
                    // Simple heuristic: repeated bytes or patterns
                    if (srcBytes[srcPos] == srcBytes[srcPos + 1] && 
                        srcBytes[srcPos] == srcBytes[srcPos + 2]) {
                        canCompress = true;
                    }
                }
                if (canCompress) break;
                srcPos++;
            }
            
            size_t literalLength = srcPos - literalStart;
            if (literalLength > 0) {
                if (dstPos + literalLength + 1 > dstCapacity) {
                    // Not enough space
                    break;
                }
                
                dstBytes[dstPos++] = literalLength & 0x7F;
                memcpy(dstBytes + dstPos, srcBytes + literalStart, literalLength);
                dstPos += literalLength;
            }
        }
    }
    
    return dstPos;
}

size_t ZSTD_DecompressStream::decompress(const void* src, size_t srcSize, void* dst, size_t dstCapacity) {
    if (srcSize == 0 || dstCapacity == 0) return 0;
    
    const uint8_t* srcBytes = static_cast<const uint8_t*>(src);
    uint8_t* dstBytes = static_cast<uint8_t*>(dst);
    
    size_t srcPos = 0;
    size_t dstPos = 0;
    
    while (srcPos < srcSize && dstPos < dstCapacity) {
        uint8_t header = srcBytes[srcPos++];
        
        if (header & 0x80) {
            // Match
            if (srcPos + 1 >= srcSize) break;
            
            size_t matchLen = (header & 0x7F) + 4;
            uint16_t matchDist = (srcBytes[srcPos] << 8) | srcBytes[srcPos + 1];
            srcPos += 2;
            
            if (matchDist == 0 || matchDist > dstPos) {
                // Invalid match distance
                break;
            }
            
            size_t toCopy = std::min(matchLen, dstCapacity - dstPos);
            size_t matchPos = dstPos - matchDist;
            
            for (size_t i = 0; i < toCopy; i++) {
                dstBytes[dstPos++] = dstBytes[matchPos + i];
            }
            
            if (toCopy < matchLen) {
                break; // Output buffer full
            }
        } else {
            // Literals
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