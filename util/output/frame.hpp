#ifndef FRAME_HPP
#define FRAME_HPP

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>

class frame {
private:
    std::vector<uint8_t> _data;
    std::unordered_map<size_t, uint8_t> overheadmap;
    size_t width;
    size_t height;
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
public:
    // to do: compress rle option, zigzag the frame and then rle, do a diff and then rle. should only support addition diff
    //convert to hex code instead and then run an option
    //decompress to return original data
    std::vector<uint8_t> compressFrameRLE() {
        if (cformat == compresstype::ZIGZAG){
            cformat = compresstype::ZIGZAGRLE;
        } else if (cformat == compresstype::DIFF){
            cformat = compresstype::DIFFRLE;
        } else {
            cformat = compresstype::RLE;
        }
        std::vector<uint8_t> compressed;
        if (_data.empty()) return compressed;
        size_t i = 0;

        while (i < _data.size()) {

        }
        
    }
    std::vector<uint8_t> compressFrameZigZag() {
        if (cformat != compresstype::RAW) {
            //FAIL
        }
        cformat = compresstype::ZIGZAG;
    }
    std::vector<uint8_t> compressFrameDiff() {
        if (cformat != compresstype::RAW) {
            //FAIL should decompress and recompress or just return false?
        }
        cformat = compresstype::DIFF;
    }

    std::vector<uint8_t> compressFrameHuffman() {

    }
    //get compression ratio
};

#endif