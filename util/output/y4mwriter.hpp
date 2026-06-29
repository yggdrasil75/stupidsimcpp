#ifndef Y4M_WRITER_HPP
#define Y4M_WRITER_HPP

#include "frame.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

class y4mWriter {
public:
    static void save(const std::string& filename, std::vector<frame>& frames, int fps = 30) {
        if (frames.empty()) {
            throw std::invalid_argument("Cannot save video: Frame vector is empty.");
        }

        size_t width = frames[0].getWidth();
        size_t height = frames[0].getHeight();

        if (width == 0 || height == 0) {
            throw std::invalid_argument("Cannot save video: Frames have zero dimensions.");
        }

        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        out << "YUV4MPEG2 W" << width << " H" << height << " F" << fps << ":1 Ip A1:1 C444\n";

        size_t numPixels = width * height;
        std::vector<uint8_t> planeY(numPixels);
        std::vector<uint8_t> planeU(numPixels);
        std::vector<uint8_t> planeV(numPixels);

        for (size_t f = 0; f < frames.size(); ++f) {
            if (frames[f].getWidth() != width || frames[f].getHeight() != height) {
                throw std::runtime_error("All frames must have the exact same dimensions for Y4M export.");
            }

            bool wasCompressed = frames[f].isCompressed();
            frame::compresstype originalCType = frames[f].getCompressionType();
            
            if (wasCompressed) {
                frames[f].decompress();
            }

            const std::vector<uint8_t>& data = frames[f].getData();
            frame::colormap format = frames[f].colorFormat;
            
            size_t channels = 3;
            if (format == frame::colormap::RGBA || format == frame::colormap::BGRA) channels = 4;
            if (format == frame::colormap::B) channels = 1;

            for (size_t i = 0; i < numPixels; ++i) {
                size_t px = i * channels;
                uint8_t r = 0, g = 0, b = 0;

                if (format == frame::colormap::RGB || format == frame::colormap::RGBA) {
                    r = data[px];
                    g = data[px+1];
                    b = data[px+2];
                } else if (format == frame::colormap::BGR || format == frame::colormap::BGRA) {
                    b = data[px];
                    g = data[px+1];
                    r = data[px+2];
                } else if (format == frame::colormap::B) {
                    r = g = b = data[px];
                }

                int Y = (19595 * r + 38470 * g +  7471 * b) >> 16;
                int U = -11059 * r - 21709 * g + 32768 * b + 8388608;
                int V =  32768 * r - 27439 * g -  5329 * b + 8388608;

                planeY[i] = static_cast<uint8_t>(std::clamp(Y, 0, 255));
                planeU[i] = static_cast<uint8_t>(std::clamp(U >> 16, 0, 255));
                planeV[i] = static_cast<uint8_t>(std::clamp(V >> 16, 0, 255));
            }

            out << "FRAME\n";
            out.write(reinterpret_cast<const char*>(planeY.data()), planeY.size());
            out.write(reinterpret_cast<const char*>(planeU.data()), planeU.size());
            out.write(reinterpret_cast<const char*>(planeV.data()), planeV.size());

            if (wasCompressed) {
                switch(originalCType) {
                    case frame::compresstype::LZ78:
                        frames[f].compressFrameLZ78();
                        break;
                    case frame::compresstype::RLE:
                        frames[f].compressFrameRLE();
                        break;
                    default: break;
                }
            }
        }
        
        out.close();
        std::cout << "Successfully exported " << frames.size() << " frames to " << filename << std::endl;
    }
};

#endif