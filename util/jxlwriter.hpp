#ifndef JXL_WRITER_HPP
#define JXL_WRITER_HPP

#include <vector>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>
#include <filesystem>
#include "vec3.hpp"
#include "timing_decorator.hpp"
#include <jxl/encode.h>
#include <jxl/thread_parallel_runner.h>

class JXLWriter {
private:
    // Helper function to create directory if it doesn't exist
    static bool createDirectoryIfNeeded(const std::string& filename) {
        std::filesystem::path filePath(filename);
        std::filesystem::path directory = filePath.parent_path();
        
        // If there's a directory component and it doesn't exist, create it
        if (!directory.empty() && !std::filesystem::exists(directory)) {
            return std::filesystem::create_directories(directory);
        }
        return true;
    }

    // Helper function to convert Vec3 pixels to interleaved RGB data
    static std::vector<uint8_t> convertToRGB(const std::vector<std::vector<Vec3>>& pixels, int width, int height) {
        std::vector<uint8_t> rgbData(width * height * 3);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const Vec3& color = pixels[y][x];
                int offset = (y * width + x) * 3;
                
                rgbData[offset] = static_cast<uint8_t>(std::clamp(color.x * 255.0f, 0.0f, 255.0f));     // R
                rgbData[offset + 1] = static_cast<uint8_t>(std::clamp(color.y * 255.0f, 0.0f, 255.0f)); // G
                rgbData[offset + 2] = static_cast<uint8_t>(std::clamp(color.z * 255.0f, 0.0f, 255.0f)); // B
            }
        }
        
        return rgbData;
    }

    // Helper function to convert flat Vec3 vector to interleaved RGB data
    static std::vector<uint8_t> convertToRGB(const std::vector<Vec3>& pixels, int width, int height) {
        std::vector<uint8_t> rgbData(width * height * 3);
        
        for (int i = 0; i < width * height; ++i) {
            const Vec3& color = pixels[i];
            int offset = i * 3;
            
            rgbData[offset] = static_cast<uint8_t>(std::clamp(color.x * 255.0f, 0.0f, 255.0f));     // R
            rgbData[offset + 1] = static_cast<uint8_t>(std::clamp(color.y * 255.0f, 0.0f, 255.0f)); // G
            rgbData[offset + 2] = static_cast<uint8_t>(std::clamp(color.z * 255.0f, 0.0f, 255.0f)); // B
        }
        
        return rgbData;
    }

public:
    // Save a 2D vector of Vec3 (RGB) colors as JXL
    // Vec3 components: x = red, y = green, z = blue (values in range [0,1])
    static bool saveJXL(const std::string& filename, const std::vector<std::vector<Vec3>>& pixels,
                       float quality = 90.0f, int effort = 7) {
        if (pixels.empty() || pixels[0].empty()) {
            return false;
        }
        
        int height = static_cast<int>(pixels.size());
        int width = static_cast<int>(pixels[0].size());
        
        // Validate that all rows have the same width
        for (const auto& row : pixels) {
            if (row.size() != width) {
                return false;
            }
        }
        
        return saveJXL(filename, pixels, width, height, quality, effort);
    }
    
    static bool saveJXL(const std::string& filename, const std::vector<Vec3>& pixels, 
                       int width, int height, float quality = 90.0f, int effort = 7) {
        if (pixels.size() != width * height) {
            return false;
        }
        
        // Convert to 2D vector format
        std::vector<std::vector<Vec3>> pixels2D(height, std::vector<Vec3>(width));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                pixels2D[y][x] = pixels[y * width + x];
            }
        }
        
        return saveJXL(filename, pixels2D, width, height, quality, effort);
    }
    
    // Save from 1D vector of uint8_t pixels (RGB order: pixels[i]=r, pixels[i+1]=g, pixels[i+2]=b)
    static bool saveJXL(const std::string& filename, const std::vector<uint8_t>& pixels, 
                       int width, int height, float quality = 90.0f, int effort = 7) {
        TIME_FUNCTION;
        if (pixels.size() != width * height * 3) {
            return false;
        }
        
        // Create directory if needed
        if (!createDirectoryIfNeeded(filename)) {
            return false;
        }
        
        // Create encoder
        JxlEncoder* enc = JxlEncoderCreate(nullptr);
        if (!enc) {
            return false;
        }
        
        // Create thread pool (use nullptr for single-threaded)
        void* runner = JxlThreadParallelRunnerCreate(nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
        if (JxlEncoderSetParallelRunner(enc, JxlThreadParallelRunner, runner) != JXL_ENC_SUCCESS) {
            JxlThreadParallelRunnerDestroy(runner);
            JxlEncoderDestroy(enc);
            return false;
        }
        
        // Configure basic encoder settings
        JxlEncoderFrameSettings* options = JxlEncoderFrameSettingsCreate(enc, nullptr);
        
        // Set quality/distance (distance = 0 is lossless, higher = more lossy)
        float distance = (100.0f - quality) / 10.0f;  // Convert quality to distance
        if (quality >= 100.0f) {
            JxlEncoderSetFrameLossless(options, JXL_TRUE);
        } else {
            JxlEncoderSetFrameDistance(options, distance);
        }
        
        JxlEncoderFrameSettingsSetOption(options, JXL_ENC_FRAME_SETTING_EFFORT, effort);
        
        // Set up basic image info
        JxlBasicInfo basic_info;
        JxlEncoderInitBasicInfo(&basic_info);
        basic_info.xsize = width;
        basic_info.ysize = height;
        basic_info.bits_per_sample = 8;
        basic_info.exponent_bits_per_sample = 0;
        basic_info.uses_original_profile = JXL_FALSE;
        
        if (JxlEncoderSetBasicInfo(enc, &basic_info) != JXL_ENC_SUCCESS) {
            JxlThreadParallelRunnerDestroy(runner);
            JxlEncoderDestroy(enc);
            return false;
        }
        
        // Set color encoding to sRGB
        JxlColorEncoding color_encoding = {};
        JxlColorEncodingSetToSRGB(&color_encoding, JXL_FALSE); // is_gray = false
        if (JxlEncoderSetColorEncoding(enc, &color_encoding) != JXL_ENC_SUCCESS) {
            JxlThreadParallelRunnerDestroy(runner);
            JxlEncoderDestroy(enc);
            return false;
        }
        
        // Set up pixel format
        JxlPixelFormat pixel_format = {3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};
        
        // Add image frame
        if (JxlEncoderAddImageFrame(options, &pixel_format, 
                                   (void*)pixels.data(), 
                                   pixels.size()) != JXL_ENC_SUCCESS) {
            JxlThreadParallelRunnerDestroy(runner);
            JxlEncoderDestroy(enc);
            return false;
        }
        
        // Mark the end of input
        JxlEncoderCloseInput(enc);
        
        // Compress to buffer
        std::vector<uint8_t> compressed;
        compressed.resize(4096);  // Start with 4KB
        uint8_t* next_out = compressed.data();
        size_t avail_out = compressed.size();
        JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
        
        while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
            process_result = JxlEncoderProcessOutput(enc, &next_out, &avail_out);
            if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
                size_t offset = next_out - compressed.data();
                compressed.resize(compressed.size() * 2);
                next_out = compressed.data() + offset;
                avail_out = compressed.size() - offset;
            }
        }
        
        bool success = false;
        if (process_result == JXL_ENC_SUCCESS) {
            // Write to file
            size_t compressed_size = next_out - compressed.data();
            std::ofstream file(filename, std::ios::binary);
            if (file) {
                file.write(reinterpret_cast<const char*>(compressed.data()), compressed_size);
                success = file.good();
            }
        }
        
        // Cleanup
        JxlThreadParallelRunnerDestroy(runner);
        JxlEncoderDestroy(enc);
        
        return success;
    }
    
private:
    static bool saveJXL(const std::string& filename, const std::vector<std::vector<Vec3>>& pixels, 
                       int width, int height, float quality, int effort) {
        if (!createDirectoryIfNeeded(filename)) {
            return false;
        }
        
        std::vector<uint8_t> rgbData = convertToRGB(pixels, width, height);
        
        return saveJXL(filename, rgbData, width, height, quality, effort);
    }
};

#endif