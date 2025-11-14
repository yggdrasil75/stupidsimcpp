#ifndef VIDEO_HPP
#define VIDEO_HPP

#include "frame.hpp"
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <iostream>
#include "../timing_decorator.hpp"

class video {
private:
    std::vector<std::vector<std::pair<uint8_t, uint32_t>>> compressed_frames_;
    size_t width_;
    size_t height_;
    std::vector<char> channels_;
    double fps_;
    bool use_differential_encoding_;

    // Compress frame using differential encoding
    std::vector<std::pair<uint8_t, uint32_t>> compress_with_differential(
        const frame& current_frame, const frame* previous_frame = nullptr) const {
        TIME_FUNCTION;
        
        if (previous_frame == nullptr) {
            // First frame - compress normally
            return current_frame.compress_rle();
        }
        
        // Create differential frame
        std::vector<uint8_t> diff_data(current_frame.size());
        
        const std::vector<uint8_t>& current_data = current_frame.data();
        const std::vector<uint8_t>& prev_data = previous_frame->data();
        
        // Calculate difference between frames
        for (size_t i = 0; i < current_data.size(); ++i) {
            // Use modulo arithmetic to handle unsigned byte overflow
            diff_data[i] = (current_data[i] - prev_data[i]) & 0xFF;
        }
        
        // Create temporary frame for differential data
        frame diff_frame(diff_data, width_, height_, channels_);
        
        // Compress the differential data
        return diff_frame.compress_rle();
    }
    
    // Decompress differential frame
    frame decompress_differential(const std::vector<std::pair<uint8_t, uint32_t>>& compressed_diff,
                                 const frame& previous_frame) const {
        TIME_FUNCTION;
        
        frame diff_frame;
        diff_frame.decompress_rle(compressed_diff);
        
        // Reconstruct original frame from differential
        std::vector<uint8_t> reconstructed_data(diff_frame.size());
        const std::vector<uint8_t>& diff_data = diff_frame.data();
        const std::vector<uint8_t>& prev_data = previous_frame.data();
        
        for (size_t i = 0; i < diff_data.size(); ++i) {
            // Reverse the differential encoding
            reconstructed_data[i] = (prev_data[i] + diff_data[i]) & 0xFF;
        }
        
        return frame(reconstructed_data, width_, height_, channels_);
    }

public:
    // Default constructor
    video() : width_(0), height_(0), fps_(30.0), use_differential_encoding_(true) {}
    
    // Constructor with dimensions and settings
    video(size_t width, size_t height, const std::vector<char>& channels = {'\0'},
          double fps = 30.0, bool use_differential = true)
        : width_(width), height_(height), channels_(channels), fps_(fps),
          use_differential_encoding_(use_differential) {
        
        if (width == 0 || height == 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
        if (channels.empty()) {
            throw std::invalid_argument("Channels list cannot be empty");
        }
        if (fps <= 0) {
            throw std::invalid_argument("FPS must be positive");
        }
    }
    
    // Constructor with initializer list for channels
    video(size_t width, size_t height, std::initializer_list<char> channels,
          double fps = 30.0, bool use_differential = true)
        : video(width, height, std::vector<char>(channels), fps, use_differential) {}
    
    // Accessors
    size_t width() const noexcept { return width_; }
    size_t height() const noexcept { return height_; }
    const std::vector<char>& channels() const noexcept { return channels_; }
    double fps() const noexcept { return fps_; }
    bool use_differential_encoding() const noexcept { return use_differential_encoding_; }
    size_t frame_count() const noexcept { return compressed_frames_.size(); }
    size_t channels_count() const noexcept { return channels_.size(); }
    
    // Check if video is empty
    bool empty() const noexcept {
        return compressed_frames_.empty() || width_ == 0 || height_ == 0;
    }
    
    // Add a frame to the video sequence
    void add_frame(const frame& new_frame) {
        TIME_FUNCTION;
        // Validate frame dimensions and channels
        if (new_frame.width() != width_ || new_frame.height() != height_) {
            throw std::invalid_argument("Frame dimensions must match video dimensions");
        }
        if (new_frame.channels() != channels_) {
            throw std::invalid_argument("Frame channels must match video channels");
        }
        
        if (compressed_frames_.empty() || !use_differential_encoding_) {
            // First frame or differential encoding disabled - compress normally
            compressed_frames_.push_back(new_frame.compress_rle());
        } else {
            // Get the previous frame for differential encoding
            frame prev_frame = get_frame(frame_count() - 1);
            compressed_frames_.push_back(compress_with_differential(new_frame, &prev_frame));
        }
    }
    
    // Add frame with move semantics
    void add_frame(frame&& new_frame) {
        add_frame(new_frame); // Just call the const version
    }
    
    // Get a specific frame
    frame get_frame(size_t index) const {
        if (index >= compressed_frames_.size()) {
            throw std::out_of_range("Frame index out of range");
        }
        
        if (index == 0 || !use_differential_encoding_) {
            // First frame or no differential encoding - decompress normally
            frame result;
            result.decompress_rle(compressed_frames_[index]);
            
            // Set dimensions and channels
            result.resize(width_, height_, channels_);
            return result;
        } else {
            // Differential encoded frame - need previous frame to reconstruct
            frame prev_frame = get_frame(index - 1);
            return decompress_differential(compressed_frames_[index], prev_frame);
        }
    }
    
    // Get multiple frames as a sequence
    std::vector<frame> get_frames(size_t start_index, size_t count) const {
        TIME_FUNCTION;
        if (start_index >= compressed_frames_.size()) {
            throw std::out_of_range("Start index out of range");
        }
        
        count = std::min(count, compressed_frames_.size() - start_index);
        std::vector<frame> frames;
        frames.reserve(count);
        
        for (size_t i = start_index; i < start_index + count; ++i) {
            frames.push_back(get_frame(i));
        }
        
        return frames;
    }
    
    // Get all frames
    std::vector<frame> get_all_frames() const {
        return get_frames(0, compressed_frames_.size());
    }
    
    // Remove a frame
    void remove_frame(size_t index) {
        if (index >= compressed_frames_.size()) {
            throw std::out_of_range("Frame index out of range");
        }
        compressed_frames_.erase(compressed_frames_.begin() + index);
    }
    
    // Clear all frames
    void clear_frames() noexcept {
        compressed_frames_.clear();
    }
    
    // Replace a frame
    void replace_frame(size_t index, const frame& new_frame) {
        TIME_FUNCTION;
        if (index >= compressed_frames_.size()) {
            throw std::out_of_range("Frame index out of range");
        }
        
        // Validate frame dimensions and channels
        if (new_frame.width() != width_ || new_frame.height() != height_) {
            throw std::invalid_argument("Frame dimensions must match video dimensions");
        }
        if (new_frame.channels() != channels_) {
            throw std::invalid_argument("Frame channels must match video channels");
        }
        
        if (index == 0 || !use_differential_encoding_) {
            compressed_frames_[index] = new_frame.compress_rle();
        } else {
            frame prev_frame = get_frame(index - 1);
            compressed_frames_[index] = compress_with_differential(new_frame, &prev_frame);
        }
        
        // If this isn't the last frame, we need to update the next frame's differential encoding
        if (use_differential_encoding_ && index + 1 < compressed_frames_.size()) {
            frame current_frame = get_frame(index);
            frame next_frame_original = get_frame(index + 1);
            compressed_frames_[index + 1] = compress_with_differential(next_frame_original, &current_frame);
        }
    }
    
    // Set FPS
    void set_fps(double fps) {
        if (fps <= 0) {
            throw std::invalid_argument("FPS must be positive");
        }
        fps_ = fps;
    }
    
    // Enable/disable differential encoding
    void set_differential_encoding(bool enabled) {
        TIME_FUNCTION;
        if (use_differential_encoding_ == enabled) {
            return; // No change needed
        }
        
        if (!compressed_frames_.empty() && enabled != use_differential_encoding_) {
            // Need to recompress all frames with new encoding setting
            auto original_frames = get_all_frames();
            clear_frames();
            use_differential_encoding_ = enabled;
            
            for (const auto& f : original_frames) {
                add_frame(f);
            }
        } else {
            use_differential_encoding_ = enabled;
        }
    }
    
    // Get video duration in seconds
    double duration() const noexcept {
        TIME_FUNCTION;
        return compressed_frames_.size() / fps_;
    }
    
    // Calculate total compressed size in bytes
    size_t total_compressed_size() const noexcept {
        TIME_FUNCTION;
        size_t total = 0;
        for (const auto& compressed_frame : compressed_frames_) {
            total += compressed_frame.size() * sizeof(std::pair<uint8_t, uint32_t>);
        }
        return total;
    }
    
    // Calculate total uncompressed size in bytes
    size_t total_uncompressed_size() const noexcept {
        TIME_FUNCTION;
        return compressed_frames_.size() * width_ * height_ * channels_.size();
    }
    
    // Calculate overall compression ratio
    double overall_compression_ratio() const noexcept {
        TIME_FUNCTION;
        if (empty()) {
            return 1.0;
        }
        size_t uncompressed = total_uncompressed_size();
        if (uncompressed == 0) {
            return 1.0;
        }
        return static_cast<double>(uncompressed) / total_compressed_size();
    }
    
    // Calculate average frame compression ratio
    double average_frame_compression_ratio() const {
        TIME_FUNCTION;
        if (empty()) {
            return 1.0;
        }
        
        double total_ratio = 0.0;
        for (size_t i = 0; i < compressed_frames_.size(); ++i) {
            frame f = get_frame(i);
            total_ratio += f.get_compression_ratio();
        }
        
        return total_ratio / compressed_frames_.size();
    }
    
    // Get compression statistics
    struct compression_stats {
        size_t total_frames;
        size_t total_compressed_bytes;
        size_t total_uncompressed_bytes;
        double overall_ratio;
        double average_frame_ratio;
        double video_duration;
    };
    
    compression_stats get_compression_stats() const {
        TIME_FUNCTION;
        compression_stats stats;
        stats.total_frames = compressed_frames_.size();
        stats.total_compressed_bytes = total_compressed_size();
        stats.total_uncompressed_bytes = total_uncompressed_size();
        stats.overall_ratio = overall_compression_ratio();
        stats.average_frame_ratio = average_frame_compression_ratio();
        stats.video_duration = duration();
        return stats;
    }
    
    // Extract a sub-video
    video subvideo(size_t start_frame, size_t frame_count) const {
        TIME_FUNCTION;
        if (start_frame >= compressed_frames_.size()) {
            throw std::out_of_range("Start frame out of range");
        }
        
        frame_count = std::min(frame_count, compressed_frames_.size() - start_frame);
        video result(width_, height_, channels_, fps_, use_differential_encoding_);
        
        for (size_t i = start_frame; i < start_frame + frame_count; ++i) {
            result.compressed_frames_.push_back(compressed_frames_[i]);
        }
        
        return result;
    }
    
    // Append another video (must have same dimensions and channels)
    void append_video(const video& other) {
        TIME_FUNCTION;
        if (other.width_ != width_ || other.height_ != height_ || other.channels_ != channels_) {
            throw std::invalid_argument("Videos must have same dimensions and channels");
        }
        
        // If both use differential encoding, we can directly append compressed frames
        if (use_differential_encoding_ && other.use_differential_encoding_) {
            compressed_frames_.insert(compressed_frames_.end(),
                                    other.compressed_frames_.begin(),
                                    other.compressed_frames_.end());
        } else {
            // Otherwise, we need to decompress and recompress
            auto other_frames = other.get_all_frames();
            for (const auto& frame : other_frames) {
                add_frame(frame);
            }
        }
    }
    
    // Save/Load functionality (basic serialization)
    std::vector<uint8_t> serialize() const {
        TIME_FUNCTION;
        // Simple serialization format:
        // [header][compressed_frame_data...]
        // Header: width(4), height(4), channels_count(1), channels_data(n), fps(8), frame_count(4)
        
        std::vector<uint8_t> result;
        
        // Header
        auto add_uint32 = [&result](uint32_t value) {
            for (int i = 0; i < 4; ++i) {
                result.push_back((value >> (i * 8)) & 0xFF);
            }
        };
        
        auto add_double = [&result](double value) {
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
            for (size_t i = 0; i < sizeof(double); ++i) {
                result.push_back(bytes[i]);
            }
        };
        
        // Write header
        add_uint32(static_cast<uint32_t>(width_));
        add_uint32(static_cast<uint32_t>(height_));
        result.push_back(static_cast<uint8_t>(channels_.size()));
        for (char c : channels_) {
            result.push_back(static_cast<uint8_t>(c));
        }
        add_double(fps_);
        result.push_back(use_differential_encoding_ ? 1 : 0);
        add_uint32(static_cast<uint32_t>(compressed_frames_.size()));
        
        // Write compressed frames
        for (const auto& compressed_frame : compressed_frames_) {
            add_uint32(static_cast<uint32_t>(compressed_frame.size()));
            for (const auto& run : compressed_frame) {
                result.push_back(run.first);
                add_uint32(run.second);
            }
        }
        
        return result;
    }
    
    // Deserialize from byte data
    static video deserialize(const std::vector<uint8_t>& data) {
        TIME_FUNCTION;
        if (data.size() < 4 + 4 + 1 + 8 + 1 + 4) { // Minimum header size
            throw std::invalid_argument("Invalid video data: too short");
        }
        
        size_t pos = 0;
        auto read_uint32 = [&data, &pos]() {
            if (pos + 4 > data.size()) throw std::invalid_argument("Unexpected end of data");
            uint32_t value = 0;
            for (int i = 0; i < 4; ++i) {
                value |= static_cast<uint32_t>(data[pos++]) << (i * 8);
            }
            return value;
        };
        
        auto read_double = [&data, &pos]() {
            if (pos + sizeof(double) > data.size()) throw std::invalid_argument("Unexpected end of data");
            double value;
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
            for (size_t i = 0; i < sizeof(double); ++i) {
                bytes[i] = data[pos++];
            }
            return value;
        };
        
        // Read header
        uint32_t width = read_uint32();
        uint32_t height = read_uint32();
        uint8_t channels_count = data[pos++];
        
        std::vector<char> channels;
        for (uint8_t i = 0; i < channels_count; ++i) {
            if (pos >= data.size()) throw std::invalid_argument("Unexpected end of data");
            channels.push_back(static_cast<char>(data[pos++]));
        }
        
        double fps = read_double();
        bool use_diff = data[pos++] != 0;
        uint32_t frame_count = read_uint32();
        
        video result(width, height, channels, fps, use_diff);
        
        // Read compressed frames
        for (uint32_t i = 0; i < frame_count; ++i) {
            if (pos + 4 > data.size()) throw std::invalid_argument("Unexpected end of data");
            uint32_t runs_count = read_uint32();
            
            std::vector<std::pair<uint8_t, uint32_t>> compressed_frame;
            for (uint32_t j = 0; j < runs_count; ++j) {
                if (pos + 5 > data.size()) throw std::invalid_argument("Unexpected end of data");
                uint8_t value = data[pos++];
                uint32_t count = read_uint32();
                compressed_frame.emplace_back(value, count);
            }
            
            result.compressed_frames_.push_back(compressed_frame);
        }
        
        return result;
    }
};

#endif