#ifndef VIDEO_HPP
#define VIDEO_HPP


#include "frame.hpp"
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include "../timing_decorator.hpp"

class video {
private:
    std::vector<std::vector<std::pair<uint8_t, uint32_t>>> compressed_frames_;
    std::unordered_map<size_t, size_t> keyframe_indices_; // Maps frame index to keyframe index
    size_t width_;
    size_t height_;
    std::vector<char> channels_;
    double fps_;
    bool use_differential_encoding_;
    size_t keyframe_interval_;

    // Compress frame using differential encoding
    std::vector<std::pair<uint8_t, uint32_t>> compress_with_differential(
        const frame& current_frame, const frame* previous_frame = nullptr) const {
        TIME_FUNCTION;
        
        if (previous_frame == nullptr) {
            // First frame or keyframe - compress normally
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

    // Find the nearest keyframe index for a given frame index
    size_t find_nearest_keyframe(size_t frame_index) const {
        if (keyframe_indices_.empty()) return 0;
        
        // Keyframes are stored at intervals, so we can calculate the nearest one
        size_t keyframe_idx = (frame_index / keyframe_interval_) * keyframe_interval_;
        
        // Make sure the keyframe exists
        if (keyframe_idx >= compressed_frames_.size()) {
            // Find the last available keyframe
            for (size_t i = frame_index; i > 0; --i) {
                if (keyframe_indices_.count(i)) {
                    return i;
                }
            }
            return 0;
        }
        
        return keyframe_idx;
    }

    // Build keyframe indices (call this when frames change)
    void rebuild_keyframe_indices() {
        keyframe_indices_.clear();
        for (size_t i = 0; i < compressed_frames_.size(); i += keyframe_interval_) {
            if (i < compressed_frames_.size()) {
                keyframe_indices_[i] = i;
            }
        }
        // Always ensure frame 0 is a keyframe
        if (!compressed_frames_.empty() && !keyframe_indices_.count(0)) {
            keyframe_indices_[0] = 0;
        }
    }

    // Get frame with keyframe optimization - much faster for random access
    frame get_frame_optimized(size_t index) const {
        if (index >= compressed_frames_.size()) {
            throw std::out_of_range("Frame index out of range");
        }
        
        // If it's a keyframe or we're not using differential encoding, decompress directly
        if (keyframe_indices_.count(index) || !use_differential_encoding_) {
            frame result;
            result.decompress_rle(compressed_frames_[index]);
            result.resize(width_, height_, channels_);
            return result;
        }
        
        // Find the nearest keyframe
        size_t keyframe_idx = find_nearest_keyframe(index);
        
        // Decompress the keyframe first
        frame current_frame = get_frame_optimized(keyframe_idx);
        
        // Then decompress all frames from keyframe to target frame
        for (size_t i = keyframe_idx + 1; i <= index; ++i) {
            current_frame = decompress_differential(compressed_frames_[i], current_frame);
        }
        
        return current_frame;
    }

public:
    // Default constructor
    video() : width_(0), height_(0), fps_(30.0), use_differential_encoding_(true), keyframe_interval_(50) {}
    
    // Constructor with dimensions and settings
    video(size_t width, size_t height, const std::vector<char>& channels = {'\0'},
          double fps = 30.0, bool use_differential = true, size_t keyframe_interval = 50)
        : width_(width), height_(height), channels_(channels), fps_(fps),
          use_differential_encoding_(use_differential), keyframe_interval_(keyframe_interval) {
        
        if (width == 0 || height == 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
        if (channels.empty()) {
            throw std::invalid_argument("Channels list cannot be empty");
        }
        if (fps <= 0) {
            throw std::invalid_argument("FPS must be positive");
        }
        if (keyframe_interval == 0) {
            throw std::invalid_argument("Keyframe interval must be positive");
        }
    }
    
    // Constructor with initializer list for channels
    video(size_t width, size_t height, std::initializer_list<char> channels,
          double fps = 30.0, bool use_differential = true, size_t keyframe_interval = 50)
        : video(width, height, std::vector<char>(channels), fps, use_differential, keyframe_interval) {}
    
    // Accessors
    size_t width() const noexcept { return width_; }
    size_t height() const noexcept { return height_; }
    const std::vector<char>& channels() const noexcept { return channels_; }
    double fps() const noexcept { return fps_; }
    bool use_differential_encoding() const noexcept { return use_differential_encoding_; }
    size_t frame_count() const noexcept { return compressed_frames_.size(); }
    size_t channels_count() const noexcept { return channels_.size(); }
    size_t keyframe_interval() const noexcept { return keyframe_interval_; }
    const std::unordered_map<size_t, size_t>& keyframe_indices() const noexcept { return keyframe_indices_; }
    
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
        
        size_t new_index = compressed_frames_.size();
        
        if (compressed_frames_.empty() || !use_differential_encoding_) {
            // First frame or differential encoding disabled - compress normally
            compressed_frames_.push_back(new_frame.compress_rle());
        } else {
            // Check if this should be a keyframe
            bool is_keyframe = (new_index % keyframe_interval_ == 0);
            
            if (is_keyframe) {
                // Keyframe - compress normally
                compressed_frames_.push_back(new_frame.compress_rle());
                keyframe_indices_[new_index] = new_index;
            } else {
                // Regular frame - use differential encoding from previous frame
                frame prev_frame = get_frame_optimized(new_index - 1);
                compressed_frames_.push_back(compress_with_differential(new_frame, &prev_frame));
            }
        }
        
        // Ensure we have keyframe at index 0
        if (compressed_frames_.size() == 1) {
            keyframe_indices_[0] = 0;
        }
    }
    
    // Add frame with move semantics
    void add_frame(frame&& new_frame) {
        add_frame(new_frame); // Just call the const version
    }
    
    // Get a specific frame (uses optimized version with keyframes)
    frame get_frame(size_t index) const {
        TIME_FUNCTION;
        if (!use_differential_encoding_ || keyframe_indices_.empty()) {
            // Fallback to original method if no optimization possible
            if (index >= compressed_frames_.size()) {
                throw std::out_of_range("Frame index out of range");
            }
            
            if (index == 0 || !use_differential_encoding_) {
                frame result;
                result.decompress_rle(compressed_frames_[index]);
                result.resize(width_, height_, channels_);
                return result;
            } else {
                frame prev_frame = get_frame(index - 1);
                return decompress_differential(compressed_frames_[index], prev_frame);
            }
        }
        
        return get_frame_optimized(index);
    }
    
    // Get multiple frames as a sequence (optimized for sequential access)
    std::vector<frame> get_frames(size_t start_index, size_t count) const {
        TIME_FUNCTION;
        if (start_index >= compressed_frames_.size()) {
            throw std::out_of_range("Start index out of range");
        }
        
        count = std::min(count, compressed_frames_.size() - start_index);
        std::vector<frame> frames;
        frames.reserve(count);
        
        if (!use_differential_encoding_ || keyframe_indices_.empty()) {
            // Original sequential method
            for (size_t i = start_index; i < start_index + count; ++i) {
                frames.push_back(get_frame(i));
            }
        } else {
            // Optimized method: start from nearest keyframe
            size_t current_index = start_index;
            size_t keyframe_idx = find_nearest_keyframe(start_index);
            
            // Get the keyframe
            frame current_frame = get_frame_optimized(keyframe_idx);
            
            // If we started before the keyframe (shouldn't happen), handle it
            if (keyframe_idx > start_index) {
                // This is a fallback - should not normally occur
                current_frame = get_frame_optimized(start_index);
                current_index = start_index + 1;
            } else if (keyframe_idx < start_index) {
                // Decode frames from keyframe to start_index
                for (size_t i = keyframe_idx + 1; i < start_index; ++i) {
                    current_frame = decompress_differential(compressed_frames_[i], current_frame);
                }
            }
            
            // Now add the requested frames
            for (size_t i = start_index; i < start_index + count; ++i) {
                if (i > keyframe_idx) {
                    current_frame = decompress_differential(compressed_frames_[i], current_frame);
                }
                frames.push_back(current_frame);
            }
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
        rebuild_keyframe_indices();
    }
    
    // Clear all frames
    void clear_frames() noexcept {
        compressed_frames_.clear();
        keyframe_indices_.clear();
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
        
        bool was_keyframe = keyframe_indices_.count(index);
        bool should_be_keyframe = (index % keyframe_interval_ == 0);
        
        if (index == 0 || !use_differential_encoding_ || should_be_keyframe) {
            // Keyframe or no differential encoding - compress normally
            compressed_frames_[index] = new_frame.compress_rle();
            if (should_be_keyframe) {
                keyframe_indices_[index] = index;
            }
        } else {
            // Differential frame
            frame prev_frame = get_frame_optimized(index - 1);
            compressed_frames_[index] = compress_with_differential(new_frame, &prev_frame);
            
            // Remove from keyframes if it was one but shouldn't be
            if (was_keyframe && !should_be_keyframe) {
                keyframe_indices_.erase(index);
            }
        }
        
        // If this isn't the last frame, we need to update the next frame's differential encoding
        if (use_differential_encoding_ && index + 1 < compressed_frames_.size()) {
            frame current_frame = get_frame_optimized(index);
            frame next_frame_original = get_frame_optimized(index + 1);
            compressed_frames_[index + 1] = compress_with_differential(next_frame_original, &current_frame);
        }
        
        // Rebuild keyframe indices if we changed keyframe status
        if (was_keyframe != should_be_keyframe) {
            rebuild_keyframe_indices();
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
    
    // Set keyframe interval and rebuild indices
    void set_keyframe_interval(size_t interval) {
        if (interval == 0) {
            throw std::invalid_argument("Keyframe interval must be positive");
        }
        
        if (interval != keyframe_interval_) {
            keyframe_interval_ = interval;
            if (!compressed_frames_.empty()) {
                // Rebuild keyframe indices with new interval
                rebuild_keyframe_indices();
                
                // If we have frames, we may need to recompress some as keyframes
                if (use_differential_encoding_) {
                    auto original_frames = get_all_frames();
                    clear_frames();
                    
                    for (const auto& f : original_frames) {
                        add_frame(f);
                    }
                }
            }
        }
    }
    
    // Force a specific frame to be a keyframe
    void make_keyframe(size_t index) {
        if (index >= compressed_frames_.size()) {
            throw std::out_of_range("Frame index out of range");
        }
        
        if (!keyframe_indices_.count(index)) {
            // Recompress this frame as a keyframe
            frame original_frame = get_frame_optimized(index);
            compressed_frames_[index] = original_frame.compress_rle();
            keyframe_indices_[index] = index;
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
        size_t keyframe_count;
        size_t keyframe_interval;
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
        stats.keyframe_count = keyframe_indices_.size();
        stats.keyframe_interval = keyframe_interval_;
        return stats;
    }
    
    // Extract a sub-video
    video subvideo(size_t start_frame, size_t frame_count) const {
        TIME_FUNCTION;
        if (start_frame >= compressed_frames_.size()) {
            throw std::out_of_range("Start frame out of range");
        }
        
        frame_count = std::min(frame_count, compressed_frames_.size() - start_frame);
        video result(width_, height_, channels_, fps_, use_differential_encoding_, keyframe_interval_);
        
        // Add frames one by one to maintain proper keyframe structure
        for (size_t i = start_frame; i < start_frame + frame_count; ++i) {
            result.add_frame(get_frame(i));
        }
        
        return result;
    }
    
    // Append another video (must have same dimensions and channels)
    void append_video(const video& other) {
        TIME_FUNCTION;
        if (other.width_ != width_ || other.height_ != height_ || other.channels_ != channels_) {
            throw std::invalid_argument("Videos must have same dimensions and channels");
        }
        
        // Add frames one by one to maintain proper keyframe structure
        auto other_frames = other.get_all_frames();
        for (const auto& frame : other_frames) {
            add_frame(frame);
        }
    }
    
    // Save/Load functionality (basic serialization) - updated for keyframes
    std::vector<uint8_t> serialize() const {
        TIME_FUNCTION;
        // Simple serialization format:
        // [header][compressed_frame_data...]
        // Header: width(4), height(4), channels_count(1), channels_data(n), fps(8), 
        //         use_diff(1), keyframe_interval(4), frame_count(4), keyframe_count(4), keyframe_indices...
        
        std::vector<uint8_t> result;
        
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
        add_uint32(static_cast<uint32_t>(keyframe_interval_));
        add_uint32(static_cast<uint32_t>(compressed_frames_.size()));
        
        // Write keyframe indices
        add_uint32(static_cast<uint32_t>(keyframe_indices_.size()));
        for (const auto& kv : keyframe_indices_) {
            add_uint32(static_cast<uint32_t>(kv.first));
        }
        
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
        if (data.size() < 4 + 4 + 1 + 8 + 1 + 4 + 4 + 4) { // Minimum header size
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
        uint32_t keyframe_interval = read_uint32();
        uint32_t frame_count = read_uint32();
        
        video result(width, height, channels, fps, use_diff, keyframe_interval);
        
        // Read keyframe indices
        uint32_t keyframe_count = read_uint32();
        for (uint32_t i = 0; i < keyframe_count; ++i) {
            uint32_t keyframe_index = read_uint32();
            result.keyframe_indices_[keyframe_index] = keyframe_index;
        }
        
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