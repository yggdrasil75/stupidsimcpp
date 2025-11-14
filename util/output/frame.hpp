#ifndef FRAME_HPP
#define FRAME_HPP

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <initializer_list>
#include <utility>

class frame {
private:
    std::vector<uint8_t> data_;
    size_t width_;
    size_t height_;
    std::vector<char> channels_;

    void validate_dimensions() const {
        size_t expected_size = width_ * height_ * channels_.size();
        if (data_.size() != expected_size) {
            throw std::invalid_argument("Data size does not match dimensions");
        }
        if (width_ == 0 || height_ == 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
        if (channels_.empty()) {
            throw std::invalid_argument("Channels list cannot be empty");
        }
    }
public:
    // Default constructor
    frame() : width_(0), height_(0) {}
    
    // Constructor with dimensions and channel names
    frame(size_t width, size_t height, const std::vector<char>& channels = {'\0'})
        : width_(width), height_(height), channels_(channels) {
        if (width == 0 || height == 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
        if (channels.empty()) {
            throw std::invalid_argument("Channels list cannot be empty");
        }
        data_.resize(width * height * channels.size());
    }
    
    // Constructor with initializer list for channels
    frame(size_t width, size_t height, std::initializer_list<char> channels)
        : width_(width), height_(height), channels_(channels) {
        if (width == 0 || height == 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
        if (channels.size() == 0) {
            throw std::invalid_argument("Channels list cannot be empty");
        }
        data_.resize(width * height * channels.size());
    }
    
    // Constructor with existing data
    frame(const std::vector<uint8_t>& data, size_t width, size_t height, 
          const std::vector<char>& channels = {'\0'})
        : data_(data), width_(width), height_(height), channels_(channels) {
        validate_dimensions();
    }
    
    // Move constructor with data
    frame(std::vector<uint8_t>&& data, size_t width, size_t height, 
          const std::vector<char>& channels = {'\0'})
        : data_(std::move(data)), width_(width), height_(height), channels_(channels) {
        validate_dimensions();
    }
    
    // Copy constructor
    frame(const frame&) = default;
    
    // Move constructor
    frame(frame&&) = default;
    
    // Copy assignment
    frame& operator=(const frame&) = default;
    
    // Move assignment
    frame& operator=(frame&&) = default;
    
    // Accessors
    size_t width() const noexcept { return width_; }
    size_t height() const noexcept { return height_; }
    const std::vector<char>& channels() const noexcept { return channels_; }
    size_t channels_count() const noexcept { return channels_.size(); }
    size_t size() const noexcept { return data_.size(); }
    size_t total_pixels() const noexcept { return width_ * height_; }
    
    // Data access
    const std::vector<uint8_t>& data() const noexcept { return data_; }
    std::vector<uint8_t>& data() noexcept { return data_; }
    
    // Raw pointer access (const and non-const)
    const uint8_t* raw_data() const noexcept { return data_.data(); }
    uint8_t* raw_data() noexcept { return data_.data(); }
    
    // Pixel access by channel index
    uint8_t& at(size_t row, size_t col, size_t channel_idx) {
        if (row >= height_ || col >= width_ || channel_idx >= channels_.size()) {
            throw std::out_of_range("Pixel coordinates or channel index out of range");
        }
        return data_[(row * width_ + col) * channels_.size() + channel_idx];
    }
    
    const uint8_t& at(size_t row, size_t col, size_t channel_idx) const {
        if (row >= height_ || col >= width_ || channel_idx >= channels_.size()) {
            throw std::out_of_range("Pixel coordinates or channel index out of range");
        }
        return data_[(row * width_ + col) * channels_.size() + channel_idx];
    }
    
    // Pixel access by channel name (returns first occurrence)
    uint8_t& at(size_t row, size_t col, char channel_name) {
        return at(row, col, get_channel_index(channel_name));
    }
    
    const uint8_t& at(size_t row, size_t col, char channel_name) const {
        return at(row, col, get_channel_index(channel_name));
    }
    
    // Get channel index by name (returns first occurrence)
    size_t get_channel_index(char channel_name) const {
        for (size_t i = 0; i < channels_.size(); ++i) {
            if (channels_[i] == channel_name) {
                return i;
            }
        }
        throw std::out_of_range("Channel name not found: " + std::string(1, channel_name));
    }
    
    // Check if channel exists
    bool has_channel(char channel_name) const {
        for (char c : channels_) {
            if (c == channel_name) {
                return true;
            }
        }
        return false;
    }
    
    // Get all values for a specific channel across the image
    std::vector<uint8_t> get_channel_data(char channel_name) const {
        size_t channel_idx = get_channel_index(channel_name);
        std::vector<uint8_t> result(total_pixels());
        size_t pixel_count = total_pixels();
        size_t channel_count = channels_.size();
        
        for (size_t i = 0; i < pixel_count; ++i) {
            result[i] = data_[i * channel_count + channel_idx];
        }
        return result;
    }
    
    // Set all values for a specific channel across the image
    void set_channel_data(char channel_name, const std::vector<uint8_t>& channel_data) {
        if (channel_data.size() != total_pixels()) {
            throw std::invalid_argument("Channel data size does not match image dimensions");
        }
        
        size_t channel_idx = get_channel_index(channel_name);
        size_t pixel_count = total_pixels();
        size_t channel_count = channels_.size();
        
        for (size_t i = 0; i < pixel_count; ++i) {
            data_[i * channel_count + channel_idx] = channel_data[i];
        }
    }
    
    // Check if frame is valid/initialized
    bool empty() const noexcept {
        return width_ == 0 || height_ == 0 || data_.empty();
    }
    
    // Resize the frame (clears existing data)
    void resize(size_t width, size_t height, const std::vector<char>& channels = {'\0'}) {
        if (width == 0 || height == 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
        if (channels.empty()) {
            throw std::invalid_argument("Channels list cannot be empty");
        }
        width_ = width;
        height_ = height;
        channels_ = channels;
        data_.resize(width * height * channels.size());
    }
    
    // Resize with initializer list for channels
    void resize(size_t width, size_t height, std::initializer_list<char> channels) {
        resize(width, height, std::vector<char>(channels));
    }
    
    // Change channel names (must maintain same number of channels)
    void set_channels(const std::vector<char>& new_channels) {
        if (new_channels.size() != channels_.size()) {
            throw std::invalid_argument("New channels count must match current channels count");
        }
        if (new_channels.empty()) {
            throw std::invalid_argument("Channels list cannot be empty");
        }
        channels_ = new_channels;
    }
    
    // Clear the frame
    void clear() noexcept {
        data_.clear();
        width_ = 0;
        height_ = 0;
        channels_.clear();
    }
    
    // Swap with another frame
    void swap(frame& other) noexcept {
        data_.swap(other.data_);
        std::swap(width_, other.width_);
        std::swap(height_, other.height_);
        channels_.swap(other.channels_);
    }
    
    // Create a deep copy
    frame clone() const {
        return frame(*this);
    }
    
    // Get string representation of channels
    std::string channels_string() const {
        return std::string(channels_.begin(), channels_.end());
    }

    // RLE Compression - Compress the entire frame data
    std::vector<std::pair<uint8_t, uint32_t>> compress_rle() const {
        if (empty()) {
            return {};
        }
        
        std::vector<std::pair<uint8_t, uint32_t>> compressed;
        
        if (data_.empty()) {
            return compressed;
        }
        
        uint8_t current_value = data_[0];
        uint32_t count = 1;
        
        for (size_t i = 1; i < data_.size(); ++i) {
            if (data_[i] == current_value && count < UINT32_MAX) {
                ++count;
            } else {
                compressed.emplace_back(current_value, count);
                current_value = data_[i];
                count = 1;
            }
        }
        
        // Add the last run
        compressed.emplace_back(current_value, count);
        
        return compressed;
    }
    
    // RLE Compression for a specific channel
    std::vector<std::pair<uint8_t, uint32_t>> compress_channel_rle(char channel_name) const {
        if (empty()) {
            return {};
        }
        
        std::vector<uint8_t> channel_data = get_channel_data(channel_name);
        std::vector<std::pair<uint8_t, uint32_t>> compressed;
        
        if (channel_data.empty()) {
            return compressed;
        }
        
        uint8_t current_value = channel_data[0];
        uint32_t count = 1;
        
        for (size_t i = 1; i < channel_data.size(); ++i) {
            if (channel_data[i] == current_value && count < UINT32_MAX) {
                ++count;
            } else {
                compressed.emplace_back(current_value, count);
                current_value = channel_data[i];
                count = 1;
            }
        }
        
        // Add the last run
        compressed.emplace_back(current_value, count);
        
        return compressed;
    }
    
    // RLE Decompression - Decompress RLE data into this frame
    void decompress_rle(const std::vector<std::pair<uint8_t, uint32_t>>& compressed_data) {
        if (compressed_data.empty()) {
            clear();
            return;
        }
        
        // Calculate total size from compressed data
        size_t total_size = 0;
        for (const auto& run : compressed_data) {
            total_size += run.second;
        }
        
        // Resize data vector to accommodate decompressed data
        data_.resize(total_size);
        
        // Decompress the data
        size_t index = 0;
        for (const auto& run : compressed_data) {
            for (uint32_t i = 0; i < run.second; ++i) {
                if (index < data_.size()) {
                    data_[index++] = run.first;
                }
            }
        }
        
        // Note: After RLE decompression, width_, height_, and channels_ might not be valid
        // The user should set these appropriately after decompression
    }
    
    // Static method to create frame from RLE compressed data with known dimensions
    static frame from_rle(const std::vector<std::pair<uint8_t, uint32_t>>& compressed_data,
                         size_t width, size_t height, 
                         const std::vector<char>& channels = {'\0'}) {
        frame result;
        result.decompress_rle(compressed_data);
        
        // Validate that decompressed data size matches expected dimensions
        size_t expected_size = width * height * channels.size();
        if (result.data_.size() != expected_size) {
            throw std::invalid_argument("Decompressed data size does not match provided dimensions");
        }
        
        result.width_ = width;
        result.height_ = height;
        result.channels_ = channels;
        
        return result;
    }
    
    // Calculate compression ratio
    double get_compression_ratio() const {
        if (empty()) {
            return 1.0;
        }
        
        auto compressed = compress_rle();
        if (compressed.empty()) {
            return 1.0;
        }
        
        size_t original_size = data_.size();
        size_t compressed_size = compressed.size() * sizeof(std::pair<uint8_t, uint32_t>);
        
        return static_cast<double>(original_size) / compressed_size;
    }
    
    // Get size of compressed data in bytes
    size_t get_compressed_size() const {
        auto compressed = compress_rle();
        return compressed.size() * sizeof(std::pair<uint8_t, uint32_t>);
    }
    
    // Check if compression would be beneficial (ratio > 1.0)
    bool would_benefit_from_compression() const {
        return get_compression_ratio() > 1.0;
    }
};

#endif