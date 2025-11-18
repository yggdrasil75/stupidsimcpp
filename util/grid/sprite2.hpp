#ifndef SPRITE2_HPP
#define SPRITE2_HPP

#include "grid2.hpp"
#include "../output/frame.hpp"

class SpriteMap2 : public Grid2 {
private:

    // id, sprite
    std::unordered_map<size_t, frame> spritesComped;
    std::unordered_map<size_t, int> Layers;
    std::unordered_map<size_t, float> Orientations;

public:

    using Grid2::Grid2;
    size_t addSprite(const Vec2& pos, frame sprite, int layer = 0, float orientation = 0.0f) {
        size_t id = addObject(pos, Vec4(0,0,0,0));
        spritesComped[id] = sprite;
        Layers[id] = layer;
        Orientations[id] = orientation;
    }

    frame getSprite(size_t id) {
        return spritesComped.at(id);
    }

    void setSprite(size_t id, const frame& sprite) {
        spritesComped[id] = sprite;
    }

    int getLayer(size_t id) {
        return Layers.at(id);
    }

    size_t setLayer(size_t id, int layer) {
        Layers[id] = layer;
        return id;
    }

    float getOrientation(size_t id) {
        return Orientations.at(id);
    }
    size_t setOrientation(size_t id, float orientation) {
        Orientations[id] = orientation;
    }

    void getGridRegionAsBGR(const Vec2& minCorner, const Vec2& maxCorner, int& width, int& height, std::vector<uint8_t>& bgrData) const {
        TIME_FUNCTION;
        // Calculate dimensions
        width = static_cast<int>(maxCorner.x - minCorner.x);
        height = static_cast<int>(maxCorner.y - minCorner.y);
        
        if (width <= 0 || height <= 0) {
            width = height = 0;
            bgrData.clear();
            bgrData.shrink_to_fit();
            return;
        }
        
        // Initialize RGB data (3 bytes per pixel: R, G, B)
        std::vector<Vec4> rgbaBuffer(width * height, Vec4(0.0f, 0.0f, 0.0f, 0.0f));
        
        // For each position in the grid, find the corresponding pixel
        for (const auto& [id, pos] : Positions) {
            size_t size = Sizes.at(id);
                
            // Calculate pixel coordinates
            int pixelXm = static_cast<int>(pos.x - size/2 - minCorner.x);
            int pixelXM = static_cast<int>(pos.x + size/2 - minCorner.x);
            int pixelYm = static_cast<int>(pos.y - size/2 - minCorner.y);
            int pixelYM = static_cast<int>(pos.y + size/2 - minCorner.y);
            
            pixelXm = std::max(0, pixelXm);
            pixelXM = std::min(width - 1, pixelXM);
            pixelYm = std::max(0, pixelYm);
            pixelYM = std::min(height - 1, pixelYM);

            // Ensure within bounds
            if (pixelXM >= minCorner.x && pixelXm < width && pixelYM >= minCorner.y && pixelYm < height) {
                const Vec4& color = Colors.at(id);
                float srcAlpha = color.a;
                float invSrcAlpha = 1.0f - srcAlpha;
                for (int py = pixelYm; py <= pixelYM; ++py){
                    for (int px = pixelXm; px <= pixelXM; ++px){
                        int index = (py * width + px) * 3;
                        Vec4& dest = rgbaBuffer[index];
                        
                        dest.r = color.r * srcAlpha + dest.r * invSrcAlpha;
                        dest.g = color.g * srcAlpha + dest.g * invSrcAlpha;
                        dest.b = color.b * srcAlpha + dest.b * invSrcAlpha;
                        dest.a = srcAlpha + dest.a * invSrcAlpha;
                    }
                }
            }
        }
        bgrData.resize(dest.size() * 4);
        for (int i = 0; i < width * height; ++i) {
            const Vec4& color = rgbaBuffer[i];
            int bgrIndex = i * 3;
            
            // Convert from [0,1] to [0,255] and store as BGR
            bgrData[bgrIndex + 0] = static_cast<uint8_t>(color.b * 255);  // Blue
            bgrData[bgrIndex + 1] = static_cast<uint8_t>(color.g * 255);  // Green
            bgrData[bgrIndex + 2] = static_cast<uint8_t>(color.r * 255);  // Red
        }
    }

    size_t removeSprite(size_t id) {
        spritesComped.erase(id);
        Layers.erase(id);
        Orientations.erase(id);
        return removeID(id);
    }

    // Remove sprite by position
    size_t removeSprite(const Vec2& pos) {
        size_t id = getPositionVec(pos);
        return removeSprite(id);
    }

    void clear() {
        Grid2::clear();
        spritesComped.clear();
        Layers.clear();
        Orientations.clear();
        spritesComped.rehash(0);
        Layers.rehash(0);
        Orientations.rehash(0);
    }

    // Get all sprite IDs
    std::vector<size_t> getAllSpriteIDs() {
        return getAllIDs();
    }

    // Check if ID has a sprite
    bool hasSprite(size_t id) const {
        return spritesComped.find(id) != spritesComped.end();
    }

    // Get number of sprites
    size_t getSpriteCount() const {
        return spritesComped.size();
    }

};

#endif