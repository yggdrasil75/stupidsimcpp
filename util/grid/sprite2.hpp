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
        return id;
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
        return id;
    }

    void getGridRegionAsBGR(const Vec2& minCorner, const Vec2& maxCorner, int& width, int& height, std::vector<uint8_t>& rgbData) const {
        TIME_FUNCTION;
        
        // Calculate dimensions
        width = static_cast<int>(maxCorner.x - minCorner.x);
        height = static_cast<int>(maxCorner.y - minCorner.y);
        
        if (width <= 0 || height <= 0) {
            width = height = 0;
            rgbData.clear();
            rgbData.shrink_to_fit();
            return;
        }
        
        // Initialize RGBA buffer for compositing
        std::vector<Vec4> rgbaBuffer(width * height, Vec4(0.0f, 0.0f, 0.0f, 0.0f));
        
        // Group sprites by layer for proper rendering order
        std::vector<std::pair<int, size_t>> layeredSprites;
        for (const auto& [id, pos] : Positions) {
            if (spritesComped.find(id) != spritesComped.end()) {
                layeredSprites.emplace_back(Layers.at(id), id);
            }
        }
        
        // Sort by layer (lower layers first)
        std::sort(layeredSprites.begin(), layeredSprites.end(), 
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Render each sprite in layer order
        for (const auto& [layer, id] : layeredSprites) {
            const Vec2& pos = Positions.at(id);
            const frame& sprite = spritesComped.at(id);
            float orientation = Orientations.at(id);
            
            // Decompress sprite if needed
            frame decompressedSprite = sprite;
            if (sprite.isCompressed()) {
                decompressedSprite.decompress();
            }
            
            const std::vector<uint8_t>& spriteData = decompressedSprite.getData();
            size_t spriteWidth = decompressedSprite.getWidth();
            size_t spriteHeight = decompressedSprite.getHeight();
            
            if (spriteData.empty() || spriteWidth == 0 || spriteHeight == 0) {
                continue;
            }
            
            // Calculate sprite bounds in world coordinates
            float halfWidth = spriteWidth / 2.0f;
            float halfHeight = spriteHeight / 2.0f;
            
            // Apply rotation if needed
            // TODO: Implement proper rotation transformation
            int pixelXm = static_cast<int>(pos.x - halfWidth - minCorner.x);
            int pixelXM = static_cast<int>(pos.x + halfWidth - minCorner.x);
            int pixelYm = static_cast<int>(pos.y - halfHeight - minCorner.y);
            int pixelYM = static_cast<int>(pos.y + halfHeight - minCorner.y);
            
            // Clamp to output bounds
            pixelXm = std::max(0, pixelXm);
            pixelXM = std::min(width - 1, pixelXM);
            pixelYm = std::max(0, pixelYm);
            pixelYM = std::min(height - 1, pixelYM);
            
            // Skip if completely outside bounds
            if (pixelXm >= width || pixelXM < 0 || pixelYm >= height || pixelYM < 0) {
                continue;
            }
            
            // Render sprite pixels
            for (int py = pixelYm; py <= pixelYM; ++py) {
                for (int px = pixelXm; px <= pixelXM; ++px) {
                    // Calculate sprite-relative coordinates
                    int spriteX = px - pixelXm;
                    int spriteY = py - pixelYm;
                    
                    // Skip if outside sprite bounds
                    if (spriteX < 0 || spriteX >= spriteWidth || spriteY < 0 || spriteY >= spriteHeight) {
                        continue;
                    }
                    
                    // Get sprite pixel color based on color format
                    Vec4 spriteColor = getSpritePixelColor(spriteData, spriteX, spriteY, spriteWidth, spriteHeight, decompressedSprite.colorFormat);
                    
                    // Alpha blending
                    int bufferIndex = py * width + px;
                    Vec4& dest = rgbaBuffer[bufferIndex];
                    
                    float srcAlpha = spriteColor.a;
                    if (srcAlpha > 0.0f) {
                        float invSrcAlpha = 1.0f - srcAlpha;
                        dest.r = spriteColor.r * srcAlpha + dest.r * invSrcAlpha;
                        dest.g = spriteColor.g * srcAlpha + dest.g * invSrcAlpha;
                        dest.b = spriteColor.b * srcAlpha + dest.b * invSrcAlpha;
                        dest.a = srcAlpha + dest.a * invSrcAlpha;
                    }
                }
            }
        }
        
        // Also render regular colored objects (from base class)
        for (const auto& [id, pos] : Positions) {
            // Skip if this is a sprite (already rendered above)
            if (spritesComped.find(id) != spritesComped.end()) {
                continue;
            }
            
            size_t size = Sizes.at(id);
            
            // Calculate pixel coordinates for colored objects
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
                for (int py = pixelYm; py <= pixelYM; ++py) {
                    for (int px = pixelXm; px <= pixelXM; ++px) {
                        int index = py * width + px;
                        Vec4& dest = rgbaBuffer[index];
                        
                        float invSrcAlpha = 1.0f - srcAlpha;
                        dest.r = color.r * srcAlpha + dest.r * invSrcAlpha;
                        dest.g = color.g * srcAlpha + dest.g * invSrcAlpha;
                        dest.b = color.b * srcAlpha + dest.b * invSrcAlpha;
                        dest.a = srcAlpha + dest.a * invSrcAlpha;
                    }
                }
            }
        }
        
        // Convert RGBA buffer to BGR output
        rgbData.resize(rgbaBuffer.size() * 3);
        for (size_t i = 0; i < rgbaBuffer.size(); ++i) {
            const Vec4& color = rgbaBuffer[i];
            size_t bgrIndex = i * 3;
            
            // Convert from [0,1] to [0,255] and store as BGR
            rgbData[bgrIndex + 2] = static_cast<uint8_t>(color.r * 255);  // R -> third position
            rgbData[bgrIndex + 1] = static_cast<uint8_t>(color.g * 255);  // G -> second position  
            rgbData[bgrIndex + 0] = static_cast<uint8_t>(color.b * 255);  // B -> first position
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

private:
    // Helper function to extract pixel color from sprite data based on color format
    Vec4 getSpritePixelColor(const std::vector<uint8_t>& spriteData, 
                            int x, int y, 
                            size_t spriteWidth, size_t spriteHeight,
                            frame::colormap format) const {
        size_t pixelIndex = y * spriteWidth + x;
        size_t channels = 3; // Default to RGB
        
        switch (format) {
            case frame::colormap::RGB:
                channels = 3;
                if (pixelIndex * channels + 2 < spriteData.size()) {
                    return Vec4(spriteData[pixelIndex * channels] / 255.0f,
                               spriteData[pixelIndex * channels + 1] / 255.0f,
                               spriteData[pixelIndex * channels + 2] / 255.0f,
                               1.0f);
                }
                break;
                
            case frame::colormap::RGBA:
                channels = 4;
                if (pixelIndex * channels + 3 < spriteData.size()) {
                    return Vec4(spriteData[pixelIndex * channels] / 255.0f,
                               spriteData[pixelIndex * channels + 1] / 255.0f,
                               spriteData[pixelIndex * channels + 2] / 255.0f,
                               spriteData[pixelIndex * channels + 3] / 255.0f);
                }
                break;
                
            case frame::colormap::BGR:
                channels = 3;
                if (pixelIndex * channels + 2 < spriteData.size()) {
                    return Vec4(spriteData[pixelIndex * channels + 2] / 255.0f,  // BGR -> RGB
                               spriteData[pixelIndex * channels + 1] / 255.0f,
                               spriteData[pixelIndex * channels] / 255.0f,
                               1.0f);
                }
                break;
                
            case frame::colormap::BGRA:
                channels = 4;
                if (pixelIndex * channels + 3 < spriteData.size()) {
                    return Vec4(spriteData[pixelIndex * channels + 2] / 255.0f,  // BGRA -> RGBA
                               spriteData[pixelIndex * channels + 1] / 255.0f,
                               spriteData[pixelIndex * channels] / 255.0f,
                               spriteData[pixelIndex * channels + 3] / 255.0f);
                }
                break;
                
            case frame::colormap::B:
                channels = 1;
                if (pixelIndex < spriteData.size()) {
                    float value = spriteData[pixelIndex] / 255.0f;
                    return Vec4(value, value, value, 1.0f);
                }
                break;
        }
        
        // Return transparent black if out of bounds
        return Vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
};

#endif