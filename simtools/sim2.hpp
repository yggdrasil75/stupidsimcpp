#ifndef SIM2_HPP
#define SIM2_HPP

#include "../util/noise2.hpp"
#include "../util/grid2.hpp"
#include "../util/vec2.hpp"
#include "../util/vec4.hpp"
#include "../util/timing_decorator.hpp"
#include <memory>
#include <string>
#include <unordered_map>

class Sim2 {
private:
    std::unique_ptr<Noise2> noiseGenerator;
    Grid2 terrainGrid;
    int gridWidth;
    int gridHeight;
    
    // Terrain generation parameters
    float scale;
    int octaves;
    float persistence;
    float lacunarity;
    uint32_t seed;
    Vec2 offset;
    
    // Terrain modification parameters
    float elevationMultiplier;
    float waterLevel;
    Vec4 landColor;
    Vec4 waterColor;
    
public:
    Sim2(int width = 512, int height = 512, uint32_t seed = 12345) 
        : gridWidth(width), gridHeight(height), scale(4.0f), octaves(4), 
          persistence(0.5f), lacunarity(2.0f), seed(seed), offset(0, 0),
          elevationMultiplier(1.0f), waterLevel(0.3f),
          landColor(0.2f, 0.8f, 0.2f, 1.0f),  // Green
          waterColor(0.2f, 0.3f, 0.8f, 1.0f)  // Blue
    {
        noiseGenerator = std::make_unique<Noise2>(seed);
        generateTerrain();
    }
    
    // Generate initial terrain
    void generateTerrain() {
        TIME_FUNCTION;
        terrainGrid = noiseGenerator->generateTerrainNoise(
            gridWidth, gridHeight, scale, octaves, persistence, seed, offset);
        
        applyTerrainColors();
    }
    
    // Regenerate terrain with current parameters
    void regenerate() {
        generateTerrain();
    }
    
    // Basic parameter modifications
    void setScale(float newScale) {
        scale = std::max(0.1f, newScale);
        generateTerrain();
    }
    
    void setOctaves(int newOctaves) {
        octaves = std::max(1, newOctaves);
        generateTerrain();
    }
    
    void setPersistence(float newPersistence) {
        persistence = std::clamp(newPersistence, 0.0f, 1.0f);
        generateTerrain();
    }
    
    void setLacunarity(float newLacunarity) {
        lacunarity = std::max(1.0f, newLacunarity);
        generateTerrain();
    }
    
    void setSeed(uint32_t newSeed) {
        seed = newSeed;
        noiseGenerator->setSeed(seed);
        generateTerrain();
    }
    
    void setOffset(const Vec2& newOffset) {
        offset = newOffset;
        generateTerrain();
    }
    
    void setElevationMultiplier(float multiplier) {
        elevationMultiplier = std::max(0.0f, multiplier);
        applyElevationModification();
    }
    
    void setWaterLevel(float level) {
        waterLevel = std::clamp(level, 0.0f, 1.0f);
        applyTerrainColors();
    }
    
    void setLandColor(const Vec4& color) {
        landColor = color;
        applyTerrainColors();
    }
    
    void setWaterColor(const Vec4& color) {
        waterColor = color;
        applyTerrainColors();
    }
    
    // Get current parameters
    float getScale() const { return scale; }
    int getOctaves() const { return octaves; }
    float getPersistence() const { return persistence; }
    float getLacunarity() const { return lacunarity; }
    uint32_t getSeed() const { return seed; }
    Vec2 getOffset() const { return offset; }
    float getElevationMultiplier() const { return elevationMultiplier; }
    float getWaterLevel() const { return waterLevel; }
    Vec4 getLandColor() const { return landColor; }
    Vec4 getWaterColor() const { return waterColor; }
    
    // Get the terrain grid
    const Grid2& getTerrainGrid() const {
        return terrainGrid;
    }
    
    // Get terrain dimensions
    int getWidth() const { return gridWidth; }
    int getHeight() const { return gridHeight; }
    
    // Get elevation at specific coordinates
    float getElevation(int x, int y) const {
        if (x < 0 || x >= gridWidth || y < 0 || y >= gridHeight) {
            return 0.0f;
        }
        return terrainGrid.colors[y * gridWidth + x].x; // Elevation stored in red channel
    }
    
    // Render to RGB image
    std::vector<uint8_t> renderToRGB(int width, int height, 
                                    const Vec4& backgroundColor = Vec4(0, 0, 0, 1)) const {
        return terrainGrid.renderToRGB(width, height, backgroundColor);
    }
    
    // Render to RGBA image
    std::vector<uint8_t> renderToRGBA(int width, int height, 
                                     const Vec4& backgroundColor = Vec4(0, 0, 0, 1)) const {
        return terrainGrid.renderToRGBA(width, height, backgroundColor);
    }
    
    // Export terrain data as heightmap (grayscale)
    Grid2 exportHeightmap() const {
        Grid2 heightmap(gridWidth * gridHeight);
        
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                int index = y * gridWidth + x;
                float elevation = terrainGrid.colors[index].x;
                heightmap.positions[index] = Vec2(x, y);
                heightmap.colors[index] = Vec4(elevation, elevation, elevation, 1.0f);
            }
        }
        
        return heightmap;
    }
    
    // Generate random seed and regenerate
    void randomizeSeed() {
        std::random_device rd;
        setSeed(rd());
    }
    
    // Reset all parameters to default
    void reset() {
        scale = 4.0f;
        octaves = 4;
        persistence = 0.5f;
        lacunarity = 2.0f;
        elevationMultiplier = 1.0f;
        waterLevel = 0.3f;
        landColor = Vec4(0.2f, 0.8f, 0.2f, 1.0f);
        waterColor = Vec4(0.2f, 0.3f, 0.8f, 1.0f);
        generateTerrain();
    }
    
    // Get terrain statistics
    struct TerrainStats {
        float minElevation;
        float maxElevation;
        float averageElevation;
        float landPercentage;
        int landArea;
        int waterArea;
    };
    
    TerrainStats getTerrainStats() const {
        TerrainStats stats = {1.0f, 0.0f, 0.0f, 0.0f, 0, 0};
        float totalElevation = 0.0f;
        
        for (const auto& color : terrainGrid.colors) {
            float elevation = color.x;
            stats.minElevation = std::min(stats.minElevation, elevation);
            stats.maxElevation = std::max(stats.maxElevation, elevation);
            totalElevation += elevation;
            
            if (elevation > waterLevel) {
                stats.landArea++;
            } else {
                stats.waterArea++;
            }
        }
        
        stats.averageElevation = totalElevation / terrainGrid.colors.size();
        stats.landPercentage = static_cast<float>(stats.landArea) / 
                              (stats.landArea + stats.waterArea) * 100.0f;
        
        return stats;
    }

private:
    void applyTerrainColors() {
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                int index = y * gridWidth + x;
                float elevation = terrainGrid.colors[index].x;
                
                // Apply water level and color based on elevation
                if (elevation <= waterLevel) {
                    // Water - optionally add depth variation
                    float depth = (waterLevel - elevation) / waterLevel;
                    Vec4 water = waterColor * (0.7f + 0.3f * depth);
                    water.w = 1.0f; // Ensure full alpha
                    terrainGrid.colors[index] = water;
                } else {
                    // Land - optionally add elevation-based color variation
                    float height = (elevation - waterLevel) / (1.0f - waterLevel);
                    Vec4 land = landColor * (0.8f + 0.2f * height);
                    land.w = 1.0f; // Ensure full alpha
                    terrainGrid.colors[index] = land;
                }
            }
        }
    }
    
    void applyElevationModification() {
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                int index = y * gridWidth + x;
                float originalElevation = terrainGrid.colors[index].x;
                
                // Apply elevation multiplier with clamping
                float newElevation = std::clamp(originalElevation * elevationMultiplier, 0.0f, 1.0f);
                terrainGrid.colors[index].x = newElevation;
                terrainGrid.colors[index].y = newElevation; // Keep grayscale for heightmap
                terrainGrid.colors[index].z = newElevation;
            }
        }
        
        applyTerrainColors();
    }
};

#endif