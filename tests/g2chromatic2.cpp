#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <unordered_set>
#include "../util/grid/grid22.hpp"
#include "../util/output/aviwriter.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/timing_decorator.cpp"

struct AnimationConfig {
    int width = 128;
    int height = 128;
    int totalFrames = 480;
    float fps = 30.0f;
    int numSeeds = 8;
};

Grid2 setup(AnimationConfig config) {
    TIME_FUNCTION;
    Grid2 grid;
    std::vector<Vec2> pos;
    std::vector<Vec4> colors;
    std::vector<float> sizes;
    for (int y = 0; y < config.height; ++y) {
        for (int x = 0; x < config.width; ++x) {
            float gradient = (x + y) / float(config.width + config.height - 2);
            pos.push_back(Vec2(x,y));
            colors.push_back(Vec4(gradient, gradient, gradient, 1.0f));
            sizes.push_back(1.0f);
        }
    }
    grid.bulkAddObjects(pos,colors,sizes);
    return grid;
}

void Preview(Grid2 grid) {
    TIME_FUNCTION;
    int width;
    int height;
    std::vector<uint8_t> rgbData;

    grid.getGridAsRGB(width,height,rgbData);
    bool success = BMPWriter::saveBMP("output/grayscalesource.bmp", rgbData, width, height);
}

std::vector<std::tuple<size_t, Vec2, Vec4>> pickSeeds(Grid2 grid, AnimationConfig config) {
    TIME_FUNCTION;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> xDist(0, config.width - 1);
    std::uniform_int_distribution<> yDist(0, config.height - 1);
    std::uniform_real_distribution<> colorDist(0.2f, 0.8f);

    std::vector<std::tuple<size_t, Vec2, Vec4>> seeds;

    for (int i = 0; i < config.numSeeds; ++i) {
        Vec2 point(xDist(gen), yDist(gen));
        Vec4 color(colorDist(gen), colorDist(gen), colorDist(gen), 1.0f);
        size_t id = grid.getPositionVec(point);
        grid.setColor(id, color);
        seeds.push_back(std::make_tuple(id,point, color));
    }
    return seeds;
}

void expandPixel(Grid2& grid, AnimationConfig config, std::vector<std::tuple<size_t, Vec2, Vec4>>& seeds, std::unordered_set<size_t>& visited) {
    TIME_FUNCTION;
    std::vector<std::tuple<size_t, Vec2, Vec4>> newseeds; 
    for (const std::tuple<size_t, Vec2, Vec4>& seed : seeds) {
        size_t id = std::get<0>(seed);
        Vec2 seedPOS = std::get<1>(seed);
        Vec4 seedColor = std::get<2>(seed);
        std::vector<size_t> neighbors = grid.getNeighbors(id);
        for (size_t neighbor : neighbors) {
            if (visited.find(neighbor) != visited.end()) {
                continue; // Skip already processed neighbors
            }
            visited.insert(neighbor);
            Vec2 neipos = grid.getPositionID(neighbor);
            Vec4 neighborColor = grid.getColor(neighbor);
            float distance = seedPOS.distance(neipos);
            float angle = seedPOS.angleTo(neipos);
            Vec4 newcolor = Vec4(neighborColor.r * ((1.1f + angle) * std::sin(seedColor.r)),
                neighborColor.g * ((1.1f + angle) * std::sin(seedColor.g)),
                neighborColor.b * ((1.1f + angle) * std::sin(seedColor.b)),
                1.0f
            );
            grid.setColor(neighbor, newcolor);
            newseeds.push_back({neighbor, neipos,newcolor});
        }
    }
    seeds = newseeds;
}

bool exportavi(std::vector<std::vector<uint8_t>> frames, AnimationConfig config) {
    TIME_FUNCTION;
    std::string filename = "output/chromatic_transformation.avi";
    
    std::cout << "Frame count: " << frames.size() << std::endl;
    std::cout << "Frame size: " << (frames.empty() ? 0 : frames[0].size()) << std::endl;
    std::cout << "Width: " << config.width << ", Height: " << config.height << std::endl;
    
    std::filesystem::path dir = "output";
    if (!std::filesystem::exists(dir)) {
        if (!std::filesystem::create_directories(dir)) {
            std::cout << "Failed to create output directory!" << std::endl;
            return false;
        }
    }
    
    bool success = AVIWriter::saveAVI(filename, frames, config.width+1, config.height+1, config.fps);
    
    if (success) {
        // Check if file actually exists
        if (std::filesystem::exists(filename)) {
            auto file_size = std::filesystem::file_size(filename);
        } 
    } else {
        std::cout << "Failed to save AVI file!" << std::endl;
    }
    
    return success;
}

int main() {
    AnimationConfig config;
    
    Grid2 grid = setup(config);
    grid.updateNeighborMap();
    Preview(grid);
    std::vector<std::tuple<size_t, Vec2, Vec4>> seeds = pickSeeds(grid,config);
    std::vector<std::vector<uint8_t>> frames;

    //memory aid
    std::unordered_set<size_t> visited;

    for (int i = 0; i < config.totalFrames; ++i){
        std::cout << "Processing frame " << i + 1 << "/" << config.totalFrames << std::endl;
        expandPixel(grid,config,seeds, visited);
        int width;
        int height;
        std::vector<uint8_t> frame;
        grid.getGridAsBGR(width,height,frame);
        frames.push_back(frame);
    }
    
    exportavi(frames,config);
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}