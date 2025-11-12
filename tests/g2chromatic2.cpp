#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "../util/grid/grid22.hpp"
#include "../util/output/aviwriter.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/timing_decorator.cpp"

struct AnimationConfig {
    int width = 1024;
    int height = 1024;
    int totalFrames = 480;
    float fps = 30.0f;
    int numSeeds = 1;
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

std::vector<std::pair<Vec2, Vec4>> pickSeeds(Grid2 grid, AnimationConfig config) {
    TIME_FUNCTION;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> xDist(0, config.width - 1);
    std::uniform_int_distribution<> yDist(0, config.height - 1);
    std::uniform_real_distribution<> colorDist(0.2f, 0.8f);

    std::vector<std::pair<Vec2, Vec4>> seeds;

    for (int i = 0; i < config.numSeeds; ++i) {
        Vec2 point(xDist(gen), yDist(gen));
        Vec4 color(colorDist(gen), colorDist(gen), colorDist(gen), colorDist(gen));
        seeds.push_back(std::make_pair(point, color));
        // Or in C++17 and later, you can use:
        // seeds.push_back({point, color});
    }
    for (int i = 0; i < seeds.size(); ++i) {
        size_t id = grid.getPositionVec(seeds[i].first);
        grid.setColor(id,seeds[i].second);
    }
    return seeds;
}

void expandPixel(Grid2& grid, AnimationConfig config, std::vector<std::pair<Vec2, Vec4>> seeds) {
    TIME_FUNCTION;
    for (int i = 0; i < seeds.size(); ++i) {
        size_t id = grid.getPositionVec(seeds[i].first);
        std::vector<size_t> neighbors = grid.getNeighbors(id);
        for (int j = 0; j < neighbors.size(); ++j) {
            size_t neighbor = neighbors[j];
            Vec4 neighborColor = grid.getColor(neighbor);
            Vec4 newcolor = (neighborColor - seeds[i].second) / float(config.width + config.height - 2);
            grid.setColor(neighbor, newcolor);
        }
    }
}

bool exportavi(std::vector<std::vector<uint8_t>> frames, AnimationConfig config) {
    TIME_FUNCTION;
    bool success = AVIWriter::saveAVI("output/g2chromatic.avi", frames, config.width, config.height);
    return success;
}

int main() {
    AnimationConfig config;
    
    Grid2 grid = setup(config);
    grid.updateNeighborMap();
    Preview(grid);
    std::vector<std::pair<Vec2, Vec4>> seeds = pickSeeds(grid,config);
    std::vector<std::vector<uint8_t>> frames;
    for (int i = 0; i < config.totalFrames; ++i){
        std::cout << "Processing frame " << i + 1 << "/" << config.totalFrames << std::endl;
        expandPixel(grid,config,seeds);
        int width;
        int height;
        std::vector<uint8_t> frame;
        grid.getGridAsBGR(width,height,frame);
        frames.push_back(frame);
    }
    

    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}