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
    int width = 2048;
    int height = 2048;
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

void expandPixel(Grid2& grid, AnimationConfig config, std::vector<std::tuple<size_t, Vec2, Vec4>>& seeds) {
    TIME_FUNCTION;
    std::vector<std::tuple<size_t, Vec2, Vec4>> newseeds; 

    std::unordered_set<size_t> visitedThisFrame; 
    for (const auto& seed : seeds) {
        visitedThisFrame.insert(std::get<0>(seed));
    }

    //#pragma omp parallel for
    for (const std::tuple<size_t, Vec2, Vec4>& seed : seeds) {
        size_t id = std::get<0>(seed);
        Vec2 seedPOS = std::get<1>(seed);
        Vec4 seedColor = std::get<2>(seed);
        std::vector<size_t> neighbors = grid.getNeighbors(id);
        for (size_t neighbor : neighbors) {
            if (visitedThisFrame.count(neighbor)) {
                continue;
            }
            visitedThisFrame.insert(neighbor);

            Vec2 neipos = grid.getPositionID(neighbor);
            Vec4 neighborColor = grid.getColor(neighbor);
            float distance = seedPOS.distance(neipos);
            float angle = seedPOS.directionTo(neipos);

            float normalizedAngle = (angle + M_PI) / (2.0f * M_PI);
            float blendFactor = 0.3f + 0.4f * std::sin(normalizedAngle * 2.0f * M_PI);
            blendFactor = std::clamp(blendFactor, 0.1f, 0.9f);
            
            Vec4 newcolor = Vec4(
                seedColor.r * blendFactor + neighborColor.r * (1.0f - blendFactor),
                seedColor.g * (1.0f - blendFactor) + neighborColor.g * blendFactor,
                seedColor.b * (0.5f + 0.5f * std::sin(normalizedAngle * 4.0f * M_PI)),
                1.0f
            );
            
            newcolor = newcolor.clamp(0.0f, 1.0f);

            grid.setColor(neighbor, newcolor);
            newseeds.emplace_back(neighbor, neipos, newcolor);
        }
    }
    seeds.clear();
    seeds.shrink_to_fit();
    seeds = std::move(newseeds);
}

//bool exportavi(std::vector<std::vector<uint8_t>> frames, AnimationConfig config) {
bool exportavi(std::vector<frame> frames, AnimationConfig config) {
    TIME_FUNCTION;
    std::string filename = "output/chromatic_transformation.avi";
    
    std::cout << "Frame count: " << frames.size() << std::endl;
    
    // Log compression statistics for all frames
    std::cout << "\n=== Frame Compression Statistics ===" << std::endl;
    size_t totalOriginalSize = 0;
    size_t totalCompressedSize = 0;
    
    for (int i = 0; i < frames.size(); ++i) {    
        totalOriginalSize += frames[i].getSourceSize();
        totalCompressedSize += frames[i].getTotalCompressedSize();
    }
    
    double overallRatio = static_cast<double>(totalOriginalSize) / totalCompressedSize;
    double overallSavings = (1.0 - 1.0/overallRatio) * 100.0;
    
    std::cout << "\n=== Overall Compression Summary ===" << std::endl;
    std::cout << "Total frames: " << frames.size() << std::endl;
    std::cout << "Compressed frames: " << frames.size() << std::endl;
    std::cout << "Total original size: " << totalOriginalSize << " bytes (" 
                << std::fixed << std::setprecision(2) << (totalOriginalSize / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "Total compressed size: " << totalCompressedSize << " bytes (" 
                << std::fixed << std::setprecision(2) << (totalCompressedSize / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "Overall compression ratio: " << std::fixed << std::setprecision(2) << overallRatio << ":1" << std::endl;
    std::cout << "Overall space savings: " << std::fixed << std::setprecision(1) << overallSavings << "%" << std::endl;
    
    std::filesystem::path dir = "output";
    if (!std::filesystem::exists(dir)) {
        if (!std::filesystem::create_directories(dir)) {
            std::cout << "Failed to create output directory!" << std::endl;
            return false;
        }
    }
    
    bool success = AVIWriter::saveAVIFromCompressedFrames(filename, frames, frames[0].getWidth(), frames[0].getHeight(), config.fps);
    
    if (success) {
        // Check if file actually exists
        if (std::filesystem::exists(filename)) {
            auto file_size = std::filesystem::file_size(filename);
            std::cout << "\nAVI file created successfully: " << filename 
                      << " (" << file_size << " bytes, " 
                      << std::fixed << std::setprecision(2) << (file_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
        } 
    } else {
        std::cout << "Failed to save AVI file!" << std::endl;
    }
    
    return success;
}

int main() {
    AnimationConfig config;
    
    Grid2 grid = setup(config);
    Preview(grid);
    std::vector<std::tuple<size_t, Vec2, Vec4>> seeds = pickSeeds(grid,config);
    std::vector<frame> frames;

    for (int i = 0; i < config.totalFrames; ++i){
        expandPixel(grid,config,seeds);
        
        // Print compression info for this frame
        if (i % 10 == 0 ) {
            frame bgrframe;
            std::cout << "Processing frame " << i + 1 << "/" << config.totalFrames << std::endl;
            bgrframe = grid.getGridAsFrame(frame::colormap::BGR);
            bgrframe.printCompressionStats();
            //(bgrframe, i + 1);
            frames.push_back(bgrframe);
        }

    }
    
    exportavi(frames,config);
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}