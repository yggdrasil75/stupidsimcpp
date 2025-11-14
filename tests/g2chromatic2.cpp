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
#include "../util/output/frame.hpp"
#include "../util/output/video.hpp"

struct AnimationConfig {
    int width = 1024;
    int height = 1024;
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
    seeds = std::move(newseeds);
}

int main() {
    AnimationConfig config;
    
    Grid2 grid = setup(config);
    Preview(grid);
    std::vector<std::tuple<size_t, Vec2, Vec4>> seeds = pickSeeds(grid,config);
    
    // Create video object with BGR channels and configured FPS
    video vid(config.width+1, config.height+1, {'B', 'G', 'R'}, config.fps, true);

    for (int i = 0; i < config.totalFrames; ++i){
        std::cout << "Processing frame " << i + 1 << "/" << config.totalFrames << std::endl;
        expandPixel(grid,config,seeds);
        
        frame outputFrame;
        grid.getGridAsFrame(outputFrame, {'B', 'G', 'R'});
        
        // Add frame to video (this will compress it using RLE + differential encoding)
        vid.add_frame(outputFrame);
        
        // Optional: Print compression stats every 50 frames
        // if ((i + 1) % 50 == 0) {
        //     auto stats = vid.get_compression_stats();
        //     std::cout << "Frame " << (i + 1) << " - Compression ratio: " << stats.overall_ratio 
        //               << ", Total compressed: " << stats.total_compressed_bytes << " bytes" << std::endl;
        // }
    }
    
    // Use the video-based AVIWriter overload
    bool success = AVIWriter::saveAVI("output/chromatic_transformation.avi", vid, config.fps);
    
    if (success) {
        // Print final compression statistics
        auto final_stats = vid.get_compression_stats();
        std::cout << "Successfully saved AVI with " << vid.frame_count() << " frames" << std::endl;
        std::cout << "Final compression statistics:" << std::endl;
        std::cout << "  Total frames: " << final_stats.total_frames << std::endl;
        std::cout << "  Video duration: " << final_stats.video_duration << " seconds" << std::endl;
        std::cout << "  Uncompressed size: " << final_stats.total_uncompressed_bytes << " bytes" << std::endl;
        std::cout << "  Compressed size: " << final_stats.total_compressed_bytes << " bytes" << std::endl;
        std::cout << "  Overall compression ratio: " << final_stats.overall_ratio << std::endl;
        std::cout << "  Average frame compression ratio: " << final_stats.average_frame_ratio << std::endl;
        
        // Optional: Save compressed video data for later use
        auto serialized_data = vid.serialize();
        std::cout << "Serialized video data size: " << serialized_data.size() << " bytes" << std::endl;
    } else {
        std::cout << "Failed to save AVI file!" << std::endl;
    }
    
    FunctionTimer::printStats(FunctionTimer::Mode::ENHANCED);
    return 0;
}