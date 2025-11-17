#ifndef SPRITE2_HPP
#define SPRITE2_HPP

#include "grid2.hpp"
#include "../output/frame.hpp"

class SpriteMap2 : public Grid2 {
private:

    // id, sprite
    //std::unordered_map<size_t, std::shared_ptr<Grid2>> Sprites;
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

    

};

#endif