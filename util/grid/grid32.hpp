#ifndef GRID_HPP
#define GRID_HPP

#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../noise/pnoise2.hpp"
#include <unordered_map>
#include <array>

//template <typename DataType, int Dimension>
class Node {
    //static_assert(Dimension == 2 || Dimension == 3, "Dimensions must be 2 or 3");
private:
public:
    NodeType type;
    Vec3f minBounds;
    Vec3f maxBounds;
    Vec4ui8 color;
    enum NodeType {
        BRANCH,
        LEAF
    };
    std::array<Node, 8> children;
    Node(const Vec3f min, const Vec3f max, NodeType t = LEAF) : type(t), minBounds(min), maxBounds(max), color(0,0,0,0) {}

    Vec3f getCenter() const {
        return (minBounds + maxBounds) * 0.5f;
    }

    int getChildIndex(const Vec3f& pos) const {
        Vec3f c = getCenter();
        int index = 0;
        if (pos.x >= c.x) index |= 1;
        if (pos.y >= c.y) index |= 2;
        if (pos.z >= c.z) index |= 4;
        return index;
    }

    std::pair<Vec3f, Vec3f> getChildBounds(int childIndex) const {
        Vec3f c = getCenter();
        Vec3f cMin = minBounds;
        Vec3f cMax = maxBounds;
        if (childIndex & 1) cMin.x = c.x;
        else cMax.x = c.x;

        if (childIndex & 2) cMin.y = c.y;
        else cMax.y = c.y;
        
        if (childIndex & 4) cMin.z = c.z;
        else cMax.z = c.z;

        return {cMin, cMax};
    }

};

class Grid {
private:
    Node root;
    Vec4ui8 DefaultBackgroundColor;
    PNoise2 noisegen;
    std::unordered_map<Vec3f, Node, Vec3f::Hash> Cache;
public:
    Grid() : {
        root = Node(Vec3f(0), Vec3f(0), Node::NodeType::BRANCH);
    };

    size_t insertVoxel(const Vec3f& pos, const Vec4ui8& color) {
        if (!contains(pos)) {
            return -1;
        }
        return insertRecusive(root.get(), pos, color, 0);
    }

    void removeVoxel(const Vec3f& pos) {
        bool removed = removeRecursive(root.get(), pos, 0);
        if (removed) {
            Cache.erase(pos);
        }
    }

    Vec4ui8 getVoxel(const Vec3f& pos) const {
        Node* node = findNode(root.get(), pos, 0);
        if (node && node->isLeaf()) {
            return node->color;
        }
        return DefaultBackgroundColor;
    }

    bool hasVoxel(const Vec3f& pos) const {
        Node* node = findNode(root.get(), pos, 0);
        return node && node->isLeaf();
    }

    bool rayIntersect(const Ray3& ray, Vec3f& hitPos, Vec4ui8& hitColor) const {
        return rayintersectRecursive(root.get(), ray, hitPos, hitColor);
    }

    std::unordered_map<Vec3f, Vec4ui8> queryRegion(const Vec3f& min, const Vec3f& max) const {
        std::unordered_map<Vec3f, Vec4ui8> result;
        queryRegionRecuse(root.get(), min, max, result);
        return result;
    }

    Grid& noiseGen(const Vec3f& min, const Vec3f& max, float minc = 0.1f, float maxc = 1.0f, bool genColor = true, int noiseMod = 42) {
        TIME_FUNCTION;
        noisegen = PNoise2(noiseMod);
        std::vector<Vec3f> poses;
        std::vector<Vec4ui8> colors;
        
        size_t estimatedSize = (max.x - min.x) * (max.y - min.y) * (max.z - min.z) * (maxc - minc);
        poses.reserve(estimatedSize);
        colors.reserve(estimatedSize);
        
    }
};

#endif
