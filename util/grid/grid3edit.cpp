#include "grid3eigen.hpp"

namespace Grid {

static const std::array<Vec3, 6> kFaceDirs = {
    Vec3(1, 0, 0), Vec3(-1, 0, 0),
    Vec3(0, 1, 0), Vec3(0, -1, 0),
    Vec3(0, 0, 1), Vec3(0, 0, -1)
};

template<typename T>
void Octree<T>::pixelToRay(const Camera& cam, int px, int py, int width, int height, Vec3& outOrigin, Vec3& outDir) const {
    const float aspect = static_cast<float>(width) / height;
    const float tanHalfFov = std::tan(cam.fovRad() * 0.5f);
    const float tanfovx = tanHalfFov * aspect;
    const float tanfovy = tanHalfFov;
    Vec3 dir = cam.direction.normalized();
    Vec3 right = cam.right();
    Vec3 up = cam.up.normalized();
    float sx = (2.0f * (px + 0.5f) / width - 1.0f) * tanfovx;
    float sy = (1.0f - 2.0f * (py + 0.5f) / height) * tanfovy;
    outOrigin = cam.origin;
    outDir = (dir + right * sx + up * sy).normalized();
}

template<typename T>
bool Octree<T>::raycastFromCamera(const Camera& cam, int px, int py, int width, int height, RayHit& hit, float maxDist) {
    Vec3 origin, dir;
    pixelToRay(cam, px, py, width, height, origin, dir);
    return raycast(origin, dir, maxDist, hit);
}

template<typename T>
bool Octree<T>::addVoxelAtPixel(const Camera& cam, int px, int py, int width, int height, Vec3 color, int objectId) {
    RayHit hit;
    if (!raycastFromCamera(cam, px, py, width, height, hit)) return false;
    auto& src = hit.node;
    int targetObj = (objectId == -2) ? src->objectId : objectId;
    Vec3 newPos = src->position + hit.normal * src->size;
    if (find(newPos, targetObj, src->size * 0.25f)) return false;

    RenderMaterial rmat = getRenderMaterial(src->renderMatIdx);
    PhysicsMaterial_ pmat;
    if (auto obj = getObject(src->objectId)) pmat = obj->getPhysicsMaterial(src->physMatIdx);
    Vec3 emit = rmat.emittanceRGB();
    float transmission = 1.0f - src->color.w();
    return insert(T{}, newPos, true, color, src->size, true, targetObj, emit.x(), rmat.roughness,
                  rmat.metallic, transmission, rmat.iorGreen(), rmat.absorption, pmat.type, pmat.mass);
}

template<typename T>
bool Octree<T>::removeVoxelAtPixel(const Camera& cam, int px, int py, int width, int height) {
    RayHit hit;
    if (!raycastFromCamera(cam, px, py, width, height, hit)) return false;
    return remove(hit.node->position, hit.node->size * 0.25f);
}

template<typename T>
bool Octree<T>::roundObject(int objectId, int passes) {
    if (passes < 1) return false;
    bool ok = false;
    for (int i = 0; i < passes; ++i) ok = smoothObject(objectId);
    return ok;
}

template<typename T>
bool Octree<T>::chamferObject(int objectId) {
    return bevelObject(objectId, 1);
}

template<typename T>
bool Octree<T>::bevelObject(int objectId, int layers) {
    if (root_ == INVALID_IDX || layers < 1) return false;
    bool any = false;
    for (int pass = 0; pass < layers; ++pass) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(objectId, nodes);
        if (nodes.empty()) break;

        std::unordered_set<std::shared_ptr<NodeData>> toRemove;
        std::vector<std::shared_ptr<NodeData>> toRemoveVec;
        for (auto& n : nodes) {
            int exposed = 0;
            for (const auto& d : kFaceDirs) {
                auto nb = find(n->position + d * n->size, objectId, n->size * 0.25f);
                if (!nb || !nb->isActive()) exposed++;
            }
            if (exposed >= 2) {
                toRemove.insert(n);
                toRemoveVec.push_back(n);
            }
        }
        if (toRemoveVec.empty()) break;
        BoundingBox b = getNodesBounds(toRemoveVec);
        int depth = 0;
        uint32_t start = getHighestCommonNode(b, root_, 0, depth);
        size -= removeSpecificNodesBatchRecursive(start, toRemove);
        any = true;
    }
    return any;
}

template<typename T>
bool Octree<T>::filletObject(int objectId, int layers) {
    if (root_ == INVALID_IDX || layers < 1) return false;
    auto obj = getObject(objectId);
    if (!obj) return false;
    Vec3 color(0.5f, 0.5f, 0.5f);
    RenderMaterial rmat;
    PhysicsMaterial_ pmat;
    bool any = false;
    for (int pass = 0; pass < layers; ++pass) {
        std::vector<std::shared_ptr<NodeData>> nodes;
        collectNodesByObjectId(objectId, nodes);
        if (nodes.empty()) break;

        std::vector<Vec3> toAdd;
        float step = nodes[0]->size;
        rmat = getRenderMaterial(nodes[0]->renderMatIdx);
        pmat = obj->getPhysicsMaterial(nodes[0]->physMatIdx);
        color = nodes[0]->color.template head<3>();

        for (auto& n : nodes) {
            for (const auto& d : kFaceDirs) {
                Vec3 cand = n->position + d * step;
                if (find(cand, objectId, step * 0.25f)) continue;
                int solid = 0;
                for (const auto& e : kFaceDirs) {
                    auto nb = find(cand + e * step, objectId, step * 0.25f);
                    if (nb && nb->isActive()) solid++;
                }
                if (solid >= 3) toAdd.push_back(cand);
            }
        }
        if (toAdd.empty()) break;
        Vec3 emit = rmat.emittanceRGB();
        for (const auto& p : toAdd) {
            if (find(p, objectId, step * 0.25f)) continue;
            insert(T{}, p, true, color, step, true, objectId, emit.x(), rmat.roughness,
                   rmat.metallic, 0.0f, rmat.iorGreen(), rmat.absorption, pmat.type, pmat.mass);
        }
        any = true;
    }
    return any;
}

template<typename T>
bool Octree<T>::extrudeFace(const Camera& cam, int px, int py, int width, int height, int layers) {
    if (layers < 1) return false;
    RayHit hit;
    if (!raycastFromCamera(cam, px, py, width, height, hit)) return false;
    auto& src = hit.node;
    int objectId = src->objectId;
    float step = src->size;
    RenderMaterial rmat = getRenderMaterial(src->renderMatIdx);
    PhysicsMaterial_ pmat;
    if (auto obj = getObject(objectId)) pmat = obj->getPhysicsMaterial(src->physMatIdx);
    Vec3 color = src->color.template head<3>();
    Vec3 emit = rmat.emittanceRGB();
    float transmission = 1.0f - src->color.w();

    bool any = false;
    for (int l = 1; l <= layers; ++l) {
        Vec3 p = src->position + hit.normal * (step * l);
        if (find(p, objectId, step * 0.25f)) continue;
        if (insert(T{}, p, true, color, step, true, objectId, emit.x(), rmat.roughness,
                   rmat.metallic, transmission, rmat.iorGreen(), rmat.absorption, pmat.type, pmat.mass))
            any = true;
    }
    return any;
}

template<typename T>
bool Octree<T>::flattenSurface(const Camera& cam, int px, int py, int width, int height, float radius) {
    RayHit hit;
    if (!raycastFromCamera(cam, px, py, width, height, hit)) return false;
    Vec3 planePoint = hit.node->position;
    Vec3 normal = hit.normal.normalized();

    auto nodes = findInRadius(hit.node->position, radius, hit.node->objectId);
    if (nodes.empty()) return false;

    std::unordered_set<std::shared_ptr<NodeData>> toRemove;
    std::vector<std::shared_ptr<NodeData>> toRemoveVec;
    for (auto& n : nodes) {
        if (normal.dot(n->position - planePoint) > n->size * 0.5f) {
            toRemove.insert(n);
            toRemoveVec.push_back(n);
        }
    }
    if (toRemoveVec.empty()) return false;
    BoundingBox b = getNodesBounds(toRemoveVec);
    int depth = 0;
    uint32_t start = getHighestCommonNode(b, root_, 0, depth);
    size -= removeSpecificNodesBatchRecursive(start, toRemove);
    return true;
}

}