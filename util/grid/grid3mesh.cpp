#include "grid3eigen.hpp"
#include "../timing_decorator.hpp"

namespace Grid {

template<typename T, typename GasT, typename IndexType>
float Octree<T, GasT, IndexType>::pickCellSize(const MeshVoxels& voxels) {
    float best = 0.0f;
    for (const auto& v : voxels) {
        float s = static_cast<float>(v->size);
        if (s > 1e-6f && (best == 0.0f || s < best)) best = s;
    }
    return best > 1e-6f ? best : 1.0f;
}

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::latticeBounds(const MeshVoxels& voxels, float cellSize,
        PointType& origin, Eigen::Vector3i& outMin, Eigen::Vector3i& outMax) {
    PointType mn(1e30f, 1e30f, 1e30f);
    PointType mx(-1e30f, -1e30f, -1e30f);
    for (const auto& v : voxels) { mn = mn.cwiseMin(v->position); mx = mx.cwiseMax(v->position); }
    origin = mn;
    const float inv = 1.0f / cellSize;
    PointType rel = (mx - mn) * inv;
    outMin = Eigen::Vector3i::Zero();
    outMax = Eigen::Vector3i(static_cast<int>(std::lround(rel.x())),
                             static_cast<int>(std::lround(rel.y())),
                             static_cast<int>(std::lround(rel.z())));
}

template<typename T, typename GasT, typename IndexType>
std::shared_ptr<NodeData_<T, IndexType>>
Octree<T, GasT, IndexType>::meshVoxelAt(const MeshGrid& g, const Eigen::Vector3i& k,
                                        int objectId, OctreeNode* ancestor) {
    PointType world = g.origin + k.cast<float>() * g.cellSize;
    return find(world, objectId, g.cellSize * 0.25f, ancestor);
}

template<typename T, typename GasT, typename IndexType>
typename Octree<T, GasT, IndexType>::OctreeNode*
Octree<T, GasT, IndexType>::meshAncestor(const MeshVoxels& voxels) {
    if (voxels.empty()) return root_.get();
    std::vector<PointType> positions;
    positions.reserve(voxels.size());
    for (const auto& v : voxels) positions.push_back(v->position);
    int depth = 0;
    OctreeNode* anc = getHighestCommonNode(positions, root_.get(), depth);
    return anc ? anc : root_.get();
}

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::mesh_naive(GridObject& obj, const MeshVoxels& voxels) {
    for (const auto& pt : voxels) {
        float h = static_cast<float>(pt->size) * 0.5f;
        const PointType& o = pt->position;
        size_t base = obj.objMesh.vertices.size();
        static const float corners[8][3] = {
            {-1,-1,-1},{+1,-1,-1},{+1,+1,-1},{-1,+1,-1},
            {-1,-1,+1},{+1,-1,+1},{+1,+1,+1},{-1,+1,+1}
        };
        for (auto& c : corners)
            obj.objMesh.vertices.push_back({pt->colorIdx, o + PointType(c[0]*h, c[1]*h, c[2]*h)});
        static const int faces[6][4] = {
            {0,1,2,3},{4,5,6,7},{0,1,5,4},
            {2,3,7,6},{0,3,7,4},{1,2,6,5}
        };
        for (auto& f : faces) {
            obj.objMesh.tris.push_back({base+f[0], base+f[1], base+f[2]});
            obj.objMesh.tris.push_back({base+f[0], base+f[2], base+f[3]});
        }
    }
}

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::mesh_surfaceNet(GridObject& obj, const MeshVoxels& voxels) {
    if (voxels.empty()) return;

    float cs = pickCellSize(voxels);
    PointType origin; Eigen::Vector3i mn, mx;
    latticeBounds(voxels, cs, origin, mn, mx);
    MeshGrid g{origin, cs};
    OctreeNode* anc = meshAncestor(voxels);
    auto occ = [&](const Eigen::Vector3i& k){ return meshVoxelAt(g, k, obj.id, anc) != nullptr; };

    for (const auto& pt : voxels) {
        const Eigen::Vector3f c = pt->position;
        const float h = static_cast<float>(pt->size) * 0.5f;
        Eigen::Vector3f relf = (c - origin) / cs;
        Eigen::Vector3i key(static_cast<int>(std::lround(relf.x())),
                            static_cast<int>(std::lround(relf.y())),
                            static_cast<int>(std::lround(relf.z())));
        for (const auto& fd : kFaces) {
            Eigen::Vector3i fn = meshFaceNormal(fd);
            if (occ(key + fn)) continue;

            Eigen::Vector3f n = fn.cast<float>();
            int a0 = (fd.axis + 1) % 3;
            int a1 = (fd.axis + 2) % 3;
            Eigen::Vector3f u = Eigen::Vector3f::Unit(a0) * h;
            Eigen::Vector3f w = Eigen::Vector3f::Unit(a1) * h;
            Eigen::Vector3f fc = c + n * h;

            Eigen::Vector3f p0 = fc - u - w;
            Eigen::Vector3f p1 = fc + u - w;
            Eigen::Vector3f p2 = fc + u + w;
            Eigen::Vector3f p3 = fc - u + w;

            uint32_t colorIdx = static_cast<uint32_t>(pt->colorIdx);
            size_t i0 = meshAddVertex(obj, colorIdx, p0);
            size_t i1 = meshAddVertex(obj, colorIdx, p1);
            size_t i2 = meshAddVertex(obj, colorIdx, p2);
            size_t i3 = meshAddVertex(obj, colorIdx, p3);
            if (fd.sign > 0) meshAddQuad(obj, i0, i1, i2, i3);
            else             meshAddQuad(obj, i0, i3, i2, i1);
        }
    }
}

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::mesh_greedy(GridObject& obj, const MeshVoxels& voxels) {
    if (voxels.empty()) return;

    float cs = pickCellSize(voxels);
    Eigen::Vector3f origin;
    Eigen::Vector3i mn, mx;
    latticeBounds(voxels, cs, origin, mn, mx);
    MeshGrid g{origin, cs};
    OctreeNode* anc = meshAncestor(voxels);
    auto occ = [&](const Eigen::Vector3i& k){ return meshVoxelAt(g, k, obj.id, anc) != nullptr; };
    const float h = cs * 0.5f;

    struct VKey { Eigen::Vector3i k; uint32_t color; };
    std::vector<VKey> vk;
    vk.reserve(voxels.size());
    for (const auto& v : voxels) {
        Eigen::Vector3f relf = (v->position - origin) / cs;
        vk.push_back({Eigen::Vector3i(static_cast<int>(std::lround(relf.x())),
                                      static_cast<int>(std::lround(relf.y())),
                                      static_cast<int>(std::lround(relf.z()))),
                      static_cast<uint32_t>(v->colorIdx)});
    }

    for (int axis = 0; axis < 3; ++axis) {
        int u = (axis + 1) % 3;
        int v = (axis + 2) % 3;
        int wU = mx[u] - mn[u] + 1;
        int wV = mx[v] - mn[v] + 1;
        if (wU <= 0 || wV <= 0) continue;

        std::unordered_map<int, std::vector<const VKey*>> buckets;
        buckets.reserve(static_cast<size_t>(mx[axis] - mn[axis] + 1));
        for (const auto& e : vk) buckets[e.k[axis]].push_back(&e);

        for (int sign = -1; sign <= 1; sign += 2) {
            Eigen::Vector3i nrm(0, 0, 0); nrm[axis] = sign;

            for (auto& [s, cells] : buckets) {
                std::vector<uint32_t> mask(static_cast<size_t>(wU) * wV, 0);
                for (const VKey* e : cells) {
                    if (occ(e->k + nrm)) continue;
                    int iu = e->k[u] - mn[u];
                    int jv = e->k[v] - mn[v];
                    mask[static_cast<size_t>(jv) * wU + iu] = e->color + 1;
                }

                for (int jv = 0; jv < wV; ++jv) {
                    for (int iu = 0; iu < wU; ) {
                        uint32_t m = mask[static_cast<size_t>(jv) * wU + iu];
                        if (m == 0) { ++iu; continue; }

                        int wdt = 1;
                        while (iu + wdt < wU &&
                               mask[static_cast<size_t>(jv) * wU + iu + wdt] == m) ++wdt;

                        int hgt = 1;
                        bool grow = true;
                        while (jv + hgt < wV && grow) {
                            for (int x = 0; x < wdt; ++x) {
                                if (mask[static_cast<size_t>(jv + hgt) * wU + iu + x] != m) {
                                    grow = false; break;
                                }
                            }
                            if (grow) ++hgt;
                        }

                        auto world = [&](int iuu, int jvv) {
                            Eigen::Vector3i kk;
                            kk[axis] = s; kk[u] = mn[u] + iuu; kk[v] = mn[v] + jvv;
                            return origin + kk.cast<float>() * cs;
                        };
                        Eigen::Vector3f off = nrm.cast<float>() * h;
                        Eigen::Vector3f c00 = world(iu, jv);
                        Eigen::Vector3f eU = Eigen::Vector3f::Unit(u) * (cs * wdt);
                        Eigen::Vector3f eV = Eigen::Vector3f::Unit(v) * (cs * hgt);
                        Eigen::Vector3f base = c00 + off
                                              - Eigen::Vector3f::Unit(u) * h
                                              - Eigen::Vector3f::Unit(v) * h;
                        Eigen::Vector3f p0 = base;
                        Eigen::Vector3f p1 = base + eU;
                        Eigen::Vector3f p2 = base + eU + eV;
                        Eigen::Vector3f p3 = base + eV;

                        uint32_t colorIdx = m - 1;
                        size_t i0 = meshAddVertex(obj, colorIdx, p0);
                        size_t i1 = meshAddVertex(obj, colorIdx, p1);
                        size_t i2 = meshAddVertex(obj, colorIdx, p2);
                        size_t i3 = meshAddVertex(obj, colorIdx, p3);
                        if (sign > 0) meshAddQuad(obj, i0, i1, i2, i3);
                        else          meshAddQuad(obj, i0, i3, i2, i1);

                        for (int y = 0; y < hgt; ++y)
                            for (int x = 0; x < wdt; ++x)
                                mask[static_cast<size_t>(jv + y) * wU + iu + x] = 0;

                        iu += wdt;
                    }
                }
            }
        }
    }
}

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::mesh_marchingCubes(GridObject& obj, const MeshVoxels& voxels, bool naiveNoLUT) {
    if (voxels.empty()) return;

    float cs = pickCellSize(voxels);
    Eigen::Vector3f origin;
    Eigen::Vector3i mn, mx;
    latticeBounds(voxels, cs, origin, mn, mx);
    MeshGrid g{origin, cs};
    OctreeNode* anc = meshAncestor(voxels);
    auto at = [&](const Eigen::Vector3i& k){ return meshVoxelAt(g, k, obj.id, anc); };

    auto cornerWorld = [&](const Eigen::Vector3i& k) {
        return origin + k.cast<float>() * cs;
    };

    std::vector<Eigen::Vector3i> cells;
    cells.reserve(voxels.size() * 8);
    for (const auto& v : voxels) {
        Eigen::Vector3f relf = (v->position - origin) / cs;
        Eigen::Vector3i K(static_cast<int>(std::lround(relf.x())),
                          static_cast<int>(std::lround(relf.y())),
                          static_cast<int>(std::lround(relf.z())));
        for (int i = 0; i < 8; ++i) cells.push_back(K - kMCcorner(i));
    }
    std::sort(cells.begin(), cells.end(),
              [](const Eigen::Vector3i& a, const Eigen::Vector3i& b){
                  if (a.z() != b.z()) return a.z() < b.z();
                  if (a.y() != b.y()) return a.y() < b.y();
                  return a.x() < b.x();
              });
    cells.erase(std::unique(cells.begin(), cells.end(),
              [](const Eigen::Vector3i& a, const Eigen::Vector3i& b){ return a == b; }),
              cells.end());

    for (const Eigen::Vector3i& base : cells) {
        int cubeIndex = 0;
        bool inside[8];
        uint32_t ccol[8];
        Eigen::Vector3f cpos[8];
        for (int i = 0; i < 8; ++i) {
            Eigen::Vector3i ck = base + kMCcorner(i);
            auto sv = at(ck);
            inside[i] = static_cast<bool>(sv);
            ccol[i] = inside[i] ? static_cast<uint32_t>(sv->colorIdx) : 0u;
            cpos[i] = cornerWorld(ck);
            if (inside[i]) cubeIndex |= (1 << i);
        }
        if (cubeIndex == 0 || cubeIndex == 255) continue;

        auto edgeColor = [&](int e) -> uint32_t {
            int a = kMCedge[e][0], b = kMCedge[e][1];
            return inside[a] ? ccol[a] : ccol[b];
        };
        auto edgeCross = [&](int e) -> Eigen::Vector3f {
            int a = kMCedge[e][0], b = kMCedge[e][1];
            return 0.5f * (cpos[a] + cpos[b]);
        };

        if (!naiveNoLUT) {
            int et = edgeTable[cubeIndex];
            if (et == 0) continue;
            for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
                int e0 = triTable[cubeIndex][i];
                int e1 = triTable[cubeIndex][i+1];
                int e2 = triTable[cubeIndex][i+2];
                size_t a = meshAddVertex(obj, edgeColor(e0), edgeCross(e0));
                size_t b = meshAddVertex(obj, edgeColor(e1), edgeCross(e1));
                size_t c = meshAddVertex(obj, edgeColor(e2), edgeCross(e2));
                obj.objMesh.tris.push_back({a, b, c});
            }
        } else {
            std::vector<Eigen::Vector3f> pts;
            std::vector<uint32_t> cols;
            for (int e = 0; e < 12; ++e) {
                int a = kMCedge[e][0], b = kMCedge[e][1];
                if (inside[a] != inside[b]) { pts.push_back(edgeCross(e)); cols.push_back(edgeColor(e)); }
            }
            if (pts.size() < 3) continue;
            Eigen::Vector3f cen = Eigen::Vector3f::Zero();
            for (auto& p : pts) cen += p;
            cen /= static_cast<float>(pts.size());
            Eigen::Vector3f nrm = Eigen::Vector3f::Zero();
            for (size_t i = 0; i < pts.size(); ++i) {
                const Eigen::Vector3f& p0 = pts[i];
                const Eigen::Vector3f& p1 = pts[(i+1) % pts.size()];
                nrm += (p0 - cen).cross(p1 - cen);
            }
            if (nrm.norm() < 1e-12f) nrm = Eigen::Vector3f::UnitZ();
            nrm.normalize();
            Eigen::Vector3f ref = (pts[0] - cen).normalized();
            Eigen::Vector3f bi = nrm.cross(ref);
            std::vector<size_t> order(pts.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](size_t A, size_t B){
                float aa = std::atan2((pts[A]-cen).dot(bi), (pts[A]-cen).dot(ref));
                float bb = std::atan2((pts[B]-cen).dot(bi), (pts[B]-cen).dot(ref));
                return aa < bb;
            });
            uint32_t cenCol = cols.empty() ? 0u : cols[order[0]];
            size_t cIdx = meshAddVertex(obj, cenCol, cen);
            for (size_t i = 0; i < order.size(); ++i) {
                size_t A = order[i], B = order[(i+1) % order.size()];
                size_t a = meshAddVertex(obj, cols[A], pts[A]);
                size_t b = meshAddVertex(obj, cols[B], pts[B]);
                obj.objMesh.tris.push_back({cIdx, a, b});
            }
        }
    }
}

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::mesh_dualContour(GridObject& obj, const MeshVoxels& voxels, bool manifold) {
    if (voxels.empty()) return;

    float cs = pickCellSize(voxels);
    Eigen::Vector3f origin;
    Eigen::Vector3i mn, mx;
    latticeBounds(voxels, cs, origin, mn, mx);
    MeshGrid g{origin, cs};
    OctreeNode* anc = meshAncestor(voxels);
    auto at = [&](const Eigen::Vector3i& k){ return meshVoxelAt(g, k, obj.id, anc); };

    auto cornerWorld = [&](const Eigen::Vector3i& k) {
        return origin + k.cast<float>() * cs;
    };

    std::vector<Eigen::Vector3i> vkeys;
    vkeys.reserve(voxels.size());
    for (const auto& v : voxels) {
        Eigen::Vector3f relf = (v->position - origin) / cs;
        vkeys.push_back(Eigen::Vector3i(static_cast<int>(std::lround(relf.x())),
                                        static_cast<int>(std::lround(relf.y())),
                                        static_cast<int>(std::lround(relf.z()))));
    }
    
    std::unordered_map<Eigen::Vector3i, long long, vec3ih> cellVert;

    auto cellHasSurface = [&](const Eigen::Vector3i& base, Eigen::Vector3f& outPos, uint32_t& outColor) -> bool {
        bool inside[8];
        uint32_t ccol[8];
        Eigen::Vector3f cpos[8];
        int mask = 0;
        for (int i = 0; i < 8; ++i) {
            Eigen::Vector3i ck = base + kMCcorner(i);
            auto sv = at(ck);
            inside[i] = static_cast<bool>(sv);
            ccol[i] = inside[i] ? static_cast<uint32_t>(sv->colorIdx) : 0u;
            cpos[i] = cornerWorld(ck);
            if (inside[i]) mask |= (1 << i);
        }
        if (mask == 0 || mask == 255) return false;

        Eigen::Vector3f acc = Eigen::Vector3f::Zero();
        uint32_t col = 0; int n = 0;
        for (int e = 0; e < 12; ++e) {
            int a = kMCedge[e][0], b = kMCedge[e][1];
            if (inside[a] != inside[b]) {
                acc += 0.5f * (cpos[a] + cpos[b]);
                col = inside[a] ? ccol[a] : ccol[b];
                ++n;
            }
        }
        if (n == 0) return false;
        Eigen::Vector3f pos = acc / static_cast<float>(n);
        if (manifold) {
            Eigen::Vector3f cellMin = cornerWorld(base);
            Eigen::Vector3f cellMax = cellMin + Eigen::Vector3f::Constant(cs);
            pos = pos.cwiseMax(cellMin).cwiseMin(cellMax);
        }
        outPos = pos;
        outColor = col;
        return true;
    };

    auto getCellVert = [&](const Eigen::Vector3i& base) -> long long {
        auto it = cellVert.find(base);
        if (it != cellVert.end()) return it->second;
        Eigen::Vector3f pos; uint32_t col;
        long long idx = -1;
        if (cellHasSurface(base, pos, col)) idx = static_cast<long long>(meshAddVertex(obj, col, pos));
        cellVert[base] = idx;
        return idx;
    };

    std::unordered_map<Eigen::Vector3i, uint8_t, vec3ih> edgeSeen;

    auto cornerInside = [&](const Eigen::Vector3i& c){ return at(c) != nullptr; };

    auto processEdge = [&](const Eigen::Vector3i& c, int axis) {
        uint8_t& seen = edgeSeen[c];
        if (seen & (1 << axis)) return;
        seen |= (1 << axis);

        Eigen::Vector3i d(0,0,0); d[axis] = 1;
        bool a = cornerInside(c);
        bool b = cornerInside(c + d);
        if (a == b) return;

        int u = (axis + 1) % 3;
        int v = (axis + 2) % 3;
        Eigen::Vector3i du(0,0,0); du[u] = 1;
        Eigen::Vector3i dv(0,0,0); dv[v] = 1;

        Eigen::Vector3i c0 = c - du - dv;
        Eigen::Vector3i c1 = c      - dv;
        Eigen::Vector3i c2 = c;
        Eigen::Vector3i c3 = c - du;

        long long i0 = getCellVert(c0);
        long long i1 = getCellVert(c1);
        long long i2 = getCellVert(c2);
        long long i3 = getCellVert(c3);
        if (i0 < 0 || i1 < 0 || i2 < 0 || i3 < 0) return;

        if (a) {
            obj.objMesh.tris.push_back({(size_t)i0,(size_t)i1,(size_t)i2});
            obj.objMesh.tris.push_back({(size_t)i0,(size_t)i2,(size_t)i3});
        } else {
            obj.objMesh.tris.push_back({(size_t)i0,(size_t)i2,(size_t)i1});
            obj.objMesh.tris.push_back({(size_t)i0,(size_t)i3,(size_t)i2});
        }
    };

    for (const Eigen::Vector3i& K : vkeys) {
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3i d(0,0,0); d[axis] = 1;
            processEdge(K, axis);
            processEdge(K - d, axis);
        }
    }
}

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::mesh_cubicMarching(GridObject& obj, const MeshVoxels& voxels) {
    if (voxels.empty()) return;

    float cs = pickCellSize(voxels);
    Eigen::Vector3f origin;
    Eigen::Vector3i mn, mx;
    latticeBounds(voxels, cs, origin, mn, mx);
    MeshGrid g{origin, cs};
    OctreeNode* anc = meshAncestor(voxels);
    auto at = [&](const Eigen::Vector3i& k){ return meshVoxelAt(g, k, obj.id, anc); };

    const float iso = 0.5f;
    auto field = [&](const Eigen::Vector3i& k) -> float {
        return at(k) != nullptr ? 1.0f : 0.0f;
    };
    auto cornerWorld = [&](const Eigen::Vector3i& k) {
        return origin + k.cast<float>() * cs;
    };

    auto catmull = [](float p0, float p1, float p2, float p3, float t) {
        float t2 = t*t, t3 = t2*t;
        return 0.5f * ((2*p1) + (-p0 + p2)*t +
                       (2*p0 - 5*p1 + 4*p2 - p3)*t2 +
                       (-p0 + 3*p1 - 3*p2 + p3)*t3);
    };
    auto edgeCrossT = [&](const Eigen::Vector3i& a, const Eigen::Vector3i& b) -> float {
        Eigen::Vector3i d = b - a;
        float p1 = field(a);
        float p2 = field(b);
        float p0 = field(a - d);
        float p3 = field(b + d);
        float lo = 0.0f, hi = 1.0f;
        float flo = p1 - iso;
        for (int it = 0; it < 12; ++it) {
            float midt = 0.5f * (lo + hi);
            float fmid = catmull(p0, p1, p2, p3, midt) - iso;
            if ((flo < 0) == (fmid < 0)) { lo = midt; flo = fmid; }
            else hi = midt;
        }
        return 0.5f * (lo + hi);
    };

    std::vector<Eigen::Vector3i> cells;
    cells.reserve(voxels.size() * 8);
    for (const auto& v : voxels) {
        Eigen::Vector3f relf = (v->position - origin) / cs;
        Eigen::Vector3i K(static_cast<int>(std::lround(relf.x())),
                          static_cast<int>(std::lround(relf.y())),
                          static_cast<int>(std::lround(relf.z())));
        for (int i = 0; i < 8; ++i) cells.push_back(K - kMCcorner(i));
    }
    std::sort(cells.begin(), cells.end(),
              [](const Eigen::Vector3i& a, const Eigen::Vector3i& b){
                  if (a.z() != b.z()) return a.z() < b.z();
                  if (a.y() != b.y()) return a.y() < b.y();
                  return a.x() < b.x();
              });
    cells.erase(std::unique(cells.begin(), cells.end(),
              [](const Eigen::Vector3i& a, const Eigen::Vector3i& b){ return a == b; }),
              cells.end());

    for (const Eigen::Vector3i& base : cells) {
        int cubeIndex = 0;
        Eigen::Vector3f cpos[8];
        Eigen::Vector3i ck[8];
        bool inside[8];
        uint32_t ccol[8];
        for (int i = 0; i < 8; ++i) {
            ck[i] = base + kMCcorner(i);
            cpos[i] = cornerWorld(ck[i]);
            auto sv = at(ck[i]);
            inside[i] = static_cast<bool>(sv);
            ccol[i] = inside[i] ? static_cast<uint32_t>(sv->colorIdx) : 0u;
            if (inside[i]) cubeIndex |= (1 << i);
        }
        int et = edgeTable[cubeIndex];
        if (et == 0) continue;

        Eigen::Vector3f vpos[12];
        uint32_t vcol[12];
        bool vset[12] = {false};
        auto cross = [&](int e) {
            int a = kMCedge[e][0], b = kMCedge[e][1];
            float t = edgeCrossT(ck[a], ck[b]);
            vpos[e] = cpos[a] + (cpos[b] - cpos[a]) * t;
            vcol[e] = inside[a] ? ccol[a] : ccol[b];
            vset[e] = true;
        };
        for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
            int e0 = triTable[cubeIndex][i];
            int e1 = triTable[cubeIndex][i+1];
            int e2 = triTable[cubeIndex][i+2];
            if (!vset[e0]) cross(e0);
            if (!vset[e1]) cross(e1);
            if (!vset[e2]) cross(e2);
            size_t a = meshAddVertex(obj, vcol[e0], vpos[e0]);
            size_t b = meshAddVertex(obj, vcol[e1], vpos[e1]);
            size_t c = meshAddVertex(obj, vcol[e2], vpos[e2]);
            obj.objMesh.tris.push_back({a, b, c});
        }
    }
}

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::mesh_dualMarching(GridObject& obj, const MeshVoxels& voxels) {
    mesh_dualContour(obj, voxels, false);
}

}