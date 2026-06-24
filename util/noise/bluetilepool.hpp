#ifndef BLUE_TILE_POOL_HPP
#define BLUE_TILE_POOL_HPP

#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <queue>
#include <utility>
#include <cmath>

namespace bluetile {

constexpr int SUB        = 128;
constexpr int SUB_AREA   = SUB * SUB;
constexpr int POOL       = 512;
constexpr int TILE       = 512;
constexpr int TILE_AREA  = TILE * TILE;
constexpr int GRID       = TILE / SUB;
constexpr int SUBS_PER_T = GRID * GRID;
constexpr int NTILES     = 32;

class VoidAndCluster {
public:
    explicit VoidAndCluster(int n, uint64_t seed) : N(n), gen(seed) {}

    std::vector<float> generate(int relaxBudget = -1) {
        const int area = N * N;
        const int budget = (relaxBudget < 0) ? area : relaxBudget;
        std::vector<uint8_t> bp(area, 0);
        std::vector<float> energy(area, 0.0f);

        int ones = area / 10;
        std::vector<int> idx(area);
        for (int i = 0; i < area; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), gen);
        #pragma omp parallel for
        for (int i = 0; i < ones; ++i) {
            bp[idx[i]] = 1;
            addEnergy(energy, idx[i], +1.0f);
        }
        int relaxIters = (relaxBudget < 0) ? 0 : budget;
        for (int iter = 0; iter < relaxIters; ++iter) {
            int c = argExtreme(energy, bp, true,  true);
            bp[c] = 0;
            addEnergy(energy, c, -1.0f);
            int v = argExtreme(energy, bp, false, false);
            bp[v] = 1;
            addEnergy(energy, v, +1.0f);
            if (c == v) break;
        }

        std::vector<int> rank(area, -1);
        {
            std::vector<uint8_t> work = bp;
            std::vector<float> e2 = energy;
            std::priority_queue<std::pair<float,int>> pq;
            for (int i = 0; i < area; ++i) if (work[i]) pq.emplace(e2[i], i);

            int rnk = ones - 1;
            while (rnk >= 0 && !pq.empty()) {
                auto [ev, c] = pq.top(); pq.pop();
                if (!work[c] || ev != e2[c]) continue;
                work[c] = 0;
                addEnergyAndRepush(e2, c, -1.0f, work, pq, true);
                rank[c] = rnk--;
            }
        }
        {
            std::vector<uint8_t> work = bp;
            std::vector<float> e2 = energy;
            std::priority_queue<std::pair<float,int>,
                std::vector<std::pair<float,int>>, std::greater<>> pq;
            for (int i = 0; i < area; ++i) if (!work[i]) pq.emplace(e2[i], i);
            int rnk = ones;
            while (rnk < area && !pq.empty()) {
                auto [ev, v] = pq.top(); pq.pop();
                if (work[v] || ev != e2[v]) continue;
                work[v] = 1;
                addEnergyAndRepush(e2, v, +1.0f, work, pq, false);
                rank[v] = rnk++;
            }
        }

        std::vector<float> out(area);
        const float inv = 1.0f / float(area);
        #pragma omp parallel for 
        for (int i = 0; i < area; ++i) out[i] = (float(rank[i]) + 0.5f) * inv;
        return out;
    }

private:
    int N;
    std::mt19937_64 gen;

    void addEnergy(std::vector<float>& e, int center, float s) {
        const float sigma = 1.9f, inv2s2 = 1.0f / (2.0f * sigma * sigma);
        int cx = center % N, cy = center / N, R = 6;
        #pragma omp parallel for
        for (int dy = -R; dy <= R; ++dy) {
            for (int dx = -R; dx <= R; ++dx) {
                int x = ((cx + dx) % N + N) % N;
                int y = ((cy + dy) % N + N) % N;
                e[y * N + x] += s * std::exp(-float(dx*dx + dy*dy) * inv2s2);
            }
        }
    }

    template<class PQ>
    void addEnergyAndRepush(std::vector<float>& e, int center, float s,
                            const std::vector<uint8_t>& work, PQ& pq, bool wantSet) {
        const float sigma = 1.9f, inv2s2 = 1.0f / (2.0f * sigma * sigma);
        int cx = center % N, cy = center / N, R = 6;
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) {
                int x = ((cx + dx) % N + N) % N;
                int y = ((cy + dy) % N + N) % N;
                int idx = y * N + x;
                e[idx] += s * std::exp(-float(dx*dx + dy*dy) * inv2s2);
                if ((work[idx] != 0) == wantSet) pq.emplace(e[idx], idx);
            }
    }

    int argExtreme(const std::vector<float>& e, const std::vector<uint8_t>& bp,
                   bool wantOne, bool wantMax) {
        int best = -1;
        float bv = wantMax ? -1e30f : 1e30f;
        #pragma omp parallel for
        for (int i = 0; i < (int)e.size(); ++i) {
            if ((bp[i] == 1) != wantOne) continue;
            if ((wantMax && e[i] > bv) || (!wantMax && e[i] < bv)) {
                bv = e[i];
                best = i;
            }
        }
        return best;
    }
};

struct Pool {
    std::vector<float> sub;
    Pool() : sub(size_t(POOL) * SUB_AREA) {}

    size_t byteSize() const { return sub.size() * sizeof(float); }
    const float* data() const { return sub.data(); }

    void build(uint64_t seed = 0x9E3779B97F4A7C15ull, int count = POOL, int relaxBudget = -1) {
        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < count; ++t) {
            VoidAndCluster vc(SUB, seed + 0x100000001B3ull * uint64_t(t + 1));
            std::vector<float> tile = vc.generate(relaxBudget);
            std::memcpy(&sub[size_t(t) * SUB_AREA], tile.data(), SUB_AREA * sizeof(float));
        }
        for (int t = count; t < POOL; ++t)
            std::memcpy(&sub[size_t(t) * SUB_AREA], &sub[size_t(t % count) * SUB_AREA], SUB_AREA * sizeof(float));
    }
};

struct Assembler {
    const Pool& pool;
    std::mt19937_64 gen;
    explicit Assembler(const Pool& p) : pool(p), gen(0) {}

    static inline void symMap(int sym, int x, int y, int& sx, int& sy) {
        int ax = x, ay = y;
        if (sym & 1) {
            int t = ax;
            ax = ay;
            ay = t;
        }
        if (sym & 2) ax = SUB - 1 - ax;
        if (sym & 4) ay = SUB - 1 - ay;
        sx = ax;
        sy = ay;
    }

    void assemble(uint64_t frameSeed, std::vector<float>& out) {
        gen.seed(frameSeed ^ 0xD1B54A32D192ED03ull);
        out.resize(size_t(NTILES) * TILE_AREA);

        std::uniform_int_distribution<int> dOrig(0, SUB - 1);
        std::uniform_int_distribution<int> dSym(0, 7);
        std::uniform_int_distribution<int> dPool(0, POOL - 1);

        #pragma omp parallel for
        for (int t = 0; t < NTILES; ++t) {
            float* tileOut = &out[size_t(t) * TILE_AREA];
            for (int gy = 0; gy < GRID; ++gy)
                for (int gx = 0; gx < GRID; ++gx) {
                    int poolIdx = dPool(gen);
                    int ox = dOrig(gen), oy = dOrig(gen);
                    int sym = dSym(gen);
                    const float* src = &pool.sub[size_t(poolIdx) * SUB_AREA];
                    for (int ly = 0; ly < SUB; ++ly)
                        for (int lx = 0; lx < SUB; ++lx) {
                            int wx = (lx + ox) % SUB;
                            int wy = (ly + oy) % SUB;
                            int sx, sy;
                            symMap(sym, wx, wy, sx, sy);
                            float v = src[sy * SUB + sx];
                            int px = gx * SUB + lx;
                            int py = gy * SUB + ly;
                            tileOut[py * TILE + px] = v;
                        }
                }
        }

        std::uniform_int_distribution<int> dT(0, NTILES - 1);
        int z = dT(gen);
        if (z != 0) {
            for (int i = 0; i < TILE_AREA; ++i)
                std::swap(out[i], out[size_t(z) * TILE_AREA + i]);
        }
    }
};

}
#endif