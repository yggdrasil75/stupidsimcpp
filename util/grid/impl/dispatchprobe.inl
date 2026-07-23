#pragma once
#ifdef VULKAN_SUPPORT

static constexpr int TILE_QUANTUM = 64;
static constexpr int TILE_MIN = 128;
static constexpr int TILE_MAX = 4096;
static constexpr int PROBE_BOUNCES = 2;
static constexpr int PROBE_WARMUP = 1;
static constexpr int PROBE_REPEATS = 3;
static constexpr int PROBE_BRACKET = 2;
static constexpr double PROBE_LO_SPLIT = 0.382;
static constexpr double PROBE_HI_SPLIT = 0.618;
static constexpr uint64_t PROBE_MEM_DIVISOR = 8;
static constexpr int PROBE_OCC_WAVES = 64;
static constexpr int PROBE_OCC_UNITS = 32;
static constexpr uint32_t PROBE_FILE_VERSION = 1;

static constexpr uint64_t WF_BYTES_PER_PATH =
        WF_PATH_STRIDE + WF_PATHHIT_STRIDE + WF_SHADOW_STRIDE + 3 * sizeof(uint32_t);

struct TileProfile {
    uint64_t deviceKey = 0;
    int tileTarget = 512;
    double mpixPerMs = 0.0;
    bool probed = false;

    bool empty() const { return !probed; }
};

static inline uint64_t deviceKeyOf(VkPhysicalDevice pd) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    uint64_t h = 0;
    for (uint32_t i = 0; i < VK_UUID_SIZE; ++i) {
        h ^= static_cast<uint64_t>(props.pipelineCacheUUID[i]) + 0x9e3779b9u + (h << 6) + (h >> 2);
    }
    h ^= static_cast<uint64_t>(props.driverVersion) + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
}

static inline int quantizeTileEdge(int edge) {
    int q = (edge / TILE_QUANTUM) * TILE_QUANTUM;
    return std::clamp(q, TILE_MIN, TILE_MAX);
}

static inline uint64_t deviceLocalBytes(VkPhysicalDevice pd) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    uint64_t largest = 0;
    for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
        if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            largest = std::max(largest, static_cast<uint64_t>(mem.memoryHeaps[i].size));
        }
    }
    return largest;
}

static inline int memoryCeilingTileEdge(VkPhysicalDevice pd) {
    const uint64_t localBytes = deviceLocalBytes(pd);
    if (localBytes == 0) return TILE_MIN;
    const uint64_t budget = localBytes / PROBE_MEM_DIVISOR;
    const uint64_t maxPaths = budget / WF_BYTES_PER_PATH;
    if (maxPaths == 0) return TILE_MIN;
    return quantizeTileEdge(static_cast<int>(std::sqrt(static_cast<double>(maxPaths))));
}

static inline int seedTileTarget(VkPhysicalDevice pd) {
    VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sub};
    vkGetPhysicalDeviceProperties2(pd, &p2);

    const int lanes = static_cast<int>(sub.subgroupSize ? sub.subgroupSize : 32u);
    const double occPaths = static_cast<double>(lanes) * PROBE_OCC_WAVES * PROBE_OCC_UNITS;
    const int byOcc = quantizeTileEdge(static_cast<int>(std::sqrt(occPaths)));
    const int byMem = memoryCeilingTileEdge(pd);
    return quantizeTileEdge(std::min(std::max(byOcc, TILE_MIN), byMem));
}

struct TileProfileStore {
    std::vector<TileProfile> entries;

    static fs::path path() {
        return fs::path("./cache/tileprofiles.json");
    }

    void load() {
        entries.clear();
        std::ifstream in(path());
        if (!in) return;
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string json = ss.str();

        if (JsonHelper::parseInt(json, "version", 0) != static_cast<int>(PROBE_FILE_VERSION)) return;

        for (const std::string& obj : JsonHelper::parseArray(json, "devices")) {
            TileProfile p;
            p.deviceKey = std::strtoull(JsonHelper::parseRaw(obj, "deviceKey").c_str(), nullptr, 10);
            p.tileTarget = JsonHelper::parseInt(obj, "tileTarget", 0);
            p.mpixPerMs = JsonHelper::parseFloat(obj, "mpixPerMs", 0.0f);
            p.probed = p.deviceKey != 0 && p.tileTarget >= TILE_MIN;
            if (!p.probed) continue;
            p.tileTarget = quantizeTileEdge(p.tileTarget);
            entries.push_back(p);
        }
    }

    void save() const {
        std::error_code ec;
        fs::create_directories(path().parent_path(), ec);
        std::ofstream out(path(), std::ios::trunc);
        if (!out) return;
        out << "{\n";
        out << "    \"version\": " << PROBE_FILE_VERSION << ",\n";
        out << "    \"devices\": [\n";
        for (size_t i = 0; i < entries.size(); ++i) {
            const TileProfile& p = entries[i];
            out << "        {\n";
            out << "            \"deviceKey\": " << p.deviceKey << ",\n";
            out << "            \"tileTarget\": " << p.tileTarget << ",\n";
            out << "            \"mpixPerMs\": " << p.mpixPerMs << "\n";
            out << "        }";
            if (i + 1 < entries.size()) out << ",";
            out << "\n";
        }
        out << "    ]\n";
        out << "}\n";
    }

    bool find(uint64_t key, TileProfile& out) const {
        for (const TileProfile& p : entries) {
            if (p.deviceKey == key) {
                out = p;
                return true;
            }
        }
        return false;
    }

    void put(const TileProfile& p) {
        for (TileProfile& e : entries) {
            if (e.deviceKey == p.deviceKey) {
                e = p;
                return;
            }
        }
        entries.push_back(p);
    }
};

inline TileProfileStore tileProfileStore;

static inline std::vector<Eigen::Matrix<int, 4, 1>> buildTiles(int width, int height, int tileTarget) {
    const int target = std::max(tileTarget, TILE_QUANTUM);
    const int nx = (width + target - 1) / target;
    const int ny = (height + target - 1) / target;

    std::vector<Eigen::Matrix<int, 4, 1>> tiles;
    tiles.reserve(static_cast<size_t>(nx) * static_cast<size_t>(ny));
    for (int j = 0; j < ny; ++j) {
        const int y0 = j * height / ny;
        const int y1 = (j + 1) * height / ny;
        for (int i = 0; i < nx; ++i) {
            const int x0 = i * width / nx;
            const int x1 = (i + 1) * width / nx;
            tiles.push_back({x0, y0, x1 - x0, y1 - y0});
        }
    }
    return tiles;
}

#endif
