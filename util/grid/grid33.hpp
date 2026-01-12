#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <iostream>
#include "../vectorlogic/vec3.hpp"

static inline uint32_t FindLowestOn(uint64_t v)
{
#if defined(_MSC_VER) && defined(TREEXY_USE_INTRINSICS)
  unsigned long index;
  _BitScanForward64(&index, v);
  return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(TREEXY_USE_INTRINSICS)
  return static_cast<uint32_t>(__builtin_ctzll(v));
#else
  static const unsigned char DeBruijn[64] = {
    0,  1,  2,  53, 3,  7,  54, 27, 4,  38, 41, 8,  34, 55, 48, 28,
    62, 5,  39, 46, 44, 42, 22, 9,  24, 35, 59, 56, 49, 18, 29, 11,
    63, 52, 6,  26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
    51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
  };
// disable unary minus on unsigned warning
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return DeBruijn[uint64_t((v & -v) * UINT64_C(0x022FDD63CC95386D)) >> 58];
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(pop)
#endif

#endif
}

inline uint32_t CountOn(uint64_t v)
{
#if defined(_MSC_VER) && defined(_M_X64)
    v = __popcnt64(v);
#elif (defined(__GNUC__) || defined(__clang__))
    v = __builtin_popcountll(v);
#else
    // Software Implementation
    v = v - ((v >> 1) & uint64_t(0x5555555555555555));
    v = (v & uint64_t(0x3333333333333333)) + ((v >> 2) & uint64_t(0x3333333333333333));
    v = (((v + (v >> 4)) & uint64_t(0xF0F0F0F0F0F0F0F)) *
         uint64_t(0x101010101010101)) >>
        56;
#endif
    return static_cast<uint32_t>(v);
}

template<uint32_t LOG2DIM>
class Mask {
private: 
    static constexpr uint32_t SIZE = std::pow(2, 3 * LOG2DIM);
    static constexpr uint32_t WORD_COUNT = SIZE / 64;
    uint64_t mWords[WORD_COUNT];

    uint32_t findFirstOn() const {
        const uint64_t *w = mWords;
        uint32_t n = 0;
        while (n < WORD_COUNT && !*w) {
            ++w;
            ++n;
        }
        return n == WORD_COUNT ? SIZE : (n << 6) + FindLowestOn(*w);
    }

    uint32_t findNextOn(uint32_t start) const {
        uint32_t n start >> 6;
        if (n >= WORD_COUNT) {
            return SIZE;
        }
        uint32_t m = start & 63;
        uint64_t b = mWords[n];
        if (b & (uint64_t(1) << m)) {
            return start;
        }
        b &= ~uint64_t(0) << m;
        while (!b && ++n < WORD_COUNT) {
            b = mWords[n];
        }
        return (!b ? SIZE : (n << 6) + FindLowestOn(b));
    }

public:
    static size_t memUsage() {
        return sizeof(Mask);
    }
    
    static uint32_t bitCount() {
        return SIZE;
    }
    
    static uint32_t wordCount() {
        return WORD_COUNT;
    }

    uint64_t getWord(size_t n) const {
        return mWords[n];
    }

    void setWord(size_t n, uint64_t v) {
        mWords[n] = v;
    }

    uint32_t countOn() const {
        uint32_t sum = 0;
        uint32_t n = WORD_COUNT;
        for (const uint64_t* w = mWords; n--; ++w) {
            sum += CountOn(*w);
        }
        return sum;
    }

    class Iterator {
    private:
        uint32_t mPos;
        const Mask* mParent;
    public:
        Iterator() : mPos(Mask::SIZE), mParent(nullptr) {}
        Iterator(uint32_t pos, const Mask* parent) : mPos(pos), mParent(parent) {}
        
        Iterator& operator=(const Iterator&) = default;
        
        uint32_t opeator*() const {
            return mPos;
        }
        
        operator bool() const {
            return mPos != Mask::SIZE;
        }

        Iterator& operator++() {
            mPos = mParent -> findNextOn(mPos + 1);
            return *this;
        }
    }

    Mask() {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = 0;
        }
    }

    Mask(bool on) {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = v;
        }
    }

    Mask(const Mask &other) {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = other.mWords[i];
        }
    }

    template<typename WordT>
    WordT getWord(int n) const {
        return reinterpret_cast<const WordT *>(mWords)[n];
    }

    Mask &operator=(const Mask &other) {
        // static_assert(sizeof(Mask) == sizeof(Mask), "Mismatching sizeof");
        // static_assert(WORD_COUNT == Mask::WORD_COUNT, "Mismatching word count");
        // static_assert(LOG2DIM == Mask::LOG2DIM, "Mismatching LOG2DIM");
        uint64_t *src = reinterpret_cast<const uint64_t* >(&other);
        uint64_t *dst = mWords;
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            *dst++ = *src++;
        }
        return *this;
    }

    bool operator==(const Mask &other) const {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            if (mWords[i] != other.mWords[i]) return false;
        }
        return true;
    }

    bool operator!=(const Mask &other) const {
        return !((*this) == other);
    }

    Iterator beginOn() const {
        return Iterator(this->findFirstOn(), this);
    }

    /// @brief return true if bit n is set.
    /// @param n the bit to check
    /// @return true otherwise return false
    bool isOn(uint32_t n) const {
        return 0 != (mWords[n >> 6] & (uint64_t(1) << (n&63)));
    }

    bool isOn() const {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            if (mWords[i] != ~uint64_t(0)) return false;
        }
        return true;
    }

    bool isOff() const {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            if (mWords[i] != ~uint64_t(0)) return true;
        }
        return false;
    }

    bool setOn(uint32_t n) {
        uint64_t &word = mWords[n >> 6];
        const uint64_t bit = (uint64_t(1) << (n & 63));
        bool wasOn = word & bit;
        word |= bit;
        return wasOn;
    }

    void setOff(uint32_t n) {
        mWords[n >> 6] &= ~(uint64_t(1) << (n & 63));
    }

    template<bool UseBranchless = true>
    void set(uint32_t n, bool On) {
        if constexpr (UseBranchless) {
            auto &word = mWords[n >> 6];
            n &= 63;
            word &= ~(uint64_t(1) << n);
            word |= uint64_t(On) << n;
        } else {
            On ? this->setOn(n) : this->setOff(n);
        }
    }

    void setOn() {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = ~uint64_t(0);
        }
    }

    void setOff() {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = uint64_t(0);
        }
    }

    void set(bool on) {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = v;
        }
    }

    void toggle() {
        uint32_t n = WORD_COUNT;
        for (auto* w = mWords; n--; ++w) {
            *w = ~*w;
        }
    }

    void toggle(uint32_t n) {
        mWords[n >> 6] ^= uint64_t(1) << (n & 63);
    }
};

template <typename DataT, int Log2DIM>
class Grid {
    constexpr static int DIM = 1 << Log2DIM;
    constexpr static int SIZE = DIM * DIM * DIM;
    std::array<DataT, SIZE> data;
    Mask<Log2DIM> mask;
};

template <typename DataT, int INNER_BITS = 2, int LEAF_BITS = 3>
class VoxelGrid {
private:
    
public:
    constexpr static int32_t Log2N = INNER_BITS + LEAF_BITS;
    using LeafGrid = Grid<DataT, LEAF_BITS>;
    using InnerGrid = Grid<std::shared_ptr<LeafGrid>, INNER_BITS>;
    using RootMap = std::unordered_map<Vec3i, InnerGrid>;
    RootMap root_map;
    const double resolution;
    const double inv_resolution;
    const double half_resolution;

    VoxelGrid(double voxel_size) : resolution(voxel_size), inv_resolution(1.0 / voxel_size), half_resolution(0.5 * voxel_size) {}
    
    size_t getMemoryUsage() const {
        size_t total_size = 0;
        for (unsigned i = 0; i < root_map.bucket_count(); ++i) {
            size_t bucket_size = root_map.bucket_size(i);
            if (bucket_size == 0) {
                total_size++;
            } else {
                total_size += bucket_size;
            }
        }
        size_t entry_size = sizeof(Vec3i) + sizeof(InnerGrid) + sizeof(void *);
        total_size += root_map.size() * entry_size;

        for (const auto& [key, inner_grid] : root_map) {
            total_size += inner_grid.mask.countOn() * sizeof(LeafGrid);
        }
        return total_size;
    }

    static inline  Veci PosToCoord(float x, float y, float z) {
        // union VI {
        //     __m128i m;
        //     int32_t i[4];
        // };
        // static __m128 RES = _mm_set1_ps(inv_resolution);
        // __m128 vect = _mm_set_ps(x, y, z, 0.0);
        // __m128 res = _mm_mul_ps(vect, RES);
        // VI out;
        // out.m = _mm_cvttps_epi32(_mm_floor_ps(res));
        // return {out.i[3], out.i[2], out.i[1]};

        return Vec3f(x,y,z).floorToI();
    }

    static inline Vec3i posToCoord(double x, double y, double z) {
        return Vec3f(x,y,z).floorToI();
    }

    static inline Vec3i posToCoord(const Vec3d &pos) {
        return pos.floorToI();
    }

    Vec3d Vec3ioPos(const Vec3i &coord) {
        return (coord.toDouble() * resolution) + half_resolution;
    }

    template <class VisitorFunction>
    void forEachCell(VisitorFunction func) {
        constexpr static int32_t MASK_LEAF = ((1 << LEAF_BITS) - 1);
        constexpr static int32_t MASK_INNER = ((1 << INNER_BITS) - 1);
        for (auto& map_it : root_map) {
            const auto& [xA, yA, zA] = (map_it.first);
            InnerGrid& inner_grid = map_it.second;
            auto& mask2 = inner_grid.mask;
            for (auto inner_it = mask2.beginOn(); inner_it; ++inner_it) {
                const int32_t inner_index = *inner_it;
                int32_t xB = xA | ((inner_index & MASK_INNER) << LEAF_BITS);
                int32_t yB = yA | (((inner_index >> INNER_BITS) & MASK_INNER) << LEAF_BITS);
                int32_t zB = zA | (((inner_index >> (INNER_BITS* 2)) & MASK_INNER) << LEAF_BITS);

                auto& leaf_grid = inner_grid.data[inner_index];
                auto& mask1 = leaf_grid->mask;
                for (auto leaf_it = mask1.beginOn(); leaf_it; ++leaf_it){
                    const int32_t leaf_index = *leaf_it;
                    Vec3i pos = Vec3i(xB | (leaf_index & MASK_LEAF), 
                                      yB | ((leaf_index >> LEAF_BITS) & MASK_LEAF),
                                      zB | ((leaf_index >> (LEAF_BITS * 2)) & MASK_LEAF));
                    func(leaf_grid->data[leaf_index], pos);
                }
            }
        }
    }


    class Accessor {
    private:
        RootMap &root_;
        Vec3i prev_root_coord_;
        Vec3i prev_inner_coord_;
        InnerGrid* prev_inner_ptr_ = nullptr;
        LeafGrid* prev_leaf_ptr_ = nullptr;
    public:
        Accessor(RootMap& root) : root_(root) {}
        bool setValue(const Vec3i& coord, const DataT& value) {
            LeafGrid* leaf_ptr = prev_leaf_ptr_;
            const Vec3i inner_key = getInnerKey(coord);
            if (inner_key != prev_inner_coord_ || !prev_leaf_ptr_) {
                InnerGrid* inner_ptr = prev_inner_ptr_;
                const Vec3i root_key = getRootKey(coord);
                if (root_key != prev_root_coord_ || !prev_inner_ptr_) {
                    auto root_it = root_.find(root_key);
                    if (root_it == root_.end()) {
                        root_it = root_insert({root_key, InnerGrid()}).first;
                    }

                    inner_ptr = &(root_it->second);
                    prev_root_coord_ = root_key;
                    prev_inner_ptr_ = inner_ptr;
                }

                const uint32_t inner_index = getInnerIndex(coord);
                auto& inner_data = inner_ptr->data[inner_index];
                if (inner_ptr->mask.setOn(inner_index)) {
                    inner_data = std::make_shared<LeafGrid>();
                }

                leaf_ptr = inner_data.get();
                prev_inner_coord_ = inner_key;
                prev_leaf_ptr_ = leaf_ptr;
            }

            const uint32_t leaf_index = getLeafIndex(coord);
            bool was_on = leaf_ptr->mask.setOn(leaf_index);
            leaf_ptr->data[leaf_index] = value;
            return was_on;
        }

        DataT* value(const Vec3i& coord) {
            LeafGrid* leaf_ptr = prev_leaf_ptr_;
            const Vec3i inner_key = getInnerKey(coord);
            if (inner_key != prev_inner_coord_ || !prev_inner_ptr_) {
                InnerGrid* inner_ptr = prev_inner_ptr_;
                const Vec3i root_key = getRootKey(coord);
                
                if (root_key != prev_root_coord_ || !prev_inner_ptr_) {
                    auto it = root_.find(root_key);
                    if (it == root_.end()) {
                        return nullptr;
                    }
                    inner_ptr = &(it->second);
                    prev_inner_coord_ = root_key;
                    prev_inner_ptr_ = inner_ptr;
                }

                const uint32_t inner_index = getInnerIndex(coord);
                auto& inner_data - inner_ptr->data[inner_index];

                if (!inner_ptr->mask.isOn(inner_index)) {
                    return nullptr;
                }

                leaf_ptr = inner_ptr->data[inner_index].get();
                prev_inner_coord_ = inner_key;
                prev_leaf_ptr_ = leaf_ptr;
            }

            const uint32_t leaf_index = getLeafIndex(coord);
            if (!leaf_ptr->mask.isOn(leaf_index)) {
                return nullptr;
            }

            return &(leaf_ptr->data[leaf_index]);
        }

        const InnerGrid* lastInnerGrid() const {
            return prev_inner_ptr_;
        }

        const LeafGrid* lastLeafGrid() const {
            return prev_leaf_ptr_;
        }
    }

    Accessor createAccessor() {
        return Accessor(root_map);
    }

    static inline Vec3i getRootKey(const Vec3i& coord) {
        constexpr static int32_t MASK = ~((1 << Log2N) - 1);
        return {coord.x & MASK, coord.y & MASK, coord.z & MASK};
    }

    static inline Vec3i getInnerKey(const Vec3i &coord)
    {
        constexpr static int32_t MASK = ~((1 << LEAF_BITS) - 1);
        return {coord.x & MASK, coord.y & MASK, coord.z & MASK};
    }

    static inline uint32_t getInnerIndex(const Vec3i &coord)
    {
        constexpr static int32_t MASK = ((1 << INNER_BITS) - 1);
        // clang-format off
        return ((coord.x >> LEAF_BITS) & MASK) +
               (((coord.y >> LEAF_BITS) & MASK) <<  INNER_BITS) +
               (((coord.z >> LEAF_BITS) & MASK) << (INNER_BITS * 2));
        // clang-format on
    }

    static inline uint32_t getLeafIndex(const Vec3i &coord)
    {
        constexpr static int32_t MASK = ((1 << LEAF_BITS) - 1);
        // clang-format off
        return (coord.x & MASK) +
               ((coord.y & MASK) <<  LEAF_BITS) +
               ((coord.z & MASK) << (LEAF_BITS * 2));
        // clang-format on
    }
};