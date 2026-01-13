#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <iostream>
#include "../vectorlogic/vec3.hpp"
#include "../basicdefines.hpp"
#include "../timing_decorator.hpp"

/// @brief Finds the index of the least significant bit set to 1 in a 64-bit integer.
/// @details Uses compiler intrinsics (_BitScanForward64, __builtin_ctzll) where available, 
/// falling back to a De Bruijn sequence multiplication lookup for portability.
/// @param v The 64-bit integer to scan.
/// @return The zero-based index of the lowest set bit. Behavior is undefined if v is 0.
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

/// @brief Counts the number of bits set to 1 (population count) in a 64-bit integer.
/// @details Uses compiler intrinsics (__popcnt64, __builtin_popcountll) where available,
/// falling back to a software Hamming weight implementation.
/// @param v The 64-bit integer to count.
/// @return The number of bits set to 1.
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
    v = (((v + (v >> 4)) & uint64_t(0xF0F0F0F0F0F0F0F)) * uint64_t(0x101010101010101)) >> 56;
#endif
    return static_cast<uint32_t>(v);
}

/// @brief A bitmask class for tracking active cells within a specific grid dimension.
/// @tparam LOG2DIM The log base 2 of the dimension size.
template<uint32_t LOG2DIM>
class Mask {
private: 
    static constexpr uint32_t SIZE = std::pow(2, 3 * LOG2DIM);
    static constexpr uint32_t WORD_COUNT = SIZE / 64;
    uint64_t mWords[WORD_COUNT];

    /// @brief Internal helper to find the linear index of the first active bit.
    /// @return The index of the first on bit, or SIZE if none are set.
    uint32_t findFirstOn() const {
        const uint64_t *w = mWords;
        uint32_t n = 0;
        while (n < WORD_COUNT && !*w) {
            ++w;
            ++n;
        }
        return n == WORD_COUNT ? SIZE : (n << 6) + FindLowestOn(*w);
    }

    /// @brief Internal helper to find the next active bit after a specific index.
    /// @param start The index to start searching from (inclusive check, though logic implies sequential access).
    /// @return The index of the next on bit, or SIZE if none are found.
    uint32_t findNextOn(uint32_t start) const {
        uint32_t n = start >> 6;
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
    /// @brief Returns the memory size of this Mask instance.
    static size_t memUsage() {
        return sizeof(Mask);
    }
    
    /// @brief Returns the total capacity (number of bits) in the mask.
    static uint32_t bitCount() {
        return SIZE;
    }
    
    /// @brief Returns the number of 64-bit words used to store the mask.
    static uint32_t wordCount() {
        return WORD_COUNT;
    }

    /// @brief Retrieves a specific 64-bit word from the mask array.
    /// @param n The index of the word.
    /// @return The word value.
    uint64_t getWord(size_t n) const {
        return mWords[n];
    }

    /// @brief Sets a specific 64-bit word in the mask array.
    /// @param n The index of the word.
    /// @param v The value to set.
    void setWord(size_t n, uint64_t v) {
        mWords[n] = v;
    }

    /// @brief Calculates the total number of bits set to 1 in the mask.
    /// @return The count of active bits.
    uint32_t countOn() const {
        uint32_t sum = 0;
        uint32_t n = WORD_COUNT;
        for (const uint64_t* w = mWords; n--; ++w) {
            sum += CountOn(*w);
        }
        return sum;
    }

    /// @brief Iterator class for traversing set bits in the Mask.
    class Iterator {
    private:
        uint32_t mPos;
        const Mask* mParent;
    public:
        /// @brief Default constructor creating an invalid end iterator.
        Iterator() : mPos(Mask::SIZE), mParent(nullptr) {}
        
        /// @brief Constructor for a specific position.
        /// @param pos The current bit index.
        /// @param parent Pointer to the Mask being iterated.
        Iterator(uint32_t pos, const Mask* parent) : mPos(pos), mParent(parent) {}
        
        /// @brief Default assignment operator.
        Iterator& operator=(const Iterator&) = default;
        
        /// @brief Dereference operator.
        /// @return The index of the current active bit.
        uint32_t operator*() const {
            return mPos;
        }
        
        /// @brief Boolean conversion operator.
        /// @return True if the iterator is valid (not at end), false otherwise.
        operator bool() const {
            return mPos != Mask::SIZE;
        }

        /// @brief Pre-increment operator. Advances to the next active bit.
        /// @return Reference to self.
        Iterator& operator++() {
            mPos = mParent -> findNextOn(mPos + 1);
            return *this;
        }
    };

    /// @brief Default constructor. Initializes all bits to 0 (off).
    Mask() {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = 0;
        }
    }

    /// @brief Constructor initializing all bits to a specific state.
    /// @param on If true, all bits are set to 1; otherwise 0.
    Mask(bool on) {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = v;
        }
    }

    /// @brief Copy constructor.
    Mask(const Mask &other) {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = other.mWords[i];
        }
    }

    /// @brief Reinterprets internal words as a different type and retrieves one.
    /// @tparam WordT The type to cast the pointer to (e.g., uint32_t).
    /// @param n The index in the reinterpreted array.
    /// @return The value at index n.
    template<typename WordT>
    WordT getWord(int n) const {
        return reinterpret_cast<const WordT *>(mWords)[n];
    }

    /// @brief Assignment operator.
    /// @param other The mask to copy from.
    /// @return Reference to self.
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

    /// @brief Equality operator.
    /// @param other The mask to compare.
    /// @return True if all bits match, false otherwise.
    bool operator==(const Mask &other) const {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            if (mWords[i] != other.mWords[i]) return false;
        }
        return true;
    }

    /// @brief Inequality operator.
    /// @param other The mask to compare.
    /// @return True if any bits differ.
    bool operator!=(const Mask &other) const {
        return !((*this) == other);
    }

    /// @brief Returns an iterator to the first active bit.
    /// @return An Iterator pointing to the first set bit.
    Iterator beginOn() const {
        return Iterator(this->findFirstOn(), this);
    }

    /// @brief Checks if a specific bit is set.
    /// @param n The bit index to check.
    /// @return True if the bit is 1, false if 0.
    bool isOn(uint32_t n) const {
        return 0 != (mWords[n >> 6] & (uint64_t(1) << (n&63)));
    }

    /// @brief Checks if all bits are set to 1.
    /// @return True if fully saturated.
    bool isOn() const {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            if (mWords[i] != ~uint64_t(0)) return false;
        }
        return true;
    }

    /// @brief Checks if all bits are set to 0.
    /// @return True if fully empty.
    bool isOff() const {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            if (mWords[i] != ~uint64_t(0)) return true;
        }
        return false;
    }

    /// @brief Sets a specific bit to 1.
    /// @param n The bit index to set.
    /// @return True if the bit was already on, false otherwise.
    bool setOn(uint32_t n) {
        uint64_t &word = mWords[n >> 6];
        const uint64_t bit = (uint64_t(1) << (n & 63));
        bool wasOn = word & bit;
        word |= bit;
        return wasOn;
    }

    /// @brief Sets a specific bit to 0.
    /// @param n The bit index to clear.
    void setOff(uint32_t n) {
        mWords[n >> 6] &= ~(uint64_t(1) << (n & 63));
    }

    /// @brief Sets a specific bit to the boolean value `On`.
    /// @param n The bit index.
    /// @param On The state to set (true=1, false=0).
    void set(uint32_t n, bool On) {
#if 1
            auto &word = mWords[n >> 6];
            n &= 63;
            word &= ~(uint64_t(1) << n);
            word |= uint64_t(On) << n;
#else
            On ? this->setOn(n) : this->setOff(n);
#endif
    }

    /// @brief Sets all bits to 1.
    void setOn() {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = ~uint64_t(0);
        }
    }

    /// @brief Sets all bits to 0.
    void setOff() {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = uint64_t(0);
        }
    }

    /// @brief Sets all bits to a specific boolean state.
    /// @param on If true, fill with 1s; otherwise 0s.
    void set(bool on) {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            mWords[i] = v;
        }
    }

    /// @brief Inverts (flips) all bits in the mask.
    void toggle() {
        uint32_t n = WORD_COUNT;
        for (auto* w = mWords; n--; ++w) {
            *w = ~*w;
        }
    }

    /// @brief Inverts (flips) a specific bit.
    /// @param n The bit index to toggle.
    void toggle(uint32_t n) {
        mWords[n >> 6] ^= uint64_t(1) << (n & 63);
    }
};

/// @brief Represents a generic grid block containing data and a presence mask.
/// @tparam DataT The type of data stored in each cell.
/// @tparam Log2DIM The log base 2 of the grid dimension (e.g., 3 for 8x8x8).
template <typename DataT, int Log2DIM>
class Grid {
public:
    constexpr static int DIM = 1 << Log2DIM;
    constexpr static int SIZE = DIM * DIM * DIM;
    std::array<DataT, SIZE> data;
    Mask<Log2DIM> mask;
};

/// @brief A sparse hierarchical voxel grid container.
/// @details Implements a 3-level structure: Root Map -> Inner Grid -> Leaf Grid.
/// @tparam DataT The type of data stored in the leaf voxels.
/// @tparam INNER_BITS Log2 dimension of the inner grid nodes (intermediate layer).
/// @tparam LEAF_BITS Log2 dimension of the leaf grid nodes (data layer).
template <typename DataT, int INNER_BITS = 2, int LEAF_BITS = 3>
class VoxelGrid {
public:
    constexpr static int32_t Log2N = INNER_BITS + LEAF_BITS;
    using LeafGrid = Grid<DataT, LEAF_BITS>;
    using InnerGrid = Grid<std::shared_ptr<LeafGrid>, INNER_BITS>;
    using RootMap = std::unordered_map<Vec3i, InnerGrid>;
    RootMap root_map;
    const double resolution;
    const double inv_resolution;
    const double half_resolution;

    /// @brief Constructs a VoxelGrid with a specific voxel size.
    /// @param voxel_size The size of a single voxel in world units.
    VoxelGrid(double voxel_size) : resolution(voxel_size), inv_resolution(1.0 / voxel_size), half_resolution(0.5 * voxel_size) {}
    
    /// @brief Calculates the approximate memory usage of the grid structure.
    /// @return The size in bytes used by the map, inner grids, and leaf grids.
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

    /// @brief Converts a 3D float position to integer grid coordinates.
    /// @param x X coordinate.
    /// @param y Y coordinate.
    /// @param z Z coordinate.
    /// @return The integer grid coordinates.
    static inline  Vec3i PosToCoord(float x, float y, float z) {
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

    /// @brief Converts a 3D double position to integer grid coordinates.
    /// @param x X coordinate.
    /// @param y Y coordinate.
    /// @param z Z coordinate.
    /// @return The integer grid coordinates.
    static inline Vec3i posToCoord(double x, double y, double z) {
        return Vec3f(x,y,z).floorToI();
    }

    /// @brief Converts a Vec3d position to integer grid coordinates.
    /// @param pos The position vector.
    /// @return The integer grid coordinates.
    static inline Vec3i posToCoord(const Vec3d &pos) {
        return pos.floorToI();
    }

    /// @brief Converts integer grid coordinates back to world position (center of voxel).
    /// @param coord The grid coordinate.
    /// @return The world position center of the voxel.
    Vec3d Vec3iToPos(const Vec3i& coord) const {
        return (coord.toDouble() * resolution) + half_resolution;
    }

    /// @brief Iterates over every active cell in the grid and applies a visitor function.
    /// @tparam VisitorFunction The type of the callable (DataT& val, Vec3i pos).
    /// @param func The function to execute for each active voxel.
    template <class VisitorFunction>
    void forEachCell(VisitorFunction func) {
        constexpr static int32_t MASK_LEAF = ((1 << LEAF_BITS) - 1);
        constexpr static int32_t MASK_INNER = ((1 << INNER_BITS) - 1);
        for (auto& map_it : root_map) {
            const Vec3i& root_coord = map_it.first;
            int32_t xA = root_coord.x;
            int32_t yA = root_coord.y;
            int32_t zA = root_coord.z;
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

    /// @brief Helper class to accelerate random access to the VoxelGrid by caching recent paths.
    class Accessor {
    private:
        RootMap &root_;
        Vec3i prev_root_coord_;
        Vec3i prev_inner_coord_;
        InnerGrid* prev_inner_ptr_ = nullptr;
        LeafGrid* prev_leaf_ptr_ = nullptr;
    public:
        /// @brief Constructs an Accessor for a specific root map.
        /// @param root Reference to the grid's root map.
        Accessor(RootMap& root) : root_(root) {}

        /// @brief Sets a value at a specific coordinate, creating nodes if they don't exist.
        /// @param coord The grid coordinate.
        /// @param value The value to store.
        /// @return True if the voxel was already active, false if it was newly activated.
        bool setValue(const Vec3i& coord, const DataT& value) {
            LeafGrid* leaf_ptr = prev_leaf_ptr_;
            const Vec3i inner_key = getInnerKey(coord);
            if (inner_key != prev_inner_coord_ || !prev_leaf_ptr_) {
                InnerGrid* inner_ptr = prev_inner_ptr_;
                const Vec3i root_key = getRootKey(coord);
                if (root_key != prev_root_coord_ || !prev_inner_ptr_) {
                    auto root_it = root_.find(root_key);
                    if (root_it == root_.end()) {
                        root_it = root_.insert({root_key, InnerGrid()}).first;
                    }

                    inner_ptr = &(root_it->second);
                    prev_root_coord_ = root_key;
                    prev_inner_ptr_ = inner_ptr;
                }

                const uint32_t inner_index = getInnerIndex(coord);
                auto& inner_data = inner_ptr->data[inner_index];
                if (inner_ptr->mask.setOn(inner_index) == false) {
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

        /// @brief Retrieves a pointer to the value at a coordinate.
        /// @param coord The grid coordinate.
        /// @return Pointer to the data if active, otherwise nullptr.
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
                auto& inner_data = inner_ptr->data[inner_index];

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

        /// @brief Returns the most recently accessed InnerGrid pointer.
        /// @return Pointer to the cached InnerGrid.
        const InnerGrid* lastInnerGrid() const {
            return prev_inner_ptr_;
        }

        /// @brief Returns the most recently accessed LeafGrid pointer.
        /// @return Pointer to the cached LeafGrid.
        const LeafGrid* lastLeafGrid() const {
            return prev_leaf_ptr_;
        }
    };

    /// @brief Creates a new Accessor instance for this grid.
    /// @return An Accessor object.
    Accessor createAccessor() {
        return Accessor(root_map);
    }

    /// @brief Computes the key for the RootMap based on global coordinates.
    /// @param coord The global grid coordinate.
    /// @return The base coordinate for the root key (masked).
    static inline Vec3i getRootKey(const Vec3i& coord) {
        constexpr static int32_t MASK = ~((1 << Log2N) - 1);
        return {coord.x & MASK, coord.y & MASK, coord.z & MASK};
    }

    /// @brief Computes the key for locating an InnerGrid (intermediate block).
    /// @param coord The global grid coordinate.
    /// @return The coordinate masked to the InnerGrid resolution.
    static inline Vec3i getInnerKey(const Vec3i &coord)
    {
        constexpr static int32_t MASK = ~((1 << LEAF_BITS) - 1);
        return {coord.x & MASK, coord.y & MASK, coord.z & MASK};
    }

    /// @brief Computes the linear index within an InnerGrid for a given coordinate.
    /// @param coord The global grid coordinate.
    /// @return The linear index (0 to size of InnerGrid).
    static inline uint32_t getInnerIndex(const Vec3i &coord)
    {
        constexpr static int32_t MASK = ((1 << INNER_BITS) - 1);
        // clang-format off
        return ((coord.x >> LEAF_BITS) & MASK) +
               (((coord.y >> LEAF_BITS) & MASK) <<  INNER_BITS) +
               (((coord.z >> LEAF_BITS) & MASK) << (INNER_BITS * 2));
        // clang-format on
    }

    /// @brief Computes the linear index within a LeafGrid for a given coordinate.
    /// @param coord The global grid coordinate.
    /// @return The linear index (0 to size of LeafGrid).
    static inline uint32_t getLeafIndex(const Vec3i &coord)
    {
        constexpr static int32_t MASK = ((1 << LEAF_BITS) - 1);
        // clang-format off
        return (coord.x & MASK) +
               ((coord.y & MASK) <<  LEAF_BITS) +
               ((coord.z & MASK) << (LEAF_BITS * 2));
        // clang-format on
    }

    /// @brief Sets the color of a voxel at a specific world position.
    /// @details Assumes DataT is compatible with Vec3ui8.
    /// @param worldPos The 3D world position.
    /// @param color The color value to set.
    /// @return True if the voxel previously existed, false if created.
    bool setVoxelColor(const Vec3d& worldPos, const Vec3ui8& color) {
        Vec3i coord = posToCoord(worldPos);
        Accessor accessor = createAccessor();
        return accessor.setValue(coord, color);
    }
    
    /// @brief Retrieves the color of a voxel at a specific world position.
    /// @details Assumes DataT is compatible with Vec3ui8.
    /// @param worldPos The 3D world position.
    /// @return Pointer to the color if exists, nullptr otherwise.
    Vec3ui8* getVoxelColor(const Vec3d& worldPos) {
        Vec3i coord = posToCoord(worldPos);
        Accessor accessor = createAccessor();
        return accessor.value(coord);
    }
    
    /// @brief Renders the grid to an RGB buffer 
    /// @details Iterates all cells and projects them onto a 2D plane defined by viewDir and upDir.
    /// @param buffer The output buffer (will be resized to width * height * 3).
    /// @param width Width of the output image.
    /// @param height Height of the output image.
    /// @param viewOrigin the position of the camera
    /// @param viewDir The direction the camera is looking.
    /// @param upDir The up vector of the camera.
    /// @param fov the field of view for the camera
    void renderToRGB(std::vector<uint8_t>& buffer, int width, int height, const Vec3d& viewOrigin, 
                 const Vec3d& viewDir, const Vec3d& upDir, float fov = 80) {
        TIME_FUNCTION;
        buffer.resize(width * height * 3);
        std::fill(buffer.begin(), buffer.end(), 0);
        
        // Normalize view direction and compute right vector
        Vec3d viewDirN = viewDir.normalized();
        Vec3d upDirN = upDir.normalized();
        Vec3d rightDir = viewDirN.cross(upDirN).normalized();
        
        // Recompute up vector to ensure orthogonality
        Vec3d realUpDir = rightDir.cross(viewDirN).normalized();
        
        // Compute focal length based on FOV
        double aspectRatio = static_cast<double>(width) / static_cast<double>(height);
        double fovRad = fov * M_PI / 180.0;
        double focalLength = 0.5 / tan(fovRad * 0.5); // Reduced for wider view
        
        // Pixel to world scaling
        double pixelWidth = 2.0 * focalLength / width;
        double pixelHeight = 2.0 * focalLength / height;
        
        // Create an accessor for efficient voxel lookup
        Accessor accessor = createAccessor();
        
        // For each pixel in the output image
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Calculate pixel position in camera space
                double u = (x - width * 0.5) * pixelWidth;
                double v = (height * 0.5 - y) * pixelHeight;
                
                // Compute ray direction in world space
                Vec3d rayDirWorld = viewDirN * focalLength + 
                                rightDir * u + 
                                realUpDir * v;
                rayDirWorld = rayDirWorld.normalized();
                
                // Set up ray marching
                Vec3d rayPos = viewOrigin;
                double maxDistance = 1000.0; // Increased maximum ray distance
                double stepSize = resolution * 0.5; // Smaller step size
                
                // Ray marching loop
                bool hit = false;
                for (double t = 0; t < maxDistance && !hit; t += stepSize) {
                    rayPos = viewOrigin + rayDirWorld * t;
                    
                    // Check if we're inside the grid bounds
                    if (rayPos.x < 0 || rayPos.y < 0 || rayPos.z < 0 ||
                        rayPos.x >= 128 || rayPos.y >= 128 || rayPos.z >= 128) {
                        continue;
                    }
                    
                    // Convert world position to voxel coordinate
                    Vec3i coord = posToCoord(rayPos);
                    
                    // Look up voxel value using accessor
                    DataT* voxelData = accessor.value(coord);
                    
                    if (voxelData) {
                        // Voxel hit - extract color
                        Vec3ui8* colorPtr = reinterpret_cast<Vec3ui8*>(voxelData);
                        
                        // Get buffer index for this pixel
                        size_t pixelIdx = (y * width + x) * 3;
                        
                        // Simple distance-based attenuation
                        double distance = t;
                        double attenuation = 1.0 / (1.0 + distance * 0.01);
                        
                        // Store color in buffer with attenuation
                        buffer[pixelIdx]     = static_cast<uint8_t>(colorPtr->x * attenuation);
                        buffer[pixelIdx + 1] = static_cast<uint8_t>(colorPtr->y * attenuation);
                        buffer[pixelIdx + 2] = static_cast<uint8_t>(colorPtr->z * attenuation);
                        
                        hit = true;
                        break; // Stop ray marching after hitting first voxel
                    }
                }
            }
        }
    }
};