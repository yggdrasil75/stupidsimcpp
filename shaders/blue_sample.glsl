#ifndef BLUE_SAMPLE_GLSL
#define BLUE_SAMPLE_GLSL

const int  BT_TILE   = 512;
const int  BT_NTILES = 512;
const int  BT_TILE_AREA = BT_TILE * BT_TILE;

layout(std430, binding = 18) readonly buffer BlueTileBuffer { float blueTiles[]; };

struct BlueState {
    uint  seqIndex;
    uint  dim;
    float keyA;
    float keyB;
};

float bt_vdc2(uint i) {
    i = (i << 16u) | (i >> 16u);
    i = ((i & 0x55555555u) << 1u) | ((i & 0xAAAAAAAAu) >> 1u);
    i = ((i & 0x33333333u) << 2u) | ((i & 0xCCCCCCCCu) >> 2u);
    i = ((i & 0x0F0F0F0Fu) << 4u) | ((i & 0xF0F0F0F0u) >> 4u);
    i = ((i & 0x00FF00FFu) << 8u) | ((i & 0xFF00FF00u) >> 8u);
    return float(i) * 2.3283064365386963e-10;
}
// Radix-3 van der Corput -> [0,1). Pairs with radix-2 to give a Halton(2,3)
// sequence: well-stratified in 2D. Integer-only, exact on all GPUs.
float bt_vdc3(uint i) {
    float f = 1.0, r = 0.0;
    // 20 digits covers our index range comfortably.
    for (int k = 0; k < 20; ++k) {
        f /= 3.0;
        r += f * float(i % 3u);
        i /= 3u;
        if (i == 0u) break;
    }
    return r;
}

// Integer hash for picking the key texel / decorrelating block repeats.
uint bt_hash(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// Fetch the blue scramble key for a pixel from a given tile.
// Pixels congruent mod 512 would otherwise read identical keys, so we offset
// the lookup by a hash of which 512-block the pixel is in.
float bt_key(ivec2 coord, uint tile) {
    uint bx = uint(coord.x) / uint(BT_TILE);
    uint by = uint(coord.y) / uint(BT_TILE);
    uint blockHash = bt_hash(bx * 73856093u ^ by * 19349663u ^ tile * 83492791u);
    uint lx = (uint(coord.x) + blockHash)            % uint(BT_TILE);
    uint ly = (uint(coord.y) + (blockHash >> 16u))   % uint(BT_TILE);
    uint idx = tile * uint(BT_TILE_AREA) + ly * uint(BT_TILE) + lx;
    return blueTiles[idx];
}

float bt_keyD(ivec2 coord, uint sampleIndex, uint dim) {
    uint tile = bt_hash(sampleIndex * 0x9E3779B9u ^ dim * 0x85EBCA6Bu)
              % uint(BT_NTILES);
    return bt_key(coord, tile);
}

BlueState blueInit(ivec2 coord, int sampleIndex) {
    BlueState s;
    s.seqIndex = uint(sampleIndex);
    s.dim      = bt_hash(uint(coord.x) ^ (uint(coord.y) << 16u)) & 0xFFFFu;
    s.keyA     = uintBitsToFloat(uint(coord.x) | (uint(coord.y) << 16u));
    s.keyB     = uintBitsToFloat(uint(sampleIndex));
    return s;
}

BlueState blueResume(uint pixelIndex, int sampleIndex, uint width,
                     uint packedSeq, uint packedDim) {
    ivec2 coord = ivec2(int(pixelIndex % width), int(pixelIndex / width));
    BlueState s = blueInit(coord, sampleIndex);
    s.seqIndex = packedSeq;
    s.dim      = packedDim;
    return s;
}

float blueNext(inout BlueState s) {
    uint packedCoord = floatBitsToUint(s.keyA);
    ivec2 coord = ivec2(int(packedCoord & 0xFFFFu), int(packedCoord >> 16u));
    uint sampleIndex = floatBitsToUint(s.keyB);

    uint pairIdx = s.seqIndex + (s.dim >> 1u) * 0x68E31DA4u;
    float u   = ((s.dim & 1u) == 0u) ? bt_vdc2(pairIdx) : bt_vdc3(pairIdx);
    float key = bt_keyD(coord + ((s.dim & 1u) == 0u ? ivec2(0) : ivec2(1)),
                        sampleIndex, s.dim);
    s.dim += 1u;
    float r = u + key;
    return r - floor(r);
}

#endif
