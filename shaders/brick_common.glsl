// Brick world traversal.  Included by brick_*.comp.
// Layout must match util/grid/brick.hpp (GPUBrickHeader, GPUPaletteEntry, flatten order).

#define BRICK_DIM 8
#define BRICK_VOXELS 512

struct GPUBrickHeader {
    int bx;
    int by;
    int bz;
    uint objectId;
    uint matWordOffset;
    uint occWordOffset;
    uint paletteOffset;
    uint voxelCount;
    float originX;
    float originY;
    float originZ;
    uint pad;
};

struct GPUPaletteEntry {
    uint colorRGBA8;
    uint renderMatIdx;
    uint physMatIdx;
    uint pad;
};

// The five brick bindings sit at BRICK_BINDING_BASE .. +4; the fast pipeline uses 11..15,
// the wavefront pipeline (whose own buffers occupy 0..21) uses 22..26.
#ifndef BRICK_BINDING_BASE
#define BRICK_BINDING_BASE 11
#endif

layout(std430, binding = BRICK_BINDING_BASE + 0) readonly buffer BrickHeaders { GPUBrickHeader bricks[]; };
layout(std430, binding = BRICK_BINDING_BASE + 1) readonly buffer BrickMats { uint brickMatWords[]; };
layout(std430, binding = BRICK_BINDING_BASE + 2) readonly buffer BrickOcc { uint brickOccWords[]; };
layout(std430, binding = BRICK_BINDING_BASE + 3) readonly buffer BrickPalette { GPUPaletteEntry palette[]; };
layout(binding = BRICK_BINDING_BASE + 4) uniform BrickParams {
    float voxelSize;
    uint brickCount;
    uint pad0;
    uint pad1;
} bp;

bool brickOccupied(uint b, ivec3 c) {
    int i = c.x + c.y * BRICK_DIM + c.z * BRICK_DIM * BRICK_DIM;
    uint word = brickOccWords[bricks[b].occWordOffset + uint(i >> 5)];
    return ((word >> uint(i & 31)) & 1u) != 0u;
}

uint brickPaletteIndex(uint b, ivec3 c) {
    int i = c.x + c.y * BRICK_DIM + c.z * BRICK_DIM * BRICK_DIM;
    uint word = brickMatWords[bricks[b].matWordOffset + uint(i >> 2)];
    return bricks[b].paletteOffset + ((word >> uint((i & 3) * 8)) & 0xFFu);
}

vec3 brickMinWorld(uint b) {
    vec3 origin = vec3(bricks[b].originX, bricks[b].originY, bricks[b].originZ);
    return origin + vec3(bricks[b].bx, bricks[b].by, bricks[b].bz) * (bp.voxelSize * float(BRICK_DIM));
}

vec3 brickVoxelCenter(uint b, ivec3 c) {
    return brickMinWorld(b) + (vec3(c) + vec3(0.5)) * bp.voxelSize;
}

struct BrickHit {
    float t;
    float tExit;
    vec3 normal;
    uint brick;
    ivec3 cell;
    uint pal;
    uint objectId;
};

struct BrickDDA {
    ivec3 cell;
    ivec3 step;
    vec3 tDelta;
    vec3 tNext;
    float t;
    vec3 normal;
    bool consumed;
};

///@brief Ray/box slab test against a brick. Returns false if the brick is missed entirely.
bool brickBounds(uint b, vec3 ro, vec3 invD, out float tIn, out float tOut) {
    vec3 bmin = brickMinWorld(b);
    vec3 bmax = bmin + vec3(float(BRICK_DIM) * bp.voxelSize);
    vec3 t0 = (bmin - ro) * invD;
    vec3 t1 = (bmax - ro) * invD;
    vec3 tmin3 = min(t0, t1);
    vec3 tmax3 = max(t0, t1);
    tIn = max(max(tmin3.x, tmin3.y), tmin3.z);
    tOut = min(min(tmax3.x, tmax3.y), tmax3.z);
    return tOut >= max(tIn, 0.0);
}

///@brief Position the iterator on the cell the ray enters at tEnter (>= 0)
void brickDDAStart(uint b, vec3 ro, vec3 rd, vec3 invD, float tEnter, out BrickDDA s) {
    const float vs = bp.voxelSize;
    vec3 bmin = brickMinWorld(b);
    vec3 p = (ro + rd * tEnter - bmin) / vs;
    // Silhouette rays graze a brick's boundary plane: the entry point lands a hair outside
    // on some axis. Clamping it into the boundary cell would report a voxel the ray never
    // crosses and give it an exit distance before its entry. Treat a clear overshoot as a
    // miss (the iterator starts outside), and never let a boundary crossing precede tEnter.
    const float slack = 1e-3;
    if (any(lessThan(p, vec3(-slack))) || any(greaterThan(p, vec3(float(BRICK_DIM) + slack)))) {
        s.cell = ivec3(-1);
        s.step = ivec3(0);
        s.tDelta = vec3(1e30);
        s.tNext = vec3(1e30);
        s.t = tEnter;
        s.normal = vec3(0.0);
        s.consumed = true;
        return;
    }
    p = clamp(p, vec3(1e-4), vec3(float(BRICK_DIM) - 1e-4));

    s.cell = ivec3(floor(p));
    s.step = ivec3(sign(rd));
    s.tDelta = abs(invD) * vs;
    vec3 nextBoundary = (vec3(s.cell) + max(vec3(s.step), vec3(0.0))) * vs + bmin;
    s.tNext = max((nextBoundary - ro) * invD, vec3(tEnter));
    if (rd.x == 0.0) s.tNext.x = 1e30;
    if (rd.y == 0.0) s.tNext.y = 1e30;
    if (rd.z == 0.0) s.tNext.z = 1e30;
    s.t = tEnter;
    s.consumed = false;

    // normal of the brick face we came through; none if the ray started inside
    vec3 t0 = (bmin - ro) * invD;
    vec3 t1 = (bmin + vec3(float(BRICK_DIM) * vs) - ro) * invD;
    vec3 tmin3 = min(t0, t1);
    float m = max(max(tmin3.x, tmin3.y), tmin3.z);
    if (tEnter <= 0.0) s.normal = vec3(0.0);
    else if (m == tmin3.x) s.normal = vec3(-float(s.step.x), 0.0, 0.0);
    else if (m == tmin3.y) s.normal = vec3(0.0, -float(s.step.y), 0.0);
    else s.normal = vec3(0.0, 0.0, -float(s.step.z));
}

///@brief Step the iterator into the next cell along the ray
void brickDDAStep(inout BrickDDA s) {
    if (s.tNext.x < s.tNext.y && s.tNext.x < s.tNext.z) {
        s.t = s.tNext.x;
        s.tNext.x += s.tDelta.x;
        s.cell.x += s.step.x;
        s.normal = vec3(-float(s.step.x), 0.0, 0.0);
    } else if (s.tNext.y < s.tNext.z) {
        s.t = s.tNext.y;
        s.tNext.y += s.tDelta.y;
        s.cell.y += s.step.y;
        s.normal = vec3(0.0, -float(s.step.y), 0.0);
    } else {
        s.t = s.tNext.z;
        s.tNext.z += s.tDelta.z;
        s.cell.z += s.step.z;
        s.normal = vec3(0.0, 0.0, -float(s.step.z));
    }
    s.consumed = false;
}

bool brickDDAInside(BrickDDA s) {
    return !(any(lessThan(s.cell, ivec3(0))) || any(greaterThanEqual(s.cell, ivec3(BRICK_DIM))));
}

bool brickDDANext(uint b, inout BrickDDA s, float tMax, out BrickHit hit) {
    if (!brickDDAInside(s)) return false;
    for (int iter = 0; iter < 3 * BRICK_DIM + 1; ++iter) {
        if (!s.consumed && brickOccupied(b, s.cell)) {
            if (s.t >= tMax) return false;
            hit.t = s.t;
            hit.normal = s.normal;
            hit.brick = b;
            hit.cell = s.cell;
            hit.pal = brickPaletteIndex(b, s.cell);
            hit.objectId = bricks[b].objectId;
            // extend over following cells of the same palette entry; opaque hits never
            // need their exit distance, so skip the walk for them
            bool opaque = (palette[hit.pal].colorRGBA8 >> 24) >= 254u;
            for (int run = 0; run < 3 * BRICK_DIM && !opaque; ++run) {
                BrickDDA peek = s;
                brickDDAStep(peek);
                if (!brickDDAInside(peek) || !brickOccupied(b, peek.cell)) break;
                if (brickPaletteIndex(b, peek.cell) != hit.pal) break;
                s = peek;
            }
            hit.tExit = min(s.tNext.x, min(s.tNext.y, s.tNext.z));
            s.consumed = true;
            return true;
        }
        brickDDAStep(s);
        if (s.t >= tMax) return false;
        if (!brickDDAInside(s)) return false;
    }
    return false;
}

///@brief First occupied voxel in brick b along the ray, or false
bool brickDDAFirst(uint b, vec3 ro, vec3 rd, vec3 invD, float tEnter, float tMax, out BrickHit hit) {
    BrickDDA s;
    brickDDAStart(b, ro, rd, invD, tEnter, s);
    return brickDDANext(b, s, tMax, hit);
}

///@brief Closest occupied voxel over the whole scene. skipObjectId excludes one object (-1 = none).
bool brickTrace(vec3 ro, vec3 rd, vec3 invD, float tMax, int skipObjectId, out BrickHit best) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoneEXT, 0xFF, ro, 0.0, rd, tMax);
    float tBest = tMax;
    bool found = false;
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) != gl_RayQueryCandidateIntersectionAABBEXT) continue;
        uint b = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, false));
        if (int(bricks[b].objectId) == skipObjectId) continue;
        float tIn, tOut;
        if (!brickBounds(b, ro, invD, tIn, tOut)) continue;
        if (tIn >= tBest) continue;
        BrickHit h;
        if (brickDDAFirst(b, ro, rd, invD, max(tIn, 0.0), min(tOut, tBest), h) && h.t < tBest) {
            tBest = h.t;
            best = h;
            found = true;
            rayQueryGenerateIntersectionEXT(rq, h.t);
        }
    }
    return found;
}

vec4 brickPaletteColor(uint pal) {
    uint c = palette[pal].colorRGBA8;
    vec4 rgba = vec4(float(c & 0xFFu), float((c >> 8) & 0xFFu), float((c >> 16) & 0xFFu), float((c >> 24) & 0xFFu));
    return rgba / 255.0;
}

// A voxel is identified across pipeline stages as brick * BRICK_VOXELS + cell, which fits
// the int slots that used to carry point-array indices.
int brickVoxelId(uint b, ivec3 c) {
    return int(b * uint(BRICK_VOXELS) + uint(c.x + c.y * BRICK_DIM + c.z * BRICK_DIM * BRICK_DIM));
}

///@brief The voxel behind a voxel id in the point-array format, so shading code that was
///       written against GPURenderData keeps working (extent is always one cell).
GPURenderData brickPoint(int id) {
    uint b = uint(id) / uint(BRICK_VOXELS);
    int i = int(uint(id) % uint(BRICK_VOXELS));
    ivec3 c = ivec3(i & 7, (i >> 3) & 7, i >> 6);
    uint pal = brickPaletteIndex(b, c);
    GPURenderData p;
    p.position = brickVoxelCenter(b, c);
    p.size = bp.voxelSize;
    p.color = palette[pal].colorRGBA8;
    p.materialIdx = palette[pal].renderMatIdx;
    p.objectId = int(bricks[b].objectId);
    p.extent = 0u;
    return p;
}
