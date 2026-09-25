// Shared declarations for the GPU physics pipeline (PBF fluids and shape
// matched rigid bodies). Every phys_*.comp includes this. Bindings must
// match PhysGpu in physics_gpu.inl.

layout(local_size_x = 256) in;

const uint TYPE_FLUID = 1u;
const uint TYPE_RIGID = 2u;
const uint TYPE_SOFT  = 3u;
const uint NO_OBJ = 0xFFFFFFFFu;
const float PI = 3.14159265358979f;

// Per rigid or soft object. start/count index the particle range, cm is the
// rest centre of mass on upload and the current one after the reduce, R is
// the rotation carried between substeps, dv/dw are the restitution response
// applied in finalize. Scalar layout so std430 packs it without padding.
struct Obj {
    uint start;
    uint count;
    uint jointBonds;
    uint fiberBonds;
    float alpha;
    float cm[3];
    float R[9];
    float dv[3];
    float dw[3];
    float mass;
};

struct IBond {
    uint a;
    uint b;
    float rest;
    float tension;
    float compression;
    float stiffness;
    uint broken;
    uint pad;
};

// other is a particle index unless isFixed, then fixedPos is the anchor.
// kind 0 = joint or anchor, 1 = muscle fibre.
struct XBond {
    uint other;
    uint isFixed;
    float rest;
    float stiffness;
    vec3 fixedPos;
    uint kind;
};

// xyz = position at substep start, w = size
layout(std430, binding = 0) buffer PosB { vec4 pos[]; };
// xyz = velocity, w = mass
layout(std430, binding = 1) buffer VelB { vec4 vel[]; };
// xyz = predicted position, w = lambda
layout(std430, binding = 2) buffer PredB { vec4 pred[]; };
// xyz = pending displacement
layout(std430, binding = 3) buffer DispB { vec4 disp[]; };
// x = object slot, y = type, z = hash cell, w = floatBitsToUint(restitution)
layout(std430, binding = 4) buffer InfoB { uvec4 info[]; };
// xyz = rest position relative to object rest cm, w = friction
layout(std430, binding = 5) buffer RestB { vec4 rest[]; };
layout(std430, binding = 6) buffer CountB { uint cellCount[]; };
layout(std430, binding = 7) buffer SortB { uint sorted[]; };
layout(std430, binding = 8) buffer ObjB { Obj objs[]; };
layout(std430, binding = 9) buffer IBondB { IBond ibonds[]; };
layout(std430, binding = 10) buffer XBondB { XBond xbonds[]; };
// CSR offsets into xbonds, numParticles + 1
layout(std430, binding = 11) buffer XOffB { uint xoff[]; };
// occupancy bit grid of static voxels
layout(std430, binding = 12) buffer OccB { uint occ[]; };
// 3 x vec4 per static primitive, see collectStaticVoxels
layout(std430, binding = 13) buffer StatB { vec4 statics[]; };
// xyz = last solid contact normal, w = 1 if touching
layout(std430, binding = 14) buffer CNormB { vec4 cnorm[]; };
layout(std430, binding = 15) buffer VelTmpB { vec4 velTmp[]; };
// exclusive prefix of cellCount, numCells + 1
layout(std430, binding = 16) buffer StartB { uint cellStart[]; };

// FLAG_GRAVITY_POINT: gravity pulls toward gravity instead of along it.
// FLAG_SOLID_BOUNDARY: particles are clamped to the domain box.
const uint FLAG_GRAVITY_POINT = 1u;
const uint FLAG_SOLID_BOUNDARY = 2u;

layout(push_constant) uniform PC {
    vec3 gridLo;
    float occCell;
    uvec3 gridDim;
    uint numParticles;
    vec3 gravity;
    float gravityStrength;
    vec3 domLo;
    float dt;
    vec3 domHi;
    float h;
    float hashCellSize;
    uint numHashCells;
    float damping;
    float xsph;
    float surfaceTension;
    uint count;
    uint iteration;
    uint flags;
} pc;

ivec3 hashCoord(vec3 p) {
    return ivec3(floor(p / pc.hashCellSize));
}

uint hashCell(ivec3 c) {
    uint n = pc.numHashCells;
    uint hsh = uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u;
    return hsh & (n - 1u);
}

// poly6 and spiky kernels with support radius h
float W(float r) {
    float hh = pc.h;
    if (r >= hh) return 0.0;
    float d = hh * hh - r * r;
    return 315.0 / (64.0 * PI * pow(hh, 9.0)) * d * d * d;
}

vec3 gradW(vec3 d, float r) {
    float hh = pc.h;
    if (r >= hh || r < 1e-6) return vec3(0.0);
    float t = hh - r;
    return -45.0 / (PI * pow(hh, 6.0)) * t * t * (d / r);
}

bool occupiedCell(ivec3 c) {
    if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(pc.gridDim)))) return false;
    uint idx = uint(c.x) + uint(c.y) * pc.gridDim.x + uint(c.z) * pc.gridDim.x * pc.gridDim.y;
    return (occ[idx >> 5u] & (1u << (idx & 31u))) != 0u;
}

ivec3 occCoord(vec3 p) {
    return ivec3(floor((p - pc.gridLo) / pc.occCell));
}

bool occupiedAt(vec3 p) {
    return occupiedCell(occCoord(p));
}

// Sweeps x0 -> xs through the occupancy grid (Amanatides-Woo). If a solid
// cell is entered the point is stopped at the entry face. Returns true on
// hit and writes the face normal.
bool sweepSolid(vec3 x0, inout vec3 xs, out vec3 n) {
    n = vec3(0.0);
    vec3 d = xs - x0;
    float len = length(d);
    if (len < 1e-7) return false;
    // a start already inside a solid is left to pushOutSolid
    if (occupiedAt(x0)) return false;
    vec3 dir = d / len;
    float cs = pc.occCell;
    ivec3 c = occCoord(x0);
    ivec3 stepv = ivec3(sign(dir));
    vec3 cellMin = pc.gridLo + vec3(c) * cs;
    vec3 tMax;
    vec3 tDelta;
    for (int a = 0; a < 3; ++a) {
        if (abs(dir[a]) < 1e-9) {
            tMax[a] = 1e30;
            tDelta[a] = 1e30;
            continue;
        }
        float edge = (dir[a] > 0.0) ? cellMin[a] + cs : cellMin[a];
        tMax[a] = (edge - x0[a]) / dir[a];
        tDelta[a] = cs / abs(dir[a]);
    }
    for (int i = 0; i < 128; ++i) {
        int ax = (tMax.x < tMax.y) ? ((tMax.x < tMax.z) ? 0 : 2) : ((tMax.y < tMax.z) ? 1 : 2);
        float t = tMax[ax];
        if (t >= len) return false;
        c[ax] += stepv[ax];
        if (occupiedCell(c)) {
            n = vec3(0.0);
            n[ax] = -float(stepv[ax]);
            xs = x0 + dir * max(t - 1e-4, 0.0);
            return true;
        }
        tMax[ax] += tDelta[ax];
    }
    return false;
}

// Pushes a sphere of radius r at xs out of every nearby solid cell and
// returns the summed push normal.
vec3 pushOutSolid(inout vec3 xs, float r, vec3 fallbackDir) {
    float cs = pc.occCell;
    int reach = min(2, int(ceil(r / cs)));
    vec3 nsum = vec3(0.0);
    ivec3 c0 = occCoord(xs);
    for (int dz = -reach; dz <= reach; ++dz)
    for (int dy = -reach; dy <= reach; ++dy)
    for (int dx = -reach; dx <= reach; ++dx) {
        ivec3 c = c0 + ivec3(dx, dy, dz);
        if (!occupiedCell(c)) continue;
        vec3 lo = pc.gridLo + vec3(c) * cs;
        vec3 hi = lo + cs;
        vec3 q = clamp(xs, lo, hi);
        vec3 dv = xs - q;
        float dist = length(dv);
        if (dist >= r) continue;
        if (dist > 1e-6) {
            vec3 nn = dv / dist;
            xs += nn * (r - dist);
            nsum += nn;
        } else {
            // centre inside the cell: leave through the nearest face
            vec3 pen = min(xs - lo, hi - xs);
            int ax = (pen.x < pen.y) ? ((pen.x < pen.z) ? 0 : 2) : ((pen.y < pen.z) ? 1 : 2);
            vec3 nn = vec3(0.0);
            float s = (xs[ax] - lo[ax] < hi[ax] - xs[ax]) ? -1.0 : 1.0;
            if (dot(fallbackDir, fallbackDir) > 0.0 && abs(fallbackDir[ax]) > 0.5) s = sign(fallbackDir[ax]);
            nn[ax] = s;
            xs += nn * (pen[ax] + r);
            nsum += nn;
        }
    }
    return nsum;
}

// Full solid collision for one particle: sweep from its substep start
// position, push out, clamp to the domain. Writes the contact normal into
// cnorm[i] with w = 1 when anything was touched.
void collideSolid(uint i, inout vec3 xs) {
    vec3 x0 = pos[i].xyz;
    float r = pos[i].w * 0.5;
    vec3 n;
    vec3 nsum = vec3(0.0);
    float hit = 0.0;
    if (sweepSolid(x0, xs, n)) {
        nsum += n;
        hit = 1.0;
    }
    vec3 pn = pushOutSolid(xs, r, nsum);
    if (dot(pn, pn) > 0.0) {
        nsum += pn;
        hit = 1.0;
    }
    if ((pc.flags & FLAG_SOLID_BOUNDARY) != 0u) {
        vec3 lo = pc.domLo + r;
        vec3 hi = pc.domHi - r;
        vec3 cl = clamp(xs, lo, hi);
        if (cl != xs) {
            nsum += sign(cl - xs);
            hit = 1.0;
            xs = cl;
        }
    }
    if (hit > 0.0) {
        float l = length(nsum);
        cnorm[i] = vec4(l > 1e-6 ? nsum / l : vec3(0.0, 0.0, 1.0), 1.0);
    }
}

vec3 objCm(uint o) {
    return vec3(objs[o].cm[0], objs[o].cm[1], objs[o].cm[2]);
}

mat3 objR(uint o) {
    return mat3(objs[o].R[0], objs[o].R[1], objs[o].R[2],
                objs[o].R[3], objs[o].R[4], objs[o].R[5],
                objs[o].R[6], objs[o].R[7], objs[o].R[8]);
}

vec3 gravityAt(vec3 p) {
    if ((pc.flags & FLAG_GRAVITY_POINT) != 0u) {
        vec3 d = pc.gravity - p;
        float l = length(d);
        return (l > 1e-4) ? d / l * pc.gravityStrength : vec3(0.0);
    }
    return pc.gravity;
}
