#ifndef SHAPES_GLSL
#define SHAPES_GLSL

///@brief One render point. Layout matches GPURenderData in rendering.inl.
///       shapeType 0: run of `extent` cubes of `size` starting at position
///       shapeType 1: oriented box, half extents = size * unpackExtent(extent) / 2
///       shapeType 2: capsule, radius a, half length b, axis = rot * +Y
struct GPURenderData {
    vec3 position;
    float size;
    uint color;
    uint materialIdx;
    int  objectId;
    uint extent;
    uint rot;
    float a;
    float b;
    uint shapeType;
};

const uint SHAPE_BOX = 0u;
const uint SHAPE_OBB = 1u;
const uint SHAPE_CAPSULE = 2u;

const uint EXTENT_STATIC_BIT = 1u << 30;
const uint EXTENT_REUSE_BIT  = 1u << 31;

bool extentIsStatic(uint e) {
    return (e & EXTENT_STATIC_BIT) != 0u;
}

bool extentIsReusable(uint e) {
    return (e & EXTENT_REUSE_BIT) != 0u;
}

///@brief Per-axis cell counts stored in an extent word
vec3 unpackExtent(uint e) {
    return vec3(float((e & 0x3FFu) + 1u), float(((e >> 10) & 0x3FFu) + 1u), float(((e >> 20) & 0x3FFu) + 1u));
}

///@brief Inverse of packQuat in shapes.inl, returns (x, y, z, w)
vec4 unpackQuat(uint p) {
    uint big = p & 3u;
    float c[4];
    float sum = 0.0;
    uint shift = 2u;
    for (int i = 0; i < 4; ++i) {
        if (uint(i) == big) {
            c[i] = 0.0;
            continue;
        }
        float v = (float((p >> shift) & 0x3FFu) / 1023.0 * 2.0 - 1.0) * 0.70710678;
        c[i] = v;
        sum += v * v;
        shift += 10u;
    }
    c[big] = sqrt(max(0.0, 1.0 - sum));
    return vec4(c[0], c[1], c[2], c[3]);
}

vec3 quatRotate(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

vec3 quatInvRotate(vec4 q, vec3 v) {
    return quatRotate(vec4(-q.xyz, q.w), v);
}

///@brief OBB half extents in world units
vec3 shapeHalf(GPURenderData p) {
    return unpackExtent(p.extent) * (0.5 * p.size);
}

///@brief Half diagonal of the world AABB of an OBB or capsule
vec3 shapeAabbRadius(GPURenderData p) {
    vec4 q = unpackQuat(p.rot);
    if (p.shapeType == SHAPE_CAPSULE) {
        vec3 e = quatRotate(q, vec3(0.0, p.b, 0.0));
        return abs(e) + vec3(p.a);
    }
    vec3 h = shapeHalf(p);
    return abs(quatRotate(q, vec3(h.x, 0.0, 0.0)))
         + abs(quatRotate(q, vec3(0.0, h.y, 0.0)))
         + abs(quatRotate(q, vec3(0.0, 0.0, h.z)));
}

vec3 ptBoundsMin(GPURenderData p) {
    if (p.shapeType != SHAPE_BOX) return p.position - shapeAabbRadius(p);
    return p.position - p.size * 0.5;
}

vec3 ptBoundsMax(GPURenderData p) {
    if (p.shapeType != SHAPE_BOX) return p.position + shapeAabbRadius(p);
    return p.position + p.size * 0.5 + p.size * (unpackExtent(p.extent) - vec3(1.0));
}

///@brief Slab test against a box centred on the origin
///@param n Receives the face normal at the entry (or exit when inside)
bool slabBox(vec3 o, vec3 d, vec3 h, out float tIn, out float tOut, out vec3 n) {
    vec3 invD = 1.0 / d;
    vec3 t0 = (-h - o) * invD;
    vec3 t1 = ( h - o) * invD;
    vec3 tmin3 = min(t0, t1);
    vec3 tmax3 = max(t0, t1);
    tIn  = max(max(tmin3.x, tmin3.y), tmin3.z);
    tOut = min(min(tmax3.x, tmax3.y), tmax3.z);
    n = vec3(0.0);
    if (tOut < max(0.0, tIn)) return false;
    bool inside = tIn < 0.0;
    vec3 slab = inside ? tmax3 : tmin3;
    float key = inside ? tOut : tIn;
    vec3 sgn = vec3(d.x < 0.0 ? 1.0 : -1.0, d.y < 0.0 ? 1.0 : -1.0, d.z < 0.0 ? 1.0 : -1.0);
    vec3 mask = vec3(0.0, 0.0, 1.0);
    if (key == slab.x) mask = vec3(1.0, 0.0, 0.0);
    else if (key == slab.y) mask = vec3(0.0, 1.0, 0.0);
    n = (inside ? -sgn : sgn) * mask;
    return true;
}

///@brief Ray interval through a capsule in its local frame: segment (0,-h,0)-(0,h,0), radius r.
///       Union of a clipped cylinder and two spheres, all convex and overlapping.
bool rayCapsuleLocal(vec3 o, vec3 d, float r, float h, out float tIn, out float tOut) {
    bool any = false;
    tIn = 1e30;
    tOut = -1e30;
    float a = d.x * d.x + d.z * d.z;
    float b = 2.0 * (o.x * d.x + o.z * d.z);
    float c = o.x * o.x + o.z * o.z - r * r;
    if (a > 1e-12) {
        float disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
            float sq = sqrt(disc);
            float t0 = (-b - sq) / (2.0 * a);
            float t1 = (-b + sq) / (2.0 * a);
            if (abs(d.y) > 1e-12) {
                float ta = (-h - o.y) / d.y;
                float tb = (h - o.y) / d.y;
                float lo = max(t0, min(ta, tb));
                float hi = min(t1, max(ta, tb));
                if (hi >= lo) {
                    any = true;
                    tIn = min(tIn, lo);
                    tOut = max(tOut, hi);
                }
            } else if (abs(o.y) <= h) {
                any = true;
                tIn = min(tIn, t0);
                tOut = max(tOut, t1);
            }
        }
    } else if (c <= 0.0 && abs(d.y) > 1e-12) {
        float ta = (-h - o.y) / d.y;
        float tb = (h - o.y) / d.y;
        any = true;
        tIn = min(tIn, min(ta, tb));
        tOut = max(tOut, max(ta, tb));
    }
    for (int i = 0; i < 2; ++i) {
        vec3 oc = o - vec3(0.0, i == 0 ? -h : h, 0.0);
        float b2 = dot(oc, d);
        float c2 = dot(oc, oc) - r * r;
        float disc = b2 * b2 - c2;
        if (disc < 0.0) continue;
        float sq = sqrt(disc);
        any = true;
        tIn = min(tIn, -b2 - sq);
        tOut = max(tOut, -b2 + sq);
    }
    return any && tOut >= max(0.0, tIn);
}

///@brief Entry hit for any primitive (exit when the origin is inside, normal flipped) plus exit distance
bool rayPrimIntersect(vec3 ro, vec3 rd, GPURenderData pt, out float t, out vec3 normal, out float tExit) {
    float tIn;
    float tOut;
    vec3 n;
    t = 0.0;
    normal = vec3(0.0);
    tExit = 0.0;
    if (pt.shapeType == SHAPE_CAPSULE) {
        vec4 q = unpackQuat(pt.rot);
        vec3 o = quatInvRotate(q, ro - pt.position);
        vec3 d = quatInvRotate(q, rd);
        if (!rayCapsuleLocal(o, d, pt.a, pt.b, tIn, tOut)) return false;
        bool inside = tIn < 0.0;
        t = inside ? tOut : tIn;
        vec3 l = o + d * t;
        l.y -= clamp(l.y, -pt.b, pt.b);
        n = quatRotate(q, normalize(l));
        if (inside) n = -n;
    } else if (pt.shapeType == SHAPE_OBB) {
        vec4 q = unpackQuat(pt.rot);
        vec3 o = quatInvRotate(q, ro - pt.position);
        vec3 d = quatInvRotate(q, rd);
        if (!slabBox(o, d, shapeHalf(pt), tIn, tOut, n)) return false;
        t = (tIn < 0.0) ? tOut : tIn;
        n = quatRotate(q, n);
    } else {
        vec3 c = (ptBoundsMin(pt) + ptBoundsMax(pt)) * 0.5;
        vec3 h = (ptBoundsMax(pt) - ptBoundsMin(pt)) * 0.5;
        if (!slabBox(ro - c, rd, h, tIn, tOut, n)) return false;
        t = (tIn < 0.0) ? tOut : tIn;
    }
    tExit = tOut;
    normal = n;
    return true;
}

///@brief Entry and exit distances only
bool rayPrimInterval(vec3 ro, vec3 rd, GPURenderData pt, out float tIn, out float tOut) {
    vec3 n;
    if (pt.shapeType == SHAPE_BOX) {
        vec3 c = (ptBoundsMin(pt) + ptBoundsMax(pt)) * 0.5;
        vec3 h = (ptBoundsMax(pt) - ptBoundsMin(pt)) * 0.5;
        return slabBox(ro - c, rd, h, tIn, tOut, n);
    }
    vec4 q = unpackQuat(pt.rot);
    vec3 o = quatInvRotate(q, ro - pt.position);
    vec3 d = quatInvRotate(q, rd);
    if (pt.shapeType == SHAPE_CAPSULE) return rayCapsuleLocal(o, d, pt.a, pt.b, tIn, tOut);
    return slabBox(o, d, shapeHalf(pt), tIn, tOut, n);
}

///@brief Signed distance from a world point to the primitive surface
float primSdf(vec3 p, GPURenderData pt) {
    if (pt.shapeType == SHAPE_CAPSULE) {
        vec3 l = quatInvRotate(unpackQuat(pt.rot), p - pt.position);
        l.y -= clamp(l.y, -pt.b, pt.b);
        return length(l) - pt.a;
    }
    vec3 q;
    if (pt.shapeType == SHAPE_OBB) {
        q = abs(quatInvRotate(unpackQuat(pt.rot), p - pt.position)) - shapeHalf(pt);
    } else {
        vec3 c = (ptBoundsMin(pt) + ptBoundsMax(pt)) * 0.5;
        q = abs(p - c) - (ptBoundsMax(pt) - ptBoundsMin(pt)) * 0.5;
    }
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
#endif
