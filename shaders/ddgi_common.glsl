#ifndef DDGI_COMMON_GLSL
#define DDGI_COMMON_GLSL

const int DDGI_IRR_RES    = 8;
const int DDGI_VIS_RES    = 16;
const int DDGI_IRR_BORDER = DDGI_IRR_RES + 2;
const int DDGI_VIS_BORDER = DDGI_VIS_RES + 2;
const int DDGI_PROBE_RAYS = 128;

const float DDGI_MAX_PROBE_DIST = 1e30;

struct DDGIVolume {
    vec4 originSpacing;
    ivec4 counts;
    vec4 params;
    ivec3 state;
    float fireflyClamp;
};

vec3 octToDir(vec2 o) {
    vec3 d = vec3(o.x, o.y, 1.0 - abs(o.x) - abs(o.y));
    if (d.z < 0.0) {
        float x = d.x;
        d.x = (1.0 - abs(d.y)) * (x >= 0.0 ? 1.0 : -1.0);
        d.y = (1.0 - abs(x)) * (d.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(d);
}

vec2 dirToOct(vec3 d) {
    float s = abs(d.x) + abs(d.y) + abs(d.z);
    vec2 o = d.xy / max(1e-8, s);
    if (d.z < 0.0) {
        float x = o.x;
        o.x = (1.0 - abs(o.y)) * (x >= 0.0 ? 1.0 : -1.0);
        o.y = (1.0 - abs(x)) * (o.y >= 0.0 ? 1.0 : -1.0);
    }
    return o;
}

vec3 ddgiRayDir(int rayIdx, int rayCount, uint frameSeed) {
    const float PHI_G = 1.6180339887;
    float i = float(rayIdx) + 0.5;
    float cosT = 1.0 - 2.0 * i / float(rayCount);
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float phi = 2.0 * PI * fract(i * PHI_G + float(frameSeed) * 0.6180339887);
    return normalize(vec3(cos(phi) * sinT, sin(phi) * sinT, cosT));
}

ivec3 ddgiProbeCoord(int probeIdx, ivec3 counts) {
    int xy = counts.x * counts.y;
    return ivec3(probeIdx % counts.x, (probeIdx / counts.x) % counts.y, probeIdx / xy);
}

int ddgiProbeIndex(ivec3 c, ivec3 counts) {
    return c.x + c.y * counts.x + c.z * counts.x * counts.y;
}

vec3 ddgiProbePosition(ivec3 c, DDGIVolume v) {
    return v.originSpacing.xyz + vec3(c) * v.originSpacing.w;
}

int ddgiIrrBase(int probeIdx) { return probeIdx * DDGI_IRR_BORDER * DDGI_IRR_BORDER; }
int ddgiVisBase(int probeIdx) { return probeIdx * DDGI_VIS_BORDER * DDGI_VIS_BORDER; }

#endif
