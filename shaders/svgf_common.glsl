
#ifndef SVGF_COMMON_GLSL
#define SVGF_COMMON_GLSL

const int PIX_STRIDE  = 5;
const int GBUF_STRIDE = 8;
const int HIST_STRIDE = 12;

const float SVGF_SKY_DEPTH = 999000.0;

float svgfLum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

bool svgfIsSky(float depth) { return depth >= SVGF_SKY_DEPTH; }

bool svgfInvalid(vec3 c) {
    return any(isnan(c)) || any(isinf(c)) || any(lessThan(c, vec3(-1e-3)));
}

vec3 svgfAlbedoFloor(vec3 a) { return max(a, vec3(0.02)); }

#endif
