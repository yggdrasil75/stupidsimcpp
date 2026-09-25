#pragma once
namespace Grid {

enum class ShapeType : uint8_t {
    AABB     = 0,
    OOBB     = 1,
    SPHERE   = 2,
    CAPSULE  = 3,
    CYLINDER = 4,
    TRI      = 5,
    COUNT
};

struct ShapeParams {
    float qx = 0.0f, qy = 0.0f, qz = 0.0f, qw = 1.0f;
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    float e1x = 0.0f, e1y = 0.0f, e1z = 0.0f;
    float e2x = 0.0f, e2y = 0.0f, e2z = 0.0f;

    Eigen::Quaternionf quat() const { return Eigen::Quaternionf(qw, qx, qy, qz); }
    void setQuat(const Eigen::Quaternionf& q) { qx = q.x(); qy = q.y(); qz = q.z(); qw = q.w(); }
    Vec3 abc() const { return Vec3(a, b, c); }
    Vec3 e1() const { return Vec3(e1x, e1y, e1z); }
    Vec3 e2() const { return Vec3(e2x, e2y, e2z); }
    void setE1(const Vec3& v) { e1x = v.x(); e1y = v.y(); e1z = v.z(); }
    void setE2(const Vec3& v) { e2x = v.x(); e2y = v.y(); e2z = v.z(); }
};

static inline BoundingBox shapeAABB(ShapeType shape, const Vec3& position, float size,
                                    uint32_t extent, const ShapeParams& p) {
    switch (shape) {
        case ShapeType::AABB: {
            const Vec3 h = Vec3::Constant(0.5f * size);
            const Vec3 lo = position - h;
            const Vec3 hi = position + h + Vec3::Constant(size).cwiseProduct(unpackExtent(extent) - Vec3::Ones());
            return {lo, hi};
        }
        case ShapeType::OOBB: {
            const Eigen::Matrix3f R = p.quat().toRotationMatrix().cwiseAbs();
            const Vec3 he = p.abc();
            const Vec3 r = R * he;
            return {position - r, position + r};
        }
        case ShapeType::SPHERE: {
            const Vec3 r = Vec3::Constant(p.a);
            return {position - r, position + r};
        }
        case ShapeType::CAPSULE:
        case ShapeType::CYLINDER: {
            const Vec3 axis = p.quat() * Vec3::UnitY();
            const Vec3 c0 = position - axis * p.b;
            const Vec3 c1 = position + axis * p.b;
            const Vec3 rad = Vec3::Constant(p.a);
            return {c0.cwiseMin(c1) - rad, c0.cwiseMax(c1) + rad};
        }
        case ShapeType::TRI: {
            const Vec3 v0 = position;
            const Vec3 v1 = position + p.abc();
            const Vec3 v2 = position + p.e1();
            Vec3 lo = v0.cwiseMin(v1).cwiseMin(v2);
            Vec3 hi = v0.cwiseMax(v1).cwiseMax(v2);
            return {lo, hi};
        }
        default: {
            const Vec3 h = Vec3::Constant(0.5f * size);
            return {position - h, position + h};
        }
    }
}

namespace detail {
    template<typename T, typename = void>
    struct has_member_merge : std::false_type {};
    template<typename T>
    struct has_member_merge<T, std::void_t<decltype(T::merge(std::declval<const T&>(), std::declval<const T&>()))>>
        : std::true_type {};

    template<typename T, typename = void>
    struct has_adl_merge : std::false_type {};
    template<typename T>
    struct has_adl_merge<T, std::void_t<decltype(merge(std::declval<const T&>(), std::declval<const T&>()))>>
        : std::true_type {};
}

template<typename T>
inline T mergePayload(const T& a, const T& b) {
    if constexpr (detail::has_member_merge<T>::value) {
        return T::merge(a, b);
    } else if constexpr (detail::has_adl_merge<T>::value) {
        return merge(a, b);
    } else {
        (void)b;
        return a;
    }
}

template<typename T>
inline constexpr bool hasMergeFunction = detail::has_member_merge<T>::value || detail::has_adl_merge<T>::value;

}