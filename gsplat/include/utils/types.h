#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include "utils/macros.h"

namespace gsplat {

template <typename T> struct Maybe {
    bool _has_value = false;
    T _value;

    GSPLAT_HOST_DEVICE inline T get() const {
        return this->_has_value ? this->_value : T{};
    }

    GSPLAT_HOST_DEVICE inline bool has_value() const {
        return this->_has_value;
    }

    GSPLAT_HOST_DEVICE inline void set(const T &v) {
        this->_value = v;
        this->_has_value = true;
    }
};

template <typename T, size_t N>
inline GSPLAT_HOST_DEVICE std::array<T, N>
make_array(const T *ptr, size_t offset = 0) {
    std::array<T, N> arr{}; // zero-initialize
    if (!ptr) {
        return arr;
    }
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = ptr[offset + i];
    }
    return arr;
}

struct MaybeValidRay {
    glm::fvec3 o;
    glm::fvec3 d;
    bool valid_flag;
};

struct MaybeValidPoint3D {
    glm::fvec3 p;
    bool valid_flag;
};

struct MaybeValidPoint2D {
    glm::fvec2 p;
    bool valid_flag;
};

struct MaybeValidGaussian3D {
    glm::fvec3 mean;
    glm::fmat3 covar;
    bool valid_flag;
};

struct MaybeValidGaussian2D {
    glm::fvec2 mean;
    glm::fmat2 covar;
    bool valid_flag;
};

struct SE3Mat {
    glm::fvec3 t;
    glm::fmat3 R;

    GSPLAT_HOST_DEVICE SE3Mat() {}
    GSPLAT_HOST_DEVICE SE3Mat(glm::fvec3 t, glm::fmat3 R) : t(t), R(R) {}
};

struct SE3Quat {
    glm::fvec3 t;
    glm::fquat q;

    GSPLAT_HOST_DEVICE SE3Quat() {}
    GSPLAT_HOST_DEVICE SE3Quat(glm::fvec3 t, glm::fquat q) : t(t), q(q) {}
};

enum class ShutterType {
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
    GLOBAL
};

enum class CameraType { PINHOLE, ORTHO, FISHEYE };

} // namespace gsplat