#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include "core/macros.h"
#include "core/tensor.h"

namespace tinyrend {

template <typename T, size_t N>
inline GSPLAT_HOST_DEVICE std::array<T, N> make_array(const T *ptr, size_t offset = 0) {
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

enum class ShutterType {
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
    GLOBAL
};

enum class CameraType { PINHOLE, ORTHO, FISHEYE };

} // namespace tinyrend