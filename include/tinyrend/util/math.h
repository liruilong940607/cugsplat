#pragma once

#include <cstdint>

#include "tinyrend/common/macros.h"
#include "tinyrend/common/mat.h"
#include "tinyrend/common/vec.h"

namespace tinyrend::math {

template <size_t SKIP_FIRST_N = 0, size_t N>
inline TREND_HOST_DEVICE bool is_all_zero(std::array<float, N> const &arr) {
#pragma unroll
    for (size_t i = SKIP_FIRST_N; i < N; ++i) {
        if (fabsf(arr[i]) >= std::numeric_limits<float>::epsilon())
            return false;
    }
    return true;
}

} // namespace tinyrend::math