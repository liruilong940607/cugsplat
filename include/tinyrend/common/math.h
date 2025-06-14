#pragma once

#include <cstdint>

#include "tinyrend/common/macros.h"

namespace tinyrend {

template <typename T> inline TREND_HOST_DEVICE T rsqrt(const T x) {
#ifdef __CUDACC__
    return ::rsqrt(x); // use CUDA's fast rsqrt()
#else
    return 1.0 / std::sqrt(x); // use standard sqrt on CPU
#endif
}

} // namespace tinyrend