#pragma once

namespace tinyrend {

#ifdef __CUDACC__
#define TREND_HOST_DEVICE __host__ __device__
#else
#define TREND_HOST_DEVICE
#endif

} // namespace tinyrend