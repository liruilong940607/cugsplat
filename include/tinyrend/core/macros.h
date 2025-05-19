#pragma once

namespace tinyrend {

#ifdef __CUDACC__
#define GSPLAT_HOST_DEVICE __host__ __device__
#else
#define GSPLAT_HOST_DEVICE
#endif

} // namespace tinyrend