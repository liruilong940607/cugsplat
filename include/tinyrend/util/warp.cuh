#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "tinyrend/common/mat.h"
#include "tinyrend/common/vec.h"

namespace tinyrend::warp {

namespace cg = cooperative_groups;

template <uint32_t DIM, class WarpT>
inline __device__ void warpSum(float *val, WarpT &warp) {
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<float>());
    }
}

template <class WarpT> inline __device__ void warpSum(float &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(fvec4 &val, WarpT &warp) {
    val[0] = cg::reduce(warp, val[0], cg::plus<float>());
    val[1] = cg::reduce(warp, val[1], cg::plus<float>());
    val[2] = cg::reduce(warp, val[2], cg::plus<float>());
    val[3] = cg::reduce(warp, val[3], cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(fvec3 &val, WarpT &warp) {
    val[0] = cg::reduce(warp, val[0], cg::plus<float>());
    val[1] = cg::reduce(warp, val[1], cg::plus<float>());
    val[2] = cg::reduce(warp, val[2], cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(fvec2 &val, WarpT &warp) {
    val[0] = cg::reduce(warp, val[0], cg::plus<float>());
    val[1] = cg::reduce(warp, val[1], cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(fmat4 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
    warpSum(val[3], warp);
}

template <class WarpT> inline __device__ void warpSum(fmat3 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
}

template <class WarpT> inline __device__ void warpSum(fmat2 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}

template <class WarpT> inline __device__ void warpMax(float &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::greater<float>());
}

} // namespace tinyrend::warp