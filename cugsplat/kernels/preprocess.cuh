#include <glm/glm.hpp>
#include <cub/cub.cuh>

namespace cugsplat {

using namespace glm;

inline __device__ auto compute_radius(float opacity, fmat2 covar) -> fvec2 {
    constexpr float alpha_threshold = 1.0f / 255.0f;
    if (opacity < alpha_threshold) {
        return fvec2(0.0f, 0.0f);
    }

    // Compute opacity-aware bounding box.
    // https://arxiv.org/pdf/2402.00525 Section B.2
    float extend = 3.33f;
    extend = min(extend, sqrt(2.0f * __logf(opacity / alpha_threshold)));

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    auto const b = 0.5f * (covar[0][0] + covar[1][1]);
    auto const det = determinant(covar);
    auto const tmp = sqrtf(max(0.01f, b * b - det));
    auto const v1 = b + tmp; // larger eigenvalue
    auto const r1 = extend * sqrtf(v1);
    auto const radius_x = ceilf(min(extend * sqrtf(covar[0][0]), r1));
    auto const radius_y = ceilf(min(extend * sqrtf(covar[1][1]), r1));
    return fvec2(radius_x, radius_y);
}

struct PreprocessParameters {
    uint32_t render_width;
    uint32_t render_height;
    float near_plane;
    float far_plane;
    float margin_factor;
    float filter_size;
};

template <
class DeviceCameraModel, 
class DeviceGaussianIn, 
class DeviceGaussianOut, 
bool PACKED, 
// Total number of threads in a block, only used in PACKED mode
int NUM_THREADS>
__global__ void PreprocessKernel(
    const DeviceCameraModel d_camera,
    const DeviceGaussianIn d_gaussians_in,
    const PreprocessParameters& params,
    // outputs
    DeviceGaussianOut d_gaussians_out,
    int32_t* __restrict__ block_cnts,
    int32_t* __restrict__ block_offsets
) {
    // Parallelize over [num_cameras, num_gaussians]. should be launched
    // as a 2D grid of threads, with each thread processing a single
    // camera and a single gaussian.

    // Kernel should be launched with dim3(blockDim.x, blockDim.y) and 
    // dim3(gridDim.x, gridDim.y) such that threadIdx.z == 0 and blockIdx.z == 0
    auto const cidx = blockIdx.x * blockDim.x + threadIdx.x; // camera index
    auto const pidx = blockIdx.y * blockDim.y + threadIdx.y; // gaussian index
    if (cidx >= d_camera.get_n() || pidx >= d_gaussians_in.get_n()) {
        return;
    }

    // Shift pointers
    d_camera.set_index(cidx);
    d_gaussians_in.set_index(pidx);

    // Preprocess the gaussian. Results are saved in `d_gaussians_out` locally.
    auto const valid_flag = d_gaussians_out.preprocess(
        d_camera, d_gaussians_in, params
    );

    if constexpr (PACKED) {
        auto const block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        int32_t thread_data = static_cast<int32_t>(valid_flag);
        if (block_cnts != nullptr) {
            // First pass: compute the block-wide sum. I.E How many gaussians will be output
            // by this block of threads. 
            int32_t aggregate = 0;
            if (__syncthreads_or(thread_data)) {
                typedef cub::BlockReduce<int32_t, NUM_THREADS> BlockReduce;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                aggregate = BlockReduce(temp_storage).Sum(thread_data);
            }
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                block_cnts[block_idx] = aggregate;
            }
            return;
        } else {
            // Second pass: write the gaussian to the output buffer
            if (__syncthreads_or(thread_data)) {
                typedef cub::BlockScan<int32_t, NUM_THREADS> BlockScan;
                __shared__ typename BlockScan::TempStorage temp_storage;
                BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
            }
            thread_data += block_offsets[block_idx];
            if (valid_flag) {
                // Write the gaussian to the output buffer
                d_gaussians_out.set_index(thread_data);
                d_gaussians_out.export();
            }
        }
    } else {
        if (valid_flag) {
            // Write the gaussian to the output buffer
            d_gaussians_out.set_index(cidx * num_gaussians + pidx);
            d_gaussians_out.export();
        }
    }
}

} // namespace cugsplat