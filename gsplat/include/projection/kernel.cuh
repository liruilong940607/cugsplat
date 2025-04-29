#pragma once

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat::device {

namespace cg = cooperative_groups;

/**
 * @brief PreprocessFwdKernel performs preprocessing of input primitives from a
 * specific camera view.
 *
 * This kernel is designed to be launched with a 2D thread grid over [camera x
 * primitive], and supports two modes:
 *   - PACKED = false: Write output directly to a known index
 *   - PACKED = true: Write output in a compacted format, requiring a two-pass
 * reduction/scan
 *
 * This kernel will parallelize over [num_cameras, num_primitives], and is
 * expected to be launched with a 2D grid of threads, with each thread
 * processing a single {camera, primitive} pair. I.E.: dim3
 * blockDim(THREADS_FOR_CAMERA, THREADS_FOR_PRIMITIVE); dim3 gridDim(
 *          (num_cameras + THREADS_FOR_CAMERA - 1) / THREADS_FOR_CAMERA,
 *          (num_primitives + THREADS_FOR_PRIMITIVE - 1) / THREADS_FOR_PRIMITIVE
 *      );
 *
 * @tparam DeviceCameraModel
 *         A lightweight device-side structure that represents a camera model.
 *         Must implement:
 *           __device__ void set_index(int);
 *           __device__ int get_n() const;
 *
 * @tparam DevicePrimitiveIn
 *         A device-side structure for accessing input primitive data.
 *         Must implement:
 *           __device__ void set_index(int);
 *           __device__ int get_n() const;
 *
 * @tparam DevicePrimitiveOut
 *         A device-side structure for storing output primitive data.
 *         Must implement:
 *           __device__ void set_index(int);
 *           __device__ void set_value(const auto&);
 *
 * @tparam Operator
 *         A device-side structure for preprocessing primitives.
 *         Must implement:
 *           __device__ auto forward(DeviceCameraModel&, DevicePrimitiveIn&) ->
 * std::tuple<auto, bool>;
 *
 * @tparam PACKED
 *         If true, enables two-pass "packed" processing:
 *           - Pass 1: Count valid primitives per block (output to `block_cnts`)
 *           - Pass 2: Write compacted output using `block_offsets`
 *         If false, output is written to (camera_index * num_primitives +
 * primitive_index)
 *
 * @tparam THREADS_PER_BLOCK
 *         Total number of threads per CUDA block. Required when PACKED = true.
 *         Must match the number of threads in the kernel launch.
 *
 * @param d_camera Device-side camera model instance
 * @param d_primitives_in Device-side input primitive data
 * @param d_primitives_out Device-side output primitive data
 * @param op Preprocessing operator instance
 * @param block_cnts [PACKED mode only] Buffer for storing block-wise primitive
 * counts
 * @param block_offsets [PACKED mode only] Buffer for storing block-wise output
 * offsets
 * @param camera_ids [PACKED mode only] Buffer for storing camera indices of
 * output primitives
 * @param primitive_ids [PACKED mode only] Buffer for storing primitive indices
 * of output primitives
 */
template <
    class DeviceCameraModel,
    class DevicePrimitiveIn,
    class DevicePrimitiveOut,
    class Operator,
    bool PACKED,
    int THREADS_PER_BLOCK>
__global__ void PreprocessFwdKernel(
    DeviceCameraModel d_camera,
    DevicePrimitiveIn d_primitives_in,
    DevicePrimitiveOut d_primitives_out,
    Operator op,
    // helpers for packed mode
    int32_t *block_cnts,
    int32_t *block_offsets,
    int32_t *camera_ids,
    int32_t *primitive_ids
) {
    auto const cidx = blockIdx.x * blockDim.x + threadIdx.x; // camera index
    auto const pidx = blockIdx.y * blockDim.y + threadIdx.y; // primitive index
    auto const num_cameras = d_camera.get_n();
    auto const num_primitives_in = d_primitives_in.get_n();
    if (cidx >= num_cameras || pidx >= num_primitives_in) {
        return;
    }

    // Shift pointers
    d_camera.set_index(cidx);
    d_primitives_in.set_index(pidx);

    // Preprocess the primitive
    auto const &[primitive_out, valid_flag] =
        op.forward(d_camera, d_primitives_in);

    if constexpr (PACKED) {
        auto const block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        uint32_t thread_data = static_cast<uint32_t>(valid_flag);
        if (block_cnts != nullptr) {
            // First pass: compute the block-wide sum. I.E How many primitives
            // will be output by this block of threads.
            uint32_t aggregate = 0;
            if (__syncthreads_or(thread_data)) {
                typedef cub::BlockReduce<uint32_t, THREADS_PER_BLOCK>
                    BlockReduce;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                aggregate = BlockReduce(temp_storage).Sum(thread_data);
            }
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                block_cnts[block_idx] = aggregate;
            }
            return;
        } else {
            // Second pass: write the primitive to the output buffer
            if (__syncthreads_or(thread_data)) {
                typedef cub::BlockScan<uint32_t, THREADS_PER_BLOCK> BlockScan;
                __shared__ typename BlockScan::TempStorage temp_storage;
                BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
            }
            thread_data += block_offsets[block_idx];
            if (valid_flag) {
                // Write the primitive to the output buffer
                auto const oidx = thread_data;
                d_primitives_out.set_index(oidx);
                d_primitives_out.set_value(primitive_out);
                camera_ids[thread_data] = cidx;
                primitive_ids[thread_data] = pidx;
            }
        }
    } else {
        if (valid_flag) {
            // Write the primitive to the output buffer
            auto const oidx = cidx * num_primitives_in + pidx;
            d_primitives_out.set_index(oidx);
            d_primitives_out.set_value(primitive_out);
        }
    }
}

template <
    class DeviceCameraModel,
    class DeviceCameraModelGrad,
    class DevicePrimitiveIn,
    class DevicePrimitiveInGrad,
    class DevicePrimitiveOut,
    class DevicePrimitiveOutGrad,
    class Operator,
    bool PACKED>
__global__ void PreprocessBwdKernel(
    DeviceCameraModel d_camera,
    DeviceCameraModelGrad d_camera_grad,
    DevicePrimitiveIn d_primitives_in,
    DevicePrimitiveInGrad d_primitives_in_grad,
    DevicePrimitiveOut d_primitives_out,
    DevicePrimitiveOutGrad d_primitives_out_grad,
    const int32_t *camera_ids,
    const int32_t *primitive_ids,
    // outputs
    Operator op
) {
    int32_t cidx, pidx, oidx;
    if constexpr (PACKED) {
        // Launch with a 1D grid of threads
        auto const oidx = blockIdx.y * blockDim.x + threadIdx.x;
        auto const num_outputs = d_primitives_out.get_n();
        if (oidx >= num_outputs) {
            return;
        }
        cidx = camera_ids[oidx];
        pidx = primitive_ids[oidx];
    } else {
        // Launch with a 2D grid of threads
        cidx = blockIdx.x * blockDim.x + threadIdx.x; // camera index
        pidx = blockIdx.y * blockDim.y + threadIdx.y; // primitive index
        auto const num_cameras = d_camera.get_n();
        auto const num_primitives_in = d_primitives_in.get_n();
        if (cidx >= num_cameras || pidx >= num_primitives_in) {
            return;
        }
        oidx = cidx * num_primitives_in + pidx;
    }

    // Shift pointers
    d_camera.set_index(cidx);
    d_primitives_in.set_index(pidx);
    d_primitives_out.set_index(oidx);
    d_primitives_out_grad.set_index(oidx);

    // Preprocess the primitive.
    auto const &[primitive_in_grad, camera_grad, valid_flag] = op.backward(
        d_camera, d_primitives_in, d_primitives_out, d_primitives_out_grad
    );

    if (valid_flag) {
        // Write out results with warp-level reduction
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        if (d_primitives_in.requires_grad) {
            auto warp_group_g = cg::labeled_partition(warp, pidx);
            primitive_in_grad.warp_sum(warp);
            d_primitives_in_grad.set_index(pidx);
            d_primitives_in_grad.atomic_add(primitive_in_grad);
        }
        if (d_camera.requires_grad) {
            auto warp_group_c = cg::labeled_partition(warp, cidx);
            camera_grad.warp_sum(warp);
            d_camera_grad.set_index(cidx);
            d_camera_grad.atomic_add(camera_grad);
        }
    }
}

} // namespace gsplat::device