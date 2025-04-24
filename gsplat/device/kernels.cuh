#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat::device {

/**
 * @brief PreprocessKernel performs preprocessing of input primitives from a specific camera view.
 * 
 * This kernel is designed to be launched with a 2D thread grid over [camera x primitive],
 * and supports two modes:
 *   - PACKED = false: Write output directly to a known index
 *   - PACKED = true: Write output in a compacted format, requiring a two-pass reduction/scan
 *
 * This kernel will parallelize over [num_cameras, num_primitives], and is expected to be 
 * launched with a 2D grid of threads, with each thread processing a single {camera, primitive} pair.
 * I.E.:
 *      dim3 blockDim(THREADS_FOR_CAMERA, THREADS_FOR_PRIMITIVE);
 *      dim3 gridDim(
 *          (num_cameras + THREADS_FOR_CAMERA - 1) / THREADS_FOR_CAMERA, 
 *          (num_primitives + THREADS_FOR_PRIMITIVE - 1) / THREADS_FOR_PRIMITIVE
 *      );
 * 
 * 
 * @tparam DeviceCameraModel
 *         A lightweight device-side structure that represents a camera model.
 *         Must implement:
 *           __device__ void set_index(int);
 *           __device__ int get_n() const;
 *
 *         // Example:
 *         struct DummyCameraModel {
 *             __device__ void set_index(int) {}
 *             __device__ int get_n() const { return 1; }
 *         };
 * 
 * @tparam DevicePrimitiveIn
 *         A device-side structure for accessing input primitive data.
 *         Must implement:
 *           __device__ void set_index(int);
 *           __device__ int get_n() const;
 *
 *         // Example:
 *         struct DummyPrimitiveIn {
 *             __device__ void set_index(int) {}
 *             __device__ int get_n() const { return 1; }
 *         };
 *
 * @tparam DevicePrimitiveOut
 *         A device-side structure for storing and exporting output primitives.
 *         Must implement:
 *           __device__ bool preprocess(DeviceCameraModel&, DevicePrimitiveIn&);
 *           __device__ void write_to_buffer();
 *
 *         // Example:
 *         struct DummyPrimitiveOut {
 *             template <typename Cam, typename In, typename Param>
 *             __device__ bool preprocess(Cam&, In&) { return false; }
 *             __device__ void write_to_buffer(uint32_t) {}
 *         };
 *
 * @tparam PACKED
 *         If true, enables two-pass "packed" processing:
 *           - Pass 1: Count valid primitives per block (output to `block_cnts`)
 *           - Pass 2: Write compacted output using `block_offsets`
 *         If false, output is written to (camera_index * num_primitives + primitive_index)
 *
 * @tparam THREADS_PER_BLOCK
 *         Total number of threads per CUDA block. Required when PACKED = true. Must
 *         match the number of threads in the kernel launch.
 */
template <
class DeviceCameraModel, 
class DevicePrimitiveIn, 
class DevicePrimitiveOut, 
bool PACKED, 
int THREADS_PER_BLOCK>
__global__ void PreprocessKernel(
    DeviceCameraModel d_camera,
    DevicePrimitiveIn d_primitives_in,
    // outputs
    DevicePrimitiveOut d_primitives_out,
    int32_t* block_cnts,
    int32_t* block_offsets
) {
    auto const cidx = blockIdx.x * blockDim.x + threadIdx.x; // camera index
    auto const pidx = blockIdx.y * blockDim.y + threadIdx.y; // primitive index
    auto const num_cameras = d_camera.get_n();
    auto const num_primitives = d_primitives_in.get_n();
    if (cidx >= num_cameras || pidx >= num_primitives) {
        return;
    }

    // Shift pointers
    d_camera.set_index(cidx);
    d_primitives_in.set_index(pidx);

    // Preprocess the primitive. Results are saved in `d_primitives_out` locally.
    auto const valid_flag = d_primitives_out.preprocess(d_camera, d_primitives_in);

    if constexpr (PACKED) {
        auto const block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        uint32_t thread_data = static_cast<uint32_t>(valid_flag);
        if (block_cnts != nullptr) {
            // First pass: compute the block-wide sum. I.E How many primitives will be output
            // by this block of threads. 
            uint32_t aggregate = 0;
            if (__syncthreads_or(thread_data)) {
                typedef cub::BlockReduce<uint32_t, THREADS_PER_BLOCK> BlockReduce;
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
                // Write the primitive to the output buffer:
                // `thread_data` is the index of where to write the primitive
                d_primitives_out.write_to_buffer(thread_data);
            }
        }
    } else {
        if (valid_flag) {
            // Write the primitive to the output buffer
            d_primitives_out.write_to_buffer(cidx * num_primitives + pidx);
        }
    }
}

} // namespace gsplat::device