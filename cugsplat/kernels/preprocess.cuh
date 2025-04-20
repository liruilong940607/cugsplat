#include <glm/glm.hpp>
#include <cub/cub.cuh>

using namespace glm;

template <class PrimitiveProjected>
struct PreprocessResult {
    fvec2 image_point;
    float depth;
    PrimitiveProjected primitive_projected;
    fvec2 center;
    fvec2 radius;
};

struct PreprocessParameters {
    uint32_t render_width;
    uint32_t render_height;
    float near_plane;
    float far_plane;
    float margin_factor;
    float filter_size;
};

template <class CameraModel, class PrimitiveIn, class PrimitiveProjected>
__forceinline__ __device__ auto preprocess_impl(
    const CameraModel d_camera,
    const PrimitiveIn d_primitives_in,
    const PreprocessParameters& params
) -> std::pair<PreprocessResult<PrimitiveProjected>, bool> {
    PreprocessResult<PrimitiveProjected> result;

    // Check: If the primitive is outside the camera frustum, skip it
    auto const &[image_point, depth] = d_camera.point_world_to_image(d_primitives_in.get_position());
    if (depth < params.near_plane || depth > params.far_plane) {
        return {result, false};
    }

    // Check: If the primitive is outside the image plane, skip it
    auto const min_x = - params.margin_factor * params.render_width;
    auto const min_y = - params.margin_factor * params.render_height;
    auto const max_x = (1 + params.margin_factor) * params.render_width;
    auto const max_y = (1 + params.margin_factor) * params.render_height;
    if (image_point.x < min_x || image_point.x > max_x ||
        image_point.y < min_y || image_point.y > max_y) {
        return {result, false};
    }

    // Compute the projected primitive on the image plane
    auto const &[primitive_projected, projected_valid_flag] = d_camera.primitive_world_to_image(d_primitives_in);
    if (!projected_valid_flag) {
        return {result, false};
    }
    primitive_projected.set_filter_size(params.filter_size);

    // Compute the bounding box of this primitive on the image plane
    auto const &[center, radius] = primitive_projected.compute_aabb();

    // Check again if the primitive is outside the image plane
    if (center.x - radius.x < 0 || center.x + radius.x > params.render_width ||
        center.y - radius.y < 0 || center.y + radius.y > params.render_height) {
        return {result, false};
    }

    result.image_point = image_point;
    result.depth = depth;
    result.primitive_projected = primitive_projected;
    result.center = center;
    result.radius = radius;
    return {result, true};
}


template <
class CameraModel, 
class PrimitiveIn, 
class PrimitiveOut, 
class PrimitiveProjected, 
bool PACKED, 
// Total number of threads in a block, only used in PACKED mode
int NUM_THREADS>
__global__ void PreprocessKernel(
    const int num_cameras,
    const CameraModel d_camera,
    const int num_primitives,
    const PrimitiveIn d_primitives_in,
    const PreprocessParameters& params,
    // outputs
    PrimitiveOut d_primitives_out,
    int32_t* __restrict__ block_cnts,
    int32_t* __restrict__ block_offsets
) {
    // Parallelize over [num_cameras, num_primitives]. should be launched
    // as a 2D grid of threads, with each thread processing a single
    // camera and a single primitive.

    // Kernel should be launched with dim3(blockDim.x, blockDim.y) and dim3(gridDim.x, gridDim.y)
    // such that threadIdx.z == 0 and blockIdx.z == 0
    auto const cidx = blockIdx.x * blockDim.x + threadIdx.x; // camera index
    auto const pidx = blockIdx.y * blockDim.y + threadIdx.y; // primitive index
    if (cidx >= num_cameras || pidx >= num_primitives) {
        return;
    }

    // Shift pointers
    d_camera.set_index(cidx);
    d_primitives_in.set_index(pidx);

    // Preprocess the primitive
    auto const &[preprocess_result, valid_flag] = preprocess_impl(d_camera, d_primitives_in, params);

    if constexpr (PACKED) {
        auto const block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        int32_t thread_data = static_cast<int32_t>(valid_flag);
        if (block_cnts != nullptr) {
            // First pass: compute the block-wide sum. I.E How many primitives will be output
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
            // Second pass: write the primitive to the output buffer
            if (__syncthreads_or(thread_data)) {
                typedef cub::BlockScan<int32_t, NUM_THREADS> BlockScan;
                __shared__ typename BlockScan::TempStorage temp_storage;
                BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
            }
            thread_data += block_offsets[block_idx];
            if (valid_flag) {
                d_primitives_out.set_index(thread_data);
                d_primitives_out.set_data(d_primitives_in, preprocess_result);
            }
        }
    } else {
        if (valid_flag) {
            d_primitives_out.set_index(cidx * num_primitives + pidx);
            d_primitives_out.set_data(d_primitives_in, preprocess_result);
        }
    }
}
