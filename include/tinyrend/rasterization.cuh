#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace tinyrend {

/*
    Rasterize primitives to image buffers (tile-based)

    dim3 threads = {tile_w, tile_h, 1};
    dim3 grid = {n_tiles_w, n_tiles_h, n_images};

    Each block of threads rasterizes one tile with size (tile_w, tile_h):
    - First all threads will collabratively load all primitives that intersect with this
   tile into shared memory,
    - Then each thread will rasterize its assigned pixel.

    This function expects the primitive-tile intersection has been computed already.
    Say we have in total `n_tiles` tiles to be rasterized, in total `nnz` primtives.
    and in total `n_isects` intersections are found between primitives and tiles.

    The intersection data should be stored in the following format:
    - isect_primitive_ids: [n_isects] Stores the primitive id for each intersection
    - isect_prefix_sum_per_tile: [n_tiles] Stores the prefix-sum of the number of
   intersections per tile

    In other words, isect_primitive_ids[start:end) stores all the intersections for the
   i-th tile. where `start=isect_prefix_sum_per_tile[i - 1]` and
   `end=isect_prefix_sum_per_tile[i]`.

    The outputs are:
    - buffer_alpha: [n_images, image_h, image_w, 1] Stores the alpha value for each tile
*/
__global__ void rasterization(
    const uint32_t n_primitives,
    const uint32_t n_tiles,
    const uint32_t n_isects,
    const uint32_t n_images,
    const uint32_t image_h,
    const uint32_t image_w,
    // intersection data
    const uint32_t *isect_primitive_ids,       // [n_isects]
    const uint32_t *isect_prefix_sum_per_tile, // [n_tiles]
    // primitive data
    // const float* opacities, // [nnz]
    // outputs
    float *buffer_alpha // [n_images, image_h, image_w, 1]
) {
    auto const tile_w = blockDim.x;
    auto const tile_h = blockDim.y;
    auto const tile_x = blockIdx.x;
    auto const tile_y = blockIdx.y;
    auto const pixel_x = tile_x * tile_w + threadIdx.x;
    auto const pixel_y = tile_y * tile_h + threadIdx.y;
    auto const pixel_id = pixel_y * image_w + pixel_x;
    auto const image_id = blockIdx.z;
    auto const tile_id = tile_y * tile_w + tile_x;
    auto const threads_per_block = blockDim.x * blockDim.y;
    auto const thread_rank = threadIdx.x + threadIdx.y * blockDim.x;

    // Check if the pixel is inside the image. If not, we still keep this thread
    // alive to help with loading primitives into shared memory.
    auto const inside = pixel_x < image_w && pixel_y < image_h;
    auto done = !inside;

    // First, figure out which primitives intersect with the current tile.
    auto const isect_start = tile_id == 0 ? 0 : isect_prefix_sum_per_tile[tile_id - 1];
    auto const isect_end = isect_prefix_sum_per_tile[tile_id];

    // Since each thread is responsible for loading one primitive into shared memory,
    // we can load at most `threads_per_block` primitives at a time as a batch. Then
    // how many batches do we need to load all the primitives?
    auto const n_batches =
        (isect_end - isect_start + threads_per_block - 1) / threads_per_block;

    // Load the primitives into shared memory.
    extern __shared__ char shmem[];
    // uint32_t *shmem_ids = reinterpret_cast<uint32_t*>(shmem);
    // float *shmem_opacities = reinterpret_cast<float*>(shmem_ids + threads_per_block);

    // Init values
    float T = 1.0f;                 // current transmittance
    uint32_t primitive_cur_idx = 0; // current primitive index

    for (uint32_t b = 0; b < n_batches; ++b) {
        // resync all threads before beginning next batch and early stop if entire tile
        // is done
        if (__syncthreads_count(done) >= threads_per_block) {
            break;
        }

        // Load the next batch of primitives into shared memory.
        auto const isect_start_cur_batch = isect_start + b * threads_per_block;
        auto const idx = isect_start_cur_batch + thread_rank;
        if (idx < isect_end) {
            auto const primitive_id = isect_primitive_ids[idx];
            // shmem_ids[thread_rank] = primitive_id;
            // shmem_opacities[thread_rank] = opacities[primitive_id];
        }

        // wait for other threads to collect the primitives in batch
        __syncthreads();

        // Now, the job of this thread is to rasterize all the primitives in the shared
        // memory to the current pixel.
        uint32_t cur_batch_size =
            min(threads_per_block, isect_end - isect_start_cur_batch);
        for (uint32_t t = 0; (t < cur_batch_size) && (!done); ++t) {
            // auto const sigma = ...; // TODO: get sigma from the primitive
            // auto const opacity = shmem_opacities[t];
            // auto const alpha = min(0.999f, opacity * __expf(-sigma));
            auto const alpha = 1.0f;
            if (alpha < 1.0f / 255.0f) {
                continue;
            }

            auto const next_T = T * (1.0f - alpha);
            if (next_T < 1e-4f) { // this pixel is done: exclusive
                done = true;
                break;
            }

            // auto const weight = alpha * T;
            primitive_cur_idx = isect_start_cur_batch + t;
            T = next_T;
        }
    }

    if (inside) {
        auto const offset_pixel = image_id * image_h * image_w + pixel_id;
        buffer_alpha[offset_pixel] = 1.0f - T;
    }
}

} // namespace tinyrend
