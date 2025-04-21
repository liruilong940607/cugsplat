
// template <typename T>
// struct always_false : std::false_type {};

// template <, 
// class DeviceGaussianOut,
// class DeviceCameraModel, 
// class DeviceGaussianIn>
// inline __host__ __device__ auto preprocess_impl_dispatch(
//     const DeviceCameraModel d_camera,
//     const DeviceGaussianIn d_gaussians_in,
//     const PreprocessParameters& params
// ) {
//     if constexpr (std::is_same<DeviceGaussianOut, DeviceGaussian3DIn3D>::value) {
//         return preprocess_impl_3dgs();
//     } else if constexpr (std::is_same<DeviceGaussianOut, DeviceGaussian2DIn2D>::value) {
//         return preprocess_impl_2dgs();
//     } else {
//         static_assert(always_false<DeviceGaussianOut>::value, "Unsupported DeviceGaussianOut type.");
//     }
// }



// template <typename TypeMu, typename TypeQuat, typename TypeScale, typename TypeCovar>
// struct Gaussian {
//     float opacity;
//     TypeMu mu;
//     TypeQuat quat;
//     TypeScale scale;
//     TypeCovar covar;
// };

// // template <>
// // __forceinline__ __device__
// // auto Gaussian<fvec2, fvec3, fvec2, mat2>::compute_aabb() const -> std::pair<fvec2, fvec2> {
// //     // Compute opacity-aware bounding box.
// //     // https://arxiv.org/pdf/2402.00525 Section B.2
// //     float extend = 3.33f;
// //     extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));

// //     // compute tight rectangular bounding box (non differentiable)
// //     // https://arxiv.org/pdf/2402.00525
// //     float b = 0.5f * (covar[0][0] + covar[1][1]);
// //     float tmp = sqrtf(max(0.01f, b * b - det));
// //     float v1 = b + tmp; // larger eigenvalue
// //     float r1 = extend * sqrtf(v1);
// //     radius_x = ceilf(min(extend * sqrtf(covar[0][0]), r1));
// //     radius_y = ceilf(min(extend * sqrtf(covar[1][1]), r1));
// // }

// using Gaussian3DIn3D = Gaussian<fvec3, fvec4, fvec3, mat3>;
// using Gaussian2DIn3D = Gaussian<fvec2, fvec4, fvec3, mat3>;
// using Gaussian2DIn2D = Gaussian<fvec2, fvec3, fvec2, mat2>;


// template <typename TypeMu, typename TypeQuat, typename TypeScale, typename TypeCovar>
// struct DeviceGaussian {
//     uint32_t num_elements;
//     uint32_t index;

//     void set_index(uint32_t index) {index = index;}
//     void set_n(uint32_t n) {num_elements = n;}
//     uint32_t get_index() const {return index;}
//     uint32_t get_n() const {return num_elements;}

//     float* __restrict__ opacity;
//     TypeMu* __restrict__ mu;
//     TypeQuat* __restrict__ quat;
//     TypeScale* __restrict__ scale;
//     TypeCovar* __restrict__ covar;

//     // TOOD: what if they are nullptr?
//     void set_opacity(float opacity) {this->opacity[index] = opacity;}
//     void set_mu(TypeMu mu) {this->mu[index] = mu;}
//     void set_quat(TypeQuat quat) {this->quat[index] = quat;}
//     void set_scale(TypeScale scale) {this->scale[index] = scale;}
//     void set_covar(TypeCovar covar) {this->covar[index] = covar;}
//     float get_opacity() const {return opacity[index];}
//     TypeMu get_mu() const {return mu[index];}
//     TypeQuat get_quat() const {return quat[index];}
//     TypeScale get_scale() const {return scale[index];}
//     TypeCovar get_covar() const {return covar[index];}
// };

// using DeviceGaussian3DIn3D = DeviceGaussian<fvec3, fvec4, fvec3, mat3>;
// using DeviceGaussian2DIn3D = DeviceGaussian<fvec3, fvec4, fvec2, mat3>;

// struct DeviceGaussian2DIn2D : DeviceGaussian<fvec2, fvec3, fvec2, mat2> {
//     __forceinline__ __device__ void set_data(const Gaussian2DIn2D& g2d) {
//         opacity[index] = g2d.opacity;
//         mu[index] = g2d.get_mu();
//         quat[index] = g3d.get_quat();
//         scale[index] = g2d.get_scale();
//         covar[index] = g2d.get_covar();
//     }

    
// };


// struct PreprocessParameters {
//     uint32_t render_width;
//     uint32_t render_height;
//     float near_plane;
//     float far_plane;
//     float margin_factor;
//     float filter_size;
// };


// // struct PreprocessResult {
// //     fvec2 image_point;
// //     float depth;
// //     DeviceGaussian2DIn2D gaussian_projected;
// //     fvec2 center;
// //     fvec2 radius;
// // };

// // template <class DeviceCameraModel, class DeviceGaussianIn>
// // __forceinline__ __device__ auto preprocess_impl(
// //     const DeviceCameraModel d_camera,
// //     const DeviceGaussianIn d_gaussians_in,
// //     const PreprocessParameters& params
// // ) -> std::pair<PreprocessResult, bool> {
// //     PreprocessResult result;

// //     // Check: If the gaussian is outside the camera frustum, skip it
// //     auto const &[image_point, depth] = d_camera.point_world_to_image(d_gaussians_in.get_mu());
// //     if (depth < params.near_plane || depth > params.far_plane) {
// //         return {result, false};
// //     }

// //     // Check: If the gaussian is outside the image plane, skip it
// //     auto const min_x = - params.margin_factor * params.render_width;
// //     auto const min_y = - params.margin_factor * params.render_height;
// //     auto const max_x = (1 + params.margin_factor) * params.render_width;
// //     auto const max_y = (1 + params.margin_factor) * params.render_height;
// //     if (image_point.x < min_x || image_point.x > max_x ||
// //         image_point.y < min_y || image_point.y > max_y) {
// //         return {result, false};
// //     }

// //     // Compute the projected gaussian on the image plane
// //     auto const &[gaussian_projected, projected_valid_flag] = d_camera.gaussian_world_to_image(d_gaussians_in);
// //     if (!projected_valid_flag) {
// //         return {result, false};
// //     }

// //     // Compute the bounding box of this gaussian on the image plane
// //     auto const &[center, radius] = gaussian_projected.compute_aabb();

// //     // Check again if the gaussian is outside the image plane
// //     if (center.x - radius.x < 0 || center.x + radius.x > params.render_width ||
// //         center.y - radius.y < 0 || center.y + radius.y > params.render_height) {
// //         return {result, false};
// //     }

// //     result.image_point = image_point;
// //     result.depth = depth;
// //     result.gaussian_projected = gaussian_projected;
// //     result.center = center;
// //     result.radius = radius;
// //     return {result, true};
// // }


// template <
// class DeviceCameraModel, 
// class DeviceGaussianIn, 
// class DeviceGaussianOut, 
// bool PACKED, 
// // Total number of threads in a block, only used in PACKED mode
// int NUM_THREADS>
// __global__ void PreprocessKernel(
//     const DeviceCameraModel d_camera,
//     const DeviceGaussianIn d_gaussians_in,
//     const PreprocessParameters& params,
//     // outputs
//     DeviceGaussianOut d_gaussians_out,
//     int32_t* __restrict__ block_cnts,
//     int32_t* __restrict__ block_offsets
// ) {
//     // Parallelize over [num_cameras, num_gaussians]. should be launched
//     // as a 2D grid of threads, with each thread processing a single
//     // camera and a single gaussian.

//     // Kernel should be launched with dim3(blockDim.x, blockDim.y) and 
//     // dim3(gridDim.x, gridDim.y) such that threadIdx.z == 0 and blockIdx.z == 0
//     auto const cidx = blockIdx.x * blockDim.x + threadIdx.x; // camera index
//     auto const pidx = blockIdx.y * blockDim.y + threadIdx.y; // gaussian index
//     if (cidx >= d_camera.get_n() || pidx >= d_gaussians_in.get_n()) {
//         return;
//     }

//     // Shift pointers
//     d_camera.set_index(cidx);
//     d_gaussians_in.set_index(pidx);

//     // Preprocess the gaussian
//     auto const &[
//         preprocess_result, valid_flag
//     ] = preprocess_impl<DeviceGaussianOut>(d_camera, d_gaussians_in, params);

//     if constexpr (PACKED) {
//         auto const block_idx = blockIdx.y * gridDim.x + blockIdx.x;
//         int32_t thread_data = static_cast<int32_t>(valid_flag);
//         if (block_cnts != nullptr) {
//             // First pass: compute the block-wide sum. I.E How many gaussians will be output
//             // by this block of threads. 
//             int32_t aggregate = 0;
//             if (__syncthreads_or(thread_data)) {
//                 typedef cub::BlockReduce<int32_t, NUM_THREADS> BlockReduce;
//                 __shared__ typename BlockReduce::TempStorage temp_storage;
//                 aggregate = BlockReduce(temp_storage).Sum(thread_data);
//             }
//             if (threadIdx.x == 0 && threadIdx.y == 0) {
//                 block_cnts[block_idx] = aggregate;
//             }
//             return;
//         } else {
//             // Second pass: write the gaussian to the output buffer
//             if (__syncthreads_or(thread_data)) {
//                 typedef cub::BlockScan<int32_t, NUM_THREADS> BlockScan;
//                 __shared__ typename BlockScan::TempStorage temp_storage;
//                 BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
//             }
//             thread_data += block_offsets[block_idx];
//             if (valid_flag && d_gaussians_out != nullptr) {
//                 d_gaussians_out.set_index(thread_data);
//                 d_gaussians_out.set_data(preprocess_result);
//             }
//         }
//     } else {
//         if (valid_flag && d_gaussians_out != nullptr) {
//             d_gaussians_out.set_index(cidx * num_gaussians + pidx);
//             d_gaussians_out.set_data(preprocess_result);
//         }
//     }
// }

// template <typename TypeMu, typename TypeQuat, typename TypeScale, typename TypeCovar>
// struct Gaussian {
//     float opacity;
//     TypeMu mu;
//     TypeQuat quat;
//     TypeScale scale;
//     TypeCovar covar;
// };

// using Gaussian3DIn3D = Gaussian<fvec3, fvec4, fvec3, mat3>;
// using Gaussian2DIn3D = Gaussian<fvec2, fvec4, fvec3, mat3>;
// using Gaussian2DIn2D = Gaussian<fvec2, fvec3, fvec2, mat2>;

// template <typename TypeMu, typename TypeQuat, typename TypeScale, typename TypeCovar>
// struct DeviceGaussian {
//     uint32_t num_elements;
//     uint32_t index;

//     void set_index(uint32_t index) {index = index;}
//     void set_n(uint32_t n) {num_elements = n;}
//     uint32_t get_index() const {return index;}
//     uint32_t get_n() const {return num_elements;}

//     float* __restrict__ opacity;
//     TypeMu* __restrict__ mu;
//     TypeQuat* __restrict__ quat;
//     TypeScale* __restrict__ scale;
//     TypeCovar* __restrict__ covar;
// };


// struct DeviceGaussian3DIn3D : public DeviceGaussian<fvec3, fvec4, fvec3, mat3> {
//     __forceinline__ __device__ void set_data(const Gaussian3DIn3D& g3d, const Gaussian2DIn2D& g2d) {
//         filter_size = g2d.get_filter();
//         if (filter_size == 0) {
//             // just set g3d to it.
//         }
//         opacity[index] = g3d.opacity;
// };