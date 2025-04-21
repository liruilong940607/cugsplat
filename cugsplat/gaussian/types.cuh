// #include <glm/glm.hpp>
// #include <cub/cub.cuh>

// #include "util/macros.cuh"

// namespace cugsplat {

// using namespace glm;

// struct Gaussian3DIn3D {
//     float opacity;
//     fvec3 mu;
//     fvec4 quat;
//     fvec3 scale;
//     fmat3 covar;
// }

// struct DeviceGaussian3DIn3D {
//     uint32_t n;
//     uint32_t index;
//     float* __restrict__ opacity;
//     fvec3* __restrict__ mu;
//     fvec4* __restrict__ quat;
//     fvec3* __restrict__ scale;
//     fmat3* __restrict__ covar;

//     DEFINE_VALUE_SETGET(uint32_t, n)
//     DEFINE_VALUE_SETGET(uint32_t, index)
//     DEFINE_POINTER_SETGET(float, opacity, index)
//     DEFINE_POINTER_SETGET(fvec3, mu, index)
//     DEFINE_POINTER_SETGET(fvec4, quat, index)
//     DEFINE_POINTER_SETGET(fvec3, scale, index)
//     DEFINE_POINTER_SETGET(fmat3, covar, index)
// };

// struct Gaussian2DIn3D {
//     float opacity;
//     fvec3 mu;
//     fvec4 quat;
//     fvec2 scale;
//     fmat3 covar;
// };

// struct DeviceGaussian2DIn3D {
//     uint32_t n;
//     uint32_t index;
//     float* __restrict__ opacity;
//     fvec3* __restrict__ mu;
//     fvec4* __restrict__ quat;
//     fvec2* __restrict__ scale;
//     fmat3* __restrict__ covar;

//     DEFINE_VALUE_SETGET(uint32_t, n)
//     DEFINE_VALUE_SETGET(uint32_t, index)
//     DEFINE_POINTER_SETGET(float, opacity, index)
//     DEFINE_POINTER_SETGET(fvec3, mu, index)
//     DEFINE_POINTER_SETGET(fvec4, quat, index)
//     DEFINE_POINTER_SETGET(fvec2, scale, index)
//     DEFINE_POINTER_SETGET(fmat3, covar, index)
// };

// struct Gaussian2DIn2D {
//     float opacity;
//     fvec2 mu;
//     fvec2 scale;
//     fmat2 covar;
// };

// struct DeviceGaussian2DIn2D {
//     uint32_t n;
//     uint32_t index;
//     float* __restrict__ opacity;
//     fvec2* __restrict__ mu;
//     fvec2* __restrict__ scale;
//     fmat2* __restrict__ covar;

//     DEFINE_VALUE_SETGET(uint32_t, n)
//     DEFINE_VALUE_SETGET(uint32_t, index)
//     DEFINE_POINTER_SETGET(float, opacity, index)
//     DEFINE_POINTER_SETGET(fvec2, mu, index)
//     DEFINE_POINTER_SETGET(fvec2, quat, index)
//     DEFINE_POINTER_SETGET(fmat2, covar, index)
// };

// } // namespace cugsplat