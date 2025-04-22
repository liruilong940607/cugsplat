// #include <glm/glm.hpp>
// #include <cub/cub.cuh>

// #include "util/macros.cuh"

// namespace cugsplat {

// using namespace glm;

// struct Gaussian3DIn3D {
//     float opacity;
//     glm::fvec3 mu;
//     glm::fvec4 quat;
//     glm::fvec3 scale;
//     fmat3 covar;
// }

// struct DeviceGaussian3DIn3D {
//     uint32_t n;
//     uint32_t index;
//     float* opacity;
//     glm::fvec3* mu;
//     glm::fvec4* quat;
//     glm::fvec3* scale;
//     fmat3* covar;

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
//     glm::fvec3 mu;
//     glm::fvec4 quat;
//     glm::fvec2 scale;
//     fmat3 covar;
// };

// struct DeviceGaussian2DIn3D {
//     uint32_t n;
//     uint32_t index;
//     float* opacity;
//     glm::fvec3* mu;
//     glm::fvec4* quat;
//     glm::fvec2* scale;
//     fmat3* covar;

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
//     glm::fvec2 mu;
//     glm::fvec2 scale;
//     fmat2 covar;
// };

// struct DeviceGaussian2DIn2D {
//     uint32_t n;
//     uint32_t index;
//     float* opacity;
//     glm::fvec2* mu;
//     glm::fvec2* scale;
//     fmat2* covar;

//     DEFINE_VALUE_SETGET(uint32_t, n)
//     DEFINE_VALUE_SETGET(uint32_t, index)
//     DEFINE_POINTER_SETGET(float, opacity, index)
//     DEFINE_POINTER_SETGET(fvec2, mu, index)
//     DEFINE_POINTER_SETGET(fvec2, quat, index)
//     DEFINE_POINTER_SETGET(fmat2, covar, index)
// };

// } // namespace cugsplat