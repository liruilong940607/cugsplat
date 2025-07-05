#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include "tinyrend/common/macros.h" // for TREND_HOST_DEVICE
#include "tinyrend/common/mat.h"
#include "tinyrend/common/math.h"
#include "tinyrend/common/vec.h"

namespace tinyrend::camera::impl::orthogonal {

/// \brief Project a 3D point in camera space to 2D image space using
/// orthogonal projection.
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \return Projected 2D point in image space
TREND_HOST_DEVICE inline auto project(
    fvec3 const &camera_point, fvec2 const &focal_length, fvec2 const &principal_point
) -> fvec2 {
    auto const xy = fvec2(camera_point[0], camera_point[1]);
    auto const image_point = focal_length * xy + principal_point;
    return image_point;
}

/// \brief Compute the Jacobian of the projection function.
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \return Jacobian of the projection function
TREND_HOST_DEVICE inline auto
project_jac(fvec3 const &camera_point, fvec2 const &focal_length) -> fmat3x2 {
    auto const J = fmat3x2{focal_length[0], 0.f, 0.f, focal_length[1], 0.f, 0.f};
    return J;
}

/// \brief Compute the Hessian of the projection: H = d²(image_point) / d(camera_point)²
/// \param camera_point 3D point in camera space (x, y, z)
/// \param focal_length Focal length in pixels (fx, fy)
/// \return Array of two 3x3 Hessian matrices (H1 = ∂²u/∂p², H2 = ∂²v/∂p²)
TREND_HOST_DEVICE inline auto project_hess(
    fvec3 const &camera_point, fvec2 const &focal_length
) -> std::array<fmat3, 2> {
    // Hessian is zero for orthogonal projection
    return {fmat3{}, fmat3{}};
}

/// \brief Unproject a 2D image point to a ray in camera space.
/// \param image_point 2D point in image space
/// \param focal_length Focal length in pixels (fx, fy)
/// \param principal_point Principal point in pixels (cx, cy)
/// \return Ray in camera space (origin, direction)
TREND_HOST_DEVICE inline auto unproject(
    fvec2 const &image_point, fvec2 const &focal_length, fvec2 const &principal_point
) -> std::pair<fvec3, fvec3> {
    auto const xy = (image_point - principal_point) / focal_length;
    auto const origin = fvec3{xy[0], xy[1], 0.f};
    auto const dir = fvec3{0.f, 0.f, 1.f};
    return {origin, dir};
}

} // namespace tinyrend::camera::impl::orthogonal
