#pragma once

#include <algorithm>
#include <cstdint>

#include "tinyrend/common/macros.h" // for TREND_HOST_DEVICE
#include "tinyrend/common/mat.h"
#include "tinyrend/common/quat.h"
#include "tinyrend/common/vec.h"

namespace tinyrend::se3 {

/// \brief Interpolate two SE(3) poses using quaternion rotation
/// \param ratio Interpolation ratio (0 for pose1, 1 for pose2)
/// \param rot1 First rotation as quaternion
/// \param transl1 First translation
/// \param rot2 Second rotation as quaternion
/// \param transl2 Second translation
/// \return Pair of interpolated rotation and translation
TREND_HOST_DEVICE inline auto interpolate(
    const float ratio,
    const fquat &rot1,
    const fvec3 &transl1,
    const fquat &rot2,
    const fvec3 &transl2
) -> std::pair<fquat, fvec3> {
    auto const transl = (1.f - ratio) * transl1 + ratio * transl2;
    auto const rot = slerp(rot1, rot2, ratio);
    return {rot, transl};
}

/// \brief Interpolate two SE(3) poses using matrix rotation
/// \param ratio Interpolation ratio (0 for pose1, 1 for pose2)
/// \param rot1 First rotation as 3x3 matrix
/// \param transl1 First translation
/// \param rot2 Second rotation as 3x3 matrix
/// \param transl2 Second translation
/// \return Pair of interpolated rotation and translation
TREND_HOST_DEVICE inline auto interpolate(
    const float ratio,
    const fmat3 &rot1,
    const fvec3 &transl1,
    const fmat3 &rot2,
    const fvec3 &transl2
) -> std::pair<fmat3, fvec3> {
    auto const transl = (1.f - ratio) * transl1 + ratio * transl2;
    auto const rot = slerp(quat_cast(rot1), quat_cast(rot2), ratio);
    return {mat3_cast(rot), transl};
}

/// \brief Transform a point using SE(3) matrix
/// \param rot Rotation matrix
/// \param transl Translation vector
/// \param point Point to transform
/// \return Transformed point
TREND_HOST_DEVICE inline auto
transform_point(const fmat3 &rot, const fvec3 &transl, const fvec3 &point) -> fvec3 {
    return rot * point + transl;
}

/// \brief Transform a point using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param transl Translation vector
/// \param point Point to transform
/// \return Transformed point
TREND_HOST_DEVICE inline auto
transform_point(const fquat &rot, const fvec3 &transl, const fvec3 &point) -> fvec3 {
    return mat3_cast(rot) * point + transl;
}

/// \brief Inverse transform a point using SE(3) matrix
/// \param rot Rotation matrix
/// \param transl Translation vector
/// \param point Point to inverse transform
/// \return Inverse transformed point
TREND_HOST_DEVICE inline auto
invtransform_point(const fmat3 &rot, const fvec3 &transl, const fvec3 &point) -> fvec3 {
    return rot.transpose() * (point - transl);
}

/// \brief Inverse transform a point using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param transl Translation vector
/// \param point Point to inverse transform
/// \return Inverse transformed point
TREND_HOST_DEVICE inline auto
invtransform_point(const fquat &rot, const fvec3 &transl, const fvec3 &point) -> fvec3 {
    return mat3_cast(rot).transpose() * (point - transl);
}

/// \brief Transform a direction using SE(3) matrix
/// \param rot Rotation matrix
/// \param dir Direction to transform
/// \return Transformed direction
TREND_HOST_DEVICE inline auto
transform_dir(const fmat3 &rot, const fvec3 &dir) -> fvec3 {
    return rot * dir;
}

/// \brief Transform a direction using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param dir Direction to transform
/// \return Transformed direction
TREND_HOST_DEVICE inline auto
transform_dir(const fquat &rot, const fvec3 &dir) -> fvec3 {
    return mat3_cast(rot) * dir;
}

/// \brief Inverse transform a direction using SE(3) matrix
/// \param rot Rotation matrix
/// \param dir Direction to inverse transform
/// \return Inverse transformed direction
TREND_HOST_DEVICE inline auto
invtransform_dir(const fmat3 &rot, const fvec3 &dir) -> fvec3 {
    return rot.transpose() * dir;
}

/// \brief Inverse transform a direction using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param dir Direction to inverse transform
/// \return Inverse transformed direction
TREND_HOST_DEVICE inline auto
invtransform_dir(const fquat &rot, const fvec3 &dir) -> fvec3 {
    return mat3_cast(rot).transpose() * dir;
}

/// \brief Transform a ray using SE(3) matrix
/// \param rot Rotation matrix
/// \param transl Translation vector
/// \param ray_o Ray origin
/// \param ray_d Ray direction
/// \return Tuple of transformed ray origin and direction
TREND_HOST_DEVICE inline auto transform_ray(
    const fmat3 &rot, const fvec3 &transl, const fvec3 &ray_o, const fvec3 &ray_d
) -> std::tuple<fvec3, fvec3> {
    return {rot * ray_o + transl, rot * ray_d};
}

/// \brief Transform a ray using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param transl Translation vector
/// \param ray_o Ray origin
/// \param ray_d Ray direction
/// \return Tuple of transformed ray origin and direction
TREND_HOST_DEVICE inline auto transform_ray(
    const fquat &rot, const fvec3 &transl, const fvec3 &ray_o, const fvec3 &ray_d
) -> std::tuple<fvec3, fvec3> {
    auto const R = mat3_cast(rot);
    return {R * ray_o + transl, R * ray_d};
}

/// \brief Inverse transform a ray using SE(3) matrix
/// \param rot Rotation matrix
/// \param transl Translation vector
/// \param ray_o Ray origin
/// \param ray_d Ray direction
/// \return Tuple of inverse transformed ray origin and direction
TREND_HOST_DEVICE inline auto invtransform_ray(
    const fmat3 &rot, const fvec3 &transl, const fvec3 &ray_o, const fvec3 &ray_d
) -> std::tuple<fvec3, fvec3> {
    auto const R_inv = rot.transpose();
    return {R_inv * (ray_o - transl), R_inv * ray_d};
}

/// \brief Inverse transform a ray using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param transl Translation vector
/// \param ray_o Ray origin
/// \param ray_d Ray direction
/// \return Tuple of inverse transformed ray origin and direction
TREND_HOST_DEVICE inline auto invtransform_ray(
    const fquat &rot, const fvec3 &transl, const fvec3 &ray_o, const fvec3 &ray_d
) -> std::tuple<fvec3, fvec3> {
    auto const R_inv = mat3_cast(inverse(rot));
    return {R_inv * (ray_o - transl), R_inv * ray_d};
}

/// \brief Transform a covariance matrix using SE(3) matrix
/// \param rot Rotation matrix
/// \param covar Covariance matrix to transform
/// \return Transformed covariance matrix
TREND_HOST_DEVICE inline auto
transform_covar(const fmat3 &rot, const fmat3 &covar) -> fmat3 {
    return rot * covar * rot.transpose();
}

/// \brief Transform a covariance matrix using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param covar Covariance matrix to transform
/// \return Transformed covariance matrix
TREND_HOST_DEVICE inline auto
transform_covar(const fquat &rot, const fmat3 &covar) -> fmat3 {
    auto const R = mat3_cast(rot);
    return R * covar * R.transpose();
}

/// \brief Inverse transform a covariance matrix using SE(3) matrix
/// \param rot Rotation matrix
/// \param covar Covariance matrix to inverse transform
/// \return Inverse transformed covariance matrix
TREND_HOST_DEVICE inline auto
invtransform_covar(const fmat3 &rot, const fmat3 &covar) -> fmat3 {
    return rot.transpose() * covar * rot;
}

/// \brief Inverse transform a covariance matrix using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param covar Covariance matrix to inverse transform
/// \return Inverse transformed covariance matrix
TREND_HOST_DEVICE inline auto
invtransform_covar(const fquat &rot, const fmat3 &covar) -> fmat3 {
    auto const R = mat3_cast(rot);
    return R.transpose() * covar * R;
}

} // namespace tinyrend::se3