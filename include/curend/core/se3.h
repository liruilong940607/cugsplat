#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp> // glm slerp

#include "curend/core/macros.h" // for GSPLAT_HOST_DEVICE

namespace curend::se3 {

/// \brief Interpolate two SE(3) poses using quaternion rotation
/// \param ratio Interpolation ratio (0 for pose1, 1 for pose2)
/// \param rot1 First rotation as quaternion
/// \param transl1 First translation
/// \param rot2 Second rotation as quaternion
/// \param transl2 Second translation
/// \return Pair of interpolated rotation and translation
GSPLAT_HOST_DEVICE inline auto interpolate(
    const float ratio,
    const glm::fquat &rot1,
    const glm::fvec3 &transl1,
    const glm::fquat &rot2,
    const glm::fvec3 &transl2
) -> std::pair<glm::fquat, glm::fvec3> {
    auto const transl = (1.f - ratio) * transl1 + ratio * transl2;
    auto const rot = glm::slerp(rot1, rot2, ratio);
    return {rot, transl};
}

/// \brief Interpolate two SE(3) poses using matrix rotation
/// \param ratio Interpolation ratio (0 for pose1, 1 for pose2)
/// \param rot1 First rotation as 3x3 matrix
/// \param transl1 First translation
/// \param rot2 Second rotation as 3x3 matrix
/// \param transl2 Second translation
/// \return Pair of interpolated rotation and translation
GSPLAT_HOST_DEVICE inline auto interpolate(
    const float ratio,
    const glm::fmat3 &rot1,
    const glm::fvec3 &transl1,
    const glm::fmat3 &rot2,
    const glm::fvec3 &transl2
) -> std::pair<glm::fmat3, glm::fvec3> {
    auto const transl = (1.f - ratio) * transl1 + ratio * transl2;
    auto const rot = glm::slerp(glm::quat_cast(rot1), glm::quat_cast(rot2), ratio);
    return {glm::mat3_cast(rot), transl};
}

/// \brief Transform a point using SE(3) matrix
/// \param rot Rotation matrix
/// \param transl Translation vector
/// \param point Point to transform
/// \return Transformed point
GSPLAT_HOST_DEVICE inline auto transform_point(
    const glm::fmat3 &rot, const glm::fvec3 &transl, const glm::fvec3 &point
) -> glm::fvec3 {
    return rot * point + transl;
}

/// \brief Transform a point using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param transl Translation vector
/// \param point Point to transform
/// \return Transformed point
GSPLAT_HOST_DEVICE inline auto transform_point(
    const glm::fquat &rot, const glm::fvec3 &transl, const glm::fvec3 &point
) -> glm::fvec3 {
    return glm::mat3_cast(rot) * point + transl;
}

/// \brief Inverse transform a point using SE(3) matrix
/// \param rot Rotation matrix
/// \param transl Translation vector
/// \param point Point to inverse transform
/// \return Inverse transformed point
GSPLAT_HOST_DEVICE inline auto invtransform_point(
    const glm::fmat3 &rot, const glm::fvec3 &transl, const glm::fvec3 &point
) -> glm::fvec3 {
    return glm::transpose(rot) * (point - transl);
}

/// \brief Inverse transform a point using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param transl Translation vector
/// \param point Point to inverse transform
/// \return Inverse transformed point
GSPLAT_HOST_DEVICE inline auto invtransform_point(
    const glm::fquat &rot, const glm::fvec3 &transl, const glm::fvec3 &point
) -> glm::fvec3 {
    return glm::transpose(glm::mat3_cast(rot)) * (point - transl);
}

/// \brief Transform a direction using SE(3) matrix
/// \param rot Rotation matrix
/// \param dir Direction to transform
/// \return Transformed direction
GSPLAT_HOST_DEVICE inline auto
transform_dir(const glm::fmat3 &rot, const glm::fvec3 &dir) -> glm::fvec3 {
    return rot * dir;
}

/// \brief Transform a direction using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param dir Direction to transform
/// \return Transformed direction
GSPLAT_HOST_DEVICE inline auto
transform_dir(const glm::fquat &rot, const glm::fvec3 &dir) -> glm::fvec3 {
    return glm::mat3_cast(rot) * dir;
}

/// \brief Inverse transform a direction using SE(3) matrix
/// \param rot Rotation matrix
/// \param dir Direction to inverse transform
/// \return Inverse transformed direction
GSPLAT_HOST_DEVICE inline auto
invtransform_dir(const glm::fmat3 &rot, const glm::fvec3 &dir) -> glm::fvec3 {
    return glm::transpose(rot) * dir;
}

/// \brief Inverse transform a direction using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param dir Direction to inverse transform
/// \return Inverse transformed direction
GSPLAT_HOST_DEVICE inline auto
invtransform_dir(const glm::fquat &rot, const glm::fvec3 &dir) -> glm::fvec3 {
    return glm::transpose(glm::mat3_cast(rot)) * dir;
}

/// \brief Transform a ray using SE(3) matrix
/// \param rot Rotation matrix
/// \param transl Translation vector
/// \param ray_o Ray origin
/// \param ray_d Ray direction
/// \return Tuple of transformed ray origin and direction
GSPLAT_HOST_DEVICE inline auto transform_ray(
    const glm::fmat3 &rot,
    const glm::fvec3 &transl,
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d
) -> std::tuple<glm::fvec3, glm::fvec3> {
    return {rot * ray_o + transl, rot * ray_d};
}

/// \brief Transform a ray using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param transl Translation vector
/// \param ray_o Ray origin
/// \param ray_d Ray direction
/// \return Tuple of transformed ray origin and direction
GSPLAT_HOST_DEVICE inline auto transform_ray(
    const glm::fquat &rot,
    const glm::fvec3 &transl,
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d
) -> std::tuple<glm::fvec3, glm::fvec3> {
    auto const R = glm::mat3_cast(rot);
    return {R * ray_o + transl, R * ray_d};
}

/// \brief Inverse transform a ray using SE(3) matrix
/// \param rot Rotation matrix
/// \param transl Translation vector
/// \param ray_o Ray origin
/// \param ray_d Ray direction
/// \return Tuple of inverse transformed ray origin and direction
GSPLAT_HOST_DEVICE inline auto invtransform_ray(
    const glm::fmat3 &rot,
    const glm::fvec3 &transl,
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d
) -> std::tuple<glm::fvec3, glm::fvec3> {
    auto const R_inv = glm::transpose(rot);
    return {R_inv * (ray_o - transl), R_inv * ray_d};
}

/// \brief Inverse transform a ray using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param transl Translation vector
/// \param ray_o Ray origin
/// \param ray_d Ray direction
/// \return Tuple of inverse transformed ray origin and direction
GSPLAT_HOST_DEVICE inline auto invtransform_ray(
    const glm::fquat &rot,
    const glm::fvec3 &transl,
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d
) -> std::tuple<glm::fvec3, glm::fvec3> {
    auto const R_inv = glm::mat3_cast(glm::inverse(rot));
    return {R_inv * (ray_o - transl), R_inv * ray_d};
}

/// \brief Transform a covariance matrix using SE(3) matrix
/// \param rot Rotation matrix
/// \param covar Covariance matrix to transform
/// \return Transformed covariance matrix
GSPLAT_HOST_DEVICE inline auto
transform_covar(const glm::fmat3 &rot, const glm::fmat3 &covar) -> glm::fmat3 {
    return rot * covar * glm::transpose(rot);
}

/// \brief Transform a covariance matrix using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param covar Covariance matrix to transform
/// \return Transformed covariance matrix
GSPLAT_HOST_DEVICE inline auto
transform_covar(const glm::fquat &rot, const glm::fmat3 &covar) -> glm::fmat3 {
    auto const R = glm::mat3_cast(rot);
    return R * covar * glm::transpose(R);
}

/// \brief Inverse transform a covariance matrix using SE(3) matrix
/// \param rot Rotation matrix
/// \param covar Covariance matrix to inverse transform
/// \return Inverse transformed covariance matrix
GSPLAT_HOST_DEVICE inline auto
invtransform_covar(const glm::fmat3 &rot, const glm::fmat3 &covar) -> glm::fmat3 {
    return glm::transpose(rot) * covar * rot;
}

/// \brief Inverse transform a covariance matrix using SE(3) quaternion
/// \param rot Rotation quaternion
/// \param covar Covariance matrix to inverse transform
/// \return Inverse transformed covariance matrix
GSPLAT_HOST_DEVICE inline auto
invtransform_covar(const glm::fquat &rot, const glm::fmat3 &covar) -> glm::fmat3 {
    auto const R = glm::mat3_cast(rot);
    return glm::transpose(R) * covar * R;
}

} // namespace curend::se3