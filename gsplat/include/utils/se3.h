#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp> // glm slerp

#include "core/macros.h" // for GSPLAT_HOST_DEVICE

namespace gsplat::se3 {

// Interpolate two SE3 poses. Translation is interpolated linearly, rotation
// is interpolated using slerp. pose1 corresponds to ratio=0, pose2 corresponds
// to ratio=1.
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

// Interpolate two SE3 poses. Translation is interpolated linearly, rotation
// is interpolated using slerp. pose1 corresponds to ratio=0, pose2 corresponds
// to ratio=1.
GSPLAT_HOST_DEVICE inline auto interpolate(
    const float ratio,
    const glm::fmat3 &rot1,
    const glm::fvec3 &transl1,
    const glm::fmat3 &rot2,
    const glm::fvec3 &transl2
) -> std::pair<glm::fmat3, glm::fvec3> {
    auto const transl = (1.f - ratio) * transl1 + ratio * transl2;
    auto const rot =
        glm::slerp(glm::quat_cast(rot1), glm::quat_cast(rot2), ratio);
    return {glm::mat3_cast(rot), transl};
}

// Transform a point using SE3 matrix
GSPLAT_HOST_DEVICE inline auto transform_point(
    const glm::fmat3 &rot, const glm::fvec3 &transl, const glm::fvec3 &point
) -> glm::fvec3 {
    return rot * point + transl;
}

// Transform a point using SE3 matrix
GSPLAT_HOST_DEVICE inline auto transform_point(
    const glm::fquat &rot, const glm::fvec3 &transl, const glm::fvec3 &point
) -> glm::fvec3 {
    return glm::mat3_cast(rot) * point + transl;
}

// Inverse transform a point using SE3 matrix
GSPLAT_HOST_DEVICE inline auto invtransform_point(
    const glm::fmat3 &rot, const glm::fvec3 &transl, const glm::fvec3 &point
) -> glm::fvec3 {
    return glm::transpose(rot) * (point - transl);
}

// Inverse transform a point using SE3 matrix
GSPLAT_HOST_DEVICE inline auto invtransform_point(
    const glm::fquat &rot, const glm::fvec3 &transl, const glm::fvec3 &point
) -> glm::fvec3 {
    return glm::transpose(glm::mat3_cast(rot)) * (point - transl);
}

// Transform a direction using SE3 matrix
GSPLAT_HOST_DEVICE inline auto
transform_dir(const glm::fmat3 &rot, const glm::fvec3 &dir) -> glm::fvec3 {
    return rot * dir;
}

// Transform a direction using SE3 matrix
GSPLAT_HOST_DEVICE inline auto
transform_dir(const glm::fquat &rot, const glm::fvec3 &dir) -> glm::fvec3 {
    return glm::mat3_cast(rot) * dir;
}

// Inverse transform a direction using SE3 matrix
GSPLAT_HOST_DEVICE inline auto
invtransform_dir(const glm::fmat3 &rot, const glm::fvec3 &dir) -> glm::fvec3 {
    return glm::transpose(rot) * dir;
}

// Inverse transform a direction using SE3 matrix
GSPLAT_HOST_DEVICE inline auto
invtransform_dir(const glm::fquat &rot, const glm::fvec3 &dir) -> glm::fvec3 {
    return glm::transpose(glm::mat3_cast(rot)) * dir;
}

// Transform a ray using SE3 matrix
GSPLAT_HOST_DEVICE inline auto transform_ray(
    const glm::fmat3 &rot,
    const glm::fvec3 &transl,
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d
) -> std::tuple<glm::fvec3, glm::fvec3> {
    return {rot * ray_o + transl, rot * ray_d};
}

// Transform a ray using SE3 matrix
GSPLAT_HOST_DEVICE inline auto transform_ray(
    const glm::fquat &rot,
    const glm::fvec3 &transl,
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d
) -> std::tuple<glm::fvec3, glm::fvec3> {
    auto const R = glm::mat3_cast(rot);
    return {R * ray_o + transl, R * ray_d};
}

// Inverse transform a ray using SE3 matrix
GSPLAT_HOST_DEVICE inline auto invtransform_ray(
    const glm::fmat3 &rot,
    const glm::fvec3 &transl,
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d
) -> std::tuple<glm::fvec3, glm::fvec3> {
    auto const R_inv = glm::transpose(rot);
    return {R_inv * (ray_o - transl), R_inv * ray_d};
}

// Inverse transform a ray using SE3 matrix
GSPLAT_HOST_DEVICE inline auto invtransform_ray(
    const glm::fquat &rot,
    const glm::fvec3 &transl,
    const glm::fvec3 &ray_o,
    const glm::fvec3 &ray_d
) -> std::tuple<glm::fvec3, glm::fvec3> {
    auto const R_inv = glm::mat3_cast(glm::inverse(rot));
    return {R_inv * (ray_o - transl), R_inv * ray_d};
}

// Transform a covariance matrix using SE3 matrix
GSPLAT_HOST_DEVICE inline auto
transform_covar(const glm::fmat3 &rot, const glm::fmat3 &covar) -> glm::fmat3 {
    return rot * covar * glm::transpose(rot);
}

// Transform a covariance matrix using SE3 matrix
GSPLAT_HOST_DEVICE inline auto
transform_covar(const glm::fquat &rot, const glm::fmat3 &covar) -> glm::fmat3 {
    auto const R = glm::mat3_cast(rot);
    return R * covar * glm::transpose(R);
}

// Inverse transform a covariance matrix using SE3 matrix
GSPLAT_HOST_DEVICE inline auto invtransform_covar(
    const glm::fmat3 &rot, const glm::fmat3 &covar
) -> glm::fmat3 {
    return glm::transpose(rot) * covar * rot;
}

// Inverse transform a covariance matrix using SE3 matrix
GSPLAT_HOST_DEVICE inline auto invtransform_covar(
    const glm::fquat &rot, const glm::fmat3 &covar
) -> glm::fmat3 {
    auto const R = glm::mat3_cast(rot);
    return glm::transpose(R) * covar * R;
}

} // namespace gsplat::se3