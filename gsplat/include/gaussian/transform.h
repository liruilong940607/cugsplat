#pragma once
#include <glm/glm.hpp>
#include <tuple.h>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE

// Transforms a 3D position from world space to camera space.
// [R | t] is the world-to-camera transformation.
inline GSPLAT_HOST_DEVICE auto point_world_to_camera(
    const glm::fvec3 &point_world, const glm::fmat3 &R, const glm::fvec3 &t
) -> glm::fvec3 {
    return R * point_world + t;
}

inline GSPLAT_HOST_DEVICE auto point_world_to_camera_vjp(
    // inputs
    const glm::fvec3 &point_world,
    const glm::fmat3 &R,
    // output gradients
    const glm::fvec3 &v_point_camera
) -> std::tuple<glm::fvec3, glm::fmat3, glm::fvec3> {
    auto const v_R = glm::outerProduct(v_point_camera, point_world);
    auto const v_t = v_point_camera;
    auto const v_point_world = glm::transpose(R) * v_point_camera;
    return {v_point_world, v_R, v_t};
}

// Transform a 3D covariance from world space to camera space.
// [R | t] is the world-to-camera transformation.
inline GSPLAT_HOST_DEVICE auto covariance_world_to_camera(
    const glm::fmat3 &covar_world, const glm::fmat3 &R
) -> glm::fmat3 {
    return R * covar_world * glm::transpose(R);
}

inline GSPLAT_HOST_DEVICE auto covariance_world_to_camera_vjp(
    // inputs
    const glm::fmat3 &covar_world,
    const glm::fmat3 &R,
    // output gradients
    const glm::fmat3 &v_covar_camera
) -> std::tuple<glm::fmat3, glm::fmat3> {
    auto const v_R =
        (v_covar_camera * R * glm::transpose(covar_world) +
         glm::transpose(v_covar_camera) * R * covar_world);
    auto const v_covar_world = glm::transpose(R) * v_covar_camera * R;
    return {v_covar_world, v_R};
}
