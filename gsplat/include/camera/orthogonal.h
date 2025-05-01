#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE
#include "core/math.h"
#include "core/tensor.h"
#include "utils/solver.h" // for solver_newton

namespace gsplat {

struct OrthogonalProjection {
    Tensor<glm::fvec2> focal_length;
    Tensor<glm::fvec2> principal_point;

    GSPLAT_HOST_DEVICE
    OrthogonalProjection() {}

    GSPLAT_HOST_DEVICE
    OrthogonalProjection(
        const glm::fvec2 *focal_length, const glm::fvec2 *principal_point
    )
        : focal_length(focal_length), principal_point(principal_point) {}

    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point(const glm::fvec3 &camera_point
    ) -> std::pair<glm::fvec2, bool> {
        auto const xy = glm::fvec2(camera_point);
        auto const focal_length = this->focal_length.get_data();
        auto const principal_point = this->principal_point.get_data();
        auto const image_point = focal_length * xy + principal_point;
        return {image_point, true};
    }

    GSPLAT_HOST_DEVICE auto image_point_to_camera_ray(
        const glm::fvec2 &image_point, bool normalize = true
    ) -> std::tuple<glm::fvec3, glm::fvec3, bool> {
        auto const focal_length = this->focal_length.get_data();
        auto const principal_point = this->principal_point.get_data();
        auto const xy = (image_point - principal_point) / focal_length;
        // for orthogonal projection, the camera ray origin is
        // (u, v, 0) and the direction is (0, 0, 1)
        auto const origin = glm::fvec3{xy[0], xy[1], 0.f};
        auto const dir = glm::fvec3{0.f, 0.f, 1.f};
        return {origin, dir, true};
    }

    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point_jacobian(const glm::fvec3 &camera_point
    ) -> std::pair<glm::fmat3x2, bool> {
        auto const focal_length = this->focal_length.get_data();
        auto const J =
            glm::fmat3x2{focal_length[0], 0.f, 0.f, focal_length[1], 0.f, 0.f};
        return {J, true};
    }
};

} // namespace gsplat