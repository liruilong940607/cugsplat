#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <tuple>

#include "core/macros.h" // for GSPLAT_HOST_DEVICE
#include "core/math.h"
#include "core/types.h"   // for Maybe
#include "utils/solver.h" // for solver_newton

namespace gsplat {

template <class Derived> struct OrthogonalProjectionImpl {
    GSPLAT_HOST_DEVICE auto
    camera_point_to_image_point(const glm::fvec3 &camera_point
    ) -> std::pair<glm::fvec2, bool> {
        auto const derived = static_cast<Derived *>(this);
        auto const xy = glm::fvec2(camera_point);
        auto const focal_length = derived->get_focal_length();
        auto const principal_point = derived->get_principal_point();
        auto const image_point = focal_length * xy + principal_point;
        return {image_point, true};
    }

    GSPLAT_HOST_DEVICE auto image_point_to_camera_ray(
        const glm::fvec2 &image_point, bool normalize = true
    ) -> std::tuple<glm::fvec3, glm::fvec3, bool> {
        auto const derived = static_cast<Derived *>(this);
        auto const focal_length = derived->get_focal_length();
        auto const principal_point = derived->get_principal_point();
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
        auto const derived = static_cast<Derived *>(this);
        auto const focal_length = derived->get_focal_length();
        auto const J =
            glm::fmat3x2{focal_length[0], 0.f, 0.f, focal_length[1], 0.f, 0.f};
        return {J, true};
    }
};

struct OrthogonalProjection : OrthogonalProjectionImpl<OrthogonalProjection> {
    glm::fvec2 focal_length;
    glm::fvec2 principal_point;

    GSPLAT_HOST_DEVICE
    OrthogonalProjection(glm::fvec2 focal_length, glm::fvec2 principal_point)
        : focal_length(focal_length), principal_point(principal_point) {}

    GSPLAT_HOST_DEVICE auto get_focal_length() const { return focal_length; }
    GSPLAT_HOST_DEVICE auto get_principal_point() const {
        return principal_point;
    }
};

struct BatchedOrthogonalProjection
    : OrthogonalProjectionImpl<BatchedOrthogonalProjection> {
    uint32_t n{0}, idx{0};
    GSPLAT_HOST_DEVICE void set_index(uint32_t i) { idx = i; }
    GSPLAT_HOST_DEVICE int get_n() const { return n; }

    // pointer to device memory
    const glm::fvec2 *focal_length_ptr;
    const glm::fvec2 *principal_point_ptr;

    // cache
    Maybe<glm::fvec2> focal_length;
    Maybe<glm::fvec2> principal_point;

    GSPLAT_HOST_DEVICE BatchedOrthogonalProjection(
        uint32_t n,
        const glm::fvec2 *focal_length_ptr,
        const glm::fvec2 *principal_point_ptr
    )
        : n(n), focal_length_ptr(focal_length_ptr),
          principal_point_ptr(principal_point_ptr) {}

    GSPLAT_HOST_DEVICE auto get_focal_length() {
        if (!focal_length.has_value()) {
            focal_length.set(focal_length_ptr[idx]);
        }
        return focal_length.get();
    }
    GSPLAT_HOST_DEVICE auto get_principal_point() {
        if (!principal_point.has_value()) {
            principal_point.set(principal_point_ptr[idx]);
        }
        return principal_point.get();
    }
};

} // namespace gsplat