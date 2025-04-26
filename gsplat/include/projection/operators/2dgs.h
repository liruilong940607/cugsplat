#pragma once

#include <glm/glm.hpp>
#include <stdint.h>

#include "camera/model.h"
#include "utils/gaussian.h"
#include "gaussian/utils.h"
#include "core/types.h"

namespace gsplat {

struct ProjectionOperator3DGS {
    // pointers to output buffer
    float *opacity_ptr;
    glm::fvec2 *mean_ptr;
    glm::fmat3 *transform_ptr;
    float *depth_ptr;
    glm::fvec2 *radius_ptr;

    // parameters
    const float alpha_threshold = 1.0f / 255.0f;

    // cache: internal state to be written to output buffer
    float opacity;
    glm::fvec2 mean;
    glm::fmat3 transform;
    float depth;
    glm::fvec2 radius;

    template <class CameraProjection, class CameraPose, class Gaussian>
    inline GSPLAT_HOST_DEVICE bool preprocess(
        CameraModel<CameraProjection, CameraPose> &camera, Gaussian &gaussian
    ) {
        // Compute projected center.
        auto const world_point = gaussian.get_mean();
        auto const &[camera_point, image_point, point_valid_flag, pose] =
            camera._world_point_to_image_point(world_point);
        if (!point_valid_flag) {
            return false;
        }

        // Compute the projected gaussian on the image plane (TODO)
        auto const KWH = glm::fmat2x3{};

        // Compute aabb
        auto const M = transpose(fmat3(KWH[0], KWH[1], image_point));
        auto const M0 = M[0], M1 = M[1], M2 = M[2];
        auto const temp_point = glm::fvec3(1.0f, 1.0f, -1.0f);
        auto const distance = glm::compAdd(temp_point * M2 * M2);
        if (distance == 0.0f) {
            return false;
        }
        auto const f = (1.0f / distance) * temp_point;
        auto const center =
            glm::fvec2(glm::compAdd(f * M0 * M2), glm::compAdd(f * M1 * M2));
        auto const temp =
            glm::fvec2(glm::compAdd(f * M0 * M0), glm::compAdd(f * M1 * M1));
        auto const half_extend = center * center - temp;
        auto const radius =
            3.33f * glm::sqrt(glm::max(fvec2(1e-4f), half_extend));

        // Check again if the gaussian is outside the image plane
        auto const &[render_width, render_height] = camera.resolution;
        if (center.x - radius.x < 0 || center.x + radius.x > render_width ||
            center.y - radius.y < 0 || center.y + radius.y > render_height) {
            return false;
        }

        // this->opacity = opacity;
        this->mean = image_point;
        this->transform = transform;
        this->depth = camera_point.z;
        this->radius = radius;
        return true;
    }

    inline GSPLAT_HOST_DEVICE void write_to_buffer(uint32_t index) {
        // this->opacity_ptr[index] = this->opacity;
        this->mean_ptr[index] = this->mean;
        this->transform_ptr[index] = this->transform;
        this->depth_ptr[index] = this->depth;
        this->radius_ptr[index] = this->radius;
    }
};

} // namespace gsplat