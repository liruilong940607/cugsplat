#include "camera/model.h"
#include "camera/opencv_pinhole.h"
#include <stdio.h>

using namespace gsplat;

int main() {
    auto const focal_length = glm::fvec2(800.0f, 600.0f);
    auto const principal_point = glm::fvec2(400.0f, 300.0f);
    std::array<float, 6> radial_coeffs = {
        0.1f, 0.01f, -0.01f, 0.01f, -0.01f, 0.01f
    };
    std::array<float, 2> tangential_coeffs = {0.1f, -0.1f};
    std::array<float, 4> thin_prism_coeffs = {0.1f, -0.1f, -0.05f, -0.2f};

    std::array<uint32_t, 2> resolution = {800, 600};

    auto projector = BatchedOpencvPinholeProjection(
        1,
        &focal_length,
        &principal_point,
        &radial_coeffs,
        &tangential_coeffs,
        &thin_prism_coeffs
    );
    projector.set_index(0);

    auto const pose_start = SE3Quat{glm::fvec3(0.f), glm::fmat3(1.f)};
    auto camera_model = CameraModel(resolution, projector, pose_start);

    auto const world_point = glm::fvec3(0.2f, 0.4f, 3.0f);
    auto const &[image_point, depth_, valid_flag] =
        camera_model.world_point_to_image_point(world_point);
    printf(
        "Image Point: (%f, %f), Valid: %d\n",
        image_point.x,
        image_point.y,
        valid_flag
    );

    auto const &[world_ray_o, world_ray_d, valid_flag_] =
        camera_model.image_point_to_world_ray(image_point);
    printf(
        "World Ray Origin: (%f, %f, %f), Direction: (%f, %f, %f), Valid: %d\n",
        world_ray_o.x,
        world_ray_o.y,
        world_ray_o.z,
        world_ray_d.x,
        world_ray_d.y,
        world_ray_d.z,
        valid_flag_
    );

    const float depth = 12.9f; // arbitrary depth
    auto const world_point_ = world_ray_o + depth * world_ray_d;
    auto const &[image_point_, depth__, valid_flag__] =
        camera_model.world_point_to_image_point(world_point_);
    printf(
        "Image Point (verify): (%f, %f), Valid: %d\n",
        image_point_.x,
        image_point_.y,
        valid_flag__
    );
}