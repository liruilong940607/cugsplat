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

    // auto projector = OpencvPinholeProjection(
    //     focal_length,
    //     principal_point,
    //     radial_coeffs,
    //     tangential_coeffs,
    //     thin_prism_coeffs
    // );

    auto projector = BatchedOpencvPinholeProjection(
        1,
        &focal_length,
        &principal_point,
        &radial_coeffs,
        &tangential_coeffs,
        &thin_prism_coeffs
    );
    projector.set_index(0);

    auto const camera_point = glm::fvec3(0.2f, 0.4f, 3.0f);
    auto const &[image_point, point_valid_flag] =
        projector.camera_point_to_image_point(camera_point);
    printf(
        "Image Point: (%f, %f), Valid: %d\n",
        image_point.x,
        image_point.y,
        point_valid_flag
    );
    auto const &[ray_o, ray_d, ray_valid_flag] =
        projector.image_point_to_camera_ray(image_point);
    printf(
        "Camera Ray Origin: (%f, %f, %f), Direction: (%f, %f, %f), Valid: %d\n",
        ray_o.x,
        ray_o.y,
        ray_o.z,
        ray_d.x,
        ray_d.y,
        ray_d.z,
        ray_valid_flag
    );

    auto const depth = 12.9f; // arbitrary depth
    auto const camera_point_ = ray_o + depth * ray_d;
    auto const &[image_point_, point_valid_flag_] =
        projector.camera_point_to_image_point(camera_point_);
    printf(
        "Image Point (verify): (%f, %f), Valid: %d\n",
        image_point_.x,
        image_point_.y,
        point_valid_flag_
    );
}