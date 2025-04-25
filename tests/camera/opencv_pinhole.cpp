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

    auto const &[J, valid_flag] =
        projector.camera_point_to_image_point_jacobian(camera_point);
    printf("Jacobian: \n");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            printf("%f ", J[j][i]);
        }
        printf("\n");
    }

    // finite difference to verify the Jacobian
    auto const delta = 1e-5f;
    auto const &[image_point_plus_x, _] =
        projector.camera_point_to_image_point(camera_point + glm::fvec3(delta, 0.f, 0.f));
    auto const &[image_point_plus_y, __] =
        projector.camera_point_to_image_point(camera_point + glm::fvec3(0.f, delta, 0.f));
    auto const &[image_point_plus_z, ___] =
        projector.camera_point_to_image_point(camera_point + glm::fvec3(0.f, 0.f, delta));
    auto const &[image_point_minus_x, ____] =
        projector.camera_point_to_image_point(camera_point - glm::fvec3(delta, 0.f, 0.f));
    auto const &[image_point_minus_y, _____] =
        projector.camera_point_to_image_point(camera_point - glm::fvec3(0.f, delta, 0.f));
    auto const &[image_point_minus_z, ______] =
        projector.camera_point_to_image_point(camera_point - glm::fvec3(0.f, 0.f, delta));
    auto const J_x = (image_point_plus_x - image_point_minus_x) / (2.f * delta);
    auto const J_y = (image_point_plus_y - image_point_minus_y) / (2.f * delta);
    auto const J_z = (image_point_plus_z - image_point_minus_z) / (2.f * delta);
    auto const J_ = glm::fmat3x2{J_x, J_y, J_z};
    printf("Jacobian (finite difference): \n");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            printf("%f ", J_[j][i]);
        }
        printf("\n");
    }

}