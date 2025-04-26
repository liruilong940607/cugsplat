#include "camera/model.h"
#include "camera/opencv_fisheye.h"
#include "camera/opencv_pinhole.h"
#include "camera/orthogonal.h"
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

using namespace gsplat;

template <class CameraProjection, class CameraPose>
void test_round_trip_consistency(
    CameraModel<CameraProjection, CameraPose> &camera_model,
    const glm::fvec2 &image_point,
    const char *test_name
) {
    // Image point to world ray
    auto const &[world_ray_o, world_ray_d, ray_valid] =
        camera_model.image_point_to_world_ray(image_point);

    if (!ray_valid) {
        printf("\n%s: Invalid ray\n", test_name);
        return;
    }

    // Choose a point at arbitrary deph
    const float t = 5.0f;
    auto const world_point = world_ray_o + t * world_ray_d;

    // World point back to image point
    auto const &[image_point_roundtrip, z_depth, point_valid] =
        camera_model.world_point_to_image_point(world_point);

    // Check consistency
    const float image_tol = 1e-4f;
    const float depth_tol = 1e-5f;
    bool image_consistent =
        glm::all(glm::equal(image_point, image_point_roundtrip, image_tol));
    bool consistent = point_valid && image_consistent;

    if (!consistent) {
        printf("\n%s:\n", test_name);
        printf(
            "  Initial Image Point: %s\n", glm::to_string(image_point).c_str()
        );
        printf("  World Ray Origin: %s\n", glm::to_string(world_ray_o).c_str());
        printf(
            "  World Ray Direction: %s\n", glm::to_string(world_ray_d).c_str()
        );
        printf(
            "  World Point (at t=%.2f): %s\n",
            t,
            glm::to_string(world_point).c_str()
        );
        printf(
            "  Round-trip Image Point: %s\n",
            glm::to_string(image_point_roundtrip).c_str()
        );
        printf("  Actual Depth (z): %.2f\n", z_depth);
        printf("  Point Valid: %d\n", point_valid);
        if (!image_consistent) {
            printf(
                "  Image Point Error: %.6f\n",
                glm::length(image_point - image_point_roundtrip)
            );
        }
    }
}

void test_pinhole_camera() {
    printf("\n=== Testing Pinhole Camera ===\n");

    // Test case 1: Basic pinhole camera with no distortion
    {
        printf("\nTest 1: Basic pinhole camera\n");
        auto const focal_length = glm::fvec2(800.0f, 600.0f);
        auto const principal_point = glm::fvec2(400.0f, 300.0f);
        std::array<uint32_t, 2> resolution = {800, 600};

        auto projector =
            BatchedOpencvPinholeProjection(1, &focal_length, &principal_point);
        projector.set_index(0);

        auto const pose_start = SE3Mat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        auto const image_point = glm::fvec2(400.0f, 300.0f);
        test_round_trip_consistency(
            camera_model, image_point, "Round-trip test"
        );
        auto const image_point2 = glm::fvec2(0.0f, 0.0f);
        test_round_trip_consistency(
            camera_model, image_point2, "Round-trip test"
        );
    }

    // Test case 2: Pinhole camera with distortion
    {
        printf("\nTest 2: Pinhole camera with distortion\n");
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

        auto const pose_start = SE3Mat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        auto const image_point = glm::fvec2(400.0f, 300.0f);
        test_round_trip_consistency(
            camera_model, image_point, "Round-trip test with distortion"
        );
        auto const image_point2 = glm::fvec2(0.0f, 0.0f);
        test_round_trip_consistency(
            camera_model, image_point2, "Round-trip test with distortion"
        );
    }
}

void test_fisheye_camera() {
    printf("\n=== Testing Fisheye Camera ===\n");

    // Test case 1: Basic fisheye camera
    {
        printf("\nTest 1: Basic fisheye camera\n");
        auto const focal_length = glm::fvec2(800.0f, 600.0f);
        auto const principal_point = glm::fvec2(400.0f, 300.0f);
        std::array<uint32_t, 2> resolution = {800, 600};

        auto projector =
            BatchedOpencvFisheyeProjection(1, &focal_length, &principal_point);
        projector.set_index(0);

        auto const pose_start = SE3Mat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        auto const image_point = glm::fvec2(400.0f, 300.0f);
        test_round_trip_consistency(
            camera_model, image_point, "Round-trip test"
        );
        auto const image_point2 = glm::fvec2(0.0f, 0.0f);
        test_round_trip_consistency(
            camera_model, image_point2, "Round-trip test"
        );
    }

    // Test case 2: Fisheye camera with distortion
    {
        printf("\nTest 2: Fisheye camera with distortion\n");
        auto const focal_length = glm::fvec2(800.0f, 600.0f);
        auto const principal_point = glm::fvec2(400.0f, 300.0f);
        std::array<float, 4> k = {0.1f, 0.01f, -0.01f, 0.01f};
        std::array<uint32_t, 2> resolution = {800, 600};

        auto projector = BatchedOpencvFisheyeProjection(
            1, &focal_length, &principal_point, &k
        );
        projector.set_index(0);

        auto const pose_start = SE3Mat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        auto const image_point = glm::fvec2(400.0f, 300.0f);
        test_round_trip_consistency(
            camera_model, image_point, "Round-trip test with distortion"
        );
        auto const image_point2 = glm::fvec2(0.0f, 0.0f);
        test_round_trip_consistency(
            camera_model, image_point2, "Round-trip test with distortion"
        );
    }
}

void test_orthogonal_camera() {
    printf("\n=== Testing Orthogonal Camera ===\n");

    // Test case 1: Basic orthogonal camera
    {
        printf("\nTest 1: Basic orthogonal camera\n");
        auto const focal_length = glm::fvec2(800.0f, 600.0f);
        auto const principal_point = glm::fvec2(400.0f, 300.0f);
        std::array<uint32_t, 2> resolution = {800, 600};

        auto projector =
            BatchedOrthogonalProjection(1, &focal_length, &principal_point);
        projector.set_index(0);

        auto const pose_start = SE3Mat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        auto const image_point = glm::fvec2(400.0f, 300.0f);
        test_round_trip_consistency(
            camera_model, image_point, "Round-trip test"
        );
        auto const image_point2 = glm::fvec2(0.0f, 0.0f);
        test_round_trip_consistency(
            camera_model, image_point2, "Round-trip test"
        );
    }
}

void test_camera_transformations() {
    printf("\n=== Testing Camera Transformations ===\n");

    auto const focal_length = glm::fvec2(800.0f, 600.0f);
    auto const principal_point = glm::fvec2(400.0f, 300.0f);
    std::array<uint32_t, 2> resolution = {800, 600};

    auto projector =
        BatchedOpencvPinholeProjection(1, &focal_length, &principal_point);
    projector.set_index(0);

    // Test case 1: Camera rotation and translation
    {
        printf("\nTest 1: Camera rotation\n");
        auto const rotation =
            glm::angleAxis(glm::radians(45.0f), glm::fvec3(0.0f, 1.0f, 0.0f));
        auto const translation = glm::fvec3(0.0f, 0.0f, 0.0f);
        auto const pose_start = SE3Quat{translation, rotation};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        auto const image_point = glm::fvec2(400.0f, 300.0f);
        test_round_trip_consistency(
            camera_model, image_point, "Round-trip test"
        );
        auto const image_point2 = glm::fvec2(0.0f, 0.0f);
        test_round_trip_consistency(
            camera_model, image_point2, "Round-trip test"
        );
    }
}

void test_rolling_shutter() {
    printf("\n=== Testing Rolling Shutter ===\n");

    auto const focal_length = glm::fvec2(800.0f, 600.0f);
    auto const principal_point = glm::fvec2(400.0f, 300.0f);
    std::array<uint32_t, 2> resolution = {800, 600};

    auto projector =
        BatchedOpencvPinholeProjection(1, &focal_length, &principal_point);
    projector.set_index(0);

    // Test case 1: Top-to-bottom rolling shutter
    {
        printf("\nTest 1: Top-to-bottom rolling shutter\n");
        auto const pose_start =
            SE3Quat{glm::fvec3(0.f), glm::identity<glm::fquat>()};
        auto const pose_end = SE3Quat{
            glm::fvec3(0.f),
            glm::angleAxis(glm::radians(10.0f), glm::fvec3(0.0f, 1.0f, 0.0f))
        };

        auto camera_model = CameraModel(
            resolution,
            projector,
            pose_start,
            pose_end,
            ShutterType::ROLLING_TOP_TO_BOTTOM
        );

        auto const image_point = glm::fvec2(400.0f, 300.0f);
        test_round_trip_consistency(
            camera_model, image_point, "Round-trip test"
        );
        auto const image_point2 = glm::fvec2(0.0f, 0.0f);
        test_round_trip_consistency(
            camera_model, image_point2, "Round-trip test"
        );
    }
}

int main() {
    test_pinhole_camera();
    test_fisheye_camera();
    test_orthogonal_camera();
    test_camera_transformations();
    test_rolling_shutter();
    return 0;
}