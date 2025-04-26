#include "camera/model.h"
#include "camera/opencv_pinhole.h"
#include "camera/opencv_fisheye.h"
#include "camera/orthogonal.h"
#include <stdio.h>
#include <glm/gtx/string_cast.hpp>

using namespace gsplat;

void test_pinhole_camera() {
    printf("\n=== Testing Pinhole Camera ===\n");
    
    // Test case 1: Basic pinhole camera with no distortion
    {
        printf("\nTest 1: Basic pinhole camera\n");
        auto const focal_length = glm::fvec2(800.0f, 600.0f);
        auto const principal_point = glm::fvec2(400.0f, 300.0f);
        std::array<uint32_t, 2> resolution = {800, 600};

        auto projector = BatchedOpencvPinholeProjection(1, &focal_length, &principal_point);
        projector.set_index(0);

        auto const pose_start = SE3Quat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        // Test point in front of camera
        auto const world_point = glm::fvec3(0.2f, 0.4f, 3.0f);
        auto const &[image_point, depth, valid_flag] = camera_model.world_point_to_image_point(world_point);
        printf("Point in front of camera:\n");
        printf("  World Point: %s\n", glm::to_string(world_point).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point).c_str(), depth, valid_flag);

        // Test point behind camera
        auto const world_point_behind = glm::fvec3(0.2f, 0.4f, -3.0f);
        auto const &[image_point_behind, depth_behind, valid_behind] = 
            camera_model.world_point_to_image_point(world_point_behind);
        printf("\nPoint behind camera:\n");
        printf("  World Point: %s\n", glm::to_string(world_point_behind).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point_behind).c_str(), depth_behind, valid_behind);
    }

    // Test case 2: Pinhole camera with distortion
    {
        printf("\nTest 2: Pinhole camera with distortion\n");
        auto const focal_length = glm::fvec2(800.0f, 600.0f);
        auto const principal_point = glm::fvec2(400.0f, 300.0f);
        std::array<float, 6> radial_coeffs = {0.1f, 0.01f, -0.01f, 0.01f, -0.01f, 0.01f};
        std::array<float, 2> tangential_coeffs = {0.1f, -0.1f};
        std::array<float, 4> thin_prism_coeffs = {0.1f, -0.1f, -0.05f, -0.2f};
        std::array<uint32_t, 2> resolution = {800, 600};

        auto projector = BatchedOpencvPinholeProjection(
            1, &focal_length, &principal_point, &radial_coeffs, 
            &tangential_coeffs, &thin_prism_coeffs
        );
        projector.set_index(0);

        auto const pose_start = SE3Quat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        // Test point with distortion
        auto const world_point = glm::fvec3(0.2f, 0.4f, 3.0f);
        auto const &[image_point, depth, valid_flag] = camera_model.world_point_to_image_point(world_point);
        printf("Point with distortion:\n");
        printf("  World Point: %s\n", glm::to_string(world_point).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point).c_str(), depth, valid_flag);
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

        auto projector = BatchedOpencvFisheyeProjection(1, &focal_length, &principal_point);
        projector.set_index(0);

        auto const pose_start = SE3Quat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        // Test point in front of camera
        auto const world_point = glm::fvec3(0.2f, 0.4f, 3.0f);
        auto const &[image_point, depth, valid_flag] = camera_model.world_point_to_image_point(world_point);
        printf("Point in front of camera:\n");
        printf("  World Point: %s\n", glm::to_string(world_point).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point).c_str(), depth, valid_flag);

        // Test point at large angle
        auto const world_point_large_angle = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const &[image_point_large, depth_large, valid_large] = 
            camera_model.world_point_to_image_point(world_point_large_angle);
        printf("\nPoint at large angle:\n");
        printf("  World Point: %s\n", glm::to_string(world_point_large_angle).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point_large).c_str(), depth_large, valid_large);
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

        auto projector = BatchedOrthogonalProjection(1, &focal_length, &principal_point);
        projector.set_index(0);

        auto const pose_start = SE3Quat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        // Test point in front of camera
        auto const world_point = glm::fvec3(0.2f, 0.4f, 3.0f);
        auto const &[image_point, depth, valid_flag] = camera_model.world_point_to_image_point(world_point);
        printf("Point in front of camera:\n");
        printf("  World Point: %s\n", glm::to_string(world_point).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point).c_str(), depth, valid_flag);

        // Test point at different depth
        auto const world_point_deep = glm::fvec3(0.2f, 0.4f, 10.0f);
        auto const &[image_point_deep, depth_deep, valid_deep] = 
            camera_model.world_point_to_image_point(world_point_deep);
        printf("\nPoint at different depth:\n");
        printf("  World Point: %s\n", glm::to_string(world_point_deep).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point_deep).c_str(), depth_deep, valid_deep);
    }
}

void test_camera_transformations() {
    printf("\n=== Testing Camera Transformations ===\n");
    
    auto const focal_length = glm::fvec2(800.0f, 600.0f);
    auto const principal_point = glm::fvec2(400.0f, 300.0f);
    std::array<uint32_t, 2> resolution = {800, 600};

    auto projector = BatchedOpencvPinholeProjection(1, &focal_length, &principal_point);
    projector.set_index(0);

    // Test case 1: Camera rotation
    {
        printf("\nTest 1: Camera rotation\n");
        auto const rotation = glm::angleAxis(glm::radians(45.0f), glm::fvec3(0.0f, 1.0f, 0.0f));
        auto const pose_start = SE3Quat{glm::fvec3(0.f), rotation};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        auto const world_point = glm::fvec3(1.0f, 0.0f, 1.0f);
        auto const &[image_point, depth, valid_flag] = camera_model.world_point_to_image_point(world_point);
        printf("Point with camera rotation:\n");
        printf("  World Point: %s\n", glm::to_string(world_point).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point).c_str(), depth, valid_flag);
    }

    // Test case 2: Camera translation
    {
        printf("\nTest 2: Camera translation\n");
        auto const translation = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const pose_start = SE3Quat{translation, glm::fmat3(1.f)};
        auto camera_model = CameraModel(resolution, projector, pose_start);

        auto const world_point = glm::fvec3(1.0f, 0.0f, 1.0f);
        auto const &[image_point, depth, valid_flag] = camera_model.world_point_to_image_point(world_point);
        printf("Point with camera translation:\n");
        printf("  World Point: %s\n", glm::to_string(world_point).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point).c_str(), depth, valid_flag);
    }
}

void test_rolling_shutter() {
    printf("\n=== Testing Rolling Shutter ===\n");
    
    auto const focal_length = glm::fvec2(800.0f, 600.0f);
    auto const principal_point = glm::fvec2(400.0f, 300.0f);
    std::array<uint32_t, 2> resolution = {800, 600};

    auto projector = BatchedOpencvPinholeProjection(1, &focal_length, &principal_point);
    projector.set_index(0);

    // Test case 1: Top-to-bottom rolling shutter
    {
        printf("\nTest 1: Top-to-bottom rolling shutter\n");
        auto const pose_start = SE3Quat{glm::fvec3(0.f), glm::fmat3(1.f)};
        auto const pose_end = SE3Quat{glm::fvec3(0.f), 
            glm::angleAxis(glm::radians(10.0f), glm::fvec3(0.0f, 1.0f, 0.0f))};
        
        auto camera_model = CameraModel(
            resolution, projector, pose_start, pose_end, 
            ShutterType::ROLLING_TOP_TO_BOTTOM
        );

        // Test point at top of image
        auto const world_point_top = glm::fvec3(0.2f, 0.4f, 3.0f);
        auto const &[image_point_top, depth_top, valid_top] = 
            camera_model.world_point_to_image_point(world_point_top);
        printf("Point at top of image:\n");
        printf("  World Point: %s\n", glm::to_string(world_point_top).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point_top).c_str(), depth_top, valid_top);

        // Test point at bottom of image
        auto const world_point_bottom = glm::fvec3(0.2f, -0.4f, 3.0f);
        auto const &[image_point_bottom, depth_bottom, valid_bottom] = 
            camera_model.world_point_to_image_point(world_point_bottom);
        printf("\nPoint at bottom of image:\n");
        printf("  World Point: %s\n", glm::to_string(world_point_bottom).c_str());
        printf("  Image Point: %s, Depth: %f, Valid: %d\n", 
               glm::to_string(image_point_bottom).c_str(), depth_bottom, valid_bottom);
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