#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <cmath>
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "cugsplat/camera/shutter.h"
#include "cugsplat/impl.h"
#include "helpers.h"

using namespace cugsplat::impl;
using namespace cugsplat::shutter;

// Test projection with perfect camera shutter
auto test_projection() -> int {
    int fails = 0;

    // Test parameters
    const uint32_t width = 640;
    const uint32_t height = 480;
    const float near_plane = 0.0f;
    const float far_plane = 100.0f;
    const float margin_factor = 0.15f;

    // Camera intrinsics (3x3 matrix)
    const float Ks[9] = {
        100.0f,
        0.0f,
        320.0f, // focal_x, skew, principal_x
        0.0f,
        100.0f,
        240.0f, // 0, focal_y, principal_y
        0.0f,
        0.0f,
        1.0f // 0, 0, 1
    };

    // Camera poses (4x4 matrices)
    // Start pose: Identity
    const float viewmats0[16] = {
        1.0f,
        0.0f,
        0.0f,
        0.0f, // Identity rotation
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f
    };

    // End pose: Small rotation (5 degrees around Y) and small translation
    const float angle = 5.0f * 3.14159f / 180.0f; // 5 degrees in radians
    const float cos_angle = std::cos(angle);
    const float sin_angle = std::sin(angle);
    const float viewmats1[16] = {
        cos_angle,
        0.0f,
        sin_angle,
        0.1f, // Small rotation around Y + small X translation
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        -sin_angle,
        0.0f,
        cos_angle,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f
    };

    // Gaussian parameters
    const float means[3] = {1.0f, 1.0f, 1.0f};       // World position
    const float quats[4] = {1.0f, 0.0f, 0.0f, 0.0f}; // Identity rotation
    const float scales[3] = {0.1f, 0.1f, 0.1f};      // Scale

    // Test different shutter types
    const std::array<Type, 5> shutter_types = {
        Type::GLOBAL,
        Type::ROLLING_TOP_TO_BOTTOM,
        Type::ROLLING_LEFT_TO_RIGHT,
        Type::ROLLING_BOTTOM_TO_TOP,
        Type::ROLLING_RIGHT_TO_LEFT
    };

    for (const auto &shutter_type : shutter_types) {
        // Test projection
        auto [image_point, depth, cov2d, valid] = projection<CameraType::PINHOLE>(
            Ks,
            near_plane,
            far_plane,
            viewmats0,
            viewmats1,
            shutter_type,
            width,
            height,
            means,
            quats,
            scales,
            margin_factor
        );

        // Verify results
        if (!valid) {
            printf("\n=== Testing projection_perfect_camera_shutter ===\n");
            printf(
                "\n[FAIL] Test for shutter type %d:\n", static_cast<int>(shutter_type)
            );
            printf("  Projection failed\n");
            fails += 1;
            continue;
        }
    }

    return fails;
}

auto main() -> int {
    int fails = 0;
    fails += test_projection();

    if (fails == 0) {
        printf("\nAll tests passed!\n");
    } else {
        printf("\n%d tests failed!\n", fails);
    }

    return fails;
}