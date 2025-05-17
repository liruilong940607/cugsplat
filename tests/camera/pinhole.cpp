#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <chrono>
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "curend/camera/pinhole.h"

using namespace curend::pinhole;

// Test project function (distorted pinhole)
auto test_project_distorted() -> int {
    int fails = 0;

    {
        auto const camera_point = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);
        auto const radial_coeffs =
            std::array<float, 6>{0.8f, -0.6f, 0.2f, 0.1f, -0.1f, 0.1f};
        auto const tangential_coeffs = std::array<float, 2>{0.0f, 0.0f};
        auto const thin_prism_coeffs = std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};

        // project with distortion
        auto const [image_point, valid] = project(
            camera_point,
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs
        );

        // unproject with distortion
        auto const [ray_dir, ray_valid] = unproject(
            image_point,
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs
        );

        // expected ray direction
        auto const expected_ray_dir = glm::normalize(camera_point);

        if (!valid || !ray_valid || !is_close(ray_dir, expected_ray_dir)) {
            printf("\n=== Testing unproject (distorted pinhole) ===\n");
            printf("\n[FAIL] Test 1:\n");
            printf("  Ray direction: %s\n", glm::to_string(ray_dir).c_str());
            printf("  Expected: %s\n", glm::to_string(expected_ray_dir).c_str());
            fails += 1;
        }
    }

    return fails;
}

auto main() -> int {
    int fails = 0;
    fails += test_project_distorted();

    if (fails == 0) {
        printf("\nAll tests passed!\n");
    } else {
        printf("\n%d tests failed!\n", fails);
    }

    return fails;
}