#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <chrono>
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "tinyrend/camera/orthogonal.h"

using namespace tinyrend::camera::orthogonal;

// Test project function
auto test_project() -> int {
    int fails = 0;

    {
        auto const camera_point = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);

        // project
        auto const image_point = project(camera_point, focal_length, principal_point);

        // unproject
        auto const [ray_origin, ray_dir] =
            unproject(image_point, focal_length, principal_point);
        auto const camera_point_unproj = ray_origin + ray_dir * camera_point.z;

        if (!is_close(camera_point_unproj, camera_point)) {
            printf("\n=== Testing unproject (orthogonal) ===\n");
            printf("\n[FAIL] Test 1:\n");
            printf(
                "  Camera Point Unprojected: %s\n",
                glm::to_string(camera_point_unproj).c_str()
            );
            printf("  Expected: %s\n", glm::to_string(camera_point).c_str());
            fails += 1;
        }
    }

    return fails;
}

auto main() -> int {
    int fails = 0;
    fails += test_project();

    if (fails == 0) {
        printf("\nAll tests passed!\n");
    } else {
        printf("\n%d tests failed!\n", fails);
    }

    return fails;
}