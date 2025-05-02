#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "camera/shutter.h"

using namespace gsplat::shutter;

// Test point_world_to_image function with quaternion rotation
auto test_point_world_to_image_quat() -> int {
    int fails = 0;

    // Test case 1: Global shutter
    {
        auto const world_point = glm::fvec3(1.0f, 1.0f, 5.0f);
        auto const resolution = std::array<uint32_t, 2>{640, 480};
        auto const pose_r_start =
            glm::fquat(1.0f, 0.0f, 0.0f, 0.0f); // Identity rotation
        auto const pose_t_start = glm::fvec3(0.0f, 0.0f, 0.0f);
        auto const pose_r_end =
            glm::fquat(1.0f, 0.0f, 0.0f, 0.0f); // Identity rotation
        auto const pose_t_end = glm::fvec3(0.0f, 0.0f, 0.0f);

        // Simple projection function that just divides by z
        auto project_fn = [](const glm::fvec3 &p
                          ) -> std::pair<glm::fvec2, bool> {
            return {glm::fvec2(p.x / p.z, p.y / p.z), true};
        };

        auto const result = point_world_to_image(
            project_fn,
            resolution,
            world_point,
            pose_r_start,
            pose_t_start,
            pose_r_end,
            pose_t_end,
            Type::GLOBAL
        );

        if (result.valid_flag) {
            printf("\n=== Testing point_world_to_image (quaternion) ===\n");
            printf("\n[FAIL] Test 1: Global shutter should not be valid\n");
            fails += 1;
        }
    }

    // Test case 2: Rolling shutter with rotation
    {
        auto const world_point = glm::fvec3(1.0f, 1.0f, 5.0f);
        auto const resolution = std::array<uint32_t, 2>{640, 480};
        auto const pose_r_start =
            glm::fquat(1.0f, 0.0f, 0.0f, 0.0f); // Identity rotation
        auto const pose_t_start = glm::fvec3(0.0f, 0.0f, 0.0f);
        auto const pose_r_end =
            glm::fquat(0.707106781f, 0.0f, 0.707106781f, 0.0f); // 90° around y
        auto const pose_t_end = glm::fvec3(0.0f, 0.0f, 0.0f);

        // Simple projection function that just divides by z
        auto project_fn = [](const glm::fvec3 &p
                          ) -> std::pair<glm::fvec2, bool> {
            return {glm::fvec2(p.x / p.z, p.y / p.z), true};
        };

        auto const shutter_type = Type::ROLLING_TOP_TO_BOTTOM;

        auto const result = point_world_to_image(
            project_fn,
            resolution,
            world_point,
            pose_r_start,
            pose_t_start,
            pose_r_end,
            pose_t_end,
            shutter_type
        );

        if (!result.valid_flag) {
            printf("\n[FAIL] Test 2: Rolling shutter should be valid\n");
            fails += 1;
        } else {
            // Check if the result is reasonable
            auto const t = relative_frame_time(
                result.image_point, resolution, shutter_type
            );
            auto const &[pose_r_rs, pose_t_rs] = gsplat::se3::interpolate(
                t, pose_r_start, pose_t_start, pose_r_end, pose_t_end
            );

            if (!is_close(pose_r_rs, result.pose_r, 1e-4f) ||
                !is_close(pose_t_rs, result.pose_t, 1e-4f)) {
                printf("\n[FAIL] Test 2: Rolling shutter result incorrect\n");
                printf("  Expected r: %s\n", glm::to_string(pose_r_rs).c_str());
                printf("  Expected t: %s\n", glm::to_string(pose_t_rs).c_str());
                printf(
                    "  Result r: %s\n", glm::to_string(result.pose_r).c_str()
                );
                printf(
                    "  Result t: %s\n", glm::to_string(result.pose_t).c_str()
                );
                fails += 1;
            }
        }
    }

    return fails;
}

// Test point_world_to_image function with matrix rotation
auto test_point_world_to_image_mat() -> int {
    int fails = 0;

    // Test case 1: Global shutter
    {
        auto const world_point = glm::fvec3(1.0f, 1.0f, 5.0f);
        auto const resolution = std::array<uint32_t, 2>{640, 480};
        auto const pose_r_start = glm::fmat3(1.0f); // Identity rotation
        auto const pose_t_start = glm::fvec3(0.0f, 0.0f, 0.0f);
        auto const pose_r_end = glm::fmat3(1.0f); // Identity rotation
        auto const pose_t_end = glm::fvec3(0.0f, 0.0f, 0.0f);

        // Simple projection function that just divides by z
        auto project_fn = [](const glm::fvec3 &p
                          ) -> std::pair<glm::fvec2, bool> {
            return {glm::fvec2(p.x / p.z, p.y / p.z), true};
        };

        auto const result = point_world_to_image(
            project_fn,
            resolution,
            world_point,
            pose_r_start,
            pose_t_start,
            pose_r_end,
            pose_t_end,
            Type::GLOBAL
        );

        if (result.valid_flag) {
            printf("\n=== Testing point_world_to_image (matrix) ===\n");
            printf("\n[FAIL] Test 1: Global shutter should not be valid\n");
            fails += 1;
        }
    }

    // Test case 2: Rolling shutter with rotation
    {
        auto const world_point = glm::fvec3(1.0f, 1.0f, 5.0f);
        auto const resolution = std::array<uint32_t, 2>{640, 480};
        auto const pose_r_start = glm::fmat3(1.0f); // Identity rotation
        auto const pose_t_start = glm::fvec3(0.0f, 0.0f, 0.0f);
        // 90° rotation around y-axis
        auto const pose_r_end =
            glm::fmat3(0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        auto const pose_t_end = glm::fvec3(0.0f, 0.0f, 0.0f);

        // Simple projection function that just divides by z
        auto project_fn = [](const glm::fvec3 &p
                          ) -> std::pair<glm::fvec2, bool> {
            return {glm::fvec2(p.x / p.z, p.y / p.z), true};
        };
        auto const shutter_type = Type::ROLLING_LEFT_TO_RIGHT;

        auto const result = point_world_to_image(
            project_fn,
            resolution,
            world_point,
            pose_r_start,
            pose_t_start,
            pose_r_end,
            pose_t_end,
            shutter_type
        );

        if (!result.valid_flag) {
            printf("\n[FAIL] Test 2: Rolling shutter should be valid\n");
            fails += 1;
        } else {
            // Check if the result is reasonable
            auto const t = relative_frame_time(
                result.image_point, resolution, shutter_type
            );
            auto const &[pose_r_rs, pose_t_rs] = gsplat::se3::interpolate(
                t, pose_r_start, pose_t_start, pose_r_end, pose_t_end
            );

            if (!is_close(pose_r_rs, result.pose_r, 1e-4f) ||
                !is_close(pose_t_rs, result.pose_t, 1e-4f)) {
                printf("\n[FAIL] Test 2: Rolling shutter result incorrect\n");
                printf("  Expected r: %s\n", glm::to_string(pose_r_rs).c_str());
                printf("  Expected t: %s\n", glm::to_string(pose_t_rs).c_str());
                printf(
                    "  Result r: %s\n", glm::to_string(result.pose_r).c_str()
                );
                printf(
                    "  Result t: %s\n", glm::to_string(result.pose_t).c_str()
                );
                fails += 1;
            }
        }
    }

    return fails;
}

auto main() -> int {
    int fails = 0;

    fails += test_point_world_to_image_quat();
    fails += test_point_world_to_image_mat();

    if (fails > 0) {
        printf("\nTotal number of failures: %d\n", fails);
        return 1;
    }

    printf("\nAll tests passed!\n");
    return 0;
}