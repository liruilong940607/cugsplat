#include <algorithm>
#include <cstdint>

#include "../helpers.h"
#include "tinyrend/camera/model.h"

using namespace tinyrend;
using namespace tinyrend::camera;

int test_perfect_pinhole() {
    int fails = 0;

    // Test case 1: Perfect pinhole camera model
    {
        ShutterPoses shutter_poses;
        shutter_poses.start = ShutterPoses::Pose{fvec3::zero(), fquat::identity()};
        shutter_poses.end = ShutterPoses::Pose{fvec3::ones(), fquat::identity()};

        auto parameters = PerfectPinholeCameraModelImpl::Parameters{};
        parameters.resolution = {640, 480};
        parameters.shutter_type =
            tinyrend::camera::impl::shutter::Type::ROLLING_TOP_TO_BOTTOM;
        parameters.margin_factor = 0.0f;
        parameters.focal_length = fvec2(100.0f, 100.0f);
        parameters.principal_point = fvec2(320.0f, 240.0f);
        auto model = PerfectPinholeCameraModelImpl(parameters);

        auto const world_point = fvec3(1.0f, 1.0f, 1.0f);
        auto const image_point =
            model.world_point_to_image_point(world_point, shutter_poses);
        // printf("image_point_p: %s\n", image_point.p.to_string().c_str());
        // printf("image_point_z: %f\n", image_point.z);
        // printf("image_point_valid_flag: %d\n", image_point.valid_flag);

        auto const world_ray =
            model.image_point_to_world_ray(image_point.p, shutter_poses);
        // printf("world_ray_o: %s\n", world_ray.o.to_string().c_str());
        // printf("world_ray_d: %s\n", world_ray.d.to_string().c_str());
        // printf("world_ray_valid_flag: %d\n", world_ray.valid_flag);

        auto const t_dist = length(world_point - world_ray.o);
        auto const world_point_reproj = world_ray.o + world_ray.d * t_dist;
        // printf("world_point_reproj: %s\n", world_point_reproj.to_string().c_str());

        fails += CHECK(image_point.valid_flag && world_ray.valid_flag, "");
        fails += CHECK(is_close(world_point_reproj, world_point), "");
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_pinhole();

    if (fails > 0) {
        printf("[camera/model.cpp] %d tests failed!\n", fails);
    } else {
        printf("[camera/model.cpp] All tests passed!\n");
    }

    return fails;
}