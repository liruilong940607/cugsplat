#pragma once

#include <stdint.h>
#include <glm/glm.hpp>

#include "utils/macros.h"
#include "utils/types.h"
#include "utils/gaussian.h"

#include "device/common.cuh"

namespace gsplat::device {    

    
struct DeviceOperator3DGSIn3D {

    template <class DeviceCameraModel>
    inline GSPLAT_HOST_DEVICE float image_depth(DeviceCameraModel &d_camera) {
        auto const world_point = this->get_mean();
        auto const camera_point = d_camera.point_world_to_camera(world_point);
        return camera_point.z;
    }

    template <class DeviceCameraModel>
    inline GSPLAT_HOST_DEVICE auto world_to_image(
        DeviceCameraModel &d_camera
    ) -> std::tuple<glm::fvec2, glm::fmat2, bool> {
        auto const world_point = this->get_mean();
        auto const world_covar = quat_scale_to_covar(
            this->get_quat(), this->get_scale());
        auto const image_point = d_camera.point_world_to_image(world_point);
        auto const J = d_camera.jacobian_world_to_image(world_point);
        auto const image_covar = J * world_covar * transpose(J);
        return {image_point, image_covar, true};
    }
};

struct  DevicePrimitiveInWorld3DGS: DevicePrimitive3DGSIn3D, DeviceOperator3DGSIn3D 
{
    using DevicePrimitive3DGSIn3D::DevicePrimitive3DGSIn3D;
};

} // namespace gsplat::device

