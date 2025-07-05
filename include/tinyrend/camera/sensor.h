#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include "tinyrend/camera/impl/fisheye.h"
#include "tinyrend/camera/impl/orthogonal.h"
#include "tinyrend/camera/impl/pinhole.h"
#include "tinyrend/camera/impl/shutter.h"
#include "tinyrend/common/macros.h" // for TREND_HOST_DEVICE
#include "tinyrend/common/vec.h"

namespace tinyrend::camera {

template <typename Derived> struct BaseSensor {}

} // namespace tinyrend::camera
