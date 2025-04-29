#pragma once

namespace gsplat {

#ifdef __CUDACC__
#define GSPLAT_HOST_DEVICE __host__ __device__
#else
#define GSPLAT_HOST_DEVICE
#endif

#define GET_FIELD(field_name)                                                  \
    GSPLAT_HOST_DEVICE decltype(field_name.get()) get_##field_name() const {   \
        return field_name.get();                                               \
    }

#define GET_FIELD_FROM_PTR(field_name)                                         \
    GSPLAT_HOST_DEVICE decltype(field_name.get()) get_##field_name() {         \
        if (!field_name.has_value()) {                                         \
            field_name.set(field_name##_ptr[idx]);                             \
        }                                                                      \
        return field_name.get();                                               \
    }

} // namespace gsplat