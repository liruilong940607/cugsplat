

namespace cugsplat {

#define DEFINE_POINTER_SETGET(TYPE, NAME, INDEX)                           \
    __host__ __device__ void set_##NAME(TYPE value) { this->NAME[INDEX] = value; }  \
    __host__ __device__ TYPE get_##NAME() const { return this->NAME[INDEX]; }

#define DEFINE_VALUE_SETGET(TYPE, NAME)                                    \
    __host__ __device__ void set_##NAME(TYPE value) { this->NAME = value; }         \
    __host__ __device__ TYPE get_##NAME() const { return this->NAME; }

#define DEFINE_DEVICE_POINTER_SETGET(TYPE, NAME, INDEX)                           \
    __device__ void set_##NAME(TYPE value) { this->NAME[INDEX] = value; }  \
    __device__ TYPE get_##NAME() const { return this->NAME[INDEX]; }

#define DEFINE_DEVICE_VALUE_SETGET(TYPE, NAME)                                    \
    __device__ void set_##NAME(TYPE value) { this->NAME = value; }         \
    __device__ TYPE get_##NAME() const { return this->NAME; }

#define DEFINE_HOST_POINTER_SETGET(TYPE, NAME, INDEX)                           \
    __host__ void set_##NAME(TYPE value) { this->NAME[INDEX] = value; }  \
    __host__ TYPE get_##NAME() const { return this->NAME[INDEX]; }

#define DEFINE_HOST_VALUE_SETGET(TYPE, NAME)                                    \
    __host__ void set_##NAME(TYPE value) { this->NAME = value; }         \
    __host__ TYPE get_##NAME() const { return this->NAME; }

} // namespace cugsplat