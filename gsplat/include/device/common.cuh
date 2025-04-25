#pragma once

#include <cstdint>
#include "utils/macros.h"
#include "utils/types.h"

namespace gsplat {

#define GSPLAT_DECL_CACHE(fname, ftype) ftype fname;
#define GSPLAT_DECL_GETTER(fname, ftype)                                   \
    GSPLAT_HOST_DEVICE ftype get_##fname() {                               \
        return this->fname;                                                \
    }
#define GSPLAT_DEFINE_PRIMITIVE(NAME, FIELD_LIST)                          \
struct NAME {                                                              \
    FIELD_LIST(GSPLAT_DECL_CACHE)                                          \
    FIELD_LIST(GSPLAT_DECL_GETTER)                                                                                          \
};

#define GSPLAT_DECL_DEIVCE_PTR(fname, ftype)   ftype* fname##_ptr{};
#define GSPLAT_DECL_DEVICE_CACHE(fname, ftype) Maybe<ftype> fname;
#define GSPLAT_DECL_DEVICE_GETTER(fname, ftype)                            \
    GSPLAT_HOST_DEVICE ftype get_##fname() {                               \
        if (!fname.has_value()) fname.set(fname##_ptr[idx]);               \
        return fname.get();                                                \
    }
#define GSPLAT_DEVICE_CTOR_PARAM(fname, ftype) ftype* fname##_ptr,
#define GSPLAT_DEVICE_CTOR_INIT(fname, ftype) this->fname##_ptr = fname##_ptr;
#define GSPLAT_DEFINE_DEVICE_PRIMITIVE(NAME, FIELD_LIST)                   \
struct NAME {                                                              \
    uint32_t n{0}, idx{0};                                                 \
    GSPLAT_HOST_DEVICE void set_index(uint32_t i) { idx = i; }             \
    GSPLAT_HOST_DEVICE int  get_n() const { return n; }                    \
    /*  (a) pointers, (b) caches, (c) getters  */                          \
    FIELD_LIST(GSPLAT_DECL_DEIVCE_PTR)                                     \
    FIELD_LIST(GSPLAT_DECL_DEVICE_CACHE)                                   \
    FIELD_LIST(GSPLAT_DECL_DEVICE_GETTER)                                  \
    /* (d) constructor: assign all pointer fields */                       \
    GSPLAT_HOST_DEVICE NAME(                                               \
        FIELD_LIST(GSPLAT_DEVICE_CTOR_PARAM) uint32_t n                    \
    ) {                                                                    \
        this->n = n;                                                       \
        FIELD_LIST(GSPLAT_DEVICE_CTOR_INIT)                                \
    }                                                                      \
    /* (e) default constructor */                                          \
    GSPLAT_HOST_DEVICE NAME() {}                                           \
};

#define FIELDS_3DGS_IN_3D(X)                      \
    X(opacity , float      )                      \
    X(mean    , glm::fvec3 )                      \
    X(quat    , glm::fvec4 )                      \
    X(scale   , glm::fvec3 )
GSPLAT_DEFINE_DEVICE_PRIMITIVE(Primitive3DGSIn3D, FIELDS_3DGS_IN_3D)
GSPLAT_DEFINE_DEVICE_PRIMITIVE(DevicePrimitive3DGSIn3D, FIELDS_3DGS_IN_3D)

#define FIELDS_2DGS_IN_3D(X)                      \
    X(opacity , float      )                      \
    X(mean    , glm::fvec3 )                      \
    X(quat    , glm::fvec4 )                      \
    X(scale   , glm::fvec2 )
GSPLAT_DEFINE_DEVICE_PRIMITIVE(Primitive2DGSIn3D, FIELDS_2DGS_IN_3D)
GSPLAT_DEFINE_DEVICE_PRIMITIVE(DevicePrimitive2DGSIn3D, FIELDS_2DGS_IN_3D)

#define FIELDS_2DGS_IN_2D(X)                      \
    X(opacity , float      )                      \
    X(mean    , glm::fvec2 )                      \
    X(covar   , glm::fmat2 )
GSPLAT_DEFINE_DEVICE_PRIMITIVE(Primitive2DGSIn2D, FIELDS_2DGS_IN_2D)
GSPLAT_DEFINE_DEVICE_PRIMITIVE(DevicePrimitive2DGSIn2D, FIELDS_2DGS_IN_2D)

#undef GSPLAT_DECL_CACHE
#undef GSPLAT_DECL_GETTER
#undef GSPLAT_DEFINE_PRIMITIVE
#undef GSPLAT_DECL_DEIVCE_PTR
#undef GSPLAT_DECL_DEVICE_CACHE
#undef GSPLAT_DECL_DEVICE_GETTER
#undef GSPLAT_DEVICE_CTOR_PARAM
#undef GSPLAT_DEVICE_CTOR_INIT
#undef GSPLAT_DEFINE_DEVICE_PRIMITIVE

} // namespace gsplat
