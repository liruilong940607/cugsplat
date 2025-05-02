#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

namespace pybind11 {
namespace detail {

// Primary template (unspecialized)
template <typename T> struct glm_vec_caster;

template <typename T> struct glm_mat_caster;

// ---------- Vector Caster ----------
template <glm::length_t L>
struct glm_vec_caster<glm::vec<L, float, glm::defaultp>> {
    using glm_type = glm::vec<L, float, glm::defaultp>;

    static constexpr auto type_name() {
        if constexpr (L == 2)
            return _("Tuple[float, float]");
        else if constexpr (L == 3)
            return _("Tuple[float, float, float]");
        else if constexpr (L == 4)
            return _("Tuple[float, float, float, float]");
        else if constexpr (L == 5)
            return _("Tuple[float, float, float, float, float]");
        else if constexpr (L == 6)
            return _("Tuple[float, float, float, float, float, float]");
        else
            return _("Tuple[float, ...]");
    }

  public:
    PYBIND11_TYPE_CASTER(glm_type, type_name());

    bool load(handle src, bool) {
        if (!isinstance<sequence>(src))
            return false;
        auto seq = reinterpret_borrow<sequence>(src);
        if (seq.size() != L)
            return false;
        for (size_t i = 0; i < L; ++i)
            value[i] = seq[i].cast<float>();
        return true;
    }

    static handle cast(glm_type const &src, return_value_policy, handle) {
        tuple t(L);
        for (size_t i = 0; i < L; ++i)
            t[i] = py::cast(src[i]);
        return t.release();
    }
};

template <> struct type_caster<glm::fvec2> : glm_vec_caster<glm::fvec2> {};
template <> struct type_caster<glm::fvec3> : glm_vec_caster<glm::fvec3> {};
template <> struct type_caster<glm::fvec4> : glm_vec_caster<glm::fvec4> {};

// ---------- Matrix Caster ----------
template <glm::length_t C, glm::length_t R>
struct glm_mat_caster<glm::mat<C, R, float, glm::defaultp>> {
    using glm_type = glm::mat<C, R, float, glm::defaultp>;

    // glm::fmat3x2 (C=3, R=2) is 3 columns x 2 rows (2x3 in python)
    static constexpr auto type_name() {
        if constexpr (C == 2 && R == 2)
            return _("List[List[float]] 2x2");
        else if constexpr (C == 3 && R == 3)
            return _("List[List[float]] 3x3");
        else if constexpr (C == 4 && R == 4)
            return _("List[List[float]] 4x4");
        else if constexpr (C == 2 && R == 3)
            return _("List[List[float]] 3x2");
        else if constexpr (C == 3 && R == 2)
            return _("List[List[float]] 2x3");
        else
            return _("List[List[float]] CxR");
    }

  public:
    PYBIND11_TYPE_CASTER(glm_type, type_name());

    bool load(handle src, bool) {
        if (!isinstance<sequence>(src))
            return false;
        auto outer = reinterpret_borrow<sequence>(src);
        if (outer.size() != R)
            return false; // row count

        for (size_t i = 0; i < R; ++i) {
            auto inner_any = outer[i];
            if (!isinstance<sequence>(inner_any))
                return false;
            auto inner = reinterpret_borrow<sequence>(inner_any);
            if (inner.size() != C)
                return false; // column count

            for (size_t j = 0; j < C; ++j) {
                // Python: row i, col j → GLM: col j, row i
                value[j][i] = inner[j].cast<float>();
            }
        }

        return true;
    }

    static handle cast(glm_type const &src, return_value_policy, handle) {
        tuple rows(R);
        for (size_t i = 0; i < R; ++i) {
            tuple row(C);
            for (size_t j = 0; j < C; ++j) {
                row[j] = py::cast(src[j][i]);
            }
            rows[i] = std::move(row);
        }
        return rows.release();
    }
};

template <> struct type_caster<glm::fmat2> : glm_mat_caster<glm::fmat2> {};
template <> struct type_caster<glm::fmat3> : glm_mat_caster<glm::fmat3> {};
template <> struct type_caster<glm::fmat4> : glm_mat_caster<glm::fmat4> {};
template <> struct type_caster<glm::fmat2x3> : glm_mat_caster<glm::fmat2x3> {};
template <> struct type_caster<glm::fmat3x2> : glm_mat_caster<glm::fmat3x2> {};

// -- quat --
template <> struct type_caster<glm::fquat> {
  public:
    PYBIND11_TYPE_CASTER(
        glm::fquat, _("Tuple[float, float, float, float]")
    ); // (w, x, y, z)

    bool load(handle src, bool) {
        if (!isinstance<sequence>(src))
            return false;
        auto seq = reinterpret_borrow<sequence>(src);
        if (seq.size() != 4)
            return false;
        value = glm::fquat(
            seq[0].cast<float>(), // w
            seq[1].cast<float>(), // x
            seq[2].cast<float>(), // y
            seq[3].cast<float>()  // z
        );
        return true;
    }

    static handle cast(glm::fquat const &src, return_value_policy, handle) {
        return py::make_tuple(src.w, src.x, src.y, src.z).release();
    }
};

} // namespace detail
} // namespace pybind11