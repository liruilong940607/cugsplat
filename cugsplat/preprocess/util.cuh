#pragma once

#include <glm/glm.hpp>

namespace cugsplat::preprocess {

template <typename T>
struct Maybe {
    bool _has_value = false;
    T _value;

    __device__ inline T get() const {
        return this->_has_value ? this->_value : T{};
    }

    __device__ inline bool has_value() const {
        return this->_has_value;
    }

    __device__ inline void set(const T& v) {
        this->_value = v;
        this->_has_value = true;
    }
};

// Solve the tight axis-aligned bounding box radius for a Gaussian defined as
//      sigma = 1/2 * xᵀ * covar⁻¹ * x
//      y = prefactor * exp(-sigma)
// at the given y value.
inline __device__ auto solve_tight_radius(
    glm::fmat2 covar, float prefactor, float y = 1.0f / 255.0f
) -> glm::fvec2 {
    if (prefactor < y) {
        return glm::fvec2(0.0f);
    }

    // Threshold distance squared on ellipse
    float sigma = -logf(y / prefactor);
    float Q = 2.0f * sigma;

    // Eigenvalues of covariance matrix
    float det = determinant(covar);
    float half_trace = 0.5f * (covar[0][0] + covar[1][1]);
    float discrim = sqrtf(max(0.f, half_trace * half_trace - det));
    float lambda1 = half_trace + discrim;
    float lambda2 = half_trace - discrim;

    // Compute unit eigenvectors
    glm::fvec2 v1, v2;
    if (covar[0][1] == 0.0f) {
        // pick the axis that corresponds to the larger eigenvalue
        if (covar[0][0] >= covar[1][1]) {
            v1 = glm::fvec2(1,0);
        } else {
            v1 = glm::fvec2(0,1);
        }
    } else {
        v1 = glm::normalize(glm::fvec2(lambda1 - covar[1][1], covar[0][1]));
    }
    v2 = glm::fvec2(-v1.y, v1.x);  // perpendicular

    // Scale eigenvectors with eigenvalues
    v1 *= sqrtf(Q * lambda1);
    v2 *= sqrtf(Q * lambda2);

    // Compute max extent along world x/y axes (bounding box)
    auto const radius = glm::sqrt(v1 * v1 + v2 * v2);
    return radius;
}


} // namespace cugsplat::preprocess