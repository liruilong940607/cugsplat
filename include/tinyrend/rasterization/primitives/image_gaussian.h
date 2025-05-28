#pragma once

#include <cstdint>
#include <glm/glm.hpp>

#include "tinyrend/core/macros.h" // for GSPLAT_HOST_DEVICE

namespace tinyrend::rasterization {

struct ImageGaussians {
    /*
    A collection of Gaussian primitives defined on a image plane.
    */

    // Pointers to the device memory (shared across all threads)
    glm::fvec2 *mu;     // [N, 2]
    glm::fvec3 *conics; // [N, 3]

    // Per-thread data
    uint32_t _image_id;
    uint32_t _pixel_x;
    uint32_t _pixel_y;
    void *_shmem_ptr;
    uint32_t _shmem_n_primitives;

    inline GSPLAT_HOST_DEVICE auto initialize(
        uint32_t image_id,
        uint32_t pixel_x,
        uint32_t pixel_y,
        void *shmem_ptr,
        uint32_t shmem_n_primitives
    ) -> bool {
        _image_id = image_id;
        _pixel_x = pixel_x;
        _pixel_y = pixel_y;
        _shmem_ptr = shmem_ptr;
        _shmem_n_primitives = shmem_n_primitives;
        return true;
    }

    inline GSPLAT_HOST_DEVICE auto
    load_to_shared_memory(uint32_t shmem_id, uint32_t global_id) -> void {
        glm::fvec2 *shmem_ptr_mu = reinterpret_cast<glm::fvec2 *>(_shmem_ptr);
        glm::fvec3 *shmem_ptr_conics =
            reinterpret_cast<glm::fvec3 *>(&shmem_ptr_mu[_shmem_n_primitives]);
        shmem_ptr_mu[shmem_id] = mu[global_id];
        shmem_ptr_conics[shmem_id] = conics[global_id];
    }

    inline GSPLAT_HOST_DEVICE auto get_light_attenuation(uint32_t shmem_id) -> float {
        glm::fvec2 *shmem_ptr_mu = reinterpret_cast<glm::fvec2 *>(_shmem_ptr);
        glm::fvec3 *shmem_ptr_conics =
            reinterpret_cast<glm::fvec3 *>(&shmem_ptr_mu[_shmem_n_primitives]);
        auto const mu = shmem_ptr_mu[shmem_id];
        auto const conic = shmem_ptr_conics[shmem_id];

        auto const dx = _pixel_x - mu.x;
        auto const dy = _pixel_y - mu.y;
        auto const sigma =
            0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
        return exp(-sigma);
    }
};

} // namespace tinyrend::rasterization