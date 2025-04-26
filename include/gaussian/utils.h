inline GSPLAT_HOST_DEVICE auto quat_scale_to_scaled_rotmat(
    glm::fvec4 const &quat, glm::fvec3 const &scale
) -> glm::fmat3 {
    auto const R = quat_to_rotmat(quat);
    auto const M =
        glm::fmat3(R[0] * scale[0], R[1] * scale[1], R[2] * scale[2]);
    return M;
}

inline GSPLAT_HOST_DEVICE auto quat_scale_to_scaled_rotmat_vjp(
    // inputs
    glm::fvec4 const &quat,
    glm::fvec3 const &scale,
    // output gradients
    glm::fmat3 const &v_M
) -> std::pair<glm::fvec4, glm::fvec3> {
    auto const R = quat_to_rotmat(quat);
    // Scale each column of v_M by its corresponding scale component
    auto const v_R = glm::fmat3(
        v_M[0] * scale[0],
        v_M[1] * scale[1],
        v_M[2] * scale[2]
    );
    auto const v_quat = quat_to_rotmat_vjp(quat, v_R);
    auto const v_scale = glm::fvec3{
        glm::dot(v_M[0], R[0]), glm::dot(v_M[1], R[1]), glm::dot(v_M[2], R[2])
    };
    return {v_quat, v_scale};
}

inline GSPLAT_HOST_DEVICE auto quat_scale_to_covar_vjp(
    // inputs
    glm::fvec4 const &quat,
    glm::fvec3 const &scale,
    // output gradients
    glm::fmat3 const &v_covar
) -> std::pair<glm::fvec4, glm::fvec3> {
    auto const R = quat_to_rotmat(quat);
    auto const M =
        glm::fmat3(R[0] * scale[0], R[1] * scale[1], R[2] * scale[2]);

    auto const v_M = (v_covar + glm::transpose(v_covar)) * M;
    auto const v_R =
        glm::fmat3(v_M[0] * scale[0], v_M[1] * scale[1], v_M[2] * scale[2]);

    auto const v_quat = quat_to_rotmat_vjp(quat, v_R);
    auto const v_scale = glm::fvec3{
        glm::dot(v_M[0], R[0]), glm::dot(v_M[1], R[1]), glm::dot(v_M[2], R[2])
    };
    return {v_quat, v_scale};
} 