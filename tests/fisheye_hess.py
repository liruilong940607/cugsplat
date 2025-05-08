
import torch

focal_length = torch.tensor([2.0, 3.0])

def fwd(image_point):
    invz = 1.0 / image_point[2]
    invz2 = invz * invz
    xy = image_point[:2] * invz
    r = torch.norm(xy)
    invr = 1.0 / r
    theta = torch.atan(r)
    s = theta * invr

    invr2 = invr * invr

    J_theta_r = 1.0 / (1.0 + r * r)
    J_theta_r2 = J_theta_r * J_theta_r
    tmp = (J_theta_r - s) * invr * invr
    J_uv_xy = s * torch.eye(2) + tmp * torch.outer(xy, xy)

    d_r_d_xy = xy * invr
    d_s_d_r = J_theta_r * invr - theta * invr2
    d_tmp_d_r = invr2 * (- 2.0 * J_theta_r2 * r - 3.0 * d_s_d_r)

    # d_tmp_d_r = (d_J_theta_r_d_r - d_s_d_r) * invr * invr + (J_theta_r - s) * -2.0 * invr * invr * invr
    # d_tmp_d_r = (d_J_theta_r_d_r - d_s_d_r) * invr2 + (J_theta_r - s) * -2.0 * invr3
    # d_tmp_d_r = invr2 * (- 2.0 * J_theta_r2 * r - 3.0 * d_s_d_r)

    # print ("d_tmp_d_r", d_tmp_d_r)

    d_J_uv_xy_d_x = d_s_d_r * d_r_d_xy[0] * torch.eye(2) + d_tmp_d_r * d_r_d_xy[0] * torch.outer(xy, xy) + tmp * torch.tensor([[2.0 * xy[0], xy[1]], [xy[1], 0.0]])
    d_J_uv_xy_d_y = d_s_d_r * d_r_d_xy[1] * torch.eye(2) + d_tmp_d_r * d_r_d_xy[1] * torch.outer(xy, xy) + tmp * torch.tensor([[0.0, xy[0]], [xy[0], 2.0 * xy[1]]])

    # print ("d_J_uv_xy_d_x", d_J_uv_xy_d_x)
    # print ("d_J_uv_xy_d_y", d_J_uv_xy_d_y)

    J_im_xy = torch.tensor([[focal_length[0], 0.0], [0.0, focal_length[1]]]) @ J_uv_xy
    zero = torch.zeros(())
    J_xy_cam = torch.stack([
        torch.stack([invz, zero, -xy[0] * invz]),
        torch.stack([zero, invz, -xy[1] * invz])
    ])
    J = J_im_xy @ J_xy_cam

    d_J_im_xy_d_x = torch.tensor([[focal_length[0], 0.0], [0.0, focal_length[1]]]) @  d_J_uv_xy_d_x
    d_J_im_xy_d_y = torch.tensor([[focal_length[0], 0.0], [0.0, focal_length[1]]]) @  d_J_uv_xy_d_y

    # print ("d_J_im_xy_d_x", d_J_im_xy_d_x)
    # print ("d_J_im_xy_d_y", d_J_im_xy_d_y)

    d_J_xy_cam_d_x = torch.tensor([[0.0, 0.0, -invz], [0.0, 0.0, 0.0]])
    d_J_xy_cam_d_y = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -invz]])
    d_J_xy_cam_d_z_direct = torch.tensor([[-invz2, 0.0, xy[0] * invz2], [0.0, -invz2, xy[1] * invz2]])

    # print ("d_J_xy_cam_d_x", d_J_xy_cam_d_x)
    # print ("d_J_xy_cam_d_y", d_J_xy_cam_d_y)

    d_J_d_x = d_J_im_xy_d_x @ J_xy_cam + J_im_xy @ d_J_xy_cam_d_x
    d_J_d_y = d_J_im_xy_d_y @ J_xy_cam + J_im_xy @ d_J_xy_cam_d_y
    
    print ("d_J_d_x", d_J_d_x)
    print ("d_J_d_y", d_J_d_y)

    # _d_J_d_x = invz * torch.tensor([
    #     d_J_im_xy_d_x[0][0], d_J_im_xy_d_x[1][0],
    #     d_J_im_xy_d_x[0][1], d_J_im_xy_d_x[1][1],
    #     -d_J_im_xy_d_x[0][0] * xy[0] - d_J_im_xy_d_x[0][1] * xy[1] - J_im_xy[0][0],
    #     -d_J_im_xy_d_x[1][0] * xy[0] - d_J_im_xy_d_x[1][1] * xy[1] - J_im_xy[1][0]
    # ]).reshape(3, 2).t()
    # _d_J_d_y = invz * torch.tensor([
    #     d_J_im_xy_d_y[0][0], d_J_im_xy_d_y[1][0],
    #     d_J_im_xy_d_y[0][1], d_J_im_xy_d_y[1][1],
    #     -d_J_im_xy_d_y[0][0] * xy[0] - d_J_im_xy_d_y[0][1] * xy[1] - J_im_xy[0][1],
    #     -d_J_im_xy_d_y[1][0] * xy[0] - d_J_im_xy_d_y[1][1] * xy[1] - J_im_xy[1][1]
    # ]).reshape(3, 2).t()
    # print ("_d_J_d_x", _d_J_d_x)
    # print ("_d_J_d_y", _d_J_d_y)


    d_J_d_cam_x = d_J_d_x * invz
    d_J_d_cam_y = d_J_d_y * invz
    d_J_d_cam_z = -d_J_d_x * xy[0] * invz - d_J_d_y * xy[1] * invz + J_im_xy @ d_J_xy_cam_d_z_direct

    # print ("d_J_d_cam_x", d_J_d_cam_x)
    # print ("d_J_d_cam_y", d_J_d_cam_y)
    # print ("d_J_d_cam_z", d_J_d_cam_z)

    H1 = torch.stack([
        d_J_d_cam_x[0][0], d_J_d_cam_x[0][1], d_J_d_cam_x[0][2],
        d_J_d_cam_y[0][0], d_J_d_cam_y[0][1], d_J_d_cam_y[0][2],
        d_J_d_cam_z[0][0], d_J_d_cam_z[0][1], d_J_d_cam_z[0][2]
    ]).reshape(3, 3)

    H2 = torch.stack([
        d_J_d_cam_x[1][0], d_J_d_cam_x[1][1], d_J_d_cam_x[1][2],
        d_J_d_cam_y[1][0], d_J_d_cam_y[1][1], d_J_d_cam_y[1][2],
        d_J_d_cam_z[1][0], d_J_d_cam_z[1][1], d_J_d_cam_z[1][2]
    ]).reshape(3, 3)
    
    print ("H1", H1)
    print ("H2", H2)

    return J

image_point = torch.rand(3)

# use torch to compute jacobian
jac = torch.autograd.functional.jacobian(fwd, image_point)
# print("jac_x", jac[..., 0])
# print("jac_y", jac[..., 1])
print ("jac", jac[0], jac[1])

# manually compute jacobian
