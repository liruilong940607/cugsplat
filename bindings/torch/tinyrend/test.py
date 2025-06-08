import torch

from _backend import _C

device = "cuda"

camera_points = torch.randn(1, 1, 3, device=device)
focal_lengths = torch.randn(1, 1, 2, device=device)
principal_points = torch.randn(1, 1, 2, device=device)
image_points = _C.fisheye_project(camera_points, focal_lengths, principal_points)
print (image_points)


n_images = 1
image_height = 28
image_width = 22
tile_width = 8
tile_height = 16
opacities = torch.tensor([0.5, 0.7], device=device)
isect_primitive_ids = torch.tensor([0, 1], device=device, dtype=torch.uint32)
isect_prefix_sum_per_tile = torch.tensor([2], device=device, dtype=torch.uint32)

render_alpha = _C.rasterize_simple_planer_forward(
    opacities, 
    n_images, image_height, image_width, tile_width, tile_height, 
    isect_primitive_ids, isect_prefix_sum_per_tile
)
assert render_alpha.shape == (n_images, image_height, image_width, 1)
assert (render_alpha[:, :tile_height, :tile_width, :] == 0.5 + (1 - 0.5) * 0.7).all()
