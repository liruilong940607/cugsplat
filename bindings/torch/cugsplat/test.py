import torch
from _backend import _C

device = "cpu"

camera_points = torch.randn(1, 3, device=device)
focal_lengths = torch.randn(1, 2, device=device)
principal_points = torch.randn(1, 2, device=device)
image_points = _C.fisheye_project(camera_points, focal_lengths, principal_points)
print (image_points)