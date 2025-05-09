import torch
from _backend import _C

camera_points = torch.randn(1, 3)
focal_lengths = torch.randn(1, 2)
principal_points = torch.randn(1, 2)
image_points = _C.fisheye_project(camera_points, focal_lengths, principal_points)
print (image_points)
