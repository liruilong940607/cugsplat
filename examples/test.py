import os
import glob

import torch
from torch.utils.cpp_extension import load

def build_extension():
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sources = [
        os.path.join(REPO_ROOT, "examples", "bindings.cpp")
    ] + list(glob.glob(os.path.join(REPO_ROOT, "include", "cugsplat", "kernels", "*.cu")))
    extra_include_paths = [
        os.path.join(REPO_ROOT, "include"),
        os.path.join(REPO_ROOT, "third_party", "glm"),
    ]
    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "-use_fast_math", "--extended-lambda"]

    return load(
        name="cugsplat",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        verbose=True,
    )

_C = build_extension()

camera_points = torch.randn(1, 3)
focal_lengths = torch.randn(1, 2)
principal_points = torch.randn(1, 2)
image_points = _C.fisheye_project(camera_points, focal_lengths, principal_points)
print (image_points)
