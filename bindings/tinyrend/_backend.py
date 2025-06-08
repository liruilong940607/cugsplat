import os
import glob

from torch.utils.cpp_extension import load

def build_extension():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

    sources = [os.path.join(CURRENT_DIR, "bindings.cpp")] 
    sources += list(glob.glob(os.path.join(REPO_ROOT, "launcher", "tinyrend", "**/*.cu"), recursive=True))
    
    extra_include_paths = [
        os.path.join(REPO_ROOT, "include"),
        os.path.join(REPO_ROOT, "launcher"),
        os.path.join(REPO_ROOT, "third_party", "glm"),
    ]
    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "-use_fast_math", "--extended-lambda"]

    return load(
        name="tinyrend",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        verbose=True,
    )

_C = build_extension()