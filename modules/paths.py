import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir  # noqa: F401

import modules.safe  # noqa: F401


# data_path = cmd_opts_pre.data
sys.path.insert(0, script_path)

repositories_path = os.path.join(script_path, "repositories")

path_dirs = [
    (os.path.join(repositories_path, 'CodeFormer'), 'inference_codeformer.py', 'CodeFormer', []),
    (os.path.join(repositories_path, 'BLIP'), 'models/blip.py', 'BLIP', []),
    (os.path.join(repositories_path, 'k-diffusion'), 'k_diffusion/sampling.py', 'k_diffusion', ["atstart"]),
]

paths = {}

for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        if "atstart" in options:
            sys.path.insert(0, d)
        else:
            sys.path.append(d)
        paths[what] = d


class Prioritize:
    def __init__(self, name):
        self.name = name
        self.path = None

    def __enter__(self):
        self.path = sys.path.copy()
        sys.path = [paths[self.name]] + sys.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.path = self.path
        self.path = None
