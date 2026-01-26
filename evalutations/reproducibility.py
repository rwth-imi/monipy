
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Sets global seed and enforces deterministic behavior in PyTorch, NumPy, random, and environment.

    Args:
        seed (int): Seed value to set.
        deterministic (bool): Whether to enforce full determinism. May reduce speed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

