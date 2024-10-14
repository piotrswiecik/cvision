import torch

from typing import Literal



def get_device() -> Literal["cpu", "cuda", "mps"]:
    """Get the available GPU device or set CPU as default.
    Note that MPS usage is not recommended as of Oct'24 due to PyTorch support issues 
    for various nn module components.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"