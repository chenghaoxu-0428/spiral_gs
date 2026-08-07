# Fused 3D total variation

CUDA implementation of a fused 3D total variation loss. Implemented as a part of the paper:
> FaCT-GS: Fast and Scalable CT Reconstruction with Gaussian Splatting

### [Main Repository](https://github.com/PaPieta/fact-gs) | [Paper](TBA) | [Project Page](TBA)

#### Related repositories (applied in the paper):
[Fast Gaussian Splatting Voxelizer](https://github.com/PaPieta/gs-voxelizer) | [Fast CT Rasterizer](https://github.com/PaPieta/gs-ct-rasterizer) | [Fused SSIM](https://github.com/rahul-goel/fused-ssim) (2D and 3D) 

## Prerequirements

1. You must have PyTorch installed with CUDA backend, and an NVIDIA GPU

## Installation
```bash
pip install . --no-build-isolation
```
## Usage
```python
import torch
from fused_3d_tv import tv3d_loss

tensor = torch.randn(2, 1, 16, 32, 32, device="cuda", requires_grad=True)
loss = tv3d_loss(tensor)  # scalar sum of absolute diffs along D/H/W
loss.backward()
```

Each kernel stage pulls an `8x8x8` neighborhood into shared memory and performs all operations (forward accumulation or backward gradients).


## Reference implementation
For validation, the package ships with tests that compare against the Python equivalent below:
```python
def reference_tv3d(vol: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(torch.diff(vol, dim=2))
    dy = torch.abs(torch.diff(vol, dim=3))
    dz = torch.abs(torch.diff(vol, dim=4))
    return dx.sum() + dy.sum() + dz.sum()
```
Run `pytest -q` to verify the kernels against this baseline.

## Acknowledgements

Inspired by [Fused SSIM](https://github.com/rahul-goel/fused-ssim).
