#pragma once

#include <torch/extension.h>

/**
 * Compute the fused 3D total variation loss for a 5D tensor.
 * 
 * Inputs:
 * @param volume Tensor shaped [B, C, D, H, W] on CUDA.
 *
 * Outputs:
 * @return Scalar tensor containing the accumulated TV3D penalty.
 */
torch::Tensor tv3d_forward(torch::Tensor volume);

/**
 * Compute gradients of the fused 3D TV loss w.r.t. the input tensor.
 * 
 * Inputs:
 * @param volume Tensor shaped [B, C, D, H, W] on CUDA.
 * @param grad_output Scalar gradient propagated from autograd.
 *
 * Outputs:
 * @return Tensor of shape [B, C, D, H, W] with gradients for each voxel.
 */
torch::Tensor tv3d_backward(torch::Tensor volume, torch::Tensor grad_output);
