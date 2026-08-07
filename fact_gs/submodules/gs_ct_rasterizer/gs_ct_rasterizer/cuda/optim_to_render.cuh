#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ALPHA_THRESHOLD (1/0.00001f)

/**
 * Converts gaussian parameters from 3D optimization format to 2D rasterization format.
 * Position from [0,1] to pixel-based, scale and quat to conics
 * Prepares radii for assigning gaussians to volume tiles.
 * Each thread processes one gaussian.
 * 
 * Template parameter:
 * @tparam CHANNELS Number of channels in the input intensities.
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param pos3d 3D positions in optimization format [0-1] range.
 * @param scale3d 3D scale parameters.
 * @param quat Quaternions representing rotation (w, x, y, z order).
 * @param intensities Per-gaussian intensity/opactiy values used to estimate screen-space extent.
 * @param viewmatrix View matrix.
 * @param projmatrix Projection matrix.
 * @param focal_x,focal_y Focal length in x and y.
 * @param tan_fovx,tan_fovy Tangents of the field of view angles in x and y.
 * @param image_size Image dimensions in pixels.
 * @param grid Grid dimensions for volume tiles.
 * @param mode 0: parallel beam, 1: cone beam
 * 
 * @return Outputs:
 * @param pos2d_out 2D positions in pixel coordinates.
 * @param conics_mu_out Conic parameters a, b, c for each gaussian, and mu (integration factor).
 * @param radii_out Radii for each gaussian for tile assignment, set to 0 if out of frustum.
 * @param num_tiles_hit_out Number of tiles hit by each gaussian.
 * @param tile_min_out Minimum tile indices (x,y) intersected by each gaussian.
 * @param tile_max_out Maximum tile indices (x,y) intersected by each gaussian.
 */
template <int CHANNELS> __global__ void optim_to_render_forward(
    const int num_gaussians,
    const float3* __restrict__ pos3d,
    const float3* __restrict__ scale3d,
    const float4* __restrict__ quat,
    const float* __restrict__ intensities,
    const float* viewmatrix,
	const float* projmatrix,
    const float focal_x, float focal_y,
    const float tan_fovx, const float tan_fovy,
    const uint2 image_size,
	const dim3 grid,
    const int mode, // 0: parallel beam, 1: cone beam
    float2* __restrict__ pos2d_out,
    float4* __restrict__ conics_mu_out,
    float* __restrict__ radii_out,
	int32_t* __restrict__ num_tiles_hit_out,
	int2* __restrict__ tile_min_out,
	int2* __restrict__ tile_max_out
);


/**
 * Backward pass for optim_to_render_forward. 
 * Computes gradients w.r.t. input parameters (pos3d, scale3d, quat).
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param pos3d 3D positions in optimization format [0-1] range.
 * @param scale3d 3D scale parameters.
 * @param quat Quaternions representing rotation (w, x, y, z order).
 * @param viewmatrix View matrix.
 * @param projmatrix Projection matrix.
 * @param focal_x,focal_y Focal length in x and y.
 * @param tan_fovx,tan_fovy Tangents of the field of view angles in x and y.
 * @param mode 0: parallel beam, 1: cone beam
 * @param radii Radii for each gaussian for tile assignment
 * @param pos2d_grad_in Incoming gradient w.r.t. output 2D positions.
 * @param conics_mu_grad_in Incoming gradient w.r.t. output conic parameters and mu.
 * 
 * @return Outputs:
 * @param pos3d_grad_out Gradient w.r.t. input positions.
 * @param scale3d_grad_out Gradient w.r.t. input scales.
 * @param quat_grad_out Gradient w.r.t. input quaternions.
 */
__global__ void optim_to_render_backward(
	const int num_gaussians,
	const float3* __restrict__ pos3d,
	const float3* __restrict__ scale3d,
	const float4* __restrict__ quat,
	const float* viewmatrix,
	const float* projmatrix,
    const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const int mode, // 0: parallel beam, 1: cone beam
	const float* radii,
	const float2* __restrict__ pos2d_grad_in,
	const float4* __restrict__ conics_mu_grad_in,
	float3* __restrict__ pos3d_grad_out,
	float3* __restrict__ scale3d_grad_out,
	float4* __restrict__ quat_grad_out
);
