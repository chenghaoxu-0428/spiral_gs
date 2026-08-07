#pragma once

#include <tuple>
#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)


/**
 * optim_to_render_forward kernel binding. Kernel description:
 * Converts gaussian parameters from 3D optimization format to 2D rasterization format.
 * Position from [0,1] to pixel-based, scale and quat to conics
 * Prepares radii for assigning gaussians to volume tiles.
 * Each thread processes one gaussian.
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param pos3d 3D positions in optimization format [0-1] range.
 * @param scale3d 3D scale parameters.
 * @param quat Quaternions representing rotation (w, x, y, z order).
 * @param intensities Per-gaussian intensity/opactiy values used when estimating extents.
 * @param viewmatrix Pointer to view matrix.
 * @param projmatrix Pointer to projection matrix.
 * @param tan_fovx, tan_fovy Tangent of the field of view angles in x and y directions.
 * @param image_height, image_width Height and width of the output image in pixels.
 * @param mode Mode of operation (0: parallel beam, 1: cone beam).
 * 
 * @return Outputs:
 * 2D vectors storing visualization positions (xy)
 * Conic parameters a, b, c, and mu (integration factor) for each gaussian.
 * Minimum enclosing radii of each gaussian for image tile assignment.
 * Minimum...
 * and maximum tile indices,
 * Number of tiles hit per gaussian.
*/
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
optim_to_render_forward_torch(
    const int num_gaussians,
    const torch::Tensor &pos3d,
    const torch::Tensor &scale3d,
    const torch::Tensor &quat,
    const torch::Tensor &intensities,
    const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
    const float tan_fovx, const float tan_fovy,
    const unsigned image_height,
    const unsigned image_width,
    const int mode
);


/**
 * optim_to_render_backward kernel binding. Kernel description:
 * Backward pass for optim_to_render_forward. 
 * Computes gradients w.r.t. input parameters (pos3d, scale3d, quat).
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param pos3d 3D positions in optimization format [0-1] range.
 * @param scale3d 3D scale parameters.
 * @param quat Quaternions representing rotation (w, x, y, z order).
 * @param viewmatrix Pointer to view matrix.
 * @param projmatrix Pointer to projection matrix.
 * @param tan_fovx, tan_fovy Tangent of the field of view angles in x and y directions.
 * @param image_height, image_width Height and width of the output image in pixels.
 * @param mode Mode of operation (0: parallel beam, 1: cone beam).
 * @param radii Minimum enclosing radii of each gaussian for image tile assignment (from forward pass).
 * @param pos2d_grad_in Gradient of the loss w.r.t. the 2D positions.
 * @param conics_mu_grad_in Gradient of the loss w.r.t. the conic parameters.
 * 
 * @return Outputs:
 * Gradient w.r.t. input position.
 * Gradient w.r.t. input scale.
 * Gradient w.r.t. input quaternion.
 */
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
optim_to_render_backward_torch(
    const int num_gaussians,
    const torch::Tensor &pos3d,
    const torch::Tensor &scale3d,
    const torch::Tensor &quat,
    const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
    const float tan_fovx, const float tan_fovy,
    const unsigned image_height,
    const unsigned image_width,
    const int mode,
    const torch::Tensor &radii,
    const torch::Tensor &pos2d_grad_in,
    const torch::Tensor &conics_mu_grad_in
);

/**
 * Combined binding that maps gaussians to tiles, sorts intersections
 * with a radix sort, and builds tile bins.
 * 
 * Inputs:
 * @param num_gaussians Number of gaussians.
 * @param num_intersects Number of total intersections.
 * @param tile_min Minimum tile indices (x,y) each gaussian intersects.
 * @param tile_max Maximum tile indices (x,y) each gaussian intersects.
 * @param cum_tiles_hit Cumulative count of tiles hit by gaussians.
 * @param img_size Dimensions of the image (height, width).
 * 
 * @return Outputs:
 * Sorted gaussian IDs assigned to consecutive tiles.
 * Tile bins storing start/end offsets of gaussians intersecting each tile.
 */
std::tuple<
    torch::Tensor,
    torch::Tensor> bin_and_sort_gaussians_torch(
    const int num_gaussians,
    const int num_intersects,
    const torch::Tensor &tile_min,
    const torch::Tensor &tile_max,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int> img_size
);



/**
 * rasterize_forward kernel binding. Kernel description:
 * Forward rasterization pass of a set of 2D Gaussians into an image grid.
 * Each thread processes one pixel in the image. Block size defined in config.h.
 * 
 * Inputs:
 * @param image_size Dimensions of the image (height, width).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections.
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos2d 4D vectors storing gaussian positions (xyz) and unused w component.
 * @param conics_mu Conic parameters a, b, c for each gaussian, and mu (integration factor).
 * @param intensities Intensity values for each gaussian.
 * 
 * @return Outputs:
 * Rasterized output image (Y, X) or (H, W).
 */

torch::Tensor rasterize_forward_torch(
    const std::tuple<int, int> image_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &pos2d,
    const torch::Tensor &conics_mu,
    const torch::Tensor &intensities
);

/**
 * rasterize_backward kernel binding. Kernel description:
 * Backward rasterization pass computing gradients w.r.t. input parameters. 
 * Each thread processes one pixel in the image. Block size defined in config.h.
 * 
 * Inputs:
 * @param image_size Dimensions of the image (height, width).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections.
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos2d 4D vectors storing gaussian positions (xyz) and unused w component.
 * @param conics_mu Conic parameters a, b, c for each gaussian, and mu (integration factor).
 * @param intensities Intensity values for each gaussian.
 * @param img_grad_in Gradient w.r.t. output image.
 * 
 * @return Outputs:
 * Gradient w.r.t. input 2D positions.
 * Gradient w.r.t. input conics a, b, c, and mu.
 * Gradient w.r.t. input intensities.
 */
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor> rasterize_backward_torch(
    const std::tuple<int, int> image_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &pos2d,
    const torch::Tensor &conics_mu,
    const torch::Tensor &intensities,
    const torch::Tensor &img_grad_in
);

/**
 * Alternative rasterize_backward binding that dispatches the per-gaussian CUDA kernel.
 * Mirrors the behavior of rasterize_backward_torch while reusing the bucket metadata
 * used by the forward pass to distribute work by gaussian.
 */
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor> rasterize_backward_per_gaussian_torch(
    const std::tuple<int, int> image_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &pos2d,
    const torch::Tensor &conics_mu,
    const torch::Tensor &intensities,
    const torch::Tensor &img_grad_in
);
