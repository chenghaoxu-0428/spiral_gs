#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>


/**
 * Forward rasterization kernel.
 * 
 * Forward rasterization pass of a set of 2D Gaussians into an image grid.
 * Each thread processes one pixel in the image. Block size defined in config.h.
 * 
 *  Template parameter:
 * @tparam CHANNELS Number of channels in the output image.
 * 
 * Inputs:
 * @param tile_bounds Dimensions of the image in tiles (width, height).
 * @param img_size Dimensions of the image in pixels (width, height).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos2d 2D positions of gaussians (xy).
 * @param conics_mu  Conic parameters a, b, c for each gaussian, and mu (integration factor).
 * @param intensities Intensity values for each gaussian and channel.
 * 
 * @return Outputs:
 * @param out_img Rasterized output image
 */
template<int CHANNELS> __global__ void rasterize_forward(
    const dim3 tile_bounds,
    const uint2 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ pos2d,
    const float4* __restrict__ conics_mu,
    const float* __restrict__ intensities,
    float* __restrict__ out_img
);

/**
 * Backward rasterization kernel.
 * 
 * Backward rasterization pass computing gradients w.r.t. input parameters. 
 * Each thread processes one pixel in the image. Block size defined in config.h.
 * 
 *  Template parameter:
 * @tparam CHANNELS Number of channels in the output image.
 * 
 * Inputs:
 * @param tile_bounds Dimensions of the image in tiles (width, height).
 * @param img_size Dimensions of the image in pixels (width, height).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos2d 2D positions of gaussians (xy).
 * @param conics_mu  Conic parameters a, b, c for each gaussian, and mu (integration factor).
 * @param intensities Intensity values for each gaussian and channel.
 * @param img_grad_in Incoming gradient of the loss w.r.t. the output image.
 * 
 * @return Outputs:
 * @param pos2d_grad_out Gradient of the loss w.r.t. the 2D positions.
 * @param conics_mu_grad_out Gradient of the loss w.r.t. the conic parameters and mu.
 * @param intensities_grad_out Gradient of the loss w.r.t. the intensities
 */
template<int CHANNELS> __global__ void rasterize_backward(
    const dim3 tile_bounds,
    const uint2 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ pos2d,
    const float4* __restrict__ conics_mu,
    const float* __restrict__ intensities,
    const float* __restrict__ img_grad_in,
    float2* __restrict__ pos2d_grad_out,
    float4* __restrict__ conics_mu_grad_out,
    float* __restrict__ intensities_grad_out
);

/**
 * Alternative backward rasterization kernel where work is distributed per-gaussian.
 * Each warp in a block processes one gaussian-tile pair and iterates over voxels.
 * 
 * Template parameter:
 * @tparam CHANNELS Number of channels in the output image.
 * 
 * Inputs:
 * @param tile_bounds Dimensions of the image in tiles (width, height).
 * @param img_size Dimensions of the image in pixels (width, height).
 * @param gaussian_ids_sorted Sorted gaussian IDs corresponding to intersections
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * @param pos2d 2D positions of gaussians (xy).
 * @param conics_mu  Conic parameters a, b, c for each gaussian, and mu (integration factor).
 * @param intensities Intensity values for each gaussian and channel.
 * @param img_grad_in Incoming gradient of the loss w.r.t. the output image.
 * @param per_tile_bucket_offset Precomputed offsets to access tile bins from bucket_to_tile.
 * @param bucket_to_tile Mapping from bucket index to tile index for non-empty tiles.
 * @param total_buckets Total number of non-empty buckets (tiles).
 * 
 * @return Outputs:
 * @param pos2d_grad_out Gradient of the loss w.r.t. the 2D positions.
 * @param conics_mu_grad_out Gradient of the loss w.r.t. the conic parameters and mu.
 * @param intensities_grad_out Gradient of the loss w.r.t. the intensities.
 */
template<int CHANNELS> __global__ void rasterize_backward_per_gaussian(
    const dim3 tile_bounds,
    const uint2 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ pos2d,
    const float4* __restrict__ conics_mu,
    const float* __restrict__ intensities,
    const float* __restrict__ img_grad_in,
    const uint32_t* __restrict__ per_tile_bucket_offset,
    const uint32_t* __restrict__ bucket_to_tile,
    const uint32_t total_buckets,
    float2* __restrict__ pos2d_grad_out,
    float4* __restrict__ conics_mu_grad_out,
    float* __restrict__ intensities_grad_out
);

/** Kernel to compute the number of gaussians in each bucket. 
 * 
 * Inputs:
 * @param num_tiles Number of tiles.
 * @param tile_bins Start and end indices for each tile's bin of intersections.
 * 
 * Output:
 * @param bucket_counts Number of gaussians in each bucket.
 */
__global__ void rasterize_compute_bucket_counts_kernel(
    const int num_tiles,
    const int2* __restrict__ tile_bins,
    uint32_t* __restrict__ bucket_counts
);

/** Kernel to build a mapping from buckets to tiles based on the bucket counts and tile_bins.
 * 
 * Inputs:
 * @param num_tiles Number of tiles.
 * @param bucket_counts Number of gaussians in each bucket.
 * @param bucket_offsets Offsets for each bucket to access the output array.
 * 
 * Output:
 * @param bucket_to_tile Mapping from bucket index to tile index for non-empty buckets.
 */
__global__ void rasterize_build_bucket_to_tile_kernel(
    const int num_tiles,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t* __restrict__ bucket_to_tile
);
