#include <cstdint>
#include <cuda_runtime.h>
#include "bin_and_sort_gaussians.cuh"
#include "config.h"

/**
 * Map each intersection from tile ID to a gaussian ID. 
 * Outputs two arrays of size cum_tiles_hit[-1] containing tile-gaussian pairs.
 * Gaussian dimension is slower.
 */
__global__ void map_gaussian_to_intersects(
    const int num_gaussians,
    int2* __restrict__ tile_min,
    int2* __restrict__ tile_max,
    const int32_t* __restrict__ cum_tiles_hit,
    const int2 tile_bounds,
    int32_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians)
        return;

    // update the intersection info for all tiles this gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];;
    for (int i = tile_min[idx].y; i < tile_max[idx].y; ++i) {
        for (int j = tile_min[idx].x; j < tile_max[idx].x; ++j) {
            // isect_id is tile ID and depth as int32
            isect_ids[cur_idx] = i * tile_bounds.x + j; // tile id
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            ++cur_idx; // handles gaussians that hit more than one tile
            
        }
    }
}

/**
 * Given sorted intersection IDs, find its bins (how many gaussians in each tile).
 */ 
__global__ void get_tile_bin_edges(
    const int num_intersects, const int32_t* __restrict__ isect_ids_sorted, int2* __restrict__ tile_bins
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    int32_t cur_tile_idx = isect_ids_sorted[idx];
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            tile_bins[cur_tile_idx].x = 0;
        if (idx == num_intersects - 1)
            tile_bins[cur_tile_idx].y = num_intersects;
    }
    if (idx == 0)
        return;
    int32_t prev_tile_idx = isect_ids_sorted[idx - 1];
    if (prev_tile_idx != cur_tile_idx) {
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x = idx;
        return;
    }
}
