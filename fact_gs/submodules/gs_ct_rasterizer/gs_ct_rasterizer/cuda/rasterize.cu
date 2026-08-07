#include "rasterize.cuh"
#include "config.h"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>

namespace cg = cooperative_groups;

inline __device__ void warpSum4(float4& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
    val.w = cg::reduce(tile, val.w, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}

/** 
 * Forward rasterization pass of a set of 2D Gaussians into an image grid.
 * Each thread processes one pixel in the image. Block size defined in config.h.
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
)
{
    auto block = cg::this_thread_block();
    int32_t tile_id = block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y  + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;

    // Get the start and end indices of the gaussians in this tile
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // Clamp this value to the last pixel
    int32_t pix_id = i * img_size.x + j;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches

    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float4 conic_mu_batch[BLOCK_SIZE];
    __shared__ float intensity_batch[BLOCK_SIZE][CHANNELS];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();

    // **** max 4 channels for speed ***
    float pix_out[CHANNELS] = {0.f};

    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = pos2d[g_id];
            conic_mu_batch[tr] = conics_mu[g_id];
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) 
                intensity_batch[tr][c] = intensities[CHANNELS * g_id + c];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float4 conic_mu = conic_mu_batch[t];
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            const float sigma = 0.5f * (conic_mu.x * delta.x * delta.x +
                                        conic_mu.z * delta.y * delta.y) +
                                        conic_mu.y * delta.x * delta.y;
            
            if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
                continue;
            }
            
            const float alpha = exp(-sigma) * conic_mu.w;

            float mean_intensity = 0.f;
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                mean_intensity += intensity_batch[t][c];
            } 
            mean_intensity /= CHANNELS;

            if (alpha * mean_intensity < 0.00001f) { 
                continue;
            }

            for (int c = 0; c < CHANNELS; ++c) {
                pix_out[c] += intensity_batch[t][c] * alpha;
            }
        }
    }

    if (inside) {
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            out_img[CHANNELS * pix_id + c] = pix_out[c];
        }
    }

}

/** 
 * Backward pass for rasterization of a set of 2D Gaussians into an image grid.
 * Each thread processes one pixel in the image. Block size defined in config.h.
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
)
{
    auto block = cg::this_thread_block();
    int32_t tile_id = block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y  + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;

    // Get the start and end indices of the gaussians in this tile
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // Clamp this value to the last pixel
    int32_t pix_id = i * img_size.x + j;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches

    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float4 conic_mu_batch[BLOCK_SIZE];
    __shared__ float intensity_batch[BLOCK_SIZE][CHANNELS];

    const float* img_grad_ = inside ? &img_grad_in[CHANNELS * pix_id] : nullptr;

    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    //copy img_grad to local memory
    float img_grad_local[CHANNELS];
    if (inside) {
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            img_grad_local[c] = img_grad_[c];
        }
    }

    // Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * img_size.x;
	const float ddely_dy = 0.5 * img_size.y;


    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = pos2d[g_id];
            conic_mu_batch[tr] = conics_mu[g_id];
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) 
                intensity_batch[tr][c] = intensities[CHANNELS * g_id + c];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            bool valid = inside;

            const float4 conic_mu = conic_mu_batch[t];
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            const float sigma = 0.5f * (conic_mu.x * delta.x * delta.x +
                                        conic_mu.z * delta.y * delta.y) +
                                        conic_mu.y * delta.x * delta.y;
            
            if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
                valid = false;
            }

            float  intensities_grad_local[CHANNELS] = {0.f};
            float4 conics_mu_grad_local = {0.f, 0.f, 0.f, 0.f};
            float2 pos2d_grad_local = {0.f, 0.f};

            const float G = exp(-sigma);
            const float alpha = G * conic_mu.w;

            float mean_intensity = 0.f;
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                mean_intensity += intensity_batch[t][c];
            } 
            mean_intensity /= CHANNELS;

            if (alpha * mean_intensity  < 0.00001f) 
                valid = false;

            if (valid) {

                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c) {
                    intensities_grad_local[c] += alpha * img_grad_local[c];
                }
                float dL_dalpha = 0.f;
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c) {
                    dL_dalpha += intensity_batch[t][c] * img_grad_local[c];
                }
                float sigma_grad = -alpha * dL_dalpha;

                conics_mu_grad_local = {0.5f * sigma_grad * delta.x * delta.x,
                                        sigma_grad * delta.x * delta.y,
                                        0.5f * sigma_grad * delta.y * delta.y,
                                        G * dL_dalpha
                                       };

                pos2d_grad_local = {
                    sigma_grad * (conic_mu.x * delta.x + conic_mu.y * delta.y) * ddelx_dx,
                    sigma_grad * (conic_mu.y * delta.x + conic_mu.z * delta.y) * ddely_dy
                };

            }

            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                warpSum(intensities_grad_local[c], warp);
            }
            warpSum4(conics_mu_grad_local, warp);
            warpSum2(pos2d_grad_local, warp);

            if (warp.thread_rank() == 0) {
                int32_t g_id = id_batch[t];
                float* intensities_grad_ptr = (float*)(intensities_grad_out);
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c) {
                    atomicAdd(intensities_grad_ptr + CHANNELS * g_id + c, intensities_grad_local[c]);
                }
                float* conics_mu_grad_ptr = (float*)(conics_mu_grad_out);
                atomicAdd(conics_mu_grad_ptr + 4 * g_id + 0, conics_mu_grad_local.x);
                atomicAdd(conics_mu_grad_ptr + 4 * g_id + 1, conics_mu_grad_local.y);
                atomicAdd(conics_mu_grad_ptr + 4 * g_id + 2, conics_mu_grad_local.z);
                atomicAdd(conics_mu_grad_ptr + 4 * g_id + 3, conics_mu_grad_local.w);

                float* pos2d_grad_ptr = (float*)(pos2d_grad_out);
                atomicAdd(pos2d_grad_ptr + 2 * g_id + 0, pos2d_grad_local.x);
                atomicAdd(pos2d_grad_ptr + 2 * g_id + 1, pos2d_grad_local.y);
            }
        }
    }
}

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
)
{
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warps_per_block = block.size() / warp.size();
    const uint32_t global_bucket_idx =
        block.group_index().x * warps_per_block + warp.meta_group_rank();
    if (global_bucket_idx >= total_buckets)
        return;

    const uint32_t tile_id = bucket_to_tile[global_bucket_idx];
    const int2 range = tile_bins[tile_id];
    const int num_splats = range.y - range.x;
    if (num_splats <= 0)
        return;

    const uint32_t tile_bucket_base =
        tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
    const uint32_t bucket_idx_in_tile = global_bucket_idx - tile_bucket_base;
    const uint32_t splat_idx_in_tile =
        bucket_idx_in_tile * warp.size() + warp.thread_rank();
    const bool valid_splat = splat_idx_in_tile < (uint32_t)num_splats;
    const uint32_t splat_idx_global = range.x + splat_idx_in_tile;

    const int tiles_per_row = tile_bounds.x;
    const int tile_y = tile_id / tiles_per_row;
    const int tile_x = tile_id - tile_y * tiles_per_row;
    // Map tile id back to spatial coordinates so we can iterate over pixels.
    const int tile_origin_x = tile_x * BLOCK_X;
    const int tile_origin_y = tile_y * BLOCK_Y;

    int32_t g_id = 0;
    float2 xy = {0.f, 0.f};
    float4 conic_mu = {0.f, 0.f, 0.f, 0.f};
    float gaussian_intensity[CHANNELS] = {0.f};
    float mean_intensity = 0.f;

    // Each warp processes one gaussian-tile pair, so we only need to load one.
    if (valid_splat) {
        g_id = gaussian_ids_sorted[splat_idx_global];
        xy = pos2d[g_id];
        conic_mu = conics_mu[g_id];
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            const float value = intensities[CHANNELS * g_id + c];
            gaussian_intensity[c] = value;
            mean_intensity += value;
        }
        mean_intensity /= CHANNELS;
    }

    float intensities_grad_local[CHANNELS] = {0.f};
    float4 conics_mu_grad_local = {0.f, 0.f, 0.f, 0.f};
    float2 pos2d_grad_local = {0.f, 0.f};

    float px = 0.f;
    float py = 0.f;
    int pixel_active = 0;
    float pixel_grad[CHANNELS] = {0.f};

    const float ddelx_dx = 0.5f * (float)img_size.x;
    const float ddely_dy = 0.5f * (float)img_size.y;

    // Simulate the shuffle-based pixel traversal from the renderer: each pass
    // shifts the pixel state down the warp while lane 0 brings in the next
    // pixel for this tile. This lets us reuse the same dataflow as
    // per-gaussian rendering without storing per-pixel state in shared memory.
    const int pipeline_iters = BLOCK_SIZE + warp.size() - 1;
    for (int iter = 0; iter < pipeline_iters; ++iter) {
        pixel_active = warp.shfl_up(pixel_active, 1);
        px = warp.shfl_up(px, 1);
        py = warp.shfl_up(py, 1);
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            pixel_grad[c] = warp.shfl_up(pixel_grad[c], 1);
        }

        const int idx = iter - warp.thread_rank();
        const bool idx_in_range = (idx >= 0) && (idx < BLOCK_SIZE);

        if (idx_in_range && warp.thread_rank() == 0) {
            // Lane 0 computes global pixel coordinates and pulls gradients.
            const int local_y = idx / BLOCK_X;
            const int local_x = idx - local_y * BLOCK_X;
            const int global_x = tile_origin_x + local_x;
            const int global_y = tile_origin_y + local_y;

            const bool inside =
                (global_x < img_size.x) && (global_y < img_size.y);
            pixel_active = inside ? 1 : 0;
            if (inside) {
                px = (float)global_x;
                py = (float)global_y;
                const int pix_id = global_y * img_size.x + global_x;
                const float* grad_ptr = img_grad_in + CHANNELS * pix_id;
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c) {
                    pixel_grad[c] = grad_ptr[c];
                }
            } else {
                px = py = 0.f;
                #pragma unroll
                for (int c = 0; c < CHANNELS; ++c) {
                    pixel_grad[c] = 0.f;
                }
            }
        }

        const bool process = valid_splat && idx_in_range && pixel_active;
        // The original pixel-wise kernel accumulated into shared memory per
        // pixel; here we only accumulate when both a pixel and gaussian are
        // valid for this warp.
        if (!process)
            continue;

        const float2 delta = {xy.x - px, xy.y - py};
        const float sigma = 0.5f * (conic_mu.x * delta.x * delta.x +
                                    conic_mu.z * delta.y * delta.y) +
                            conic_mu.y * delta.x * delta.y;

        if (sigma < 0.f || isnan(sigma) || isinf(sigma))
            continue;

        const float G = exp(-sigma);
        const float alpha = G * conic_mu.w;

        if (alpha * mean_intensity < 0.00001f)
            continue;

        float dL_dalpha = 0.f;
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            const float grad = pixel_grad[c];
            intensities_grad_local[c] += alpha * grad;
            dL_dalpha += gaussian_intensity[c] * grad;
        }

        const float sigma_grad = -alpha * dL_dalpha;

        conics_mu_grad_local.x += 0.5f * sigma_grad * delta.x * delta.x;
        conics_mu_grad_local.y += sigma_grad * delta.x * delta.y;
        conics_mu_grad_local.z += 0.5f * sigma_grad * delta.y * delta.y;
        conics_mu_grad_local.w += G * dL_dalpha;

        pos2d_grad_local.x +=
            sigma_grad * (conic_mu.x * delta.x + conic_mu.y * delta.y) * ddelx_dx;
        pos2d_grad_local.y +=
            sigma_grad * (conic_mu.y * delta.x + conic_mu.z * delta.y) * ddely_dy;
    }

    if (valid_splat) {
        // Unlike rasterize_backward (which atomically adds per pixel), each warp
        // accumulates a gaussian's entire contribution before touching global
        // memory, so we only emit one set of atomics per gaussian.
        float* intensities_grad_ptr = intensities_grad_out + CHANNELS * g_id;
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            atomicAdd(intensities_grad_ptr + c, intensities_grad_local[c]);
        }

        float* conics_mu_grad_ptr = (float*)(conics_mu_grad_out);
        atomicAdd(conics_mu_grad_ptr + 4 * g_id + 0, conics_mu_grad_local.x);
        atomicAdd(conics_mu_grad_ptr + 4 * g_id + 1, conics_mu_grad_local.y);
        atomicAdd(conics_mu_grad_ptr + 4 * g_id + 2, conics_mu_grad_local.z);
        atomicAdd(conics_mu_grad_ptr + 4 * g_id + 3, conics_mu_grad_local.w);

        float* pos2d_grad_ptr = (float*)(pos2d_grad_out);
        atomicAdd(pos2d_grad_ptr + 2 * g_id + 0, pos2d_grad_local.x);
        atomicAdd(pos2d_grad_ptr + 2 * g_id + 1, pos2d_grad_local.y);
    }
}

// Kernel to compute the number of gaussians in each bucket. 
__global__ void rasterize_compute_bucket_counts_kernel(
    const int num_tiles,
    const int2* __restrict__ tile_bins,
    uint32_t* __restrict__ bucket_counts
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tiles)
        return;
    const int2 range = tile_bins[idx];
    int splats = range.y - range.x;
    // Each bucket can hold up to 32 gaussians, mirroring the rendering path.
    splats = splats < 0 ? 0 : splats;
    bucket_counts[idx] = (uint32_t)((splats + 31) / 32);
}

// Kernel to build a mapping from buckets to tiles based on the bucket counts and tile_bins
__global__ void rasterize_build_bucket_to_tile_kernel(
    const int num_tiles,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_offsets,
    uint32_t* __restrict__ bucket_to_tile
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tiles)
        return;
    const uint32_t count = bucket_counts[idx];
    if (count == 0 || bucket_to_tile == nullptr)
        return;
    const uint32_t base = idx == 0 ? 0 : bucket_offsets[idx - 1];
    // Fill the mapping so warp i directly knows which tile to sample from.
    for (uint32_t b = 0; b < count; ++b) {
        bucket_to_tile[base + b] = (uint32_t)idx;
    }
}

template __global__ void rasterize_forward<1>(const dim3, const uint2, const int32_t*, 
    const int2*, const float2*,const float4*,const float*,float*);

template __global__ void rasterize_forward<2>(const dim3, const uint2, const int32_t*, 
    const int2*, const float2*,const float4*,const float*,float*);

template __global__ void rasterize_forward<3>(const dim3, const uint2, const int32_t*, 
    const int2*, const float2*,const float4*,const float*,float*);

template __global__ void rasterize_forward<4>(const dim3, const uint2, const int32_t*, 
    const int2*, const float2*,const float4*,const float*,float*);


template __global__ void rasterize_backward<1>(const dim3, const uint2, const int32_t*, 
    const int2*, const float2*,const float4*,const float*,const float*,float2*,float4*,float*);

template __global__ void rasterize_backward<2>(const dim3, const uint2, const int32_t*, 
    const int2*, const float2*,const float4*,const float*,const float*,float2*,float4*,float*);

template __global__ void rasterize_backward<3>(const dim3, const uint2, const int32_t*, 
    const int2*, const float2*,const float4*,const float*,const float*,float2*,float4*,float*);

template __global__ void rasterize_backward<4>(const dim3, const uint2, const int32_t*, 
    const int2*, const float2*,const float4*,const float*,const float*,float2*,float4*,float*);

template __global__ void rasterize_backward_per_gaussian<1>(const dim3, const uint2, const int32_t*,
    const int2*, const float2*, const float4*, const float*, const float*, const uint32_t*, const uint32_t*, const uint32_t, float2*, float4*, float*);

template __global__ void rasterize_backward_per_gaussian<2>(const dim3, const uint2, const int32_t*,
    const int2*, const float2*, const float4*, const float*, const float*, const uint32_t*, const uint32_t*, const uint32_t, float2*, float4*, float*);

template __global__ void rasterize_backward_per_gaussian<3>(const dim3, const uint2, const int32_t*,
    const int2*, const float2*, const float4*, const float*, const float*, const uint32_t*, const uint32_t*, const uint32_t, float2*, float4*, float*);

template __global__ void rasterize_backward_per_gaussian<4>(const dim3, const uint2, const int32_t*,
    const int2*, const float2*, const float4*, const float*, const float*, const uint32_t*, const uint32_t*, const uint32_t, float2*, float4*, float*);
