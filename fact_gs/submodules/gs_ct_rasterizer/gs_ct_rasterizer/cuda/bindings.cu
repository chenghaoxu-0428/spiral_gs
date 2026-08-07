#include "bindings.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cstdint>
#include "config.h"

#include "bin_and_sort_gaussians.cuh"
#include "optim_to_render.cuh"
#include "rasterize.cuh"

/**
 * Zero out an existing tensor in-place, respecting its device and dtype.
 * Cuda implementation uses cudaMemsetAsync. The goal is to avoid torch::zeros allocation where possible.
 * 
 * Inputs:
 * @param tensor Tensor to be zeroed.
 */
inline void zero_tensor(torch::Tensor &tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return;
    }
    if (tensor.is_cuda()) {
        auto stream = at::cuda::getCurrentCUDAStream();
        CUDA_CALL(cudaMemsetAsync(
            tensor.data_ptr(),
            0,
            tensor.numel() * tensor.element_size(),
            stream.stream()));
    } else {
        tensor.zero_();
    }
}


/**
 * Kernel description:
 * Converts gaussian parameters from 3D optimization format to 2D rasterization format.
 * Position from [0,1] to pixel-based, scale and quat to conics
 * Prepares radii for assigning gaussians to volume tiles.
 * Each thread processes one gaussian.
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
) {
    CHECK_INPUT(pos3d);
    CHECK_INPUT(scale3d);
    CHECK_INPUT(quat);
    CHECK_INPUT(intensities);
    CHECK_INPUT(viewmatrix);
    CHECK_INPUT(projmatrix);

    const int channels = intensities.size(1);

    if (channels > MAX_CHANNELS) {
        throw std::runtime_error("Number of channels exceeds MAX_CHANNELS");
    }
    
    uint2 image_size = make_uint2(image_width, image_height);

    const float focal_y = image_height / (2.0f * tan_fovy);  // tan_fovy=1 when parallel
	const float focal_x = image_width / (2.0f * tan_fovx);  // tan_fovx=1 when parallel

    dim3 tile_grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1);

    auto float_options = pos3d.options().dtype(torch::kFloat32);
    auto int_options = pos3d.options().dtype(torch::kInt32);

    torch::Tensor pos2d_out = torch::empty({num_gaussians, 2}, float_options);
    torch::Tensor conics_mu_out = torch::empty({num_gaussians, 4}, float_options);
    torch::Tensor radii_out = torch::empty({num_gaussians}, float_options);
    torch::Tensor num_tiles_hit_out = torch::empty({num_gaussians}, int_options);
    torch::Tensor tile_min_out = torch::empty({num_gaussians, 2}, int_options);
    torch::Tensor tile_max_out = torch::empty({num_gaussians, 2}, int_options);

    switch(channels)
    {
    case 1:
        optim_to_render_forward<1><<<
            (num_gaussians + N_THREADS - 1) / N_THREADS,
            N_THREADS>>>(
            num_gaussians,
            (float3*)pos3d.contiguous().data_ptr<float>(),
            (float3*)scale3d.contiguous().data_ptr<float>(),
            (float4*)quat.contiguous().data_ptr<float>(),
            intensities.contiguous().data_ptr<float>(),
            (float*)viewmatrix.contiguous().data_ptr<float>(),
            (float*)projmatrix.contiguous().data_ptr<float>(),
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            image_size,
            tile_grid,
            mode,
            (float2*)pos2d_out.contiguous().data_ptr<float>(),
            (float4*)conics_mu_out.contiguous().data_ptr<float>(),
            (float*)radii_out.contiguous().data_ptr<float>(),
            (int32_t*)num_tiles_hit_out.contiguous().data_ptr<int32_t>(),
            (int2*)tile_min_out.contiguous().data_ptr<int>(),
            (int2*)tile_max_out.contiguous().data_ptr<int>()
        );
        break;
    case 2:
            optim_to_render_forward<2><<<
                (num_gaussians + N_THREADS - 1) / N_THREADS,
                N_THREADS>>>(
                num_gaussians,
                (float3*)pos3d.contiguous().data_ptr<float>(),
                (float3*)scale3d.contiguous().data_ptr<float>(),
                (float4*)quat.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                (float*)viewmatrix.contiguous().data_ptr<float>(),
                (float*)projmatrix.contiguous().data_ptr<float>(),
                focal_x, focal_y,
                tan_fovx, tan_fovy,
                image_size,
                tile_grid,
                mode,
                (float2*)pos2d_out.contiguous().data_ptr<float>(),
                (float4*)conics_mu_out.contiguous().data_ptr<float>(),
                (float*)radii_out.contiguous().data_ptr<float>(),
                (int32_t*)num_tiles_hit_out.contiguous().data_ptr<int32_t>(),
                (int2*)tile_min_out.contiguous().data_ptr<int>(),
                (int2*)tile_max_out.contiguous().data_ptr<int>()
            );
            break;
    case 3:
            optim_to_render_forward<3><<<
                (num_gaussians + N_THREADS - 1) / N_THREADS,
                N_THREADS>>>(
                num_gaussians,
                (float3*)pos3d.contiguous().data_ptr<float>(),
                (float3*)scale3d.contiguous().data_ptr<float>(),
                (float4*)quat.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                (float*)viewmatrix.contiguous().data_ptr<float>(),
                (float*)projmatrix.contiguous().data_ptr<float>(),
                focal_x, focal_y,
                tan_fovx, tan_fovy,
                image_size,
                tile_grid,
                mode,
                (float2*)pos2d_out.contiguous().data_ptr<float>(),
                (float4*)conics_mu_out.contiguous().data_ptr<float>(),
                (float*)radii_out.contiguous().data_ptr<float>(),
                (int32_t*)num_tiles_hit_out.contiguous().data_ptr<int32_t>(),
                (int2*)tile_min_out.contiguous().data_ptr<int>(),
                (int2*)tile_max_out.contiguous().data_ptr<int>()
            );
            break;
    case 4:
            optim_to_render_forward<4><<<
                (num_gaussians + N_THREADS - 1) / N_THREADS,
                N_THREADS>>>(
                num_gaussians,
                (float3*)pos3d.contiguous().data_ptr<float>(),
                (float3*)scale3d.contiguous().data_ptr<float>(),
                (float4*)quat.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                (float*)viewmatrix.contiguous().data_ptr<float>(),
                (float*)projmatrix.contiguous().data_ptr<float>(),
                focal_x, focal_y,
                tan_fovx, tan_fovy,
                image_size,
                tile_grid,
                mode,
                (float2*)pos2d_out.contiguous().data_ptr<float>(),
                (float4*)conics_mu_out.contiguous().data_ptr<float>(),
                (float*)radii_out.contiguous().data_ptr<float>(),
                (int32_t*)num_tiles_hit_out.contiguous().data_ptr<int32_t>(),
                (int2*)tile_min_out.contiguous().data_ptr<int>(),
                (int2*)tile_max_out.contiguous().data_ptr<int>()
            );
            break;
    default:
        break;
    }
    return std::make_tuple(
        pos2d_out,
        conics_mu_out,
        radii_out,
        tile_min_out,
        tile_max_out,
        num_tiles_hit_out
    );
}

/**
 * Kernel description:
 * Backward pass for optim_to_render_forward. 
 * Computes gradients w.r.t. input parameters (pos3d, scale3d, quat).
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
) {
    const float focal_y = image_height / (2.0f * tan_fovy);  // tan_fovy=1 when parallel
	const float focal_x = image_width / (2.0f * tan_fovx);  // tan_fovx=1 when parallel

    torch::Tensor pos3d_grad_out = 
        torch::empty({num_gaussians, 3}, pos3d.options().dtype(torch::kFloat32));
    zero_tensor(pos3d_grad_out);
    torch::Tensor scale3d_grad_out = 
        torch::empty({num_gaussians, 3}, pos3d.options().dtype(torch::kFloat32));
    zero_tensor(scale3d_grad_out);
    torch::Tensor quat_grad_out = 
        torch::empty({num_gaussians, 4}, pos3d.options().dtype(torch::kFloat32));
    zero_tensor(quat_grad_out);

    optim_to_render_backward<<<
        (num_gaussians + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_gaussians,
        (float3*)pos3d.contiguous().data_ptr<float>(),
        (float3*)scale3d.contiguous().data_ptr<float>(),
        (float4*)quat.contiguous().data_ptr<float>(),
        (float*)viewmatrix.contiguous().data_ptr<float>(),
        (float*)projmatrix.contiguous().data_ptr<float>(),
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        mode,
        (float*)radii.contiguous().data_ptr<float>(),
        (float2*)pos2d_grad_in.contiguous().data_ptr<float>(),
        (float4*)conics_mu_grad_in.contiguous().data_ptr<float>(),
        (float3*)pos3d_grad_out.contiguous().data_ptr<float>(),
        (float3*)scale3d_grad_out.contiguous().data_ptr<float>(),
        (float4*)quat_grad_out.contiguous().data_ptr<float>()
    );
    return std::make_tuple(
        pos3d_grad_out,
        scale3d_grad_out,
        quat_grad_out
    );

}


/**
 *  Combined binding that maps gaussians to tiles, sorts intersections
 *  with a radix sort, and builds tile bins.
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
) {
    CHECK_INPUT(tile_min);
    CHECK_INPUT(tile_max);
    CHECK_INPUT(cum_tiles_hit);

    int2 tile_bounds = make_int2(
        (std::get<1>(img_size) + BLOCK_X - 1) / BLOCK_X,
        (std::get<0>(img_size) + BLOCK_Y - 1) / BLOCK_Y
    );
    int num_tiles = tile_bounds.x * tile_bounds.y;

    auto options_i32 = tile_min.options().dtype(torch::kInt32);

    if (num_intersects <= 0) {
        torch::Tensor gaussian_ids_sorted = torch::empty({0}, options_i32);
        torch::Tensor tile_bins = torch::empty({num_tiles, 2}, options_i32);
        zero_tensor(tile_bins);
        return std::make_tuple(gaussian_ids_sorted, tile_bins);
    }

    auto tile_min_c = tile_min.contiguous();
    auto tile_max_c = tile_max.contiguous();
    auto cum_tiles_hit_c = cum_tiles_hit.contiguous();

    torch::Tensor isect_ids = torch::empty({num_intersects}, options_i32);
    torch::Tensor gaussian_ids = torch::empty({num_intersects}, options_i32);

    // Generate the unsorted tile/gaussian pairs by iterating over the
    // per-gaussian tile ranges. 
    map_gaussian_to_intersects<<<
        (num_gaussians + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_gaussians,
        (int2*)tile_min_c.data_ptr<int>(),
        (int2*)tile_max_c.data_ptr<int>(),
        (int32_t*)cum_tiles_hit_c.data_ptr<int32_t>(),
        tile_bounds,
        (int32_t*)isect_ids.data_ptr<int32_t>(),
        (int32_t*)gaussian_ids.data_ptr<int32_t>()
    );

    torch::Tensor isect_ids_sorted = torch::empty({num_intersects}, options_i32);
    torch::Tensor gaussian_ids_sorted = torch::empty({num_intersects}, options_i32);

    size_t temp_storage_bytes = 0;
    auto stream = at::cuda::getCurrentCUDAStream();

    // Query how much temporary storage the radix sort needs for this workload.
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        temp_storage_bytes,
        (int32_t*)isect_ids.data_ptr<int32_t>(),
        (int32_t*)isect_ids_sorted.data_ptr<int32_t>(),
        (int32_t*)gaussian_ids.data_ptr<int32_t>(),
        (int32_t*)gaussian_ids_sorted.data_ptr<int32_t>(),
        num_intersects,
        0,
        sizeof(int32_t) * 8,
        stream.stream());

    torch::Tensor temp_storage;
    uint8_t* temp_ptr = nullptr;
    if (temp_storage_bytes > 0) {
        temp_storage = torch::empty({static_cast<int64_t>(temp_storage_bytes)}, tile_min.options().dtype(torch::kUInt8));
        temp_ptr = temp_storage.data_ptr<uint8_t>();
    }

    // Perform the actual sort using the allocated scratch buffer.
    cub::DeviceRadixSort::SortPairs(
        temp_ptr,
        temp_storage_bytes,
        (int32_t*)isect_ids.data_ptr<int32_t>(),
        (int32_t*)isect_ids_sorted.data_ptr<int32_t>(),
        (int32_t*)gaussian_ids.data_ptr<int32_t>(),
        (int32_t*)gaussian_ids_sorted.data_ptr<int32_t>(),
        num_intersects,
        0,
        sizeof(int32_t) * 8,
        stream.stream());

    torch::Tensor tile_bins =
        torch::empty({num_tiles, 2}, options_i32);
    zero_tensor(tile_bins);

    // Use the sorted tile IDs to find per-tile start/end offsets.
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        (int32_t*)isect_ids_sorted.data_ptr<int32_t>(),
        (int2*)tile_bins.data_ptr<int>()
    );

    return std::make_tuple(gaussian_ids_sorted, tile_bins);
}



/**
 * Kernel description:
 * Forward rasterization pass of a set of 2D Gaussians into an image grid.
 * Each thread processes one pixel in the image. Block size defined in config.h.
 */
torch::Tensor rasterize_forward_torch(
    const std::tuple<int, int> image_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &pos2d,
    const torch::Tensor &conics_mu,
    const torch::Tensor &intensities
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(pos2d);
    CHECK_INPUT(conics_mu);
    CHECK_INPUT(intensities);

    dim3 block_dim3 = dim3(BLOCK_X, BLOCK_Y, 1);

    uint2 image_size_uint2 = make_uint2(
        std::get<1>(image_size),
        std::get<0>(image_size)
    );

    dim3 tile_bounds_dim3 = dim3(
        (image_size_uint2.x + BLOCK_X - 1) / BLOCK_X,
        (image_size_uint2.y + BLOCK_Y - 1) / BLOCK_Y,
        1
    );

    const int channels = intensities.size(1);

    if (channels > MAX_CHANNELS) {
        throw std::runtime_error("Number of channels exceeds MAX_CHANNELS");
    }

    torch::Tensor out_img =
        torch::empty(
            {image_size_uint2.y, image_size_uint2.x, channels}, 
            pos2d.options().dtype(torch::kFloat32)
        );

    switch(channels) {
        case 1:
            rasterize_forward<1><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                image_size_uint2,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float2*)pos2d.contiguous().data_ptr<float>(),
                (float4*)conics_mu.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                out_img.contiguous().data_ptr<float>()
            );
            break;
        case 2:
            rasterize_forward<2><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                image_size_uint2,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float2*)pos2d.contiguous().data_ptr<float>(),
                (float4*)conics_mu.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                out_img.contiguous().data_ptr<float>()
            );
            break;
        case 3:
            rasterize_forward<3><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                image_size_uint2,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float2*)pos2d.contiguous().data_ptr<float>(),
                (float4*)conics_mu.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                out_img.contiguous().data_ptr<float>()
            );
            break;
        case 4:
            rasterize_forward<4><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                image_size_uint2,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float2*)pos2d.contiguous().data_ptr<float>(),
                (float4*)conics_mu.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                out_img.contiguous().data_ptr<float>()
            );
            break;
        default:
            throw std::runtime_error("Unsupported number of channels");
    }

    return out_img;
}

/**
 * Kernel description:
 * Backward rasterization pass computing gradients w.r.t. input parameters. 
 * Each thread processes one pixel in the image. Block size defined in config.h.
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
) {
    CHECK_INPUT(pos2d);
    CHECK_INPUT(intensities);

    const int channels = intensities.size(1);
    const int num_gaussians = pos2d.size(0);

    if (channels > MAX_CHANNELS) {
        throw std::runtime_error("Number of channels exceeds MAX_CHANNELS");
    }

    uint2 image_size_uint2 = make_uint2(
        std::get<1>(image_size),
        std::get<0>(image_size)
    );

    dim3 tile_bounds_dim3 = dim3(
        (image_size_uint2.x + BLOCK_X - 1) / BLOCK_X,
        (image_size_uint2.y + BLOCK_Y - 1) / BLOCK_Y,
        1
    );

    dim3 block_dim3 = dim3(BLOCK_X, BLOCK_Y, 1);

    torch::Tensor pos2d_grad_out = 
        torch::empty({num_gaussians, 2}, pos2d.options());
    torch::Tensor conics_mu_grad_out = 
        torch::empty({num_gaussians, 4}, pos2d.options());
    torch::Tensor intensities_grad_out = 
        torch::empty({num_gaussians, channels}, pos2d.options());

    zero_tensor(pos2d_grad_out);
    zero_tensor(conics_mu_grad_out);
    zero_tensor(intensities_grad_out);

    switch (channels) {
        case 1:
            rasterize_backward<1><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                image_size_uint2,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float2*)pos2d.contiguous().data_ptr<float>(),
                (float4*)conics_mu.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                img_grad_in.contiguous().data_ptr<float>(),
                (float2*)pos2d_grad_out.contiguous().data_ptr<float>(),
                (float4*)conics_mu_grad_out.contiguous().data_ptr<float>(),
                intensities_grad_out.contiguous().data_ptr<float>()
            );
            break;
        case 2:
            rasterize_backward<2><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                image_size_uint2,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float2*)pos2d.contiguous().data_ptr<float>(),
                (float4*)conics_mu.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                img_grad_in.contiguous().data_ptr<float>(),
                (float2*)pos2d_grad_out.contiguous().data_ptr<float>(),
                (float4*)conics_mu_grad_out.contiguous().data_ptr<float>(),
                intensities_grad_out.contiguous().data_ptr<float>()
            );
            break;
        case 3:
            rasterize_backward<3><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                image_size_uint2,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float2*)pos2d.contiguous().data_ptr<float>(),
                (float4*)conics_mu.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                img_grad_in.contiguous().data_ptr<float>(),
                (float2*)pos2d_grad_out.contiguous().data_ptr<float>(),
                (float4*)conics_mu_grad_out.contiguous().data_ptr<float>(),
                intensities_grad_out.contiguous().data_ptr<float>()
            );
            break;
        case 4:
            rasterize_backward<4><<<tile_bounds_dim3,block_dim3>>>(
                tile_bounds_dim3,
                image_size_uint2,
                (int32_t*)gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
                (int2*)tile_bins.contiguous().data_ptr<int>(),
                (float2*)pos2d.contiguous().data_ptr<float>(),
                (float4*)conics_mu.contiguous().data_ptr<float>(),
                intensities.contiguous().data_ptr<float>(),
                img_grad_in.contiguous().data_ptr<float>(),
                (float2*)pos2d_grad_out.contiguous().data_ptr<float>(),
                (float4*)conics_mu_grad_out.contiguous().data_ptr<float>(),
                intensities_grad_out.contiguous().data_ptr<float>()
            );
            break;
        default:
            throw std::runtime_error("Unsupported number of channels");
    }

    return std::make_tuple(
        pos2d_grad_out,
        conics_mu_grad_out,
        intensities_grad_out
    );
}

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
)
{
    CHECK_INPUT(pos2d);
    CHECK_INPUT(intensities);

    const int channels = intensities.size(1);
    const int num_gaussians = pos2d.size(0);

    if (channels > MAX_CHANNELS) {
        throw std::runtime_error("Number of channels exceeds MAX_CHANNELS");
    }

    uint2 image_size_uint2 = make_uint2(
        std::get<1>(image_size),
        std::get<0>(image_size)
    );

    dim3 tile_bounds_dim3 = dim3(
        (image_size_uint2.x + BLOCK_X - 1) / BLOCK_X,
        (image_size_uint2.y + BLOCK_Y - 1) / BLOCK_Y,
        1
    );

    const int threads = N_THREADS;
    const int warps_per_block = threads / 32;

    torch::Tensor pos2d_grad_out =
        torch::empty({num_gaussians, 2}, pos2d.options());
    torch::Tensor conics_mu_grad_out =
        torch::empty({num_gaussians, 4}, pos2d.options());
    torch::Tensor intensities_grad_out =
        torch::empty({num_gaussians, channels}, pos2d.options());

    zero_tensor(pos2d_grad_out);
    zero_tensor(conics_mu_grad_out);
    zero_tensor(intensities_grad_out);

    const int num_tiles = tile_bounds_dim3.x * tile_bounds_dim3.y;

    auto tile_bins_contig = tile_bins.contiguous();
    torch::Tensor bucket_counts;
    torch::Tensor bucket_offsets;
    int total_buckets = 0;

    if (num_tiles > 0) {
        bucket_counts = torch::empty({num_tiles}, tile_bins.options().dtype(torch::kInt32));
        bucket_offsets = torch::empty({num_tiles}, tile_bins.options().dtype(torch::kInt32));

        rasterize_compute_bucket_counts_kernel<<<
            (num_tiles + N_THREADS - 1) / N_THREADS,
            N_THREADS>>>(
            num_tiles,
            (int2*)tile_bins_contig.data_ptr<int>(),
            (uint32_t*)bucket_counts.data_ptr<int>()
        );

        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            (uint32_t*)bucket_counts.data_ptr<int>(),
            (uint32_t*)bucket_offsets.data_ptr<int>(),
            num_tiles
        );

        torch::Tensor temp_storage;
        if (temp_storage_bytes > 0) {
            temp_storage = torch::empty(
                {static_cast<long>(temp_storage_bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(bucket_counts.device())
            );
        }

        cub::DeviceScan::InclusiveSum(
            temp_storage_bytes > 0 ? temp_storage.data_ptr<uint8_t>() : nullptr,
            temp_storage_bytes,
            (uint32_t*)bucket_counts.data_ptr<int>(),
            (uint32_t*)bucket_offsets.data_ptr<int>(),
            num_tiles
        );

        CUDA_CALL(cudaMemcpy(
            &total_buckets,
            ((uint32_t*)bucket_offsets.data_ptr<int>()) + num_tiles - 1,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        ));

        if (total_buckets > 0) {
            torch::Tensor bucket_to_tile = torch::empty({total_buckets}, tile_bins.options().dtype(torch::kInt32));
            rasterize_build_bucket_to_tile_kernel<<<
                (num_tiles + N_THREADS - 1) / N_THREADS,
                N_THREADS>>>(
                num_tiles,
                (uint32_t*)bucket_counts.data_ptr<int>(),
                (uint32_t*)bucket_offsets.data_ptr<int>(),
                (uint32_t*)bucket_to_tile.data_ptr<int>()
            );

            auto gaussian_ids_sorted_contig = gaussian_ids_sorted.contiguous();
            auto pos2d_contig = pos2d.contiguous();
            auto conics_mu_contig = conics_mu.contiguous();
            auto intensities_contig = intensities.contiguous();
            auto img_grad_contig = img_grad_in.contiguous();

            const int blocks = (total_buckets + warps_per_block - 1) / warps_per_block;

            switch (channels) {
                case 1:
                    rasterize_backward_per_gaussian<1><<<blocks, threads>>>(
                        tile_bounds_dim3,
                        image_size_uint2,
                        (int32_t*)gaussian_ids_sorted_contig.data_ptr<int32_t>(),
                        (int2*)tile_bins_contig.data_ptr<int>(),
                        (float2*)pos2d_contig.data_ptr<float>(),
                        (float4*)conics_mu_contig.data_ptr<float>(),
                        intensities_contig.data_ptr<float>(),
                        img_grad_contig.data_ptr<float>(),
                        (uint32_t*)bucket_offsets.data_ptr<int>(),
                        (uint32_t*)bucket_to_tile.data_ptr<int>(),
                        total_buckets,
                        (float2*)pos2d_grad_out.contiguous().data_ptr<float>(),
                        (float4*)conics_mu_grad_out.contiguous().data_ptr<float>(),
                        intensities_grad_out.contiguous().data_ptr<float>()
                    );
                    break;
                case 2:
                    rasterize_backward_per_gaussian<2><<<blocks, threads>>>(
                        tile_bounds_dim3,
                        image_size_uint2,
                        (int32_t*)gaussian_ids_sorted_contig.data_ptr<int32_t>(),
                        (int2*)tile_bins_contig.data_ptr<int>(),
                        (float2*)pos2d_contig.data_ptr<float>(),
                        (float4*)conics_mu_contig.data_ptr<float>(),
                        intensities_contig.data_ptr<float>(),
                        img_grad_contig.data_ptr<float>(),
                        (uint32_t*)bucket_offsets.data_ptr<int>(),
                        (uint32_t*)bucket_to_tile.data_ptr<int>(),
                        total_buckets,
                        (float2*)pos2d_grad_out.contiguous().data_ptr<float>(),
                        (float4*)conics_mu_grad_out.contiguous().data_ptr<float>(),
                        intensities_grad_out.contiguous().data_ptr<float>()
                    );
                    break;
                case 3:
                    rasterize_backward_per_gaussian<3><<<blocks, threads>>>(
                        tile_bounds_dim3,
                        image_size_uint2,
                        (int32_t*)gaussian_ids_sorted_contig.data_ptr<int32_t>(),
                        (int2*)tile_bins_contig.data_ptr<int>(),
                        (float2*)pos2d_contig.data_ptr<float>(),
                        (float4*)conics_mu_contig.data_ptr<float>(),
                        intensities_contig.data_ptr<float>(),
                        img_grad_contig.data_ptr<float>(),
                        (uint32_t*)bucket_offsets.data_ptr<int>(),
                        (uint32_t*)bucket_to_tile.data_ptr<int>(),
                        total_buckets,
                        (float2*)pos2d_grad_out.contiguous().data_ptr<float>(),
                        (float4*)conics_mu_grad_out.contiguous().data_ptr<float>(),
                        intensities_grad_out.contiguous().data_ptr<float>()
                    );
                    break;
                case 4:
                    rasterize_backward_per_gaussian<4><<<blocks, threads>>>(
                        tile_bounds_dim3,
                        image_size_uint2,
                        (int32_t*)gaussian_ids_sorted_contig.data_ptr<int32_t>(),
                        (int2*)tile_bins_contig.data_ptr<int>(),
                        (float2*)pos2d_contig.data_ptr<float>(),
                        (float4*)conics_mu_contig.data_ptr<float>(),
                        intensities_contig.data_ptr<float>(),
                        img_grad_contig.data_ptr<float>(),
                        (uint32_t*)bucket_offsets.data_ptr<int>(),
                        (uint32_t*)bucket_to_tile.data_ptr<int>(),
                        total_buckets,
                        (float2*)pos2d_grad_out.contiguous().data_ptr<float>(),
                        (float4*)conics_mu_grad_out.contiguous().data_ptr<float>(),
                        intensities_grad_out.contiguous().data_ptr<float>()
                    );
                    break;
                default:
                    throw std::runtime_error("Unsupported number of channels");
            }
        }
    }

    return std::make_tuple(
        pos2d_grad_out,
        conics_mu_grad_out,
        intensities_grad_out
    );
}
