#include "tv3d.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

namespace {

constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 8;
constexpr int BLOCK_Z = 8;
constexpr int BLOCK_SIZE = BLOCK_X * BLOCK_Y * BLOCK_Z;

/**
 * Compute the flattened linear index for a 3D voxel coordinate.
 *
 * Inputs:
 * @param z Depth coordinate.
 * @param y Height coordinate.
 * @param x Width coordinate.
 * @param H Volume height.
 * @param W Volume width.
 *
 * Outputs:
 * @return Row-major index inside a contiguous DHW volume.
 */
__device__ __forceinline__ long long linear_index(
    int z, int y, int x, int H, int W) {
    return (static_cast<long long>(z) * H + y) * W + x;
}

/**
 * Sample a value from the volume with zero padding outside bounds.
 *
 * Inputs:
 * @param ptr Pointer to the base of the DHW volume.
 * @param z Depth coordinate to read.
 * @param y Height coordinate to read.
 * @param x Width coordinate to read.
 * @param D Volume depth.
 * @param H Volume height.
 * @param W Volume width.
 * 
 * Outputs:
 * @return Sampled value or 0 when the index is out of bounds.
 */
__device__ __forceinline__ float global_value(
    const float* __restrict__ ptr,
    int z, int y, int x,
    int D, int H, int W) {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.f;
    }
    return ptr[linear_index(z, y, x, H, W)];
}

/**
 * Compute sign without branching to keep gradients stable.
 *
 * @param v Input value.
 *
 * @return 1 if v>0, -1 if v<0, 0 otherwise.
 */
__device__ __forceinline__ float signed_unit(float v) {
    return (v > 0.f) - (v < 0.f);
}

/**
 * Forward TV3D kernel accumulating partial sums per (B, C) tile.
 *
 * Each block loads a BLOCK_X * BLOCK_Y * BLOCK_Z tile, computes TV
 * contributions along +x/+y/+z neighbors, and writes a partial sum.
 *
 * Inputs:
 * @param volume Pointer to contiguous [bc_count, D, H, W] volume tensor.
 * @param D Volume depth.
 * @param H Volume height.
 * @param W Volume width.
 * @param bc_count Number of batch-channel tiles.
 * @param volume_stride Elements between consecutive batch-channel tiles.
 *
 * Outputs:
 * @param partials Accumulator storing per (B, C) partial TV3D sums.
 */
__global__ void tv3d_forward_kernel(
    const float* __restrict__ volume,
    float* __restrict__ partials,
    int D, int H, int W,
    long long bc_count,
    long long volume_stride) {
    __shared__ float tile[BLOCK_Z][BLOCK_Y][BLOCK_X];
    __shared__ float reduce_buf[BLOCK_SIZE];

    const int tile_origin_x = blockIdx.x * BLOCK_X;
    const int tile_origin_y = blockIdx.y * BLOCK_Y;
    const int tile_origin_z = blockIdx.z * BLOCK_Z;

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int lz = threadIdx.z;
    const int gx = tile_origin_x + lx;
    const int gy = tile_origin_y + ly;
    const int gz = tile_origin_z + lz;
    const bool inside = (gx < W) && (gy < H) && (gz < D);
    const int tid = lz * (BLOCK_X * BLOCK_Y) + ly * BLOCK_X + lx;

    for (long long bc = 0; bc < bc_count; ++bc) {
        const float* volume_ptr = volume + bc * volume_stride;

        float value = 0.f;
        if (inside) {
            value = volume_ptr[linear_index(gz, gy, gx, H, W)];
        }
        tile[lz][ly][lx] = value;
        __syncthreads();

        float contrib = 0.f;
        if (inside) {
            if (gx + 1 < W) {
                float right = (lx + 1 < BLOCK_X)
                    ? tile[lz][ly][lx + 1]
                    : global_value(volume_ptr, gz, gy, gx + 1, D, H, W);
                contrib += fabsf(right - value);
            }
            if (gy + 1 < H) {
                float down = (ly + 1 < BLOCK_Y)
                    ? tile[lz][ly + 1][lx]
                    : global_value(volume_ptr, gz, gy + 1, gx, D, H, W);
                contrib += fabsf(down - value);
            }
            if (gz + 1 < D) {
                float front = (lz + 1 < BLOCK_Z)
                    ? tile[lz + 1][ly][lx]
                    : global_value(volume_ptr, gz + 1, gy, gx, D, H, W);
                contrib += fabsf(front - value);
            }
        }

        reduce_buf[tid] = contrib;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(partials + bc, reduce_buf[0]);
        }
        __syncthreads();
    }
}

/**
 * Backward TV3D kernel computing gradients for each voxel.
 *
 * Inputs:
 * @param volume Pointer to contiguous [bc_count, D, H, W] volume tensor.
 * @param grad_scale Scalar multiplier from upstream grad_output.
 * @param D Volume depth.
 * @param H Volume height.
 * @param W Volume width.
 * @param bc_count Number of batch-channel tiles.
 * @param volume_stride Elements between consecutive tiles.
 *
 * Outputs:
 * @param grad_volume Gradients w.r.t. the original volume values.
 */
__global__ void tv3d_backward_kernel(
    const float* __restrict__ volume,
    float* __restrict__ grad_volume,
    float grad_scale,
    int D, int H, int W,
    long long bc_count,
    long long volume_stride) {
    __shared__ float tile[BLOCK_Z][BLOCK_Y][BLOCK_X];

    const int tile_origin_x = blockIdx.x * BLOCK_X;
    const int tile_origin_y = blockIdx.y * BLOCK_Y;
    const int tile_origin_z = blockIdx.z * BLOCK_Z;

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int lz = threadIdx.z;
    const int gx = tile_origin_x + lx;
    const int gy = tile_origin_y + ly;
    const int gz = tile_origin_z + lz;
    const bool inside = (gx < W) && (gy < H) && (gz < D);

    for (long long bc = 0; bc < bc_count; ++bc) {
        const float* volume_ptr = volume + bc * volume_stride;
        float* grad_ptr = grad_volume + bc * volume_stride;

        float value = 0.f;
        if (inside) {
            value = volume_ptr[linear_index(gz, gy, gx, H, W)];
        }
        tile[lz][ly][lx] = value;
        __syncthreads();

        if (inside) {
            float grad = 0.f;

            if (gx > 0) {
                float neighbor = (lx > 0)
                    ? tile[lz][ly][lx - 1]
                    : global_value(volume_ptr, gz, gy, gx - 1, D, H, W);
                grad += signed_unit(value - neighbor);
            }
            if (gx + 1 < W) {
                float neighbor = (lx + 1 < BLOCK_X)
                    ? tile[lz][ly][lx + 1]
                    : global_value(volume_ptr, gz, gy, gx + 1, D, H, W);
                grad -= signed_unit(neighbor - value);
            }

            if (gy > 0) {
                float neighbor = (ly > 0)
                    ? tile[lz][ly - 1][lx]
                    : global_value(volume_ptr, gz, gy - 1, gx, D, H, W);
                grad += signed_unit(value - neighbor);
            }
            if (gy + 1 < H) {
                float neighbor = (ly + 1 < BLOCK_Y)
                    ? tile[lz][ly + 1][lx]
                    : global_value(volume_ptr, gz, gy + 1, gx, D, H, W);
                grad -= signed_unit(neighbor - value);
            }

            if (gz > 0) {
                float neighbor = (lz > 0)
                    ? tile[lz - 1][ly][lx]
                    : global_value(volume_ptr, gz - 1, gy, gx, D, H, W);
                grad += signed_unit(value - neighbor);
            }
            if (gz + 1 < D) {
                float neighbor = (lz + 1 < BLOCK_Z)
                    ? tile[lz + 1][ly][lx]
                    : global_value(volume_ptr, gz + 1, gy, gx, D, H, W);
                grad -= signed_unit(neighbor - value);
            }

            grad_ptr[linear_index(gz, gy, gx, H, W)] = grad * grad_scale;
        }
        __syncthreads();
    }
}

} // namespace

/**
 * Launch the CUDA forward kernel and return the scalar TV3D loss.
 */
torch::Tensor tv3d_forward(torch::Tensor volume) {
    TORCH_CHECK(volume.is_cuda(), "tv3d_forward expects a CUDA tensor");
    TORCH_CHECK(volume.scalar_type() == torch::kFloat32,
                "tv3d_forward currently supports float32 inputs");
    TORCH_CHECK(volume.dim() == 5,
                "tv3d_forward expects input shaped [B, C, D, H, W]");

    auto volume_contig = volume.contiguous();

    const auto B = volume_contig.size(0);
    const auto C = volume_contig.size(1);
    const auto D = static_cast<int>(volume_contig.size(2));
    const auto H = static_cast<int>(volume_contig.size(3));
    const auto W = static_cast<int>(volume_contig.size(4));

    TORCH_CHECK(D > 0 && H > 0 && W > 0, "Input spatial dimensions must be > 0");

    const long long bc_count =
        static_cast<long long>(B) * static_cast<long long>(C);
    const long long volume_stride =
        static_cast<long long>(D) * H * W;

    c10::cuda::CUDAGuard device_guard(volume_contig.device());

    auto partials = torch::zeros({bc_count}, volume_contig.options());

    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (W + BLOCK_X - 1) / BLOCK_X,
        (H + BLOCK_Y - 1) / BLOCK_Y,
        (D + BLOCK_Z - 1) / BLOCK_Z
    );

    auto stream = at::cuda::getCurrentCUDAStream();
    tv3d_forward_kernel<<<grid, block, 0, stream>>>(
        volume_contig.data_ptr<float>(),
        partials.data_ptr<float>(),
        D, H, W,
        bc_count,
        volume_stride
    );
    AT_CUDA_CHECK(cudaGetLastError());

    return partials.sum();
}

/**
 * Launch the CUDA backward kernel to compute gradients of TV3D loss.
 */
torch::Tensor tv3d_backward(torch::Tensor volume, torch::Tensor grad_output) {
    TORCH_CHECK(volume.is_cuda(), "tv3d_backward expects a CUDA tensor");
    TORCH_CHECK(volume.scalar_type() == torch::kFloat32,
                "tv3d_backward currently supports float32 inputs");
    TORCH_CHECK(volume.dim() == 5,
                "tv3d_backward expects input shaped [B, C, D, H, W]");
    TORCH_CHECK(grad_output.is_cuda(),
                "tv3d_backward grad_output must live on CUDA");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32,
                "tv3d_backward grad_output must be float32");
    TORCH_CHECK(grad_output.numel() == 1,
                "tv3d_backward expects a scalar grad_output");
    TORCH_CHECK(grad_output.device() == volume.device(),
                "grad_output must be on the same device as input volume");

    auto volume_contig = volume.contiguous();

    const auto B = volume_contig.size(0);
    const auto C = volume_contig.size(1);
    const auto D = static_cast<int>(volume_contig.size(2));
    const auto H = static_cast<int>(volume_contig.size(3));
    const auto W = static_cast<int>(volume_contig.size(4));

    const long long bc_count =
        static_cast<long long>(B) * static_cast<long long>(C);
    const long long volume_stride =
        static_cast<long long>(D) * H * W;

    c10::cuda::CUDAGuard device_guard(volume_contig.device());

    auto grad = torch::empty_like(volume_contig);
    const float grad_scale = grad_output.item<float>();

    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (W + BLOCK_X - 1) / BLOCK_X,
        (H + BLOCK_Y - 1) / BLOCK_Y,
        (D + BLOCK_Z - 1) / BLOCK_Z
    );

    auto stream = at::cuda::getCurrentCUDAStream();
    tv3d_backward_kernel<<<grid, block, 0, stream>>>(
        volume_contig.data_ptr<float>(),
        grad.data_ptr<float>(),
        grad_scale,
        D, H, W,
        bc_count,
        volume_stride
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return grad;
}
