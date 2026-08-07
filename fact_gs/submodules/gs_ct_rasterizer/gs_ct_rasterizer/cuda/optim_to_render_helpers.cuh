
#include "config.h"
#include "stdio.h"
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


/**
 * Given a point and a maximum radius, compute the bounding rectangle of tiles that the point can affect.
 * 
 * Input:
 * @param p 2D position of the point.
 * @param max_radius Maximum radius of the point's influence in x and y directions.
 * @param grid Grid dimensions (number of tiles in x and y directions).
 * Output:
 * @param rect_min Minimum tile indices (x_min, y_min) that the point can affect.
 * @param rect_max Maximum tile indices (x_max, y_max) that the point
 */
__forceinline__ __device__ void getRect(
    const float2 p,
    const float2 max_radius,
	dim3 grid,
    int2& rect_min,
    int2& rect_max
)
{
	const int grid_x = static_cast<int>(grid.x);
	const int grid_y = static_cast<int>(grid.y);
	rect_min = {
		min(grid_x, max(0, static_cast<int>((p.x - max_radius.x) / BLOCK_X))),
		min(grid_y, max(0, static_cast<int>((p.y - max_radius.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid_x, max(0, static_cast<int>((p.x + max_radius.x + BLOCK_X - 1) / BLOCK_X))),
		min(grid_y, max(0, static_cast<int>((p.y + max_radius.y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

/**
 * Apply a 4x3 or 4x4 transformation matrix to a point or vector.
 * 
 * Input:
 * @param p 3D point or vector to be transformed.
 * @param matrix 4x3 or 4x4 transformation matrix in column-major order.
 * 
 * Output:
 * Transformed 3D point or vector.
 */

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

/**
 * Apply a 4x4 transformation matrix to a point.
 * 
 * Input:
 * @param p 3D point to be transformed.
 * @param matrix 4x4 transformation matrix in column-major order.
 * 
 * Output:
 * Transformed 4D point.
 */
__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

/**
 * Apply a 4x3 transformation matrix to a vector.
 * 
 * Input:
 * @param p 3D vector to be transformed.
 * @param matrix 4x3 transformation matrix in column-major order.
 * 
 * Output:
 * Transformed 3D vector.
 */
__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

/**
 * Apply the transpose of a 4x3 transformation matrix to a vector.
 * 
 * Input:
 * @param p 3D vector to be transformed.
 * @param matrix 4x3 transformation matrix in column-major order.
 * 
 * Output:
 * Transformed 3D vector.
 */
__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

/**
 * Compute the z derivative of the normalized float3 vector v with respect to changes in v (dv).
 * 
 * Input:
 * @param v The original vector that is being normalized.
 * @param dv The change in the original vector v.
 * 
 * Output:
 * The derivative of the normalized vector with respect to changes in v.
 */
__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

/**
 * Compute the derivative of the normalized float3 vector v with respect to changes in v (dv) for all components.
 * 
 * Input:
 * @param v The original vector that is being normalized.
 * @param dv The change in the original vector v.
 * 
 * Output:
 * The derivative of the normalized vector with respect to changes in v for all components.
 */
__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

/**
 * Compute the derivative of the normalized float4 vector v with respect to changes in v (dv) for all components.
 * 
 * Input:
 * @param v The original vector that is being normalized.
 * @param dv The change in the original vector v.
 * 
 * Output:
 * The derivative of the normalized vector with respect to changes in v for all components.
 */
__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

/**
 * Compute the sigmoid function of x.
 * 
 * Input:
 * @param x The input value for which to compute the sigmoid function.
 * 
 * Output:
 * The sigmoid of x, defined as 1 / (1 + exp(-x)).
 */
__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

/**
 * Convert quaternion to rotation matrix.
 * 
 * Input: 
 * @param quat quaternion (w,x,y,z)
 * 
 * @return Output: rotation matrix (column-major)
 */
inline __device__ glm::mat3 quat_to_rotmat(const float4 quat) {
    
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

	// Baseline doesn't normalize quaternions, use below for complete equivalence
	// float w = quat.x;
    // float x = quat.y;
    // float y = quat.z;
    // float z = quat.w;

    // glm matrices are column-major
    return glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z), 2.f * (x * z + w * y), 
		2.f * (x * y + w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x),
        2.f * (x * z - w * y), 2.f * (y * z + w * x), 1.f - 2.f * (x * x + y * y)
    );
}
