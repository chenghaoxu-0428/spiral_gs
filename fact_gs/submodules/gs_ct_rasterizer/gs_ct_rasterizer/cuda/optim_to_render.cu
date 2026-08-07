#include "optim_to_render.cuh"
#include "optim_to_render_helpers.cuh"
#include <cooperative_groups.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace cg = cooperative_groups;

/**
 * Forward version of 2D covariance matrix computation
 * Section 6.2 EWA Volume Resampling Filter
 * Converts gaussian parameters from 3D optimization format to 2D rasterization format.
 * Each thread processes one gaussian.
 * 
 * Inputs:
 * @param mean: 3D position of the gaussian in world space
 * @param focal_x, focal_y: focal length in x and y directions
 * @param tan_fovx, tan_fovy: Tangents of the field of view angles in x and y.
 * @param cov3D: 3D covariance matrix of the gaussian in world space.
 * @param viewmatrix: View matrix.
 * @param mode: 0: parallel beam, 1: cone beam
 * 
 * Outputs:
 * @return float4 containing the 2D covariance matrix entries (cov_xx, cov_xy, cov_yy) and integration factor mu.
 */
__device__ float4 computeCov2D_forward(
    const float3& mean, 
    float focal_x, 
    float focal_y,
    float tan_fovx, 
    float tan_fovy, 
    const glm::mat3 cov3D, 
    const float* viewmatrix, 
    const int mode)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.

	//! This is the transformation from Source (world) to view (camera)
	float3 t = transformPoint4x3(mean, viewmatrix);
    glm::mat3 J;
	if (mode == 0){  // parallel beam
		const float limx = 1.3f;
		const float limy = 1.3f;
		t.x = min(limx, max(-limx, t.x));
		t.y = min(limx, max(-limx, t.y));

		// J is eye
		J = glm::mat3(
		focal_x, 0.0f, 0.0f,
		0.0f, focal_y, 0.0f,
		0.0f, 0.0f, 1.0f);
	}
	else  // cone beam
	{	
		const float limx = 1.3f * tan_fovx;
		const float limy = 1.3f * tan_fovy;
		const float txtz = t.x / t.z;
		const float tytz = t.y / t.z;
		// t = Phi^-1(x), eq(26)
		t.x = min(limx, max(-limx, txtz)) * t.z;
		t.y = min(limy, max(-limy, tytz)) * t.z;
		// Jacobian of Affine approximation of projection transformation
		// eq(29)
		const float l = sqrt(t.x * t.x +  t.y * t.y + t.z * t.z);
		J = glm::mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			t.x / l, t.y / l, t.z / l);  //! We add the third row for further computation.
	}
	
	// Viewing transformation
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	
	glm::mat3 M = W * J;
    
	// J W Sigma W^M J^M
	glm::mat3 cov = glm::transpose(M) * glm::transpose(cov3D) * M;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	//! We do not add low pass filter.
	cov[0][0] += 0.0f;
	cov[1][1] += 0.0f;

	//! Compute integration bias factor mu (Eq. 7 in our paper) 
	//! Check issue #4 regarding dicussion of ray-space and world-space scales.
	float hata = cov[0][0];
	float hatb = cov[0][1];
	float hatc = cov[0][2];
	float hatd = cov[1][1];
	float hate = cov[1][2];
	float hatf = cov[2][2];
	float diamond = hata * hatd - hatb * hatb;
	float circ = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	float mu_square = 2 * M_PI * circ / diamond;
	float mu = 0.0f;
	if (mu_square > 0.0f){
		mu = sqrt(2 * M_PI * circ / diamond); 
	}

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]), float(mu) };
}

/**
 * Converts gaussian parameters from 3D optimization format to 2D rasterization format.
 * Position from [0,1] to pixel-based, scale and quat to conics
 * Prepares radii for assigning gaussians to volume tiles.
 * Each thread processes one gaussian.
 */
template<int CHANNELS> __global__ void optim_to_render_forward(
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
)
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
	radii_out[idx] = 0.0f; // Mark as invalid unless proven otherwise
	num_tiles_hit_out[idx] = 0;
	tile_min_out[idx] = make_int2(0, 0);
	tile_max_out[idx] = make_int2(0, 0);

    // Compute center in pixel coordinates
	float3 p_orig = pos3d[idx];

    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	// In-frustum check
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
    if (p_view.z <= 0.2f){
	    return;
	}
    //Scale matrix
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = scale3d[idx].x;
    S[1][1] = scale3d[idx].y;
    S[2][2] = scale3d[idx].z;

    // Rotation matrix from quaternion
    glm::mat3 R = quat_to_rotmat(quat[idx]);

    // 3D Covariance matrix
    glm::mat3 M = S * R;
    glm::mat3 Mt = glm::transpose(M);
    glm::mat3 cov3d = Mt * M;
    
    // 2D Covariance matrix
    float4 cov2d = computeCov2D_forward(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3d, viewmatrix, mode);

    // Invert covariance (EWA algorithm)
	float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
	if (det == 0.0f)
		return;
    float det_inv = 1.f / det;
	float4 conic_mu = { cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv , cov2d.w };

    // Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix).
	float mid = 0.5f * (cov2d.x + cov2d.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det)); 
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// Keep tile assignment independent of the current density.  The previous
	// density-derived bound could collapse to zero/NaN for weak Gaussians and
	// permanently remove the very gradients needed to grow them.  A 3-sigma
	// covariance bound matches the legacy rasterizer's support; half a pixel is
	// added because tiles are selected from pixel-centre coordinates.
	float2 radius2 = make_float2(
		max(0.0f, 3.0f * sqrtf(max(0.0f, cov2d.x)) + 0.5f),
		max(0.0f, 3.0f * sqrtf(max(0.0f, cov2d.z)) + 0.5f));

	int2 rect_min, rect_max;
	float2 pos2d = {
        ((p_proj.x + 1.0f) * image_size.x - 1.0f) * 0.5f,
        ((p_proj.y + 1.0f) * image_size.y - 1.0f) * 0.5f
    };
	
	getRect(pos2d, radius2, grid, rect_min, rect_max);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

    // Write outputs
	num_tiles_hit_out[idx] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
	tile_min_out[idx] = rect_min;
	tile_max_out[idx] = rect_max;
    pos2d_out[idx] = pos2d;
    conics_mu_out[idx] = conic_mu;
    radii_out[idx] = radius;
}

// Backward version of INVERSE 2D covariance matrix computation
__device__ void computeCov2D_backward(
	const float3 pos3d,
	const glm::mat3 cov3d, 
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float4 conics_mu_grad_in,
	float3& pos3d_grad_out,
	glm::mat3& cov3d_grad_out,
	const int mode)
{
	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 t = transformPoint4x3(pos3d, view_matrix);

	glm::mat3 J;
	float x_grad_mul, y_grad_mul;
	if (mode == 0){  //! parallel beam
		const float limx = 1.3f;
		const float limy = 1.3f;
		t.x = min(limx, max(-limx, t.x));
		t.y = min(limx, max(-limx, t.y));
		
		x_grad_mul = t.x < -limx || t.x > limx ? 0 : 1;
		y_grad_mul = t.y < -limy || t.y > limy ? 0 : 1;

		J = glm::mat3(
		focal_x, 0.0f, 0.0f,
		0.0f, focal_y, 0.0f,
		0, 0, 1.0f);
	}
	else  //! cone beam
	{
		const float limx = 1.3f * tan_fovx;
		const float limy = 1.3f * tan_fovy;
		const float txtz = t.x / t.z;
		const float tytz = t.y / t.z;
		t.x = min(limx, max(-limx, txtz)) * t.z;
		t.y = min(limy, max(-limy, tytz)) * t.z;
		
		x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
		y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

		const float l = sqrt(t.x * t.x +  t.y * t.y + t.z * t.z);
		J = glm::mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			t.x / l, t.y / l, t.z / l);  //! We need last row for computation.
	}

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 M = W * J;

	glm::mat3 cov = glm::transpose(M) * glm::transpose(cov3d) * M;
	// Use helper variables for 2D covariance entries. More compact.
	float hata = cov[0][0] += 0.0f;
	float hatb = cov[0][1];
	float hatc = cov[0][2];
	float hatd = cov[1][1] += 0.0f;
	float hate = cov[1][2];
	float hatf = cov[2][2];

	float dL_dhata = 0, dL_dhatb = 0, dL_dhatc = 0, dL_dhatd = 0, dL_dhate = 0, dL_dhatf = 0;
	float denom = hata * hatd - hatb * hatb;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
	float diamond = hata * hatd - hatb * hatb;

	//! mu part gradient
	float circ = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	float mu_square = 2 * M_PI * circ / diamond;
	float mu = 0.0f;
	if (mu_square > 0.0f){
		mu = sqrt(2 * M_PI * circ / diamond);
	}
	float pi_mu= M_PI / (mu + 0.0000001f);
	float circ_diamond = circ / diamond;

	glm::mat3 cov2d_grad(0.0f);
	if (denom2inv != 0.0f && mu != 0.0f)
	{
		// exp(*) part gradient
		dL_dhata = denom2inv * (-hatd * hatd * conics_mu_grad_in.x + hatb * hatd * conics_mu_grad_in.y + (denom - hata * hatd) * conics_mu_grad_in.z);  // We remove 2 here because in render we do not *0.5
		dL_dhatd = denom2inv * (-hata * hata * conics_mu_grad_in.z + hata * hatb * conics_mu_grad_in.y + (denom - hata * hatd) * conics_mu_grad_in.x);  // We remove 2 here because in render we do not *0.5
		dL_dhatb = denom2inv * (2 * hatb * hatd * conics_mu_grad_in.x - (denom + 2 * hatb * hatb) * conics_mu_grad_in.y + 2 * hata * hatb * conics_mu_grad_in.z);

		dL_dhata += pi_mu * ((hatd * hatf - hate * hate) / diamond -  hatd * circ_diamond / diamond) * conics_mu_grad_in.w;
		dL_dhatb += pi_mu * ((2 * hatc * hate - 2 * hatf * hatb) / diamond + 2 * hatb * circ_diamond / diamond) * conics_mu_grad_in.w;
		dL_dhatc += pi_mu * ((2 * hatb * hate - 2 * hatd * hatc) / diamond) * conics_mu_grad_in.w;
		dL_dhatd += pi_mu * ((hata * hatf - hatc * hatc) / diamond -  hata *circ_diamond / diamond) * conics_mu_grad_in.w;
		dL_dhate += pi_mu * ((2 * hatb * hatc - 2 * hata * hate) / diamond) * conics_mu_grad_in.w;
		dL_dhatf += pi_mu * ((hata * hatd - hatb * hatb) / diamond) * conics_mu_grad_in.w;
		cov2d_grad = glm::mat3(
			dL_dhata, 0.5f * dL_dhatb, 0.5f * dL_dhatc,
			0.5f * dL_dhatb, dL_dhatd, 0.5f * dL_dhate,
			0.5f * dL_dhatc, 0.5f * dL_dhate, dL_dhatf);
		
		cov3d_grad_out =  M * cov2d_grad * glm::transpose(M);
	}

	if (mode == 1){
		// Gradients of loss w.r.t. M
		// glm::mat3 cov3d_T = glm::transpose(cov3d);        // equals cov3d as it is symmetric
		glm::mat3 dL_dM = 2.0f * cov3d * M * cov2d_grad;

		// Gradients of loss w.r.t. J
		glm::mat3 Wt = glm::transpose(W);
		glm::mat3 dL_dJ = Wt * dL_dM;

		float tx = t.x;
		float ty = t.y;
		float tz = t.z;
		float inv_tz = 1.f / tz;
		float inv_tz2 = inv_tz * inv_tz;
		float inv_tz3 = inv_tz2 * inv_tz;
		float circledcirc = sqrt(tx * tx + ty * ty + tz * tz);
		float inv_circledcirc3 = 1 / (circledcirc * circledcirc * circledcirc);
		float dL_dtx = x_grad_mul * (-focal_x*inv_tz2*dL_dJ[0][2] + (1/circledcirc - tx*tx*inv_circledcirc3)*dL_dJ[2][0] - tx*ty*inv_circledcirc3*dL_dJ[2][1] - tx*tz*inv_circledcirc3*dL_dJ[2][2]);
		float dL_dty = y_grad_mul * (-focal_y*inv_tz2*dL_dJ[1][2] - tx*ty*inv_circledcirc3*dL_dJ[2][0] + (1/circledcirc - ty*ty*inv_circledcirc3)*dL_dJ[2][1] - ty*tz*inv_circledcirc3*dL_dJ[2][2]);
		float dL_dtz = -focal_x*inv_tz2*dL_dJ[0][0] + 2*focal_x*tx*inv_tz3*dL_dJ[0][2] - focal_y*inv_tz2*dL_dJ[1][1] + 2*focal_y*ty*inv_tz3*dL_dJ[1][2] - tx*tz*inv_circledcirc3*dL_dJ[2][0] - ty*tz*inv_circledcirc3*dL_dJ[2][1] + (1/circledcirc-tz*tz*inv_circledcirc3)*dL_dJ[2][2];

		pos3d_grad_out = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	}
	else{
		// Parallel beam case: J is constant, so no further gradients to compute.
		pos3d_grad_out = {0.0f, 0.0f, 0.0f};
	}
}


/**
 * Backward pass for optim_to_render_forward. 
 * Computes gradients w.r.t. input parameters (pos3d, scale3d, quat).
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
)
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_gaussians || !(radii[idx] > 0)) return;

	// Recompute 3D covariance matrix from scale and quaternion
	// Scale matrix
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = scale3d[idx].x;
    S[1][1] = scale3d[idx].y;
    S[2][2] = scale3d[idx].z;

    // Rotation matrix from quaternion
    glm::mat3 R = quat_to_rotmat(quat[idx]);

    // 3D Covariance matrix
    glm::mat3 M = S * R;
    glm::mat3 Mt = glm::transpose(M);
    glm::mat3 cov3d = Mt * M;

	glm::mat3 cov3d_grad = glm::mat3(0.0f);

	computeCov2D_backward(
		pos3d[idx],
		cov3d,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		viewmatrix,
		conics_mu_grad_in[idx],
		pos3d_grad_out[idx],
		cov3d_grad,
		mode);
	
	// Taking care of gradients from the screenspace points
	float4 pos3d_hom = transformPoint4x4(pos3d[idx], projmatrix);
	float pos3d_w = 1.0f / (pos3d_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	float3 pos3d_grad_temp;
	float mul1 = (projmatrix[0] * pos3d[idx].x + projmatrix[4] * pos3d[idx].y + projmatrix[8] * pos3d[idx].z + projmatrix[12]) * pos3d_w * pos3d_w;
	float mul2 = (projmatrix[1] * pos3d[idx].x + projmatrix[5] * pos3d[idx].y + projmatrix[9] * pos3d[idx].z + projmatrix[13]) * pos3d_w * pos3d_w;
	pos3d_grad_temp.x = (projmatrix[0] * pos3d_w - projmatrix[3] * mul1) * pos2d_grad_in[idx].x + (projmatrix[1] * pos3d_w - projmatrix[3] * mul2) * pos2d_grad_in[idx].y;
	pos3d_grad_temp.y = (projmatrix[4] * pos3d_w - projmatrix[7] * mul1) * pos2d_grad_in[idx].x + (projmatrix[5] * pos3d_w - projmatrix[7] * mul2) * pos2d_grad_in[idx].y;
	pos3d_grad_temp.z = (projmatrix[8] * pos3d_w - projmatrix[11] * mul1) * pos2d_grad_in[idx].x + (projmatrix[9] * pos3d_w - projmatrix[11] * mul2) * pos2d_grad_in[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D also affects it.
	pos3d_grad_out[idx].x += pos3d_grad_temp.x;
	pos3d_grad_out[idx].y += pos3d_grad_temp.y;
	pos3d_grad_out[idx].z += pos3d_grad_temp.z;

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scale3d){
		float s = rsqrtf(
        	quat[idx].w * quat[idx].w + quat[idx].x * quat[idx].x + quat[idx].y * quat[idx].y + quat[idx].z * quat[idx].z
    	);
		float r = quat[idx].x * s;
		float x = quat[idx].y * s;
		float y = quat[idx].z * s;
		float z = quat[idx].w * s;
		// Convert per-element covariance loss gradients to matrix form
		glm::mat3 dL_dSigma = cov3d_grad;

		// Compute loss gradient w.r.t. matrix M
		glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

		glm::mat3 Rt = glm::transpose(R);
		glm::mat3 dL_dMt = glm::transpose(dL_dM);

		// Gradients of loss w.r.t. scale
		scale3d_grad_out[idx].x = glm::dot(Rt[0], dL_dMt[0]);
		scale3d_grad_out[idx].y = glm::dot(Rt[1], dL_dMt[1]);
		scale3d_grad_out[idx].z = glm::dot(Rt[2], dL_dMt[2]);

		dL_dMt[0] *= scale3d[idx].x;
		dL_dMt[1] *= scale3d[idx].y;
		dL_dMt[2] *= scale3d[idx].z;
		// Gradients of loss w.r.t. normalized quaternion
		quat_grad_out[idx].x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
		quat_grad_out[idx].y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
		quat_grad_out[idx].z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
		quat_grad_out[idx].w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	}

}

template __global__ void optim_to_render_forward<1>(const int, const float3*, const float3*, const float4*, const float*, const float*, const float*,
	const float, float, const float, const float, const uint2, const dim3, const int, float2*, float4*, float*, int32_t*, int2*, int2*);

template __global__ void optim_to_render_forward<2>(const int, const float3*, const float3*, const float4*, const float*, const float*, const float*,
	const float, float, const float, const float, const uint2, const dim3, const int, float2*, float4*, float*, int32_t*, int2*, int2*);

template __global__ void optim_to_render_forward<3>(const int, const float3*, const float3*, const float4*, const float*, const float*, const float*,
	const float, float, const float, const float, const uint2, const dim3, const int, float2*, float4*, float*, int32_t*, int2*, int2*);

template __global__ void optim_to_render_forward<4>(const int, const float3*, const float3*, const float4*, const float*, const float*, const float*,
	const float, float, const float, const float, const uint2, const dim3, const int, float2*, float4*, float*, int32_t*, int2*, int2*);
