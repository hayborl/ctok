#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "emicp.h"

struct updateA_functor
{
	float R[9], T[3], sigma;
	updateA_functor(const Mat& m_R, const Mat& m_T, float _sigma)
	{
		sigma = _sigma;
		memcpy(R, (float*)m_R.data, 9 * sizeof(float));
		memcpy(T, (float*)m_T.data, 3 * sizeof(float));
	}
	__host__ __device__ float operator()(const float3& p1, const float3& p2)
	{
		float tmp0, tmp1, tmp2;
		tmp0 = p1.x - (R[0] * p2.x + R[1] * p2.y + R[2] * p2.z + T[0]);
		tmp1 = p1.y - (R[3] * p2.x + R[4] * p2.y + R[5] * p2.z + T[1]);
		tmp2 = p1.z - (R[6] * p2.x + R[7] * p2.y + R[8] * p2.z + T[2]);
		tmp0 = tmp0 * tmp0 + tmp1 * tmp1 + tmp2 * tmp2;
		tmp0 /= sigma;
		return expf(-tmp0);
	}
};

void EMICP::updateA(Mat &A, const Mat &objSet, 
	const Mat &R, const Mat &T, bool withCuda)
{
	int rowsA = objSet.rows;
	int colsA = m_modSet.rows;
	
	float3* arr_obj = new float3[rowsA];
	memcpy(arr_obj, (float3*)objSet.data, rowsA * sizeof(float3));
	float3* arr_mod = new float3[colsA];
	memcpy(arr_mod, (float3*)m_modSet.data, colsA * sizeof(float3));

	try
	{
		thrust::host_vector<float> h_A;

		if (withCuda)
		{
			thrust::device_vector<float> d_A(rowsA * colsA);
			thrust::device_vector<float3> d_obj(arr_obj, arr_obj + rowsA);
			thrust::device_vector<float3> d_mod(arr_mod, arr_mod + colsA);

			for (int i = 0; i < rowsA; i++)
			{
				thrust::constant_iterator<float3> tmp(d_obj[i]);
				thrust::transform(d_mod.begin(), d_mod.end(), tmp, 
					d_A.begin() + i * colsA, updateA_functor(R, T, m_sigma_p2));
			}

			h_A = d_A;
		}
		else
		{
			h_A = thrust::host_vector<float>(rowsA * colsA);
			thrust::host_vector<float3> h_obj(arr_obj, arr_obj + rowsA);
			thrust::host_vector<float3> h_mod(arr_mod, arr_mod + colsA);

			for (int i = 0; i < rowsA; i++)
			{
				thrust::constant_iterator<float3> tmp(h_obj[i]);
				thrust::transform(h_mod.begin(), h_mod.end(), tmp, 
					h_A.begin() + i * colsA, updateA_functor(R, T, m_sigma_p2));
			}
		}

		float* h_A_ptr = thrust::raw_pointer_cast(&h_A[0]);
		A = Mat(rowsA, colsA, CV_32FC1);
		memcpy((float*)A.data, h_A_ptr, rowsA * colsA * sizeof(float));
	}
	catch (thrust::system_error e)
	{
		cout << "System Error: " << e.what() << endl;
	}

	delete[] arr_obj;
	delete[] arr_mod;
}

struct normalizeRow_functor
{
	int cols;
	normalizeRow_functor(const int& _cols) : cols(_cols){}
	__host__ __device__ float operator()(const float& x, const float& lambda)
	{
		if (lambda > 10e-7f)
		{
			return x / lambda;
		}
		else
		{
			return 1.0f / cols;
		}
	}
};

void EMICP::normalizeRows(Mat &mat, const Mat &alpha, bool withCuda)
{
	int rows = mat.rows;
	int cols = mat.cols;
	float* arr_mat = new float[rows * cols];
	memcpy(arr_mat, (float*)mat.data, rows * cols * sizeof(float));
	float* arr_alpha = new float[rows];
	memcpy(arr_alpha, (float*)alpha.data, rows * sizeof(float));

	try
	{
		thrust::host_vector<float> h_mat(rows * cols);

		if (withCuda)
		{
			thrust::device_vector<float> d_mat(arr_mat, arr_mat + rows * cols);
			thrust::device_vector<float> d_alpha(arr_alpha, arr_alpha + rows);

			for (int i = 0; i < rows; i++)
			{
				thrust::constant_iterator<float> tmp(d_alpha[i]);
				thrust::device_vector<float>::iterator 
					begin = d_mat.begin() + i * cols;
				thrust::transform(begin, begin + cols, 
					tmp, begin, normalizeRow_functor(cols));
			}

			h_mat = d_mat;
		}
		else
		{
			thrust::copy(arr_mat, arr_mat + rows * cols, h_mat.begin());
			thrust::host_vector<float> h_alpha(arr_alpha, arr_alpha + rows);

			for (int i = 0; i < rows; i++)
			{
				thrust::constant_iterator<float> tmp(h_alpha[i]);
				thrust::host_vector<float>::iterator 
					begin = h_mat.begin() + i * cols;
				thrust::transform(begin, begin + cols, 
					tmp, begin, normalizeRow_functor(cols));
			}
		}

		float* h_mat_ptr = thrust::raw_pointer_cast(&h_mat[0]);
		memcpy((float*)mat.data, h_mat_ptr, rows * cols * sizeof(float));
	}
	catch (thrust::system_error e)
	{
		cout << "System Error: " << e.what() << endl;
	}

	delete[] arr_mat;
	delete[] arr_alpha;
}

#define BLOCK_SIZE 128

__global__ void kernelUpdateA(PtrStepSz<float> d_mod, PtrStepSz<float> d_obj,
	PtrStepSz<float> d_R, PtrStepSz<float> d_T, 
	PtrStepSz<float> d_A, float sigma_p2)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;

#define arrR(r, c) d_R.ptr(r)[c]
#define arrT(r) d_T.ptr(r)[0]
#define mod(r, c) d_mod.ptr(r)[c]
#define obj(r, c) d_obj.ptr(r)[c]
#define A(r, c) d_A.ptr(r)[c]

	if (r < d_obj.rows && c < d_mod.rows)
	{
		float tmp[3];
		for (int i = 0; i < 3; i++)
		{
			tmp[i] = mod(c, i) - (arrR(i, 0) * obj(r, 0) 
				+ arrR(i, 1) * obj(r, 1) 
				+ arrR(i, 2) * obj(r, 2) + arrT(i));
			tmp[i] *= tmp[i];
		}
		tmp[0] += tmp[1];
		tmp[0] += tmp[2];
		tmp[0] /= sigma_p2;
		tmp[0] = expf(-tmp[0]);

		A(r, c) = tmp[0];
	}
}

void EMICP::cuda_updateA(Mat &h_A, const Mat &objSet, 
	const Mat &modSet, const Mat &h_R, const Mat &h_T)
{
	assert(objSet.cols == 3 && modSet.cols == 3);

	int rowsA = objSet.rows;
	int colsA = modSet.rows;
	GpuMat d_obj, d_mod, d_R, d_T;
	d_obj.upload(objSet);
	d_mod.upload(modSet);
	d_R.upload(h_R);
	d_T.upload(h_T);

	int rowsA1 = rowsA - 1;
	int colsA1 = colsA - 1;
	dim3 dimBlockForA(rowsA1 % BLOCK_SIZE, colsA1 % BLOCK_SIZE);
	dim3 dimGridForA(rowsA1 / dimBlockForA.x + 1, colsA1 / dimBlockForA.y + 1);

	GpuMat d_A(rowsA, colsA, CV_32FC1);
	kernelUpdateA<<<dimGridForA, dimBlockForA>>>
		(d_mod, d_obj, d_R, d_T, d_A, m_sigma_p2);

	d_A.download(h_A);
	cout << h_A << endl;waitKey();

	d_obj.release();
	d_mod.release();
	d_R.release();
	d_T.release();
	d_A.release();
}