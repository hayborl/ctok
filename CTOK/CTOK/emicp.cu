#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "emicp.h"

struct updateA_functor
{
	double sigma;
	updateA_functor(double _sigma)
	{
		sigma = _sigma;
	}
	__host__ __device__ double operator()(const double3 &p1, const double3 &p2)
	{
		double tmp0, tmp1, tmp2;
		tmp0 = p1.x - p2.x;
		tmp1 = p1.y - p2.y;
		tmp2 = p1.z - p2.z;
		tmp0 = tmp0 * tmp0 + tmp1 * tmp1 + tmp2 * tmp2;
		tmp0 /= sigma;
		return exp(-tmp0);
	}
};

void EMICP::updateA( Mat &A, const Mat &objSet, 
	const Mat &modSet, bool withCuda )
{
	
	if (withCuda)
	{
		Mat tmpObj = convertMat(objSet);
		Mat tmpMod = convertMat(modSet);
		cuda_updateA(A, tmpObj, tmpMod);
	}
	else
	{
		int rowsA = objSet.rows;
		int colsA = modSet.rows;

		double3* arr_obj = new double3[rowsA];
		memcpy(arr_obj, (double3*)objSet.data, rowsA * sizeof(double3));
		double3* arr_mod = new double3[colsA];
		memcpy(arr_mod, (double3*)modSet.data, colsA * sizeof(double3));

		try
		{
			thrust::host_vector<double> h_A = 
				thrust::host_vector<double>(rowsA * colsA);
			thrust::host_vector<double3> h_obj(arr_obj, arr_obj + rowsA);
			thrust::host_vector<double3> h_mod(arr_mod, arr_mod + colsA);

			for (int i = 0; i < rowsA; i++)
			{
				thrust::constant_iterator<double3> tmp(h_obj[i]);
				thrust::transform(h_mod.begin(), h_mod.end(), tmp, 
					h_A.begin() + i * colsA, updateA_functor(m_sigma_p2));
			}

			double* h_A_ptr = thrust::raw_pointer_cast(&h_A[0]);
			A = Mat(rowsA, colsA, CV_64FC1);
			memcpy((double*)A.data, h_A_ptr, rowsA * colsA * sizeof(double));
		}
		catch (thrust::system_error e)
		{
			cout << "System Error: " << e.what() << endl;
		}

		delete[] arr_obj;
		delete[] arr_mod;
	}
}

struct normalizeRow_functor
{
	__host__ __device__ double operator()(const double &x, const double &lambda)
	{
		return (x / (lambda + 1e-7));
	}
};

struct sqrt_functor
{
	__host__ __device__ double operator()(const double &x)
	{
		return sqrt(x);
	}
};

void EMICP::normalizeRows(Mat &mat, const Mat &alpha, bool withCuda, bool withSqrt)
{
	if (withCuda)
	{
		cuda_normalizeRows(mat, alpha, withSqrt);
	}
	else
	{
		int rows = mat.rows;
		int cols = mat.cols;
		double* arr_mat = new double[rows * cols];
		memcpy(arr_mat, (double*)mat.data, rows * cols * sizeof(double));
		double* arr_alpha = new double[rows];
		memcpy(arr_alpha, (double*)alpha.data, rows * sizeof(double));

		try
		{
			thrust::host_vector<double> h_mat(arr_mat, arr_mat + rows * cols);
			thrust::host_vector<double> h_alpha(arr_alpha, arr_alpha + rows);

			for (int i = 0; i < rows; i++)
			{
				thrust::constant_iterator<double> tmp(h_alpha[i]);
				thrust::host_vector<double>::iterator 
					begin = h_mat.begin() + i * cols;
				thrust::transform(begin, begin + cols, 
					tmp, begin, normalizeRow_functor());
			}

			if (withSqrt)
			{
				thrust::transform(h_mat.begin(), h_mat.end(), 
					h_mat.begin(), sqrt_functor());
			}

			double* h_mat_ptr = thrust::raw_pointer_cast(&h_mat[0]);
			memcpy((double*)mat.data, h_mat_ptr, rows * cols * sizeof(double));
		}
		catch (thrust::system_error e)
		{
			cout << "System Error: " << e.what() << endl;
		}

		delete[] arr_mat;
		delete[] arr_alpha;
	}
}

#define BLOCK_SIZE 128

__global__ void kernelUpdateA(PtrStepSz<double> d_mod, PtrStepSz<double> d_obj,
	PtrStep<double> d_A, double sigma_p2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < d_obj.rows && x < d_mod.rows)
	{
		double tmp[3];
		for (int i = 0; i < 3; i++)
		{
			tmp[i] = d_mod(x, i) - d_obj(y, i);
			tmp[i] *= tmp[i];
		}
		tmp[0] += tmp[1];
		tmp[0] += tmp[2];
		tmp[0] /= sigma_p2;
		tmp[0] = exp(-tmp[0]);

		d_A(y, x) = tmp[0];
	}
}

void EMICP::cuda_updateA(Mat &h_A, const Mat &objSet, const Mat &modSet)
{
	assert(objSet.cols == 3 && modSet.cols == 3);

	int rowsA = objSet.rows;
	int colsA = modSet.rows;
	GpuMat d_obj, d_mod;
	d_obj.upload(objSet);
	d_mod.upload(modSet);

	dim3 block(16, 16);
	dim3 grid((colsA + block.x - 1) / block.x, 
		(rowsA + block.y - 1) / block.y);

	GpuMat d_A(rowsA, colsA, CV_64FC1);
	kernelUpdateA<<<grid, block>>>(d_mod, d_obj, d_A, m_sigma_p2);

	d_A.download(h_A);

	d_obj.release();
	d_mod.release();
	d_A.release();
}

__global__ void kernelNormalizeRows(PtrStepSz<double> d_mat, 
	PtrStepSz<double> d_alpha, int withSqrt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < d_mat.rows && x < d_mat.cols)
	{
		if (withSqrt)
		{
			d_mat(y, x) = sqrt(d_mat(y, x) / (d_alpha(y, 0) + 1e-7));
		}
		else
		{
			d_mat(y, x) = d_mat(y, x) / (d_alpha(y, 0) + 1e-7);
		}
	}
}

void EMICP::cuda_normalizeRows(Mat &mat, 
	const Mat &alpha, bool withSqrt)
{
	int rows = mat.rows;
	int cols = mat.cols;
	GpuMat d_mat, d_alpha;
	d_mat.upload(mat);
	d_alpha.upload(alpha);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, 
		(rows + block.y - 1) / block.y);
	kernelNormalizeRows<<<grid, block>>>(d_mat, d_alpha, (int)withSqrt);
	
	d_mat.download(mat);

	d_mat.release();
	d_alpha.release();
}