#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "emicp.h"

#define BLOCK_SIZE 128

// __global__ void kernelUpdateA(int rowsA, int colsA,
// 	const float* d_modx, const float* d_mody, const float* d_modz, 
// 	const float* d_objx, const float* d_objy, const float* d_objz,
// 	const float* d_R, const float* d_T, float* d_A, float sigma_p2)
// {
// 	int r =  blockIdx.x * blockDim.x + threadIdx.x;
// 	int c =  blockIdx.y * blockDim.y + threadIdx.y;
// 
// 	// Shared memory
// 	__shared__ float modxShare[BLOCK_SIZE];
// 	__shared__ float modyShare[BLOCK_SIZE];
// 	__shared__ float modzShare[BLOCK_SIZE];
// 	__shared__ float objxShare[BLOCK_SIZE];
// 	__shared__ float objyShare[BLOCK_SIZE];
// 	__shared__ float objzShare[BLOCK_SIZE];
// 	__shared__ float RShare[9]; // BLOCK_SIZE >= 9 is assumed
// 	__shared__ float TShare[3]; // BLOCK_SIZE >= 3 is assumed
// 
// 	if(threadIdx.y == 0)
// 	{
// 		if(threadIdx.x < 9)
// 		{
// 			RShare[threadIdx.x] = d_R[threadIdx.x];
// 			if(threadIdx.x < 3)
// 			{
// 				TShare[threadIdx.x] = d_T[threadIdx.x];
// 			}
// 		}
// 	}
// 	
// 	if(r < rowsA && c < colsA)
// 	{ // check for only inside the matrix A
// 		if(threadIdx.x == 0)
// 		{
// 			modxShare[threadIdx.y] = d_modx[c];
// 			modyShare[threadIdx.y] = d_mody[c];
// 			modzShare[threadIdx.y] = d_modz[c];
// 		}
// 		if(threadIdx.y == 0){
// 			objxShare[threadIdx.x] = d_objx[r];
// 			objyShare[threadIdx.x] = d_objy[r];
// 			objzShare[threadIdx.x] = d_objz[r];
// 		}
// 		
// 		__syncthreads();
// 
// #define modx modxShare[threadIdx.y]
// #define mody modyShare[threadIdx.y]
// #define modz modzShare[threadIdx.y]
// #define objx objxShare[threadIdx.x]
// #define objy objyShare[threadIdx.x]
// #define objz objzShare[threadIdx.x]
// #define R(i) RShare[i]
// #define T(i) TShare[i]
// 
// 		float tmpX = modx - (R(0)*objx + R(1)*objy + R(2)*objz + T(0));
// 		float tmpY = mody - (R(3)*objx + R(4)*objy + R(5)*objz + T(1));
// 		float tmpZ = modz - (R(6)*objx + R(7)*objy + R(8)*objz + T(2));
// 
// 		__syncthreads();
// 
// 		tmpX *= tmpX;
// 		tmpY *= tmpY;
// 		tmpZ *= tmpZ;
// 
// 		tmpX += tmpY;
// 		tmpX += tmpZ;
// 
// 		tmpX /= sigma_p2;
// 		tmpX = expf(-tmpX);
// 
// 		d_A[c * rowsA + r] = tmpX;
// 	}
// }
// 
// void EMICP::cuda_updateA(Mat &A, const Mat &objSet, 
// 	const Mat &modSet, float* h_R, float* h_T)
// {
// 	assert(objSet.cols == 3 && modSet.cols == 3);
// 
// 	const int floatSize = sizeof(float);
// 	float* d_R, *d_T;
// 	cudaMalloc((void**)&d_R, 9 * floatSize);
// 	cudaMemcpy(d_R, h_R, 9 * floatSize, cudaMemcpyHostToDevice);
// 	cudaMalloc((void**)&d_T, 3 * floatSize);
// 	cudaMemcpy(d_T, h_T, 3 * floatSize, cudaMemcpyHostToDevice);
// 
// 	int objSize = objSet.rows;
// 	int modSize = modSet.rows;
// 	float* h_obj = new float[objSize];
// 	float* h_mod = new float[modSize];
// 
// 	float* d_mod, *d_obj;
// 
// 	size_t size = modSize * 3 * floatSize;
// 	Mat modT = modSet.t();
// 	memcpy(h_mod, (float*)modT.data, size);
// 	cudaMalloc((void**)&d_mod, size);
// 	cudaMemcpy(d_mod, h_mod, size, cudaMemcpyHostToDevice);
// 	float* d_modx = &d_mod[modSize * 0];
// 	float* d_mody = &d_mod[modSize * 1];
// 	float* d_modz = &d_mod[modSize * 2];
// 
// 	size = objSize * 3 * floatSize;
// 	Mat objT = objSet.t();
// 	memcpy(h_obj, (float*)objT.data, objSize * 3 * floatSize);
// 	cudaMalloc((void**)&d_obj, size);
// 	cudaMemcpy(d_obj, h_obj, size, cudaMemcpyHostToDevice);
// 	float* d_objx = &d_obj[objSize * 0];
// 	float* d_objy = &d_obj[objSize * 1];
// 	float* d_objz = &d_obj[objSize * 2];
// 
// 	int rowsA = objSize;
// 	int colsA = modSize;
// 	float* h_A = new float[rowsA * colsA];
// 	float* d_A;
// 	size = rowsA * colsA * floatSize;
// 	cudaMalloc((void**)&d_A, size);
// 
// 	dim3 dimBlockForA(BLOCK_SIZE, BLOCK_SIZE);
// 	dim3 dimGridForA((rowsA + dimBlockForA.x - 1) / dimBlockForA.x, 
// 		(colsA + dimBlockForA.y - 1) / dimBlockForA.y);
// 
// 	kernelUpdateA<<<dimGridForA, dimBlockForA>>>
// 		(rowsA, colsA, d_modx, d_mody, d_modz, 
// 		d_objx, d_objy, d_objz, d_R, d_T, d_A, m_sigma_p2);
// 
// 	cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
// 	A = Mat(rowsA, colsA, CV_32FC1, h_A);
// 
// 	cudaFree(&d_mod);
// 	cudaFree(&d_obj);
// 	cudaFree(&d_R);
// 	cudaFree(&d_T);
// 	cudaFree(&d_A);
// 
// 	delete[] h_obj;
// 	delete[] h_mod;
// 	delete[] h_A;
// }

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