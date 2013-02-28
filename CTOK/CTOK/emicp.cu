#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cublas.h"

#include "emicp.h"

#define BLOCK_SIZE 256

__global__ void kernelUpdateA(int rowsA, int colsA,
	const float* d_modx, const float* d_mody, const float* d_modz, 
	const float* d_objx, const float* d_objy, const float* d_objz,
	const float* d_R, const float* d_T, float* d_A, float sigma_p2)
{
	int r =  blockIdx.x * blockDim.x + threadIdx.x;
	int c =  blockIdx.y * blockDim.y + threadIdx.y;

	// Shared memory
	__shared__ float modxShare[BLOCK_SIZE];
	__shared__ float modyShare[BLOCK_SIZE];
	__shared__ float modzShare[BLOCK_SIZE];
	__shared__ float objxShare[BLOCK_SIZE];
	__shared__ float objyShare[BLOCK_SIZE];
	__shared__ float objzShare[BLOCK_SIZE];
	__shared__ float RShare[9]; // BLOCK_SIZE >= 9 is assumed
	__shared__ float TShare[3]; // BLOCK_SIZE >= 3 is assumed

	if(threadIdx.y == 0)
	{
		if(threadIdx.x < 9)
		{
			RShare[threadIdx.x] = d_R[threadIdx.x];
			if(threadIdx.x < 3)
			{
				TShare[threadIdx.x] = d_T[threadIdx.x];
			}
		}
	}
	
	if(r < rowsA && c < colsA)
	{ // check for only inside the matrix A
		if(threadIdx.x == 0)
		{
			modxShare[threadIdx.y] = d_modx[c];
			modyShare[threadIdx.y] = d_mody[c];
			modzShare[threadIdx.y] = d_modz[c];
		}
		if(threadIdx.y == 0){
			objxShare[threadIdx.x] = d_objx[r];
			objyShare[threadIdx.x] = d_objy[r];
			objzShare[threadIdx.x] = d_objz[r];
		}
		
		__syncthreads();

#define modx modxShare[threadIdx.y]
#define mody modyShare[threadIdx.y]
#define modz modzShare[threadIdx.y]
#define objx objxShare[threadIdx.x]
#define objy objyShare[threadIdx.x]
#define objz objzShare[threadIdx.x]
#define R(i) RShare[i]
#define T(i) TShare[i]

		float tmpX = modx - (R(0)*objx + R(1)*objy + R(2)*objz + T(0));
		float tmpY = mody - (R(3)*objx + R(4)*objy + R(5)*objz + T(1));
		float tmpZ = modz - (R(6)*objx + R(7)*objy + R(8)*objz + T(2));

		__syncthreads();

		tmpX *= tmpX;
		tmpY *= tmpY;
		tmpZ *= tmpZ;

		tmpX += tmpY;
		tmpX += tmpZ;

		tmpX /= sigma_p2;
		tmpX = expf(-tmpX);

		d_A[c * rowsA + r] = tmpX;
	}
}

void EMICP::cuda_updateA(Mat &A, const Mat &objSet, 
	const Mat &modSet, float* h_R, float* h_T)
{
	assert(objSet.cols == 3 && modSet.cols == 3);

	const int floatSize = sizeof(float);
	float* d_R, *d_T;
	cudaMalloc((void**)&d_R, 9 * floatSize);
	cudaMemcpy(d_R, h_R, 9 * floatSize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_T, 3 * floatSize);
	cudaMemcpy(d_T, h_T, 3 * floatSize, cudaMemcpyHostToDevice);

	int objSize = objSet.rows;
	int modSize = modSet.rows;
	float* h_obj = new float[objSize];
	float* h_mod = new float[modSize];

	float* d_mod, *d_obj;

	size_t size = modSize * 3 * floatSize;
	Mat modT = modSet.t();
	memcpy(h_mod, (float*)modT.data, size);
	cudaMalloc((void**)&d_mod, size);
	cudaMemcpy(d_mod, h_mod, size, cudaMemcpyHostToDevice);
	float* d_modx = &d_mod[modSize * 0];
	float* d_mody = &d_mod[modSize * 1];
	float* d_modz = &d_mod[modSize * 2];

	size = objSize * 3 * floatSize;
	Mat objT = objSet.t();
	memcpy(h_obj, (float*)objT.data, objSize * 3 * floatSize);
	cudaMalloc((void**)&d_obj, size);
	cudaMemcpy(d_obj, h_obj, size, cudaMemcpyHostToDevice);
	float* d_objx = &d_obj[objSize * 0];
	float* d_objy = &d_obj[objSize * 1];
	float* d_objz = &d_obj[objSize * 2];

	int rowsA = objSize;
	int colsA = modSize;
	float* h_A = new float[rowsA * colsA];
	float* d_A;
	size = rowsA * colsA * floatSize;
	cudaMalloc((void**)&d_A, size);

	dim3 dimBlockForA(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGridForA((rowsA + dimBlockForA.x - 1) / dimBlockForA.x, 
		(colsA + dimBlockForA.y - 1) / dimBlockForA.y);

	kernelUpdateA<<<dimGridForA, dimBlockForA>>>
		(rowsA, colsA, d_modx, d_mody, d_modz, 
		d_objx, d_objy, d_objz, d_R, d_T, d_A, m_sigma_p2);

	cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	A = Mat(rowsA, colsA, CV_32FC1, h_A);

	cudaFree(&d_mod);
	cudaFree(&d_obj);
	cudaFree(&d_R);
	cudaFree(&d_T);
	cudaFree(&d_A);

	delete[] h_obj;
	delete[] h_mod;
	delete[] h_A;
}