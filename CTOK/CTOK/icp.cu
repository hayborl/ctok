#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "sm_20_atomic_functions.h"

#include "icp.h"

char* fileName = "icp.cu";

inline void __checkCudaErrors(cudaError err, const char *file = NULL, const int line = 0)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

__global__ void kernelTransform(const float* p, const float* R, 
	const float* T, float* res, int count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < count)
	{
		float tp[3];
		tp[0] = p[i * 3 + 0];
		tp[1] = p[i * 3 + 1];
		tp[2] = p[i * 3 + 2];
		float resp[3];
		for(int y = 0; y < 3; y++)
		{
			float tmp = 0;
			for (int x = 0; x < 3; x++)
			{
				int index = y * 3 + x;
				tmp += R[index] * tp[x];
			}
			resp[y] = tmp + T[y];
		}
		res[i * 3 + 0] = resp[0];
		res[i * 3 + 1] = resp[1];
		res[i * 3 + 2] = resp[2];
	}
}

void cuda_transform(const float* p, int count, const float* R, 
	const float* T, float* res)
{
	int size;
	// Load A and B to the device
	float* pd;
	size = count * 3 * sizeof(float);
	__checkCudaErrors(cudaMalloc((void**)&pd, size), fileName, 56);
	__checkCudaErrors(cudaMemcpy(pd, p, size, cudaMemcpyHostToDevice), 
		fileName, 57);
	float* Rd;
	size = 3 * 3 * sizeof(float);
	__checkCudaErrors(cudaMalloc((void**)&Rd, size), fileName, 61);
	__checkCudaErrors(cudaMemcpy(Rd, R, size, cudaMemcpyHostToDevice), 
		fileName, 62);
	// Allocate C on the device
	float* Td;
	size = 3 * 1 * sizeof(float);
	__checkCudaErrors(cudaMalloc((void**)&Td, size), fileName, 67);
	__checkCudaErrors(cudaMemcpy(Td, T, size, cudaMemcpyHostToDevice), 
		fileName, 68);
	float* resd;
	size = count * 3 * sizeof(float);
	__checkCudaErrors(cudaMalloc((void**)&resd, size), fileName, 72);
	// Compute the execution configuration assuming
	// the matrix dimensions are multiples of BLOCK_SIZE
	// Launch the device computation
	kernelTransform<<<count / BLOCK_SIZE + 1, BLOCK_SIZE>>>
		(pd, Rd, Td, resd, count);
	// Read C from the device
	__checkCudaErrors(cudaMemcpy(res, resd, size, cudaMemcpyDeviceToHost), 
		fileName, 80);
	// Free device memory
	cudaFree(pd);
	cudaFree(Rd);
	cudaFree(Td);
	cudaFree(resd);
}

EXTERN_C void cuda_transformPointCloud(Mat input, 
	Mat* output, Mat transformMat)
{
	int rows = input.rows;
	int size = rows * 3;
	float* p = new float[size];
	memcpy(p, (float*)input.data, size * sizeof(float));

	Mat MR = transformMat(Rect(0, 0, 3, 3)).clone();
	Mat MT = transformMat(Rect(3, 0, 1, 3)).clone();
	float R[9], T[3];
	memcpy(R, (float*)MR.data, 9 * sizeof(float));
	memcpy(T, (float*)MT.data, 3 * sizeof(float));

	float* res = new float[size];
	cuda_transform(p, rows, R, T, res);
	memcpy((float*)output->data, res, size * sizeof(float));

	delete[] p;
	delete[] res;
}
