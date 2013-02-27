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

__device__ void float3Add(float3* fs, float x, float y, 
	float z, unsigned int index)
{
	fs[index].x += x;
	fs[index].y += y;
	fs[index].z += z;
}

#define FLOAT3_ADD(f1, f2) f1.x += f2.x; f1.y += f2.y; f1.z += f2.z;
#define FLOAT3_ASSIGN(f1, f2) f1.x = f2.x; f1.y = f2.y; f1.z = f2.z;

__global__ void kernelComputeTransformation(float *cpIn, float *opIn, 
	unsigned int n, const char* weights, float* cpOut, float* opOut, float3* ccOut)
{
	extern __shared__ float cpData[];
	extern __shared__ float opData[];
	extern __shared__ float3 ccData0[];
	extern __shared__ float3 ccData1[];
	extern __shared__ float3 ccData2[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int i = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
	unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;

	float cpSum[3], opSum[3];
	float3 ccSum[3];
	for (int j = 0; j < 3; j++)
	{
		cpSum[j] = 0;
		opSum[j] = 0;
		ccSum[j] = make_float3(0, 0, 0);
	}

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		if (weights[i] == 1)
		{
			unsigned int i3 = i * 3;
			for (int j = 0; j < 3; j++)
			{
				unsigned int ti = i3 + j;
				float op = opIn[ti];
				cpSum[j] += cpIn[ti];
				opSum[j] += op;
				float3Add(ccSum, op * cpIn[i3 + 0], 
					op * cpIn[i3 + 1], op * cpIn[i3 + 2], j);
			}
		}

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (i + BLOCK_SIZE < n)
		{
			if (weights[i + BLOCK_SIZE] == 1)
			{
				unsigned int i3 = (i + BLOCK_SIZE) * 3;
				for (int j = 0; j < 3; j++)
				{
					unsigned int ti = i3 + j;
					float op = opIn[ti];
					cpSum[j] += cpIn[ti];
					opSum[j] += op;
					float3Add(ccSum, op * cpIn[i3 + 0], 
						op * cpIn[i3 + 1], op * cpIn[i3 + 2], j);
				}
			}	
		}

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	float* cpTmp = cpData;
	float* opTmp = opData;
	for (int j = 0; j < 3; j++)
	{
		cpTmp[tid] = cpSum[j];
		opTmp[tid] = opSum[j];
		cpTmp += blockDim.x;
		opTmp += blockDim.x;
	}
	ccData0[tid] = ccSum[0];
	ccData1[tid] = ccSum[1];
	ccData2[tid] = ccSum[2];
	__syncthreads();


	// do reduction in shared mem
	if (BLOCK_SIZE >= 512)
	{
		if (tid < 256)
		{
			unsigned int tid2 = tid + 256;
			float* cpTmp = cpData;
			float* opTmp = opData;
			for (int j = 0; j < 3; j++)
			{
				cpSum[j] += cpTmp[tid2];
				opSum[j] += opTmp[tid2];
				cpTmp[tid] = cpSum[j];
				opTmp[tid] = opSum[j];
				cpTmp += blockDim.x;
				opTmp += blockDim.x;
			}
			FLOAT3_ADD(ccSum[0], ccData0[tid2]);
			FLOAT3_ADD(ccSum[1], ccData0[tid2]);
			FLOAT3_ADD(ccSum[2], ccData0[tid2]);
			ccData0[tid] = ccSum[0];
			ccData1[tid] = ccSum[1];
			ccData2[tid] = ccSum[2];
		}

		__syncthreads();
	}

	if (BLOCK_SIZE >= 256)
	{
		if (tid < 128)
		{
			unsigned int tid2 = tid + 128;
			float* cpTmp = cpData;
			float* opTmp = opData;
			for (int j = 0; j < 3; j++)
			{
				cpSum[j] += cpTmp[tid2];
				opSum[j] += opTmp[tid2];
				cpTmp[tid] = cpSum[j];
				opTmp[tid] = opSum[j];
				cpTmp += blockDim.x;
				opTmp += blockDim.x;
			}
			FLOAT3_ADD(ccSum[0], ccData0[tid2]);
			FLOAT3_ADD(ccSum[1], ccData0[tid2]);
			FLOAT3_ADD(ccSum[2], ccData0[tid2]);
			ccData0[tid] = ccSum[0];
			ccData1[tid] = ccSum[1];
			ccData2[tid] = ccSum[2];
		}

		__syncthreads();
	}

	if (BLOCK_SIZE >= 128)
	{
		if (tid <  64)
		{
			unsigned int tid2 = tid + 64;
			float* cpTmp = cpData;
			float* opTmp = opData;
			for (int j = 0; j < 3; j++)
			{
				cpSum[j] += cpTmp[tid2];
				opSum[j] += opTmp[tid2];
				cpTmp[tid] = cpSum[j];
				opTmp[tid] = opSum[j];
				cpTmp += blockDim.x;
				opTmp += blockDim.x;
			}
			FLOAT3_ADD(ccSum[0], ccData0[tid2]);
			FLOAT3_ADD(ccSum[1], ccData0[tid2]);
			FLOAT3_ADD(ccSum[2], ccData0[tid2]);
			ccData0[tid] = ccSum[0];
			ccData1[tid] = ccSum[1];
			ccData2[tid] = ccSum[2];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float* cpmem = cpData;
		volatile float* opmem = opData;
		volatile float3* ccmem0 = ccData0;
		volatile float3* ccmem1 = ccData1;
		volatile float3* ccmem2 = ccData2;

		if (BLOCK_SIZE >=  64)
		{
			unsigned int tid2 = tid + 32;
			cpmem = cpData;
			opmem = opData;
			for (int j = 0; j < 3; j++)
			{
				cpSum[j] += cpmem[tid2];
				opSum[j] += opmem[tid2];
				cpmem[tid] = cpSum[j];
				opmem[tid] = opSum[j];
				cpmem += blockDim.x;
				opmem += blockDim.x;
			}
			FLOAT3_ADD(ccSum[0], ccmem0[tid2]);
			FLOAT3_ADD(ccSum[1], ccmem1[tid2]);
			FLOAT3_ADD(ccSum[2], ccmem2[tid2]);
			FLOAT3_ASSIGN(ccmem0[tid], ccSum[0]);
			FLOAT3_ASSIGN(ccmem1[tid], ccSum[1]);
			FLOAT3_ASSIGN(ccmem2[tid], ccSum[2]);
		}

		if (BLOCK_SIZE >=  32)
		{
			unsigned int tid2 = tid + 16;
			cpmem = cpData;
			opmem = opData;
			for (int j = 0; j < 3; j++)
			{
				cpSum[j] += cpmem[tid2];
				opSum[j] += opmem[tid2];
				cpmem[tid] = cpSum[j];
				opmem[tid] = opSum[j];
				cpmem += blockDim.x;
				opmem += blockDim.x;
			}
			FLOAT3_ADD(ccSum[0], ccmem0[tid2]);
			FLOAT3_ADD(ccSum[1], ccmem1[tid2]);
			FLOAT3_ADD(ccSum[2], ccmem2[tid2]);
			FLOAT3_ASSIGN(ccmem0[tid], ccSum[0]);
			FLOAT3_ASSIGN(ccmem1[tid], ccSum[1]);
			FLOAT3_ASSIGN(ccmem2[tid], ccSum[2]);
		}

		if (BLOCK_SIZE >=  16)
		{
			unsigned int tid2 = tid + 8;
			cpmem = cpData;
			opmem = opData;
			for (int j = 0; j < 3; j++)
			{
				cpSum[j] += cpmem[tid2];
				opSum[j] += opmem[tid2];
				cpmem[tid] = cpSum[j];
				opmem[tid] = opSum[j];
				cpmem += blockDim.x;
				opmem += blockDim.x;
			}
			FLOAT3_ADD(ccSum[0], ccmem0[tid2]);
			FLOAT3_ADD(ccSum[1], ccmem1[tid2]);
			FLOAT3_ADD(ccSum[2], ccmem2[tid2]);
			FLOAT3_ASSIGN(ccmem0[tid], ccSum[0]);
			FLOAT3_ASSIGN(ccmem1[tid], ccSum[1]);
			FLOAT3_ASSIGN(ccmem2[tid], ccSum[2]);
		}

		if (BLOCK_SIZE >=   8)
		{
			unsigned int tid2 = tid + 4;
			cpmem = cpData;
			opmem = opData;
			for (int j = 0; j < 3; j++)
			{
				cpSum[j] += cpmem[tid2];
				opSum[j] += opmem[tid2];
				cpmem[tid] = cpSum[j];
				opmem[tid] = opSum[j];
				cpmem += blockDim.x;
				opmem += blockDim.x;
			}
			FLOAT3_ADD(ccSum[0], ccmem0[tid2]);
			FLOAT3_ADD(ccSum[1], ccmem1[tid2]);
			FLOAT3_ADD(ccSum[2], ccmem2[tid2]);
			FLOAT3_ASSIGN(ccmem0[tid], ccSum[0]);
			FLOAT3_ASSIGN(ccmem1[tid], ccSum[1]);
			FLOAT3_ASSIGN(ccmem2[tid], ccSum[2]);
		}

		if (BLOCK_SIZE >=   4)
		{
			unsigned int tid2 = tid + 2;
			cpmem = cpData;
			opmem = opData;
			for (int j = 0; j < 3; j++)
			{
				cpSum[j] += cpmem[tid2];
				opSum[j] += opmem[tid2];
				cpmem[tid] = cpSum[j];
				opmem[tid] = opSum[j];
				cpmem += blockDim.x;
				opmem += blockDim.x;
			}
			FLOAT3_ADD(ccSum[0], ccmem0[tid2]);
			FLOAT3_ADD(ccSum[1], ccmem1[tid2]);
			FLOAT3_ADD(ccSum[2], ccmem2[tid2]);
			FLOAT3_ASSIGN(ccmem0[tid], ccSum[0]);
			FLOAT3_ASSIGN(ccmem1[tid], ccSum[1]);
			FLOAT3_ASSIGN(ccmem2[tid], ccSum[2]);
		}

		if (BLOCK_SIZE >=   2)
		{
			unsigned int tid2 = tid + 1;
			cpmem = cpData;
			opmem = opData;
			for (int j = 0; j < 3; j++)
			{
				cpSum[j] += cpmem[tid2];
				opSum[j] += opmem[tid2];
				cpmem[tid] = cpSum[j];
				opmem[tid] = opSum[j];
				cpmem += blockDim.x;
				opmem += blockDim.x;
			}
			FLOAT3_ADD(ccSum[0], ccmem0[tid2]);
			FLOAT3_ADD(ccSum[1], ccmem1[tid2]);
			FLOAT3_ADD(ccSum[2], ccmem2[tid2]);
			FLOAT3_ASSIGN(ccmem0[tid], ccSum[0]);
			FLOAT3_ASSIGN(ccmem1[tid], ccSum[1]);
			FLOAT3_ASSIGN(ccmem2[tid], ccSum[2]);
		}
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		float* cpOutTmp = cpOut;
		float* opOutTmp = opOut;
		float* cpTmp = cpData;
		float* opTmp = opData;
		for (int j = 0; j < 3; j++)
		{
			cpOutTmp[bid] = cpTmp[0];
			opOutTmp[bid] = opTmp[0];
			cpOutTmp += gridDim.x;
			opOutTmp += gridDim.x;
			cpTmp += blockDim.x;
			opTmp += blockDim.x;
		}
		ccOut[bid] = ccData0[0];
		ccOut[bid + gridDim.x] = ccData1[0];
		ccOut[bid + gridDim.x * 2] = ccData2[0];
	}
}

void cuda_computeTrans(const float* cpIn, const float* opIn, 
	unsigned int n, const char* weights, float* cpOut, float* opOut, float3* ccOut)
{
	float* cpInd, *opInd, *cpOutd, *opOutd;
	float3* ccOutd;
	int copySize = n * 3 * sizeof(float);
	__checkCudaErrors(cudaMalloc((void**)&cpInd, copySize), fileName, 160);
	__checkCudaErrors(cudaMemcpy(cpInd, cpIn, copySize, 
		cudaMemcpyHostToDevice), fileName, 161);
	__checkCudaErrors(cudaMalloc((void**)&opInd, copySize), fileName, 163);
	__checkCudaErrors(cudaMemcpy(opInd, opIn, copySize, 
		cudaMemcpyHostToDevice), fileName, 164);
	char* weightsd;
	copySize = n * sizeof(char);
	__checkCudaErrors(cudaMalloc((void**)&weightsd, copySize), fileName, 178);
	__checkCudaErrors(cudaMemcpy(weightsd, weights, copySize, 
		cudaMemcpyHostToDevice), fileName, 179);
	int blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
	copySize = blocks * 3 * sizeof(float);
	__checkCudaErrors(cudaMalloc((void**)&cpOutd, copySize), fileName, 167);
	__checkCudaErrors(cudaMalloc((void**)&opOutd, copySize), fileName, 167);
	copySize = blocks * 3 * sizeof(float3);
	__checkCudaErrors(cudaMalloc((void**)&ccOutd, copySize), fileName, 167);

	int smemSize = BLOCK_SIZE * 3 * sizeof(float);
	kernelComputeTransformation<<<blocks, BLOCK_SIZE, smemSize>>>(cpInd, 
		opInd, n, weightsd, cpOutd, opOutd, ccOutd);

	float* cpOutT = new float[blocks * 3];
	float* opOutT = new float[blocks * 3];
	float3* ccOutT = new float3[blocks * 3];
	copySize = blocks * 3 * sizeof(float);
	cudaMemcpy(cpOutT, cpOutd, copySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(opOutT, opOutd, copySize, cudaMemcpyDeviceToHost);
	copySize = blocks * 3 * sizeof(float3);
	cudaMemcpy(ccOutT, ccOutd, copySize, cudaMemcpyDeviceToHost);

	float* cpTmp = cpOutT;
	float* opTmp = opOutT;
	float3* ccTmp = ccOutT;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < blocks; j++)
		{
			cpOut[i] += cpTmp[j];
			opOut[i] += opTmp[j];
			FLOAT3_ADD(ccOut[i], ccTmp[j]);
		}
		cpTmp += blocks;
		opTmp += blocks;
		ccTmp += blocks;
	}

	cudaFree(cpInd);
	cudaFree(opInd);
	cudaFree(cpOutd);
	cudaFree(opOutd);
	cudaFree(ccOutd);
	cudaFree(weightsd);
	delete[] cpOutT;
	delete[] opOutT;
	delete[] ccOutT;
}

void ICP::cuda_computeTransformation(const Mat &closest, char* weights, 
	Mat* ccMatrix, Mat* meanMatObj, Mat* meanMatMod)
{
	int rows = m_objSet.rows;
	float* cph = new float[rows * 3];
	int size = rows * 3 * sizeof(float);
	memcpy(cph, (float*)closest.data, size);
	float* oph = new float[rows * 3];
	memcpy(oph, (float*)m_objSet.data, size);
	float cp[3] = {0}, op[3] = {0};
	float3 cc[3] = {make_float3(0, 0, 0)};
	cuda_computeTrans(cph, oph, rows, weights, cp, op, cc);

	*ccMatrix = Mat(3, 3, CV_32FC1, cc).clone();
	*meanMatObj = Mat(3, 1, CV_32FC1, op).clone();
	*meanMatMod = Mat(3, 1, CV_32FC1, cp).clone();

	cout << *ccMatrix << endl << *meanMatMod << *meanMatObj << endl;

	delete[] cph;
	delete[] oph;
}