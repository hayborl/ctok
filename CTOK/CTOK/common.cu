#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "common.h"

#define BLOCK_SIZE 512
#define GRID_SIZE 512

char* fileName = "common.cu";

inline void __checkCudaErrors(cudaError err, 
	const char *file = NULL, const int line = 0)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// __global__ void kernelTransform(const float* p, const float* R, 
// 	const float* T, float* res, int count)
// {
// 	int i = blockIdx.x * blockDim.x + threadIdx.x;
// 
// 	if (i < count)
// 	{
// 		float tp[3];
// 		tp[0] = p[i * 3 + 0];
// 		tp[1] = p[i * 3 + 1];
// 		tp[2] = p[i * 3 + 2];
// 		float resp[3];
// 		for(int y = 0; y < 3; y++)
// 		{
// 			float tmp = 0;
// 			for (int x = 0; x < 3; x++)
// 			{
// 				int index = y * 3 + x;
// 				tmp += R[index] * tp[x];
// 			}
// 			resp[y] = tmp + T[y];
// 		}
// 		res[i * 3 + 0] = resp[0];
// 		res[i * 3 + 1] = resp[1];
// 		res[i * 3 + 2] = resp[2];
// 	}
// }
// 
// void cuda_transform(const float* p, int count, const float* R, 
// 	const float* T, float* res)
// {
// 	int size;
// 	// Load A and B to the device
// 	float* pd;
// 	size = count * 3 * sizeof(float);
// 	__checkCudaErrors(cudaMalloc((void**)&pd, size), fileName, 56);
// 	__checkCudaErrors(cudaMemcpy(pd, p, size, cudaMemcpyHostToDevice), 
// 		fileName, 57);
// 	float* Rd;
// 	size = 3 * 3 * sizeof(float);
// 	__checkCudaErrors(cudaMalloc((void**)&Rd, size), fileName, 61);
// 	__checkCudaErrors(cudaMemcpy(Rd, R, size, cudaMemcpyHostToDevice), 
// 		fileName, 62);
// 	// Allocate C on the device
// 	float* Td;
// 	size = 3 * 1 * sizeof(float);
// 	__checkCudaErrors(cudaMalloc((void**)&Td, size), fileName, 67);
// 	__checkCudaErrors(cudaMemcpy(Td, T, size, cudaMemcpyHostToDevice), 
// 		fileName, 68);
// 	float* resd;
// 	size = count * 3 * sizeof(float);
// 	__checkCudaErrors(cudaMalloc((void**)&resd, size), fileName, 72);
// 	// Compute the execution configuration assuming
// 	// the matrix dimensions are multiples of BLOCK_SIZE
// 	// Launch the device computation
// 	kernelTransform<<<count / BLOCK_SIZE + 1, BLOCK_SIZE>>>
// 		(pd, Rd, Td, resd, count);
// 	// Read C from the device
// 	__checkCudaErrors(cudaMemcpy(res, resd, size, cudaMemcpyDeviceToHost), 
// 		fileName, 80);
// 	// Free device memory
// 	cudaFree(pd);
// 	cudaFree(Rd);
// 	cudaFree(Td);
// 	cudaFree(resd);
// }
// 
// void cuda_transformPointCloud(Mat input, 
// 	Mat* output, Mat transformMat)
// {
// 	*output = Mat(input.rows, input.cols, input.type());
// 
// 	int rows = input.rows;
// 	int size = rows * 3;
// 	float* p = new float[size];
// 	memcpy(p, (float*)input.data, size * sizeof(float));
// 
// 	Mat MR = transformMat(Rect(0, 0, 3, 3)).clone();
// 	Mat MT = transformMat(Rect(3, 0, 1, 3)).clone();
// 	float R[9], T[3];
// 	memcpy(R, (float*)MR.data, 9 * sizeof(float));
// 	memcpy(T, (float*)MT.data, 3 * sizeof(float));
// 
// 	float* res = new float[size];
// 	cuda_transform(p, rows, R, T, res);
// 	memcpy((float*)output->data, res, size * sizeof(float));
// 
// 	delete[] p;
// 	delete[] res;
// }

struct transform_functor
{
	float R[9], T[3];
	transform_functor(const Mat& m_R, const Mat& m_T)
	{
		memcpy(R, (float*)m_R.data, 9 * sizeof(float));
		memcpy(T, (float*)m_T.data, 3 * sizeof(float));
	}
	__host__ __device__ float3 operator()(const float3& pt) const
	{
		float3 tmp;
		tmp.x = R[0] * pt.x + R[1] * pt.y + R[2] * pt.z + T[0];
		tmp.y = R[3] * pt.x + R[4] * pt.y + R[5] * pt.z + T[1];
		tmp.z = R[6] * pt.x + R[7] * pt.y + R[8] * pt.z + T[2];
		return tmp;
	}
};

void transformPointCloud(Mat input, Mat* output, 
	Mat transformMat, bool withCuda)
{
	Mat m_R = transformMat(Rect(0, 0, 3, 3)).clone();
	Mat m_T = transformMat(Rect(3, 0, 1, 3)).clone();

	int num = input.rows;
	float3* arr_in = new float3[num];
	memcpy(arr_in, (float3*)input.data, num * sizeof(float3));

	try
	{
		thrust::host_vector<float3> h_out(num);
		if (withCuda)
		{
			thrust::device_vector<float3> d_in(arr_in, arr_in + num);
			thrust::device_vector<float3> d_out(num);

			thrust::transform(d_in.begin(), d_in.end(), 
				d_out.begin(), transform_functor(m_R, m_T));

			h_out = d_out;
		}
		else
		{
			thrust::host_vector<float3> h_in(arr_in, arr_in + num);

			thrust::transform(h_in.begin(), h_in.end(), 
				h_out.begin(), transform_functor(m_R, m_T));
		}

		float3* h_out_ptr = thrust::raw_pointer_cast(&h_out[0]);

		*output = Mat(input.rows, input.cols, input.type());
		memcpy((float3*)output->data, h_out_ptr, num * sizeof(float3));
	}
	catch (thrust::system_error e)
	{
		cout << "System Error: " << e.what() << endl;
	}

	delete[] arr_in;
}