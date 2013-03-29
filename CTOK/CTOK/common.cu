#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "common.h"

#define BLOCK_SIZE 512
#define GRID_SIZE 512

char* fileName = "common.cu";

inline void __checkCudaErrors(cudaError err, 
	const char* file = NULL, const int line = 0)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

struct transform_functor
{
	float R[9], T[3];
	transform_functor(const Mat &m_R, const Mat &m_T)
	{
		memcpy(R, (float*)m_R.data, 9 * sizeof(float));
		memcpy(T, (float*)m_T.data, 3 * sizeof(float));
	}
	__host__ __device__ float3 operator()(const float3 &pt) const
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
	if (num == 0)
	{
		return;
	}
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