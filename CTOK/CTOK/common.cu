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
	double R[9], T[3];
	transform_functor(const Mat &m_R, const Mat &m_T)
	{
		memcpy(R, (double*)m_R.data, 9 * sizeof(double));
		memcpy(T, (double*)m_T.data, 3 * sizeof(double));
	}
	__host__ __device__ double3 operator()(const double3 &pt) const
	{
		double3 tmp;
		tmp.x = R[0] * pt.x + R[1] * pt.y + R[2] * pt.z + T[0];
		tmp.y = R[3] * pt.x + R[4] * pt.y + R[5] * pt.z + T[1];
		tmp.z = R[6] * pt.x + R[7] * pt.y + R[8] * pt.z + T[2];
		return tmp;
	}
};

void transformPointCloud(const Mat &input, Mat &output, 
	const Mat &transformMat, bool withCuda)
{
	if (isIdentity(transformMat))
	{
		output = input.clone();
		return;
	}
	Mat m_R = transformMat(Rect(0, 0, 3, 3)).clone();
	Mat m_T = transformMat(Rect(3, 0, 1, 3)).clone();

	int num = input.rows;
	if (num == 0)
	{
		return;
	}
	double3* arr_in = new double3[num];
	memcpy(arr_in, (double3*)input.data, num * sizeof(double3));

	try
	{
		thrust::host_vector<double3> h_out(num);
		if (withCuda)
		{
			thrust::device_vector<double3> d_in(arr_in, arr_in + num);
			thrust::device_vector<double3> d_out(num);

			thrust::transform(d_in.begin(), d_in.end(), 
				d_out.begin(), transform_functor(m_R, m_T));

			h_out = d_out;
		}
		else
		{
			thrust::host_vector<double3> h_in(arr_in, arr_in + num);

			thrust::transform(h_in.begin(), h_in.end(), 
				h_out.begin(), transform_functor(m_R, m_T));
		}

		double3* h_out_ptr = thrust::raw_pointer_cast(&h_out[0]);

		output.create(input.rows, input.cols, input.type());
		memcpy((double3*)output.data, h_out_ptr, num * sizeof(double3));
	}
	catch (thrust::system_error e)
	{
		cout << "System Error: " << e.what() << endl;
	}

	delete[] arr_in;
}