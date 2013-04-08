#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "abstracticp.h"

struct dist2_functor
{
	__host__ __device__ float operator()(const float3 &p1, const float3 &p2)
	{
		float tmp0, tmp1, tmp2;
		tmp0 = p1.x - p2.x;
		tmp1 = p1.y - p2.y;
		tmp2 = p1.z - p2.z;
		tmp0 = tmp0 * tmp0 + tmp1 * tmp1 + tmp2 * tmp2;
		return tmp0;
	}
};

double AbstractICP::computeError(const Mat &objAfterTrans, 
	const Mat &mod, const Mat &lambda, bool withCuda)
{
	assert(objAfterTrans.rows == mod.rows);
	assert(objAfterTrans.rows == lambda.rows);
	if (withCuda)
	{
		return 0;
	}
	else
	{
		float error = 0;
		int rows = objAfterTrans.rows;

		float3* arr_obj = new float3[rows];
		memcpy(arr_obj, (float3*)objAfterTrans.data, rows * sizeof(float3));
		float3* arr_mod = new float3[rows];
		memcpy(arr_mod, (float3*)mod.data, rows * sizeof(float3));
		float* arr_lambda = new float[rows];
		memcpy(arr_lambda, (float*)lambda.data, rows * sizeof(float));

		try
		{
			thrust::host_vector<float3> h_obj(arr_obj, arr_obj + rows);
			thrust::host_vector<float3> h_mod(arr_mod, arr_mod + rows);
			thrust::host_vector<float> h_lambda(
				arr_lambda, arr_lambda + rows);

			thrust::host_vector<float> tmp(rows);
			thrust::transform(h_obj.begin(), h_obj.end(), 
				h_mod.begin(), tmp.begin(), dist2_functor());
			thrust::transform(tmp.begin(), tmp.end(), 
				h_lambda.begin(), tmp.begin(), thrust::multiplies<float>());
			error = thrust::reduce(tmp.begin(), 
				tmp.end(), 0.0f, thrust::plus<float>());
		}
		catch (thrust::system_error e)
		{
			cout << "System Error: " << e.what() << endl;
		}

		delete[] arr_obj;
		delete[] arr_mod;
		delete[] arr_lambda;

		return error;
	}
}