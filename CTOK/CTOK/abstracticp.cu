#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "abstracticp.h"

struct dist2_functor
{
	__host__ __device__ double operator()(const double3 &p1, const double3 &p2)
	{
		double tmp0, tmp1, tmp2;
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

	double error = 0;
	int rows = objAfterTrans.rows;

	double3* arr_obj = new double3[rows];
	memcpy(arr_obj, (double3*)objAfterTrans.data, rows * sizeof(double3));
	double3* arr_mod = new double3[rows];
	memcpy(arr_mod, (double3*)mod.data, rows * sizeof(double3));
	double* arr_lambda = new double[rows];
	memcpy(arr_lambda, (double*)lambda.data, rows * sizeof(double));

	if (withCuda)
	{
		try
		{
			thrust::device_vector<double3> d_obj(arr_obj, arr_obj + rows);
			thrust::device_vector<double3> d_mod(arr_mod, arr_mod + rows);
			thrust::device_vector<double> d_lambda(
				arr_lambda, arr_lambda + rows);

			thrust::device_vector<double> tmp(rows);
			thrust::transform(d_obj.begin(), d_obj.end(), 
				d_mod.begin(), tmp.begin(), dist2_functor());
			thrust::transform(tmp.begin(), tmp.end(), 
				d_lambda.begin(), tmp.begin(), thrust::multiplies<double>());
			error = thrust::reduce(tmp.begin(), 
				tmp.end(), 0.0f, thrust::plus<double>());
		}
		catch (thrust::system_error e)
		{
			cout << "System Error: " << e.what() << endl;
		}
	}
	else
	{
		double error = 0;
		int rows = objAfterTrans.rows;

		double3* arr_obj = new double3[rows];
		memcpy(arr_obj, (double3*)objAfterTrans.data, rows * sizeof(double3));
		double3* arr_mod = new double3[rows];
		memcpy(arr_mod, (double3*)mod.data, rows * sizeof(double3));
		double* arr_lambda = new double[rows];
		memcpy(arr_lambda, (double*)lambda.data, rows * sizeof(double));

		try
		{
			thrust::host_vector<double3> h_obj(arr_obj, arr_obj + rows);
			thrust::host_vector<double3> h_mod(arr_mod, arr_mod + rows);
			thrust::host_vector<double> h_lambda(
				arr_lambda, arr_lambda + rows);

			thrust::host_vector<double> tmp(rows);
			thrust::transform(h_obj.begin(), h_obj.end(), 
				h_mod.begin(), tmp.begin(), dist2_functor());
			thrust::transform(tmp.begin(), tmp.end(), 
				h_lambda.begin(), tmp.begin(), thrust::multiplies<double>());
			error = thrust::reduce(tmp.begin(), 
				tmp.end(), 0.0f, thrust::plus<double>());
		}
		catch (thrust::system_error e)
		{
			cout << "System Error: " << e.what() << endl;
		}
	}

	delete[] arr_obj;
	delete[] arr_mod;
	delete[] arr_lambda;

	return error;
}