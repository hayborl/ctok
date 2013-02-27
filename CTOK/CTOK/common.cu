#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "common.h"

__device__ int cntd[1];

__global__ void cuda_findNeighbor(float* pSet, float3 p, 
	const size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size && cntd[0] == 0)
	{
		int offset = idx * 3;
		float sum = 0;
		float tmp = p.x - pSet[offset];
		sum += tmp * tmp;
		tmp = p.y - pSet[offset + 1];
		sum += tmp * tmp;
		tmp = p.z - pSet[offset + 2];
		sum += tmp * tmp;
		if (sum < DISTANCE_RANGE)
		{
			cntd[0]++;
		}
	}
}

EXTERN_C void cuda_pushBackPoint(float* pSet1, float* pSet2,  
	const size_t size1, const size_t size2, Mat pointColor,
	vector<Vec3f>& v, vector<Vec3b>& c)
{
	float* pSet1d;
	size_t copySize = size1 * 3 * sizeof(float);
	if (size1 > 0)
	{
		cudaMalloc((void**)&pSet1d, copySize);
		cudaMemcpy(pSet1d, pSet1, copySize, cudaMemcpyHostToDevice);
	}

	Point3f p;
	Vec3b color;
	Vec3f vp;
	for (size_t i = 0; i < size2; i ++/*= SAMPLE_INTERVAL*/)
	{
		size_t off = i * 3;
		p = Point3f(pSet2[off], pSet2[off + 1], pSet2[off + 2]);
		if (p != Point3f(0, 0, 0))
		{
			vp = Vec3f(p);
			vp[2] = -vp[2];
			bool flag = false;

			if (size1 > 0)
			{
				float3 pd;
				pd.x = vp[0];
				pd.y = vp[1];
				pd.z = vp[2];

				int cnt = 0;
// 				cudaMemcpyToSymbol(cntd, &cnt, sizeof(int));
// 				cuda_findNeighbor<<<size1 / BLOCK_SIZE + 1, 
// 					BLOCK_SIZE>>>(pSet1d, pd, size1);
// 				cudaMemcpyFromSymbol(&cnt, cntd, sizeof(int));
				
				flag = (cnt > 0);
			}
			if (!flag)
			{
				v.push_back(vp);
				color = pointColor.at<Vec3b>((int)i, 0);
				c.push_back(Vec3b(color[2], color[1], color[0]));
			}
		}
	}

	if (size1 > 0)
	{
		cudaFree(pSet1d);
	}
}