__global__ void cuda_getDistance(const float* ps, const float* p, 
	const int count, float* dis, int* idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < count)
	{
		float distance = 0;
		int offset = i * 3;
		for (int j = 0; j < 3; j++)
		{
			distance += (p[j] - ps[offset + j]) * (p[j] - ps[offset + j]);
		}
		dis[i] = distance;
		idx[i] = i;
	}
}

__global__ void cuda_oddSort(float* arr, int* idx, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i + 1) % 2 == 0)
	{
		if(arr[i] > arr[i + 1] && i + 1 < size)  
		{  
			float tp = arr[i];  
			arr[i] = arr[i + 1];  
			arr[i + 1] = tp;
			int ti = idx[i];
			idx[i] = idx[i + 1];
			idx[i + 1] = ti;
		}
	}
}

__global__ void cuda_evenSort(float* arr, int* idx, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i + 1) % 2 == 1)
	{
		if(arr[i] > arr[i + 1] && i + 1 < size)  
		{  
			float tp = arr[i];  
			arr[i] = arr[i + 1];  
			arr[i + 1] = tp;
			int ti = idx[i];
			idx[i] = idx[i + 1];
			idx[i + 1] = ti;
		}
	}
}

void cuda_sort(float* arr, int* idx, int size)
{
	float* arrd;
	int s = size * sizeof(float);
	__checkCudaErrors(cudaMalloc((void**)&arrd, s), fileName, 167);
	__checkCudaErrors(cudaMemcpy(arrd, arr, s, 
		cudaMemcpyHostToDevice), fileName, 168);
	int* idxd;
	s = size * sizeof(int);
	__checkCudaErrors(cudaMalloc((void**)&idxd, s), fileName, 172);
	__checkCudaErrors(cudaMemcpy(idxd, idx, s, 
		cudaMemcpyHostToDevice), fileName, 173);
	int gridSize = size / BLOCK_SIZE + 1;
	for (int i = 0; i < size; i++)
	{
		cuda_evenSort<<<gridSize, BLOCK_SIZE>>>(arrd, idxd, size);
		cuda_oddSort<<<gridSize, BLOCK_SIZE>>>(arrd, idxd, size);
	}
	s = size * sizeof(float);
	__checkCudaErrors(cudaMemcpy(arr, arrd, s, cudaMemcpyDeviceToHost), 
		fileName, 182);
	s = size * sizeof(int);
	__checkCudaErrors(cudaMemcpy(idx, idxd, s, cudaMemcpyDeviceToHost), 
		fileName, 185);

	cudaFree(arrd);
	cudaFree(idxd);
}

void cuda_getNearest(const float* p, const float* ps, 
	const int size, float* cp, float& dis)
{
	float* pd;
	int s = 3 * sizeof(float);
	cudaMalloc((void**)&pd, s);
	cudaMemcpy(pd, p, s, cudaMemcpyHostToDevice);
	float* psd;
	s = size * 3 * sizeof(float);
	cudaMalloc((void**)&psd, s);
	cudaMemcpy(psd, ps, s, cudaMemcpyHostToDevice);
	float* disd;
	s = size * sizeof(float);
	cudaMalloc((void**)&disd, s);
	int* idxd;
	s = size * sizeof(int);
	cudaMalloc((void**)&idxd, s);

	int gridSize = size / BLOCK_SIZE + 1;
	cuda_getDistance<<<gridSize, BLOCK_SIZE>>>(psd, pd, size, disd, idxd);
	float* diss = new float[size];
	int* idx = new int[size];
	s = size * sizeof(float);
	cudaMemcpy(diss, disd, s, cudaMemcpyDeviceToHost);
	s = size * sizeof(int);
	cudaMemcpy(idx, idxd, s, cudaMemcpyDeviceToHost);
/*	cuda_sort(diss, idx, size);*/
	int offset = idx[0] * 3;
	cp[0] = ps[offset]; cp[1] = ps[offset + 1]; cp[2] = ps[offset + 2];
	dis = diss[0];

	cudaFree(pd);
	cudaFree(psd);
	cudaFree(disd);
	cudaFree(idxd);

	delete[] diss;
	delete[] idx;
}

EXTERN_C void cuda_getClosestPoints(const Mat &objSet, const Mat &modSet,
	vector<double> &diss, double &sum, Mat* resSet)
{
	int orows = objSet.rows;
	int mrows = modSet.rows;
	float* mp = new float[mrows * 3];
	memcpy(mp, (float*)modSet.data, mrows * 3 * sizeof(float));
	for (int i = 0; i < orows; i++)
	{
		float op[3], cp[3], distance;
		Vec3f p = Vec3f(objSet.at<Point3f>(i, 0));
		op[0] = p[0]; op[1] = p[1], op[2] = p[2];
		cuda_getNearest(op, mp, mrows, cp, distance);

		resSet->at<Point3f>(i, 0) = Point3f(Mat(3, 1, CV_32FC1, cp));
		diss[i] = distance;
		sum += distance;
	}
	sum /= (double)orows;

	delete[] mp;
}