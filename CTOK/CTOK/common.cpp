#include "cuda_runtime.h"

#include "common.h"

void saveData(const char* filename, InputArray _pts)
{
	Mat pts = _pts.getMat();
	assert(pts.type() == CV_64FC3);
	assert(pts.isContinuous());
	int total = (int)pts.total();

	Vec3d* ptr = (Vec3d*)pts.data;

	FILE* fp;
	fopen_s(&fp, filename, "wt");
	fprintf(fp, "%d\n", total);
	for (int i = 0; i < total; i++)
	{
		Vec3d point = ptr[i];
		fprintf(fp, "%f,%f,%f\n", point[0], point[1], point[2]);
	}
	fclose(fp);
}

void saveData(const char* filename, const Triangulation::VertexVector pts)
{
	FILE* fp;
	fopen_s(&fp, filename, "wt");
	fprintf(fp, "%d\n", pts.size());
	for (int i = 0; i < pts.size(); i++)
	{
		Triangulation::Vertex point = pts[i];
		fprintf(fp, "%f,%f,%f\n", point.m_xyz[0], 
			point.m_xyz[1], point.m_xyz[2]);
	}
	fclose(fp);
}

bool isIdentity( const Mat &mat )
{
	if (mat.rows != mat.cols)
	{
		return false;
	}
	Mat identityMat = Mat::eye(mat.size(), mat.type());
	Mat cmp;
	compare(mat, identityMat, cmp, CMP_NE);
	return countNonZero(cmp) == 0;
}

//CUDA初始化
bool initCuda()
{
	int count; 
	//传回有计算能力的设备数(≥1)，没有回传回1，device 0是一个仿真装置，不支持CUDA功能
	cudaError_t errorId = cudaGetDeviceCount(&count);
	if (errorId != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", 
			(int)errorId, cudaGetErrorString(errorId));
		return false;
	}

	if (count == 0) //没有cuda计算能力的设备
	{
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for (i = 0; i < count; i++)
	{
		cudaDeviceProp prop; //设备属性
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) //取得设备信息
		{
			if (prop.major >= 1) //cuda计算能力
			{
				break;
			}
		}
	}

	if (i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA 1.x\n");
		return false;
	}

	cudaSetDevice(i); //设置cuda运行的GPU设备
	return true;
}

Mat convertMat( const Mat &mat )
{
	Mat* subs = new Mat[mat.channels()];
	split(mat, subs);

	Mat tmp(mat.rows, mat.cols * mat.channels(), CV_64FC1);
#pragma omp parallel for
	for (int i = 0; i < mat.channels(); i++)
	{
		Mat roi = tmp(Rect(i, 0, 1, mat.rows));
		subs[i].copyTo(roi);
	}

	return tmp.clone();
}

void plotTwoPoint3DSet( Mat objSet, Mat modSet )
{
	Mat pointImg = Mat::zeros(768, 1024, CV_8UC3);

	Mat tmp[3];
	split(objSet, tmp);
	double maxX1, minX1, maxY1, minY1, minZ1, maxZ1;
	double maxX2, minX2, maxY2, minY2, minZ2, maxZ2;
	minMaxIdx(tmp[0], &minX1, &maxX1);
	minMaxIdx(tmp[1], &minY1, &maxY1);
	minMaxIdx(tmp[2], &minZ1, &maxZ1);

	split(modSet, tmp);
	minMaxIdx(tmp[0], &minX2, &maxX2);
	minMaxIdx(tmp[1], &minY2, &maxY2);
	minMaxIdx(tmp[2], &minZ2, &maxZ2);

	double maxX = maxX1 > maxX2 ? maxX1 : maxX2;
	double maxY = maxY1 > maxY2 ? maxY1 : maxY2;
	double maxZ = maxZ1 > maxZ2 ? maxZ1 : maxZ2;
	double minX = minX1 < minX2 ? minX1 : minX2;
	double minY = minY1 < minY2 ? minY1 : minY2;
	double minZ = minZ1 < minZ2 ? minZ1 : minZ2;

	for (int i = 0; i < objSet.rows; i++)
	{
		for (int j = 0; j < objSet.cols; j++)
		{
			int x = (int)((objSet.at<Point3d>(i, j).x - minX) * 500 / (maxX - minX));
			int y = (int)((maxY - objSet.at<Point3d>(i, j).y) * 375 / (maxY - minY));
			pointImg.at<Vec3b>(y, x) = Vec3b(0, 0, 255);

			x = (int)((objSet.at<Point3d>(i, j).x - minX) * 500 / (maxX - minX));
			y = (int)((maxZ - objSet.at<Point3d>(i, j).z) * 375 / (maxZ - minZ)) + 375;
			pointImg.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
		}
	}
	for (int i = 0; i < modSet.rows; i++)
	{
		for (int j = 0; j < modSet.cols; j++)
		{
			int x = (int)((modSet.at<Point3d>(i, j).x - minX) * 500 / (maxX - minX));
			int y = (int)((maxY - modSet.at<Point3d>(i, j).y) * 375 / (maxY - minY));
			pointImg.at<Vec3b>(y, x) += Vec3b(0, 255, 0);

			x = (int)((modSet.at<Point3d>(i, j).x - minX) * 500 / (maxX - minX));
			y = (int)((maxZ - modSet.at<Point3d>(i, j).z) * 375 / (maxZ - minZ)) + 375;
			pointImg.at<Vec3b>(y, x) += Vec3b(0, 255, 0);
		}
	}
	Mat seperator(1, 1024, CV_8UC3, Scalar::all(255));
	Mat roi = pointImg(Rect(0, 375, 1024, 1));
	seperator.copyTo(roi);
	imshow("Point Cloud", pointImg);
	waitKey();
}

Vec3d computeNormal( ANNpointArray pts, 
	ANNidxArray idxs, const int &k, Mat &barycenter)
{
	Mat M = Mat::zeros(3, 3, CV_64FC1);
	Mat mean = Mat::zeros(3, 1, CV_64FC1);
	for (int i = 0; i < k; i++)
	{
		Mat pMat(3, 1, CV_64FC1, pts[idxs[i]]);
		M += pMat * pMat.t();
		mean += pMat;
	}
	M /= (double)k;
	mean /= (double)k;
	M -= mean * mean.t();

	barycenter = mean.clone();

	Mat eigenValues(3, 1, CV_64FC1);
	Mat eigenVector(3, 3, CV_64FC1);
	eigen(M, eigenValues, eigenVector);

	int minEigenIndex = 0;
	minMaxIdx(eigenValues, NULL, NULL, &minEigenIndex, NULL);

	return normalize(Vec3d(eigenVector.row(minEigenIndex)));
}

void simplifyPoints( Mat inPts, Mat& outPts, 
	const int& k, const double& alpha )
{
	int num = inPts.rows;
	assert(num > 0);

	ANNpointArray pts = annAllocPts(num, 3);
#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		Vec3d p = inPts.at<Vec3d>(i, 0);
		pts[i][0] = p[0];
		pts[i][1] = p[1];
		pts[i][2] = p[2];
	}
	ANNkd_tree* kdTree = new ANNkd_tree(pts, num, 3);

	ANNidxArray idxs = new ANNidx[k];
	ANNdistArray dists = new ANNdist[k];

	Mat sumdMat(num, 1, CV_64FC1);
	for (int i = 0; i < num; i++)
	{
		ANNpoint q = pts[i];
		kdTree->annkSearch(q, k, idxs, dists);
		Mat barycenter;
		Vec3d normalVector = computeNormal(pts, idxs, k, barycenter);
		double sumd = 0;
		for (int j = 0; j < k; j++)
		{
			ANNpoint p = pts[idxs[j + 1]];
			Vec3d v(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
			sumd += fabs(v.ddot(normalVector));
		}
		sumdMat.at<double>(i, 0) = sumd;
	}

	double thres = alpha * sum(sumdMat)[0] / (double)num;
	outPts = inPts.clone();
	int cnt = 0;
#pragma omp parallel for
	for (int i = 0; i < num; i++)
	{
		if (sumdMat.at<double>(i, 0) < thres)
		{
			outPts.at<Vec3d>(i, 0) = Vec3d(0, 0, 0);
			cnt++;
		}
	}
	cout << "pre:" << num << " now:" << num - cnt << endl;
}