#include "features.h"

#include "opencv2/nonfree/nonfree.hpp"

using namespace gpu;

extern bool hasCuda;

void Features::getHSVColorHistDes( const Mat& image, Mat& descriptors )
{
	assert(image.channels() == 3);

	int rows = image.rows;
	int cols = image.cols;

	Mat tmp(rows, cols, image.type());
	cvtColor(image, tmp, CV_BGR2HSV);

	int hist[72];
	memset(hist, 0, 72 * sizeof(int));

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			Vec3b scalar = tmp.at<Vec3b>(i, j);
			double h = scalar[0] * 2;
			double s = scalar[1] / 255.0;
			double v = scalar[2] / 255.0;

			int H = convertH(h);
			int S = convertS(s);
			int V = convertV(v);

			int G = 9 * H + 3 * S + V;
			hist[G]++;
		}
	}

	int n = rows * cols;
	descriptors = Mat(1, 72, CV_32FC1);
	for (int i = 0; i < 72; i++)
	{
		float temp = (float)hist[i] / (float)n;
		descriptors.at<float>(0, i) = temp;
	}
}

void Features::getGLCMDes( const Mat& image, Mat& descriptors )
{
	double normGlcm[GRAY_LEVEL * GRAY_LEVEL];

	descriptors = Mat(4, 4, CV_32FC1);
	for (int i = 0; i < 4; i++)
	{
		getGLCM(image, normGlcm, i, 1);

		double glcm_asm = 0, glcm_ent = 0, 
			glcm_con = 0, glcm_idm = 0;
		for (int j = 0; j < GRAY_LEVEL * GRAY_LEVEL; j++)
		{
			int ii = j / GRAY_LEVEL;
			int ij = j % GRAY_LEVEL;
			int dd = (ii - ij) * (ii - ij);

			glcm_asm += normGlcm[j] * normGlcm[j];
			glcm_ent += -normGlcm[j] * log(normGlcm[j]+ 1e-7);
			glcm_con += dd * normGlcm[j];
			glcm_idm += normGlcm[j] / (1 + dd);
		}
		descriptors.at<float>(i, 0) = (float)glcm_asm;
		descriptors.at<float>(i, 1) = (float)glcm_ent;
		descriptors.at<float>(i, 2) = (float)glcm_con;
		descriptors.at<float>(i, 3) = (float)glcm_idm;
	}

	float totalSum = (float)sum(descriptors)[0];
	descriptors /= totalSum;
}

int Features::convertH( double h )
{
	if (h >= 316 || h <= 20)
	{
		return 0;
	}
	else if (h > 20 && h <= 40)
	{
		return 1;
	}
	else if (h > 40 && h <= 75)
	{
		return 2;
	}
	else if (h > 75 && h <= 155)
	{
		return 3;
	}
	else if (h > 155 && h <= 190)
	{
		return 4;
	}
	else if (h > 190 && h <= 270)
	{
		return 5;
	}
	else if (h > 270 && h <= 295)
	{
		return 6;
	}
	else
	{
		return 7;
	}
}

int Features::convertS( double s )
{
	if (s <= 0.2)
	{
		return 0;
	}
	else if (s > 0.2 && s <= 0.7)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}

int Features::convertV( double v )
{
	if (v <= 0.2)
	{
		return 0;
	}
	else if (v > 0.2 && v <= 0.7)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}

void Features::getGLCM( const Mat& image, double* normGlcm, 
	int orientation, int step )
{
	Mat tmp(image.rows, image.cols, image.type());
	if (image.channels() != 1)
	{
		cvtColor(image, tmp, CV_BGR2GRAY);
	}
	else
	{
		tmp = image.clone();
	}

	int dx = step, dy = 0;
	switch (orientation)
	{
	case ORIENTATION_0:
	default:
		dx = step;
		dy = 0;
		break;
	case ORIENTATION_45:
		dx = step; 
		dy = step;
		break;
	case ORIENTATION_90:
		dx = 0; 
		dy = step;
		break;
	case ORIENTATION_135:
		dx = -step;
		dy = -step;
	}

	int divided = 256 / GRAY_LEVEL;
	int glcm[GRAY_LEVEL][GRAY_LEVEL];
	memset(glcm, 0, GRAY_LEVEL * GRAY_LEVEL * sizeof(int));
	int totalNum = 0;

	for (int i = 0; i < tmp.rows; i++)
	{
		if (i + dy >= tmp.rows || i + dy < 0)
		{
			continue;
		}
		for (int j = 0; j < tmp.cols; j++)
		{
			if (j + dx >= tmp.cols || j + dy < 0)
			{
				continue;
			}
			uchar tmp1 = tmp.at<uchar>(i, j) / GRAY_LEVEL;
			uchar tmp2 = tmp.at<uchar>(i + dy, j + dx) / GRAY_LEVEL;
			glcm[tmp1][tmp2]++;
			totalNum++;
		}
	}

	for (int i = 0; i < GRAY_LEVEL; i++)
	{
		for (int j = 0; j < GRAY_LEVEL; j++)
		{
			normGlcm[i * GRAY_LEVEL + j] = 
				(double)glcm[i][j] / (double)totalNum;
		}
	}
}

void getSurfPointsSet( const Mat& objColorImg, const Mat& objPointCloud, 
	const Mat& objPointIndex, const Mat& modColorImg, 
	const Mat& modPointCloud, const Mat& modPointIndex, 
	Mat& objSetOrigin, Mat& objSet, Mat& modSet, xn::DepthGenerator dg)
{
	Mat colorImgs[2], keyPointsImgs[2], descriptors[2];
	vector<KeyPoint> keyPoints[2];
	vector<DMatch> matches;

	colorImgs[0] = objColorImg;
	colorImgs[1] = modColorImg;

	if (hasCuda)
	{
		GpuMat imgsColorGpu0, imgsColorGpu1;
		GpuMat imgsGpu0, imgsGpu1;
		imgsColorGpu0.upload(objColorImg);
		imgsColorGpu1.upload(modColorImg);
		cvtColor(imgsColorGpu0, imgsGpu0, CV_BGR2GRAY);
		cvtColor(imgsColorGpu1, imgsGpu1, CV_BGR2GRAY);

		SURF_GPU surfGpu;
		GpuMat keyPointsGpu0, keyPointsGpu1;
		GpuMat desGpu0, desGpu1;
		surfGpu(imgsGpu0, GpuMat(), keyPointsGpu0, desGpu0);
		surfGpu(imgsGpu1, GpuMat(), keyPointsGpu1, desGpu1);

		BFMatcher_GPU matcher;
		GpuMat trainIdx, distance;
		matcher.matchSingle(desGpu0, desGpu1, trainIdx, distance);

		surfGpu.downloadKeypoints(keyPointsGpu0, keyPoints[0]);
		surfGpu.downloadKeypoints(keyPointsGpu1, keyPoints[1]);
		BFMatcher_GPU::matchDownload(trainIdx, distance, matches);
	}
	else
	{
		// 特征算子
		Ptr<Feature2D> surf;
		DescriptorMatcher* matcher = new FlannBasedMatcher;

		initModule_nonfree();
		surf = Algorithm::create<Feature2D>("Feature2D.SURF");

#pragma omp parallel for
		for (int i = 0; i < 2; i++)
		{
			// 检测关键点
			surf->detect(colorImgs[i], keyPoints[i]);

			// 计算特征描述子
			surf->compute(colorImgs[i], keyPoints[i], descriptors[i]);
		}

		// 匹配描述子
		matcher->match(descriptors[0], descriptors[1], matches);
	}

	int ptCount = (int)matches.size();
	vector<Point2f> points0;
	vector<Point2f> points1;
	double minDist = 100000;
	double maxDist = 0;

	for (int i = 0; i < ptCount; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist)
		{
			minDist = dist;
		}
		if (dist > maxDist)
		{
			maxDist = dist;
		}
	}
	for (int i = 0; i < ptCount; i++)
	{
		double dist = matches[i].distance;
		if (dist < (minDist + maxDist) / 2)
		{
			points0.push_back(keyPoints[0][matches[i].queryIdx].pt);
			points1.push_back(keyPoints[1][matches[i].trainIdx].pt);
		}
	}
	// 用RANSAC方法计算基本矩阵
	Mat tmp = findHomography(points0, points1, CV_RANSAC).clone();

	vector<Point2f> tmpKeyPoints, transPoints;
	KeyPoint::convert(keyPoints[0], tmpKeyPoints);
	perspectiveTransform(tmpKeyPoints, transPoints, tmp);

	vector<Point3f> modTmpSet, objTmpSetOrign;
	Ptr<XnPoint3D> proj = new XnPoint3D[objPointCloud.rows];
	Ptr<XnPoint3D> real = new XnPoint3D[objPointCloud.rows];
	int cnt = 0;
	for (int i = 0; i < tmpKeyPoints.size(); i++)
	{
		int x = (int)(tmpKeyPoints[i].x);
		int y = (int)(tmpKeyPoints[i].y);
		int index = objPointIndex.at<int>(y, x);
		if (index != -1)
		{
			Point3f p = objPointCloud.at<Point3f>(index, 0);
			// 			if (p == Point3f(0, 0, 0))
			// 			{
			// 				continue;
			// 			}
			objTmpSetOrign.push_back(p);
			Point2f p2d = transPoints[i];
			proj[cnt].X = p2d.x;
			proj[cnt].Y = p2d.y;
			proj[cnt].Z = p.z;
			cnt++;
		}
	}
	dg.ConvertProjectiveToRealWorld(cnt, proj, real);

	objSet = Mat(cnt, 1, DataType<Point3f>::type);
#pragma omp parallel for
	for (int i = 0; i < cnt; i++)
	{
		objSet.at<Point3f>(i, 0) = Point3f(real[i].X, real[i].Y, real[i].Z);
	}

	for (int i = 0; i < keyPoints[1].size(); i++)
	{
		int x = (int)(keyPoints[1][i].pt.x);
		int y = (int)(keyPoints[1][i].pt.y);
		int index = modPointIndex.at<int>(y, x);
		if (index != -1)
		{
			modTmpSet.push_back(modPointCloud.at<Point3f>(index, 0));
		}
	}

	objSetOrigin = Mat(objTmpSetOrign, true);
	modSet = Mat(modTmpSet, true);
}

void getFeaturePoints(xn::DepthGenerator dg,				
	const Mat &colorImgNow, const Mat &depthImgNow,		
	const Mat &colorImgPre, const Mat &depthImgPre,		
	Mat &objSet, Mat &modSet, Mat &objSetAT, Mat &mask)
{
	Mat colorImgs[2], keyPointsImgs[2], descriptors[2];
	vector<KeyPoint> keyPoints[2];
	vector<DMatch> matches;

	colorImgs[0] = colorImgNow;
	colorImgs[1] = colorImgPre;

	if (hasCuda)
	{
		GpuMat imgsColorGpu0, imgsColorGpu1;
		GpuMat imgsGpu0, imgsGpu1;
		imgsColorGpu0.upload(colorImgNow);
		imgsColorGpu1.upload(colorImgPre);
		cvtColor(imgsColorGpu0, imgsGpu0, CV_BGR2GRAY);
		cvtColor(imgsColorGpu1, imgsGpu1, CV_BGR2GRAY);

		SURF_GPU surfGpu;
		GpuMat keyPointsGpu0, keyPointsGpu1;
		GpuMat desGpu0, desGpu1;
		surfGpu(imgsGpu0, GpuMat(), keyPointsGpu0, desGpu0);
		surfGpu(imgsGpu1, GpuMat(), keyPointsGpu1, desGpu1);

		BFMatcher_GPU matcher;
		GpuMat trainIdx, distance;
		matcher.matchSingle(desGpu0, desGpu1, trainIdx, distance);

		surfGpu.downloadKeypoints(keyPointsGpu0, keyPoints[0]);
		surfGpu.downloadKeypoints(keyPointsGpu1, keyPoints[1]);
		BFMatcher_GPU::matchDownload(trainIdx, distance, matches);
	}
	else
	{
		// 特征算子
		Ptr<Feature2D> surf;
		DescriptorMatcher* matcher = new FlannBasedMatcher;

		initModule_nonfree();
		surf = Algorithm::create<Feature2D>("Feature2D.SURF");

#pragma omp parallel for
		for (int i = 0; i < 2; i++)
		{
			// 检测关键点
			surf->detect(colorImgs[i], keyPoints[i]);

			// 计算特征描述子
			surf->compute(colorImgs[i], keyPoints[i], descriptors[i]);
		}

		// 匹配描述子
		matcher->match(descriptors[0], descriptors[1], matches);
	}

	int ptCount = (int)matches.size();
	vector<Point2f> points0;
	vector<Point2f> points1;
	double minDist = 100000;
	double maxDist = 0;

	// 求最大最小距离
	for (int i = 0; i < ptCount; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist)
		{
			minDist = dist;
		}
		if (dist > maxDist)
		{
			maxDist = dist;
		}
	}
	// 去除距离过大的特征点
	for (int i = 0; i < ptCount; i++)
	{
		double dist = matches[i].distance;
		if (dist < (minDist + maxDist) / 2)
		{
			points0.push_back(keyPoints[0][matches[i].queryIdx].pt);
			points1.push_back(keyPoints[1][matches[i].trainIdx].pt);
		}
	}
	// 用RANSAC方法计算基本矩阵
	Mat H = findHomography(points0, points1, CV_RANSAC).clone();

	// 将特征点通过基本矩阵转换
	vector<Point2f> tmpKeyPoints, transPoints;
	KeyPoint::convert(keyPoints[0], tmpKeyPoints);
	perspectiveTransform(tmpKeyPoints, transPoints, H);

	int rows = colorImgNow.rows;
	int cols = colorImgNow.cols;
	int size = rows * cols;

	mask = Mat(colorImgs[1].rows, colorImgs[1].cols, CV_8UC1, Scalar::all(0));
	warpPerspective(mask, mask, H, Size(cols, rows),
		INTER_LINEAR + WARP_INVERSE_MAP, 0, Scalar::all(255));

	vector<Point3f> modTmpSet, objTmpSet, objTmpSetAT;
	Ptr<XnPoint3D> proj = new XnPoint3D[size];
	Ptr<XnPoint3D> real = new XnPoint3D[size];
	Ptr<XnPoint3D> projAT = new XnPoint3D[size];
	Ptr<XnPoint3D> realAT = new XnPoint3D[size];
	int cnt = 0;
	for (int i = 0; i < tmpKeyPoints.size(); i++)
	{
		int x = (int)(tmpKeyPoints[i].x);
		int y = (int)(tmpKeyPoints[i].y);
		ushort z = depthImgNow.at<ushort>(y, x);
		if (z != 0)
		{
			proj[cnt].X = (float)x;
			proj[cnt].Y = (float)y;
			proj[cnt].Z = (float)z;

			Point2f p2d = transPoints[i];
			projAT[cnt].X = p2d.x;
			projAT[cnt].Y = p2d.y;
			projAT[cnt].Z = (float)z;
			cnt++;
		}
	}
	dg.ConvertProjectiveToRealWorld(cnt, proj, real);
	dg.ConvertProjectiveToRealWorld(cnt, projAT, realAT);

	objSet = Mat(cnt, 1, DataType<Point3f>::type);
	objSetAT = Mat(cnt, 1, DataType<Point3f>::type);
#pragma omp parallel for
	for (int i = 0; i < cnt; i++)
	{
		objSet.at<Point3f>(i, 0) = Point3f(real[i].X, real[i].Y, real[i].Z);
		objSetAT.at<Point3f>(i, 0) = Point3f(
			realAT[i].X, realAT[i].Y, realAT[i].Z);
	}

	cnt = 0;
	for (int i = 0; i < keyPoints[1].size(); i++)
	{
		int x = (int)(keyPoints[1][i].pt.x);
		int y = (int)(keyPoints[1][i].pt.y);
		ushort z = depthImgPre.at<ushort>(y, x);
		if (z != 0)
		{
			proj[cnt].X = (float)x;
			proj[cnt].Y = (float)y;
			proj[cnt].Z = (float)z;
			cnt++;
		}
	}
	dg.ConvertProjectiveToRealWorld(cnt, proj, real);
	modSet = Mat(cnt, 1, DataType<Point3f>::type);
#pragma omp parallel for
	for (int i = 0; i < cnt; i++)
	{
		modSet.at<Point3f>(i, 0) = Point3f(real[i].X, real[i].Y, real[i].Z);
	}
}