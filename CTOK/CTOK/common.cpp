#include "cuda_runtime.h"

#include "opencv2/nonfree/nonfree.hpp"
#include "common.h"

bool hasCuda = true;

void saveData(const char* filename, const Mat& mat, int flag)      
{
	FILE* fp;
	fopen_s(&fp, filename, "wt");
	if (3 != flag)
	{
		fprintf(fp, "%02d\n", mat.rows);
		fprintf(fp, "%02d\n", mat.cols);
	}
	switch (flag)
	{
	case 0:
		for(int y = 0; y < mat.rows; y++)
		{
			for(int x = 0; x < mat.cols; x++)
			{
				short depth = mat.at<short>(y, x);   
				fprintf(fp, "%d\n", depth);
			}
		}
		break;
	case 1:
		for(int y = 0; y < mat.rows; y++)
		{
			for(int x = 0; x < mat.cols; x++)
			{
				uchar disp = mat.at<uchar>(y,x);
				fprintf(fp, "%d\n", disp);
			}
		}
		break;
	case 2:
		for(int y = 0; y < mat.rows; y++)
		{
			for(int x = 0; x < mat.cols; x++)
			{
				float disp = mat.at<float>(y,x);
				fprintf(fp, "%10.4f\n", disp);
			}
		}
		break;
	case 3:
		for(int y = 0; y < mat.rows; y++)
		{
			for(int x = 0; x < mat.cols; x++)
			{
				Vec3f point = mat.at<Vec3f>(y, x);   // Vec3f 是 template 类定义
				fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
			}
		}
		break;
	case 4:
		imwrite(filename, mat);
		break;
	default:
		break;
	}

	fclose(fp);
}

//CUDA初始化
bool initCuda()
{
	int count; 
	//传回有计算能力的设备数(≥1)，没有回传回1，device 0是一个仿真装置，不支持CUDA功能
	cudaGetDeviceCount(&count);

	if(count == 0) //没有cuda计算能力的设备
	{
		fprintf(stderr,"There is no device.\n");
		return false;
	}

	int i;
	for(i=0;i<count;i++)
	{
		cudaDeviceProp prop; //设备属性
		if (cudaGetDeviceProperties(&prop,i)==cudaSuccess) //取得设备数据，brief Returns information about the compute-device
		{
			if (prop.major>=1) //cuda计算能力
			{
				break;
			}
		}
	}

	if (i==count)
	{
		fprintf(stderr,"There is no device supporting CUDA 2.x\n");
		return false;
	}

	cudaSetDevice(i); //brief Set device to be used for GPU executions
	return true;
}

Mat convertMat( const Mat &mat )
{
	Mat* subs = new Mat[mat.channels()];
	split(mat, subs);

	Mat tmp(mat.rows, mat.cols * mat.channels(), CV_32FC1);
#pragma omp parallel for
	for (int i = 0; i < mat.channels(); i++)
	{
		Mat roi = tmp(Rect(i, 0, 1, mat.rows));
		subs[i].copyTo(roi);
	}

	return tmp.clone();
}

void transformPointCloud(Mat input, Mat* output, Mat transformMat)
{
	*output = input.clone();
	Mat R = transformMat(Rect(0, 0, 3, 3));
	Mat T = transformMat(Rect(3, 0, 1, 3));
	Mat tmp = input.clone();
#pragma omp parallel for
	for (int i = 0; i < tmp.rows; i++)
	{
		for (int j = 0; j < tmp.cols; j++)
		{
			Point3f p = tmp.at<Point3f>(i, j);
			Mat point(p);
			point = R * point + T;
			output->at<Point3f>(i, j) = Point3f(point);
		}
	}
}

Mat getFeaturePointCloud( const Mat &colorImg, 
	const Mat &pointCloud, const Mat &pointIndices )
{
	Mat keyPointsImg;
	vector<KeyPoint> keyPoints;
	FeatureDetector* detector = new FastFeatureDetector;

	initModule_nonfree();

	detector->detect(colorImg, keyPoints); // 检测关键点
	// 	drawKeypoints(colorImg, keyPoints, keyPointsImg, 
	// 		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	// 	imshow("KeyPoint", keyPointsImg);
	// 	waitKey();

	vector<Point3f> tmpSet;
	for (int i = 0; i < keyPoints.size(); i++)
	{
		int x = (int)(keyPoints[i].pt.x);
		int y = (int)(keyPoints[i].pt.y);
		int index = pointIndices.at<int>(y, x);
		if (index != -1)
		{
			tmpSet.push_back(pointCloud.at<Point3f>(index, 0));
		}
	}

	return Mat(tmpSet, true).clone();
}

void getSurfPointsSet( const Mat &objColorImg, const Mat &objPointCloud, 
	const Mat &objPointIndex, const Mat &modColorImg, 
	const Mat &modPointCloud, const Mat &modPointIndex, 
	Mat* objSetOrigin, Mat* objSet, Mat* modSet, xn::DepthGenerator dg)
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

	// 把Keypoint转换为Mat
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

	vector<Point3f> objTmpSet, modTmpSet, objTmpSetOrign;
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
			objTmpSetOrign.push_back(p);
			Point2f p2d = transPoints[i];
			proj[cnt].X = p2d.x;
			proj[cnt].Y = p2d.y;
			proj[cnt].Z = p.z;
			cnt++;
		}
	}
	dg.ConvertProjectiveToRealWorld(cnt, proj, real);
	for (int i = 0; i < cnt; i++)
	{
		objTmpSet.push_back(Point3f(real[i].X, real[i].Y, real[i].Z));
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

	*objSetOrigin = Mat(objTmpSetOrign, true);
	*objSet = Mat(objTmpSet, true);
	*modSet = Mat(modTmpSet, true);
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
			int x = (int)((objSet.at<Point3f>(i, j).x - minX) * 500 / (maxX - minX));
			int y = (int)((maxY - objSet.at<Point3f>(i, j).y) * 375 / (maxY - minY));
			pointImg.at<Vec3b>(y, x) = Vec3b(0, 0, 255);

			x = (int)((objSet.at<Point3f>(i, j).x - minX) * 500 / (maxX - minX));
			y = (int)((maxZ - objSet.at<Point3f>(i, j).z) * 375 / (maxZ - minZ)) + 375;
			pointImg.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
		}
	}
	for (int i = 0; i < modSet.rows; i++)
	{
		for (int j = 0; j < modSet.cols; j++)
		{
			int x = (int)((modSet.at<Point3f>(i, j).x - minX) * 500 / (maxX - minX));
			int y = (int)((maxY - modSet.at<Point3f>(i, j).y) * 375 / (maxY - minY));
			pointImg.at<Vec3b>(y, x) += Vec3b(0, 255, 0);

			x = (int)((modSet.at<Point3f>(i, j).x - minX) * 500 / (maxX - minX));
			y = (int)((maxZ - modSet.at<Point3f>(i, j).z) * 375 / (maxZ - minZ)) + 375;
			pointImg.at<Vec3b>(y, x) += Vec3b(0, 255, 0);
		}
	}
	Mat seperator(1, 1024, CV_8UC3, Scalar::all(255));
	Mat roi = pointImg(Rect(0, 375, 1024, 1));
	seperator.copyTo(roi);
	imshow("Point Cloud", pointImg);
	waitKey();
}

void getRotateMatrix( Vec4f q, float* R )
{
	float q0 = q[0];
	float q1 = q[1];
	float q2 = q[2];
	float q3 = q[3];

	R[0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
	R[1] = 2 * (q1 * q2 - q0 * q3);
	R[2] = 2 * (q1 * q3 + q0 * q2);
	R[3] = 2 * (q1 * q2 + q0 * q3);
	R[4] = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3;
	R[5] = 2 * (q2 * q3 - q0 * q1);
	R[6] = 2 * (q1 * q3 - q0 * q2);
	R[7] = 2 * (q2 * q3 + q0 * q1);
	R[8] = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2;
}