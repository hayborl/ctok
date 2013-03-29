#ifndef COMMON_H
#define COMMON_H

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/opencv.hpp"
#include "XnCppWrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <assert.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/find.h>

#include "ANN/ANN.h"
#pragma comment(lib, "ANN.lib")

#include "timer.h"

using namespace cv;
using namespace std;
using namespace gpu; 

#define EXTERN_C extern "C"

#define DISTANCE_MAX 100000000
#define DISTANCE_RANGE 100
#define DISTANCE_THRE 1500
#define OUTPUT	true

typedef unsigned int uint;

void saveData(const char* filename, const Mat& mat, int flag = 0);
void saveData(const char* filename, const vector<Vec3f> pts);

bool initCuda();

// 将m*n k通道转换为m*(n*k) 单通道
Mat convertMat(const Mat& mat);

void transformPointCloud(Mat input, Mat* output, 
	Mat transformMat, bool withCuda = false);

void plotTwoPoint3DSet(Mat objSet, Mat modSet);

void getRotateMatrix(Vec4f q, float* R);							// 将四元数转换为旋转矩阵

double bhattacharyyaDistance(const Mat& mat1, const Mat& mat2);	// 计算巴氏距离
double computeDistance(pair<Mat, Mat> des1, pair<Mat, Mat> des2);	// 根据描述子计算相似度

// 根据已知点集估计出所有点拟合的平面的法向量
Vec3f computeNormal(ANNpointArray pts, ANNidxArray idxs, const int& k);

void simplifyPoints(Mat inPts, Mat& outPts, 
	const int& k, const double& alpha);

#endif