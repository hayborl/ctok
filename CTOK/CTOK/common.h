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
#pragma comment(lib, "libs/ANN.lib")

#include "triangulation.h"

#include "timer.h"

using namespace cv;
using namespace std;
using namespace gpu; 

#define EXTERN_C extern "C"

#define DISTANCE_MAX 100000000
#define DISTANCE_RANGE 100
#define DISTANCE_THRE 1500
#define OUTPUT	false

typedef unsigned int uint;

void saveData(const char* filename, InputArray _pts);
void saveData(const char* filename, const Triangulation::VertexVector pts);

bool initCuda();

bool isIdentity(const Mat &mat);

// 将m*n k通道转换为m*(n*k) 单通道
Mat convertMat(const Mat &mat);

void transformPointCloud(const Mat &input, Mat &output, 
	const Mat &transformMat, bool withCuda = false);

void plotTwoPoint3DSet(Mat objSet, Mat modSet);

// 根据已知点集估计出所有点拟合的平面的法向量
Vec3d computeNormal(ANNpointArray pts, ANNidxArray idxs, const int &k);

void simplifyPoints(Mat inPts, Mat &outPts, 
	const int &k, const double &alpha);

#endif