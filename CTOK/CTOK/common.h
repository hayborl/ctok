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

#include "ANN/ANN.h"
#pragma comment(lib, "ANN.lib")

using namespace cv;
using namespace std;
using namespace gpu; 

#define EXTERN_C extern "C"

#define DISTANCE_MAX 100000000
#define DISTANCE_RANGE 100
#define DISTANCE_THRE 1500
#define OUTPUT	true

typedef unsigned int uint;

class My_Timer : public TickMeter
{
public:
	enum TimeUnit{MICRO, MILLI, SEC};
	void start()
	{
		TickMeter::reset();
		TickMeter::start();
	}
	double stop(TimeUnit unit = SEC)
	{
		TickMeter::stop();
		switch (unit)
		{
		case MICRO:
			return getTimeMicro();
		case MILLI:
			return getTimeMilli();
		case SEC:
		default:
			return getTimeSec();
		}
	}
};

#define RUNANDTIME(timer, fun, output, s) timer.start(); fun; \
	if(output) cout << timer.stop() << "s " << s << endl;

void saveData(const char* filename, const Mat& mat, int flag = 0);

bool initCuda();

EXTERN_C void cuda_pushBackPoint(float* pSet1, float* pSet2,  
	const size_t size1, const size_t size2, Mat pointColor,
	vector<Vec3f>& v, vector<Vec3b>& c);

extern bool hasCuda;

static My_Timer global_timer;

typedef struct tag_Transformation
{
	Vec4f q;
	Vec3f t;
} Transformation;

// 将m*n k通道转换为m*(n*k) 单通道
Mat convertMat(const Mat &mat);

void transformPointCloud(Mat input, Mat* output, Mat transformMat);
EXTERN_C void cuda_transformPointCloud(Mat input, 
	Mat* output, Mat transformMat);

Mat getFeaturePointCloud(const Mat &colorImg, 
	const Mat &pointCloud, const Mat &pointIndices);

void getSurfPointsSet(const Mat &objColorImg, const Mat &objPointCloud, 
	const Mat &objPointIndex, const Mat &modColorImg, 
	const Mat &modPointCloud, const Mat &modPointIndex, 
	Mat* objSetOrigin, Mat* objSet, Mat* modSet, xn::DepthGenerator dg);

void plotTwoPoint3DSet(Mat objSet, Mat modSet);

void getRotateMatrix(Vec4f q, float* R);						// 将四元数转换为旋转矩阵

#endif