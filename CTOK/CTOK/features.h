#ifndef FEATURES_H
#define FEATURES_H

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/opencv.hpp"
#include "XnCppWrapper.h"
using namespace cv;

#define GRAY_LEVEL 16

#define ORIENTATION_0	0
#define ORIENTATION_45	1 
#define ORIENTATION_90	2
#define ORIENTATION_135	3

class Features
{
public:
	Features(void){}
	~Features(void){}

	void getHSVColorHistDes(const Mat &image, Mat &descriptors);	// HSV颜色直方图
	void getGLCMDes(const Mat &image, Mat &descriptors);			// 灰度共生矩阵

private:
	int convertH(double h);
	int convertS(double s);
	int convertV(double v);
	void getGLCM(const Mat &image, 
		double* normGlcm, int orientation, int step);
};

double bhattacharyyaDistance(const Mat &mat1, const Mat &mat2);		// 计算巴氏距离
double computeDistance(pair<Mat, Mat> des1, pair<Mat, Mat> des2);	// 根据描述子计算相似度

void getSurfPointsSet(const Mat &objColorImg, const Mat &objPointCloud, 
	const Mat &objPointIndex, const Mat &modColorImg, 
	const Mat &modPointCloud, const Mat &modPointIndex, 
	Mat &objSetOrigin, Mat &objSet, Mat &modSet, xn::DepthGenerator dg);

void getFeaturePoints(xn::DepthGenerator dg,				// OpenNI，用以将点转换成真实世界的点
	const Mat &colorImgNow, const Mat &depthImgNow,			// 当前帧的彩色图与深度图
	const Mat &colorImgPre, const Mat &depthImgPre,			// 前一帧的彩色图与深度图
	Mat &objSet, Mat &modSet, Mat &objSetAT, Mat &mask);	// 依次为当前帧的特征点集，前一帧的特征点集，
															// 匹配转换后当前帧的特征点集，当前帧相比前一帧多出的部分

#endif //FEATURES_H