#ifndef FEATURES_H
#define FEATURES_H

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/opencv.hpp"
#include "XnCppWrapper.h"
using namespace cv;
using namespace std;

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
	void getAvgHash(const Mat &image, Mat &descriptors);			// 平均hash算法
	void getPHash(const Mat &image, Mat &descriptors);				// pHash算法

private:
	int convertH(double h);
	int convertS(double s);
	int convertV(double v);
	void getGLCM(const Mat &image, 
		double* normGlcm, int orientation, int step);
};

int hammingDistance(const Mat &mat1, const Mat &mat2);				// 计算汉明距离

double bhattacharyyaDistance(const Mat &mat1, const Mat &mat2);		// 计算巴氏距离
double computeDistance(pair<Mat, Mat> des1, pair<Mat, Mat> des2);	// 根据描述子计算相似度

void getSurfPointsSet(const Mat &objColorImg, const Mat &objPointCloud, 
	const Mat &objPointIndex, const Mat &modColorImg, 
	const Mat &modPointCloud, const Mat &modPointIndex, 
	Mat &objSetOrigin, Mat &objSet, Mat &modSet, xn::DepthGenerator dg);

void get2DFeaturePoints(const Mat &colorImg,					// 彩色图
	vector<KeyPoint> &keyPoints, Mat &descriptor);				// 提取出的特征点及其描述子

double pairwiseMatch(const vector<KeyPoint> &queryKeypoints,	// 查找的特征点集
	const vector<KeyPoint> &trainKeypoints,						// 训练的特征点集
	const Mat &queryDescriptors, const Mat &trainDescriptors,	// 两个描述子集
	Mat &H,	vector<pair<int, int>> &matchesPoints);				// Homography矩阵,相匹配的点集
																// 返回两张图匹配的score

bool convert2DTo3D(xn::DepthGenerator dg, const Mat &H,			// OpenNI，用以将点转换成真实世界的点，Homography矩阵
	const Mat &depthImgNow, const Mat &depthImgPre,				// 当前帧、前一帧的深度图
	const vector<KeyPoint> &keypointsNow,						// 当前帧的特征点集
	const vector<KeyPoint> &keypointsPre,						// 前一帧的特征点集
	const vector<pair<int, int>> &matchesPoints,				// 相匹配的点集
	vector<Vec2d> &oldLoc, vector<Vec2d> &newLoc,				// 在前一帧、当前帧图像上对应的位置
	Mat &objSet, Mat &modSet, Mat &objSetAT, Mat &mask);		// 依次为当前帧的特征点集，前一帧的特征点集，
																// 匹配转换后当前帧的特征点集，当前帧相比前一帧多出的部分

void convert2DTo3D(xn::DepthGenerator dg, const Mat &depthImg, 
	const vector<KeyPoint> &keypoints, Mat &points);

void fullMatch(const int &index, 
	const vector<Mat> &descriptors,
	const vector<vector<KeyPoint>> &keypoints,
	vector<vector<pair<int, int>>> &matches);

#endif //FEATURES_H