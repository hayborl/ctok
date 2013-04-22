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

	void getHSVColorHistDes(const Mat &image, Mat &descriptors);	// HSV��ɫֱ��ͼ
	void getGLCMDes(const Mat &image, Mat &descriptors);			// �Ҷȹ�������
	void getAvgHash(const Mat &image, Mat &descriptors);			// ƽ��hash�㷨
	void getPHash(const Mat &image, Mat &descriptors);				// pHash�㷨

private:
	int convertH(double h);
	int convertS(double s);
	int convertV(double v);
	void getGLCM(const Mat &image, 
		double* normGlcm, int orientation, int step);
};

int hammingDistance(const Mat &mat1, const Mat &mat2);				// ���㺺������

double bhattacharyyaDistance(const Mat &mat1, const Mat &mat2);		// ������Ͼ���
double computeDistance(pair<Mat, Mat> des1, pair<Mat, Mat> des2);	// ���������Ӽ������ƶ�

void getSurfPointsSet(const Mat &objColorImg, const Mat &objPointCloud, 
	const Mat &objPointIndex, const Mat &modColorImg, 
	const Mat &modPointCloud, const Mat &modPointIndex, 
	Mat &objSetOrigin, Mat &objSet, Mat &modSet, xn::DepthGenerator dg);

void get2DFeaturePoints(const Mat &colorImg,					// ��ɫͼ
	vector<KeyPoint> &keyPoints, Mat &descriptor);				// ��ȡ���������㼰��������

double pairwiseMatch(const vector<KeyPoint> &queryKeypoints,	// ���ҵ������㼯
	const vector<KeyPoint> &trainKeypoints,						// ѵ���������㼯
	const Mat &queryDescriptors, const Mat &trainDescriptors,	// ���������Ӽ�
	Mat &H,	vector<pair<int, int>> &matchesPoints);				// Homography����,��ƥ��ĵ㼯
																// ��������ͼƥ���score

bool convert2DTo3D(xn::DepthGenerator dg, const Mat &H,			// OpenNI�����Խ���ת������ʵ����ĵ㣬Homography����
	const Mat &depthImgNow, const Mat &depthImgPre,				// ��ǰ֡��ǰһ֡�����ͼ
	const vector<KeyPoint> &keypointsNow,						// ��ǰ֡�������㼯
	const vector<KeyPoint> &keypointsPre,						// ǰһ֡�������㼯
	const vector<pair<int, int>> &matchesPoints,				// ��ƥ��ĵ㼯
	vector<Vec2d> &oldLoc, vector<Vec2d> &newLoc,				// ��ǰһ֡����ǰ֡ͼ���϶�Ӧ��λ��
	Mat &objSet, Mat &modSet, Mat &objSetAT, Mat &mask);		// ����Ϊ��ǰ֡�������㼯��ǰһ֡�������㼯��
																// ƥ��ת����ǰ֡�������㼯����ǰ֡���ǰһ֡����Ĳ���

void convert2DTo3D(xn::DepthGenerator dg, const Mat &depthImg, 
	const vector<KeyPoint> &keypoints, Mat &points);

void fullMatch(const int &index, 
	const vector<Mat> &descriptors,
	const vector<vector<KeyPoint>> &keypoints,
	vector<vector<pair<int, int>>> &matches);

#endif //FEATURES_H