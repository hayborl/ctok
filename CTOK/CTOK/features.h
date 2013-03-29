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

	void getHSVColorHistDes(const Mat &image, Mat &descriptors);	// HSV��ɫֱ��ͼ
	void getGLCMDes(const Mat &image, Mat &descriptors);			// �Ҷȹ�������

private:
	int convertH(double h);
	int convertS(double s);
	int convertV(double v);
	void getGLCM(const Mat &image, 
		double* normGlcm, int orientation, int step);
};

double bhattacharyyaDistance(const Mat &mat1, const Mat &mat2);		// ������Ͼ���
double computeDistance(pair<Mat, Mat> des1, pair<Mat, Mat> des2);	// ���������Ӽ������ƶ�

void getSurfPointsSet(const Mat &objColorImg, const Mat &objPointCloud, 
	const Mat &objPointIndex, const Mat &modColorImg, 
	const Mat &modPointCloud, const Mat &modPointIndex, 
	Mat &objSetOrigin, Mat &objSet, Mat &modSet, xn::DepthGenerator dg);

void getFeaturePoints(xn::DepthGenerator dg,				// OpenNI�����Խ���ת������ʵ����ĵ�
	const Mat &colorImgNow, const Mat &depthImgNow,			// ��ǰ֡�Ĳ�ɫͼ�����ͼ
	const Mat &colorImgPre, const Mat &depthImgPre,			// ǰһ֡�Ĳ�ɫͼ�����ͼ
	Mat &objSet, Mat &modSet, Mat &objSetAT, Mat &mask);	// ����Ϊ��ǰ֡�������㼯��ǰһ֡�������㼯��
															// ƥ��ת����ǰ֡�������㼯����ǰ֡���ǰһ֡����Ĳ���

#endif //FEATURES_H