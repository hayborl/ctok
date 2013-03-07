#ifndef FEATURES_H
#define FEATURES_H

#include "common.h"

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

	void getHSVColorHistDes(const Mat& image, Mat& descriptors);	// HSV颜色直方图
	void getGLCMDes(const Mat& image, Mat& descriptors);			// 灰度共生矩阵

private:
	int convertH(double h);
	int convertS(double s);
	int convertV(double v);
	void getGLCM(const Mat& image, 
		double* normGlcm, int orientation, int step);
};


#endif //FEATURES_H