#ifndef BUNDLEADJUSTMENT_H
#define BUNDLEADJUSTMENT_H

#include "opencv2/opencv.hpp"

#include "Math/v3d_linear_utils.h"

using namespace V3D;
using namespace cv;

class BundleAdjustment
{
public:
	BundleAdjustment(){};

	static void setIntrinsic(const Mat &intrinsicMat);
	void runBundleAdjustment(Mat &cam, Mat &points, 
		const vector<Vec2d> &oldLoc, const vector<Vec2d> &newLoc);

private:
	enum {CAM_NUM = 2};

	static Matrix3x3d m_intrinsic;	// ÄÚÓ¦¾ØÕó
	static double m_f0;				// ½¹¾à
};

#endif