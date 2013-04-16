#ifndef BUNDLEADJUSTMENT_H
#define BUNDLEADJUSTMENT_H

#include "opencv2/opencv.hpp"

#include "Math/v3d_linear_utils.h"
#include "Geometry/v3d_cameramatrix.h"

#pragma comment(lib, "libs/colamd.lib")
#pragma comment(lib, "libs/ldl.lib")
#pragma comment(lib, "libs/SSBA.lib")

using namespace V3D;
using namespace cv;

class BundleAdjustment
{
public:
	BundleAdjustment(){};

	static void setIntrinsic(const Mat &intrinsicMat);
	static void runBundleAdjustment(Mat &oldCam, Mat &newCam, Mat &points, 
		const vector<Vec2d> &oldLoc, const vector<Vec2d> &newLoc);
	static void runBundleAdjustment(vector<Mat> &camPoses, Mat points, 
		const vector<vector<KeyPoint>> &keypoints);

private:
	enum {CAM_NUM = 2};

	static Matrix3x3d m_intrinsic;	// ÄÚÓ¦¾ØÕó
	static double m_f0;				// ½¹¾à

	static void showErrorStatistics(double const f0,
		StdDistortionFunction const& distortion,
		vector<CameraMatrix> const& cams,
		vector<Vector3d> const& Xs,
		vector<Vector2d> const& measurements,
		vector<int> const& correspondingView,
		vector<int> const& correspondingPoint);
};

#endif