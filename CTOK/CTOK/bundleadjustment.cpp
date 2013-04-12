#include "bundleadjustment.h"

#include "Math/v3d_linear.h"
#include "Geometry/v3d_metricbundle.h"
#include "Geometry/v3d_cameramatrix.h"

#include <iostream>

using namespace std;

Matrix3x3d BundleAdjustment::m_intrinsic = Matrix3x3d();
double BundleAdjustment::m_f0 = 0.0;

void BundleAdjustment::setIntrinsic( const Mat &intrinsicMat )
{
	makeIdentityMatrix(m_intrinsic);
	m_intrinsic[0][0] = intrinsicMat.at<double>(0, 0);
	m_intrinsic[0][1] = intrinsicMat.at<double>(0, 1);
	m_intrinsic[0][2] = intrinsicMat.at<double>(0, 2);
	m_intrinsic[1][1] = intrinsicMat.at<double>(1, 1);
	m_intrinsic[1][2] = intrinsicMat.at<double>(1, 2);

	m_f0 = m_intrinsic[0][0];
}

void BundleAdjustment::runBundleAdjustment( Mat &cam, Mat &points, 
	const vector<Vec2d> &oldLoc, const vector<Vec2d> &newLoc )
{
	assert(points.rows == oldLoc.size());
	assert(oldLoc.size() == newLoc.size());

	int N = CAM_NUM;
	int M = points.rows;
	int K = 2 * M;

	double r_f0 = 1.0 / m_f0;

	Matrix3x3d KNorm = m_intrinsic;
	// Normalize the intrinsic to have unit focal length.
	scaleMatrixIP(r_f0, KNorm);
	KNorm[2][2] = 1.0;

	vector<Vector3d> pts(M);
	for (int i = 0; i < M; i++)
	{
		Vec3d v = points.at<Vec3d>(i, 0);
		pts[i][0] = v[0];
		pts[i][1] = v[1];
		pts[i][2] = v[2];
	}
	cout << "Read the 3D points." << endl;

	vector<int> camIdFwdMap(N);
	map<int, int> camIdBwdMap;

	vector<CameraMatrix> cams(N);

	Matrix3x3d R;
	makeIdentityMatrix(R);
	Vector3d T;
	cams[0].setIntrinsic(KNorm);
	cams[0].setRotation(R);
	cams[0].setTranslation(T);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			R[i][j] = cam.at<double>(i, j);
		}
		T[i] = cam.at<double>(i, 3);
	}
	cams[1].setIntrinsic(KNorm);
	cams[1].setRotation(R);
	cams[1].setTranslation(T);
	cout << "Read the cameras." << endl;

	vector<Vector2d> measurements(K);
	vector<int> correspondingView(K);
	vector<int> correspondingPoint(K);

	for (int i = 0; i < M; i++)
	{
		int idx = 2 * i;
		Vector3d p;

		Vec2d v = oldLoc[i];
		p[0] = v[0];
		p[1] = v[1];
		p[2] = 1.0;

		scaleVectorIP(r_f0, p);
		measurements[idx] = makeVector2(p[0], p[1]);
		correspondingView[idx] = 0;
		correspondingPoint[idx] = i;

		idx++;
		v = newLoc[i];
		p[0] = v[0];
		p[1] = v[1];
		p[2] = 1.0;

		scaleVectorIP(r_f0, p);
		measurements[idx] = makeVector2(p[0], p[1]);
		correspondingView[idx] = 1;
		correspondingPoint[idx] = i;
	}
	cout << "Read " << K << " valid 2D measurements." << endl;

	const double inlierThreshold = 2.0 / m_f0;

	StdMetricBundleOptimizer opt(inlierThreshold, cams, pts, 
		measurements, correspondingView, correspondingPoint);
	opt.maxIterations = 50;
	opt.minimize();

	for (int i = 0; i < M; i++)
	{
		Vec3d v = points.at<Vec3d>(i, 0);
		v[0] = pts[i][0];
		v[1] = pts[i][1];
		v[2] = pts[i][2];
	}
	Matrix3x4d RT = cams[1].getOrientation();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			cam.at<double>(i, j) = RT[i][j];
		}
	}
}
