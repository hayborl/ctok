#include "bundleadjustment.h"

#include "Math/v3d_linear.h"
#include "Geometry/v3d_metricbundle.h"
#include "Math/v3d_optimization.h"

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

void BundleAdjustment::runBundleAdjustment( Mat &oldCam, Mat &newCam, 
	Mat &points, const vector<Vec2d> &oldLoc, const vector<Vec2d> &newLoc )
{
	assert(points.rows == oldLoc.size());
	assert(oldLoc.size() == newLoc.size());

	int N = CAM_NUM;
	int M = points.rows;
	int K = 2 * M;

	double r_f0 = 1.0 / m_f0;

	StdDistortionFunction distortion;
// 	distortion.k1 = -0.1296;
// 	distortion.k2 = 0.45;
// 	distortion.p1 = -0.0005;
// 	distortion.p2 = -0.002;

	Matrix3x3d KNorm = m_intrinsic;
	// Normalize the intrinsic to have unit focal length.
	scaleMatrixIP(r_f0, KNorm);
	KNorm[2][2] = 1.0;

	vector<Vector3d> pts(M);
#pragma omp parallel for
	for (int i = 0; i < M; i++)
	{
		Vec3d v = points.at<Vec3d>(i, 0);
		pts[i][0] = v[0];
		pts[i][1] = v[1];
		pts[i][2] = v[2];
	}
/*	cout << "Read the 3D points." << endl;*/

	vector<CameraMatrix> cams(N);

	Matrix3x3d R;
	Vector3d T;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			R[i][j] = oldCam.at<double>(i, j);
		}
		T[i] = oldCam.at<double>(i, 3);
	}
	cams[0].setIntrinsic(KNorm);
	cams[0].setRotation(R);
	cams[0].setTranslation(T);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			R[i][j] = newCam.at<double>(i, j);
		}
		T[i] = oldCam.at<double>(i, 3);
	}
	cams[1].setIntrinsic(KNorm);
	cams[1].setRotation(R);
	cams[1].setTranslation(T);
/*	cout << "Read the cameras." << endl;*/

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
/*	cout << "Read " << K << " valid 2D measurements." << endl;*/

	const double inlierThreshold = 2.0 / m_f0;

	Matrix3x3d K0 = cams[0].getIntrinsic();

// 	showErrorStatistics(m_f0, distortion, cams, pts, measurements, 
// 		correspondingView, correspondingPoint);

// 	V3D::optimizerVerbosenessLevel = 1;
	CommonInternalsMetricBundleOptimizer opt(FULL_BUNDLE_FOCAL_LENGTH_PP,
		inlierThreshold, K0, distortion, cams, pts, 
		measurements, correspondingView, correspondingPoint);
	opt.maxIterations = 50;
	opt.minimize();
// 	cout << "optimizer status = " << opt.status << endl;
// 
// 	showErrorStatistics(m_f0, distortion, cams, pts, measurements, 
// 		correspondingView, correspondingPoint);

#pragma omp parallel for
	for (int i = 0; i < M; i++)
	{
		Vec3d v;
		v[0] = pts[i][0];
		v[1] = pts[i][1];
		v[2] = pts[i][2];
		points.at<Vec3d>(i, 0) = v;
	}
	Matrix3x4d RT = cams[0].getOrientation();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			oldCam.at<double>(i, j) = RT[i][j];
		}
	}
	RT = cams[1].getOrientation();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			newCam.at<double>(i, j) = RT[i][j];
		}
	}
}

void BundleAdjustment::runBundleAdjustment( vector<Mat> &camPoses, 
	const Mat &points, const vector<vector<KeyPoint>> &keypoints, 
	const vector<vector<pair<int, int>>> &matches )
{
	assert(keypoints.size() == camPoses.size());
	assert(matches.size() == keypoints.size() - 1);
	int N = (int)camPoses.size();
	assert(keypoints[N - 1].size() == points.rows);
	int M = points.rows;
	int K = 0;

	double r_f0 = 1.0 / m_f0;

	StdDistortionFunction distortion;

	Matrix3x3d KNorm = m_intrinsic;
	// Normalize the intrinsic to have unit focal length.
	scaleMatrixIP(r_f0, KNorm);
	KNorm[2][2] = 1.0;

	vector<Vector3d> pts(M);
	vector<int> ptMaps(M);
	int cnt = 0;
//#pragma omp parallel for
	for (int i = 0; i < M; i++)
	{
		Vec3d v = points.at<Vec3d>(i, 0);
		pts[i][0] = v[0];
		pts[i][1] = v[1];
		pts[i][2] = v[2];
		if (v[2] != 0)
		{
			ptMaps[i] = i;
		}
		else
		{
			ptMaps[i] = -1;
			cnt++;
		}
	}
/*	cout << "Read the 3D points." << endl;*/

	vector<CameraMatrix> cams(N);

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		Matrix3x3d R;
		Vector3d T;

		cams[i].setIntrinsic(KNorm);
		Mat pose = camPoses[i];

		for (int m = 0; m < 3; m++)
		{
			for (int n = 0; n < 3; n++)
			{
				R[m][n] = pose.at<double>(m, n);
			}
			T[m] = pose.at<double>(m, 3);
		}
		cams[i].setRotation(R);
		cams[i].setTranslation(T);
	}
	/*	cout << "Read the cameras." << endl;*/

	vector<Vector2d> measurements;
	vector<int> correspondingView;
	vector<int> correspondingPoint;

	K = 0;
	for (int i = 0; i < N - 1; i++)
	{
		for (int j = 0; j < matches[i].size(); j++)
		{
			if (ptMaps[matches[i][j].first] == -1)
			{
				continue;
			}
			Point2f pt = keypoints[i][matches[i][j].second].pt;
			Vector3d p = makeVector3((double)pt.x, (double)pt.y, 1.0); 
			scaleVectorIP(r_f0, p);
			measurements.push_back(makeVector2(p[0], p[1]));
			correspondingView.push_back(i);
			correspondingPoint.push_back(matches[i][j].first);
			K++;
		}
	}
	for (int i = 0; i < keypoints[N - 1].size(); i++)
	{
		if (ptMaps[i] == -1)
		{
			continue;
		}
		Point2f pt = keypoints[N - 1][i].pt;
		Vector3d p = makeVector3((double)pt.x, (double)pt.y, 1.0); 
		scaleVectorIP(r_f0, p);
		measurements.push_back(makeVector2(p[0], p[1]));
		correspondingView.push_back(N - 1);
		correspondingPoint.push_back(i);
		K++;
	}
/*	cout << "Read " << K << " valid 2D measurements." << endl;*/

	const double inlierThreshold = 2.0 / m_f0;

// 	showErrorStatistics(m_f0, distortion, cams, pts, measurements, 
// 		correspondingView, correspondingPoint);
// 	V3D::optimizerVerbosenessLevel = 1;

	Matrix3x3d K0 = cams[0].getIntrinsic();
	CommonInternalsMetricBundleOptimizer opt(FULL_BUNDLE_FOCAL_LENGTH_PP,
		inlierThreshold, K0, distortion, cams, pts, 
		measurements, correspondingView, correspondingPoint);
	opt.maxIterations = 50;
	opt.minimize();
// 	cout << "optimizer status = " << opt.status << endl;
// 
// 	showErrorStatistics(m_f0, distortion, cams, pts, measurements, 
// 		correspondingView, correspondingPoint);

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		Matrix3x4d RT = cams[i].getOrientation();
		for (int j = 0; j < 16; j++)
		{
			int subi = j / 4;
			int subj = j % 4;
			if (subi < 3)
			{
				camPoses[i].at<double>(subi, subj) = RT[subi][subj];
			}
			else
			{
				camPoses[i].at<double>(subi, subj) = 0;
			}
		}
	}
}

void BundleAdjustment::showErrorStatistics( double const f0, 
	StdDistortionFunction const& distortion,
	vector<CameraMatrix> const& cams, 
	vector<Vector3d> const& Xs, 
	vector<Vector2d> const& measurements, 
	vector<int> const& correspondingView, 
	vector<int> const& correspondingPoint )
{
	
	int const K = (int)measurements.size();

	double meanReprojectionError = 0.0;
	for (int k = 0; k < K; ++k)
	{
		int const i = correspondingView[k];
		int const j = correspondingPoint[k];
		Vector2d p = cams[i].projectPoint(distortion, Xs[j]);

		double reprojectionError = norm_L2(f0 * (p - measurements[k]));
		meanReprojectionError += reprojectionError;
	}
	cout << "mean reprojection error (in pixels): " << meanReprojectionError/K << endl;
}
