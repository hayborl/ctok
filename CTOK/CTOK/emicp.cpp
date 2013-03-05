#include "emicp.h"

EMICP::EMICP(const Mat &objSet, const Mat &modSet, 
	float sigma_p2, float sigma_inf, 
	float sigma_factor, float d_02)
{
	assert(objSet.cols == 1 && modSet.cols == 1);
	assert(!objSet.empty() && !modSet.empty());

	m_objSet = objSet.clone() / 1000.0f;
	m_modSet = modSet.clone() / 1000.0f;
	m_tr.q = Vec4f(1, 0, 0, 0);
	m_tr.t = Vec3f(0, 0, 0);
	m_sigma_p2 = sigma_p2;
	m_sigma_inf = sigma_inf;
	m_sigma_factor = sigma_factor;
	m_d_02 = d_02;
}

void EMICP::run(Mat* initObjSet)
{
	int rowsA = m_objSet.rows;
	int colsA = m_modSet.rows;
	Mat objSet = m_objSet.clone();
	Mat modSet = convertMat(m_modSet);

	int iter = 0;

	while (m_sigma_p2 > m_sigma_inf)
	{
		float tempR[9], tempT[3];
		getRotateMatrix(m_tr.q, tempR);
		Mat R(3, 3, CV_32FC1, tempR);
		memcpy(tempT, m_tr.t.val, 3 * sizeof(float));
		Mat T(3, 1, CV_32FC1, tempT);

		Mat A(rowsA, colsA, CV_32FC1);
		RUNANDTIME(global_timer, updateA(A, objSet, R, T),
			OUTPUT && SUBOUTPUT, "update A Matrix");

		Mat C = Mat::ones(rowsA, 1, CV_32FC1);
		Mat ones = Mat::ones(colsA, 1, CV_32FC1);
		float alpha = expf(-m_d_02 / m_sigma_p2);
		C = C * alpha + A * ones;
		RUNANDTIME(global_timer, normalizeRows2(A, C), 
			OUTPUT && SUBOUTPUT, "normalize rows of A with C");
		Mat tmpM = A.clone();
		sqrt(tmpM, A);

		Mat lambda = A * ones;
		Mat modSetPrime = A * modSet;
		RUNANDTIME(global_timer, normalizeRows(modSetPrime, lambda),
			OUTPUT && SUBOUTPUT, "normalize rows of mod with lambda");

		Mat tmpObjSet = convertMat(objSet);
		RUNANDTIME(global_timer, m_tr = 
			computeTransformation(tmpObjSet, modSetPrime, lambda),
			OUTPUT && SUBOUTPUT, "compute transformation");
		Mat transformMat = getTransformMat();
		RUNANDTIME(global_timer, transformPointCloud(
			m_objSet, &objSet, transformMat), 
			OUTPUT && SUBOUTPUT, "transform points.");

		m_sigma_p2 *= m_sigma_factor;
	}
	m_tr.t *= 1000.0f;
}

void EMICP::cuda_run(Mat* initObjSet)
{
	int rowsA = m_objSet.rows;
	int colsA = m_modSet.rows;
	Mat objSet = m_objSet.clone();
	Mat modSet = convertMat(m_modSet);

	while (m_sigma_p2 > m_sigma_inf)
	{
		Mat tmpObjSet = convertMat(objSet);

		float tempR[9], tempT[3];
		getRotateMatrix(m_tr.q, tempR);
		Mat R(3, 3, CV_32FC1, tempR);
		memcpy(tempT, m_tr.t.val, 3 * sizeof(float));
		Mat T(3, 1, CV_32FC1, tempT);

		Mat A(rowsA, colsA, CV_32FC1);
		RUNANDTIME(global_timer, cuda_updateA(A, tmpObjSet, 
			modSet, R, T), OUTPUT && SUBOUTPUT, 
			"update A Matrix");

		Mat C = Mat::ones(rowsA, 1, CV_32FC1);
		Mat ones = Mat::ones(colsA, 1, CV_32FC1);
		float alpha = expf(-m_d_02 / m_sigma_p2);
		C = C * alpha + A * ones;
		RUNANDTIME(global_timer, normalizeRows(A, C), 
			OUTPUT && SUBOUTPUT, "normalize rows of A with C");
		Mat tmpM = A.clone();
		sqrt(tmpM, A);

		Mat lambda = A * ones;
		Mat modSetPrime = A * modSet;
		RUNANDTIME(global_timer, normalizeRows(modSetPrime, lambda),
			OUTPUT && SUBOUTPUT, "normalize rows of mod with lambda");

		RUNANDTIME(global_timer, m_tr = 
			computeTransformation(tmpObjSet, modSetPrime, lambda),
			OUTPUT && SUBOUTPUT, "compute transformation");
		Mat transformMat = getTransformMat();
		RUNANDTIME(global_timer, transformPointCloud(
			m_objSet, &objSet, transformMat), 
			OUTPUT && SUBOUTPUT, "transform points.");

		m_sigma_p2 *= m_sigma_factor;
		waitKey();
	}
	m_tr.t *= 1000.0f;
}

// void EMICP::updateA( Mat &A, const Mat &objSet, 
// 	const Mat &R, const Mat &T, bool withCuda )
// {
// #pragma omp parallel for
// 	for (int c = 0; c < A.cols; c++)
// 	{
// 		Mat mp = Mat(m_modSet.at<Point3f>(c, 0));
// 		for (int r = 0; r < A.rows; r++)
// 		{
// 			Mat op = Mat(objSet.at<Point3f>(r, 0));
// 			Mat tmp = mp - (R * op + T);
// 			double d = tmp.dot(tmp) / m_sigma_p2;
// 			d = exp(-d);
// 			A.at<float>(r, c) = (float)d;
// 		}
// 	}
// }

// void EMICP::normalizeRows( Mat &A, const Mat &C, bool withCuda )
// {
// 	int rowsA = A.rows;
// 	int colsA = A.cols;
// #pragma omp parallel for
// 	for (int r = 0; r < rowsA; r++)
// 	{
// 		Mat row = A(Rect(0, r, colsA, 1));
// 		float c = C.at<float>(r, 0);
// 		if (c > 10e-7f)
// 		{
// 			row = row / c;
// 		}
// 		else
// 		{
// 			float tmp = 1.0f / colsA;
// 			Mat(rowsA, colsA, CV_32FC1, Scalar::all(tmp)).copyTo(row);
// 		}
// 	}
// }

void EMICP::normalizeRows2( Mat &A, const Mat &C )
{
	int rowsA = A.rows;
	int colsA = A.cols;
#pragma omp parallel for
	for (int r = 0; r < rowsA; r++)
	{
		Mat row = A(Rect(0, r, colsA, 1));
		float c = C.at<float>(r, 0);
		if (c > 10e-7f)
		{
			row = row / c;
		}
		else
		{
			float tmp = 1.0f / colsA;
			Mat(rowsA, colsA, CV_32FC1, Scalar::all(tmp)).copyTo(row);
		}
	}
}
