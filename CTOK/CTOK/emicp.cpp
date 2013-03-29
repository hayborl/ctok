#include "emicp.h"

EMICP::EMICP(const Mat &objSet, const Mat &modSet, 
	float sigma_p2, float sigma_inf, 
	float sigma_factor, float d_02)
{
	assert(objSet.cols == 1 && modSet.cols == 1);
	assert(!objSet.empty() && !modSet.empty());

	m_objSet = objSet.clone() / 1000.0f;
	m_modSet = modSet.clone() / 1000.0f;
	m_tr = Mat::eye(4, 4, CV_32FC1);
	m_sigma_p2 = sigma_p2;
	m_sigma_inf = sigma_inf;
	m_sigma_factor = sigma_factor;
	m_d_02 = d_02;
}

void EMICP::run(bool withCuda, Mat* initObjSet)
{
	Mat objSet = initObjSet->clone() / 1000.0f;
	Mat modSet = convertMat(m_modSet);
	int rowsA = objSet.rows;
	int colsA = modSet.rows;
	Transformation tr;

	int iter = 0;

	Mat A(rowsA, colsA, CV_32FC1);

	Mat preTransformMat = Mat::zeros(4, 4, CV_32FC1);
	Mat nowTransformMat = Mat::zeros(4, 4, CV_32FC1);

	do
	{
		preTransformMat = nowTransformMat;

		RUNANDTIME(global_timer, updateA(A, objSet, m_modSet, withCuda),
			OUTPUT && SUBOUTPUT, "update A Matrix");

		Mat C = Mat::ones(rowsA, 1, CV_32FC1);
		Mat ones = Mat::ones(colsA, 1, CV_32FC1);
		float alpha = expf(-m_d_02 / m_sigma_p2);
		C = C * alpha + A * ones;
		RUNANDTIME(global_timer, normalizeRows(A, C, withCuda, true), 
			OUTPUT && SUBOUTPUT, "normalize rows of A with C");
// 		Mat tmpM = A.clone();
// 		sqrt(tmpM, A);

		Mat lambda = A * ones;
		Mat modSetPrime = A * modSet;
		RUNANDTIME(global_timer, 
			normalizeRows(modSetPrime, lambda),
			OUTPUT && SUBOUTPUT, "normalize rows of mod with lambda");

		Mat tmpObjSet = convertMat(m_objSet);
		RUNANDTIME(global_timer, tr = 
			computeTransformation(tmpObjSet, modSetPrime, lambda),
			OUTPUT && SUBOUTPUT, "compute transformation");
		nowTransformMat = getTransformMat(tr);
		RUNANDTIME(global_timer, transformPointCloud(
			m_objSet, &objSet, nowTransformMat, withCuda), 
			OUTPUT && SUBOUTPUT, "transform points.");

		m_sigma_p2 *= m_sigma_factor;
	}while (m_sigma_p2 > m_sigma_inf && 
		abs(sum(nowTransformMat - preTransformMat)[0]) > 1e-5);
	tr.t *= 1000.0f;
	m_tr = getTransformMat(tr);
}
