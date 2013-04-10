#include "abstracticp.h"

cv::Mat AbstractICP::getTransformMat(const Transformation &tr)
{
	double tempR[9], tempT[3];
	getRotateMatrix(tr.q, tempR);
	Mat R(3, 3, CV_64FC1, tempR);
	memcpy(tempT, tr.t.val, 3 * sizeof(double));
	Mat T(3, 1, CV_64FC1, tempT);

	Mat mat = Mat::eye(4, 4, CV_64FC1);
	Mat roi = mat(Rect(0, 0, 3, 3));
	R.copyTo(roi);
	roi = mat(Rect(3, 0, 1, 3));
	T.copyTo(roi);

	return mat.clone();
}

void AbstractICP::initTransform( Mat &initObj, bool withCuda )
{
	assert(m_objSet.cols == 1 && m_modSet.cols == 1);
	assert(m_objSet.type() == DataType<Point3d>::type);
	assert(m_modSet.type() == DataType<Point3d>::type);

	int objRows = m_objSet.rows;
	int modRows = m_modSet.rows;

	Mat objSet, modSet;

	Mat objCMat, modCMat;

#pragma omp parallel sections
	{
#pragma omp section
		{
			objSet = convertMat(m_objSet);
			getConstructionMat(objSet, objCMat);
		}
#pragma omp section
		{
			modSet = convertMat(m_modSet);
			getConstructionMat(modSet, modCMat);
		}
	}
	Mat ones = Mat::ones(objCMat.rows, 1, CV_64FC1);
	Transformation tr = computeTransformation(objCMat, modCMat, ones);
	Mat transformMat = getTransformMat(tr);
	transformPointCloud(m_objSet, initObj, transformMat, withCuda);
}

Transformation AbstractICP::computeTransformation( const Mat &objSet, 
	const Mat &modSet, const Mat &lambda )
{
	assert(objSet.cols == 3 && modSet.cols == 3);
	assert(objSet.rows == modSet.rows && objSet.rows == lambda.rows);
	int rows = objSet.rows;

	Mat meanMatMod = modSet.t() * lambda;
	Mat meanMatObj = objSet.t() * lambda;
	double sumLambda = sum(lambda)[0];
	meanMatMod /= sumLambda;
	meanMatObj /= sumLambda;

	//compute cross-covariance matrix
	Mat ccMatrix = Mat::zeros(3, 3, CV_64FC1);
	Mat tmpObjSet = objSet.clone();
#pragma omp parallel for
	for (int c = 0; c < 3; c++)
	{
		Mat col = tmpObjSet(Rect(c, 0, 1, rows));
		Mat(col.mul(lambda)).copyTo(col);
	}
	tmpObjSet /= sumLambda;
	ccMatrix = tmpObjSet.t() * modSet - meanMatObj * meanMatMod.t();

	//compute the trace of cross-covariance matrix
	double tr = trace(ccMatrix)[0];

	//compute the cyclic components of Q
	Mat ccMatrixT = ccMatrix.t();
	Mat A = ccMatrix - ccMatrixT;
	Mat delta(3, 1, CV_64FC1);
	delta.at<double>(0, 0) = A.at<double>(1, 2);
	delta.at<double>(1, 0) = A.at<double>(2, 0);
	delta.at<double>(2, 0) = A.at<double>(0, 1);
	Mat tempMat = ccMatrix + ccMatrixT - tr * Mat::eye(3, 3, CV_64FC1);

	Mat Q(4, 4, CV_64FC1);
	Q.at<double>(0, 0) = tr;
	Mat roi = Q(Rect(1, 0, 3, 1));
	((Mat)delta.t()).copyTo(roi);
	roi = Q(Rect(0, 1, 1, 3));
	delta.copyTo(roi);
	roi = Q(Rect(1, 1, 3, 3));
	tempMat.copyTo(roi);

	//compute the eigenvalues and eigenvector
	Mat eigenValues(4, 1, CV_64FC1);
	Mat eigenVector(4, 4, CV_64FC1);
	eigen(Q, eigenValues, eigenVector);
	Transformation RT;
	RT.q = Vec4f(eigenVector.row(0));

	//get the rotation matrix
	double tempR[9];
	getRotateMatrix(RT.q, tempR);
	Mat R(3, 3, CV_64FC1, tempR);

	Mat T = meanMatMod - R * meanMatObj;
	RT.t = Vec3d(T);

	return RT;
}

Transformation AbstractICP::cuda_computeTransformation( 
	const Mat &objSet, const Mat &modSet, const Mat &lambda )
{
	assert(objSet.cols == 3 && modSet.cols == 3);
	assert(objSet.rows == modSet.rows && objSet.rows == lambda.rows);
	int rows = objSet.rows;

	Mat ones = Mat::ones(3, 1, CV_64FC1);
	Mat ones33 = Mat::ones(3, 3, CV_64FC1);
	GpuMat gpuOnes, gpuOnes33;
	gpuOnes.upload(ones);
	gpuOnes33.upload(ones33);

	GpuMat gpuObj, gpuMod, gpuLambda;
	gpuObj.upload(objSet);
	gpuMod.upload(modSet);
	gpuLambda.upload(lambda);

	double sumLambda = sum(gpuLambda)[0];
	GpuMat gpuModMatC, gpuObjMatC;

	gemm(gpuObj, gpuLambda, 1.0f / sumLambda, 
		gpuOnes, 0, gpuObjMatC, GEMM_1_T);
	gemm(gpuMod, gpuLambda, 1.0f / sumLambda, 
		gpuOnes, 0, gpuModMatC, GEMM_1_T);

	//compute cross-covariance matrix
	GpuMat gpuCcMatrix(3, 3, CV_64FC1);
	GpuMat tmpObjSet = gpuObj.clone();
	for (int c = 0; c < 3; c++)
	{
		GpuMat col = tmpObjSet(Rect(c, 0, 1, rows));
		multiply(col, gpuLambda, col);
	}
	GpuMat tmpMat1, tmpMat2;
	gemm(tmpObjSet, gpuMod, 1.0f, gpuOnes33, 0, tmpMat1, GEMM_1_T);
	gemm(gpuObjMatC, gpuModMatC, 1.0f, gpuOnes33, 0, tmpMat2, GEMM_2_T);
	subtract(tmpMat1, tmpMat2, gpuCcMatrix);

	Mat ccMatrix(3, 3, CV_64FC1);
	gpuCcMatrix.download(ccMatrix);

	//compute the trace of cross-covariance matrix
	double tr = trace(ccMatrix)[0];

	//compute the cyclic components of Q
	Mat ccMatrixT = ccMatrix.t();
	Mat A = ccMatrix - ccMatrixT;
	Mat delta(3, 1, CV_64FC1);
	delta.at<double>(0, 0) = A.at<double>(1, 2);
	delta.at<double>(1, 0) = A.at<double>(2, 0);
	delta.at<double>(2, 0) = A.at<double>(0, 1);
	Mat tempMat = ccMatrix + ccMatrixT - tr * Mat::eye(3, 3, CV_64FC1);

	Mat Q(4, 4, CV_64FC1);
	Q.at<double>(0, 0) = tr;
	Mat roi = Q(Rect(1, 0, 3, 1));
	((Mat)delta.t()).copyTo(roi);
	roi = Q(Rect(0, 1, 1, 3));
	delta.copyTo(roi);
	roi = Q(Rect(1, 1, 3, 3));
	tempMat.copyTo(roi);

	//compute the eigenvalues and eigenvector
	Mat eigenValues(4, 1, CV_64FC1);
	Mat eigenVector(4, 4, CV_64FC1);
	eigen(Q, eigenValues, eigenVector);

	int maxEigenIndex = 0;
	minMaxIdx(eigenValues, NULL, NULL, NULL, &maxEigenIndex);
	Transformation RT;
	RT.q = Vec4f(eigenVector.row(maxEigenIndex));

	//get the rotation matrix
	double tempR[9];
	getRotateMatrix(RT.q, tempR);
	Mat R(3, 3, CV_64FC1, tempR);

	Mat objMatC, modMatC;
	gpuObjMatC.download(objMatC);
	gpuModMatC.download(modMatC);
	Mat T = modMatC - R * objMatC;
	RT.t = Vec3d(T);

	return RT;
}

void AbstractICP::getConstructionMat( const Mat &in, Mat &out )
{
	assert(in.channels() == 1);
	assert(in.type() == CV_64FC1);

	out.create(4, 3, CV_64FC1);
	int rows = in.rows;
	Mat ones = Mat::ones(rows, 1, CV_64FC1);

	Mat mean = in.t() * ones;
	mean /= (double)rows;

	Mat ccMatrix = in.t() * in / ((double)rows) - mean * mean.t();
	Mat eigenValues(ccMatrix.rows, 1, CV_64FC1);
	Mat eigenVector(ccMatrix.rows, ccMatrix.cols, CV_64FC1);
	eigen(ccMatrix, eigenValues, eigenVector);

	Mat roi = out(Rect(0, 0, 3, 3));
	eigenVector.copyTo(roi);
	roi = out(Rect(0, 3, 3, 1));
	Mat(mean.t()).copyTo(roi);
}

void getRotateMatrix( Vec4d q, double* R )
{
	double q0 = q[0];
	double q1 = q[1];
	double q2 = q[2];
	double q3 = q[3];

	R[0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
	R[1] = 2 * (q1 * q2 - q0 * q3);
	R[2] = 2 * (q1 * q3 + q0 * q2);
	R[3] = 2 * (q1 * q2 + q0 * q3);
	R[4] = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3;
	R[5] = 2 * (q2 * q3 - q0 * q1);
	R[6] = 2 * (q1 * q3 - q0 * q2);
	R[7] = 2 * (q2 * q3 + q0 * q1);
	R[8] = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2;
}
