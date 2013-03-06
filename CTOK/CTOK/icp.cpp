#include "icp.h"

// Vec3f computeNormal( vector<pair<_Examplar, double>> points )
// {
// 	Mat M = Mat::zeros(3, 3, CV_64FC1);
// 	Mat mean = Mat::zeros(3, 1, CV_64FC1);
// 	for (int i = 0; i < points.size(); i++)
// 	{
// 		Mat pMat(points[i].first.data(), true);
// 		M += pMat * pMat.t();
// 		mean += pMat;
// 	}
// 	M /= (double)points.size();
// 	mean /= (double)points.size();
// 	M -= mean * mean.t();
// 
// 	Mat eigenValues(3, 1, CV_32FC1);
// 	Mat eigenVector(3, 3, CV_32FC1);
// 	eigen(M, eigenValues, eigenVector);
// 
// 	int minEigenIndex = 0;
// 	minMaxIdx(eigenValues, NULL, NULL, &minEigenIndex, NULL);
// 
// 	return normalize(Vec3f(eigenVector.row(minEigenIndex)));
// }

ICP::ICP( const Mat& objSet, const Mat& modSet, int iterMax, double epsilon )
{
	assert(objSet.cols == 1 && modSet.cols == 1);
	assert(!objSet.empty() && !modSet.empty());

	m_objSet = objSet.clone();
	m_modSet = modSet.clone();
	m_iterMax = iterMax;
	m_epsilon = epsilon;
	m_tr = Mat::eye(4, 4, CV_32FC1);

	RUNANDTIME(global_timer, createKDTree(),
		OUTPUT && SUBOUTPUT, "create kdTree");
}

void ICP::run(bool withCuda, Mat* initObjSet)
{
	assert(!m_objSet.empty() && !m_modSet.empty());

	double d_pre = 100000, d_now = 100000;
	int iterCnt = 0;

	Mat objSet = initObjSet->clone();
	Transformation tr;

/*	plotTwoPoint3DSet(objSet, m_modSet);*/

	do 
	{
		d_pre = d_now;

		Mat closestSet;
		Mat lambda(objSet.rows, 1, CV_32FC1);
		RUNANDTIME(global_timer, closestSet = 
			getClosestPointsSet(objSet, d_now, lambda, KDTREE).clone(), 
			OUTPUT && SUBOUTPUT, "compute closest points.");
		Mat tmpObjSet = convertMat(m_objSet);
		Mat tmpModSet = convertMat(closestSet);
		RUNANDTIME(global_timer, tr = 
			computeTransformation(tmpObjSet, tmpModSet, lambda), 
			OUTPUT && SUBOUTPUT, "compute transformation");
		Mat transformMat = getTransformMat(tr);
		RUNANDTIME(global_timer, transformPointCloud(
			m_objSet, &objSet, transformMat, withCuda), 
			OUTPUT && SUBOUTPUT, "transform points.");

		iterCnt++;
	} while (fabs(d_now - d_pre) > m_epsilon && iterCnt <= m_iterMax);

	m_tr = getTransformMat(tr);
/*	waitKey();*/

/*	plotTwoPoint3DSet(objSet, m_modSet);*/
}

Mat ICP::getClosestPointsSet( const Mat& objSet, double& d,
	Mat& lambda, Method method )
{
	int rows = objSet.rows;
	Mat closestSet(rows, 1, DataType<Point3f>::type);
	vector<double> dists(rows);
	double threshold = 0;
	int cnt = 0;

	switch (method)
	{
	case BASIC:
		for (int i = 0; i < rows; i++)
		{
			int minIndex = 0;
			float minDistance = DISTANCE_MAX;
			Point3f oPoint = objSet.at<Point3f>(i, 0);
			for (int j = 0; j < m_modSet.rows; j++)
			{
				Point3f p = m_modSet.at<Point3f>(j, 0) - oPoint;
				float distance = sqrt(p.dot(p));
				if (distance < minDistance)
				{
					minDistance = distance;
					minIndex = j;
				}
			}
			closestSet.at<Point3f>(i, 0) = m_modSet.at<Point3f>(minIndex, 0);
			dists[i] = minDistance;
			threshold += minDistance;
			cnt++;
		}
		break;
	case KDTREE:
	default:
		for (int i = 0; i < rows; i++)
		{
			Vec3f oPoint = objSet.at<Vec3f>(i, 0);
			ANNpoint qp = annAllocPt(ICP_DIMS);
			qp[0] = oPoint[0]; 
			qp[1] = oPoint[1];
			qp[2] = oPoint[2];
			ANNidx idx[1];
			ANNdist dist[1];
			m_kdTree->annkSearch(qp, 1, idx, dist);
			closestSet.at<Point3f>(i, 0) = Point3f(
				(float)m_modPts[idx[0]][0], (float)m_modPts[idx[0]][1], 
				(float)m_modPts[idx[0]][2]);
			dists[i] = dist[0];
			threshold += dist[0];
			cnt++;
		}
		break;
	case POINT_TO_PLANE:
		{
// 			int cnt = 0;
// 			for (int i = 0; i < rows; i++)
// 			{
// 				Vec3f oPoint = Vec3f(objSet.at<Point3f>(i, 0));
// 				_Examplar exm(ICP_DIMS);
// 				for (int j = 0; j < ICP_DIMS; j++)
// 				{
// 					exm[j] = oPoint[j];
// 				}
// 				vector<pair<_Examplar, double>> results;
// 				if (m_kdTree.findNearest(exm, DISTANCE_RANGE, results) < 3)
// 				{
// 					closestSet.at<Point3f>(i, 0) = Point3f(0, 0, 0);
// 					dists[i] = -1;
// 				}
// 				else
// 				{
// 					ExamplarPairCompare epc;
// 					sort(results.begin(), results.end(), epc);
// 					Vec3f normal = computeNormal(results);
// 					_Examplar mExm = results[0].first;
// 					Vec3f v((float)mExm[0], (float)mExm[1], (float)mExm[2]);
// 					v = v - oPoint;
// 					float dotProd = v.dot(normal);
// 					float distance = fabs(dotProd);
// 					Vec3f mPoint = oPoint - dotProd * normal;
// 					closestSet.at<Point3f>(i, 0) = Point3f(mPoint);
// 					dists[i] = (double)distance;
// 					threshold += distance;
// 					cnt++;
// 				}
// 			}
		}
		break;
	case CUDA:
/*		cuda_getClosestPoints(objSet, m_modSet, dists, threshold, &closestSet);*/
		break;
	}

	threshold /= (double)cnt;
	d = 0;
	for (int r = 0; r < rows; r++)
	{
		double dist = dists[r];
		if (dist < threshold && dist > 0)
		{
			float l = 1.0f;
			lambda.at<float>(r, 0) = l;
			d += l * dist;
		}
		else
		{
			lambda.at<float>(r, 0) = 0;
		}
	}
	d /= (double)cnt;
	return closestSet.clone();
}

void ICP::createKDTree()
{
	m_modPts = annAllocPts(m_modSet.rows, ICP_DIMS);
#pragma omp parallel for
	for (int r = 0; r < m_modSet.rows; r++)
	{
		Vec3f p = m_modSet.at<Vec3f>(r, 0);
		m_modPts[r][0] = p[0];
		m_modPts[r][1] = p[1]; 
		m_modPts[r][2] = p[2];	
	}
	m_kdTree = new ANNkd_tree(m_modPts, m_modSet.rows, ICP_DIMS);
}
