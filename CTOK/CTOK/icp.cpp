#include "icp.h"

ICP::ICP( const Mat &objSet, const Mat &modSet, int iterMax, double epsilon )
{
	assert(objSet.cols == 1 && modSet.cols == 1);
	assert(!objSet.empty() && !modSet.empty());

	m_objSet = objSet.clone();
	m_modSet = modSet.clone();
	m_iterMax = iterMax;
	m_epsilon = epsilon;
	m_tr = Mat::eye(4, 4, CV_64FC1);

	RUNANDTIME(global_timer, createKDTree(),
		OUTPUT && SUBOUTPUT, "create kdTree");
}

void ICP::run(bool withCuda, InputArray initObjSet)
{
	assert(!m_objSet.empty() && !m_modSet.empty());

	double d_pre = 100000, d_now = 100000;
	int iterCnt = 0;
	Mat objSet;
	Transformation tr;

	if (initObjSet.empty())
	{
		objSet = m_objSet.clone();
	}
	else
	{
		objSet = initObjSet.getMat();
	}

/*	plotTwoPoint3DSet(objSet, m_modSet);*/

	do 
	{
		d_pre = d_now;

		Mat closestSet;
		Mat lambda(objSet.rows, 1, CV_64FC1);
		RUNANDTIME(global_timer, closestSet = 
			getClosestPointsSet(objSet, lambda, KDTREE).clone(), 
			OUTPUT && SUBOUTPUT, "compute closest points.");
		Mat tmpObjSet = convertMat(m_objSet);
		Mat tmpModSet = convertMat(closestSet);
		RUNANDTIME(global_timer, tr = 
			computeTransformation(tmpObjSet, tmpModSet, lambda), 
			OUTPUT && SUBOUTPUT, "compute transformation");
		Mat transformMat = getTransformMat(tr);
		RUNANDTIME(global_timer, transformPointCloud(
			m_objSet, objSet, transformMat, withCuda), 
			OUTPUT && SUBOUTPUT, "transform points.");
		RUNANDTIME(global_timer, 
			d_now = computeError(objSet, closestSet, lambda, withCuda),
			OUTPUT && SUBOUTPUT, "compute error.");

		iterCnt++;
	} while (fabs(d_pre - d_now) > m_epsilon && iterCnt <= m_iterMax);

	m_tr = getTransformMat(tr);
/*	waitKey();*/

/*	plotTwoPoint3DSet(objSet, m_modSet);*/
}

Mat ICP::getClosestPointsSet( const Mat &objSet, Mat &lambda, Method method )
{
	int rows = objSet.rows;
	Mat closestSet(rows, 1, DataType<Point3d>::type);
	vector<double> dists(rows);
	double threshold = 0;
	int cnt = 0;

	switch (method)
	{
	case BASIC:
		for (int i = 0; i < rows; i++)
		{
			int minIndex = 0;
			double minDistance = DISTANCE_MAX;
			Point3d oPoint = objSet.at<Point3d>(i, 0);
			for (int j = 0; j < m_modSet.rows; j++)
			{
				Point3d p = m_modSet.at<Point3d>(j, 0) - oPoint;
				double distance = sqrt(p.dot(p));
				if (distance < minDistance)
				{
					minDistance = distance;
					minIndex = j;
				}
			}
			closestSet.at<Point3d>(i, 0) = m_modSet.at<Point3d>(minIndex, 0);
			dists[i] = minDistance;
			threshold += minDistance;
			cnt++;
		}
		break;
	case KDTREE:
	default:
		for (int i = 0; i < rows; i++)
		{
			Vec3d oPoint = objSet.at<Vec3d>(i, 0);
			ANNpoint qp = annAllocPt(ICP_DIMS);
			qp[0] = oPoint[0]; 
			qp[1] = oPoint[1];
			qp[2] = oPoint[2];
			ANNidx idx[1];
			ANNdist dist[1];
			m_kdTree->annkSearch(qp, 1, idx, dist);
			closestSet.at<Point3d>(i, 0) = Point3d(
				m_modPts[idx[0]][0], m_modPts[idx[0]][1], 
				m_modPts[idx[0]][2]);
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
// 				Vec3d oPoint = Vec3d(objSet.at<Point3d>(i, 0));
// 				_Examplar exm(ICP_DIMS);
// 				for (int j = 0; j < ICP_DIMS; j++)
// 				{
// 					exm[j] = oPoint[j];
// 				}
// 				vector<pair<_Examplar, double>> results;
// 				if (m_kdTree.findNearest(exm, DISTANCE_RANGE, results) < 3)
// 				{
// 					closestSet.at<Point3d>(i, 0) = Point3d(0, 0, 0);
// 					dists[i] = -1;
// 				}
// 				else
// 				{
// 					ExamplarPairCompare epc;
// 					sort(results.begin(), results.end(), epc);
// 					Vec3d normal = computeNormal(results);
// 					_Examplar mExm = results[0].first;
// 					Vec3d v((double)mExm[0], (double)mExm[1], (double)mExm[2]);
// 					v = v - oPoint;
// 					double dotProd = v.dot(normal);
// 					double distance = fabs(dotProd);
// 					Vec3d mPoint = oPoint - dotProd * normal;
// 					closestSet.at<Point3d>(i, 0) = Point3d(mPoint);
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
	for (int r = 0; r < rows; r++)
	{
		double dist = dists[r];
		if (dist < threshold && dist > 0)
		{
			double l = 1.0f;
			lambda.at<double>(r, 0) = l;
		}
		else
		{
			lambda.at<double>(r, 0) = 0;
		}
	}
	return closestSet.clone();
}

void ICP::createKDTree()
{
	m_modPts = annAllocPts(m_modSet.rows, ICP_DIMS);
#pragma omp parallel for
	for (int r = 0; r < m_modSet.rows; r++)
	{
		Vec3d p = m_modSet.at<Vec3d>(r, 0);
		m_modPts[r][0] = p[0];
		m_modPts[r][1] = p[1]; 
		m_modPts[r][2] = p[2];	
	}
	m_kdTree = new ANNkd_tree(m_modPts, m_modSet.rows, ICP_DIMS);
}
