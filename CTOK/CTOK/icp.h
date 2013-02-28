#ifndef ICP_H
#define ICP_H

#include "abstracticp.h"
#include "kdtree.h"

class ICP : public AbstractICP
{
public:

#define ICP_DIMS 3

	ICP(){}
	// iterMax：最大迭代次数；epsilon：精度
	ICP(const Mat &objSet, const Mat &modSet, 
		int iterMax = 50, double epsilon = 1e-6);

	void run(Mat* initObjSet = NULL);		// 运行ICP算法
	void cuda_run(Mat* initObjSet = NULL);

	// 获取对应点的方法
	// BASIC：直接遍历；KDTREE：构建KDTree；POINT_TO_PLANE：点到面
	enum Method{BASIC, KDTREE, POINT_TO_PLANE, CUDA};	

private:
	vector<Vec3f> m_modNormal;	// 固定点云的法向量
	int m_iterMax;				// 最大迭代次数
	double m_epsilon;			// 迭代容差精度
	KDTree_R::KDTree m_kdTree;	// modSet 的KDTree

	int m_cnt;					// 实际起作用的点的数目

	Mat getClosestPointsSet(const Mat &objSet, double &d,
		Mat &lambda, Method method = KDTREE);					// 计算固定点云集中与目标点云最近的点云及其权重

	void createKDTree();										// 创建KDTree						
};

// 将存放点云数据的Mat转换成ExamplarSet供KDTree使用
KDTree_R::ExamplarSet convertMatToExmSet(const Mat &mat);

// 根据已知点集估计出所有点拟合的平面的法向量
Vec3f computeNormal(vector<pair<KDTree_R::_Examplar, double>> points);

// EXTERN_C void cuda_getClosestPoints(const Mat &objSet, const Mat &modSet,
// 	vector<double> &diss, double &sum, Mat* resSet);

#endif