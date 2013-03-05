#ifndef ABSTRACTICP_H
#define ABSTRACTICP_H

#include "common.h"

#define SUBOUTPUT true

class AbstractICP
{
public:
	AbstractICP(){}
	~AbstractICP(){}

	Mat getTransformMat();												// 得到变换矩阵
	virtual void run(bool withCuda, Mat* initObjSet = NULL){};		// 运行ICP算法

protected:
	Mat m_objSet;				// 待配准的点云
	Mat m_modSet;				// 固定点云
	Transformation m_tr;		// 变换向量

	Transformation computeTransformation(const Mat &objSet, 
		const Mat &modSet, const Mat &lambda);						// 根据固定点云集与待配准的点云计算变换向量

	Transformation cuda_computeTransformation(const Mat &objSet,
		const Mat &modSet, const Mat &lambda);						// 使用cuda来计算变换矩阵
};

#endif