#ifndef ABSTRACTICP_H
#define ABSTRACTICP_H

#include "common.h"

#define SUBOUTPUT false

typedef struct tag_Transformation
{
	Vec4f q;
	Vec3f t;
} Transformation;

class AbstractICP
{
public:
	AbstractICP(){}
	~AbstractICP(){}

	Mat getFinalTransformMat(){return m_tr.clone();}				// 得到变换矩阵
	Mat getTransformMat(const Transformation &tr);					// 得到变换矩阵
	virtual void run(bool withCuda, 
		InputArray initObjSet = noArray()){};						// 运行ICP算法

protected:
	Mat m_objSet;				// 待配准的点云
	Mat m_modSet;				// 固定点云
	Mat m_tr;					// 变换矩阵

	void initTransform(Mat &initObj, bool withCuda);				// 进行粗配准得到配准后的点云

	// lambda为每个点的权重的矩阵
	Transformation computeTransformation(const Mat &objSet, 
		const Mat &modSet, const Mat &lambda);						// 根据固定点云集与待配准的点云计算变换向量

	Transformation cuda_computeTransformation(const Mat &objSet,
		const Mat &modSet, const Mat &lambda);						// 使用cuda来计算变换矩阵

	double computeError(const Mat &objAfterTrans, 
		const Mat &mod, const Mat &lambda, bool withCuda);			// 计算误差

private:
	void getConstructionMat(const Mat &in, Mat &out);				// 用主成分分析法建立构造矩阵
};

void getRotateMatrix(Vec4f q, float* R);							// 将四元数转换为旋转矩阵

#endif