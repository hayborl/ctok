#ifndef ABSTRACTICP_H
#define ABSTRACTICP_H

#include "common.h"

#define SUBOUTPUT false

class Quaternion
{
public:
	Quaternion(Vec3d v);
	Quaternion(Vec4d q) : m_q(q){}
	Quaternion(const Quaternion &q);
	Quaternion(double q0 = 0.0, double q1 = 0.0, 
		double q2 = 0.0, double q3 = 0.0);

	void normalize();							// 归一化
	Mat toMatrix()const;						// 将四元数转换为旋转矩阵
	double mag2()const;							// 模的平方
	double mag()const;							// 模
	Quaternion getConjugate()const;				// 获得共轭四元数
	Quaternion operator=(const Quaternion &q);
	Quaternion operator*(const Quaternion &q);	// 四元数相乘
	Vec3d operator*(const Vec3d &v);			// 将点经四元数代表的旋转矩阵变换
	const double& operator[](int i)const;
	double& operator[](int i);

private:
	Vec4d m_q;
};

class Transformation
{
public:
	Transformation(Quaternion q = Quaternion(1.0), 
		Vec3d t = Vec3d(0.0, 0.0, 0.0)) : m_q(q), m_t(t){}

	Transformation operator=(const Transformation &t);
	Transformation operator*(const Transformation &t);
	Vec3d operator*(const Vec3d &v);			// 将点做变换
	Mat toMatrix()const;
	void normalizaQuaternion(){m_q.normalize();}

	Quaternion m_q;
	Vec3d m_t;
};

class AbstractICP
{
public:
	AbstractICP(){}
	~AbstractICP(){}

	Transformation getFinalTransformation(){return m_tr;}			// 得到最终变换
	Mat getFinalTransformMat(){return m_tr.toMatrix();}			// 得到最终变换矩阵
	virtual void run(bool withCuda, 
		InputArray initObjSet = noArray()){};						// 运行ICP算法

protected:
	Mat m_objSet;				// 待配准的点云
	Mat m_modSet;				// 固定点云
	Transformation m_tr;		// 变换矩阵

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

#endif