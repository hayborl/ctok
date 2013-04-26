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

	void normalize();							// ��һ��
	Mat toMatrix()const;						// ����Ԫ��ת��Ϊ��ת����
	double mag2()const;							// ģ��ƽ��
	double mag()const;							// ģ
	Quaternion getConjugate()const;				// ��ù�����Ԫ��
	Quaternion operator=(const Quaternion &q);
	Quaternion operator*(const Quaternion &q);	// ��Ԫ�����
	Vec3d operator*(const Vec3d &v);			// ���㾭��Ԫ���������ת����任
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
	Vec3d operator*(const Vec3d &v);			// �������任
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

	Transformation getFinalTransformation(){return m_tr;}			// �õ����ձ任
	Mat getFinalTransformMat(){return m_tr.toMatrix();}			// �õ����ձ任����
	virtual void run(bool withCuda, 
		InputArray initObjSet = noArray()){};						// ����ICP�㷨

protected:
	Mat m_objSet;				// ����׼�ĵ���
	Mat m_modSet;				// �̶�����
	Transformation m_tr;		// �任����

	void initTransform(Mat &initObj, bool withCuda);				// ���д���׼�õ���׼��ĵ���

	// lambdaΪÿ�����Ȩ�صľ���
	Transformation computeTransformation(const Mat &objSet, 
		const Mat &modSet, const Mat &lambda);						// ���ݹ̶����Ƽ������׼�ĵ��Ƽ���任����

	Transformation cuda_computeTransformation(const Mat &objSet,
		const Mat &modSet, const Mat &lambda);						// ʹ��cuda������任����

	double computeError(const Mat &objAfterTrans, 
		const Mat &mod, const Mat &lambda, bool withCuda);			// �������

private:
	void getConstructionMat(const Mat &in, Mat &out);				// �����ɷַ����������������
};

#endif