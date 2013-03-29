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

	Mat getFinalTransformMat(){return m_tr.clone();}					// �õ��任����
	Mat getTransformMat(const Transformation &tr);						// �õ��任����
	virtual void run(bool withCuda, Mat* initObjSet = NULL){};			// ����ICP�㷨

protected:
	Mat m_objSet;				// ����׼�ĵ���
	Mat m_modSet;				// �̶�����
	Mat m_tr;					// �任����

	// lambdaΪÿ�����Ȩ�صľ���
	Transformation computeTransformation(const Mat &objSet, 
		const Mat &modSet, const Mat &lambda);						// ���ݹ̶����Ƽ������׼�ĵ��Ƽ���任����

	Transformation cuda_computeTransformation(const Mat &objSet,
		const Mat &modSet, const Mat &lambda);						// ʹ��cuda������任����
};

void getRotateMatrix(Vec4f q, float* R);							// ����Ԫ��ת��Ϊ��ת����

#endif