#ifndef ABSTRACTICP_H
#define ABSTRACTICP_H

#include "common.h"

#define SUBOUTPUT true

class AbstractICP
{
public:
	AbstractICP(){}
	~AbstractICP(){}

	Mat getTransformMat();												// �õ��任����
	virtual void run(bool withCuda, Mat* initObjSet = NULL){};		// ����ICP�㷨

protected:
	Mat m_objSet;				// ����׼�ĵ���
	Mat m_modSet;				// �̶�����
	Transformation m_tr;		// �任����

	Transformation computeTransformation(const Mat &objSet, 
		const Mat &modSet, const Mat &lambda);						// ���ݹ̶����Ƽ������׼�ĵ��Ƽ���任����

	Transformation cuda_computeTransformation(const Mat &objSet,
		const Mat &modSet, const Mat &lambda);						// ʹ��cuda������任����
};

#endif