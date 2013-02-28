#ifndef ABSTRACTICP_H
#define ABSTRACTICP_H

#include "common.h"

#define SUBOUTPUT false

class AbstractICP
{
public:
	AbstractICP(){}
	~AbstractICP(){}

	Mat getTransformMat();								// �õ��任����
	virtual void run(Mat* initObjSet = NULL){};		// ����ICP�㷨
	virtual void cuda_run(Mat* initObjSet = NULL){};	// ��cuda��������

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