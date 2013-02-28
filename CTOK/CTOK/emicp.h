#ifndef EMICP_H
#define EMICP_H

#include "abstracticp.h"

EXTERN_C class EMICP : public AbstractICP
{
public:
	EMICP(){}
	// <Softassign and EM-ICP on GPU>
	EMICP(const Mat &objSet, const Mat &modSet,
		float sigma_p2 = 0.01f, float sigma_inf = 0.00001f, 
		float sigma_factor = 0.8f, float d_02 = 0.01f);
	~EMICP(){}

	void run(Mat* initObjSet = NULL);		// ����ICP�㷨
	void cuda_run(Mat* initObjSet = NULL);

private:
	float m_sigma_p2;			// square(sigma_p)
	float m_sigma_inf;			// sigma�ٽ�ֵ
	float m_sigma_factor;		// sigma�任����
	float m_d_02;				// square(d_0)

	void updateA(Mat &A, const Mat &objSet, const Mat &R, const Mat &T);	// ��������ľ���A
	void normalizeRows(Mat &mat, const Mat &alpha);						// row(A)[i] / alpha[i]

	void cuda_updateA(Mat &A, const Mat &objSet, 
		const Mat &modSet, float* h_R, float* h_T);
};

#endif