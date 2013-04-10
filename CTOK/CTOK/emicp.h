#ifndef EMICP_H
#define EMICP_H

#include "abstracticp.h"

class EMICP : public AbstractICP
{
public:
	EMICP(){}
	// <Softassign and EM-ICP on GPU>
	EMICP(const Mat &objSet, const Mat &modSet,
		double sigma_p2 = 0.01, double sigma_inf = 0.00001, 
		double sigma_factor = 0.8, double d_02 = 0.01);
	~EMICP(){}

	void run(bool withCuda, InputArray initObjSet = noArray());	// ����ICP�㷨

private:
	double m_sigma_p2;			// square(sigma_p)
	double m_sigma_inf;			// sigma�ٽ�ֵ
	double m_sigma_factor;		// sigma�任����
	double m_d_02;				// square(d_0)

	void updateA(Mat &A, const Mat &objSet,
		const Mat &modSet, bool withCuda = false);				// ��������ľ���A
	void normalizeRows(Mat &mat, const Mat &alpha, 
		bool withCuda = false, bool withSqrt = false);			// rows(A)[i] / alpha[i]

	void cuda_updateA(Mat &A, const Mat &objSet, const Mat &modSet);
	void cuda_normalizeRows(Mat &mat, 
		const Mat &alpha, bool withSqrt = false);
};

#endif