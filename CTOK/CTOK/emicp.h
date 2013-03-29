#ifndef EMICP_H
#define EMICP_H

#include "abstracticp.h"

class EMICP : public AbstractICP
{
public:
	EMICP(){}
	// <Softassign and EM-ICP on GPU>
	EMICP(const Mat &objSet, const Mat &modSet,
		float sigma_p2 = 0.01f, float sigma_inf = 0.00001f, 
		float sigma_factor = 0.8f, float d_02 = 0.01f);
	~EMICP(){}

	void run(bool withCuda, Mat* initObjSet = NULL);		// 运行ICP算法

private:
	float m_sigma_p2;			// square(sigma_p)
	float m_sigma_inf;			// sigma临界值
	float m_sigma_factor;		// sigma变换因子
	float m_d_02;				// square(d_0)

	void updateA(Mat &A, const Mat &objSet,
		const Mat &modSet, bool withCuda = false);				// 计算所需的矩阵A
	void normalizeRows(Mat &mat, const Mat &alpha, 
		bool withCuda = false, bool withSqrt = false);			// rows(A)[i] / alpha[i]

	void cuda_updateA(Mat &A, const Mat &objSet, const Mat &modSet);
	void cuda_normalizeRows(Mat &mat, 
		const Mat &alpha, bool withSqrt = false);
};

#endif