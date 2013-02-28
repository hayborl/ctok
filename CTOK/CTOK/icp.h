#ifndef ICP_H
#define ICP_H

#include "abstracticp.h"
#include "kdtree.h"

class ICP : public AbstractICP
{
public:

#define ICP_DIMS 3

	ICP(){}
	// iterMax��������������epsilon������
	ICP(const Mat &objSet, const Mat &modSet, 
		int iterMax = 50, double epsilon = 1e-6);

	void run(Mat* initObjSet = NULL);		// ����ICP�㷨
	void cuda_run(Mat* initObjSet = NULL);

	// ��ȡ��Ӧ��ķ���
	// BASIC��ֱ�ӱ�����KDTREE������KDTree��POINT_TO_PLANE���㵽��
	enum Method{BASIC, KDTREE, POINT_TO_PLANE, CUDA};	

private:
	vector<Vec3f> m_modNormal;	// �̶����Ƶķ�����
	int m_iterMax;				// ����������
	double m_epsilon;			// �����ݲ��
	KDTree_R::KDTree m_kdTree;	// modSet ��KDTree

	int m_cnt;					// ʵ�������õĵ����Ŀ

	Mat getClosestPointsSet(const Mat &objSet, double &d,
		Mat &lambda, Method method = KDTREE);					// ����̶����Ƽ�����Ŀ���������ĵ��Ƽ���Ȩ��

	void createKDTree();										// ����KDTree						
};

// ����ŵ������ݵ�Matת����ExamplarSet��KDTreeʹ��
KDTree_R::ExamplarSet convertMatToExmSet(const Mat &mat);

// ������֪�㼯���Ƴ����е���ϵ�ƽ��ķ�����
Vec3f computeNormal(vector<pair<KDTree_R::_Examplar, double>> points);

// EXTERN_C void cuda_getClosestPoints(const Mat &objSet, const Mat &modSet,
// 	vector<double> &diss, double &sum, Mat* resSet);

#endif