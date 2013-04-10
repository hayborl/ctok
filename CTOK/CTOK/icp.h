#ifndef ICP_H
#define ICP_H

#include "abstracticp.h"

class ICP : public AbstractICP
{
public:

#define ICP_DIMS 3

	ICP(){}
	// iterMax��������������epsilon������
	ICP(const Mat &objSet, const Mat &modSet, 
		int iterMax = 50, double epsilon = 1e-6);

	void run(bool withCuda, InputArray initObjSet = noArray());		// ����ICP�㷨

	// ��ȡ��Ӧ��ķ���
	// BASIC��ֱ�ӱ�����KDTREE������KDTree��POINT_TO_PLANE���㵽��
	enum Method{BASIC, KDTREE, POINT_TO_PLANE, CUDA};	

private:
	vector<Vec3d> m_modNormal;	// �̶����Ƶķ�����
	int m_iterMax;				// ����������
	double m_epsilon;			// �����ݲ��
	ANNkd_tree* m_kdTree;		// modSet ��KDTree
	ANNpointArray m_modPts;		// for kdTree

	int m_cnt;					// ʵ�������õĵ����Ŀ

	Mat getClosestPointsSet(const Mat &objSet,
		Mat &lambda, Method method = KDTREE);					// ����̶����Ƽ�����Ŀ���������ĵ��Ƽ���Ȩ��

	void createKDTree();										// ����KDTree						
};

#endif