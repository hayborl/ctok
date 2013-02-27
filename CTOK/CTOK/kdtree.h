/*! \file kdtree.h
    \brief declaration of KDTree
    
    \author RaySaint 

*/

#ifndef KDTREE_H
#define KDTREE_H

#include "kdtreenode.h"

namespace KDTree_R
{
	class KDTree	//k-d tree�ṹ����
	{
	public:
		KDTreeNode *_root;		//k-d tree�ĸ��ڵ�
	public:
		KDTree():_root(NULL){}
		void create(const ExamplarSet &exm_set);		//����k-d tree��ʵ���ϵ���createKDTree
		void destroy();									//����k-d tree��ʵ���ϵ���destroyKDTree
		~KDTree(){ destroyKDTree(_root); }
		std::pair<_Examplar, double> findNearest(_Examplar target);	//��������ڵ㺯��������ֵ��pair����
																	//ʵ���ǵ���findNearest_i
		//���Ҿ�����range��Χ�ڵĽ��ڵ㣬�����������ڵ�ĸ�����ʵ���ǵ���findNearest_range
		int findNearest(_Examplar target, double range, std::vector<std::pair<_Examplar, double>> &res_nearest);
	private:
		KDTreeNode* createKDTree(const ExamplarSet &exm_set);
		void destroyKDTree(KDTreeNode *root);
		std::pair<_Examplar, double> findNearest_i(KDTreeNode *root, _Examplar target);
		int findNearest_range(KDTreeNode *root, _Examplar target, double range, 
			std::vector<std::pair<_Examplar, double>> &res_nearest);
	};
}

#endif