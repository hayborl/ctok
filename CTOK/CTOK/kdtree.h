/*! \file kdtree.h
    \brief declaration of KDTree
    
    \author RaySaint 

*/

#ifndef KDTREE_H
#define KDTREE_H

#include "kdtreenode.h"

namespace KDTree_R
{
	class KDTree	//k-d tree结构定义
	{
	public:
		KDTreeNode *_root;		//k-d tree的根节点
	public:
		KDTree():_root(NULL){}
		void create(const ExamplarSet &exm_set);		//创建k-d tree，实际上调用createKDTree
		void destroy();									//销毁k-d tree，实际上调用destroyKDTree
		~KDTree(){ destroyKDTree(_root); }
		std::pair<_Examplar, double> findNearest(_Examplar target);	//查找最近邻点函数，返回值是pair类型
																	//实际是调用findNearest_i
		//查找距离在range范围内的近邻点，返回这样近邻点的个数，实际是调用findNearest_range
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