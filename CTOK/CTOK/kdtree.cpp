#include "kdtree.h"
#include <limits> 

using namespace KDTree_R;

void KDTree::create( const ExamplarSet &exm_set )
{
	_root = createKDTree(exm_set);
}

KDTreeNode* KDTree::createKDTree( const ExamplarSet &exm_set )
{
	if(exm_set.empty())
		return NULL;

	ExamplarSet exm_set_copy(exm_set);

	int dims = exm_set_copy.getDims();
	int size = exm_set_copy.getSize();

	//! 计算每个维的方差，选出方差值最大的维
	double var_max = -0.1; 
	double avg, var;
	int dim_max_var = -1;
	for(int i=0;i<dims;i++)
	{
		avg = 0;
		var = 0;
		//! 求某一维的总和
		for(int j=0;j<size;j++)
		{
			avg += exm_set_copy[j][i];
		}
		//! 求平均
		avg /= size;
		//! 求方差
		for(int j=0;j<size;j++)
		{
			var += ( exm_set_copy[j][i] - avg ) * 
				( exm_set_copy[j][i] - avg );
		}
		var /= size;
		if(var > var_max)
		{
			var_max = var;
			dim_max_var = i;
		}
	}

	//! 确定节点的数据矢量

	_HyperRectangle hr = exm_set_copy.calculateRange();
	exm_set_copy.sortByDim(dim_max_var);
	int mid = size / 2;
	_Examplar exm_split = exm_set_copy.examplarAt(mid);
	exm_set_copy.remove(mid);

	//! 确定左右节点

	ExamplarSet exm_set_left(0, exm_set_copy.getDims());
	ExamplarSet exm_set_right(0, exm_set_copy.getDims());
	exm_set_right.remove(0);

	int size_new = exm_set_copy.getSize();
	for(int i=0;i<size_new;i++)
	{
		_Examplar temp = exm_set_copy[i];
		if( temp.dataAt(dim_max_var) < 
			exm_split.dataAt(dim_max_var) )
			exm_set_left.push_back(temp);
		else
			exm_set_right.push_back(temp);
	}

	KDTreeNode *pNewNode = new KDTreeNode(0, 0, 0, dim_max_var, exm_split, hr);
	pNewNode->_left_child = createKDTree(exm_set_left);
	if(pNewNode->_left_child != NULL)
		pNewNode->_left_child->_parent = pNewNode;
	pNewNode->_right_child = createKDTree(exm_set_right);
	if(pNewNode->_right_child != NULL)
		pNewNode->_right_child->_parent = pNewNode;

	return pNewNode;
}

void KDTree::destroyKDTree( KDTreeNode *root )
{
	if(root != NULL)
	{
		destroyKDTree(root->_left_child);
		destroyKDTree(root->_right_child);
		delete root;
	}
}

void KDTree::destroy()
{
	destroyKDTree(_root);
}

std::pair<_Examplar, double> KDTree::findNearest_i( KDTreeNode *root, _Examplar target )
{
	//! 向下到达叶子节点

	KDTreeNode *pSearch = root;

	//! 堆栈用于保存搜索路径
	std::vector<KDTreeNode*> search_path;

	_Examplar nearest;

	double max_dist;

	while(pSearch != NULL)
	{
		search_path.push_back(pSearch);
		int s = pSearch->splitDim();
		if(target[s] <= pSearch->getDomElt()[s])
		{
			pSearch = pSearch->_left_child;
		}
		else
		{
			pSearch = pSearch->_right_child;
		}
	}

	nearest = search_path.back()->getDomElt();
	max_dist = Distance_exm(nearest, target);

	search_path.pop_back();

	//! 回溯搜索路径
	while(!search_path.empty())
	{
		KDTreeNode *pBack = search_path.back();
		search_path.pop_back();

		if( pBack->_left_child == NULL && pBack->_right_child == NULL)
		{
			if( Distance_exm(nearest, target) > Distance_exm(pBack->getDomElt(), target) )
			{
				nearest = pBack->getDomElt();
				max_dist = Distance_exm(pBack->getDomElt(), target);
			}
		}
		else
		{
			int s = pBack->splitDim();
			if( abs(pBack->getDomElt()[s] - target[s]) < max_dist)
			{
				if( Distance_exm(nearest, target) > Distance_exm(pBack->getDomElt(), target) )
				{
					nearest = pBack->getDomElt();
					max_dist = Distance_exm(pBack->getDomElt(), target);
				}
				if(target[s] <= pBack->getDomElt()[s])
					pSearch = pBack->_right_child;
				else
					pSearch = pBack->_left_child;
				if(pSearch != NULL)
					search_path.push_back(pSearch);
			}
		}
	}

	std::pair<_Examplar, double> res(nearest, max_dist);

	return res;
}

std::pair<_Examplar, double> KDTree::findNearest( _Examplar target )
{
	std::pair<_Examplar, double> res;
	if(_root == NULL)
	{
		res.second = std::numeric_limits<double>::infinity();
		return res;
	}
	return findNearest_i(_root, target);
}

int KDTree::findNearest( _Examplar target, double range, std::vector<std::pair<_Examplar, double>> &res_nearest )
{
	return findNearest_range(_root, target, range, res_nearest);
}

int KDTree::findNearest_range( KDTreeNode *root, _Examplar target, double range, 
	std::vector<std::pair<_Examplar, double>> &res_nearest )
{
	if(root == NULL)
		return 0;
	double dist_sq, dx;
	int ret, added_res = 0;
	dist_sq = 0;
	dist_sq = Distance_exm(root->getDomElt(), target);

	if(dist_sq <= range) {
		std::pair<_Examplar,double> temp(root->getDomElt(), dist_sq);
		res_nearest.push_back(temp);

		//! 结果个数+1

		added_res = 1;
	}

	dx = target[root->splitDim()] - root->getDomElt()[root->splitDim()];
	//! 左子树或右子树递归的查找
	ret = findNearest_range(dx <= 0.0 ? root->_left_child : root->_right_child, target, range, res_nearest);
	if(ret >= 0 && fabs(dx) < range) {
		added_res += ret;
		ret = findNearest_range(dx <= 0.0 ? root->_right_child : root->_left_child, target, range, res_nearest);
	}

	added_res += ret;
	return added_res;
}

