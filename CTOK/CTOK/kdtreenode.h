/*! \file kdtreenode.h
    \brief declaration of KDTreeNode
    
    \author RaySaint 

*/

#ifndef KDTREENODE_H
#define KDTREENODE_H

#include "examplarset.h"

namespace KDTree_R
{
	class KDTreeNode
	{
	private:
		int _split_dim;
		_Examplar _dom_elt;
		_HyperRectangle _range_hr;
	public:
		KDTreeNode *_left_child, *_right_child, *_parent;
	public:
		KDTreeNode():_left_child(0), _right_child(0), _parent(0), 
			_split_dim(0){}
		KDTreeNode(KDTreeNode *left_child, KDTreeNode *right_child, 
			KDTreeNode *parent, int split_dim, _Examplar dom_elt, _HyperRectangle range_hr):
		_left_child(left_child), _right_child(right_child), _parent(parent),
			_split_dim(split_dim), _dom_elt(dom_elt), _range_hr(range_hr){}
		KDTreeNode(const KDTreeNode &rhs);
		KDTreeNode& operator=(const KDTreeNode &rhs);
		_Examplar& getDomElt() { return _dom_elt; }
		_HyperRectangle& getHyperRectangle(){ return _range_hr; }
		int& splitDim(){ return _split_dim; }
		void create(KDTreeNode *left_child, KDTreeNode *right_child, 
			KDTreeNode *parent, int split_dim, _Examplar dom_elt,  _HyperRectangle range_hr);
	};
}

#endif