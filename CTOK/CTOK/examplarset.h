/*! \file examplarset.h
    \brief declaration of ExamplarSet
    
    \author RaySaint 

*/

#ifndef EXAMPLARSET_H
#define EXAMPLARSET_H

#include <memory.h>
#include <stdio.h>
#include "assert.h"
#include <vector>

namespace KDTree_R
{
	struct _Examplar
	{
	public:

		_Examplar():dom_dims(0){}		//数据维度初始化为0
		_Examplar(const std::vector<double> elt, int dims)	//带有完整的两个参数的constructor
		{													//这里const是为了保护原数据不被修改
			if(dims > 0)
			{
				dom_elt = elt;
				dom_dims = dims;
			}
			else
			{
				dom_dims = 0;
			}
		}
		_Examplar(int dims)	//只含有维度信息的constructor
		{
			if(dims > 0)
			{
				dom_elt.resize(dims);
				dom_dims = dims;
			}
			else
			{
				dom_dims = 0;
			}
		}
		_Examplar(const _Examplar& rhs)	//copy-constructor
		{
			if(rhs.dom_dims > 0)
			{
				dom_elt = rhs.dom_elt;
				dom_dims = rhs.dom_dims;
			}
			else
			{
				dom_dims = 0;
			}
		}
		_Examplar& operator=(const _Examplar& rhs)	//重载"="运算符
		{
			if(this == &rhs) 
				return *this;

			releaseExamplarMem();

			if(rhs.dom_dims > 0)
			{
				dom_elt = rhs.dom_elt;
				dom_dims = rhs.dom_dims;
			}

			return *this;
		}
		~_Examplar()
		{
			releaseExamplarMem();
		}
		double& dataAt(int dim)	//定义访问控制函数
		{
			assert(dim < dom_dims);
			return dom_elt[dim];
		}
		double& operator[](int dim)	//重载"[]"运算符，实现下标访问
		{
			return dataAt(dim);
		}
		const double& dataAt(int dim) const	//定义只读访问函数
		{
			assert(dim < dom_dims);
			return dom_elt[dim];
		}
		const double& operator[](int dim) const	//重载"[]"运算符，实现下标只读访问
		{
			return dataAt(dim);
		}
		void create(int dims)	//创建数据向量
		{
			releaseExamplarMem();
			if(dims > 0)
			{
				dom_elt.resize(dims);	//控制数据向量维度
				dom_dims = dims;
			}
		}
		int getDomDims() const 	//获得数据向量维度信息
		{
			return dom_dims;
		}
		void setTo(double val)	//数据向量初始化设置
		{
			if(dom_dims > 0)
			{
				for(int i=0;i<dom_dims;i++)
				{
					dom_elt[i] = val;
				}
			}
		}
		std::vector<double> data()
		{ 
			return dom_elt; 
		}
	private:
		void releaseExamplarMem()	//清除现有数据向量
		{
			dom_elt.clear();
			dom_dims = 0;
		}
	private:
		std::vector<double> dom_elt;	//每个数据定义为一个double类型的向量
		int dom_dims;					//数据向量的维度
	};

	double Distance_exm(const _Examplar &x, const _Examplar &y);	//定义的距离函数

	class ExamplarCompare	//定义数据向量比较类，产生的对象用于sort的comp
	{
	public:
		ExamplarCompare(const int dim) : _dim(dim){}	//这里的dim是指待比较的方向
		bool
			operator()(const _Examplar &x, const _Examplar &y) const
		{
			return x[_dim] < y[_dim];
		}
	private:
		int _dim;	// don't make this const so that an assignment operator can be auto-generated
	};

	class ExamplarPairCompare	//定义数据向量比较类，产生的对象用于sort的comp
	{
	public:
		ExamplarPairCompare(){}
		bool
			operator()(const std::pair<_Examplar, double> &x, 
			const std::pair<_Examplar, double> &y) const
		{
			return x.second < y.second;
		}
	};

	struct _HyperRectangle	//定义表示数据范围的超矩形结构
	{
		_Examplar min;		//统计数据集中所有数据向量每个维度上最小值组成的一个数据向量
		_Examplar max;		//统计数据集中所有数据向量每个维度上最大值组成的一个数据向量
		_HyperRectangle() {}
		_HyperRectangle(_Examplar mx, _Examplar mn)
		{
			assert (mx.getDomDims() == mn.getDomDims());
			min = mn;
			max = mx;
		}
		_HyperRectangle(const _HyperRectangle& rhs)
		{
			min = rhs.min;
			max = rhs.max;
		}
		_HyperRectangle& operator= (const _HyperRectangle& rhs)
		{
			if(this == &rhs)
				return *this;
			min = rhs.min;
			max = rhs.max;
			return *this;
		}
		void create(_Examplar mx, _Examplar mn)
		{
			assert (mx.getDomDims() == mn.getDomDims());
			min = mn;
			max = mx;
		}
	};

	class ExamplarSet
	{
	private:
		//_Examplar *_ex_set;
		std::vector<_Examplar> _ex_set;		//定义含有若干个_Examplar类数据向量的数据集
		int _size;							//数据集大小
		int _dims;							//数据集中每个数据向量的维度
	public:
		ExamplarSet():_size(0), _dims(0){}
		ExamplarSet(std::vector<_Examplar> ex_set, int size, int dims);
		ExamplarSet(int size, int dims);
		ExamplarSet(const ExamplarSet& rhs);
		ExamplarSet& operator=(const ExamplarSet& rhs);
		~ExamplarSet(){}

		_Examplar& examplarAt(int idx)
		{ 
			assert(idx < _size);
			return _ex_set[idx]; 
		}
		_Examplar& operator[](int idx)
		{
			return examplarAt(idx);
		}
		const _Examplar& examplarAt(int idx) const
		{
			assert(idx < _size);
			return _ex_set[idx];
		}
		void create(int size, int dims);
		int getDims() const { return _dims;}
		int getSize() const { return _size;}
		_HyperRectangle calculateRange();
		bool empty() const
		{
			return (_size == 0);
		}

		void sortByDim(int dim);	//按某个方向维的排序函数
		bool remove(int idx);		//去除数据集中排序后指定位置的数据向量
		void push_back(const _Examplar& ex)	//添加某个数据向量至数据集末尾
		{
			_ex_set.push_back(ex);
			_size++;
		}

		int readData(char *strFilePath);	//从文件读取数据集
	private:
		void releaseExamplarSetMem()		//清除现有数据集
		{
			_ex_set.clear();
			_size = 0;
		}
	};
}

#endif