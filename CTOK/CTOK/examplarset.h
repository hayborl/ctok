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

		_Examplar():dom_dims(0){}		//����ά�ȳ�ʼ��Ϊ0
		_Examplar(const std::vector<double> elt, int dims)	//��������������������constructor
		{													//����const��Ϊ�˱���ԭ���ݲ����޸�
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
		_Examplar(int dims)	//ֻ����ά����Ϣ��constructor
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
		_Examplar& operator=(const _Examplar& rhs)	//����"="�����
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
		double& dataAt(int dim)	//������ʿ��ƺ���
		{
			assert(dim < dom_dims);
			return dom_elt[dim];
		}
		double& operator[](int dim)	//����"[]"�������ʵ���±����
		{
			return dataAt(dim);
		}
		const double& dataAt(int dim) const	//����ֻ�����ʺ���
		{
			assert(dim < dom_dims);
			return dom_elt[dim];
		}
		const double& operator[](int dim) const	//����"[]"�������ʵ���±�ֻ������
		{
			return dataAt(dim);
		}
		void create(int dims)	//������������
		{
			releaseExamplarMem();
			if(dims > 0)
			{
				dom_elt.resize(dims);	//������������ά��
				dom_dims = dims;
			}
		}
		int getDomDims() const 	//�����������ά����Ϣ
		{
			return dom_dims;
		}
		void setTo(double val)	//����������ʼ������
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
		void releaseExamplarMem()	//���������������
		{
			dom_elt.clear();
			dom_dims = 0;
		}
	private:
		std::vector<double> dom_elt;	//ÿ�����ݶ���Ϊһ��double���͵�����
		int dom_dims;					//����������ά��
	};

	double Distance_exm(const _Examplar &x, const _Examplar &y);	//����ľ��뺯��

	class ExamplarCompare	//�������������Ƚ��࣬�����Ķ�������sort��comp
	{
	public:
		ExamplarCompare(const int dim) : _dim(dim){}	//�����dim��ָ���Ƚϵķ���
		bool
			operator()(const _Examplar &x, const _Examplar &y) const
		{
			return x[_dim] < y[_dim];
		}
	private:
		int _dim;	// don't make this const so that an assignment operator can be auto-generated
	};

	class ExamplarPairCompare	//�������������Ƚ��࣬�����Ķ�������sort��comp
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

	struct _HyperRectangle	//�����ʾ���ݷ�Χ�ĳ����νṹ
	{
		_Examplar min;		//ͳ�����ݼ���������������ÿ��ά������Сֵ��ɵ�һ����������
		_Examplar max;		//ͳ�����ݼ���������������ÿ��ά�������ֵ��ɵ�һ����������
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
		std::vector<_Examplar> _ex_set;		//���庬�����ɸ�_Examplar���������������ݼ�
		int _size;							//���ݼ���С
		int _dims;							//���ݼ���ÿ������������ά��
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

		void sortByDim(int dim);	//��ĳ������ά��������
		bool remove(int idx);		//ȥ�����ݼ��������ָ��λ�õ���������
		void push_back(const _Examplar& ex)	//���ĳ���������������ݼ�ĩβ
		{
			_ex_set.push_back(ex);
			_size++;
		}

		int readData(char *strFilePath);	//���ļ���ȡ���ݼ�
	private:
		void releaseExamplarSetMem()		//����������ݼ�
		{
			_ex_set.clear();
			_size = 0;
		}
	};
}

#endif