#include "examplarset.h"
#include <algorithm>
#include <fstream>

using namespace KDTree_R;

ExamplarSet::ExamplarSet(std::vector<_Examplar> ex_set, int size, int dims)
{
	if(size > 0)
		_size = size;
	else
		_size = 0;
	if(dims > 0)
		_dims = dims;
	else
		_dims = 0;

	_ex_set = ex_set;
}


ExamplarSet::ExamplarSet( int size, int dims )
{
	if(size > 0)
		_size = size;
	else
		_size = 0;
	if(dims > 0)
		_dims = dims;
	else
		_dims = 0;
}

ExamplarSet::ExamplarSet( const ExamplarSet& rhs )
{
	if(rhs._size > 0)
		_size = rhs._size;
	else
		_size = 0;
	if(rhs._dims > 0)
		_dims = rhs._dims;
	else
		_dims = 0;

	_ex_set = rhs._ex_set;
}

ExamplarSet& ExamplarSet::operator=( const ExamplarSet& rhs )
{

	if(this == &rhs) 
		return *this;

	releaseExamplarSetMem();

	if(rhs._size > 0)
		_size = rhs._size;
	else
		_size = 0;
	if(rhs._dims > 0)
		_dims = rhs._dims;
	else
		_dims = 0;

	_ex_set = rhs._ex_set;

	return *this;
}

void ExamplarSet::create( int size, int dims )
{
	releaseExamplarSetMem();
	if(size > 0 && dims > 0)
	{
		_ex_set.resize(size);
		_size = size;
		_dims = dims;
		for(int i=0;i<_size;i++)
		{
			_ex_set[i].create(_dims);
		}
	}
}

_HyperRectangle ExamplarSet::calculateRange()
{
	assert(_size > 0);
	assert(_dims > 0);
	_Examplar mn(_dims);
	_Examplar mx(_dims);

	for(int j=0;j<_dims;j++)
	{
		mn.dataAt(j) = (*this)[0][j];	//初始化最小范围向量
		mx.dataAt(j) = (*this)[0][j];	//初始化最大范围向量
	}

	for(int i=1;i<_size;i++)
	{
		for(int j=0;j<_dims;j++)
		{
			if( (*this)[i][j] < mn[j] )
				mn[j] = (*this)[i][j];
			if( (*this)[i][j] > mx[j] )
				mx[j] = (*this)[i][j];
		}
	}
	_HyperRectangle hr(mx, mn);

	return hr;
}

void ExamplarSet::sortByDim( int dim )
{
	ExamplarCompare cmp(dim);
	std::sort(_ex_set.begin(), _ex_set.end(), cmp);
}

bool ExamplarSet::remove( int idx )
{
	if(idx >=0 && idx < _size)
	{
		_ex_set.erase(_ex_set.begin() + idx);
		_size --;
		return true;
	}
	else
		return false;
}

int ExamplarSet::readData( char *strFilePath )
{
	std::ifstream fin(strFilePath);
	assert(fin != NULL);

	double temp;

	int row_id = 0, column_id = 0;;
	while(!fin.eof())
	{
		//! 获得一个数据样本

		for(column_id = 0;column_id < _dims;column_id++)
		{
			fin >> temp;
			(*this)[row_id][column_id] = temp;
		}

		row_id ++;
		if(row_id == _size)
			break;
	}

	fin.close();
	return 0;
}

double KDTree_R::Distance_exm( const _Examplar &x, const _Examplar &y )
{
	double dis;
	if(x.getDomDims() == y.getDomDims() && x.getDomDims() > 0)
	{
		dis = 0.0;
		for(int i=0;i<x.getDomDims();i++)
		{
			dis += (x[i] - y[i]) * (x[i] - y[i]);
		}
		dis = sqrt(dis);
	}
	else
		dis = -1.0;
	return dis;
}
