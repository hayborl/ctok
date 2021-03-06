#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "triangulation.h"
#include "ANN/ANN.h"

using namespace std;

// 使用kmeans方法分割
void segment3DKmeans(Triangulation::Mesh mesh, 
	vector<Triangulation::Mesh> &segs, int kNum);

// 使用球状最近邻方法分割
#define SEG_K	200					// 最近邻最大个数
#define SEG_DISTANCE_RANGE 0.04		// 距离的平方(0.05m)^2

int search(double dists[], const double &dist, const int &n);

int computeLabels(const int &k, Triangulation::Mesh &mesh, 
	vector<int> &labels, map<int, int> &labelMap);
void segment3DRBNN(const int &k, 
	Triangulation::Mesh &mesh, vector<Triangulation::Mesh> &segs);

// 使用smoothness constraiant方法来分割
void segment3DSC(Triangulation::Mesh &mesh, 
	vector<Triangulation::Mesh> &segs);

#endif
