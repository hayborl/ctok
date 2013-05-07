#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "triangulation.h"
#include "ANN/ANN.h"

using namespace std;

// 使用kmeans方法分割
void segment3DKmeans(Triangulation::Mesh mesh, 
	vector<Triangulation::Mesh> &segs);

// 使用球状最近邻方法分割
#define SEG_K	200					// 最近邻最大个数
#define SEG_DISTANCE_RANGE 0.0025	// 距离的平方(0.05m)^2

int computeLabels(Triangulation::Mesh &mesh, 
	vector<int> &labels, map<int, int> &labelMap);
void segment3DRBNN(Triangulation::Mesh &mesh, 
	vector<Triangulation::Mesh> &segs);

#endif
