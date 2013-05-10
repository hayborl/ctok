#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "triangulation.h"
#include "ANN/ANN.h"

using namespace std;

// ʹ��kmeans�����ָ�
void segment3DKmeans(Triangulation::Mesh mesh, 
	vector<Triangulation::Mesh> &segs);

// ʹ����״����ڷ����ָ�
#define SEG_K	200					// �����������
#define SEG_DISTANCE_RANGE 0.04		// �����ƽ��(0.05m)^2

int computeLabels(const int &k, Triangulation::Mesh &mesh, 
	vector<int> &labels, map<int, int> &labelMap);
void segment3DRBNN(const int &k, 
	Triangulation::Mesh &mesh, vector<Triangulation::Mesh> &segs);

// ʹ��smoothness constraiant�������ָ�
void segment3DSC(Triangulation::Mesh &mesh, 
	vector<Triangulation::Mesh> &segs);

#endif
