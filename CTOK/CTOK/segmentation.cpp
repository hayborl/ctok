#include "segmentation.h"

#include "opencv2/opencv.hpp"
#include <queue>

using namespace Triangulation;
using namespace cv;

extern Mat convertMat(const Mat &mat);

void segment3DKmeans(Mesh mesh, vector<Mesh> &segs)
{
	mesh.computeVerticesNormals();

	int size = (int)mesh.getVerticesSize();
	Mat pointCloud, normals;
	pointCloud.create(size, 1, CV_64FC3);
	normals.create(size, 1, CV_64FC3);
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		Triangulation::Vertex v = mesh.getVertex(i);
		pointCloud.at<Vec3d>(i, 0) = v.m_xyz / 1000.0;
		normals.at<Vec3d>(i, 0) = v.m_normal;
	}

	Mat clusterData(pointCloud.rows, 6, CV_64FC1);
	Mat roi = clusterData(Rect(0, 0, 3, pointCloud.rows));
	convertMat(pointCloud).copyTo(roi);
	roi = clusterData(Rect(3, 0, 3, pointCloud.rows));
	convertMat(normals).copyTo(roi);
	clusterData.convertTo(clusterData, CV_32FC1);

	vector<int> labels;
	int kNum = 10;
	kmeans(clusterData, kNum, labels, 
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, kNum, 0.1), 
		kNum, KMEANS_PP_CENTERS);

	vector<Vec3b> colors(kNum);
	for (int i = 0; i < kNum; i++)
	{
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}
	vector<Triangulation::Mesh> tmpMeshs(kNum);
	for (int i = 0; i < size; i++)
	{
		Triangulation::Vertex v = mesh.getVertex(i);
		v.m_color = colors[labels[i]];
		tmpMeshs[labels[i]].addVertex(v);
	}
	for (int i = 0; i < kNum; i++)
	{
		segs.push_back(tmpMeshs[i]);
	}
}

int computeLabels( ANNpointArray verticesData, 
	const int &size, vector<int> &labels, map<int, int> &labelMap )
{
	ANNkd_tree* kdtree = new ANNkd_tree(verticesData, size, 3);

	labels.resize(size, -1);			// ��ʼ����ǩΪ-1
	map<int, set<int>> labelsEquals;	// ����ͬһ��ı�ǩ��ӳ��
	int labelCnt = 0;					// �����Ŀ
	for (int i = 0; i < size; i++)
	{
		if (labels[i] >= 0)	// �Ѿ����
		{
			continue;
		}

		ANNidx idxs[SEG_K];
		ANNdist dists[SEG_K];
		int cnt = kdtree->annkFRSearch(verticesData[i], 
			SEG_DISTANCE_RANGE, SEG_K, idxs, dists);	// ����idxs[0] = i;
		cnt = cnt > SEG_K ? SEG_K : cnt;

		// ���ھ����Ƿ����ѱ�ǵ�
		for (int j = 1; j < cnt; j++)
		{
			if (labels[i] >= 0 && labels[idxs[j]] >= 0)
			{
				if (labels[i] != labels[idxs[j]])
				{
					labelsEquals[labels[idxs[j]]].insert(labels[i]);
					labelsEquals[labels[i]].insert(labels[idxs[j]]);
				}
			}
			else if (labels[idxs[j]] >= 0)
			{
				labels[i] = labels[idxs[j]];
			}
			else if (labels[i] >= 0)
			{
				labels[idxs[j]] = labels[i];
			}
		}

		if (labels[i] < 0)
		{
			labels[i] = labelCnt;
			labelCnt++;
#pragma omp parallel for
			for (int j = 1; j < cnt; j++)
			{
				labels[idxs[j]] = labels[i];
			}
		}
	}

	// �ù�������������ϲ�ͬһ��ı�ǩ
	bool* visited = new bool[labelCnt];
	memset(visited, 0, labelCnt * sizeof(bool));
	labelCnt = 0;
	map<int, set<int>>::iterator mapIter = labelsEquals.begin();
	for (; mapIter != labelsEquals.end(); mapIter++)
	{
		if (!visited[mapIter->first])
		{
			visited[mapIter->first] = true;
			labelMap[mapIter->first] = labelCnt;
			labelCnt++;
			queue<int> labelQ;
			labelQ.push(mapIter->first);
			while (!labelQ.empty())
			{
				int label = labelQ.front();
				labelQ.pop();
				set<int> labelSet = labelsEquals[label];
				set<int>::iterator setIter = labelSet.begin();
				for (; setIter != labelSet.end(); setIter++)
				{
					if (!visited[*setIter])
					{
						visited[*setIter] = true;
						labelMap[*setIter] = labelMap[mapIter->first];
						labelQ.push(*setIter);
					}
				}
			}
		}
	}

	delete[] visited;
	return labelCnt;
}


void segment3DRBNN(Mesh mesh, vector<Mesh> &segs)
{
	int size = (int)mesh.getVerticesSize();
	if (size > 0)
	{
		// ����kdtree����Ľṹ����
		ANNpointArray verticesData = annAllocPts(size, 3);
#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			Vec3d v = mesh.getVertex(i).m_xyz / 1000.0;	//ת����λmm->m
			verticesData[i][0] = v[0];
			verticesData[i][1] = v[1];
			verticesData[i][2] = v[2];
		}
		
		vector<int> labels;
		map<int, int> labelMap;
		int labelCnt = computeLabels(verticesData, size, labels, labelMap);

		vector<Vec3b> colors(labelCnt);
		for (int i = 0; i < labelCnt; i++)
		{
			colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
		}
		vector<Triangulation::Mesh> tmpMeshs(labelCnt);
		for (int i = 0; i < size; i++)
		{
			Triangulation::Vertex v = mesh.getVertex(i);
			v.m_color = colors[labelMap[labels[i]]];
			tmpMeshs[labelMap[labels[i]]].addVertex(v);
		}
		for (int i = 0; i < labelCnt; i++)
		{
			segs.push_back(tmpMeshs[i]);
		}
	}
}
