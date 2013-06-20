#include "segmentation.h"

#include "opencv2/opencv.hpp"
#include <queue>
#include "thrust/sort.h"
#include "boost/unordered/unordered_set.hpp"
#include "boost/assign/list_of.hpp"

#include <fstream>

using namespace Triangulation;
using namespace cv;

extern Mat convertMat(const Mat &mat);

void segment3DKmeans(Mesh mesh, vector<Mesh> &segs, int kNum)
{
	//mesh.computeVerticesNormals();

	int size = (int)mesh.getVerticesSize();
	Mat pointCloud, normals;
	pointCloud.create(size, 1, CV_64FC3);
	normals.create(size, 1, CV_64FC3);
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		Triangulation::Vertex v = mesh.getVertex(i);
		pointCloud.at<Vec3d>(i, 0) = v.m_xyz;
		normals.at<Vec3d>(i, 0) = v.m_normal;
	}

	Mat clusterData(pointCloud.rows, 3, CV_64FC1);
	Mat roi = clusterData(Rect(0, 0, 3, pointCloud.rows));
	convertMat(pointCloud).copyTo(roi);
// 	roi = clusterData(Rect(3, 0, 3, pointCloud.rows));
// 	convertMat(normals).copyTo(roi);
	clusterData.convertTo(clusterData, CV_32FC1);

	vector<int> labels;
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
		/*v.m_color = colors[labels[i]];*/
		tmpMeshs[labels[i]].addVertex(v);
	}
	for (int i = 0; i < kNum; i++)
	{
		segs.push_back(tmpMeshs[i]);
	}
}

int search( double dists[], const double &dist, const int &n)
{
	if (dist > dists[n - 1])
	{
		return n;
	}
	if (dist < dists[1])
	{
		return 10;
	}
	int l = 0, h = n - 1;
	while (l <= h)
	{
		int mid = (l + h) / 2;
		if (dist < dists[mid])
		{
			h = mid - 1;
		}
		else if (dist > dists[mid])
		{
			l = mid + 1;
		}
		else
		{
			return mid;
		}
	}
	return l;
}

int computeLabels( const int &k, Mesh &mesh, 
	vector<int> &labels, map<int, int> &labelMap )
{
	if (k < SEG_K * 0.15)
	{
		return -1;
	}
	int size = (int)mesh.getVerticesSize();
	// 构建kdtree所需的结构数组
	ANNpointArray verticesData = annAllocPts(size, 3);
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		Vec3d v = mesh.getVertex(i).m_xyz;	//转换单位mm->m
		verticesData[i][0] = v[0];
		verticesData[i][1] = v[1];
		verticesData[i][2] = v[2];
	}
	ANNkd_tree* kdtree = new ANNkd_tree(verticesData, size, 3);

	std::ofstream os("LPD.txt");

	labels.resize(size, -1);			// 初始化标签为-1
	map<int, set<int>> labelsEquals;	// 属于同一类的标签的映射
	int labelCnt = 0;					// 类别数目
	for (int i = 0; i < size; i++)
	{
		if (labels[i] >= 0 /*&& mesh.getVertex(i).m_isInner*/)	// 已经标记
		{
			continue;
		}

		ANNidxArray idxs = new ANNidx[k];
		ANNdistArray dists = new ANNdist[k];
// 		int cnt = kdtree->annkFRSearch(verticesData[i], 
// 			distanceRange, SEG_K, idxs, dists);	// 其中idxs[0] = i;
		kdtree->annkSearch(verticesData[i], k, idxs, dists);
//		cnt = cnt > SEG_K ? SEG_K : cnt;

		double LPD = k / (CV_PI * dists[k - 1]);
		os << LPD << endl;
		int cnt = k;

// 		if (LPD > 100000)
// 		{
// 			cnt = int(k * 0.15);
// 		}
// 		else if (LPD > 10000)
// 		{
// 			cnt = int(k * 0.2);
// 		}
// 		else if (LPD > 1000)
// 		{
// 			cnt = int(k * 2 / log(LPD));
// 		}
// 		else
// 		{
// 			cnt = int(k * (0.75 - 0.00045 * LPD));
// 		}
		if (LPD > 10000)
		{
			cnt = k * 0.2;
		}
		else if (LPD > 1000)
		{
			double dist = dists[k - 1] / exp(log(LPD));
			cnt = search(dists, dist, k);
		}
		else
		{
			cnt = k * 0.75;
		}

//		Vec3d normal = mesh.getVertex(i).m_normal;
		// 看邻居中是否有已标记的
		for (int j = 1; j < cnt; j++)
		{
// 			Vec3d neighNormal = mesh.getVertex(j).m_normal;
// 			if (!mesh.getVertex(j).m_isInner && normal.ddot(neighNormal) < 0.9)
// 			{
// 				continue;
// 			}
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
			labelsEquals[labelCnt].insert(labelCnt);
			labelCnt++;
#pragma omp parallel for
			for (int j = 1; j < cnt; j++)
			{
				labels[idxs[j]] = labels[i];
			}
		}
	}
	os.close();

	// 用广度优先搜索来合并同一类的标签
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


void segment3DRBNN(const int &k, Mesh &mesh, vector<Mesh> &segs)
{
	int size = (int)mesh.getVerticesSize();
	if (size > 0)
	{	
//		mesh.computeVerticesNormals();

		vector<int> labels;
		map<int, int> labelMap;
		int labelCnt = computeLabels(k, mesh, labels, labelMap);
		if (labelCnt < 0)
		{
			segs.push_back(mesh);
			return;
		}

		vector<Vec3b> colors(labelCnt);
		for (int i = 0; i < labelCnt; i++)
		{
			colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
		}
		vector<Triangulation::Mesh> tmpMeshs(labelCnt);
		for (int i = 0; i < size; i++)
		{
			Triangulation::Vertex v = mesh.getVertex(i);
			/*v.m_color = colors[labelMap[labels[i]]];*/
			tmpMeshs[labelMap[labels[i]]].addVertex(v);
		}
		for (int i = 0; i < labelCnt; i++)
		{
 			if (tmpMeshs[i].getVerticesSize() > mesh.getVerticesSize() * 0.5)
			{
				segment3DKmeans(tmpMeshs[i], segs, 4);
			}
			else
			{
				segs.push_back(tmpMeshs[i]);
			}
		}
 	}
}

typedef boost::unordered::unordered_set<int> b_unordered_set_int;
#define b_list_of_int boost::assign::list_of<int>

void segment3DSC( Mesh &mesh, vector<Mesh> &segs )
{
	double residualPercent = 0.95;
	double angleTh = cos(30.0 * CV_PI / 180.0);
	mesh.computeVerticesNormals();

	int size = (int)mesh.getVerticesSize();

	int *indices = new int[size];
	double *residuals = new double[size];
	bool *visited = new bool[size];
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		residuals[i] = mesh.getVertex(i).m_residual;
		indices[i] = i;
		visited[i] = false;
	}

	thrust::sort_by_key(residuals, residuals + size, indices);

	b_unordered_set_int indexSet = b_list_of_int().range(indices, indices + size);

	while (!indexSet.empty())
	{
		Vec3b color(rand() % 256, rand() % 256, rand() % 256);
		Mesh m;
		set<int> seedIndices;

		int minIndex = *indexSet.begin();
		visited[minIndex] = true;
		seedIndices.insert(minIndex);

		Vertex v = mesh.getVertex(minIndex);
		v.m_color = color;
		m.addVertex(v);

		indexSet.erase(indexSet.begin());

		while (!seedIndices.empty())
		{
			int index = *seedIndices.begin();
			seedIndices.erase(seedIndices.begin());
			v = mesh.getVertex(index);
			for (int i = 0; i < v.m_neighbors[0]; i++)
			{
				int index2 = v.m_neighbors[i + 1];
				if (!visited[index2])
				{
					Vertex v2 = mesh.getVertex(index2);
					if (abs(v.m_normal.ddot(v2.m_normal)) > angleTh)
					{
						v2.m_color = color;
						m.addVertex(v2);
						visited[index2] = true;
						indexSet.erase(index2);
						int n = size - 1;
						while (n >= 0 && visited[indices[n]])
						{
							n--;
						}
						if (n >= 0)
						{
							if (v2.m_residual < residuals[n] * residualPercent)
							{
								seedIndices.insert(index2);
							}
						}
					}
				}
			}
		}
		if (m.getVerticesSize() > 0)
		{
			segs.push_back(m);
		}
	}
}