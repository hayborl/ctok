#include "triangulation.h"

#include "ANN/ANN.h"
#include "common.h"

#include <fstream>

using namespace Triangulation;

Vertex& Vertex::operator=( const Vertex &v )
{
	m_xyz = v.m_xyz;
	m_color = v.m_color;
	m_index = v.m_index;
	return *this;
}

bool Vertex::operator==( const Vertex &v ) const
{
	return m_xyz == v.m_xyz;
}

bool Vertex::operator<( const Vertex &v ) const
{
	if (m_xyz[0] == v.m_xyz[0])
	{
		if (m_xyz[1] == v.m_xyz[1])
		{
			return m_xyz[2] < v.m_xyz[2];
		}
		return m_xyz[1] < v.m_xyz[1];
	}
	return m_xyz[0] < v.m_xyz[0];
}

double Vertex::distance2( const Vertex &v )
{
	Vec3d d = m_xyz - v.m_xyz;
	return d.dot(d);
}

double Vertex::cross( Vertex a, Vertex b, Vertex c )
{
	Vec3d ab = b.m_xyz - a.m_xyz;
	Vec3d ac = c.m_xyz - a.m_xyz;
	return ab[0] * ac[1] - ab[1] * ac[0];
}

Triangle::Triangle( Vertex v0, Vertex v1, Vertex v2 )
{
	m_vertices[0] = v0;
	m_vertices[1] = v1;
	m_vertices[2] = v2;

// 	Vertex tmp;
// 	for (int i = 0; i < Vertex_Size - 1; i++)
// 	{
// 		for (int j = i + 1; j < Vertex_Size; j++)
// 		{
// 			if (m_vertices[j] < m_vertices[i])
// 			{
// 				tmp = m_vertices[j];
// 				m_vertices[j] = m_vertices[i];
// 				m_vertices[i] = tmp;
// 			}
// 		}
// 	}
}

Vec3d Triangle::getNormal()
{
	Vec3d edge0 = m_vertices[1].m_xyz - m_vertices[0].m_xyz;
	Vec3d edge1 = m_vertices[2].m_xyz - m_vertices[0].m_xyz;

	Vec3d normal = edge0.cross(edge1);
	normalize(normal);

	return normal;
}

void Triangle::turnBack()
{
	Vertex tmp = m_vertices[0];
	m_vertices[0] = m_vertices[2];
	m_vertices[2] = tmp;
}

int Triangle::inTriangle( Vertex v )
{
	double abc = fabs(Vertex::cross(m_vertices[0], 
		m_vertices[1], m_vertices[2]));
	double abv = fabs(Vertex::cross(m_vertices[0], m_vertices[1], v));
	double avc = fabs(Vertex::cross(m_vertices[0], v, m_vertices[2]));
	double vbc = fabs(Vertex::cross(v, m_vertices[1], m_vertices[2]));

	double diff = abc - abv - avc - vbc;

	int r = 1;
	if (fabs(diff) < DOUBLE_ERROR)
	{
		if (abv > DOUBLE_ERROR && avc > DOUBLE_ERROR && vbc > DOUBLE_ERROR)			// in
		{
			r = 0;
		}
		else if (abv < DOUBLE_ERROR && avc > DOUBLE_ERROR && vbc > DOUBLE_ERROR)	// on ab
		{
			r = -1;
		}
		else if (abv > DOUBLE_ERROR && vbc < DOUBLE_ERROR && avc > DOUBLE_ERROR)	// on bc
		{
			r = -2;
		}
		else if (abv > DOUBLE_ERROR && vbc > DOUBLE_ERROR && avc < DOUBLE_ERROR)	// on ac
		{
			r = -3;
		}
	}

	return r;
}

bool Triangle::isVertex( const int index )
{
	if (m_vertices[0].m_index == index ||
		m_vertices[1].m_index == index ||
		m_vertices[2].m_index == index)
	{
		return true;
	}
	return false;
}

bool Triangle::angleCriterion( const double &minCosAngle, 
	const double &maxCosAngle )
{
	Vec3d ab = m_vertices[1].m_xyz - m_vertices[0].m_xyz;
	Vec3d bc = m_vertices[2].m_xyz - m_vertices[1].m_xyz;
	Vec3d ac = m_vertices[2].m_xyz - m_vertices[0].m_xyz;

	double lenAB = ab.dot(ab);
	double lenBC = bc.dot(bc);
	double lenAC = ac.dot(ac);

	double maxLen = lenAB, minLen = lenAB;
	double lenMaxE0 = lenBC, lenMaxE1 = lenAC;
	double lenMinE0 = lenBC, lenMinE1 = lenAC;
	Vec3d maxE0 = bc, maxE1 = ac, minE0 = bc, minE1 = ac;

	bool maxFlag = true, minFlag = true;
	if (maxCosAngle > COS180 && maxCosAngle <  COS60)
	{
		if (maxLen < lenBC)
		{
			maxLen = lenBC;
			lenMaxE0 = lenAB;
			lenMaxE1 = lenAC;
			maxE0 = ab;
			maxE1 = ac;
		}
		if (maxLen < lenAC)
		{
			lenMaxE0 = lenAB;
			lenMaxE1 = lenBC;
			maxE0 = -ab;
			maxE1 = bc;
		}
		maxFlag = maxE0.dot(maxE1) > sqrt(lenMaxE0) 
			* sqrt(lenMaxE1) * maxCosAngle;
	}
	if (minCosAngle < COS0 && minCosAngle > COS60)
	{
		if (minLen > lenBC)
		{
			minLen = lenBC;
			lenMinE0 = lenAB;
			lenMinE1 = lenAC;
			minE0 = ab;
			minE1 = ac;
		}
		if (minLen > lenAC)
		{
			lenMinE0 = lenAB;
			lenMinE1 = lenBC;
			minE0 = -ab;
			minE1 = bc;
		}
		minFlag = minE0.dot(minE1) < sqrt(lenMinE0) 
			* sqrt(lenMinE1) * minCosAngle;
	}
	
	return minFlag && maxFlag;
}

bool Triangle::operator==( const Triangle &t ) const
{
	int cnt = 0;
	for (int i = 0; i < Vertex_Size; i++)
	{
		for (int j = 0; j < Vertex_Size; j++)
		{
			if (m_vertices[i] == t.m_vertices[j])
			{
				cnt++;
				break;
			}
		}
	}
	return cnt == Vertex_Size;
}

size_t Triangulation::hash_value( const Triangle &t )
{
	size_t seed;
	for (int i = 0; i < Triangle::Vertex_Size; i++)
	{
		Vec3d v = t.m_vertices[i].m_xyz;
		boost::hash_combine(seed, boost::hash_value(v[0]));
		boost::hash_combine(seed, boost::hash_value(v[1]));
		boost::hash_combine(seed, boost::hash_value(v[2]));
	}
	return seed;
}

Delaunay::Delaunay( const Mat &pts, const vector<Vec3b> &colors,
	double minAngle, double maxAngle)
	: m_minAngle(minAngle), m_maxAngle(maxAngle), m_curIndex(0)
{
	addVertices(pts, colors);
}

Delaunay::Delaunay( const Mat &pts, const Mat &colors,
	double minAngle, double maxAngle)
	: m_minAngle(minAngle), m_maxAngle(maxAngle), m_curIndex(0)
{
	addVertices(pts, colors);
}

void Delaunay::computeDelaunay()
{
	if (m_vertices.size() == 0)
	{
		return;
	}
	m_preSize = m_curIndex;

	// ����kdtree
	ANNpointArray verticesData = annAllocPts((int)m_vertices.size(), 3);
	for (int i = 0; i < m_vertices.size(); i++)
	{
		verticesData[i][0] = m_vertices[i].m_xyz[0];
		verticesData[i][1] = m_vertices[i].m_xyz[1];
		verticesData[i][2] = m_vertices[i].m_xyz[2];
	}
	ANNkd_tree* kdtree = new ANNkd_tree(verticesData, 
		(int)m_vertices.size(), 3);

	TriangleSet triSet;
	// ���α���ÿ���㣬Ѱ������ڣ��������ǻ�
	for (; m_curIndex < m_vertices.size(); m_curIndex++)
	{
		Vertex v = m_vertices[m_curIndex];
		ANNidx idxs[k];
		ANNdist dists[k];
		int cnt = kdtree->annkFRSearch(verticesData[m_curIndex], 
			Distance_Range, k, idxs, dists);	// ����idxs[0] = i;
		if (cnt < 4)	// �����С��4�������ܹ���������
		{
			m_beginIndicesTri.push_back((int)triSet.size());
			continue;
		}
		cnt = cnt > k ? k : cnt;

		Vec3d normal = computeNormal(verticesData, idxs, cnt);	// ���㷨����
		int id = 2;
		// �жϷ������ĸ���Ϊ0��z->y->x
		if (normal[2] != 0)		// z
		{
			id = 2;
		}
		else if (normal[1] != 0)// y
		{
			id = 1;
		}
		else if (normal[0] != 0)// x
		{
			id = 0;
		}
		else	// ������Ϊ(0, 0, 0)��
		{
			m_beginIndicesTri.push_back((int)triSet.size());
			continue;
		}

		double minDistance = -1;
		int j;
		for (j = 0; j < cnt; j++)
		{
			if (dists[j] > 0)
			{
				minDistance = dists[j];
				break;
			}
		}
		VertexVector vVector, tmpvVector;
		// ������ڵ�Ͷ�䵽�õ����ƽ����
		for (; j < cnt; j++)
		{
			Vertex tmpv = m_vertices[idxs[j]];
			if (dists[j] < u * minDistance ||		// ȥ���ǳ��ӽ��ĵ�
				(tmpv.m_index < v.m_index && tmpv.m_index >= m_preSize))	// ȥ���ѱ������ĵ�
			{
				continue;
			}
			
			Vec3d vv = tmpv.m_xyz - v.m_xyz;
			double dist2 = dists[j] * 0.75f;	// sqrt
			double alpha = vv.dot(normal);
			alpha = alpha * alpha;
			if (alpha > dist2)		// ȥ���뷨�����н�С��30�Ȼ����150�ȵĵ�
			{
				continue;
			}
			Vec3d proj = tmpv.m_xyz - alpha * normal;		// Ͷ�䵽��ƽ��
			tmpvVector.push_back(Vertex(proj, idxs[j]));
		}
		if (tmpvVector.size() < 3)	// ����3�����ܹ���������
		{
			m_beginIndicesTri.push_back((int)triSet.size());
			continue;
		}

		// ����ƽ��ת��Ϊx-yƽ����������μ���
		vVector.push_back(Vertex(Vec3d(0, 0, 0), m_curIndex));	// ԭ��
		Vec3d vx = tmpvVector[0].m_xyz - v.m_xyz;		// x��
		vx = normalize(vx);
		for (j = 0; j < tmpvVector.size(); j++)
		{
			Vec3d vv = tmpvVector[j].m_xyz - v.m_xyz;
			double x = vv.dot(vx);
			double y = vx.cross(vv)[id] / normal[id];
			Vec3d proj(x, y, 0);
			vVector.push_back(Vertex(proj, tmpvVector[j].m_index));
		}

		TriangleVector tVector;
		computeDelaunay(vVector, tVector);
// 		cout << vVector.size() << " " << tVector.size() << endl; 
// 		drawTrianglesOnPlane(tVector);
		for (j = 0; j < tVector.size(); j++)
		{
			Triangle t = tVector[j];
			t.m_vertices[0] = m_vertices[t.m_vertices[0].m_index];
			t.m_vertices[1] = m_vertices[t.m_vertices[1].m_index];
			t.m_vertices[2] = m_vertices[t.m_vertices[2].m_index];
			triSet.insert(t);
		}
		m_beginIndicesTri.push_back((int)triSet.size());
	}

	for (TriangleSet::iterator iter = triSet.begin(); 
		iter != triSet.end(); iter++)
	{
		m_triangles.push_back(*iter);
	}

	annDeallocPts(verticesData);
}

void Delaunay::computeDelaunay( const VertexVector &verSet, 
	TriangleVector &triSet )
{
	TriangleVector tmpTriSet;
	addBounding(verSet, tmpTriSet);
	for (int i = 0; i < verSet.size(); i++)
	{
/*		cout << verSet[i].m_xyz << endl;*/
		insertVertex(tmpTriSet, verSet[i]);
/*		drawTrianglesOnPlane(triSet);*/
	}
	removeBounding(tmpTriSet, triSet, verSet[0].m_index);
}

void Delaunay::saveTriangles( char* file )
{
	saveTriangles(m_triangles, file);
}

void Delaunay::saveTriangles( const TriangleVector &triSet, char* file )
{
	ofstream ofs(file);
	if (ofs)
	{
		ofs << triSet.size() << endl;
		for (int i = 0; i < triSet.size(); i++)
		{
			Triangle t = triSet[i];
			for (int j = 0; j < Triangle::Vertex_Size; j++)
			{
				ofs << t.m_vertices[j].m_xyz;
			}
			ofs << endl;
		}
	}
}

void Delaunay::addVertices( InputArray _pts, InputArray _colors )
{
	if (_pts.total() == 0 || _colors.total() == 0)
	{
		m_beginIndicesVer.push_back((int)m_vertices.size());
		return;
	}

	Mat pts = _pts.getMat();
	Mat colors = _colors.getMat();
	assert(pts.type() == CV_64FC3 && colors.type() == CV_8UC3);
	assert(pts.total() == colors.total());
	int total = (int)pts.total();
	
	assert(pts.isContinuous() && colors.isContinuous());
	Vec3d* ptsPtr = (Vec3d*)pts.data;
	Vec3b* colorsPtr = (Vec3b*)colors.data;

	int cnt = (int)m_vertices.size();
	m_beginIndicesVer.push_back(cnt);
	for (int i = 0; i < total; i++)
	{
		Vec3d xyz = ptsPtr[i] * 1000.0;
		Vec3b color = colorsPtr[i];

		m_vertices.push_back(Vertex(xyz, cnt, color));
		cnt++;
	}
	cout << "Points total number:" << m_vertices.size() << endl;
}

void Delaunay::getVertices( const int &times, Mat &out )
{
	if (times < m_beginIndicesVer.size())
	{
		int beginIndex = m_beginIndicesVer[times];
		int size;
		if (times == m_beginIndicesVer.size() - 1)
		{
			size = (int)m_vertices.size() - beginIndex;
		}
		else
		{
			size = m_beginIndicesVer[times + 1] - beginIndex;
		}

		if (size > 0)
		{
			out.create(size, 1, CV_64FC3);
#pragma omp parallel for
			for (int i = 0; i < size; i++)
			{
				out.at<Vec3d>(i, 0) = m_vertices[beginIndex + i].m_xyz;
			}
		}
	}
}

void Delaunay::updateVertices( const int &times, const Mat &in )
{
	if (times < m_beginIndicesVer.size())
	{
		int beginIndex = m_beginIndicesVer[times];
		int size;
		if (times == m_beginIndicesVer.size() - 1)
		{
			size = (int)m_vertices.size() - beginIndex;
		}
		else
		{
			size = m_beginIndicesVer[times + 1] - beginIndex;
		}

		if (size > 0 && size == in.rows)
		{
			for (int i = 0; i < size; i++)
			{
				m_vertices[beginIndex + i].m_xyz = in.at<Vec3d>(i, 0);
			}
			if (m_curIndex > beginIndex)
			{
				m_curIndex = beginIndex;
				m_triangles.erase(m_triangles.begin() 
					+ m_beginIndicesTri[times], m_triangles.end());
			}
		}
	}
}

void Delaunay::addBounding( const VertexVector &verSet, 
	TriangleVector &triSet )
{
	double max_x = -100000, max_y = -100000;
	double min_x = 100000, min_y = 100000;
	for (int i = 0; i < verSet.size(); i++)
	{
		Vec3d v = verSet[i].m_xyz;
		max_x = max_x < v[0] ? v[0] : max_x;
		min_x = min_x > v[0] ? v[0] : min_x;
		max_y = max_y < v[1] ? v[1] : max_y;
		min_y = min_y > v[1] ? v[1] : min_y;
	}

	// ����������������������Ϊ�߽�
	double dx = max_x - min_x;
	double dy = max_y - min_y;
	double mid_x = (min_x + max_x) / 2;
	double mid_y = (min_y + max_y) / 2;

	// Ϊ��ȥ���߽緽�㽲�߽���������Ϊ����
	Vertex v0(Vec3d(mid_x, max_y + dy, 0.0f), -1);
	Vertex v1(Vec3d(mid_x - dx, min_y, 0.0f), -2);
	Vertex v2(Vec3d(mid_x + dx, min_y, 0.0f), -3);

	triSet.push_back(Triangle(v0, v1, v2));
}

void Delaunay::removeBounding( TriangleVector inSet, 
	TriangleVector &outSet, const int &index )
{
	for (TriangleVector::iterator iter = inSet.begin(); 
		iter != inSet.end(); iter++)
	{
		if (iter->m_vertices[0].m_index >= 0 && 
			iter->m_vertices[1].m_index >= 0 && 
			iter->m_vertices[2].m_index >= 0 && 
			iter->isVertex(index) && 
			iter->angleCriterion(m_minAngle, m_maxAngle))
		{
			outSet.push_back(*iter);
		}
	}
}

void Delaunay::insertVertex( TriangleVector &triSet, const Vertex &v )
{
	TriangleVector tmp;
	for (TriangleVector::iterator iter = triSet.begin(); 
		iter != triSet.end();)
	{
		int r = iter->inTriangle(v);	// �жϵ��Ƿ�����������
// 		cout << iter->m_vertices[0].m_xyz 
// 			<< iter->m_vertices[1].m_xyz 
// 			<< iter->m_vertices[2].m_xyz << endl;
		switch (r)
		{
		case 0:		// in
			{
				Triangle t0(iter->m_vertices[0], iter->m_vertices[1], v);
				Triangle t1(iter->m_vertices[1], iter->m_vertices[2], v);
				Triangle t2(iter->m_vertices[2], iter->m_vertices[0], v);
				tmp.push_back(t0);
				tmp.push_back(t1);
				tmp.push_back(t2);
				iter = triSet.erase(iter);
			}
			break;
		case -1:	// on v0v1
			{
				Triangle t0(iter->m_vertices[1], iter->m_vertices[2], v);
				Triangle t1(iter->m_vertices[2], iter->m_vertices[0], v);
				tmp.push_back(t0);
				tmp.push_back(t1);
				iter = triSet.erase(iter);
			}
			break;
		case -2:	// on v1v2
			{
				Triangle t0(iter->m_vertices[0], iter->m_vertices[1], v);
				Triangle t1(iter->m_vertices[2], iter->m_vertices[0], v);
				tmp.push_back(t0);
				tmp.push_back(t1);
				iter = triSet.erase(iter);
			}
			break;
		case -3:	// on v2v0
			{
				Triangle t0(iter->m_vertices[0], iter->m_vertices[1], v);
				Triangle t1(iter->m_vertices[1], iter->m_vertices[2], v);
				tmp.push_back(t0);
				tmp.push_back(t1);
				iter = triSet.erase(iter);
			}
			break;
		default:
			iter++;
			break;
		}
		if (r <= 0)
		{
			break;
		}
	}

	for (int i = 0; i < tmp.size(); i++)
	{
		if (!flipTest(triSet, tmp[i]))	// �Ż�delaunay����
		{
			triSet.push_back(tmp[i]);
		}
/*		drawTrianglesOnPlane(triSet);*/
	}
}

bool Delaunay::flipTest( TriangleVector &triSet, Triangle t )
{
	bool flipped = false;

	Vertex a = t.m_vertices[0];
	Vertex b = t.m_vertices[1];
	Vertex c = t.m_vertices[2]; 

	TriangleVector tSet;
	for (TriangleVector::iterator iter = triSet.begin(); 
		iter != triSet.end();)
	{
		Vertex d;
		d.m_index = -100;

		// Ѱ��ӵ����ͬ��ab��������
		int satisfy[3] = {0, 0, 0};
		for (int j = 0, k = 1; j < Triangle::Vertex_Size; j++, k *= 2)
		{
			if (iter->m_vertices[j].m_index == a.m_index || 
				iter->m_vertices[j].m_index == b.m_index)
			{
				satisfy[j] = k;
			}
		}
		switch (satisfy[0] | satisfy[1] | satisfy[2])
		{
		case 3:	// v2
			if (Vertex::cross(a, c, iter->m_vertices[2]) != 0 &&
				Vertex::cross(b, c, iter->m_vertices[2]) != 0)
			{
				d = iter->m_vertices[2];
			}
			break;
		case 5: // v1
			if (Vertex::cross(a, c, iter->m_vertices[1]) != 0 &&
				Vertex::cross(b, c, iter->m_vertices[1]) != 0)
			{
				d = iter->m_vertices[1];
			}
			break;
		case 6: // v2
			if (Vertex::cross(a, c, iter->m_vertices[0]) != 0 &&
				Vertex::cross(b, c, iter->m_vertices[0]) != 0)
			{
				d = iter->m_vertices[0];
			}
			break;
		default:
			break;
		}

		if (d.m_index != -100)
		{
			if (inCircle(a, b, c, d))	// �ж�d�Ƿ���������abc�����Բ��
			{
				flipped = true;
				Triangle t0(a, d, c);
				Triangle t1(d, b, c);
				tSet.push_back(t0);
				tSet.push_back(t1);
				iter = triSet.erase(iter);
				break;
			}
			else
			{
				iter++;
			}
		}
		else
		{
			iter++;
		}
	}

	for (int i = 0; i < tSet.size(); i++)
	{
		if (!flipTest(triSet, tSet[i]))
		{
			triSet.push_back(tSet[i]);
		}
	}
	return flipped;
}

bool Delaunay::inCircle( Vertex a, Vertex b, Vertex c, Vertex p )
{
	Vec3d cb = b.m_xyz - c.m_xyz;
	Vec3d ca = a.m_xyz - c.m_xyz;
	Vec3d pb = b.m_xyz - p.m_xyz;
	Vec3d pa = a.m_xyz - p.m_xyz;

	double cross_cbca = fabs(cb[0] * ca[1] - cb[1] * ca[0]);
	double cross_pbpa = fabs(pb[0] * pa[1] - pb[1] * pa[0]);
	double dot_cbca = cb.dot(ca);
	double dot_pbpa = pb.dot(pa);
	if (cross_cbca * dot_pbpa + cross_pbpa * dot_cbca < 0)
	{
		return true;
	}
	return false;
}

void Delaunay::drawTrianglesOnPlane( const TriangleVector &triSet )
{
	int width = 1024, height = 768;
	Mat triangleImg(height, width, CV_8UC3, Scalar::all(0));
	if (triSet.size() == 0)
	{
		imshow("Triangle", triangleImg);
		waitKey();
		return;
	}

	double max_x = -100000, max_y = -100000;
	double min_x = 100000, min_y = 100000;
	for (int i = 0; i < triSet.size(); i++)
	{
		for (int j = 0; j < Triangle::Vertex_Size; j++)
		{
			Vec3d v = triSet[i].m_vertices[j].m_xyz;
			max_x = max_x < v[0] ? v[0] : max_x;
			min_x = min_x > v[0] ? v[0] : min_x;
			max_y = max_y < v[1] ? v[1] : max_y;
			min_y = min_y > v[1] ? v[1] : min_y;
		}
	}

	for (int i = 0; i < triSet.size(); i++)
	{
		Point ps[3];
		for (int j = 0; j < Triangle::Vertex_Size; j++)
		{
			double tmpx = triSet[i].m_vertices[j].m_xyz[0];
			double tmpy = triSet[i].m_vertices[j].m_xyz[1];

			double x = (tmpx - min_x) * 800 / (max_x - min_x) + 100;
			double y = (max_y - tmpy) * 600 / (max_y - min_y) + 100;

			ps[j].x = (int)x;
			ps[j].y = (int)y;
		}
		line(triangleImg, ps[0], ps[1], Scalar::all(255));
		line(triangleImg, ps[1], ps[2], Scalar::all(255));
		line(triangleImg, ps[2], ps[0], Scalar::all(255));
	}

	for (int i = 0; i < triSet.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			double tmpx = triSet[i].m_vertices[j].m_xyz[0];
			double tmpy = triSet[i].m_vertices[j].m_xyz[1];

			double fx = (tmpx - min_x) * 800 / (max_x - min_x) + 100;
			double fy = (max_y - tmpy) * 600 / (max_y - min_y) + 100;

			int x = (int)fx;
			int y = (int)fy;

			if (triSet[i].m_vertices[j].m_index < 0)
			{
				for (int m = -1; m <= 1; m++)
				{
					for (int n = -1; n <= 1; n++)
					{
						triangleImg.at<Vec3b>(y + m, x + n) = Vec3b(0, 255, 0);
					}
				}
			}
			else
			{
				for (int m = -1; m <= 1; m++)
				{
					for (int n = -1; n <= 1; n++)
					{
						triangleImg.at<Vec3b>(y + m, x + n) = Vec3b(255, 0, 255);
					}
				}
			}
		}
	}

	imshow("Triangle", triangleImg);
	waitKey();
}