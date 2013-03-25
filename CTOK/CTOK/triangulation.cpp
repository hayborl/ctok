#include "triangulation.h"

#include "ANN/ANN.h"
#include "common.h"

#include <fstream>

using namespace Triangulation;

Vertex& Vertex::operator=( const Vertex& v )
{
	m_xyz = v.m_xyz;
	m_color = v.m_color;
	m_index = v.m_index;
	return *this;
}

bool Vertex::operator==( const Vertex& v ) const
{
	return m_xyz == v.m_xyz;
}

float Vertex::distance2( const Vertex& v )
{
	Vec3f d = m_xyz - v.m_xyz;
	return d.dot(d);
}

float Vertex::cross( Vertex a, Vertex b, Vertex c )
{
	Vec3f ab = b.m_xyz - a.m_xyz;
	Vec3f ac = c.m_xyz - a.m_xyz;
	return ab[0] * ac[1] - ab[1] * ac[0];
}

Triangle::Triangle( Vertex v0, Vertex v1, Vertex v2 )
{
	m_vertices[0] = v0;
	m_vertices[1] = v1;
	m_vertices[2] = v2;
}

Vec3f Triangle::getNormal()
{
	Vec3f edge0 = m_vertices[1].m_xyz - m_vertices[0].m_xyz;
	Vec3f edge1 = m_vertices[2].m_xyz - m_vertices[0].m_xyz;

	Vec3f normal = edge0.cross(edge1);
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
	float abc = fabs(Vertex::cross(m_vertices[0], 
		m_vertices[1], m_vertices[2]));
	float abv = fabs(Vertex::cross(m_vertices[0], m_vertices[1], v));
	float avc = fabs(Vertex::cross(m_vertices[0], v, m_vertices[2]));
	float vbc = fabs(Vertex::cross(v, m_vertices[1], m_vertices[2]));

	float diff = abc - abv - avc - vbc;

	int r = 1;
	if (fabs(diff) < FLOAT_ERROR)
	{
		if (abv > FLOAT_ERROR && avc > FLOAT_ERROR && vbc > FLOAT_ERROR)		// in
		{
			r = 0;
		}
		else if (abv < FLOAT_ERROR && avc > FLOAT_ERROR && vbc > FLOAT_ERROR)	// on ab
		{
			r = -1;
		}
		else if (abv > FLOAT_ERROR && vbc < FLOAT_ERROR && avc > FLOAT_ERROR)	// on bc
		{
			r = -2;
		}
		else if (abv > FLOAT_ERROR && vbc > FLOAT_ERROR && avc < FLOAT_ERROR)	// on ac
		{
			r = -3;
		}
	}
/*	cout << diff << " " << r << endl;*/

	return r;
}

bool Triangle::operator==( const Triangle& t ) const
{
	int cnt = 0;
	for (int i = 0; i < Vertex_Size; i++)
	{
		for (int j = 0; j < Vertex_Size; j++)
		{
			if (m_vertices[i] == t.m_vertices[j])
			{
				cnt++;
			}
		}
	}
	return cnt == Vertex_Size;
}

size_t Triangulation::hash_value( const Triangle& t )
{
	size_t seed;
	for (int i = 0; i < Triangle::Vertex_Size; i++)
	{
		Vec3f v = t.m_vertices[i].m_xyz;
		boost::hash_combine(seed, boost::hash_value(v[0]));
		boost::hash_combine(seed, boost::hash_value(v[1]));
		boost::hash_combine(seed, boost::hash_value(v[2]));
	}
	return seed;
}

Delaunay::Delaunay( const Mat& pts, const vector<Vec3b>& colors )
{
	addVertices(pts, colors);
}

Delaunay::Delaunay( const Mat& pts, const Mat& colors )
{
	addVertices(pts, colors);
}

void Delaunay::computeDelaunay()
{
	assert(m_vertices.size() > 0);

	m_triangles.clear();

	// 构建kdtree
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
	// 依次遍历每个点，寻找最近邻，进行三角化
	for (int i = 0; i < m_vertices.size(); i++)
	{
		Vertex v = m_vertices[i];
		ANNidx idxs[k];
		ANNdist dists[k];
		int cnt = kdtree->annkFRSearch(verticesData[i], 
			Distance_Range, k, idxs, dists);	// 其中idxs[0] = i;
		if (cnt < 4)	// 最近邻小于4个，不能构成三角形
		{
			continue;
		}
		cnt = cnt > k ? k : cnt;

		Vec3f normal = computeNormal(verticesData, idxs, cnt);	// 计算法向量
		int id = 2;
		// 判断法向量哪个不为0，z->y->x
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
		else	// 法向量为(0, 0, 0)，
		{
			continue;
		}

		VertexVector vVector, tmpvVector;
		// 将最近邻点投射到该点的切平面上
		for (int j = 1; j < cnt; j++)
		{
			if (dists[j] < CLOSE_DISTANCE)	// 去除非常接近的点
			{
				continue;
			}
			Vertex tmpv = m_vertices[idxs[j]];
			Vec3f vv = tmpv.m_xyz - v.m_xyz;
			double dist = sqrt(dists[j]) * SQRT_3 / 2;
			double alpha = vv.dot(normal);

			if (alpha > dist || alpha < -dist)	// 去除去方向量夹角小于30度或大于150度的点
			{
				continue;
			}

			if (tmpv.m_index >= v.m_index)	// 去除已遍历过的点
			{
				Vec3f proj = tmpv.m_xyz - alpha * normal;		// 投射到切平面
				tmpvVector.push_back(Vertex(proj, idxs[j]));
			}
		}
		if (tmpvVector.size() < 3)	// 少于3个不能构成三角形
		{
			continue;
		}

		// 将切平面转换为x-y平面进行三角形计算
		vVector.push_back(Vertex(Vec3f(0, 0, 0), i));	// 原点
		Vec3f vx = tmpvVector[0].m_xyz - v.m_xyz;		// x轴
		vx = normalize(vx);
		for (int j = 0; j < tmpvVector.size(); j++)
		{
			Vec3f vv = tmpvVector[j].m_xyz - v.m_xyz;
			float x = vv.dot(vx);
			float y = vx.cross(vv)[id] / normal[id];
			Vec3f proj(x, y, 0);
			vVector.push_back(Vertex(proj, tmpvVector[j].m_index));
		}

		TriangleVector tVector;
		computeDelaunay(vVector, tVector);
// 		cout << vVector.size() << " " << tVector.size() << endl; 
// 		drawTrianglesOnPlane(tVector);
		for (int j = 0; j < tVector.size(); j++)
		{
			Triangle t = tVector[j];
			t.m_vertices[0] = m_vertices[t.m_vertices[0].m_index];
			t.m_vertices[1] = m_vertices[t.m_vertices[1].m_index];
			t.m_vertices[2] = m_vertices[t.m_vertices[2].m_index];
			triSet.insert(t);
		}
	}

	for (TriangleSet::iterator iter = triSet.begin(); 
		iter != triSet.end(); iter++)
	{
		m_triangles.push_back(*iter);
	}

	annDeallocPts(verticesData);
}

void Delaunay::computeDelaunay( const VertexVector& verSet, 
	TriangleVector& triSet )
{
	addBounding(verSet, triSet);
	for (int i = 0; i < verSet.size(); i++)
	{
/*		cout << verSet[i].m_xyz << endl;*/
		insertVertex(triSet, verSet[i]);
/*		drawTrianglesOnPlane(triSet);*/
	}
	removeBounding(triSet);
}

void Delaunay::saveTriangles( char* file )
{
	saveTriangles(m_triangles, file);
}

void Delaunay::saveTriangles( const TriangleVector& triSet, char* file )
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

void Delaunay::addVertices( const Mat& pts, const vector<Vec3b>& colors )
{
	assert(pts.rows == colors.size());
	m_vertices.clear();

	int cnt = 0;
	for (int i = 0; i < pts.rows; i++)
	{
// 		if ((double)rand() / (double)RAND_MAX > 0.5)
// 		{
// 			continue;
// 		}
		Vec3f xyz = pts.at<Vec3f>(i, 0);
		Vec3b color = colors[i];

		m_vertices.push_back(Vertex(xyz, cnt, color));
		cnt++;
	}
	cout << m_vertices.size() << endl;
}

void Delaunay::addVertices( const Mat& pts, const Mat& colors )
{
	assert(pts.rows == colors.rows);
	m_vertices.clear();

	int cnt = 0;
	for (int i = 0; i < pts.rows; i++)
	{
// 		if ((double)rand() / (double)RAND_MAX > 0.5)
// 		{
// 			continue;
// 		}
		Vec3f xyz = pts.at<Vec3f>(i, 0);
		Vec3b color = colors.at<Vec3b>(i, 0);

		m_vertices.push_back(Vertex(xyz, cnt, color));
		cnt++;
	}
	cout << m_vertices.size() << endl;
}

void Delaunay::addBounding( const VertexVector& verSet, 
	TriangleVector& triSet )
{
	float max_x = -100000, max_y = -100000;
	float min_x = 100000, min_y = 100000;
	for (int i = 0; i < verSet.size(); i++)
	{
		Vec3f v = verSet[i].m_xyz;
		max_x = max_x < v[0] ? v[0] : max_x;
		min_x = min_x > v[0] ? v[0] : min_x;
		max_y = max_y < v[1] ? v[1] : max_y;
		min_y = min_y > v[1] ? v[1] : min_y;
	}

	// 将矩形区域的外接三角形作为边界
	float dx = max_x - min_x;
	float dy = max_y - min_y;
	float mid_x = (min_x + max_x) / 2;
	float mid_y = (min_y + max_y) / 2;

	// 为了去除边界方便讲边界点的索引置为负数
	Vertex v0(Vec3f(mid_x, max_y + dy, 0.0f), -1);
	Vertex v1(Vec3f(mid_x - dx, min_y, 0.0f), -2);
	Vertex v2(Vec3f(mid_x + dx, min_y, 0.0f), -3);

	triSet.push_back(Triangle(v0, v1, v2));
}

void Delaunay::removeBounding( TriangleVector& triSet )
{
	for (TriangleVector::iterator iter = triSet.begin(); 
		iter != triSet.end();)
	{
		if (iter->m_vertices[0].m_index < 0 || 
			iter->m_vertices[1].m_index < 0 ||
			iter->m_vertices[2].m_index < 0)
		{
			iter = triSet.erase(iter);
		}
		else
		{
			iter++;
		}
	}
}

void Delaunay::insertVertex( TriangleVector& triSet, const Vertex& v )
{
	TriangleVector tmp;
	for (TriangleVector::iterator iter = triSet.begin(); 
		iter != triSet.end();)
	{
		int r = iter->inTriangle(v);	// 判断点是否在三角形内
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
		if (!flipTest(triSet, tmp[i]))	// 优化delaunay三角
		{
			triSet.push_back(tmp[i]);
		}
/*		drawTrianglesOnPlane(triSet);*/
	}
}

bool Delaunay::flipTest( TriangleVector& triSet, Triangle t )
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

		// 寻找拥有相同边ab的三角形
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
			if (inCircle(a, b, c, d))	// 判断d是否在三角形abc的外接圆内
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
	Vec3f ac = c.m_xyz - a.m_xyz;
	Vec3f ab = b.m_xyz - a.m_xyz;
	Vec3f ap = p.m_xyz - a.m_xyz;
	Vec3f bc = ac - ab;
	Vec3f ba = -ab;
	Vec3f bp = ap - ab;
	float acb = ab[0] * ac[1] - ab[1] * ac[0];
	float acp = ap[0] * ac[1] - ap[1] * ac[0];
	float bca = bc[0] * ba[1] - bc[1] * ba[0];
	float bcp = bc[0] * bp[1] - bc[1] * bp[0];

	if (acb * acp > 0 && bca * bcp > 0)	// b、p在ac同侧且a、p在bc同侧
	{
		Vec3f pa = -ap;
		Vec3f pc = ac - ap;
		float d_apc = pa.dot(pc) * sqrtf(ba.dot(ba)) * sqrtf(bc.dot(bc));
		float d_abc = ba.dot(bc) * sqrtf(pa.dot(pa)) * sqrtf(pc.dot(pc));
		if (d_apc < d_abc)
		{
			return true;
		}
	}
	return false;
}

void Delaunay::drawTrianglesOnPlane( const TriangleVector& triSet )
{
	int width = 1024, height = 768;
	Mat triangleImg(height, width, CV_8UC3, Scalar::all(0));
	if (triSet.size() == 0)
	{
		imshow("Triangle", triangleImg);
		waitKey();
		return;
	}

	float max_x = -100000, max_y = -100000;
	float min_x = 100000, min_y = 100000;
	for (int i = 0; i < triSet.size(); i++)
	{
		for (int j = 0; j < Triangle::Vertex_Size; j++)
		{
			Vec3f v = triSet[i].m_vertices[j].m_xyz;
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
			float tmpx = triSet[i].m_vertices[j].m_xyz[0];
			float tmpy = triSet[i].m_vertices[j].m_xyz[1];

			float x = (tmpx - min_x) * 800 / (max_x - min_x) + 100;
			float y = (max_y - tmpy) * 600 / (max_y - min_y) + 100;

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
			float tmpx = triSet[i].m_vertices[j].m_xyz[0];
			float tmpy = triSet[i].m_vertices[j].m_xyz[1];

			float fx = (tmpx - min_x) * 800 / (max_x - min_x) + 100;
			float fy = (max_y - tmpy) * 600 / (max_y - min_y) + 100;

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
