#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "opencv2/opencv.hpp"
#include "boost/unordered_set.hpp"

using namespace cv;

namespace Triangulation
{
#define CLOSE_DISTANCE	10		// �ж��������Ƿ���ȵ��ٽ����
#define FLOAT_ERROR		1e-3	// ����������Χ
#define SQRT_3			1.732	// ����3

	// distance_range Ѱ������ڵķ�Χ��İ뾶ƽ��
	enum{Distance_Range = 5000};

	// �������ǻ��ĵ����
	class Vertex
	{
	public:
		Vec3f m_xyz;		// x��y��z����
		Vec3b m_color;		// ��ɫ��Ϣ
		int m_index;		// ����ֵ

		Vertex() : m_index(-1){}
		Vertex(float x, float y, float z) 
			: m_index(-1){ m_xyz = Vec3f(x, y, z);}
		Vertex(Vec3f xyz, int index = -1, Vec3b color = Vec3b(0, 0, 0)) 
			: m_xyz(xyz), m_index(index), m_color(color){}

		Vertex& operator=(const Vertex& v);
		bool operator==(const Vertex& v)const;

		float distance2(const Vertex& v);	// ���֮��ľ��룬���ؾ���ƽ��

		static float cross(Vertex a, Vertex b, Vertex c);	// ab x ac��ֻ����x, y��
	};

	// ��������
	class Triangle
	{
	public:
		enum {Vertex_Size = 3};
		Vertex m_vertices[3];	// ��������

		Triangle(){}
		Triangle(Vertex v0, Vertex v1, Vertex v2);

		Vec3f getNormal();				// ��ȡ�����εķ�����
		void turnBack();				// ����v0��v2
	
		/*	�ж�һ�����Ƿ����������ڣ�
			���������ڷ���0,
			��v0v1���Ϸ���-1��
			��v1v2���Ϸ���-2��
			��v2v0���Ϸ���-3*/
		int inTriangle(Vertex v);
		bool isVertex(const int index);	// �жϵ�ǰ����ֵ�ĵ��Ƿ��������εĶ���

		bool operator==(const Triangle& t)const;
	};
	size_t hash_value(const Triangle& t);	// hash����������boost���set��

	typedef vector<Vertex> VertexVector;
	typedef vector<Triangle> TriangleVector;
	typedef boost::unordered::unordered_set<Triangle> TriangleSet;

	// ����2D delaunay���ǻ��ֵ���
	class Delaunay
	{
	public:
		TriangleVector m_triangles;	// �����μ���

		Delaunay(){}
		Delaunay(const Mat& pts, const vector<Vec3b>& colors);
		Delaunay(const Mat& pts, const Mat& colors);

		void computeDelaunay();				// ����Delaunay����
		void saveTriangles(char* file);		// ���浱ǰ���ֳ��������ε��ļ�
		void addVertices(const Mat& pts, 
			const vector<Vec3b>& colors);	// ������Լ���Ӧ����ɫ
		void addVertices(const Mat& pts, 
			const Mat& colors);				// ������Լ���Ӧ����ɫ

	private:
		VertexVector m_vertices;		// �㼯
		int m_pre_size;					// ��¼��һ��compute֮ǰ�ж��ٵ�
		enum {k = 20};

		void computeDelaunay(const VertexVector& verSet, 
			TriangleVector& triSet);	// ����ָ���㼯����������
		void addBounding(const VertexVector& verSet, 
			TriangleVector& triSet);	// ����һ���߽������ν����е��Χ����
		void removeBounding(TriangleVector& triSet, 
			const int& index);			// ������ǰ����ֵ�ĵ����ڵĲ������߽��������
		void insertVertex(TriangleVector& triSet, 
			const Vertex& v);			// �������������в���һ����
		bool flipTest(TriangleVector& triSet, 
			Triangle t);				// �Ż�������ʹ����delaunay������

		// abΪ�����ߣ�abcΪԭʼ�����Σ�abpΪ���ߵ�������, �ж�p�Ƿ���abc�����Բ��
		bool inCircle(Vertex a, Vertex b, Vertex c, Vertex p);
			
		void saveTriangles(const TriangleVector& triSet, char* file);	// ��ָ�������μ��ϱ��浽�ļ�
		void drawTrianglesOnPlane(const TriangleVector& triSet);		// ��ʾָ���������μ���
	};
}

#endif