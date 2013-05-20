#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "opencv2/opencv.hpp"
#include "boost/unordered_set.hpp"

using namespace cv;

namespace Triangulation
{
#define DOUBLE_ERROR	1e-6	// ����������Χ
#define COS0			0.0		// cos0
#define COS30			0.866	// cos30
#define COS60			0.5		// cos60
#define COS180			-1.0	// cos180
#define DISTANCE_RANGE	0.0009	// Ѱ������ڵķ�Χ��İ뾶ƽ��(0.03m)^2

	// distance_range Ѱ������ڵķ�Χ��İ뾶ƽ��
	// k���ڽ���
	enum {k = 25};

	// �������ǻ��ĵ����
	class Vertex
	{
	public:
		Vec3d m_xyz;					// x��y��z����
		Vec3b m_color;					// ��ɫ��Ϣ
		Vec3d m_normal;					// ������
		double m_residual;				// �в�ֵ
		int m_index;					// ����ֵ
		int m_neighbors[k];				// ����������ֵ, m_neighbors[0]��ʾ�ھ���Ŀ

		Vertex() : m_index(-1){m_neighbors[0] = 0;}
		Vertex(double x, double y, double z) 
			: m_index(-1){ m_xyz = Vec3d(x, y, z); 
				m_neighbors[0] = 0;m_residual = 0;}
		Vertex(Vec3d xyz, int index = -1, Vec3b color = Vec3b(0, 0, 0), 
			Vec3d normal = Vec3d(0, 0, 0), double residual = 0) 
			: m_xyz(xyz), m_index(index), m_color(color), 
				m_normal(normal), m_residual(residual)
			{m_neighbors[0] = 0;}

		Vertex& operator=(const Vertex &v);
		bool operator==(const Vertex &v)const;
		bool operator<(const Vertex &v)const;
		operator Vec3d(){return m_xyz;};

		double distance2(const Vertex &v);	// ���֮��ľ��룬���ؾ���ƽ��

		static double cross(Vertex a, Vertex b, Vertex c);	// ab x ac��ֻ����x, y��
	};

	// ��������
	class Triangle
	{
	public:
		enum {Vertex_Size = 3};
		Vertex m_vertices[3];	// ��������

		Triangle(){}
		Triangle(Vertex v0, Vertex v1, Vertex v2);

		Vec3d getNormal();				// ��ȡ�����εķ�����
		void turnBack();				// ����v0��v2
	
		/*	�ж�һ�����Ƿ����������ڣ�
			���������ڷ���0,
			��v0v1���Ϸ���-1��
			��v1v2���Ϸ���-2��
			��v2v0���Ϸ���-3*/
		int inTriangle(Vertex v);
		bool isVertex(const int index);			// �жϵ�ǰ����ֵ�ĵ��Ƿ��������εĶ���
		bool angleCriterion(const double &minCosAngle = 1.0f,
			const double &maxCosAngle = -1.0f);	// �ж��������Ƿ���������С������

		Triangle operator=(const Triangle &t);
		bool operator==(const Triangle &t)const;
	};
	size_t hash_value(const Triangle &t);		// hash����������boost���set��

	typedef vector<Vertex> VertexVector;
	typedef vector<Triangle> TriangleVector;
	typedef boost::unordered::unordered_set<Triangle> TriangleSet;

	// Mesh��
	class Mesh
	{
	public:
		int m_curIndex;							// ��ǰ���㵽�ĸ���
		TriangleVector m_triangles;				// �����μ���

		Mat m_userT;

		Mesh()
		{m_curIndex = 0; 
		 m_barycenter = Vec3d(0, 0, 0); 
		 m_userT = Mat::eye(4, 4, CV_64FC1);}
		Mesh(InputArray pts, InputArray colors);

		Vec3d barycenter(){return m_barycenter / (double)getVerticesSize();}

		void saveVertices(char *filename);
		void addVertex(const Vertex &v);
		void addVertices(InputArray _pts,
			InputArray _colors);				// ������Լ���Ӧ����ɫ
		size_t getVerticesSize()
			{return m_vertices.size();}			// ��ȡ�������Ŀ
		Vertex getVertex(const int &i)			// ��õ�i����
		{
			assert(i < m_vertices.size());
			return m_vertices[i];
		}				
		void getVertices(const int &times, 
			Mat &out);							// �õ���times�εĵ�������
		void updateVertices(const int &times,
			const Mat &in);						// �滻��times�εĵ�
		void computeVerticesNormals()
		{
			computeVerticesNormals(m_curIndex, 
				(int)m_vertices.size() - m_curIndex);
		}

		size_t getTriangleSize()
			{return m_triangles.size();}		// ��ȡ������Ƭ������Ŀ
		Triangle getTriangle(const int &i)		// ��õ�i��������Ƭ
		{
			assert(i < m_triangles.size());
			return m_triangles[i];
		}			
		void pushTriBeginIndex(const int &i)
			{m_beginIndicesTri.push_back(i);}	// ��ÿ�������������ε���ʼ����ѹ��
		void saveTriangles(char* file);			// ���浱ǰ���ֳ��������ε��ļ�

		void render(int drawType, 
			bool selected);						// ����ģ��

	private:
		VertexVector m_vertices;				// �㼯
		Vec3d m_barycenter;						// ����
		vector<int> m_beginIndicesVer;			// ÿ�μ���ĵ����ʼ����
		vector<int> m_beginIndicesTri;			// ÿ�������ɵ������ε���ʼ����

		void computeVerticesNormals(
			const int &begin, const int &size);	// ����ÿ����ķ�����
	};

	// ����2D delaunay���ǻ��ֵ���
	class Delaunay
	{
	public:				
		Delaunay(double minAngle = COS30, double maxAngle = COS180) 
			: m_minAngle(minAngle), m_maxAngle(maxAngle){}

		void computeDelaunay(Mesh &mesh);		// ����Delaunay����

		static void saveTriangles(const TriangleVector &triSet, char* file);	// ��ָ�������μ��ϱ��浽�ļ�

	private:
		int m_preSize;					// ��¼��һ��compute֮ǰ�ж��ٵ�
		double m_minAngle, m_maxAngle;	// �����С�ǵĽǶ�cosֵ����
		enum {u = 5};					// u����С�����ڵĵ�ȥ��

		void computeDelaunay(const VertexVector &verSet, 
			TriangleVector &triSet);	// ����ָ���㼯����������
		void addBounding(const VertexVector &verSet, 
			TriangleVector &triSet);	// ����һ���߽������ν����е��Χ����
		void removeBounding(TriangleVector inSet, TriangleVector &outSet,
			const int &index);			// ������ǰ����ֵ�ĵ����ڵĲ������߽��������
		void insertVertex(TriangleVector &triSet, 
			const Vertex &v);			// �������������в���һ����
		bool flipTest(TriangleVector &triSet, 
			Triangle t);				// �Ż�������ʹ����delaunay������

		// abΪ�����ߣ�abcΪԭʼ�����Σ�abpΪ���ߵ�������, �ж�p�Ƿ���abc�����Բ��
		bool inCircle(Vertex a, Vertex b, Vertex c, Vertex p);
		
		void drawTrianglesOnPlane(const TriangleVector &triSet);		// ��ʾָ���������μ���
	};
}

#endif