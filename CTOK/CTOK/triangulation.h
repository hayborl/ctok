#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "opencv2/opencv.hpp"
#include "boost/unordered_set.hpp"

using namespace cv;

namespace Triangulation
{
#define CLOSE_DISTANCE	10		// 判断两个点是否相等的临界距离
#define FLOAT_ERROR		1e-3	// 浮点数的误差范围
#define SQRT_3			1.732	// 根号3

	// distance_range 寻找最近邻的范围球的半径平方
	enum{Distance_Range = 5000};

	// 用于三角化的点的类
	class Vertex
	{
	public:
		Vec3f m_xyz;		// x、y、z坐标
		Vec3b m_color;		// 颜色信息
		int m_index;		// 索引值

		Vertex() : m_index(-1){}
		Vertex(float x, float y, float z) 
			: m_index(-1){ m_xyz = Vec3f(x, y, z);}
		Vertex(Vec3f xyz, int index = -1, Vec3b color = Vec3b(0, 0, 0)) 
			: m_xyz(xyz), m_index(index), m_color(color){}

		Vertex& operator=(const Vertex& v);
		bool operator==(const Vertex& v)const;

		float distance2(const Vertex& v);	// 求点之间的距离，返回距离平方

		static float cross(Vertex a, Vertex b, Vertex c);	// ab x ac，只考虑x, y。
	};

	// 三角形类
	class Triangle
	{
	public:
		enum {Vertex_Size = 3};
		Vertex m_vertices[3];	// 三个顶点

		Triangle(){}
		Triangle(Vertex v0, Vertex v1, Vertex v2);

		Vec3f getNormal();				// 获取三角形的法向量
		void turnBack();				// 交换v0、v2
	
		/*	判断一个点是否在三角形内，
			在三角形内返回0,
			在v0v1边上返回-1，
			在v1v2边上返回-2，
			在v2v0边上返回-3*/
		int inTriangle(Vertex v);
		bool isVertex(const int index);	// 判断当前索引值的点是否是三角形的顶点

		bool operator==(const Triangle& t)const;
	};
	size_t hash_value(const Triangle& t);	// hash函数，用于boost库的set类

	typedef vector<Vertex> VertexVector;
	typedef vector<Triangle> TriangleVector;
	typedef boost::unordered::unordered_set<Triangle> TriangleSet;

	// 进行2D delaunay三角划分的类
	class Delaunay
	{
	public:
		TriangleVector m_triangles;	// 三角形集合

		Delaunay(){}
		Delaunay(const Mat& pts, const vector<Vec3b>& colors);
		Delaunay(const Mat& pts, const Mat& colors);

		void computeDelaunay();				// 计算Delaunay三角
		void saveTriangles(char* file);		// 保存当前划分出的三角形到文件
		void addVertices(const Mat& pts, 
			const vector<Vec3b>& colors);	// 加入点以及对应的颜色
		void addVertices(const Mat& pts, 
			const Mat& colors);				// 加入点以及对应的颜色

	private:
		VertexVector m_vertices;		// 点集
		int m_pre_size;					// 记录下一次compute之前有多少点
		enum {k = 20};

		void computeDelaunay(const VertexVector& verSet, 
			TriangleVector& triSet);	// 根据指定点集计算三角形
		void addBounding(const VertexVector& verSet, 
			TriangleVector& triSet);	// 增加一个边界三角形将所有点包围在内
		void removeBounding(TriangleVector& triSet, 
			const int& index);			// 保留当前索引值的点所在的不包含边界的三角形
		void insertVertex(TriangleVector& triSet, 
			const Vertex& v);			// 往已有三角形中插入一个点
		bool flipTest(TriangleVector& triSet, 
			Triangle t);				// 优化三角形使符合delaunay的条件

		// ab为公共边，abc为原始三角形，abp为共边的三角形, 判断p是否在abc的外接圆内
		bool inCircle(Vertex a, Vertex b, Vertex c, Vertex p);
			
		void saveTriangles(const TriangleVector& triSet, char* file);	// 将指定三角形集合保存到文件
		void drawTrianglesOnPlane(const TriangleVector& triSet);		// 显示指定的三角形集合
	};
}

#endif