#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "opencv2/opencv.hpp"
#include "boost/unordered_set.hpp"

using namespace cv;

namespace Triangulation
{
#define DOUBLE_ERROR	1e-6	// 浮点数的误差范围
#define COS0			0.0		// cos0
#define COS30			0.866	// cos30
#define COS60			0.5		// cos60
#define COS180			-1.0	// cos180
#define DISTANCE_RANGE	0.0009	// 寻找最近邻的范围球的半径平方(0.03m)^2

	// distance_range 寻找最近邻的范围球的半径平方
	// k个邻近点
	enum {k = 25};

	// 用于三角化的点的类
	class Vertex
	{
	public:
		Vec3d m_xyz;					// x、y、z坐标
		Vec3b m_color;					// 颜色信息
		Vec3d m_normal;					// 法向量
		double m_residual;				// 残差值
		int m_index;					// 索引值
		int m_neighbors[k];				// 最近点的索引值, m_neighbors[0]表示邻居数目

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

		double distance2(const Vertex &v);	// 求点之间的距离，返回距离平方

		static double cross(Vertex a, Vertex b, Vertex c);	// ab x ac，只考虑x, y。
	};

	// 三角形类
	class Triangle
	{
	public:
		enum {Vertex_Size = 3};
		Vertex m_vertices[3];	// 三个顶点

		Triangle(){}
		Triangle(Vertex v0, Vertex v1, Vertex v2);

		Vec3d getNormal();				// 获取三角形的法向量
		void turnBack();				// 交换v0、v2
	
		/*	判断一个点是否在三角形内，
			在三角形内返回0,
			在v0v1边上返回-1，
			在v1v2边上返回-2，
			在v2v0边上返回-3*/
		int inTriangle(Vertex v);
		bool isVertex(const int index);			// 判断当前索引值的点是否是三角形的顶点
		bool angleCriterion(const double &minCosAngle = 1.0f,
			const double &maxCosAngle = -1.0f);	// 判断三角形是否符合最大、最小角限制

		Triangle operator=(const Triangle &t);
		bool operator==(const Triangle &t)const;
	};
	size_t hash_value(const Triangle &t);		// hash函数，用于boost库的set类

	typedef vector<Vertex> VertexVector;
	typedef vector<Triangle> TriangleVector;
	typedef boost::unordered::unordered_set<Triangle> TriangleSet;

	// Mesh类
	class Mesh
	{
	public:
		int m_curIndex;							// 当前计算到哪个点
		TriangleVector m_triangles;				// 三角形集合

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
			InputArray _colors);				// 加入点以及对应的颜色
		size_t getVerticesSize()
			{return m_vertices.size();}			// 获取点的总数目
		Vertex getVertex(const int &i)			// 获得第i个点
		{
			assert(i < m_vertices.size());
			return m_vertices[i];
		}				
		void getVertices(const int &times, 
			Mat &out);							// 得到第times次的点云数据
		void updateVertices(const int &times,
			const Mat &in);						// 替换第times次的点
		void computeVerticesNormals()
		{
			computeVerticesNormals(m_curIndex, 
				(int)m_vertices.size() - m_curIndex);
		}

		size_t getTriangleSize()
			{return m_triangles.size();}		// 获取三角面片的总数目
		Triangle getTriangle(const int &i)		// 获得第i个三角面片
		{
			assert(i < m_triangles.size());
			return m_triangles[i];
		}			
		void pushTriBeginIndex(const int &i)
			{m_beginIndicesTri.push_back(i);}	// 将每个点生成三角形的起始索引压入
		void saveTriangles(char* file);			// 保存当前划分出的三角形到文件

		void render(int drawType, 
			bool selected);						// 绘制模型

	private:
		VertexVector m_vertices;				// 点集
		Vec3d m_barycenter;						// 重心
		vector<int> m_beginIndicesVer;			// 每次加入的点的起始索引
		vector<int> m_beginIndicesTri;			// 每个点生成的三角形的起始索引

		void computeVerticesNormals(
			const int &begin, const int &size);	// 计算每个点的法向量
	};

	// 进行2D delaunay三角划分的类
	class Delaunay
	{
	public:				
		Delaunay(double minAngle = COS30, double maxAngle = COS180) 
			: m_minAngle(minAngle), m_maxAngle(maxAngle){}

		void computeDelaunay(Mesh &mesh);		// 计算Delaunay三角

		static void saveTriangles(const TriangleVector &triSet, char* file);	// 将指定三角形集合保存到文件

	private:
		int m_preSize;					// 记录下一次compute之前有多少点
		double m_minAngle, m_maxAngle;	// 最大最小角的角度cos值限制
		enum {u = 5};					// u倍最小距离内的点去除

		void computeDelaunay(const VertexVector &verSet, 
			TriangleVector &triSet);	// 根据指定点集计算三角形
		void addBounding(const VertexVector &verSet, 
			TriangleVector &triSet);	// 增加一个边界三角形将所有点包围在内
		void removeBounding(TriangleVector inSet, TriangleVector &outSet,
			const int &index);			// 保留当前索引值的点所在的不包含边界的三角形
		void insertVertex(TriangleVector &triSet, 
			const Vertex &v);			// 往已有三角形中插入一个点
		bool flipTest(TriangleVector &triSet, 
			Triangle t);				// 优化三角形使符合delaunay的条件

		// ab为公共边，abc为原始三角形，abp为共边的三角形, 判断p是否在abc的外接圆内
		bool inCircle(Vertex a, Vertex b, Vertex c, Vertex p);
		
		void drawTrianglesOnPlane(const TriangleVector &triSet);		// 显示指定的三角形集合
	};
}

#endif