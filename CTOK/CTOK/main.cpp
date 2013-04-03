#include <GL/freeglut.h>

#include "icp.h"
#include "emicp.h"
#include "segment.h"
#include "features.h"
#include "camera.h"
#include "triangulation.h"

using namespace std;
using namespace cv;
using namespace xn;

void checkOpenNIError(XnStatus result, string status)  
{   
	if (result != XN_STATUS_OK)
	{
		cerr << status << " Error: " << xnGetStatusString(result) << endl;  
		exit(-1);
	}
}

//////////////////////////////////////////////////////////////////////////
//
//---OpenGL 全局变量
vector<Vec3f> pointCloudData;
vector<Vec3b> pointColorData;
size_t pointNumber = 0;

int glWinWidth = 640, glWinHeight = 480;
int glWinPosX = 30, glWinPosY = 30;
int width = 640, height = 480;
bool breakScan = false;

Camera userCamera;
bool hasCuda = true;

Features features;							// 用以获取特征比较相似度
#define SIMILARITY_THRESHOLD 0.001			// 相似度阈值

Triangulation::Delaunay delaunay(COS30);

char* videoFile = "ttt.oni";

// 读取每一帧的彩色图与深度图
void readFrame(ImageGenerator ig, DepthGenerator dg, 
	Mat &colorImg, Mat &depthImg)
{
	ImageMetaData imageMD;     // 创建image节点元数据对象    
	DepthMetaData depthMD;     // 创建depth节点元数据对象
	Mat depthImageShow, colorImageShow;
	int cols, rows;

	//get meta data  
	dg.GetMetaData(depthMD);   
	ig.GetMetaData(imageMD);

	cols = depthMD.XRes();
	rows = depthMD.YRes();

	//OpenCV output
	Mat imgDepth16U(rows, cols, CV_16UC1, (void*)depthMD.Data());
	medianBlur(imgDepth16U, imgDepth16U, 5);

	imgDepth16U.convertTo(depthImageShow, CV_8UC1, 255 / 5000.0);
	Mat imgRGB(rows, cols, CV_8UC3, (void*)imageMD.Data());
	cvtColor(imgRGB, colorImageShow, CV_RGB2BGR);
	imshow("depth", depthImageShow);
	imshow("image", colorImageShow);

	colorImg = colorImageShow.clone();
	depthImg = imgDepth16U.clone();
}

// 根据深度图，彩色图，mask获取真实世界的点的坐标与颜色信息
void read3DPoints(DepthGenerator dg, const Mat &depthImg, 
	const Mat &colorImg, const Mat &mask, 
	Mat &realPointCloud, Mat &pointColors)
{
	int cols, rows, index = 0;
	cols = depthImg.cols;
	rows = depthImg.rows;

	Ptr<XnPoint3D> proj = new XnPoint3D[cols * rows];
	Ptr<XnPoint3D> real = new XnPoint3D[cols * rows];
	vector<Vec3b> colors;
	Mat tmpDepthImg;
	depthImg.copyTo(tmpDepthImg, mask);
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			ushort z = tmpDepthImg.at<ushort>(y, x);
			if (z != 0)
			{
				proj[index].X = (float)x;
				proj[index].Y = (float)y;
				proj[index].Z = (float)z;
				colors.push_back(colorImg.at<Vec3b>(y, x));

				index++;
			}
		}
	}

	dg.ConvertProjectiveToRealWorld(index, proj, real);

	realPointCloud = Mat(index, 1, DataType<Point3f>::type, Scalar::all(0));
#pragma omp parallel for
	for (int i = 0; i < index; i++)
	{
		realPointCloud.at<Point3f>(i, 0) = Point3f(
			real[i].X, real[i].Y, real[i].Z);
	}

	pointColors = Mat(colors, true).clone();
}

// 载入3D坐标数据及其颜色数据
void loadPointCloudAndTexture(const Mat &pointCloud, 
	const Mat &pointColor, bool clear = false)
{
	assert(pointCloud.rows == pointColor.rows && 
		pointCloud.cols == pointColor.cols);
	assert(pointCloud.cols == 1);
	if (clear)
	{
		pointCloudData.clear();
		pointColorData.clear();
		pointNumber = 0;
	}

	Vec3f p;
	Vec3b color;
	for (int i = 0; i < pointCloud.rows; i ++/*= SAMPLE_INTERVAL*/)
	{
		p = pointCloud.at<Vec3f>(i, 0);
		if (p != Vec3f(0, 0, 0)/* && (double)rand() / (double)RAND_MAX < 0.5*/)
		{
			p[2] = -p[2];
			pointCloudData.push_back(p);
			color = pointColor.at<Vec3b>(i, 0);
			pointColorData.push_back(
				Vec3b(color[2], color[1], color[0]));
		}
	}

	pointNumber = pointCloudData.size();

	cout << pointNumber << " OK" << endl;
}

// 绘制点云
void drawPoints()
{
	float x,y,z;
	// 绘制图像点云
	glPointSize(1.0);
	glBegin(GL_POINTS);
	for (int i = 0; i < pointCloudData.size(); i++)
	{
		glColor3d(pointColorData[i][0] / 255.0, 
			pointColorData[i][1] / 255.0, pointColorData[i][2] / 255.0);
		x = (float)pointCloudData[i][0];
		y = (float)pointCloudData[i][1];
		z = (float)pointCloudData[i][2];
		glVertex3f(x, y, z);
	}
	glEnd();
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < delaunay.m_triangles.size(); i++)
	{
		Triangulation::Triangle t = delaunay.m_triangles[i];
		for (int j = 0; j < Triangulation::Triangle::Vertex_Size; j++)
		{
			Triangulation::Vertex v = t.m_vertices[j];
			glColor3d(v.m_color[2] / 255.0, 
				v.m_color[1] / 255.0, v.m_color[0] / 255.0);
			glVertex3f(v.m_xyz[0], v.m_xyz[1], -v.m_xyz[2]);
		}
	}
	glEnd(); 
}

/************************************************************************/
/*                       OpenGL响应函数                                 */
/************************************************************************/

//////////////////////////////////////////////////////////////////////////
// 鼠标按键响应函数
void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			userCamera.setMouseState(true);
		}
		else
		{
			userCamera.setMouseState(false);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
// 鼠标运动响应函数
void motion(int x, int y)
{
}

void keyboard(uchar key, int x, int y)
{
	switch(key)
	{
	case 27:
		exit(0);
		break;
	case 'D':
	case 'd':
		userCamera.strafeCamera(MOVESPEEDFB);
		glutPostRedisplay();
		break;
	case 'A':
	case 'a':
		userCamera.strafeCamera(-MOVESPEEDFB);
		glutPostRedisplay();
		break;
	case 'W':
	case 'w':
		userCamera.moveCamera(MOVESPEEDFB);
		glutPostRedisplay();
		break;
	case 'S':
	case 's':
		userCamera.moveCamera(-MOVESPEEDFB);
		glutPostRedisplay();
		break;
	case 'p':
	case 'P':
		saveData("real.xyz", pointCloudData);
		cout << "save points" << endl;
		break;
	case 'r':
	case 'R':
		userCamera.reset();
		break;
	case 'b':
	case 'B':
		breakScan = true;
		break;
	default:
		break;
	}
}

void mouseEntry(int state)
{
	switch (state)
	{
	case GLUT_LEFT:
		userCamera.setMouseState(false);
		ShowCursor(TRUE);
		break;
	case GLUT_ENTERED:
		userCamera.setMouseState(false);
		ShowCursor(TRUE);
		break;
	}
}

//////////////////////////////////////////////////////////////////////////
// 三维图像显示响应函数
void renderScene(void)
{
	// clear screen and depth buffer
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	// Reset the coordinate system before modifying
	glLoadIdentity();   
	// set the camera position
	userCamera.look();

	// 对点云数据进行三角化
	// 参考自：http://www.codeproject.com/KB/openGL/OPENGLTG.aspx

	drawPoints();

	Vec3f vPos = userCamera.position();
	Vec3f vView = userCamera.view();

	// 设置camera的新位置
	userCamera.positionCamera(vPos, vView, Vec3f(0.0f, 1.0f, 0.0f));

	glFlush();
	glutSwapBuffers();
}

//////////////////////////////////////////////////////////////////////////
// 窗口变化图像重构响应函数
void reshape (int w, int h)
{
	glWinWidth = w;
	glWinHeight = h;
	glViewport (0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	gluPerspective (45, (GLfloat)w / (GLfloat)h, 1.0, 15000.0);   
	glMatrixMode (GL_MODELVIEW);
}

/************************************************************************/
/*                         主程序                                       */
/************************************************************************/
int main(int argc, char** argv)
{
	hasCuda = initCuda();
	userCamera.positionCamera(0.0f, 1.8f, 100.0f, 
		0.0f, 1.8f, 0.0f, 0.0f, 1.0f, 0.0f);		// 定位摄像机

	// OpenGL Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowPosition(glWinPosX, glWinPosY);
	glutInitWindowSize(glWinWidth, glWinHeight);
	glutCreateWindow("3D image");

	glutReshapeFunc (reshape);			// 窗口变化时重绘图像
	glutDisplayFunc(renderScene);		// 显示三维图像       
	glutMouseFunc(mouse);				// 鼠标按键响应
	glutEntryFunc(mouseEntry);			// 设置鼠标进入窗口的处理函数
	glutMotionFunc(motion);				// 鼠标移动响应
	glutKeyboardFunc(keyboard);			// 键盘按键响应
	glutIdleFunc(renderScene);			// 空闲时重绘图像

	// OpenCV Window
	namedWindow("depth", CV_WINDOW_AUTOSIZE);  
	namedWindow("image", CV_WINDOW_AUTOSIZE);  

	// OpenNI 对象
	XnStatus rc = XN_STATUS_OK;
	Context context;				//创建上下文对象
	rc = context.Init();			//上下文对象初始化 
	checkOpenNIError(rc, "initialize context");

	Player player;
	rc = context.OpenFileRecording(videoFile, player);						//打开已有的oni文件
	checkOpenNIError(rc, "Open File Recording");

	ImageGenerator imageGenerator;											//创建image generator
	rc = context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);		//获取oni文件中的image节点
	checkOpenNIError(rc, "Create Image Generator");   
	DepthGenerator depthGenerator;											//创建depth generator
	rc = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator);		//获取oni文件中的depth节点
	checkOpenNIError(rc, "Create Depth Generator"); 
	depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);

	// 开始获取并显示 Kinect 图像
	rc = context.StartGeneratingAll();
	rc = context.WaitNoneUpdateAll();

	Mat colorImgPre, colorImgNow, depthImgPre, depthImgNow;		// 前后两帧的彩色图与深度图
	Mat tr = Mat::eye(4, 4, CV_32FC1);
	pair<Mat, Mat> desPre, desNow;

	XnUInt32 frameCnt = 1;
	for (; ; frameCnt++) 
	{
		if (breakScan)
		{
			break;
		}

		rc = context.WaitOneUpdateAll(depthGenerator);
		rc = context.WaitOneUpdateAll(imageGenerator);

		if (depthGenerator.GetFrameID() < frameCnt)
		{
			break;
		}

		Mat colorImg, depthImg;
		RUNANDTIME(global_timer, readFrame(imageGenerator, depthGenerator, 
			colorImg, depthImg), OUTPUT, "read one frame.");
		colorImgPre = colorImgNow.clone();
		depthImgPre = depthImgNow.clone();
		colorImgNow = colorImg.clone();
		depthImgNow = depthImg.clone();

		desPre = desNow;
		features.getHSVColorHistDes(colorImgNow, desNow.first);
		features.getGLCMDes(colorImgNow, desNow.second);

// 		RUNANDTIME(global_timer, simplifyPoints(realPointCloud, 
// 			realPointCloud, 10, 0.9), OUTPUT, "simplify");

		Mat pointCloud, pointColors;
		if (frameCnt == 1)
		{
			Mat mask(colorImg.rows, colorImg.cols, CV_8UC1, Scalar::all(255));
			RUNANDTIME(global_timer, read3DPoints(depthGenerator, 
				depthImg, colorImg, mask, pointCloud, pointColors), 
				OUTPUT, "read 3D points");
		}
		else
		{
			double distance = computeDistance(desPre, desNow);
			if (distance < SIMILARITY_THRESHOLD)	// 判断两帧的相似度，小于阈值则不匹配
			{
				colorImgNow = colorImgPre.clone();
				depthImgNow = depthImgPre.clone();
				desNow = desPre;
			}
			else
			{
				Mat objSet, objSetAT, modSet, mask;	// 依次为当前帧特征点集，经转换后当前帧特征点集，前一帧特征点集
				RUNANDTIME(global_timer, getFeaturePoints(depthGenerator, 
					colorImgNow, depthImgNow, colorImgPre, depthImgPre, 
					objSet, modSet, objSetAT, mask), 
					OUTPUT, "get feature points.");

				/*ICP i(objSet, modSet);*/
				EMICP i(objSet, modSet, 0.01f, 0.00001f, 0.7f, 0.01f);

				RUNANDTIME(global_timer, 
					i.run(hasCuda, &objSetAT), OUTPUT, "run ICP.");
				tr = i.getFinalTransformMat().clone() * tr;
				cout << i.getFinalTransformMat() << endl << tr << endl;

				RUNANDTIME(global_timer, read3DPoints(depthGenerator, 
					depthImg, colorImg, mask, pointCloud, pointColors), 
					OUTPUT, "read 3D points");
				RUNANDTIME(global_timer, 
					transformPointCloud(pointCloud, &pointCloud, tr, hasCuda), 
					OUTPUT, "transform point cloud.");

/*				waitKey();*/
			}
		}

		if (pointCloud.rows > 0 && pointColors.rows > 0)
		{

			RUNANDTIME(global_timer, loadPointCloudAndTexture(pointCloud, 
				pointColors, false), OUTPUT, "load data");
/*			waitKey();*/
// 			RUNANDTIME(global_timer, delaunay.addVertices(pointCloud, 
// 				pointColors), OUTPUT, "load data");
// 			RUNANDTIME(global_timer, delaunay.computeDelaunay(), 
// 				OUTPUT, "delaunay");
// 			cout << delaunay.m_triangles.size() << endl;
// 			delaunay.saveTriangles("triangles.tri");
		}

		char key = waitKey(1);
		if (key == 27)
		{
			break;
		}

		glutPostRedisplay();	// 刷新画面
		glutMainLoopEvent();	// OpenCV 处理键盘响应消息后，再显示 OpenGL 图像
	}

	// destroy  
	destroyAllWindows();
	context.StopGeneratingAll();
	context.Release();

	cout << pointNumber << endl;
	glutMainLoop();
	return 0;
}


//int main()
//{
//	segment myseg;
//	Mat ldimage = imread("./images/8.jpg",0);
//	if(! ldimage.data) {
//		cout<<"load image failed\n";
//		system("pause");
//		return 0;
//	}
//	myseg.setDepthImage(ldimage);
//	RUNANDTIME(global_timer, myseg.oversegMyImage((uchar)10),
//		OUTPUT, "seg time");
//	//cout<<cos(1.57079632679489661923);
//	cout<<"good bye";
//
//	system("pause");
//	return 0;
//}