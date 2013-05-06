#include <GL/freeglut.h>

#include "icp.h"
#include "emicp.h"
#include "features.h"
#include "camera.h"
#include "triangulation.h"
#include "bundleadjustment.h"
#include "segmentation.h"
/*#include "ballpivoting.h"*/

using namespace std;
using namespace cv;
using namespace xn;

//---OpenGL 全局变量
int glWinWidth = 1280, glWinHeight = 480;
int glWinPosX = 0, glWinPosY = 0;
int width = 640, height = 480;

#define TYPE_POINT  0x1
#define TYPE_WITH_NORMAL 0x10
#define TYPE_TRIANGLES 0x100
int drawType = TYPE_POINT;

int mainWindow, subWindow1, subWindow2;

Camera userCamera;

//---OpenNI
#define CONFIG_PATH "SamplesConfig.xml"

void checkOpenNIError(XnStatus result, string status)  
{   
	if (result != XN_STATUS_OK)
	{
		cerr << status << " Error: " << xnGetStatusString(result) << endl;  
		exit(-1);
	}
}

char* videoFile = "3281.oni";
XnStatus rc = XN_STATUS_OK;
Context context;							// 上下文对象
XnDouble baseline;							// 基线长度(mm)
XnUInt64 focalLengthInPixel;				// 焦距(pixel)
ImageGenerator imageGenerator;				// image generator
DepthGenerator depthGenerator;				// depth generator

//----CTOK
bool hasCuda = true;						// 有没有cuda
bool isStarted = false;						// 是否开始重构
bool stopScan = false;						// 是否停止扫描

Features global_features;					// 用以获取特征比较相似度

Triangulation::Delaunay global_delaunay(COS30);	// 三角化
Triangulation::Mesh global_mesh;				// 模型

vector<Triangulation::Mesh> meshs;

const Mat identityMat4x4 = Mat::eye(4, 4, CV_64FC1);

/*MyMesh m;*/

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
	depthImg.create(rows, cols, CV_16UC1);
	const XnDepthPixel* pDepthMap = depthMD.Data();
	memcpy(depthImg.data, pDepthMap, cols * rows * sizeof(XnDepthPixel));
	medianBlur(depthImg, depthImg, 5);

	double min, max;
	minMaxLoc(depthImg, &min, &max);

	depthImg.convertTo(depthImageShow, CV_8UC1, 255.0 / max);
	Mat imgRGB(rows, cols, CV_8UC3, (void*)imageMD.Data());
	cvtColor(imgRGB, colorImageShow, CV_RGB2BGR);
	imshow("depth", depthImageShow);
	imshow("image", colorImageShow);
	waitKey(1);							// 稍作延迟让OpenCV窗口能正常显示

	colorImg = colorImageShow.clone();
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
// 	Mat show;
// 	tmpDepthImg.convertTo(show, CV_8UC1, 255.0 / 6000.0);
// 	imshow("show", show);
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

	realPointCloud = Mat(index, 1, DataType<Point3d>::type, Scalar::all(0));
// #pragma omp parallel for
	for (int i = 0; i < index; i++)
	{
		realPointCloud.at<Point3d>(i, 0) = Point3d(
			real[i].X / 1000.0, real[i].Y / 1000.0, real[i].Z / 1000.0);
	}

	pointColors = Mat(colors, true).clone();
}

// 绘制点云
void drawPointsSub1()
{
	if (drawType & TYPE_POINT)
	{
		// 绘制图像点云
		glPointSize(1.0);
		glBegin(GL_POINTS);
		for (int i = 0; i < global_mesh.getVerticesSize(); i++)
		{
			Triangulation::Vertex v = global_mesh.getVertex(i);
			glColor3d(v.m_color[2] / 255.0, 
				v.m_color[1] / 255.0, v.m_color[0] / 255.0);
			glVertex3d(v.m_xyz[0] / 10.0, 
				v.m_xyz[1] / 10.0, -v.m_xyz[2] / 10.0);
		}
		glEnd();
	}
	else if (drawType & TYPE_TRIANGLES)
	{
		glBegin(GL_TRIANGLES);
		for (int i = 0; i < global_mesh.m_triangles.size(); i++)
		{
			Triangulation::Triangle t = global_mesh.m_triangles[i];
			for (int j = 0; j < Triangulation::Triangle::Vertex_Size; j++)
			{
				Triangulation::Vertex v = t.m_vertices[j];
				glColor3d(v.m_color[2] / 255.0, 
					v.m_color[1] / 255.0, v.m_color[0] / 255.0);
				glVertex3d(v.m_xyz[0] / 10.0, 
					v.m_xyz[1] / 10.0, -v.m_xyz[2] / 10.0);
			}
		}
		glEnd(); 
	}
	if (drawType & TYPE_WITH_NORMAL)
	{
		glBegin(GL_LINES);
		for (int i = 0; i < global_mesh.getVerticesSize(); i++)
		{
			Triangulation::Vertex v = global_mesh.getVertex(i);
			Vec3d end = (v.m_xyz + v.m_normal) / 10.0;
			glColor3d(255.0, 0.0, 0.0);
			glVertex3d(v.m_xyz[0] / 10.0, 
				v.m_xyz[1] / 10.0, -v.m_xyz[2] / 10.0);
			glColor3d(255.0, 0.0, 0.0);
			glVertex3d(end[0], end[1], -end[2]);
		}
		glEnd();
	}
// 	MyMesh::FaceIterator fi = m.face.begin();
// 	for (; fi != m.face.end(); ++fi)
// 	{
// 		for (int i = 0; i < 3; i++)
// 		{
// 			MyMesh::VertexPointer v = fi->V(i);
// 			vcg::Color4b c = v->C();
// 			glColor3d(c[0], c[1], c[2]);
// 			glVertex3d(v->P()[0], v->P()[1], v->P()[2]);
// 		}
// 	}
}

void drawPointsSub2()
{
	glPointSize(1.0);
	glBegin(GL_POINTS);
	for (int i = 0; i < meshs.size(); i++)
	{
		for (int j = 0; j < meshs[i].getVerticesSize(); j++)
		{
			Triangulation::Vertex v = meshs[i].getVertex(j);
			glColor3d(v.m_color[2] / 255.0, 
				v.m_color[1] / 255.0, v.m_color[0] / 255.0);
			glVertex3d(v.m_xyz[0] / 10.0, 
				v.m_xyz[1] / 10.0, -v.m_xyz[2] / 10.0);
		}
	}
	glEnd();
}

/************************************************************************/
/*                       OpenGL响应函数                                 */
/************************************************************************/

// 鼠标按键响应函数
void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			userCamera.setMouseState(true);
			glutSetCursor(GLUT_CURSOR_NONE);
		}
		else
		{
			userCamera.setMouseState(false);
			glutSetCursor(GLUT_CURSOR_INHERIT);
		}
	}
}

// 鼠标运动响应函数
void motion(int x, int y)
{
}

// 键盘按键响应函数
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
/*		saveData("real.xyz", global_mesh.m_vertices);*/
		cout << "save points" << endl;
		break;
	case 'r':
	case 'R':
		userCamera.reset();
		break;
	case 'o':
	case 'O':
		stopScan = true;
		break;
	case 'b':
	case 'B':
		cout << "start" << endl;
		isStarted = true;
		break;
	case '1':
		drawType = TYPE_POINT;
		break;
	case '2':
		drawType = TYPE_POINT + TYPE_WITH_NORMAL;
		break;
	case '3':
		drawType = TYPE_TRIANGLES;
		if (stopScan)
		{
			global_delaunay.computeDelaunay(global_mesh);
		}
		break;
	default:
		break;
	}
}

// 鼠标进出窗口响应函数
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

void renderScene(void)
{
	glutSetWindow(mainWindow);
	glWinPosX = glutGet(GLUT_WINDOW_X);
	glWinPosY = glutGet(GLUT_WINDOW_Y);
	glClear(GL_COLOR_BUFFER_BIT);
	glutSwapBuffers();
}

// 三维图像显示响应函数
void renderSceneSub1(void)
{
	glutSetWindow(subWindow1);

	// clear screen and depth buffer
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	// Reset the coordinate system before modifying
	glLoadIdentity();   
	// set the camera position
	userCamera.look();

	drawPointsSub1();

	Vec3d vPos = userCamera.position();
	Vec3d vView = userCamera.view();

	// 设置camera的新位置
	userCamera.positionCamera(vPos, vView, Vec3d(0.0, 1.0, 0.0));

	glFlush();
	glutSwapBuffers();
}

void renderSceneSub2(void)
{
	glutSetWindow(subWindow2);

	// clear screen and depth buffer
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	// Reset the coordinate system before modifying
	glLoadIdentity();   
	// set the camera position
	userCamera.look();

	drawPointsSub2();

	Vec3d vPos = userCamera.position();
	Vec3d vView = userCamera.view();

	// 设置camera的新位置
	userCamera.positionCamera(vPos, vView, Vec3d(0.0, 1.0, 0.0));

	glFlush();
	glutSwapBuffers();
}

void renderSceneAll()
{
	renderSceneSub1();
	renderSceneSub2();
}

void setProjection(int w1, int h1)
{
	float ratio;
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	ratio = 1.0f * w1 / h1;
	// Reset the coordinate system before modifying
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w1, h1);

	// Set the clipping volume
	gluPerspective(45, ratio, 1.0, 15000.0);
	glMatrixMode(GL_MODELVIEW);
}

// 窗口变化图像重构响应函数
void reshape (int w, int h)
{
	if (h == 0)
	{
		h = 1;
	}
	glWinWidth = w;
	glWinHeight = h;

	glutSetWindow(subWindow1);
	// resize and reposition the sub window
	glutPositionWindow(glWinPosX, glWinPosY);
	glutReshapeWindow(glWinWidth / 2, glWinHeight);
	setProjection(glWinWidth / 2, glWinHeight);

	// set subwindow 2 as the active window
	glutSetWindow(subWindow2);
	// resize and reposition the sub window
	glutPositionWindow(glWinPosX + glWinWidth / 2, glWinPosY);
	glutReshapeWindow(glWinWidth / 2, glWinHeight);
	setProjection(glWinWidth / 2, glWinHeight);
}

// 初始化OpenNI
void initOpenNI()
{
	// OpenNI 对象
// 	EnumerationErrors errors;
// 	rc = context.InitFromXmlFile(CONFIG_PATH, &errors);
// 	if (rc == XN_STATUS_NO_NODE_PRESENT)
// 	{
// 		XnChar strError[1024];
// 		errors.ToString(strError, 1024);
// 		printf("%s\n", strError);
// 		exit(-1);
// 	}
// 	else if (rc != XN_STATUS_OK)
// 	{
// 		printf("Open failed: %s\n", xnGetStatusString(rc));
// 		exit(-1);
// 	}

	rc = context.Init();
	checkOpenNIError(rc, "Init context");
	Player player;
	rc = context.OpenFileRecording(videoFile, player);						// 打开已有的oni文件
	checkOpenNIError(rc, "Open File Recording");

	rc = context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);		// 获取oni文件中的image节点
	checkOpenNIError(rc, "Create Image Generator");   
	rc = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator);		// 获取oni文件中的depth节点
	checkOpenNIError(rc, "Create Depth Generator"); 

	// 获得像素大小
	XnDouble pixelSize = 0;
	rc = depthGenerator.GetRealProperty("ZPPS", pixelSize);
	checkOpenNIError(rc, "ZPPS");
	pixelSize *= 2.0;

	// 获得焦距（mm）
	XnUInt64 zeroPlanDistance;
	rc = depthGenerator.GetIntProperty("ZPD", zeroPlanDistance);
	checkOpenNIError(rc, "ZPD");

	// 获得基线长度(mm)
	rc = depthGenerator.GetRealProperty("LDDIS", baseline);
	checkOpenNIError(rc, "LDDIS");
	baseline *= 10;

	// 获得焦距(pixel)
	focalLengthInPixel = (XnUInt64)((double)zeroPlanDistance / (double)pixelSize);

	// 开始获取并显示 Kinect 图像
	rc = context.StartGeneratingAll();
	checkOpenNIError(rc, "Start generating");
	context.WaitNoneUpdateAll();
}

void init()
{
	glutMouseFunc(mouse);				// 鼠标按键响应
	glutEntryFunc(mouseEntry);			// 设置鼠标进入窗口的处理函数
	glutMotionFunc(motion);				// 鼠标移动响应
	glutKeyboardFunc(keyboard);			// 键盘按键响应
}

// 初始化OpenGL
void initOpenGL(int argc, char** argv)
{
	userCamera.positionCamera(0.0, 1.8, 10.0, 
		0.0, 1.8, 0.0, 0.0, 1.0, 0.0);	// 定位摄像机

	// OpenGL Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowPosition(glWinPosX, glWinPosY);
	glutInitWindowSize(glWinWidth, glWinHeight);
	mainWindow = glutCreateWindow("3D Scene");
	glutReshapeFunc (reshape);			// 窗口变化时重绘图像
	glutDisplayFunc(renderScene);		// 显示三维图像       
	glutIdleFunc(renderSceneAll);		// 空闲时重绘图像
	init();

	subWindow1 = glutCreateSubWindow(mainWindow, glWinPosX, 
		glWinPosY, glWinWidth / 2, glWinHeight);
	glutDisplayFunc(renderSceneSub1);
	init();

	subWindow2 = glutCreateSubWindow(mainWindow, glWinPosX + 
		glWinWidth / 2, glWinPosY, glWinWidth / 2, glWinHeight);
	glutDisplayFunc(renderSceneSub2);
	init();

	glEnable(GL_DEPTH_TEST);
}

// 初始化BundleAdjustment的照相机内参
void initBundleAdjustment()
{
	double K[9] = {(double)focalLengthInPixel, 0.0, 320,
		0.0, -(double)focalLengthInPixel, 240,
		0.0, 0.0, 1};
	Mat intrinsic(3, 3, CV_64FC1, K);
	BundleAdjustment::setIntrinsic(intrinsic);	// 设置内参矩阵
}

/************************************************************************/
/*                         主程序                                       */
/************************************************************************/
int main(int argc, char** argv)
{
	hasCuda = initCuda();

	initOpenGL(argc, argv);
	initOpenNI();
	initBundleAdjustment();

	// OpenCV Window
	namedWindow("depth", CV_WINDOW_AUTOSIZE);  
	namedWindow("image", CV_WINDOW_AUTOSIZE);  

	Mat _depthImgPre;					// 前一帧的深度图
	Mat _depthImgPre2;					// 前面第二帧的深度图
	Mat _dscrptPre;						// 前一帧的描述子
	Mat _mask(height, width, CV_8UC1, Scalar::all(255));

	XnUInt32 frameCnt = 0;
	int recordCnt = 0;
	vector<Mat> _camPoses;				// 记录每一帧的相机姿势
	vector<Mat> _incPoses;				// 每一步的增量
	vector<vector<KeyPoint>> _keyPoints;// 记录每一帧的特征点
	vector<Mat>	_descriptors;			// 记录每一帧的特征描述子

	Mat depthImg0;
	isStarted = true;
	for (; ; frameCnt++) 
	{
		rc = context.WaitAndUpdateAll();
		if (rc != XN_STATUS_OK)
		{
			continue;
		}

		if (stopScan || depthGenerator.GetFrameID() < frameCnt)
		{
			break;
		}

		Mat colorImg, depthImg;
		RUNANDTIME(global_timer, 
			readFrame(imageGenerator, depthGenerator, colorImg, depthImg), 
			OUTPUT, "read one frame.");

		if (!isStarted)
		{
			continue;
		}

		if (recordCnt == 0)
		{
			depthImg0 = depthImg.clone();
		}

		Mat dscrpt;
		RUNANDTIME(global_timer, 
			global_features.getPHash(colorImg, dscrpt),
			OUTPUT, "get hash using pHash");

		Mat pointCloud, pointColors;
		Mat pose = identityMat4x4.clone();
		Mat incMat = identityMat4x4.clone();
		int dist = hammingDistance(_dscrptPre, dscrpt);
		if (dist > SIMILARITY_THRESHOLD_PHASH)
		{
			vector<KeyPoint> keyPoint;
			Mat descriptor;
			RUNANDTIME(global_timer, 
				get2DFeaturePoints(colorImg, keyPoint, descriptor), 
				OUTPUT, "get feature points and descriptor.");
			if (recordCnt > 0)
			{
				Mat H;
				vector<pair<int, int>> matchesPoints;
				pair<double, double> scoreDiff;
				int last = (int)_keyPoints.size() - 1;
				RUNANDTIME(global_timer, 
					scoreDiff = pairwiseMatch(keyPoint, _keyPoints[last], 
						descriptor, _descriptors[last], H, matchesPoints), 
					OUTPUT, "pairwise matches.");
				if (scoreDiff.first < SIMILARITY_THRESHOLD_HULL &&
					scoreDiff.second < AREA_DIFF_THRESHOLD)
				{
					glutPostRedisplay();	
					glutMainLoopEvent();
					continue;
				}

				Mat objSet, objSetAT, modSet;		// 依次为当前帧特征点集，经转换后当前帧特征点集，前一帧特征点集
				vector<Vec2d> oldLoc, newLoc;
				bool success;
				RUNANDTIME(global_timer, 
					success = convert2DTo3D(depthGenerator, 
						H, depthImg, _depthImgPre, keyPoint, 
						_keyPoints[last], matchesPoints, oldLoc, 
						newLoc, objSet, modSet, objSetAT, _mask), 
					OUTPUT, "get 3D feature points.");
				if (!success)
				{
					glutPostRedisplay();	
					glutMainLoopEvent();
					continue;
				}

				/*ICP i(objSet, modSet);*/
				EMICP icp(objSet, modSet, 0.01, 0.00001, 0.7, 0.01);

				RUNANDTIME(global_timer, 
					icp.run(hasCuda, objSetAT), 
					OUTPUT, "run ICP.");
				incMat = icp.getFinalTransformMat();

				Mat tmpMat = identityMat4x4.clone();
				Mat incMatInv = incMat.inv();
				RUNANDTIME(global_timer, 
					BundleAdjustment::runBundleAdjustment(
					tmpMat, incMatInv, modSet, oldLoc, newLoc), 
					OUTPUT, "bundle adjustment.");
				incMat = tmpMat * incMatInv.inv();
				pose = _camPoses[recordCnt - 1].clone();
				pose = pose * incMat;

				if (last == -2)
				{
					Mat tmpMask(height, width, CV_8UC1, Scalar::all(255));
					vector<pair<int, int>> tmpMatchesPoints;
					pair<double, double> tmpScoreDiff;
					RUNANDTIME(global_timer, 
						tmpScoreDiff = pairwiseMatch(keyPoint, _keyPoints[0], 
							descriptor, _descriptors[0], H, tmpMatchesPoints),
						OUTPUT, "pairwise match in closure check");
					if (tmpScoreDiff.first < SIMILARITY_THRESHOLD_HULL && 
						tmpScoreDiff.second < AREA_DIFF_THRESHOLD)
					{	
						oldLoc.clear(); newLoc.clear();
						RUNANDTIME(global_timer, 
							success = convert2DTo3D(depthGenerator, 
								H, depthImg, depthImg0, keyPoint, 
								_keyPoints[0], tmpMatchesPoints, oldLoc, 
								newLoc, objSet, modSet, objSetAT, tmpMask), 
							OUTPUT, "get 3D feature points in closure check.");

						EMICP tmpIcp(objSet, modSet, 0.01, 0.00001, 0.7, 0.01);
						RUNANDTIME(global_timer, 
							tmpIcp.run(hasCuda, objSetAT), 
							OUTPUT, "run ICP in closure check.");
						incMat = tmpIcp.getFinalTransformMat().clone();

						tmpMat = identityMat4x4.clone();
						incMatInv = incMat.inv();
						RUNANDTIME(global_timer, 
							BundleAdjustment::runBundleAdjustment(
								tmpMat, incMatInv, modSet, oldLoc, newLoc), 
							OUTPUT, "bundle adjustment.");
						pose = _camPoses[0] * tmpMat * incMatInv.inv();

						Mat curPose = pose.clone();
						for (int i = recordCnt - 1; i > 0; i--)
						{
							Mat beforePose = _camPoses[i].clone();
							_camPoses[i] = curPose * _incPoses[i].inv();
							curPose = _camPoses[i].clone();
							Mat transformPose = curPose * beforePose.inv();	// 先将原来已经变换过的点变换回来，再做新的变换

							Mat pts;
							global_mesh.getVertices(i, pts);
							transformPointCloud(pts, 
								pts, transformPose, hasCuda);
							global_mesh.updateVertices(i, pts);
						}
						waitKey();
					}
				}
			}

			_keyPoints.push_back(keyPoint);
			_descriptors.push_back(descriptor);
			_camPoses.push_back(pose);
			_incPoses.push_back(incMat);
			recordCnt++;

			RUNANDTIME(global_timer, 
				read3DPoints(depthGenerator, depthImg, 
					colorImg, _mask, pointCloud, pointColors), 
				OUTPUT, "read 3D points");

			RUNANDTIME(global_timer, 
				transformPointCloud(pointCloud, pointCloud, pose, hasCuda), 
				OUTPUT, "transform point cloud.");
			
			_depthImgPre2 = _depthImgPre.clone();
			_depthImgPre = depthImg.clone();
			_dscrptPre = dscrpt.clone();

// 			waitKey();
// 			RUNANDTIME(global_timer, runBallPivoting(m, pointCloud, 
// 				pointColors), OUTPUT, "ball pivoting");
// 			waitKey();
			RUNANDTIME(global_timer, 
				global_mesh.addVertices(pointCloud, pointColors), 
				OUTPUT, "load data");
			if (drawType & TYPE_TRIANGLES)
			{
				RUNANDTIME(global_timer, 
					global_delaunay.computeDelaunay(global_mesh), 
					OUTPUT, "delaunay");
				cout << global_mesh.m_triangles.size() << endl;
			}
		}
// 				if (last - 2 >= 0)
// 				{
// 					Mat tmpH;
// 					vector<pair<int, int>> tmpMatchesPoints;
// 					double tmpScore = pairwiseMatch(keyPoint, keyPoints[last - 1], 
// 						descriptor, descriptors[last - 1], tmpH, tmpMatchesPoints);
// 					cout << score << " " << tmpScore << endl;
// 					if (tmpScore > score)
// 					{
// 						glutPostRedisplay();	
// 						glutMainLoopEvent();
// 						continue;
// 					}
// 				}
/*				waitKey();*/

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

// 				cout << pointCloud.rows << endl;
// 				RUNANDTIME(global_timer,
// 					features.getVariational(pointCloud, 
// 					pointColors, 1.5, pointCloud, pointColors),
// 					true, "get surface variational feature points");
// 				cout << pointCloud.rows << endl;
// 
// 				RUNANDTIME(global_timer, 
// 					mesh.addVertices(pointCloud, pointColors), 
// 					OUTPUT, "load data");

	RUNANDTIME(global_timer, 
		segment3DRBNN(global_mesh, meshs),
		OUTPUT, "segment 3D points");

	cout << global_mesh.getVerticesSize() << endl;
	glutMainLoop();
	return 0;
}
