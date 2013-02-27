#include <GL/freeglut.h>
#include "XnCppWrapper.h"
#include "opencv2/opencv.hpp"
#include <iostream>

#include "icp.h"
#include "emicp.h"

using namespace std;
using namespace cv;
using namespace xn;
using namespace KDTree_R;

void checkOpenNIError(XnStatus result, string status)  
{   
	if (result != XN_STATUS_OK)
	{
		cerr << status << " Error: " << xnGetStatusString(result) << endl;  
		exit(-1);
	}
} 

#define SIGN(x) ((x) < 0 ? -1 : ((x) > 0 ? 1 : 0))

//////////////////////////////////////////////////////////////////////////
//
//---OpenGL 全局变量
// float xyzdata[480][640][3];
// float texture[480][640][3];
vector<_Examplar> kPointCloudData;
vector<Vec3f> cPointCloudData;
vector<Vec3b> pointColorData;
size_t pointNumber = 0;
int glWinWidth = 640, glWinHeight = 480;
int width = 640, height = 480;
double eyex, eyey, eyez, atx, aty, atz;				// eye* - 摄像机位置，at* - 注视点位置

bool leftClickHold = false, rightClickHold = false;	//鼠标左右键是否被按住
int mx, my;											// 鼠标按键时在 OpenGL 窗口的坐标
int ry = 90, rx = 90;								// 摄像机相对注视点的观察角度
double mindepth, maxdepth;							// 深度数据的极值
double radius = 6000.0;								// 摄像机与注视点的距离

KDTree_R::KDTree kdTree;

XnDouble baseline;
XnUInt64 focalLengthInPixel;

#define SAMPLE_INTERVAL 1

void readFrame(ImageGenerator ig, DepthGenerator dg, 
	Mat* colorImg, Mat* realPointCloud, 
	Mat* pointColors, Mat* pointIndices)
{
	ImageMetaData imageMD;     //创建image节点元数据对象    
	DepthMetaData depthMD;     //创建depth节点元数据对象
	Mat depthImageShow, colorImageShow;
	int cols, rows, index = 0;

	//get meta data  
	dg.GetMetaData(depthMD);   
	ig.GetMetaData(imageMD);

	cols = depthMD.XRes();
	rows = depthMD.YRes();

	//OpenCV output
	Mat imgDepth16U(rows, cols, CV_16UC1, (void*)depthMD.Data());

	// 	int size = 7, iter = 3;
	// 	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(size, size));
	// 	Point anchor = Point(size / 2, size / 2);
	// 	dilate(imgDepth16U, imgDepth16U, kernel, anchor, iter);	//膨胀
	// 	erode(imgDepth16U, imgDepth16U, kernel, anchor, iter);	//腐蚀

	imgDepth16U.convertTo(depthImageShow, CV_8UC1, 255 / 4096.0);
	Mat imgRGB(rows, cols, CV_8UC3, (void*)imageMD.Data());
	cvtColor(imgRGB, colorImageShow, CV_RGB2BGR);
	imshow("depth", depthImageShow);
	imshow("image", colorImageShow);

	Ptr<XnPoint3D> proj = new XnPoint3D[cols * rows];
	Ptr<XnPoint3D> real = new XnPoint3D[cols * rows];
	vector<Vec3b> colors;
	Mat pointIndex(rows, cols, CV_32SC1, Scalar::all(-1));
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			ushort z = imgDepth16U.at<ushort>(y, x);
			if (z != 0)
			{
				proj[index].X = (float)x;
				proj[index].Y = (float)y;
				proj[index].Z = (float)z;
				pointIndex.at<int>(y, x) = index;
				colors.push_back(colorImageShow.at<Vec3b>(y, x));

				index++;
			}
		}
	}

	dg.ConvertProjectiveToRealWorld(index, proj, real);

	Mat pointCloud_XYZ(index, 1, DataType<Point3f>::type, Scalar::all(0));
#pragma omp parallel for
	for (int i = 0; i < index; i++)
	{
		pointCloud_XYZ.at<Point3f>(i, 0) = Point3f(
			real[i].X, real[i].Y, real[i].Z);
	}
	*realPointCloud = pointCloud_XYZ.clone();
	*colorImg = colorImageShow.clone();
	*pointColors = Mat(colors, true).clone();
	*pointIndices = pointIndex.clone();
}

// 载入3D坐标数据及其颜色数据
void loadPointCloudAndTexture(Mat pointCloud, 
	Mat pointColor, bool clear = false)
{
	assert(pointCloud.rows == pointColor.rows && 
		pointCloud.cols == pointColor.cols);
	assert(pointCloud.cols == 1);
	if (clear)
	{
		cPointCloudData.clear();
		kPointCloudData.clear();
		pointColorData.clear();
		pointNumber = 0;
	}

	if (hasCuda)
	{
		float* pSet = NULL;
		if (!cPointCloudData.empty())
		{
			size_t size = cPointCloudData.size() * 3;
			pSet = new float[size];
			memcpy(pSet, &cPointCloudData[0], size * sizeof(float));
		}
		float* oSet = new float[pointCloud.rows * 3];
		memcpy(oSet, (float*)pointCloud.data, 
			pointCloud.rows * 3 * sizeof(float));

		cuda_pushBackPoint(pSet, oSet, cPointCloudData.size(), 
			pointCloud.rows, pointColor, 
			cPointCloudData, pointColorData);

		if (pSet != NULL)
		{
			delete[] pSet;
			pSet = NULL;
		}
		delete[] oSet;

		pointNumber = cPointCloudData.size();
	}
	else
	{
		kdTree.destroy();
		vector<_Examplar> pointCloudCopy(kPointCloudData);
		ExamplarSet exmSet(pointCloudCopy, (int)pointCloudCopy.size(), ICP_DIMS);
		kdTree.create(exmSet);

		Point3f p;
		Vec3b color;
		_Examplar exm(3);
		for (int i = 0; i < pointCloud.rows; i ++/*= SAMPLE_INTERVAL*/)
		{
			p = pointCloud.at<Point3f>(i, 0);
			if (p != Point3f(0, 0, 0))
			{
				exm[0] = p.x; exm[1] = p.y; exm[2] = -p.z;
				if (kdTree.findNearest(exm).second > DISTANCE_RANGE)
				{
					kPointCloudData.push_back(exm);
					color = pointColor.at<Vec3b>(i, 0);
					pointColorData.push_back(
						Vec3b(color[2], color[1], color[0]));
				}
			}
		}

		pointNumber = kPointCloudData.size();
	}

	cout << pointNumber << " OK" << endl;
}

// 绘制点云
void drawPoints()
{
	float x,y,z;
	// 绘制图像点云
	glPointSize(1.0);
	glBegin(GL_POINTS);
	if (hasCuda)
	{
		for (int i = 0; i < cPointCloudData.size(); i++/*+= (int)radius / 10*/)
		{
// 			if (rand() < RAND_MAX * radius / (pointCloudData.size() * 100))
// 			{
// 				continue;
// 			}
			glColor3d(pointColorData[i][0] / 255.0, 
				pointColorData[i][1] / 255.0, pointColorData[i][2] / 255.0);
			x = cPointCloudData[i][0];
			y = cPointCloudData[i][1];
			z = cPointCloudData[i][2];
			glVertex3f(x, y, z);
		}
	}
	else
	{
		for (int i = 0; i < kPointCloudData.size(); i++/*+= (int)radius / 10*/)
		{
// 			if (rand() < RAND_MAX * radius / (pointCloudData.size() * 100))
// 			{
// 				continue;
// 			}
			glColor3d(pointColorData[i][0] / 255.0, 
				pointColorData[i][1] / 255.0, pointColorData[i][2] / 255.0);
			x = (float)kPointCloudData[i][0];
			y = (float)kPointCloudData[i][1];
			z = (float)kPointCloudData[i][2];
			glVertex3f(x, y, z);
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
	if(button == GLUT_LEFT_BUTTON)
	{
		if(state == GLUT_DOWN)
		{
			leftClickHold=true;
		}
		else
		{
			leftClickHold=false;
		}
	}

	if (button== GLUT_RIGHT_BUTTON)
	{
		if(state == GLUT_DOWN)
		{
			rightClickHold=true;
		}
		else
		{
			rightClickHold=false;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
// 鼠标运动响应函数
void motion(int x, int y)
{
	int rstep = 5;
	if(leftClickHold==true)
	{
		if( abs(x-mx) > abs(y-my) )
		{
			rx += SIGN(x-mx)*rstep;   
		}
		else
		{
			ry -= SIGN(y-my)*rstep;   
		}

		mx=x;
		my=y;
		glutPostRedisplay();
	}

	if(rightClickHold==true)
	{
		radius += SIGN(y-my)*100.0;
		radius = std::max( radius, 100.0 );
		mx=x;
		my=y;
		glutPostRedisplay();
	}
}

void keyboard(uchar key, int x, int y)
{
	switch(key)
	{
	case 27:
		exit(0);
		break;
	default:
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
	atx = 0.0f;
	aty = 0.0f;
	atz = ( mindepth - maxdepth ) / 2.0f;
	eyex = atx + radius * sin( CV_PI * ry / 180.0f ) * cos( CV_PI * rx/ 180.0f );
	eyey = aty + radius * cos( CV_PI * ry/ 180.0f );
	eyez = atz + radius * sin( CV_PI * ry / 180.0f ) * sin( CV_PI * rx/ 180.0f );
	eyex *= 0.5;
	eyey *= 0.5;
	eyez *= 0.5;
	gluLookAt (eyex, eyey, eyez, atx, aty, atz, 0.0, 1.0, 0.0);

	// 对点云数据进行三角化
	// 参考自：http://www.codeproject.com/KB/openGL/OPENGLTG.aspx

	drawPoints();

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
/*                          colorizeDisparity                           */
/*                      将视差图由灰度图转换为伪彩色图                  */
/************************************************************************/
void colorizeDisparity( const Mat& gray0, Mat& rgb, double maxDisp=-1.f, float S=1.f, float V=1.f )
{
	CV_Assert( !gray0.empty() );
	Mat gray;
	if (gray0.type() == CV_32FC1)
	{
		gray0.convertTo( gray, CV_8UC1 );
	}
	else if (gray0.type() == CV_8UC1)
	{
		gray0.copyTo(gray);
	}
	else
	{
		return;
	}

	if( maxDisp <= 0 )
	{
		maxDisp = 0;
		minMaxLoc( gray, 0, &maxDisp );
	}

	rgb.create( gray.size(), CV_8UC3 );
	rgb = Scalar::all(0);
	if( maxDisp < 1 )
		return;

	for( int y = 0; y < gray.rows; y++ )
	{
		for( int x = 0; x < gray.cols; x++ )
		{
			uchar d = gray.at<uchar>(y,x);
			unsigned int H = ((uchar)maxDisp - d) * 240 / (uchar)maxDisp;

			unsigned int hi = (H/60) % 6;
			float f = H/60.f - H/60;
			float p = V * (1 - S);
			float q = V * (1 - f * S);
			float t = V * (1 - (1 - f) * S);

			Point3f res;

			if( hi == 0 ) //R = V,   G = t,   B = p
				res = Point3f( p, t, V );
			if( hi == 1 ) // R = q,   G = V,   B = p
				res = Point3f( p, V, q );
			if( hi == 2 ) // R = p,   G = V,   B = t
				res = Point3f( t, V, p );
			if( hi == 3 ) // R = p,   G = q,   B = V
				res = Point3f( V, q, p );
			if( hi == 4 ) // R = t,   G = p,   B = V
				res = Point3f( V, p, t );
			if( hi == 5 ) // R = V,   G = p,   B = q
				res = Point3f( q, p, V );

			uchar b = (uchar)(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
			uchar g = (uchar)(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
			uchar r = (uchar)(std::max(0.f, std::min (res.z, 1.f)) * 255.f);

			rgb.at<Point3_<uchar> >(y,x) = Point3_<uchar>(b, g, r);     
		}
	}
}

/************************************************************************/
/*                          saveData                                    */
/*                      保存当前画面的数据                              */
/************************************************************************/


/************************************************************************/
/*                         主程序                                       */
/************************************************************************/
int main(int argc, char** argv)
{
	hasCuda = initCuda();

	// OpenGL Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowPosition(10, 320);
	glutInitWindowSize(glWinWidth, glWinHeight);
	glutCreateWindow("3D image");

	glutReshapeFunc (reshape);			// 窗口变化时重绘图像
	glutDisplayFunc(renderScene);		// 显示三维图像       
	glutMouseFunc(mouse);				// 鼠标按键响应
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
	rc = context.OpenFileRecording("test.oni", player);						//打开已有的oni文件
	checkOpenNIError(rc, "Open File Recording");

	ImageGenerator imageGenerator;											//创建image generator
	rc = context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);		//获取oni文件中的image节点
	checkOpenNIError(rc, "Create Image Generator");   
	DepthGenerator depthGenerator;											//创建depth generator
	rc = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator);		//获取oni文件中的depth节点
	checkOpenNIError(rc, "Create Depth Generator"); 
	depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);

	// 	unsigned int frameNum;
	// 	player.GetNumFrames(depthGenerator.GetName(), frameNum);

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
	rc = context.WaitNoneUpdateAll();

	Mat pointCloudPre, pointCloudNow;
	Mat pointIndicesPre, pointIndicesNow;
	Mat colorImgPre, colorImgNow;
	Mat tr = Mat::eye(4, 4, CV_32FC1);

	XnUInt32 frameCnt = 1;
	while (1)   
	{
		rc = context.WaitOneUpdateAll(depthGenerator);
		rc = context.WaitOneUpdateAll(imageGenerator);

		if (depthGenerator.GetFrameID() < frameCnt/* || frameCnt == 166*/)
		{
			break;
		}

		// 		char xyzName[20];
		// 		sprintf(xyzName, "xyz%d.xyz", frameCnt);

		Mat colorImg, realPointCloud, pointColors, pointIndices;
		RUNANDTIME(global_timer, readFrame(imageGenerator, depthGenerator, 
			&colorImg, &realPointCloud, &pointColors, &pointIndices), 
			OUTPUT, "read one frame.");

		if (realPointCloud.rows <= 0 || pointColors.rows <= 0 ||
			pointIndices.rows <= 0)
		{
			continue;
		}

		colorImgPre = colorImgNow.clone();
		colorImgNow = colorImg.clone();
		pointCloudPre = pointCloudNow.clone();
		pointCloudNow = realPointCloud.clone();
		pointIndicesPre = pointIndicesNow.clone();
		pointIndicesNow = pointIndices.clone();

// 		pointCloudPre = pointCloudNow.clone();
// 		RUNANDTIME(global_timer, 
// 			pointCloudNow = getFeaturePointCloud(colorImg, 
// 			realPointCloud, pointIndices).clone(), 
// 			OUTPUT, "get feature points.");
// 		pointCloudPre = pointCloudNow.clone();
// 		pointCloudNow = realPointCloud.clone();

		if (frameCnt != 1)
		{
// 			if (frameCnt % 2 == 1)
// 			{
// 				pointCloudPre = pointCloudNow.clone();
// 				RUNANDTIME(global_timer, pointCloudNow = getFeaturePointCloud(colorImg, 
// 					realPointCloud, pointIndices).clone(), 
// 					OUTPUT, "get feature points.");
// 
// 				if (frameCnt / 2 != 0)
// 				{
// 					ICP i(pointCloudNow, pointCloudPre);
// 					RUNANDTIME(timer, i.run(), OUTPUT, "run ICP.");
// 
// 					tr = i.getTransformMat().clone() * tr;
// // 					RUNANDTIME(global_timer, transformPointCloud(&realPointCloud, tr), 
// // 						OUTPUT, "transform point cloud.");
// // 					saveData("point3.txt", realPointCloud, 3);
// 					RUNANDTIME(global_timer, transformPCUsingCUDA(&realPointCloud, tr), 
// 						OUTPUT, "transform point cloud.");
// 					waitKey();
// 				}
// 
// 				loadPointCloudAndTexture(realPointCloud, pointColors, false);
// 			}
			Mat objSetOrigin, objSet, modSet;
			RUNANDTIME(global_timer, getSurfPointsSet(colorImgNow, 
				pointCloudNow, pointIndicesNow, colorImgPre, 
				pointCloudPre, pointIndicesPre, &objSetOrigin, 
				&objSet, &modSet, depthGenerator), 
				OUTPUT, "get feature points.");

// 			ICP i(pointCloudNow, pointCloudPre);
// 			RUNANDTIME(global_timer, i.run(), OUTPUT, "run ICP.");
			ICP i(objSetOrigin, modSet);
/*			EMICP i(objSetOrigin, modSet, 0.01f, 0.00001f, 0.7f, 0.01f);*/
			RUNANDTIME(global_timer, i.run(&objSet), OUTPUT, "run ICP.");

			tr = i.getTransformMat().clone() * tr;
			if (hasCuda)
			{
				RUNANDTIME(global_timer, 
					cuda_transformPointCloud(realPointCloud, 
					&realPointCloud, tr), OUTPUT, "transform point cloud.");
			}
			else
			{
				RUNANDTIME(global_timer, transformPointCloud(realPointCloud, 
					&realPointCloud, tr), OUTPUT, "transform point cloud.");
			}
/*			cout << tr << endl;*/
		}

		RUNANDTIME(global_timer, loadPointCloudAndTexture(realPointCloud, 
			pointColors, true), OUTPUT, "load data");
/*		waitKey();*/

		char key = waitKey(1);
		if (key == 27)
		{
			break;
		}
		frameCnt++;

		//////////////////////////////////////////////////////////////////////////
		// OpenGL显示
		//         load3dDataToGL(pointCloud_XYZ);     // 载入环境三维点云
		//         loadTextureToGL(imgBGR);			// 载入纹理数据

		glutPostRedisplay();				// 刷新画面

		// OpenCV 处理键盘响应消息后，再显示 OpenGL 图像
		glutMainLoopEvent();
	}

	// destroy  
	destroyAllWindows();
	context.StopGeneratingAll();
	context.Release();

	cout << pointNumber << endl;
	glutMainLoop();
	return 0;
}
