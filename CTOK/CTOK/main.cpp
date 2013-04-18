#include <GL/freeglut.h>

#include "icp.h"
#include "emicp.h"
#include "segment.h"
#include "features.h"
#include "camera.h"
#include "triangulation.h"
#include "bundleadjustment.h"

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

//---OpenGL ȫ�ֱ���
int glWinWidth = 640, glWinHeight = 480;
int glWinPosX = 30, glWinPosY = 30;
int width = 640, height = 480;
bool breakScan = false;

Camera userCamera;

//---OpenNI
#define CONFIG_PATH "SamplesConfig.xml"

char* videoFile = "3281.oni";
XnStatus rc = XN_STATUS_OK;
Context context;							// �����Ķ���
XnDouble baseline;							// ���߳���(mm)
XnUInt64 focalLengthInPixel;				// ����(pixel)
ImageGenerator imageGenerator;				// image generator
DepthGenerator depthGenerator;				// depth generator

//----CTOK
bool hasCuda = true;						// ��û��cuda

vector<Vec3d> pointCloudData;
vector<Vec3b> pointColorData;
size_t pointNumber = 0;

Features features;							// ���Ի�ȡ�����Ƚ����ƶ�
#define SIMILARITY_THRESHOLD 0.0001			// ���ƶ���ֵ

Triangulation::Delaunay delaunay(COS30);	// ���ǻ�

// ��ȡÿһ֡�Ĳ�ɫͼ�����ͼ
void readFrame(ImageGenerator ig, DepthGenerator dg, 
	Mat &colorImg, Mat &depthImg)
{
	ImageMetaData imageMD;     // ����image�ڵ�Ԫ���ݶ���    
	DepthMetaData depthMD;     // ����depth�ڵ�Ԫ���ݶ���
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

	colorImg = colorImageShow.clone();
}

// �������ͼ����ɫͼ��mask��ȡ��ʵ����ĵ����������ɫ��Ϣ
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

// ����3D�������ݼ�����ɫ����
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

	Vec3d p;
	Vec3b color;
	for (int i = 0; i < pointCloud.rows; i ++/*= SAMPLE_INTERVAL*/)
	{
		p = pointCloud.at<Vec3d>(i, 0);
		if (p != Vec3d(0, 0, 0)/* && (double)rand() / (double)RAND_MAX < 0.5*/)
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

// ���Ƶ���
void drawPoints()
{
	double x,y,z;
	// ����ͼ�����
	glPointSize(1.0);
	glBegin(GL_POINTS);
	for (int i = 0; i < pointCloudData.size(); i++)
	{
		glColor3d(pointColorData[i][0] / 255.0, 
			pointColorData[i][1] / 255.0, pointColorData[i][2] / 255.0);
		x = pointCloudData[i][0] * 1000.0;
		y = pointCloudData[i][1] * 1000.0;
		z = pointCloudData[i][2] * 1000.0;
		glVertex3d(x, y, z);
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
			glVertex3d(v.m_xyz[0], v.m_xyz[1], -v.m_xyz[2]);
		}
	}
	glEnd(); 
}

/************************************************************************/
/*                       OpenGL��Ӧ����                                 */
/************************************************************************/

// ��갴����Ӧ����
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

// ����˶���Ӧ����
void motion(int x, int y)
{
}

// ���̰�����Ӧ����
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

// ������������Ӧ����
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

// ��άͼ����ʾ��Ӧ����
void renderScene(void)
{
	// clear screen and depth buffer
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	// Reset the coordinate system before modifying
	glLoadIdentity();   
	// set the camera position
	userCamera.look();

	drawPoints();

	Vec3d vPos = userCamera.position();
	Vec3d vView = userCamera.view();

	// ����camera����λ��
	userCamera.positionCamera(vPos, vView, Vec3d(0.0, 1.0, 0.0));

	glFlush();
	glutSwapBuffers();
}

// ���ڱ仯ͼ���ع���Ӧ����
void reshape (int w, int h)
{
	glWinWidth = w;
	glWinHeight = h;
	glViewport (0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	gluPerspective (45, (GLdouble)w / (GLdouble)h, 1.0, 15000.0);   
	glMatrixMode (GL_MODELVIEW);
}

// ��ʼ��OpenNI
void initOpenNI()
{
	// OpenNI ����
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
	rc = context.OpenFileRecording(videoFile, player);						// �����е�oni�ļ�
	checkOpenNIError(rc, "Open File Recording");

	rc = context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);		// ��ȡoni�ļ��е�image�ڵ�
	checkOpenNIError(rc, "Create Image Generator");   
	rc = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator);		// ��ȡoni�ļ��е�depth�ڵ�
	checkOpenNIError(rc, "Create Depth Generator"); 

	// ������ش�С
	XnDouble pixelSize = 0;
	rc = depthGenerator.GetRealProperty("ZPPS", pixelSize);
	checkOpenNIError(rc, "ZPPS");
	pixelSize *= 2.0;

	// ��ý��ࣨmm��
	XnUInt64 zeroPlanDistance;
	rc = depthGenerator.GetIntProperty("ZPD", zeroPlanDistance);
	checkOpenNIError(rc, "ZPD");

	// ��û��߳���(mm)
	rc = depthGenerator.GetRealProperty("LDDIS", baseline);
	checkOpenNIError(rc, "LDDIS");
	baseline *= 10;

	// ��ý���(pixel)
	focalLengthInPixel = (XnUInt64)((double)zeroPlanDistance / (double)pixelSize);

	// ��ʼ��ȡ����ʾ Kinect ͼ��
	rc = context.StartGeneratingAll();
	checkOpenNIError(rc, "Start generating");
	context.WaitNoneUpdateAll();
}

// ��ʼ��OpenGL
void initOpenGL(int argc, char** argv)
{
	userCamera.positionCamera(0.0, 1.8, 100.0, 
		0.0, 1.8, 0.0, 0.0, 1.0, 0.0);	// ��λ�����

	// OpenGL Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowPosition(glWinPosX, glWinPosY);
	glutInitWindowSize(glWinWidth, glWinHeight);
	glutCreateWindow("3D Scene");

	glutReshapeFunc (reshape);			// ���ڱ仯ʱ�ػ�ͼ��
	glutDisplayFunc(renderScene);		// ��ʾ��άͼ��       
	glutMouseFunc(mouse);				// ��갴����Ӧ
	glutEntryFunc(mouseEntry);			// ���������봰�ڵĴ�����
	glutMotionFunc(motion);				// ����ƶ���Ӧ
	glutKeyboardFunc(keyboard);			// ���̰�����Ӧ
	glutIdleFunc(renderScene);			// ����ʱ�ػ�ͼ��
}

/************************************************************************/
/*                         ������                                       */
/************************************************************************/
int main(int argc, char** argv)
{
	hasCuda = initCuda();

	initOpenGL(argc, argv);
	initOpenNI();

	// OpenCV Window
	namedWindow("depth", CV_WINDOW_AUTOSIZE);  
	namedWindow("image", CV_WINDOW_AUTOSIZE);  

	double K[9] = {(double)focalLengthInPixel, 0.0, 320,
				   0.0, -(double)focalLengthInPixel, 240,
				   0.0, 0.0, 1};
	Mat intrinsic(3, 3, CV_64FC1, K);
	BundleAdjustment::setIntrinsic(intrinsic);

	Mat depthImgPre;					// ǰһ֡�����ͼ
	Mat mask;
	pair<Mat, Mat> desPre, desNow;
	Mat dpre;

	XnUInt32 frameCnt = 0;

	int recordCnt = 0;
	vector<Mat> camPoses;				// ��¼ÿһ֡���������
	vector<vector<KeyPoint>> keyPoints;	// ��¼ÿһ֡��������
	vector<Mat>	descriptors;			// ��¼ÿһ֡������������

	for (; ; frameCnt++) 
	{
		rc = context.WaitAndUpdateAll();
		if (rc != XN_STATUS_OK)
		{
			continue;
		}

		if (breakScan || depthGenerator.GetFrameID() < frameCnt)
		{
			break;
		}

		Mat colorImg, depthImg;
		RUNANDTIME(global_timer, readFrame(imageGenerator, depthGenerator, 
			colorImg, depthImg), OUTPUT, "read one frame.");

// 		desPre = desNow;
// 		features.getHSVColorHistDes(colorImg, desNow.first);
// 		features.getGLCMDes(colorImg, desNow.second);
		Mat des;
		features.getPHash(colorImg, des);

		Mat pointCloud, pointColors, pose;
		if (frameCnt == 0)
		{
			mask.create(colorImg.rows, colorImg.cols, CV_8UC1);
			mask.setTo(Scalar::all(255));

			RUNANDTIME(global_timer, read3DPoints(depthGenerator, 
				depthImg, colorImg, mask, pointCloud, pointColors), 
				OUTPUT, "read 3D points");

			vector<KeyPoint> keyPoint;
			Mat descriptor;
			RUNANDTIME(global_timer, get2DFeaturePoints(colorImg, keyPoint, 
				descriptor), OUTPUT, "get feature points and descriptor.");
			keyPoints.push_back(keyPoint);
			descriptors.push_back(descriptor);

			pose = Mat::eye(4, 4, CV_64FC1);
			camPoses.push_back(pose);
			recordCnt++;

			depthImgPre = depthImg.clone();
			dpre = des.clone();
		}
		else
		{
//			cout << hammingDistance(dpre, dnow) << endl;
// 			int dist = hammingDistance(dpre, des);
// 			cout << dist << endl;
// 			if (dist > 5)
// 			{
//			}
// 			waitKey();
// 			continue;
			vector<KeyPoint> keyPoint;
			Mat descriptor, H;
			RUNANDTIME(global_timer, get2DFeaturePoints(colorImg, keyPoint, 
				descriptor), OUTPUT, "get feature points and descriptor.");

			vector<vector<KeyPoint>> tmpKpts;
			vector<Mat> tmpDscrpts;
			tmpKpts.push_back(keyPoints[recordCnt - 1]);
			tmpKpts.push_back(keyPoint);
			tmpDscrpts.push_back(descriptors[recordCnt - 1]);
			tmpDscrpts.push_back(descriptor);

			vector<pair<int, int>> matchesPoints;
			double score;
			RUNANDTIME(global_timer, score = pairwiseMatch(1, 
				0, tmpKpts, tmpDscrpts, H, matchesPoints), 
				OUTPUT, "pairwise matches.");
			if (score < 0.55)
			{
// 				vector<KeyPoint> keyPoint;
// 				Mat descriptor, H;
// 				RUNANDTIME(global_timer, get2DFeaturePoints(colorImg, keyPoint, 
// 					descriptor), OUTPUT, "get feature points and descriptor.");
				keyPoints.push_back(keyPoint);
				descriptors.push_back(descriptor);
// 
// 				vector<pair<int, int>> matchesPoints;
// 				double score;
// 				RUNANDTIME(global_timer, score = pairwiseMatch(1, 
// 					0, keyPoints, descriptors, H, matchesPoints), 
// 					OUTPUT, "pairwise matches.");

				Mat objSet, objSetAT, modSet;		// ����Ϊ��ǰ֡�����㼯����ת����ǰ֡�����㼯��ǰһ֡�����㼯
				vector<Vec2d> oldLoc, newLoc;

				bool flag;
				RUNANDTIME(global_timer, flag = convert2DTo3D(depthGenerator, 
					H, depthImg, depthImgPre, recordCnt, recordCnt - 1, 
					keyPoints, matchesPoints, oldLoc, newLoc, objSet, 
					modSet, objSetAT, mask), OUTPUT, "get 3D feature points.");
// 				if (!flag || objSet.empty() || modSet.empty() || objSetAT.empty())
// 				{
// 					glutPostRedisplay();	
// 					glutMainLoopEvent();
// 					continue;
// 				}

				/*ICP i(objSet, modSet);*/
				EMICP i(objSet, modSet, 0.01, 0.00001, 0.7, 0.01);

				RUNANDTIME(global_timer, 
					i.run(hasCuda, objSetAT), OUTPUT, "run ICP.");
// 				Mat incMat = i.getFinalTransformMat();
// 				pose = camPoses[recordCnt - 1].clone();
// 				pose = pose * incMat;
// 				camPoses.push_back(pose);
// 				recordCnt++;

				Mat tmpMat = Mat::eye(4, 4, CV_64FC1);
				Mat incMat = i.getFinalTransformMat().inv();
				RUNANDTIME(global_timer, 
					BundleAdjustment::runBundleAdjustment(tmpMat, incMat,
					modSet, oldLoc, newLoc), OUTPUT, "bundle adjustment.");
				pose = camPoses[recordCnt - 1].clone();
				pose = pose * tmpMat * incMat.inv();
				camPoses.push_back(pose);
				recordCnt++;

// 				vector<vector<pair<int, int>>> matchesPairs;
// 				RUNANDTIME(global_timer, 
// 					fullMatch(recordCnt - 1, descriptors, 
// 					keyPoints, matchesPairs), OUTPUT, "full match.");
// 				Mat points;
// 				RUNANDTIME(global_timer, 
// 					convert2DTo3D(depthGenerator, depthImg, 
// 					keyPoint, points), OUTPUT, "get 3D feature points.");
// 				RUNANDTIME(global_timer, 
// 					BundleAdjustment::runBundleAdjustment(camPoses, points, 
// 					keyPoints, matchesPairs), OUTPUT, "bundle adjustment.");
// 				pose = camPoses[recordCnt - 1].clone();

				RUNANDTIME(global_timer, 
					read3DPoints(depthGenerator, depthImg, colorImg, mask, 
					pointCloud, pointColors), OUTPUT, "read 3D points");
				RUNANDTIME(global_timer, 
					transformPointCloud(pointCloud, pointCloud, 
					pose, hasCuda), OUTPUT, "transform point cloud.");

				depthImgPre = depthImg.clone();
				dpre = des.clone();
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
/*			delaunay.saveTriangles("triangles.tri");*/
		}

		char key = waitKey(1);
		if (key == 27)
		{
			break;
		}

		glutPostRedisplay();	// ˢ�»���
		glutMainLoopEvent();	// OpenCV ���������Ӧ��Ϣ������ʾ OpenGL ͼ��
	}

	// destroy  
	destroyAllWindows();
	context.StopGeneratingAll();
	context.Release();

	cout << pointNumber << endl;
	glutMainLoop();
	return 0;
}
