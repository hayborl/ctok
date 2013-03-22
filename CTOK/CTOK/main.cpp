#include <GL/freeglut.h>

#include "icp.h"
#include "emicp.h"
#include "segment.h"
#include "features.h"
#include "camera.h"

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

#define SIGN(x) ((x) < 0 ? -1 : ((x) > 0 ? 1 : 0))

//////////////////////////////////////////////////////////////////////////
//
//---OpenGL ȫ�ֱ���
// float xyzdata[480][640][3];
// float texture[480][640][3];
vector<Vec3f> pointCloudData;
vector<Vec3b> pointColorData;
size_t pointNumber = 0;
int glWinWidth = 640, glWinHeight = 480;
int width = 640, height = 480;
double eyex, eyey, eyez, atx, aty, atz;				// eye* - �����λ�ã�at* - ע�ӵ�λ��

bool leftClickHold = false, rightClickHold = false;	//������Ҽ��Ƿ񱻰�ס
int mx, my;											// ��갴��ʱ�� OpenGL ���ڵ�����
int ry = 90, rx = 90;								// ��������ע�ӵ�Ĺ۲�Ƕ�
double mindepth, maxdepth;							// ������ݵļ�ֵ
double radius = 6000.0;								// �������ע�ӵ�ľ���
Camera userCamera;

Features features;
#define SIMILARITY_THRESHOLD 0.001

#define SAMPLE_INTERVAL 1

void readFrame(ImageGenerator ig, DepthGenerator dg, 
	Mat& colorImg, Mat& depthImg)
{
	ImageMetaData imageMD;     //����image�ڵ�Ԫ���ݶ���    
	DepthMetaData depthMD;     //����depth�ڵ�Ԫ���ݶ���
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

void read3DPoints(DepthGenerator dg, const Mat& depthImg, 
	const Mat& colorImg, Mat& realPointCloud, 
	Mat& pointColors, Mat& pointIndices)
{
	int cols, rows, index = 0;
	cols = depthImg.cols;
	rows = depthImg.rows;

	Ptr<XnPoint3D> proj = new XnPoint3D[cols * rows];
	Ptr<XnPoint3D> real = new XnPoint3D[cols * rows];
	vector<Vec3b> colors;
	pointIndices = Mat(rows, cols, CV_32SC1, Scalar::all(-1));
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			ushort z = depthImg.at<ushort>(y, x);
			if (z != 0)
			{
				proj[index].X = (float)x;
				proj[index].Y = (float)y;
				proj[index].Z = (float)z;
				pointIndices.at<int>(y, x) = index;
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

// ����3D�������ݼ�����ɫ����
void loadPointCloudAndTexture(Mat pointCloud, 
	Mat pointColor, bool clear = false)
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

	ANNpointArray pts = NULL;
	ANNkd_tree* kdTree;
// 	if (pointCloudData.size() != 0)
// 	{
// 		pts = annAllocPts(pointCloudData.size(), 3);
// #pragma omp parallel for
// 		for (int r = 0; r < pointCloudData.size(); r++)
// 		{
// 			Vec3f p = pointCloudData[r];
// 			pts[r][0] = p[0];
// 			pts[r][1] = p[1]; 
// 			pts[r][2] = p[2];
// 		}
// 		kdTree = new ANNkd_tree(pts, pointCloudData.size(), 3);
// 	}

	Vec3f p;
	Vec3b color;
	for (int i = 0; i < pointCloud.rows; i ++/*= SAMPLE_INTERVAL*/)
	{
		p = pointCloud.at<Vec3f>(i, 0);
		if (p != Vec3f(0, 0, 0)/* && (double)rand() / (double)RAND_MAX < 0.5*/)
		{
			p[2] = -p[2];
			if (pts != NULL)
			{
				ANNpoint pt = annAllocPt(3);
				pt[0] = p[0];
				pt[1] = p[1];
				pt[2] = p[2];
				if (kdTree->annkFRSearch(pt, DISTANCE_RANGE, 0) > 0)
				{
					annDeallocPt(pt);
					continue;
				}
				annDeallocPt(pt);
			}
			pointCloudData.push_back(p);
			color = pointColor.at<Vec3b>(i, 0);
			pointColorData.push_back(
				Vec3b(color[2], color[1], color[0]));
		}
	}

	pointNumber = pointCloudData.size();

	cout << pointNumber << " OK" << endl;

	if (pts != NULL)
	{
		annDeallocPts(pts);
	}
}

// ���Ƶ���
void drawPoints()
{
	float x,y,z;
	// ����ͼ�����
	glPointSize(1.0);
	glBegin(GL_POINTS);
	for (int i = 0; i < pointCloudData.size(); i++/*+= (int)radius / 10*/)
	{
// 			if (rand() < RAND_MAX * radius / (pointCloudData.size() * 100))
// 			{
// 				continue;
// 			}
		glColor3d(pointColorData[i][0] / 255.0, 
			pointColorData[i][1] / 255.0, pointColorData[i][2] / 255.0);
		x = (float)pointCloudData[i][0];
		y = (float)pointCloudData[i][1];
		z = (float)pointCloudData[i][2];
		glVertex3f(x, y, z);
	}
	glEnd(); 
}

/************************************************************************/
/*                       OpenGL��Ӧ����                                 */
/************************************************************************/

//////////////////////////////////////////////////////////////////////////
// ��갴����Ӧ����
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
// ����˶���Ӧ����
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
		userCamera.setMouseState(true);
		ShowCursor(false);
		break;
	}
}

//////////////////////////////////////////////////////////////////////////
// ��άͼ����ʾ��Ӧ����
void renderScene(void)
{
	// clear screen and depth buffer
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	// Reset the coordinate system before modifying
	glLoadIdentity();   
	// set the camera position
// 	atx = 0.0f;
// 	aty = 0.0f;
// 	atz = ( mindepth - maxdepth ) / 2.0f;
// 	eyex = atx + radius * sin( CV_PI * ry / 180.0f ) * cos( CV_PI * rx/ 180.0f );
// 	eyey = aty + radius * cos( CV_PI * ry/ 180.0f );
// 	eyez = atz + radius * sin( CV_PI * ry / 180.0f ) * sin( CV_PI * rx/ 180.0f );
// 	eyex *= 0.5;
// 	eyey *= 0.5;
// 	eyez *= 0.5;
// 	gluLookAt (eyex, eyey, eyez, atx, aty, atz, 0.0, 1.0, 0.0);
	userCamera.look();

	// �Ե������ݽ������ǻ�
	// �ο��ԣ�http://www.codeproject.com/KB/openGL/OPENGLTG.aspx

	drawPoints();

	Vec3f vPos = userCamera.position();
	Vec3f vView = userCamera.view();

	// ����camera����λ��
	userCamera.positionCamera(vPos, vView, Vec3f(0.0f, 1.0f, 0.0f));

	glFlush();
	glutSwapBuffers();
}

//////////////////////////////////////////////////////////////////////////
// ���ڱ仯ͼ���ع���Ӧ����
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
/*                         ������                                       */
/************************************************************************/
int main(int argc, char** argv)
{
	hasCuda = initCuda();
	userCamera.positionCamera(0.0f, 1.8f, 100.0f, 0.0f, 1.8f, 0.0f, 0.0f, 1.0f, 0.0f);

	// OpenGL Window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowPosition(30, 30);
	glutInitWindowSize(glWinWidth, glWinHeight);
	glutCreateWindow("3D image");

	glutReshapeFunc (reshape);			// ���ڱ仯ʱ�ػ�ͼ��
	glutDisplayFunc(renderScene);		// ��ʾ��άͼ��       
	glutMouseFunc(mouse);				// ��갴����Ӧ
	glutEntryFunc(mouseEntry);			// ���������봰�ڵĴ�����
	glutMotionFunc(motion);				// ����ƶ���Ӧ
	glutKeyboardFunc(keyboard);			// ���̰�����Ӧ
	glutIdleFunc(renderScene);			// ����ʱ�ػ�ͼ��

	// OpenCV Window
	namedWindow("depth", CV_WINDOW_AUTOSIZE);  
	namedWindow("image", CV_WINDOW_AUTOSIZE);  

	// OpenNI ����
	XnStatus rc = XN_STATUS_OK;
	Context context;				//���������Ķ���
	rc = context.Init();			//�����Ķ����ʼ�� 
	checkOpenNIError(rc, "initialize context");

	Player player;
	rc = context.OpenFileRecording("ttt.oni", player);						//�����е�oni�ļ�
	checkOpenNIError(rc, "Open File Recording");

	ImageGenerator imageGenerator;											//����image generator
	rc = context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator);		//��ȡoni�ļ��е�image�ڵ�
	checkOpenNIError(rc, "Create Image Generator");   
	DepthGenerator depthGenerator;											//����depth generator
	rc = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator);		//��ȡoni�ļ��е�depth�ڵ�
	checkOpenNIError(rc, "Create Depth Generator"); 
	depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);

	// ��ʼ��ȡ����ʾ Kinect ͼ��
	rc = context.StartGeneratingAll();
	rc = context.WaitNoneUpdateAll();

	Mat pointCloudPre, pointCloudNow;
	Mat pointIndicesPre, pointIndicesNow;
	Mat colorImgPre, colorImgNow;
	Mat tr = Mat::eye(4, 4, CV_32FC1);
	pair<Mat, Mat> desPre, desNow;

	XnUInt32 frameCnt = 1;
	while (1)   
	{
		rc = context.WaitOneUpdateAll(depthGenerator);
		rc = context.WaitOneUpdateAll(imageGenerator);

		if (depthGenerator.GetFrameID() < frameCnt)
		{
			break;
		}

		Mat colorImg, depthImg, realPointCloud, pointColors, pointIndices;
		RUNANDTIME(global_timer, readFrame(imageGenerator, depthGenerator, 
			colorImg, depthImg), OUTPUT, "read one frame.");
		colorImgPre = colorImgNow.clone();
		colorImgNow = colorImg.clone();

		desPre = desNow;
		features.getHSVColorHistDes(colorImgNow, desNow.first);
		features.getGLCMDes(colorImgNow, desNow.second);

		if (frameCnt != 1)
		{
			double distance = computeDistance(desPre, desNow);
			if (distance < SIMILARITY_THRESHOLD)
			{
				colorImgNow = colorImgPre.clone();
				desNow = desPre;

				char key = waitKey(1);
				if (key == 27)
				{
					break;
				}
				frameCnt++;

				glutPostRedisplay();
				glutMainLoopEvent();
				continue;
			}
		}

		RUNANDTIME(global_timer, read3DPoints(depthGenerator, 
			depthImg, colorImg, realPointCloud, pointColors, 
			pointIndices), OUTPUT, "read 3D points");

// 		RUNANDTIME(global_timer, simplifyPoints(realPointCloud, 
// 			realPointCloud, 10, 0.9), OUTPUT, "simplify");

		if (realPointCloud.rows <= 0 || pointColors.rows <= 0 ||
			pointIndices.rows <= 0)
		{
			char key = waitKey(1);
			if (key == 27)
			{
				break;
			}
			frameCnt++;

			glutPostRedisplay();				// ˢ�»���

			// OpenCV ���������Ӧ��Ϣ������ʾ OpenGL ͼ��
			glutMainLoopEvent();
			continue;
		}

		pointCloudPre = pointCloudNow.clone();
		pointCloudNow = realPointCloud.clone();
		pointIndicesPre = pointIndicesNow.clone();
		pointIndicesNow = pointIndices.clone();

		if (frameCnt != 1)
		{
			Mat objSetOrigin, objSet, modSet;
			RUNANDTIME(global_timer, getSurfPointsSet(colorImgNow, 
				pointCloudNow, pointIndicesNow, colorImgPre, 
				pointCloudPre, pointIndicesPre, objSetOrigin, 
				objSet, modSet, depthGenerator), 
				OUTPUT, "get feature points.");

// 			ICP i(pointCloudNow, pointCloudPre);
// 			RUNANDTIME(global_timer, i.run(), OUTPUT, "run ICP.");
			//ICP i(objSetOrigin, modSet);
			EMICP i(objSetOrigin, modSet, 0.01f, 0.00001f, 0.7f, 0.01f);

			RUNANDTIME(global_timer, 
				i.run(hasCuda, &objSet), OUTPUT, "run ICP.");
			tr = i.getFinalTransformMat().clone() * tr;

			RUNANDTIME(global_timer, 
				transformPointCloud(realPointCloud, 
				&realPointCloud, tr, hasCuda), OUTPUT, "transform point cloud.");

/*			waitKey();*/
		}

		RUNANDTIME(global_timer, loadPointCloudAndTexture(realPointCloud, 
			pointColors, false), OUTPUT, "load data");

		char key = waitKey(1);
		if (key == 27)
		{
			break;
		}

		frameCnt++;

		glutPostRedisplay();				// ˢ�»���

		// OpenCV ���������Ӧ��Ϣ������ʾ OpenGL ͼ��
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