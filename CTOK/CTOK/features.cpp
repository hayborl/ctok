#include "features.h"

#include "opencv2/nonfree/nonfree.hpp"

using namespace gpu;

extern bool hasCuda;

void Features::getHSVColorHistDes( const Mat &image, Mat &descriptors )
{
	assert(image.channels() == 3);

	int rows = image.rows;
	int cols = image.cols;

	Mat tmp(rows, cols, image.type());
	cvtColor(image, tmp, CV_BGR2HSV);

	int hist[72];
	memset(hist, 0, 72 * sizeof(int));

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			Vec3b scalar = tmp.at<Vec3b>(i, j);
			double h = scalar[0] * 2;
			double s = scalar[1] / 255.0;
			double v = scalar[2] / 255.0;

			int H = convertH(h);
			int S = convertS(s);
			int V = convertV(v);

			int G = 9 * H + 3 * S + V;
			hist[G]++;
		}
	}

	int n = rows * cols;
	descriptors = Mat(1, 72, CV_64FC1);
	for (int i = 0; i < 72; i++)
	{
		double temp = (double)hist[i] / (double)n;
		descriptors.at<double>(0, i) = temp;
	}
}

void Features::getGLCMDes( const Mat &image, Mat &descriptors )
{
	double normGlcm[GRAY_LEVEL * GRAY_LEVEL];

	descriptors = Mat(4, 4, CV_64FC1);
	for (int i = 0; i < 4; i++)
	{
		getGLCM(image, normGlcm, i, 1);

		double glcm_asm = 0, glcm_ent = 0, 
			glcm_con = 0, glcm_idm = 0;
		for (int j = 0; j < GRAY_LEVEL * GRAY_LEVEL; j++)
		{
			int ii = j / GRAY_LEVEL;
			int ij = j % GRAY_LEVEL;
			int dd = (ii - ij) * (ii - ij);

			glcm_asm += normGlcm[j] * normGlcm[j];
			glcm_ent += -normGlcm[j] * log(normGlcm[j]+ 1e-7);
			glcm_con += dd * normGlcm[j];
			glcm_idm += normGlcm[j] / (1 + dd);
		}
		descriptors.at<double>(i, 0) = glcm_asm;
		descriptors.at<double>(i, 1) = glcm_ent;
		descriptors.at<double>(i, 2) = glcm_con;
		descriptors.at<double>(i, 3) = glcm_idm;
	}

	double totalSum = sum(descriptors)[0];
	descriptors /= totalSum;
}

int Features::convertH( double h )
{
	if (h >= 316 || h <= 20)
	{
		return 0;
	}
	else if (h > 20 && h <= 40)
	{
		return 1;
	}
	else if (h > 40 && h <= 75)
	{
		return 2;
	}
	else if (h > 75 && h <= 155)
	{
		return 3;
	}
	else if (h > 155 && h <= 190)
	{
		return 4;
	}
	else if (h > 190 && h <= 270)
	{
		return 5;
	}
	else if (h > 270 && h <= 295)
	{
		return 6;
	}
	else
	{
		return 7;
	}
}

int Features::convertS( double s )
{
	if (s <= 0.2)
	{
		return 0;
	}
	else if (s > 0.2 && s <= 0.7)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}

int Features::convertV( double v )
{
	if (v <= 0.2)
	{
		return 0;
	}
	else if (v > 0.2 && v <= 0.7)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}

void Features::getGLCM( const Mat &image, double* normGlcm, 
	int orientation, int step )
{
	Mat tmp(image.rows, image.cols, image.type());
	if (image.channels() != 1)
	{
		cvtColor(image, tmp, CV_BGR2GRAY);
	}
	else
	{
		tmp = image.clone();
	}

	int dx = step, dy = 0;
	switch (orientation)
	{
	case ORIENTATION_0:
	default:
		dx = step;
		dy = 0;
		break;
	case ORIENTATION_45:
		dx = step; 
		dy = step;
		break;
	case ORIENTATION_90:
		dx = 0; 
		dy = step;
		break;
	case ORIENTATION_135:
		dx = -step;
		dy = -step;
	}

	int divided = 256 / GRAY_LEVEL;
	int glcm[GRAY_LEVEL][GRAY_LEVEL];
	memset(glcm, 0, GRAY_LEVEL * GRAY_LEVEL * sizeof(int));
	int totalNum = 0;

	for (int i = 0; i < tmp.rows; i++)
	{
		if (i + dy >= tmp.rows || i + dy < 0)
		{
			continue;
		}
		for (int j = 0; j < tmp.cols; j++)
		{
			if (j + dx >= tmp.cols || j + dy < 0)
			{
				continue;
			}
			uchar tmp1 = tmp.at<uchar>(i, j) / GRAY_LEVEL;
			uchar tmp2 = tmp.at<uchar>(i + dy, j + dx) / GRAY_LEVEL;
			glcm[tmp1][tmp2]++;
			totalNum++;
		}
	}

	for (int i = 0; i < GRAY_LEVEL; i++)
	{
		for (int j = 0; j < GRAY_LEVEL; j++)
		{
			normGlcm[i * GRAY_LEVEL + j] = 
				(double)glcm[i][j] / (double)totalNum;
		}
	}
}

void Features::getVariational( const Mat &pts, Mat &variationals )
{
	if (pts.rows == 0)
	{
		return;
	}

	// 构建kdtree
	ANNpointArray ptsData = annAllocPts(pts.rows, 3);
#pragma omp parallel for
	for (int i = 0; i <pts.rows; i++)
	{
		Vec3d v = pts.at<Vec3d>(i, 0);
		ptsData[i][0] = v[0];
		ptsData[i][1] = v[1];
		ptsData[i][2] = v[2];
	}
	ANNkd_tree* kdtree = new ANNkd_tree(ptsData, pts.rows, 3);

	variationals.create(pts.rows, 1, CV_64FC1);
	for (int i = 0; i < pts.rows; i++)
	{
		ANNidx idxs[MaxK];
		ANNdist dists[MaxK];
		kdtree->annkSearch(ptsData[i], MaxK, idxs, dists);

		Mat neighbors(MaxK - 1, 1, CV_64FC3);
#pragma omp parallel for
		for (int j = 1; j < MaxK; j++)
		{
			neighbors.at<Vec3d>(j - 1) = pts.at<Vec3d>(idxs[j], 0);
		}

		Mat localVariationals(MaxK - MinK, 1, CV_64FC1);
#pragma omp parallel for
		for (int j = MinK; j < MaxK; j++)
		{
			Mat neighborsNx1 = neighbors(Rect(0, 0, 1, j));
			Mat neighborsNx3 = convertMat(neighborsNx1);
			Mat ones = Mat::ones(j, 1, CV_64FC1);
			Mat meanPt = (neighborsNx3.t() * ones) / (double)j;
			Mat ccMat = neighborsNx3.t() * neighborsNx3 / (double)j 
				- meanPt * meanPt.t();

			Mat eigenValues(3, 1, CV_64FC1);
			eigen(ccMat, eigenValues);

			double lambda0 = eigenValues.at<double>(2, 0);	// lambda0 < lambda1 < lambda2
			double lambda1 = eigenValues.at<double>(1, 0);
			double lambda2 = eigenValues.at<double>(0, 0);
			localVariationals.at<double>(j - MinK, 0) = 
				lambda0 / (lambda0 + lambda1 + lambda2);
		}
		double maxVariational = 0;
		Point maxLoc;
		minMaxLoc(localVariationals, 0, &maxVariational, 0, &maxLoc);
		variationals.at<double>(i, 0) = maxVariational;
	}
}

void Features::getAvgHash( const Mat &image, Mat &descriptors )
{
	Mat img, gray;
	resize(image, img, Size(8, 8));
	cvtColor(img, gray, CV_BGR2GRAY);
	double avg = mean(gray)[0];

	descriptors.create(8, 8, CV_8UC1);
	compare(gray, avg, descriptors, CMP_GE);
}

void Features::getPHash( const Mat &image, Mat &descriptors )
{
	Mat img, gray, floatGray, DCT, smallDCT;
	resize(image, img, Size(32, 32));
	cvtColor(img, gray, CV_BGR2GRAY);
	gray.convertTo(floatGray, CV_64FC1);
	dct(floatGray, DCT);
	smallDCT = DCT(Rect(0, 0, 8, 8)).clone();

	double avg = mean(smallDCT)[0];
	descriptors.create(8, 8, CV_8UC1);
	compare(smallDCT, avg, descriptors, CMP_GE);
}

void Features::getVariational(const Mat &inPts, 
	const Mat &inColors, const double &alpha, Mat &outPts, Mat &outColors)
{
	Mat variationals;
	getVariational(inPts, variationals);
	double threshold = mean(variationals)[0] * alpha;

	vector<Vec3d> outPtsVec;
	vector<Vec3b> outColorsVec;
	for (int i = 0; i < variationals.rows; i++)
	{
		double variational = variationals.at<double>(i, 0);
		if (variational > threshold)
		{
			outPtsVec.push_back(inPts.at<Vec3d>(i, 0));
			outColorsVec.push_back(inColors.at<Vec3b>(i, 0));
		}
	}
	outPts = Mat(outPtsVec, true);
	outColors = Mat(outColorsVec, true);
}

int hammingDistance( const Mat &mat1, const Mat &mat2 )
{
	if (mat1.empty() || mat2.empty())
	{
		return 64;
	}
	assert(mat1.rows == mat2.rows && mat1.cols == mat2.cols);
	Mat rst(mat1.rows, mat1.cols, CV_8UC1);
	compare(mat1, mat2, rst, CMP_NE);
	return countNonZero(rst);
}

double bhattacharyyaDistance(const Mat &mat1, const Mat &mat2)
{
	assert(mat1.channels() == 1 && mat2.channels() == 1);
	assert(mat1.cols == mat2.cols && mat1.rows == mat2.rows);
	Mat tmp = mat1.mul(mat2);
	sqrt(tmp, tmp);
	return -log(sum(tmp)[0]);
}

double computeDistance(pair<Mat, Mat> des1, pair<Mat, Mat> des2)
{
	double dis1 = bhattacharyyaDistance(des1.first, des2.first);
	double dis2 = bhattacharyyaDistance(des1.second, des2.second);
	return dis2;
}

void getSurfPointsSet( const Mat &objColorImg, const Mat &objPointCloud, 
	const Mat &objPointIndex, const Mat &modColorImg, 
	const Mat &modPointCloud, const Mat &modPointIndex, 
	Mat &objSetOrigin, Mat &objSet, Mat &modSet, xn::DepthGenerator dg)
{
	Mat colorImgs[2], keyPointsImgs[2], descriptors[2];
	vector<KeyPoint> keyPoints[2];
	vector<DMatch> matches;

	colorImgs[0] = objColorImg;
	colorImgs[1] = modColorImg;

	if (hasCuda)
	{
// 		GpuMat imgsColorGpu0, imgsColorGpu1;
// 		GpuMat imgsGpu0, imgsGpu1;
// 		imgsColorGpu0.upload(objColorImg);
// 		imgsColorGpu1.upload(modColorImg);
// 		cvtColor(imgsColorGpu0, imgsGpu0, CV_BGR2GRAY);
// 		cvtColor(imgsColorGpu1, imgsGpu1, CV_BGR2GRAY);
// 
// 		SURF_GPU surfGpu;
// 		GpuMat keyPointsGpu0, keyPointsGpu1;
// 		GpuMat desGpu0, desGpu1;
// 		surfGpu(imgsGpu0, GpuMat(), keyPointsGpu0, desGpu0);
// 		surfGpu(imgsGpu1, GpuMat(), keyPointsGpu1, desGpu1);
// 
// 		BFMatcher_GPU matcher;
// 		GpuMat trainIdx, distance;
// 		matcher.matchSingle(desGpu0, desGpu1, trainIdx, distance);
// 
// 		surfGpu.downloadKeypoints(keyPointsGpu0, keyPoints[0]);
// 		surfGpu.downloadKeypoints(keyPointsGpu1, keyPoints[1]);
// 		BFMatcher_GPU::matchDownload(trainIdx, distance, matches);
	}
	else
	{
		// 特征算子
		Ptr<Feature2D> surf;
		DescriptorMatcher* matcher = new FlannBasedMatcher;

		initModule_nonfree();
		surf = Algorithm::create<Feature2D>("Feature2D.SURF");

#pragma omp parallel for
		for (int i = 0; i < 2; i++)
		{
			// 检测关键点
			surf->detect(colorImgs[i], keyPoints[i]);

			// 计算特征描述子
			surf->compute(colorImgs[i], keyPoints[i], descriptors[i]);
		}

		// 匹配描述子
		matcher->match(descriptors[0], descriptors[1], matches);
	}

	int ptCount = (int)matches.size();
	vector<Point2f> points0;
	vector<Point2f> points1;
	double minDist = 100000;
	double maxDist = 0;

	for (int i = 0; i < ptCount; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist)
		{
			minDist = dist;
		}
		if (dist > maxDist)
		{
			maxDist = dist;
		}
	}
	for (int i = 0; i < ptCount; i++)
	{
		double dist = matches[i].distance;
		if (dist < (minDist + maxDist) / 2)
		{
			points0.push_back(keyPoints[0][matches[i].queryIdx].pt);
			points1.push_back(keyPoints[1][matches[i].trainIdx].pt);
		}
	}
	// 用RANSAC方法计算基本矩阵
	Mat tmp = findHomography(points0, points1, CV_RANSAC).clone();

	vector<Point2f> tmpKeyPoints, transPoints;
	KeyPoint::convert(keyPoints[0], tmpKeyPoints);
	perspectiveTransform(tmpKeyPoints, transPoints, tmp);

	vector<Point3d> modTmpSet, objTmpSetOrign;
	Ptr<XnPoint3D> proj = new XnPoint3D[objPointCloud.rows];
	Ptr<XnPoint3D> real = new XnPoint3D[objPointCloud.rows];
	int cnt = 0;
	for (int i = 0; i < tmpKeyPoints.size(); i++)
	{
		int x = (int)(tmpKeyPoints[i].x);
		int y = (int)(tmpKeyPoints[i].y);
		int index = objPointIndex.at<int>(y, x);
		if (index != -1)
		{
			Point3d p = objPointCloud.at<Point3d>(index, 0);
			// 			if (p == Point3d(0, 0, 0))
			// 			{
			// 				continue;
			// 			}
			objTmpSetOrign.push_back(p);
			Point2f p2d = transPoints[i];
			proj[cnt].X = p2d.x;
			proj[cnt].Y = p2d.y;
			proj[cnt].Z = (float)p.z;
			cnt++;
		}
	}
	dg.ConvertProjectiveToRealWorld(cnt, proj, real);

	objSet = Mat(cnt, 1, DataType<Point3d>::type);
#pragma omp parallel for
	for (int i = 0; i < cnt; i++)
	{
		objSet.at<Point3d>(i, 0) = Point3d(real[i].X, real[i].Y, real[i].Z);
	}

	for (int i = 0; i < keyPoints[1].size(); i++)
	{
		int x = (int)(keyPoints[1][i].pt.x);
		int y = (int)(keyPoints[1][i].pt.y);
		int index = modPointIndex.at<int>(y, x);
		if (index != -1)
		{
			modTmpSet.push_back(modPointCloud.at<Point3d>(index, 0));
		}
	}

	objSetOrigin = Mat(objTmpSetOrign, true);
	modSet = Mat(modTmpSet, true);
}

void get2DFeaturePoints( const Mat &colorImg, 
	vector<KeyPoint> &keyPoints, Mat &descriptor )
{
	keyPoints.clear();
	if (hasCuda)
	{
// 		GpuMat imgsColorGpu;
// 		GpuMat imgsGrayGpu;
// 		imgsColorGpu.upload(colorImg);
// 		cvtColor(imgsColorGpu, imgsGrayGpu, CV_BGR2GRAY);
// 
// 		SURF_GPU surfGpu;
// 		GpuMat keyPointsGpu;
// 		GpuMat desGpu;
// 		surfGpu(imgsGpu, GpuMat(), keyPointsGpu, desGpu);
//		
//		surfGpu.downloadKeypoints(keyPointsGpu, keyPoints);
//		desGpu.download(descriptor);
	}
	else
	{
		// 特征算子
		Ptr<Feature2D> surf;
		DescriptorMatcher* matcher = new FlannBasedMatcher;

		initModule_nonfree();
		surf = Algorithm::create<Feature2D>("Feature2D.SURF");

		// 检测关键点
		surf->detect(colorImg, keyPoints);

		// 计算特征描述子
		surf->compute(colorImg, keyPoints, descriptor);
	}
}

pair<double, double> pairwiseMatch(const vector<KeyPoint> &queryKeypoints, 
	const vector<KeyPoint> &trainKeypoints, const Mat &queryDescriptors, 
	const Mat &trainDescriptors, Mat &H, vector<pair<int, int>> &matchesPoints)
{
	matchesPoints.clear();

	vector<DMatch> matches;
	if (hasCuda)
	{
// 		GpuMat desGpu0, desGpu1;
// 		desGpu0.upload(queryDescriptors);
// 		desGpu1.upload(trainDescriptors);
// 		BFMatcher_GPU matcher;
// 		GpuMat trainIdx, distance;
// 		matcher.matchSingle(desGpu0, desGpu1, trainIdx, distance);
// 
// 		BFMatcher_GPU::matchDownload(trainIdx, distance, matches);
	}
	else
	{
		DescriptorMatcher* matcher = new FlannBasedMatcher;
		matcher->match(queryDescriptors, trainDescriptors, matches);
	}

	int ptCount = (int)matches.size();
	vector<pair<int, int>> matchesIndex;
	vector<Point2f> points0;
	vector<Point2f> points1;
	double meanDist = 0;

	// 求平均距离
	for (int i = 0; i < ptCount; i++)
	{
		double dist = matches[i].distance;
		meanDist += dist;
	}
	meanDist /= (double)ptCount;

	// 去除距离过大的特征点
	for (int i = 0; i < ptCount; i++)
	{
		double dist = matches[i].distance;
		if (dist < meanDist)
		{
			points0.push_back(queryKeypoints[matches[i].queryIdx].pt);
			points1.push_back(trainKeypoints[matches[i].trainIdx].pt);
			matchesIndex.push_back(
				make_pair(matches[i].queryIdx, matches[i].trainIdx));
		}
	}
	
	if (matchesIndex.size() < 4)
	{
		return make_pair(1.0, MAX_AREA_DIFF_PERCENT);
	}
	// 用RANSAC方法计算基本矩阵
	vector<uchar> ransacStatus;
	H = findHomography(points0, points1, ransacStatus, CV_RANSAC);

	vector<Point> tmpPoint0, tmpPoint1;
	for (int i = 0; i < (int)points0.size(); i++)
	{
		if (ransacStatus[i] != 0)
		{
			matchesPoints.push_back(matchesIndex[i]);
			Point2f sp = points0[i];
			Point dp((int)sp.x, (int)sp.y);
			tmpPoint0.push_back(dp);
			sp = points1[i];
			dp = Point((int)sp.x, (int)sp.y);
			tmpPoint1.push_back(dp);
		}
	}

	vector<vector<Point>> hull(2), approx(2);
	convexHull(tmpPoint0, hull[0]);
	convexHull(tmpPoint1, hull[1]);

	double area0 = contourArea(hull[0]);
	double area1 = contourArea(hull[1]);
	double areaDiff = abs(area1 - area0);
	areaDiff = areaDiff * 0.5 / (area0 + area1);
	double score = matchShapes(hull[0], hull[1], CV_CONTOURS_MATCH_I1, 0);
	if (area0 < DBL_EPSILON || area1 < DBL_EPSILON)
	{
		score = 1.0;
	}
	if (double(matchesPoints.size()) / double(matches.size()) 
		< FEATURE_NUM_THRESHOLD)
	{
		score = 1.0;
	}
	if (double(matchesPoints.size()) / double(trainKeypoints.size()) > 0.25)
	{
		score = 1.0;
	}
	return make_pair(score, areaDiff);
}

bool convert2DTo3D( xn::DepthGenerator dg, const Mat &H, 
	const Mat &depthImgNow, const Mat &depthImgPre, 
	const vector<KeyPoint> &keypointsNow,
	const vector<KeyPoint> &keypointsPre, 
	const vector<pair<int, int>> &matchesPoints,
	vector<Vec2d> &oldLoc, vector<Vec2d> &newLoc,
	Mat &objSet, Mat &modSet, Mat &objSetAT, Mat &mask )
{	
	if (matchesPoints.size() == 0)
	{
		return false;
	}
	// 将特征点通过基本矩阵转换
	vector<Point2f> points0;
	vector<Point2f> transPoints;
	KeyPoint::convert(keypointsNow, points0);
	perspectiveTransform(points0, transPoints, H);

	int rows = depthImgNow.rows;
	int cols = depthImgNow.cols;
	int size = (int)matchesPoints.size();

	Mat tmpMask;
	depthImgPre.convertTo(tmpMask, CV_8UC1, -255.0, 255.0);
	mask = mask & tmpMask;
	warpPerspective(mask, mask, H, Size(cols, rows),
		INTER_LINEAR + WARP_INVERSE_MAP, 0, Scalar::all(255));

	vector<Point3d> modTmpSet, objTmpSet, objTmpSetAT;
	Ptr<XnPoint3D> projO = new XnPoint3D[size];
	Ptr<XnPoint3D> realO = new XnPoint3D[size];
	Ptr<XnPoint3D> projM = new XnPoint3D[size];
	Ptr<XnPoint3D> realM = new XnPoint3D[size];
	Ptr<XnPoint3D> projAT = new XnPoint3D[size];
	Ptr<XnPoint3D> realAT = new XnPoint3D[size];
	int cnt = 0;
	for (int i = 0; i < size; i++)
	{
		Point2f op = keypointsNow[matchesPoints[i].first].pt;
		Point2f mp = keypointsPre[matchesPoints[i].second].pt;
		int ox = (int)(op.x);
		int oy = (int)(op.y);
		int mx = (int)(mp.x);
		int my = (int)(mp.y);
		ushort oz = depthImgNow.at<ushort>(oy, ox);
		ushort mz = depthImgPre.at<ushort>(my, mx);
		if (oz != 0 && mz != 0)
		{
			projO[cnt].X = (float)ox;
			projO[cnt].Y = (float)oy;
			projO[cnt].Z = (float)oz;

			Point2f p2d = transPoints[matchesPoints[i].first];
			projAT[cnt].X = p2d.x;
			projAT[cnt].Y = p2d.y;
			projAT[cnt].Z = (float)oz;

			projM[cnt].X = (float)mx;
			projM[cnt].Y = (float)my;
			projM[cnt].Z = (float)mz;

			oldLoc.push_back(Vec2d(mx, my));
			newLoc.push_back(Vec2d(ox, oy));

			cnt++;
		}
	}
	if (cnt == 0)
	{
		return false;
	}
	dg.ConvertProjectiveToRealWorld(cnt, projO, realO);
	dg.ConvertProjectiveToRealWorld(cnt, projM, realM);
	dg.ConvertProjectiveToRealWorld(cnt, projAT, realAT);

	objSet = Mat(cnt, 1, DataType<Point3d>::type);
	modSet = Mat(cnt, 1, DataType<Point3d>::type);
	objSetAT = Mat(cnt, 1, DataType<Point3d>::type);
#pragma omp parallel for
	for (int i = 0; i < cnt; i++)
	{
		objSet.at<Point3d>(i, 0) = Point3d(realO[i].X / 1000.0, realO[i].Y / 1000.0, realO[i].Z / 1000.0);
		modSet.at<Point3d>(i, 0) = Point3d(realM[i].X / 1000.0, realM[i].Y / 1000.0, realM[i].Z / 1000.0);
		objSetAT.at<Point3d>(i, 0) = Point3d(
			realAT[i].X / 1000.0, realAT[i].Y / 1000.0, realAT[i].Z / 1000.0);
	}

	return true;
}

void convert2DTo3D( xn::DepthGenerator dg, 
	const Mat &depthImg, const vector<KeyPoint> &keypoints, Mat &points )
{
	assert(keypoints.size() > 0);
	int size = (int)keypoints.size();
	vector<Point3d> tmpSet;
	Ptr<XnPoint3D> proj = new XnPoint3D[size];
	Ptr<XnPoint3D> real = new XnPoint3D[size];
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		Point2f p = keypoints[i].pt;
		int x = (int)(p.x);
		int y = (int)(p.y);
		ushort z = depthImg.at<ushort>(y, x);
		proj[i].X = (float)x;
		proj[i].Y = (float)y;
		proj[i].Z = (float)z;
	}
	dg.ConvertProjectiveToRealWorld(size, proj, real);

	points = Mat(size, 1, DataType<Point3d>::type);
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		points.at<Point3d>(i, 0) = Point3d(real[i].X, real[i].Y, real[i].Z);
	}
}

void fullMatch( const int &index, 
	const vector<Mat> &descriptors, 
	const vector<vector<KeyPoint>> &keypoints, 
	vector<vector<pair<int, int>>> &matches )
{
	assert(index > 0);
	matches.clear();
	matches.resize(index);

#pragma omp parallel for
	for (int i = 0; i < index; i++)
	{
		Mat H;
		vector<pair<int, int>> matchesPairs;
		pair<double, double> scoreDiff = pairwiseMatch(keypoints[index], 
			keypoints[i], descriptors[index], descriptors[i], H, matchesPairs);
		if (scoreDiff.first > SIMILARITY_THRESHOLD_HULL_DOWN ||
			scoreDiff.second > AREA_DIFF_THRESHOLD)
		{
			matchesPairs.clear();
		}
		matches[i] = matchesPairs;
	}
}
