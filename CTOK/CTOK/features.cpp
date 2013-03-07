#include "features.h"

void Features::getHSVColorHistDes( const Mat& image, Mat& descriptors )
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
	descriptors = Mat(1, 72, CV_32FC1);
	for (int i = 0; i < 72; i++)
	{
		float temp = (float)hist[i] / (float)n;
		descriptors.at<float>(0, i) = temp;
	}
}

void Features::getGLCMDes( const Mat& image, Mat& descriptors )
{
	double normGlcm[GRAY_LEVEL * GRAY_LEVEL];

	descriptors = Mat(4, 4, CV_32FC1);
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
		descriptors.at<float>(i, 0) = (float)glcm_asm;
		descriptors.at<float>(i, 1) = (float)glcm_ent;
		descriptors.at<float>(i, 2) = (float)glcm_con;
		descriptors.at<float>(i, 3) = (float)glcm_idm;
	}

	float totalSum = (float)sum(descriptors)[0];
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

void Features::getGLCM( const Mat& image, double* normGlcm, 
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
