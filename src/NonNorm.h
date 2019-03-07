#ifndef __NONNORM_H_
#define __NONNORM_H_
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct MatInfo{
	Mat X;
	Mat T;
};

MatInfo NONNORM(Mat &D,float rho,Mat T0, float gamma);

MatInfo DCInner(Mat &S, float rho, Mat J, float epislon, Mat &U, Mat &V);

float sum(Mat &mat);


#endif
