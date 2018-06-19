//#include "stdafx.h"
#include "cv.h"
#include "highgui.h"
#include <iostream>
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
    cv::Mat image;
    image = cv::imread("/home/hy/depthInpainting-data/adi/disp.png");
    if (image.empty())
    {
        cout << "No image";
    }
    else
    {
        cv::namedWindow("Original Image");
        cv::imshow("Original Image", image);
        cout << "This image is " << image.rows << "x" << image.cols << endl;
        cv::waitKey(0);
    }
    return 0;
}
