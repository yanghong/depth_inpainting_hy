#include <opencv2/core/core.hpp>//base data-strucure lib
#include <opencv2/highgui/highgui.hpp>//relate to UI lib
#include <iostream>

using namespace cv;

int main(){
	Mat myMat = imread("/home/hy/depthInpainting-data/adi/disp.png");
	namedWindow("image");
	imshow("image",myMat);
	waitKey(5000);
	std::cout << "Done";
	return 0;
}
