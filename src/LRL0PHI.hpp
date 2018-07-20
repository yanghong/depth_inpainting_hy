#ifndef __LRL0PHI_HPP__
#define __LRL0PHI_HPP__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <list>
#include "tnnr.h"
using namespace std;
using namespace cv;

// Low-Rank Total Variation for Image Inpainting
class LRL0PHI
{
  Mat U_;  // 原图像
  Mat U_last_;  // 最近的图像备份
  Mat mask_; //mask for the missing pixels: 0 indicates missing
  Mat I_, M_, Y_;  
  int W_, H_; // 矩阵的宽度和高度
  float rho_, dt_, h_, lambda_l0_, k_;
  float lambda_rank_, alpha_;
    
  Mat kernelx_plus_, kernelx_minus_;
  Mat kernely_plus_, kernely_minus_;
    
public:
  LRL0PHI(Mat &I, Mat &mask);
  // debug purpose
  void init_U(Mat &u0);
  Mat getU();
  Mat getY();
  Mat getM();
  //
  void setParameters(float rho, float dt, float lambda_l0, float lambda_rank, float k);
  float sub_1_val(Mat &X); // evaluate subproblem 1 to guide gradient descent
  void sub_1(int K);
  void sub_2();
  void sub_3();
  Mat compute(int K, int max_iter, string path, Mat &original, string path1);
};

#endif
