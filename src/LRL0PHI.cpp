#include "LRL0PHI.hpp"
#include "util.h"
#include <map>
#include <list>
#include <fstream>
//#include "LRL0.hpp"
#include "common.h"
#define garma 2.2
int iters1;
// class element{
// public:
//     int w;
//     int w_mask;
//     float Y;
//     float M_mean, Y_mean, I_mean;
//     list<int> G,N;
//     map<int,int> c;
// };

// list<int>::iterator ffind(list<int>::iterator start, list<int>::iterator end, int value)
// {
//     list<int>::iterator p = start;
//     while(p != end)
//     {
//         if(*p == value)
//             return p;
//         ++p;
//     }
//     return p;
// }

// float g(int iter, int k, float l)
// {
//     float ans = pow(((float)iter/k), garma) * l;
//     return ans;
// }

// map<int, int>::iterator ffind(map<int, int>::iterator start, map<int, int>::iterator end, int value)
// {
//     map<int, int>::iterator p = start;
//     while(p != end)
//     {
//         if(p->first == value)
//             return p;
//         ++p;
//     }
//     return p;
// }

// 返回U_
Mat LRL0PHI::getU()
{
    return U_;
}

// 返回M_
Mat LRL0PHI::getM()
{
    return M_;
}

// 返回Y_
Mat LRL0PHI::getY()
{
    return Y_;
}

// 初始化 M_
void LRL0PHI::init_U(Mat &u0)
{
    u0.convertTo(M_, CV_32FC1);
}

// 初始化LRL0PHI.hpp中定义的各种矩阵
// mask: uint8_t type 1 channel taking values 0 and 255
// I: uint8_t grayscale image
LRL0PHI::LRL0PHI(Mat &I, Mat &mask)
{
    W_ = I.cols;  //列（宽度）
    H_ = I.rows;  //行（高度）
    I_ = Mat::zeros(H_, W_, CV_32F);  // 全为0的同型矩阵
    mask_ = Mat::zeros(H_, W_, CV_32F);  // 全为0的同型矩阵
    
    I.convertTo(I_, CV_32FC1, 1.0);  // I转化为全为1的同型矩阵，像素类型为CV_32FC1
    mask.convertTo(mask_, CV_32FC1, 1.0/255.0);  // mask转化为全为1.0/255.0的同型矩阵，像素类型为CV_32FC1
    // initialize
    U_ = Mat::zeros(H_, W_, CV_32F); // U_初始化为0的同型矩阵
    U_last_ = Mat::zeros(H_, W_, CV_32F);  // U_last_初始化为0的同型矩阵
    // internal variables
    M_ = Mat::zeros(H_, W_, CV_32F);  // M_初始化为0的同型矩阵
    Y_ = Mat::zeros(H_, W_, CV_32F);  // Y_初始化为0的同型矩阵
    
    // make gradient operators
    kernelx_plus_ = (Mat_<float>(1,3)<<0.0,-1.0,1.0);   // 定义形状为（1，3）的矩阵
    kernelx_minus_ = (Mat_<float>(1,3)<<-1.0,1.0,0.0);   // 定义形状为（1，3）的矩阵
    kernely_plus_ = (Mat_<float>(3,1)<<0.0,-1.0,1.0);   // 定义形状为（3，1）的矩阵
    kernely_minus_ = (Mat_<float>(3,1)<<-1.0,1.0,0.0);   // 定义形状为（3，1）的矩阵
}

// 需要搞清楚这几个参数的含义?
void LRL0PHI::setParameters(float rho, float dt, float lambda_l0, float lambda_rank, float k)
{
    rho_ = rho;
    dt_ = dt;
    h_ = 1.0;
    alpha_ = 1.0;
    lambda_rank_ = lambda_rank; //10;
    lambda_l0_ = lambda_l0;
    k_ = k;
    cout << "set parameters done" << endl;
}

void LRL0PHI::sub_1(int K)
{
    int iter;

    float epsilon = 1.0e-4;
    map<int, element> I;
    // travel every pixel in U_
    for(int i = 0; i < U_.rows; i++)
    {
        for(int j = 0; j < U_.cols; j++)
        {
            int index = i*U_.cols+j;    //某一像素定位点
            element temp;               // 每一个像素都会创建一个temp 
            temp.G.push_back(index);    // G的类型应该是一个List, index是索引
            temp.w = 1;       // w系数的含义
            temp.w_mask = mask_.at<float>(i,j) == 0 ? 0 : 1;
            temp.I_mean = temp.w_mask * I_.at<float>(i,j);
            if(!(temp.I_mean == temp.I_mean))
            {
	      //                cout << mask_.at<float>(i,j) << endl;
	      //                cout << I_.at<float>(i,j) << endl;
	      //                cout << temp.I_mean << endl;
            }
            temp.M_mean = M_.at<float>(i,j);
            temp.Y_mean = Y_.at<float>(i,j);
            temp.Y = (float)(temp.w*rho_*(temp.M_mean - temp.Y_mean) + 2*temp.w_mask*temp.I_mean)/
                     (float)(temp.w*rho_ + 2*temp.w_mask);  // Y是求解ADMM最后的一个值
            //temp.Y = U_.at<float>(i,j);
            if(i != 0)
            {
                temp.N.push_back((i-1)*U_.cols+j);
                temp.c.insert(pair<int,int>((i-1)*U_.cols+j,1));
            }
            if(i != U_.rows-1)
            {
                temp.N.push_back((i+1)*U_.cols+j);
                temp.c.insert(pair<int,int>((i+1)*U_.cols+j,1));
            }
            if(j != 0)
            {
                temp.N.push_back(i*U_.cols+j-1);
                temp.c.insert(pair<int,int>(i*U_.cols+j-1,1));
            }
            if(j != U_.cols-1)
            {
                temp.N.push_back(i*U_.cols+j+1);
                temp.c.insert(pair<int,int>(i*U_.cols+j+1,1));
            }
            I.insert(pair<int,element>(index,temp));  // I从结构上看temp.N是某一像素的邻接像素
        }
    }
    cout << "I: " << I <<endl;
    // Get element I(map<int, element>) Done.
    //    cout << "1 is done" << endl;
    map<int, element>::iterator i;
    for(i = I.begin(); i != I.end(); ++i)
        for(list<int>::iterator j = i->second.G.begin(); j != i->second.G.end(); ++j)
            U_.at<float>(*j/U_.cols, *j % U_.cols) = i->second.Y;
    float beta = 0;
    iter = 0;
    // 下面的应该是区域融合算法
    while(1)
    {
        map<int,element>::iterator it = I.begin();
        while(it != I.end())
        {
            int i = it->first;
            for(list<int>::iterator j = I[i].N.begin(); j != I[i].N.end();)
            {
                float value1 = 0;
                float value2 = 0;
                float value3 = 0;
                element temp1 = I[i];
                element temp2 = I[*j];
                value1 = (temp1.w * temp1.w_mask * rho_ * pow(temp1.I_mean - temp1.M_mean + temp1.Y_mean,
                             2) / (temp1.w * rho_ + 2 * temp1.w_mask)) +
                         (temp2.w * temp2.w_mask * rho_ * pow(temp2.I_mean - temp2.M_mean + temp2.Y_mean, 2) / (temp2.w * rho_ + 2 * temp2.w_mask)) +
                         (beta * ffind(I[i].c.begin(), I[i].c.end(), *j)->second);
                float X = 0;
                X = (temp1.w*rho_*(temp1.M_mean-temp1.Y_mean) + 2*temp1.w_mask*temp1.I_mean +
                     temp2.w*rho_*(temp2.M_mean-temp2.Y_mean) + 2*temp2.w_mask*temp2.I_mean) /
                    (temp1.w*rho_ + 2*temp1.w_mask + temp2.w*rho_ + 2*temp2.w_mask);
                value2 = (temp1.w*rho_/2.0f * pow(X - temp1.M_mean + temp1.Y_mean, 2)) +
                         (temp1.w_mask * pow(X - temp1.I_mean, 2)) +
                         (temp2.w*rho_/2.0f * pow(X - temp2.M_mean + temp2.Y_mean, 2)) +
                         (temp2.w_mask * pow(X - temp2.I_mean, 2));
                float Xi = 0, Xj = 0;
                if(temp2.Y < temp1.Y)
                {
                    Xi = (temp1.w*rho_*(temp1.M_mean-temp1.Y_mean) + 2*temp1.w_mask*temp1.I_mean +
                          temp2.w*rho_*(temp2.M_mean-temp2.Y_mean+1.0f) + 2*temp2.w_mask*(temp2.I_mean+1.0f)) /
                         (temp1.w*rho_ + 2*temp1.w_mask + temp2.w*rho_ + 2*temp2.w_mask);
                    Xj = Xi - 1.0f;
                }
                else if(temp2.Y >= temp1.Y)
                {
                    Xi = (temp1.w*rho_*(temp1.M_mean-temp1.Y_mean) + 2*temp1.w_mask*temp1.I_mean +
                          temp2.w*rho_*(temp2.M_mean-temp2.Y_mean-1.0f) + 2*temp2.w_mask*(temp2.I_mean-1.0f)) /
                    (temp1.w*rho_ + 2*temp1.w_mask + temp2.w*rho_ + 2*temp2.w_mask);
                    Xj = Xi + 1.0f;
                }
                value3 = (temp1.w*rho_/2.0f * pow(Xi - temp1.M_mean + temp1.Y_mean, 2)) +
                         (temp1.w_mask * pow(Xi - temp1.I_mean, 2)) +
                         (temp2.w*rho_/2.0f * pow(Xj - temp2.M_mean + temp2.Y_mean, 2)) +
                         (temp2.w_mask * pow(Xj - temp2.I_mean, 2)) + k_*lambda_l0_;
                value1 = value1 <= epsilon ? 0 : value1;
                value2 = value2 <= epsilon ? 0 : value2;
                value3 = value3 <= epsilon ? 0 : value3;
                /*if(!(value1 == value1 && value2 == value2))
                {
                    cout << "ooooooops" << i << " " << *j << endl;
                    cout << temp2.I_mean << endl;
                    cout << temp2.M_mean << endl;
                    cout << temp2.Y_mean << endl;
                    cout << temp2.I_mean - temp2.M_mean + temp2.Y_mean << endl;
                    cout << I[*j].I_mean << endl;
                }*/
                // if(!(value1 == value1 && value2 == value2 && value3 == value3))
                //     cout <<value3 << " " << value2 << " " << value1 << endl;
                if(value3 < (value1 < value2 ? value1 : value2))  // value3 小于value1和value2中的最小值，相当于论文中的fA <= fB and fA <= fC
                {
                    temp1.Y = Xi;
                    temp2.Y = Xj;
                    j++;
                }
                else if(value2 <= (value1 < value3 ? value1 : value3))   // value2小于 value1和value3 中的最小值，相当于论文中的fB <= fA and fB <= fC
                {
                    //cout << value2 << " " << value1 << endl;
                    int temp_value = *j;
                    list<int> t1 = I[i].G;
                    list<int> t2 = I[*j].G;
                    I[i].Y = X;
                    I[i].Y_mean = (I[i].Y_mean*I[i].w + I[*j].Y_mean*I[*j].w)/(I[i].w + I[*j].w);
                    I[i].M_mean = (I[i].M_mean*I[i].w + I[*j].M_mean*I[*j].w)/(I[i].w + I[*j].w);
                    /*if(!((I[i].I_mean*I[i].w_mask + I[*j].I_mean*I[*j].w_mask)/
                         (I[i].w_mask + I[*j].w_mask) == (I[i].I_mean*I[i].w_mask + I[*j].I_mean*I[*j].w_mask)/
                         (I[i].w_mask + I[*j].w_mask)))
                    {
                        cout << "ooooooops" << i << endl;
                        cout << I[i].I_mean*I[i].w_mask << endl;
                        cout << I[*j].I_mean*I[*j].w_mask << endl;
                        cout << I[*j].I_mean << endl;
                        cout << (I[i].I_mean*I[i].w_mask + I[*j].I_mean*I[*j].w_mask) << endl;
                        cout << I[i].w_mask + I[*j].w_mask << endl;
                        cout << I[i].I_mean << endl;
                        cout << "done" << endl;
                    }*/
                    if(I[i].w_mask == 0 && I[*j].w_mask == 0)
                        I[i].I_mean = 0;
                    else
                        I[i].I_mean = (I[i].I_mean*I[i].w_mask + I[*j].I_mean*I[*j].w_mask)/
                                      (I[i].w_mask + I[*j].w_mask);
                    I[i].w = I[i].w + I[*j].w;
                    I[i].w_mask = I[i].w_mask + I[*j].w_mask;
                    I[i].G.merge(I[*j].G);
                    I[i].c.erase(temp_value);
                    j = I[i].N.erase(j);
                    list<int>::iterator k;
                    for(k = I[temp_value].N.begin(); k != I[temp_value].N.end(); ++k)
                    {
                        
                        if(*k == i)
                            continue;
                        if(ffind(I[i].N.begin(), I[i].N.end(), *k)!=I[i].N.end())
                        {
                            ffind(I[i].c.begin(),I[i].c.end(),*k)->second +=
                            ffind(I[temp_value].c.begin(),I[temp_value].c.end(),*k)->second;
                            ffind(I[*k].c.begin(),I[*k].c.end(),i)->second =
                            ffind(I[i].c.begin(),I[i].c.end(),*k)->second;
                        }
                        else{
                            I[i].N.push_back(*k);
                            list<int>temp = I[i].N;
                            I[*k].N.push_back(i);
                            I[i].c.insert(pair<int,int>(*k,ffind(I[temp_value].c.begin(),I[temp_value].c.end(),*k)->second));
                            I[*k].c.insert(pair<int,int>(i,ffind(I[temp_value].c.begin(),I[temp_value].c.end(),*k)->second));
                        }
                        I[*k].c.erase(temp_value);
                        I[*k].N.remove(temp_value);
                    }
                    I.erase(temp_value);
                    ++it;
                    break;
                }
                else{
                    j++;
                }
            }
            if(it == I.end())
                break;
            ++it;
        }
        /*Mat tem = Mat::zeros(H_, W_, CV_32F);
        Mat temp = Mat::zeros(H_, W_, CV_8UC1);
        for(map<int, element>::iterator i = I.begin(); i != I.end(); ++i)
            for(list<int>::iterator j = i->second.G.begin(); j != i->second.G.end(); ++j)
                tem.at<float>(*j/tem.cols, *j %tem.cols) = i->second.Y;
        tem.convertTo(temp, CV_8UC1);
        imshow("result", temp);
        waitKey(0);*/
	//        cout << "current beta: " << beta << endl;
        beta = g(++iter, K, lambda_l0_);
        if(beta > lambda_l0_)
            break;
    }
    //    cout << "2 is done" << endl;
    //Mat temp = Mat::zeros(H_, W_, CV_8UC1);
    //U_.convertTo(temp, CV_8UC1);
    //imshow("result", temp);
    //waitKey(0);
    //map<int, element>::iterator i;
    for(map<int, element>::iterator i = I.begin(); i != I.end(); ++i)
        for(list<int>::iterator j = i->second.G.begin(); j != i->second.G.end(); ++j)
            U_.at<float>(*j/U_.cols, *j % U_.cols) = i->second.Y;
    //U_.convertTo(temp, CV_8UC1);
    //imshow("ans", temp);
    //imwrite("./anss.png", temp);
    //waitKey(0);
    Mat temp = Mat::zeros(H_, W_, CV_8UC1);
    U_.convertTo(temp, CV_8UC1);
    string name;
    name = "ans_"+to_string(iters1)+".png";
    imwrite("./result/"+name, temp);
}

// LR
void LRL0PHI::sub_2()
{
    Mat A = U_ + Y_;
    Mat mask = 255*Mat::ones(H_, W_, CV_8UC1);
    float lambda = rho_ / 2.0 / lambda_rank_ / alpha_;
    
    // TNNR
    Mat At;
    A.convertTo(At, CV_8UC1);
    M_ = TNNR(At, mask, 9, 9, lambda);
    
    
}

void LRL0PHI::sub_3()
{
    Y_ = Y_ + (U_ - M_);
}

Mat LRL0PHI::compute(int K, int max_iter, string path, Mat &original, string path1)
{
  Mat output;
  for(int iter = 0; iter < max_iter; ++iter)
    {
      cout << "=====================" << endl;
      cout << "Iter = " << (iter + 1) << endl;
      iters1 = iter;
      sub_1(K);
      cout << "Piecewise L0 done" << endl;  // 分段 L0  Done
      sub_2();
      cout << "LR done" << endl;
      sub_3();
        
        
        
      U_.convertTo(output, CV_8UC1);
      string newpath = path + to_string(iter+1);

      imwrite(newpath + ".png", output);
      cout << "result in " << newpath + ".png" << endl;
      Mat mask = 255.0 * mask_;
      float psnr = PSNR(original, U_, mask);

      string filePath = path1 + "/";
      filePath = filePath + to_string(iter + 1);
      filePath = filePath + ".txt";
      fstream file1(filePath, ios::out);
      file1 << psnr << endl;
      cout << "PSNR = " << psnr << endl;
      file1.close();
      cout << "result psnr in " + filePath << endl;
      // when to stop needs further con sideration
	  // this code can't understand now??
      if( iter >= 1 && norm(U_, U_last_) / norm(U_last_) < 1e-3 )
	{
	  break;
	}
	
      cout << "relative error = " << norm(U_, U_last_) / norm(U_last_) << endl;
      U_.copyTo(U_last_);


      cout << endl;

    }

  return U_;
}

