#include "NonNorm.h"
#include <fstream>
#include <cmath>
#include <list>

MatInfo NONNORM(Mat &D,float rho,Mat &T0,float gamma){
	Mat U,S,V,outputT,outputPow;
	MatInfo getMatInfo;
	MatInfo returnMatInfo;
	ofstream file("temp.txt",ios::app);
    int count=0;
    // set patch
	int patchWidth = 25;
	int patchHeight = 25;
	int stepsize = 16;
	int m = D.rows;
	int n = D.cols;
	int R = m-patchWidth+1;
	int C = n-patchWidth+1;
	list<int> rr;
	list<int> cc;

	for(int si = 1; si < R; si++) {
		if(si%stepsize == 1) {
			rr.push_back(si);
		}
	}
	rr.push_back(rr.back()+1);
	for(int sj = 1; sj < C; sj++) {
		if(sj%stepsize == 1){
			cc.push_back(sj);
		}
	}
	cc.push_back(cc.back()+1);

	int rrLength = rr.size();
    int ccLength = cc.size();
	list<int>::iterator rrIter;
	list<int>::iterator ccIter;
	rrIter = rr.begin();
	ccIter = cc.begin();


	Mat T0Type;
	T0.convertTo(T0Type, CV_32FC1);
	Mat DType;
	D.convertTo(DType, CV_32FC1);

	int patchI = DType.cols/patchWidth;
    int patchJ = DType.rows/patchHeight;
	Mat result(D.rows,D.cols,CV_32FC1);
	Mat resultV(patchWidth,D.cols,CV_32FC1);


	int positionX = 0;
	int positionY = 0;

	Mat resultHTemp;
	for (int pi=0;pi<rrLength;pi++) {
			Mat resultH(stepsize,stepsize,CV_32FC1);
			if(rrIter != rr.end()) {
				positionX = *rrIter;
				++rrIter;
				ccIter = cc.begin();
			}
		for(int pj=0;pj<ccLength;pj++) {
			if(ccIter != cc.end()) {
				positionY = *ccIter;
				++ccIter;
			}
			Mat patch = DType(Range(positionX,positionX + patchWidth),Range(positionY,positionY + patchWidth));


			SVD::compute(patch, S, U, V);
			for (int i101 = 1; i101< 101; i101++) {
				getMatInfo = DCInner(S,rho,T0Type,gamma,U,V);
				subtract(getMatInfo.T,T0Type,outputT);
				pow(outputT,2,outputPow);
				float err;
				for (int row = 0;row< outputPow.rows;row++) {
					float *data = outputPow.ptr<float>(row);
					for(int col=0;col<outputPow.cols;col++) {
						err = err + data[col];
					}   
				}   
				if(err < 1e-6){
					 break;
				}   
				getMatInfo.T.copyTo(T0Type);		
			  }
			// hconcat must be rows equal
			Mat resultStepsize = getMatInfo.X(Range(0,stepsize),Range(0,stepsize));
			if (pj == 0) {
				resultH = resultStepsize;
			} else {
				hconcat(resultH,resultStepsize,resultH);
			}
			count++;
		}
        
		if (resultH.cols < D.cols) {
			int gap = D.cols - resultH.cols;
			Mat resultFill = getMatInfo.X(Range(0,patchWidth),Range(patchWidth - gap,patchWidth));
			hconcat(resultH,resultFill,resultH);
		}
		// vconcat must be columns equal
		if (pi == 0) {
			resultV = resultH;
		} else {
			vconcat(resultV,resultH,resultV);
		}
	}

    resize(resultV,result,result.size(),INTER_LINEAR);
	returnMatInfo.X = result;
	returnMatInfo.T = getMatInfo.T;
	return returnMatInfo;
}

MatInfo DCInner(Mat &S, float rho, Mat &J , float epislon, Mat &U, Mat &V) {
	ofstream file2("temp2.txt",ios::app);
	MatInfo matInfo;
	float lambda = rho;
	Mat zeroMat = Mat::zeros(J.rows,J.cols,CV_32FC1);
	Mat outputGrad;

	exp(-J/epislon,outputGrad);
	Mat grad = outputGrad/epislon;
	Mat t = max(S - lambda*grad,0.0);
	Mat tDiag = Mat::diag(t);
	Mat X = U * tDiag * V;
	X.copyTo(matInfo.X);
	t.copyTo(matInfo.T);
	return matInfo;
}

