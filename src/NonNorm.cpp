#include "NonNorm.h"

MatInfo NONNORM(Mat &D,float rho,Mat T0,float gamma){
	Mat U,S,V,outputT,outputPow;
	MatInfo getMatInfo;
	MatInfo returnMatInfo;
	// SVD -> D
	SVD::compute(D, S, U, V);
	// itera 1:100
	for (int i = 1; i< 101; i++) {
		getMatInfo = DCInner(S,rho,T0,gamma,U,V);
		subtract(getMatInfo.T,T0,outputT);
		pow(outputT,2,outputPow);
		float err = sum(outputPow);
		if(err < 1e-6){
			break;
		}
		T0 = getMatInfo.T;
	}
	Mat T = getMatInfo.T;
	returnMatInfo.X = getMatInfo.X;
	returnMatInfo.T = getMatInfo.T;
	return returnMatInfo;
}

MatInfo DCInner(Mat &S, float rho, Mat J , float epislon, Mat &U, Mat &V) {
	MatInfo matInfo;
	float lambda = 0.5/rho;
	Mat zeroMat = Mat::zeros(J.rows,J.cols,CV_32FC1);
	Mat outputGrad;

	Mat S0 = Mat::diag(S);
	exp(-J/epislon,outputGrad);
	Mat grad = outputGrad/epislon;
	Mat t = max(S0 - lambda*grad,zeroMat);
	Mat X = U *Mat::diag(t) * V.t();
	matInfo.X = X;
	matInfo.T = t;
	return matInfo;
}

float sum(Mat &mat){
	float s = 0.0f;
	for(int row=0;row<mat.rows;row++){
		uchar* data = mat.ptr<uchar>(row);
		for(int col=0;col<mat.cols;col++) {
			s = s + data[col];
		}
	}
	return s;
}
