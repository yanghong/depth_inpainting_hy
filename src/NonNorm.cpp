#include "NonNorm.h"

MatInfo NONNORM(Mat &D,float rho,Mat T0,float gamma){
	Mat U,S,V,outputT,outputPow;
	MatInfo getMatInfo;
	MatInfo returnMatInfo;

	Mat T0Type;
	T0.convertTo(T0Type, CV_32FC1);
	Mat DType;
	D.convertTo(DType, CV_32FC1);

	cout << "BreakPoint 6" << endl;

	// SVD -> D
	SVD::compute(DType, S, U, V);
	cout << "BreakPoint 7" << endl;

	// itera 1:100
	for (int i = 1; i< 101; i++) {
		cout << "BreakPoint 8" << endl;
		cout << "U: " << U.rows << " rows,cols:" << U.cols << endl;
		getMatInfo = DCInner(S,rho,T0Type,gamma,U,V);
		cout << "BreakPoint 10" << endl;
		subtract(getMatInfo.T,T0Type,outputT);
		cout << "BreakPoint 11" << endl;
		pow(outputT,2,outputPow);
		cout << "BreakPoint 12" << endl;
		float err = sum(outputPow);
		cout << "BreakPoint 13" << endl;
		if(err < 1e-6){
			break;
		}
		T0Type = getMatInfo.T;

		cout << "BreakPoint 14" << endl;
	}
	cout << "BreakPoint 15" << endl;

	Mat T = getMatInfo.T;
	returnMatInfo.X = getMatInfo.X;
	returnMatInfo.T = getMatInfo.T;
	return returnMatInfo;
}

MatInfo DCInner(Mat &S, float rho, Mat J , float epislon, Mat &U, Mat &V) {
	MatInfo matInfo;
	float lambda = 0.5/rho;
	cout << "BreakPoint 16" << endl;
	Mat zeroMat = Mat::zeros(J.rows,J.cols,CV_32FC1);
	cout << "BreakPoint 15" << endl;
	Mat outputGrad;

	Mat S0 = Mat::diag(S);
	cout << "BreakPoint 17" << endl;
	exp(-J/epislon,outputGrad);
	cout << "BreakPoint 18" << endl;
	Mat grad = outputGrad/epislon;
	cout << "BreakPoint 19" << endl;
	cout << "lambda: " << lambda << " grad:" << grad.rows << " " << grad.cols << endl;
	Mat t = max(S0 - lambda*grad,0.0);
    cout << "t: " << t.rows << " " << t.cols << endl;
	cout << lambda*grad;
	cout << "BreakPoint 20" << endl;
	Mat tDiag = Mat::diag(t);
	cout << tDiag.rows << " <-rows,tDiag,cols->" << tDiag.cols << endl;
	cout << "BreakPoint 21" << endl;
	cout << U.rows << " <-rows,U,cols->" << U.cols << endl;
	cout << V.rows << " <-rows,V,cols->" << V.cols << endl;
	Mat X = U * tDiag * V;
	cout << "BreakPoint 22" << endl;
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
