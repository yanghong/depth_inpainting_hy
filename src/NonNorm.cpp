#include "NonNorm.h"
#include <fstream>

MatInfo NONNORM(Mat &D,float rho,Mat &T0,float gamma){
	Mat U,S,V,outputT,outputPow;
	MatInfo getMatInfo;
	MatInfo returnMatInfo;
	fstream file;
	file.open("temp.txt");
    
    // set patch
	int patchWidth = 9;
	int patchHeight = 9;

	Mat T0Type;
	T0.convertTo(T0Type, CV_32FC1);
	Mat DType;
	D.convertTo(DType, CV_32FC1);

	int patchI = DType.cols/patchWidth;
    int patchJ = DType.rows/patchHeight;
	file << "patchI: " << patchI << "patchJ: " << patchJ << endl;
	Mat patch;
	
	cout << "patchI: " << patchI << endl;
	cout << "patchJ: " << patchJ << endl;
	for (int pi=0;pi<patchI;pi++) {
		for(int pj=0;pj<patchJ;pj++) {
			int positionX = pi * patchWidth;
			int positionY = pj * patchHeight;
			patch = DType(Range(positionY,positionY + 9),Range(positionX,positionX + 9));
			file << "positionX: " << positionX << endl;
			file << "positionY: " << positionY << endl;
			file << "patch: " << patch << endl;
		}
	
	}

	// SVD -> D
	SVD::compute(DType, S, U, V);

	// itera 1:100
	for (int i = 1; i< 101; i++) {
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
		T0Type = getMatInfo.T;

	}

	Mat T = getMatInfo.T;
	returnMatInfo.X = getMatInfo.X;
	returnMatInfo.T = getMatInfo.T;
	return returnMatInfo;
}

MatInfo DCInner(Mat &S, float rho, Mat &J , float epislon, Mat &U, Mat &V) {
	MatInfo matInfo;
	float lambda = 0.5/rho;
	Mat zeroMat = Mat::zeros(J.rows,J.cols,CV_32FC1);
	Mat outputGrad;

	// Mat S0 = Mat::diag(S);
	exp(-J/epislon,outputGrad);
	Mat grad = outputGrad/epislon;
	Mat t = max(S - lambda*grad,0.0);
	Mat tDiag = Mat::diag(t);
	Mat X = U * tDiag * V;
	matInfo.X = X;
	matInfo.T = t;
	return matInfo;
}

float NonSum(Mat &mat){
	ofstream filetemp3;
	filetemp3.open("temp3.txt");
	filetemp3 << "mat:" << mat << endl;
	float s = 0.0f;
	for(int row=0;row<mat.rows;row++){
		float *data = mat.ptr<float>(row);
		for(int col=0;col<mat.cols;col++) {
			s = s + data[col];
			filetemp3 << "data[" << col << "]:" << data[col] << endl;
		}
	}
	return s;
}
