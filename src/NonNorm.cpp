#include "NonNorm.h"
#include <fstream>
#include <cmath>
#include <list>

MatInfo NONNORM(Mat &D,float rho,Mat &T0,float gamma, int patchSize){
	Mat U,S,V,outputT,outputPow;
	MatInfo getMatInfo;
	MatInfo returnMatInfo;
	ofstream file("temp.txt",ios::app);
    int count=0;
    // set patch
	int patchWidth = patchSize;
	int patchHeight = patchSize;
	int stepsize = 15;
	int m = D.rows;
	int n = D.cols;
	int R = m-patchWidth+1;
	int C = n-patchWidth+1;
	list<int> rr;
	list<int> cc;

	//cout << "brakpoint 2" << endl;

	for(int si = 1; si < R; si++) {
		if(si%stepsize == 1) {
			rr.push_back(si);
		}
	}
	rr.push_back(rr.back()+stepsize);
	for(int sj = 1; sj < C; sj++) {
		if(sj%stepsize == 1){
			cc.push_back(sj);
		}
	}
	cc.push_back(cc.back()+stepsize);

	//cout << "brakpoint 3" << endl;

	int rrLength = rr.size();
    int ccLength = cc.size();
	list<int>::iterator rrIter;
	list<int>::iterator ccIter;
	rrIter = rr.begin();
	ccIter = cc.begin();
    //cout << "rrLength: " << rrLength << endl;
	//cout << "ccLength: " << ccLength << endl;

	//cout << "brakpoint 4" << endl;

	Mat T0Type;
	T0.convertTo(T0Type, CV_32FC1);
	Mat DType;
	D.convertTo(DType, CV_32FC1);

	//cout << "brakpoint 5" << endl;

	int patchI = DType.cols/patchWidth;
    int patchJ = DType.rows/patchHeight;
	Mat result(D.rows,D.cols,CV_32FC1);
	Mat resultV(patchWidth,D.cols,CV_32FC1);

	int positionX = 0;
	int positionY = 0;

	Mat resultHTemp;
	Mat resultFill;
	Mat resultStepsize;

	//cout << "brakpoint 6" << endl;
	
	int pi=0;
	int pj=0;

	for (pi;pi<rrLength;pi++) {
			//cout << "pi: " << pi << endl;
			Mat resultH(stepsize,stepsize,CV_32FC1);
			pj = 0;
			if(rrIter != rr.end()) {
				positionX = *rrIter;
				++rrIter;
				ccIter = cc.begin();
			}
		for(pj;pj<ccLength;pj++) {
			if(ccIter != cc.end()) {
				positionY = *ccIter;
				++ccIter;
			}
			//cout << "breakpoint 6.6" << endl;
			int positionYPlus = positionY + patchWidth;
			int positionXPlus = positionX + patchWidth;
			if (positionYPlus > DType.cols) {
				positionYPlus = DType.cols;
			}
			if (positionXPlus > DType.rows) {
				positionXPlus = DType.rows;
			}
			Mat patch = DType(Range(positionX,positionXPlus),Range(positionY,positionYPlus));
			//cout << "breakpoint 6.5" << endl;
			SVD::compute(patch, S, U, V);
			//cout << "breakpoint 6.2" << endl;
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
					break;}   
				getMatInfo.T.copyTo(T0Type);		
			}
			// hconcat must be rows equal
			//cout << "breakpoint 6.1" << endl;
			//cout << "getMatInfo.size: " << getMatInfo.X.rows << " " << getMatInfo.X.cols << endl;
			if (getMatInfo.X.cols < stepsize) {
				resultStepsize = getMatInfo.X(Range(0,stepsize),Range(0,getMatInfo.X.cols));
			} else {
				resultStepsize = getMatInfo.X(Range(0,stepsize),Range(0,stepsize));
			}
			if (pj == 0) {
				resultH = resultStepsize;
			} else {
				hconcat(resultH,resultStepsize,resultH);
				cout << "breakpoint 6.3" << endl;
				}
			count++;
			//cout << "resultH.cols: " << resultH.cols << " rows:"<< resultH.rows << " D.cols: " << D.cols << endl; 
		    }
			//cout << "breakpoint 7" << endl;
		if (resultH.cols < D.cols && resultH.cols > D.cols-stepsize) {
			int gap = D.cols - resultH.cols;
			//cout << "gap : " << gap << endl;
			//cout << "getMatInfo.X.cols: " << getMatInfo.X.cols << endl;
			resultFill = getMatInfo.X(Range(0,stepsize),Range(getMatInfo.X.cols - gap,getMatInfo.X.cols));
			cout << "resultFill.size: " << resultFill.rows << " " << resultFill.cols << endl;
			hconcat(resultH,resultFill,resultH);
		}
		//cout << "breakpoint 8" << endl;
		// vconcat must be columns equal
		if (pi == 0) {
			resultV = resultH;
		} else {
			vconcat(resultV,resultH,resultV);
			//if (pj >= rrLength) {	
			cout << "resultV: " << resultV.rows << " " << resultV.cols << endl;
			//}
		}
	}
    resize(resultV,result,result.size(),INTER_LINEAR);
	returnMatInfo.X = result;
	returnMatInfo.T = getMatInfo.T;
	return returnMatInfo;
}

MatInfo DCInner(Mat &S, float rho, Mat &J , float epislon, Mat &U, Mat &V) {
	ofstream file2("temp2.txt",ios::app);
//	file2 << "J : " << J << endl;
	MatInfo matInfo;
	float lambda = rho;
	//cout << "breakpoint 10 " << endl;
	//Mat zeroMat = Mat::zeros(J.rows,J.cols,CV_32FC1);
	//cout << "breakpoint 11 " << endl;
	Mat outputGrad;

    //zeroMat.at<float>(0,0) = J.at<float>(0,0);
	//zeroMat.at<float>(0,1) = J.at<float>(0,1);
	//zeroMat.at<float>(0,2) = J.at<float>(0,2);
	//zeroMat.at<float>(0,3) = J.at<float>(0,3);
	//zeroMat.at<float>(0,4) = J.at<float>(0,4);
	//zeroMat.at<float>(0,5) = J.at<float>(0,5);
	//zeroMat.at<float>(0,6) = J.at<float>(0,6);
	//zeroMat.at<float>(0,4) = J.at<float>(0,4);
	//zeroMat.at<float>(0,4) = J.at<float>(0,4);
	//zeroMat.at<float>(0,4) = J.at<float>(0,4);

	exp(-J/epislon,outputGrad);
	//exp(-zeroMat/epislon,outputGrad);
	//cout << "breakpoint 12 " << endl;
	Mat grad = outputGrad/epislon;
	//cout << "breakpoint 13 " << endl;
	Mat t = max(S - lambda*grad,0.0);
	//cout << "breakpoint 14 " << endl;
	Mat tDiag = Mat::diag(t);
	//cout << "breakpoint 15 " << endl;
	//cout << "U.size: " << U.rows << " " << U.cols << endl;
	//cout << "tDiag.size: " << tDiag.rows << " " << tDiag.cols << endl;
	//cout << "V.size: " << V.rows << " " << V.cols << endl;
	tDiag = tDiag(Range(0,V.rows),Range(0,U.cols));
	Mat X = U * tDiag * V;
	//cout << "breakpoint 16 " << endl;
	X.copyTo(matInfo.X);
	t.copyTo(matInfo.T);
	return matInfo;
}

