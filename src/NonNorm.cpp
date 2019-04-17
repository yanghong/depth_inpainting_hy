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
	int patchWidth = 20;
	int patchHeight = 20;
	int stepsize = 14;
	int m = D.rows;
	int n = D.cols;
	int R = m;
	int C = n;
	list<int> rr;
	list<int> cc;

	for(int si = 1; si < R+1; si++) {
		if(si%stepsize == 1) {
			rr.push_back(si);
		}
	}
	//cout << "rr.back() " << rr.back() << endl;
	//rr.push_back(rr.back()+1);
	for(int sj = 1; sj < C+1; sj++) {
		if(sj%stepsize == 1){
			cc.push_back(sj);
		}
	}
	//cout << "cc.back() " << cc.back() << endl;
	cc.push_back(D.cols);

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
	//cout << "point 1" << endl;
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

			// have to consider patch in edge

			//cout << "pisitionX,Y: " << positionX << " " << positionY << endl;
			int positionXWidth = positionX + patchWidth;
			int positionYWidth = positionY + patchWidth;
			Mat patch;
			// last row in cols
			if (positionX == DType.rows && positionYWidth < DType.cols){

				//cout << "1positionX,Y: " << positionX << " " << positionY << endl;
				//cout << "pisitionX,Y+patchWidth rows: " << positionXWidth << " " << positionYWidth << " DType.size: " << DType.rows << " " <<DType.cols << endl;		
				patch = DType(Range(positionX-1,positionX),Range(positionY,positionY + patchWidth));
			    //cout << "patch.size: " << patch.rows << " " << patch.cols << endl;
				T0Type = T0Type(Range(0,1),Range(0,1));
				//cout << "T0Type.size: " << T0Type.rows << " " << T0Type.cols << endl;
			
			// last col in rows
			} else if (positionY == DType.cols && positionXWidth < DType.rows){

				//cout << "2positionX,Y: " << positionX << " " << positionY << endl;
				//cout << "pisitionX,Y+patchWidth rows: " << positionXWidth << " " << positionYWidth << " DType.size: " << DType.rows << " " <<DType.cols << endl;		
				patch = DType(Range(positionX,positionX + patchWidth),Range(positionY-1,positionY));
			    //cout << "patch.size: " << patch.rows << " " << patch.cols << endl;
				T0Type = T0Type(Range(0,patch.rows),Range(0,1));
				//cout << "T0Type.size: " << T0Type.rows << " " << T0Type.cols << endl;

			// last row over cols
			} else if (positionX == DType.rows && positionYWidth > DType.cols) {

				//cout << "3positionX,Y: " << positionX << " " << positionY << endl;
				//cout << "pisitionX,Y+patchWidth rows: " << positionXWidth << " " << positionYWidth << " DType.size: " << DType.rows << " " <<DType.cols << endl;		
				patch = DType(Range(positionX-1,positionX),Range(positionY,DType.cols));
			    //cout << "patch.size: " << patch.rows << " " << patch.cols << endl;
				T0Type = T0Type(Range(0,1),Range(0,1));
				//cout << "T0Type.size: " << T0Type.rows << " " << T0Type.cols << endl;
			
			// last col over rows
			} else if (positionY == DType.cols && positionXWidth > DType.rows){
			
				//cout << "4positionX,Y: " << positionX << " " << positionY << endl;
				//cout << "pisitionX,Y+patchWidth rows: " << positionXWidth << " " << positionYWidth << " DType.size: " << DType.rows << " " <<DType.cols << endl;		
				patch = DType(Range(positionX,DType.rows),Range(positionY-1,positionY));
			    //cout << "patch.size: " << patch.rows << " " << patch.cols << endl;
				T0Type = T0Type(Range(0,patch.rows),Range(0,1));
				//cout << "T0Type.size: " << T0Type.rows << " " << T0Type.cols << endl;
			
			} else if (positionXWidth > DType.rows && positionX != DType.rows && positionYWidth > DType.cols && positionY != DType.cols) {

				//cout << "5positionX,Y: " << positionX << " " << positionY << endl;
				//cout << "pisitionX,Y+patchWidth rows: " << positionXWidth << " " << positionYWidth << " DType.size: " << DType.rows << " " <<DType.cols << endl;		
				patch = DType(Range(positionX,DType.rows),Range(positionY,DType.cols));
			    //cout << "patch.size: " << patch.rows << " " << patch.cols << endl;
				T0Type = T0Type(Range(0,DType.rows-positionX),Range(0,1));
				//cout << "T0Type.size: " << T0Type.rows << " " << T0Type.cols << endl;

			} else if (positionXWidth > DType.rows && positionX != DType.rows){

				//cout << "6positionX,Y: " << positionX << " " << positionY << endl;
				//cout << "pisitionX,Y+patchWidth rows: " << positionXWidth << " " << positionYWidth << " DType.size: " << DType.rows << " " <<DType.cols << endl;		
				patch = DType(Range(positionX,DType.rows),Range(positionY,positionYWidth));
			    //cout << "patch.size: " << patch.rows << " " << patch.cols << endl;
				T0Type = T0Type(Range(0,DType.rows-positionX),Range(0,1));
				//cout << "T0Type.size: " << T0Type.rows << " " << T0Type.cols << endl;

			} else if (positionYWidth > DType.cols && positionY != DType.cols){

				//cout << "positionX,Y: " << positionX << " " << positionY << endl;
				//cout << "pisitionX,Y+patchWidth cols: " << positionXWidth << " " << positionYWidth <<" DType.cols:  " << DType.cols <<  endl;		
				patch = DType(Range(positionX,positionXWidth),Range(positionY,DType.cols));
			    //cout << "patch.size: " << patch.rows << " " << patch.cols << endl;

			} else {

				//cout << "positionX,Y: " << positionX << " " << positionY << endl;
				//cout << "pisitionX,Y+patchWidth: " << positionXWidth << " " << positionYWidth << endl;
				patch = DType(Range(positionX,positionX + patchWidth),Range(positionY,positionY + patchWidth));

			}

			SVD::compute(patch, S, U, V);
			for (int i101 = 1; i101< 101; i101++) {
				getMatInfo = DCInner(S,rho,T0Type,gamma,U,V);
				//cout << "point-i101-1" <<endl;
				subtract(getMatInfo.T,T0Type,outputT);
				//cout << "point-i101-2" <<endl;
				pow(outputT,2,outputPow);
				//cout << "point-i101-3" <<endl;
				float err;
				for (int row = 0;row< outputPow.rows;row++) {
					float *data = outputPow.ptr<float>(row);
					for(int col=0;col<outputPow.cols;col++) {
						err = err + data[col];
					}   
				}   
				//cout << "point-i101-4" <<endl;
				if(err < 1e-6){
					cout << "err break" << endl;
					 break;
				}   
				//cout << "point-i101-5" <<endl;
				getMatInfo.T.copyTo(T0Type);
				//cout << "point-i101-6" <<endl;
			  }
			// hconcat must be rows equal
			//cout << "point 5.1" << endl;
			//cout << "getMatInfo.size: " << getMatInfo.X.rows << " " << getMatInfo.X.cols << endl;
			Mat resultStepsize;
			if (getMatInfo.X.rows < stepsize && getMatInfo.X.cols < stepsize) {
				//cout << "point 5.2" << endl;
				resultStepsize = getMatInfo.X(Range(0,getMatInfo.X.rows),Range(0,getMatInfo.X.cols));
			} else if (getMatInfo.X.cols < stepsize) {
				//cout << "point 5.3" << endl;
				resultStepsize = getMatInfo.X(Range(0,stepsize),Range(0,getMatInfo.X.cols));
			} else if (getMatInfo.X.rows < stepsize){
				//cout << "point 5.5" << endl;
				resultStepsize = getMatInfo.X(Range(0,getMatInfo.X.rows),Range(0,stepsize));
			} else {
				//cout << "point 5.4" << endl;
				resultStepsize = getMatInfo.X(Range(0,stepsize),Range(0,stepsize));
			}
			//cout << "point 5" << endl;
			if (pj == 0) {
				resultH = resultStepsize;
			} else {
				hconcat(resultH,resultStepsize,resultH);
			}
			//cout << "point 4" << endl;
			count++;
		}
        
		if (resultH.cols < D.cols) {
			//cout << "D.cols: " << D.cols << "resultH.cols: " << resultH.cols << endl;
			int gap = D.cols - resultH.cols;
			Mat resultFill = getMatInfo.X(Range(0,stepsize),Range(patchWidth - gap,patchWidth));
			//cout << "point 3" << endl;
			//cout << "resultH.size: " << resultH.rows << " " << resultH.cols << endl;
			//cout << "resultFill.size: " << resultFill.rows << " " << resultFill.cols << endl;
			hconcat(resultH,resultFill,resultH);
		}
		// vconcat must be columns equal
		if (pi == 0) {
			resultV = resultH;
		} else {
			vconcat(resultV,resultH,resultV);
		}
	}

	//cout << "point 2" << endl;
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

	//cout << "J.size: " << J.rows << " " << J.cols << endl;
	exp(-J/epislon,outputGrad);
	//cout << "point DCInner 1" << endl;
	Mat grad = outputGrad/epislon;
	//cout << "point DCInner 2" << endl;
	Mat t = max(S - lambda*grad,0.0);
	//cout << "point DCInner 3" << endl;
	Mat tDiag = Mat::diag(t);
	//cout << "point DCInner 4" << endl;
	//cout << "U.size: " << U.rows << " " << U.cols << endl;
	//cout << "tDiag.size: " << tDiag.rows << " " << tDiag.cols << endl;
	//cout << "V.size: " << V.rows << " " << V.cols << endl;
	tDiag = tDiag(Range(0,V.rows),Range(0,U.cols));
	Mat X = U * tDiag * V;
	//cout << "point DCInner 5" << endl;
	X.copyTo(matInfo.X);
	t.copyTo(matInfo.T);
	//cout << "point DCInner 6" << endl;
	return matInfo;
}

