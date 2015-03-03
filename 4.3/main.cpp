//compile g++ -o filename file.cpp and run filename
#include <iostream>
#include <math.h>
#include "vector"
#include "../lib/alglib/src/ap.h"
#include "../lib/alglib/src/alglibmisc.h"
#include "../lib/alglib/src/alglibinternal.h"
#include "../lib/alglib/src/linalg.h"
#include "../lib/alglib/src/statistics.h"
#include "../lib/alglib/src/dataanalysis.h"
#include "../lib/alglib/src/specialfunctions.h"
#include "../lib/alglib/src/solvers.h"
#include "../lib/alglib/src/optimization.h"
#include "../lib/alglib/src/diffequations.h"
#include "../lib/alglib/src/fasttransforms.h"
#include "../lib/alglib/src/integration.h"
#include "../lib/alglib/src/interpolation.h"
#include "../lib/alglib/src/ap.cpp"
#include "../lib/alglib/src/linalg.cpp"
#include "../lib/alglib/src/alglibmisc.cpp"
#include "../lib/alglib/src/alglibinternal.cpp"
#include "../lib/alglib/src/statistics.cpp"
#include "../lib/alglib/src/dataanalysis.cpp"
#include "../lib/alglib/src/specialfunctions.cpp"
#include "../lib/alglib/src/solvers.cpp"
#include "../lib/alglib/src/optimization.cpp"
#include "../lib/alglib/src/diffequations.cpp"
#include "../lib/alglib/src/fasttransforms.cpp"
#include "../lib/alglib/src/integration.cpp"
#include "../lib/alglib/src/interpolation.cpp"
#include <string>

using namespace std;
using namespace alglib;
using namespace alglib_impl;

void outMatr (vector<vector<double> > A){
    for (int i = 0; i < A.size(); ++i){
        for (int j = 0; j < A[i].size(); ++j){
            cout<<A[i][j]<<" ";
        }
        cout<<endl;
    }
}
void outVector (vector<double> B){
    for (int i = 0; i < B.size(); ++i){
        cout<< B[i]<<" ";
    }
    cout<<endl;
}
double F (double c) {
    return c * c;
}
void readMatr (vector<vector<double> > &A){
    A[0][0] = 2;
    A[0][1] = 1;
    A[1][0] = 1;
    A[1][1] = 2;
}
void readVector (vector<double> &B){
    B[0] = 4;
    B[1] = 5;
}
double* arrToRealArr (vector<vector<double> >const &A){
    double * local;
    local = new double[A.size() * A.size()];
    for (int i = 0; i < A.size(); ++i){
        for (int j = 0; j < A[i].size();++j){
            local[i * A.size() + j] = A[i][j];
        }
    }
    return local;
}
void outReal1Array (alglib::real_1d_array wr) {
    for (int i = 0; i < wr.length(); ++i )
    cout<<wr[i]<<" ";
    cout<<endl;
}

void outReal2Array (alglib::real_2d_array wr, int size) {
    for (int i = 0; i < size; ++i ){
        for (int j = 0; j < size; ++j )
            cout<<wr[i][j]<<" ";
        cout<<endl;
    }
}



void firstApprSet(vector<double>& firstAppr) {
    for (int i = 1; i < firstAppr.size(); ++i) {
        firstAppr[i] = i;
    }
}
double aMulX(vector<vector<double> > A, vector<double> X, int j){
    double res = 0;
    for (int i = 0; i < A.size(); ++i){
        res += A[j][i] * X[i];
    }
    return res;
}

double findMaxRealArr (alglib::real_1d_array wr) {
    double max = wr[0];
    for (int i = 1; i < wr.length(); ++i){
        if (wr[i] > max) max = wr[i];
    }
    return max;
}

void realArr2dToVectorMatr (alglib::real_2d_array matrix, vector<vector<double> > &A) {
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A.size(); ++j) {
            A[i][j] = matrix[i][j];
        }
    }
}

/*
*works only for good square matricies
*/
void mulMatricies (vector<vector<double> > A, vector<vector<double> > B, vector<vector<double> > &temp) {
    for (int i = 0; i < A.size(); ++i)
        for (int j = 0; j < A.size(); ++j)
            for (int k = 0; k < A.size(); ++k)
                temp[i][j] += A[k][j] * B[i][k];
}

/*
*можно найти оптимальный w как 2 - O(h) но это же слишком просто, по этому пишем эту фиготень
*/
double wOptSet ( vector<vector<double> > A, double spectr, double oh) {
    vector<vector<double> > D(A.size(), vector<double>(A.size(), 0));
    vector<vector<double> > LR(A.size(), vector<double>(A.size(), 0));
    vector<vector<double> > det(A.size(), vector<double>(A.size(), 0));
    double rDet;
    double wOpt;
    alglib::ae_int_t info;
    alglib::matinvreport rep;
    alglib::real_2d_array matrixD;
    alglib::real_2d_array matrixLR;
    alglib::real_2d_array matrixDet;
    for (int i = 0; i < A.size(); ++i ){
        D[i][i] = A[i][i];
    }
    for (int i = 0; i < A.size(); ++i )
        for (int j = 0; j < A.size(); ++j){
            if (i != j) LR[i][j] = A[i][j];
        }
    matrixD.setcontent(A.size(), A.size(), arrToRealArr(D));
    matrixLR.setcontent(A.size(), A.size(), arrToRealArr(LR));
    alglib::rmatrixinverse(matrixD, A.size(), info, rep);
    realArr2dToVectorMatr(matrixD, D);
    mulMatricies(D, LR, det);
    matrixDet.setcontent(A.size(), A.size(), arrToRealArr(det));
    // outMatr(det);
    // outReal2Array(matrixDet, det.size());
    rDet = alglib::rmatrixdet(matrixDet);
    wOpt = 2 / (1 + sqrt(1 - spectr * spectr * rDet));
    if (wOpt < 1 && wOpt > 0) {
        wOpt +=1;
    } else {
        wOpt = 2 - oh;
    }
    // cout<<"The determ is "<<rDet<<endl;
    // cout<<"The wopt is "<<wOpt<<endl;
    //checking inversed
    // cout<<"The D inverse Is:"<<endl;
    // to find determenant use RMatrixDet

    return wOpt;
}



int main(){

  /*
  *TODO: add elliptic diffequations
  *TODO: add CUDA improvements
  *эта часть задачи решает по матрице и правой части итерационный процесс верхних релаксаций
  */
  /*
  * N means matr size
  * A means main Matr
  * B means right vector
  */
  int N = 2;
  /*
  * Getting inputs A and B
  */
  vector<vector<double> > A(N, vector<double>(N, 0));
  readMatr(A);
  vector<double> B(N, 0);
  vector<double> firstAppr(N, 0);
  firstApprSet(firstAppr);
  readVector(B);
  double eps = 0.0001;
  double spectr;
  double wOpt;
  double maxDiff = 1;
  alglib::real_2d_array matrix;
  matrix.setcontent(N, N, arrToRealArr(A));

  /*
  *creating another parts
  *wr - целые части собственных чисел
  *wi - мнимые части собственных чисел
  *vl - собственный левый вектор
  *vr - собственный правый вектор
  */
  alglib::real_1d_array wr;
  alglib::real_1d_array wi;
  alglib::real_2d_array vl;
  alglib::real_2d_array vr;
  /*
  * расчет собственных чисел
  */
  alglib::rmatrixevd(matrix, N, 0, wr, wi, vl, vr);

  /*
  *допустим что спектральынй радиус матрицы это максимальное собственное число (которые все норм должны быть) без модуля, так как все должны быть положительны
  */
  spectr = findMaxRealArr(wr);
  wOpt = wOptSet(A, spectr, 1 / N);

  /*
  *main loop here
  *если я правильно понял то новые вычисления нужно тут же использовать, исхожу из этого мнения
  */
  do {
      for (int i = 0; i < A.size(); i++) {
        //   firstAppr[i] = firstAppr[i] + (B[i] - aMulX(A, firstAppr, i)) * w / ()
      }
  } while (maxDiff > eps)

  /*
  * outing
  */
  cout<<"The Matr Is:"<<endl;
  outMatr(A);
  cout<<"The Vector Is:"<<endl;
  outVector(B);
  cout<<"The first approximation Is:"<<endl;
  outVector(firstAppr);
  cout<<"The epsilon Is:"<<endl;
  cout<<eps<<endl;
  cout<<"The Vector of ownValues:"<<endl;
  outReal1Array(wr);
  cout<<"The Spectr Is:"<<endl;
  cout<<spectr<<endl;
  cout<<"The wOpt Is:"<<endl;
  cout<<wOpt<<endl;
  return 0;
}
