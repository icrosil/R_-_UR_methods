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
#include <ctime>
#include <mkl.h>

using namespace std;
using namespace alglib;
using namespace alglib_impl;

void outVector (vector<double> B){
    // int additions = B.size() / 4;
    int additions = 1;
    cout<< B[0] << " ";
    for (int i = additions; i < B.size() - 1; i += additions){
        cout<< B[i] << " ";
    }
    cout<< B[B.size() - 1] << " ";
    cout<<endl;
}
void outMatr (vector<vector<double> > A){
    // int additions = A.size() / 4;
    int additions = 1;
    outVector(A[0]);
    for (int i = additions; i < A.size() - 1; i += additions){
        outVector(A[i]);
    }
    outVector(A[A.size() - 1]);
}
double F (double x, double y, double N) {
    // cout<<x<<" "<<y<<endl;
    return (2 * sin(y) - x * x * sin(y)) / N;
}
double U (double x, double y) {
    // cout<<x<<" "<<y<<endl;
    return x * x * sin(y) + 1;
}
void readMatr (vector<vector<double> > &A){
    int sizer = (int) sqrt(A.size());
    A[0][0] = -4;
    A[A.size() - 1][A.size() - 1] = -4;
    A[0][1] = 1;
    A[A.size() - 1][A.size() - 2] = 1;
    for (int i = 1; i < A.size() - 1; i++) {
        A[i][i] = -4;
        if (!((i % sizer) == 0)) {
            A[i][i - 1] = 1;
        }
        if (!(((i - 1) % sizer) == 0)) {
            A[i][i + 1] = 1;
        }
    }
    for (int i = 0; i < A.size() - sizer; i++) {
        A[i][i + sizer] = 1;
        A[i + sizer][i] = 1;
    }
}
void readVector (vector<vector<double> >& B){
    for (int i = 0; i < B.size(); i++) {
        B[i][0] = U(i / (double) (B.size() - 1), 0);
        B[0][i] = U(0, i / (double) (B.size() - 1));
        B[B.size() - 1][i] = U(1, i / (double) (B.size() - 1));
        B[i][B.size() - 1] = U(i / (double) (B.size() - 1), 1);
    }
    for (int i = 1; i < B.size() - 1; ++i) {
        for (int j = 1; j < B.size() - 1; ++j) {
            B[i][j] = F(i / (double) (B.size() - 1), j / (double) (B.size() - 1), (B.size() - 1) * (B.size() - 1));
        }
    }
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
double findMaxRealArr (alglib::real_1d_array const wr) {
    double max = fabs(wr[0]);
    for (int i = 1; i < wr.length(); ++i) {
        if (fabs(wr[i]) > max) max = fabs(wr[i]);
    }
    return max;
}
double findMinRealArr (alglib::real_1d_array const wr) {
    double min = fabs(wr[0]);
    for (int i = 1; i < wr.length(); ++i) {
        if (fabs(wr[i]) < min) min = fabs(wr[i]);
    }
    return min;
}

void outReal2Array (alglib::real_2d_array wr, int size) {
    for (int i = 0; i < size; ++i ){
        for (int j = 0; j < size; ++j )
            cout<<wr[i][j]<<" ";
        cout<<endl;
    }
}



void firstApprSet(vector<vector<double> >& B) {
    for (int i = 0; i < B.size(); i++) {
        B[i][0] = U(i / (double) (B.size() - 1), 0);
        B[0][i] = U(0, i / (double) (B.size() - 1));
        B[B.size() - 1][i] = U(1, i / (double) (B.size() - 1));
        B[i][B.size() - 1] = U(i / (double) (B.size() - 1), 1);
    }
    for (int i = 1; i < B.size() - 1; ++i) {
        for (int j = 1; j < B.size() - 1; ++j) {
            B[i][j] = F(i / (double) (B.size() - 1), j / (double) (B.size() - 1), (B.size() - 1) * (B.size() - 1)) / 2.;
        }
    }
}
double aMulX(vector<vector<double> > A, vector<double> X, int j){
    double res = 0;
    for (int i = 0; i < A.size(); ++i){
        res += A[j][i] * X[i];
    }
    return res;
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
*совсем не факт что это штука с детерминантом оптимальная но другого у меня нет.
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
    wOpt = 2 / (1 + sqrt(1 - spectr * spectr ));//* rDet));
    if (wOpt < 1 && wOpt > 0) {
        wOpt +=1;
    } else if (wOpt < 2 && wOpt > 1){
        wOpt = wOpt;
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

double DwL (vector<vector<double> > A, int k, double w) {
    double sum = 0;
    for (int i = 0; i < k; ++i) {
        sum += A[k][i] * w;
    }
    sum += A[k][k];
    return sum;
}

double findMaxInVector ( vector<vector<double> > a) {
    double max = a[1][1];
    for (int i = 2; i < a.size() - 1; i++) {
        for (int j = 2; j < a.size() - 1; j++) {
            if (a[i][j] > max) max = a[i][j];
        }
    }
    return max;
}

void copyVectors (vector<vector<double> > in, vector<vector<double> > &out) {
    for (int i = 0; i < in.size(); i++) {
        for (int j = 0; j < in[i].size(); j++) {
            out[i][j] = in[i][j];
        }
    }
}

#ifndef N
#define N 25
#endif

int main(){
    double t0 = dsecnd();
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

  /*
  * Getting inputs A and B
  */
  vector<vector<double> > A((N - 2) * (N - 2), vector<double>((N - 2) * (N - 2), 0));
  readMatr(A);
  vector<vector<double> > B(N, vector<double>(N, 0));
  vector<vector<double> > firstAppr(N, vector<double>(N, 0));
  vector<vector<double> > changeAppr(N, vector<double>(N, 0));
  firstApprSet(firstAppr);
  readVector(B);
  double eps = 0.0001;
  double spectr;
  double wOpt;
  double maxDiff = 1;
  alglib::real_2d_array matrix;
  matrix.setcontent((N - 2) * (N - 2), (N - 2) * (N - 2), arrToRealArr(A));

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
  wOpt = wOptSet(A, spectr, 1. / N);

  /*
  *main loop here
  *если я правильно понял то новые вычисления нужно тут же использовать, исхожу из этого мнения
  */
  int k = 0;
  char aber;
  double timeChecker = dsecnd();
  do {
        cout<<"The "<<k<<" iter"<<endl;
        copyVectors(firstAppr, changeAppr);
        // cout<<"change: "<<endl;
        // outMatr(changeAppr);
        // cout<<"fa: "<<endl;
        // outMatr(firstAppr);
        // cin>>aber;
        // for (int i = 0; i < A.size(); i++) {
        //     firstAppr[i] = firstAppr[i] + (B[i] - aMulX(A, firstAppr, i)) * wOpt / (DwL(A, i, wOpt));
        // }
        for (int j = 1; j < N - 1; ++j) {
            for (int i = 1; i < N - 1; i++) {
  //               firstAppr[j][i] = (B[j][i] - (firstAppr[j][i + 1] + firstAppr[j][i - 1] +
  // firstAppr[j + 1][i] + firstAppr[j - 1][i] - 4 * firstAppr[j][i])) * wOpt / (DwL(A, i, wOpt)); + firstAppr[j][i];
                firstAppr[j][i] = (-B[j][i] + firstAppr[j + 1][i] + firstAppr[j - 1][i] + firstAppr[j][i - 1] + firstAppr[j][i + 1] - 4 * (1 - 1. / wOpt) * firstAppr[j][i]) * wOpt / 4.;
            }
        }
        for (int j = 1; j < N - 1; ++j) {
            for (int i = 1; i < N - 1; i++) {
                changeAppr[j][i] = fabs(firstAppr[j][i] - changeAppr[j][i]);
            }
        }
        outMatr(firstAppr);
        // outVector(changeAppr);
        // cout<<findMaxInVector(changeAppr)<<endl;
        maxDiff = findMaxInVector(changeAppr);
        // system("pause");
    ++k;
} while (maxDiff > eps);
timeChecker = dsecnd() - timeChecker;
cout<<"The iter is:"<<endl;
cout<<k<<endl;
firstApprSet(changeAppr);
//   /*
//   * outing
//   */
  // cout<<"The Matr Is:"<<endl;
  // outMatr(A);
  cout<<"The Vector Is:"<<endl;
  outMatr(B);
  cout<<"The first approximation Is:"<<endl;
  outMatr(changeAppr);
  cout<<"The epsilon Is:"<<endl;
  cout<<eps<<endl;
  cout<<"The Vector of ownValues:"<<endl;
  outReal1Array(wr);
  cout<<"The Spectr Is:"<<endl;
  cout<<spectr<<endl;
  cout<<"The wOpt Is:"<<endl;
  cout<<wOpt<<endl;
  cout<<"The result Is:"<<endl;
  outMatr(firstAppr);
  cout<<"The time is:"<<endl;
  cout<< dsecnd() - t0 <<" s"<<endl;
  cout<<"The time of main is:"<<endl;
  cout<< timeChecker <<" s"<<endl;
  cout<<"The 1 1 is:"<<endl;
  cout<< firstAppr[1][1]<<endl;
  cout<<"The 2 2 is:"<<endl;
  cout<< firstAppr[2][2]<<endl;
  cout<<"The N - 2 N - 2 is:"<<endl;
  cout<< firstAppr[firstAppr.size() - 2][firstAppr.size() - 2]<<endl;
  cout<<"The N - 3 N - 3 is:"<<endl;
  cout<< firstAppr[firstAppr.size() - 3][firstAppr.size() - 3]<<endl;
  return 0;
}
