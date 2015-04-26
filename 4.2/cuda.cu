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

/**
 * CUDA functions
 */

__global__ void hello (void) {

}

/**
 * Additional functions
 */
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
    return c;
}
void readMatr (vector<vector<double> > &A){
    A[0][0] = 1;
    A[A.size() - 1][A.size() - 1] = 1;
    for (int i = 1; i < A.size() - 1; i++) {
        A[i][i - 1] = -1. ;/// ((A.size() - 1) * (A.size() - 1));
        A[i][i] = 2.;// / ((A.size() - 1) * (A.size() - 1));
        A[i][i + 1] = -1.;// / ((A.size() - 1) * (A.size() - 1));
    }
}
void readVector (vector<double> &B){
    B[0] = 0;
    cout<<"The B:"<<endl;
    for (int i = 1; i < B.size() - 1; i++) {
        B[i] = F(i / (B.size() - 1.));
        cout<<B[i]<<endl;
    }
    B[B.size() - 1] = 0;
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
    double max = wr[0];
    for (int i = 1; i < wr.length(); ++i) {
        if (wr[i] > max) max = wr[i];
    }
    return max;
}
double findMinRealArr (alglib::real_1d_array const wr) {
    double min = wr[0];
    for (int i = 1; i < wr.length(); ++i) {
        if (wr[i] < min) min = wr[i];
    }
    return min;
}

double nextTau (vector<double> Tau, double ro0, int n) {
    double tk = cos ((2 * Tau.size() - 1) * M_PI / (2 * n));
    return Tau[0] / (1 + ro0 * tk);
}
/*
* TODO: make clear is it correct
*/
int findMaxIter (double eps, double ksi) {
    return floor(log (2 / eps) / (2 * sqrt(ksi)));
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

/**
 * main entrance
 */
int main(void){
    unsigned int start_time =  clock();
    double t0 = dsecnd();
  /*
  *TODO: add elliptic diffequations
  *TODO: add CUDA improvements
  *эта часть задачи решает по матрице и правой части итерационный процесс.
  */
  /*
  * N means matr size
  * A means main Matr
  * B means right vector
  */
  int N = 10;
  /*
  * Getting inputs A and B
  */
  vector<vector<double> > A(N, vector<double>(N, 0));
  readMatr(A);
  vector<double> B(N, 0);
  vector<double> Tau(1, 0);
  vector<double> firstAppr(N, 0);
  vector<double> tempAppr(N, 0);
  firstApprSet(firstAppr);
  readVector(B);
  alglib::real_2d_array matrix;
  matrix.setcontent(N, N, arrToRealArr(A));
  double eps = 0.00001;
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
  *TODO: make clear here no imaginarium parts
  *TODO: make clear here good A matr and Alphas > 0
  */
  double AlphaMax = findMaxRealArr(wr);
  double AlphaMin = findMinRealArr(wr);
  Tau[0] = 2 / (AlphaMax + AlphaMin);
  double ksi = AlphaMin / AlphaMax;
  double ro0 = (1 - ksi) / (1 + ksi);
  int maxIter = findMaxIter(eps, ksi);

  for (int i = 1; i < maxIter + 1; ++i) Tau.push_back(nextTau(Tau, ro0, maxIter));

  /*
  *main loop here
  */

  for (int i = 1; i < maxIter; ++i) {
      cout<<"The "<<i<<" iter"<<endl;
      for (int j = 0; j < N; ++j) {
          tempAppr[j] = (B[j] - aMulX(A, firstAppr, j)) * Tau[i - 1] + firstAppr[j];
      }
      firstAppr = tempAppr;
      outVector(firstAppr);
      cout<<endl;
  }


  for (int i = 0; i < firstAppr.size(); i++) {
      firstAppr[i] /= ((firstAppr.size() - 1) * (firstAppr.size() - 1));
  }
  /*
  * outing
  */
  cout<<"The Matr Is:"<<endl;
  outMatr(A);
  cout<<"The Vector Is:"<<endl;
  outVector(B);
  cout<<"The last approximation Is:"<<endl;
  outVector(firstAppr);
  cout<<"The Vector of ownValues:"<<endl;
  outReal1Array(wr);
  cout<<"The Max alpha Is:"<<endl;
  cout<<AlphaMax<<endl;
  cout<<"The Min alpha Is:"<<endl;
  cout<<AlphaMin<<endl;
  cout<<"The Tau is:"<<endl;
  outVector(Tau);
  cout<<"The ksi is:"<<endl;
  cout<<ksi<<endl;
  cout<<"The ro0 is:"<<endl;
  cout<<ro0<<endl;
  cout<<"The maxIter is:"<<endl;
  cout<<maxIter<<endl;
  // unsigned int end_time = clock(); // конечное время
  // unsigned int search_time = end_time - start_time; // искомое время
  cout<<"The time is:"<<endl;
  cout<< dsecnd() - t0 <<" ms"<<endl;
  // hello<<<1, 1>>>();
  return 0;
}
