//compile g++ -o filename file.cpp and run filename
#include <iostream>
#include <math.h>
#include "vector"
#include "alglib/src/ap.h"
#include "alglib/src/alglibmisc.h"
#include "alglib/src/alglibinternal.h"
#include "alglib/src/linalg.h"
#include "alglib/src/statistics.h"
#include "alglib/src/dataanalysis.h"
#include "alglib/src/specialfunctions.h"
#include "alglib/src/solvers.h"
#include "alglib/src/optimization.h"
#include "alglib/src/diffequations.h"
#include "alglib/src/fasttransforms.h"
#include "alglib/src/integration.h"
#include "alglib/src/interpolation.h"
#include "alglib/src/ap.cpp"
#include "alglib/src/linalg.cpp"
#include "alglib/src/alglibmisc.cpp"
#include "alglib/src/alglibinternal.cpp"
#include "alglib/src/statistics.cpp"
#include "alglib/src/dataanalysis.cpp"
#include "alglib/src/specialfunctions.cpp"
#include "alglib/src/solvers.cpp"
#include "alglib/src/optimization.cpp"
#include "alglib/src/diffequations.cpp"
#include "alglib/src/fasttransforms.cpp"
#include "alglib/src/integration.cpp"
#include "alglib/src/interpolation.cpp"
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
int main(){

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
  int N = 2;
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
  double eps = 0.0001;
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
  /*
  * outing
  */
  cout<<"The Matr Is:"<<endl;
  outMatr(A);
  cout<<"The Vector Is:"<<endl;
  outVector(B);
  cout<<"The first approximation Is:"<<endl;
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
  return 0;
}
