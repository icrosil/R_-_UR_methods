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

double nextTau (vector<double> Tau, double ro0, int n, vector<double> optTau) {
    double tk = cos (( optTau[Tau.size() - 1]) * M_PI / (2 * n));
    // return Tau[0] / (1 + ro0 * tk);
    // if ((Tau[0] / (1 + ro0 * tk) - Tau[Tau.size() - 1]) > 1.) {
        // counter--;
        // return Tau[counter];
    // } else {
        // counter = Tau.size();
        return Tau[0] / (1 + ro0 * tk);
    // }
    // return 2 / (max + min + (max - min) * tk);
}
int findMaxIter (double eps, double ksi, int N) {
    // return ceil(log (2. / eps) / (2. * sqrt(ksi)));
    return ceil(log (2. / eps) / (2. * sqrt(ksi)));
    // return ceil(log (N / eps) * N);
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
void decToDuo(vector<double> &duo, int maxIter) {
    // while (maxIter > 0) {
    //     duo.push_back(maxIter % 2);
    //     maxIter /= 2;
    // }
    vector<double> temp(0);
    while (maxIter > 1) {
        if (maxIter % 2 != 0) {
                temp.push_back(1);
                maxIter--;
        } else {
            temp.push_back(0);
            maxIter /= 2;
        }
    }
    for (int j = 0; j < temp.size(); j++) {
        /* code */
        duo.push_back(temp[temp.size() - j - 1]);
    }
}
void calculateOptTau(vector<double> &optTau, vector<double> duo) {
    vector<double> temp(0);
    double coord = 0;
    double number = 1;
    // duo[0] = 0;
    // duo.erase(duo.begin() + duo.size() - 1);
    for (int i = 0; i < duo.size(); i++) {
        temp.clear();
        coord = 0;
        if (duo[i] == 0) {
            number *= 2;
            if ((i + 1 == duo.size()) || (duo[i + 1] == 0)) {
                /*normal*/
                coord = 0;
            } else {
                /*4m+2*/
                coord = 2;
            }
            for (int j = 0; j < optTau.size(); j++) {
                temp.push_back(optTau[j]);
                temp.push_back(4 * optTau.size() + coord - optTau[j]);
            }
        } else {
            number += 1;
            /*+1*/
            for (int j = 0; j < optTau.size(); j++) {
                temp.push_back(optTau[j]);
            }
            temp.push_back(number);
        }
        optTau.clear();
        for (int j = 0; j < temp.size(); j++) {
            optTau.push_back(temp[j]);
        }
        outVector(optTau);
    }
}

#ifndef N
#define N 25
#endif

int main(){
    double t0 = dsecnd();

  // int N = 30;

  /*
  * Getting inputs A and B
  */
  vector<vector<double> > A((N - 2) * (N - 2), vector<double>((N - 2) * (N - 2), 0));
  readMatr(A);
  vector<vector<double> > B(N, vector<double>(N, 0));
  vector<double> Tau(1, 0);
  vector<vector<double> > firstAppr(N, vector<double>(N, 0));
  vector<vector<double> > tempAppr(N, vector<double>(N, 0));
  firstApprSet(firstAppr);
  readVector(B);
  alglib::real_2d_array matrix;
  matrix.setcontent((N - 2) * (N - 2), (N - 2) * (N - 2), arrToRealArr(A));
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
  alglib::rmatrixevd(matrix, (N - 2) * (N - 2), 0, wr, wi, vl, vr);
  double AlphaMax = findMaxRealArr(wr);
  double AlphaMin = findMinRealArr(wr);
  Tau[0] = 2. / (AlphaMax + AlphaMin);
  double ksi = AlphaMin / AlphaMax;
  double ro0 = (1. - ksi) / (1. + ksi);
  double ro1 = (1. - sqrt(ksi)) / (1. + sqrt(ksi));
  int maxIter = findMaxIter(eps, ksi, N);
  // cout<<"maxIter - "<<maxIter<<endl;
  // return 0;
  maxIter = maxIter * 2;
  vector<double> optTau(1, 1);
  vector<double> duo(0);
  decToDuo(duo, maxIter);
  calculateOptTau(optTau, duo);
  for (int i = 1; i < maxIter + 1; ++i) Tau.push_back(nextTau(Tau, ro0, maxIter, optTau));
  /*
  *main loop here
  */
 double timechecker = dsecnd();
  firstApprSet(tempAppr);
  for (int i = 1; i < maxIter + 1; ++i) {
      cout<<"The "<<i<<" iter"<<endl;
      cout<<"The temp is"<<endl;
      for (int j = 1; j < N - 1; ++j) {
          for (int k = 1; k < N - 1; k++) {
              cout<< (firstAppr[j][k + 1] + firstAppr[j][k - 1] +
firstAppr[j + 1][k] + firstAppr[j - 1][k] - 4 * firstAppr[j][k]) <<" ";
              tempAppr[j][k] = (-B[j][k] + (firstAppr[j][k + 1] + firstAppr[j][k - 1] +
firstAppr[j + 1][k] + firstAppr[j - 1][k] - 4 * firstAppr[j][k])) * Tau[i] + firstAppr[j][k];
          }
      }
      cout<<endl;
      firstAppr = tempAppr;
      outMatr(firstAppr);
      cout<<endl;
  }
  double tMain = dsecnd() - timechecker;
  // for (int i = 1; i < firstAppr.size() - 1; i++) {
  //     for (int j = 1; j < firstAppr.size() - 1; j++) {
  //         firstAppr[i][j] *= N;//((firstAppr.size() - 1) * (firstAppr.size() - 1));
  //     }
  // }
  /*
  * outing
  */
  firstApprSet(tempAppr);
  cout<< "The N is : " << N << endl;
  cout<<"The A(shorted) Is:"<<endl;
  outMatr(A);
  cout<<"The B(shorted) Is:"<<endl;
  outMatr(B);
  cout<<"The duo(shorted) Is:"<<endl;
  outVector(duo);
  cout<<"The opt(shorted) Is:"<<endl;
  outVector(optTau);
  cout<<"The first appr Is:"<<endl;
  outMatr(tempAppr);
  cout<<"The last approximation Is:"<<endl;
  outMatr(firstAppr);
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
  cout<<"The ro1 is:"<<endl;
  cout<<ro1<<endl;
  cout<<"The maxIter is:"<<endl;
  cout<<maxIter<<endl;
  cout<<"The time is:"<<endl;
  cout<< dsecnd() - t0 <<" s"<<endl;
  cout<<"The time of main is:"<<endl;
  cout<< tMain <<" s"<<endl;
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
