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
double F (double c) {
    return c;
}
void readMatr (vector<vector<double> > &A){
    A[0][0] = 1. ;
    A[A.size() - 1][A.size() - 1] = 1.;
    for (int i = 1; i < A.size() - 1; i++) {
        A[i][i - 1] = -1.;
        A[i][i] = 2.;
        A[i][i + 1] = -1.;
    }
}
void readVector (vector<double> &B){
    B[0] = 0;
    for (int i = 1; i < B.size() - 1; i++) {
        B[i] = F(i / (double) (B.size() - 1));
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
int findMaxIter (double eps, double ksi) {
    // return ceil(log (2. / eps) / (2. * sqrt(ksi)));
    return ceil(log (2. / eps) / (2. * sqrt(ksi)));
}
void firstApprSet(vector<double>& firstAppr) {
    for (int i = 0; i < firstAppr.size(); ++i) {
        firstAppr[i] = F(i / (double) (firstAppr.size() - 1)) / 2;
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
int main(){
    unsigned int start_time =  clock();
    double t0 = dsecnd();

  int N = 100;

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
  double AlphaMax = findMaxRealArr(wr);
  double AlphaMin = findMinRealArr(wr);
  Tau[0] = 2. / (AlphaMax + AlphaMin);
  double ksi = AlphaMin / AlphaMax;
  double ro0 = (1. - ksi) / (1. + ksi);
  double ro1 = (1. - sqrt(ksi)) / (1. + sqrt(ksi));
  int maxIter = findMaxIter(eps, ksi);
  int counter = 0;
  vector<double> optTau(1, 1);
  vector<double> duo(0);
  decToDuo(duo, maxIter);
  calculateOptTau(optTau, duo);
  for (int i = 1; i < maxIter + 1; ++i) Tau.push_back(nextTau(Tau, ro0, maxIter, optTau));

  // for (int i = 1; i < maxIter; ++i) optTau
  // for (int i = N; i < maxIter + 1; ++i) Tau.push_back(Tau[i % N]);

  /*
  *main loop here
  */

  for (int i = 1; i < maxIter + 1; ++i) {
      cout<<"The "<<i<<" iter"<<endl;
      for (int j = 0; j < N; ++j) {
          tempAppr[j] = (B[j] - aMulX(A, firstAppr, j)) * Tau[i] + firstAppr[j];
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
  firstApprSet(tempAppr);
  cout<< "The N is : " << N << endl;
  cout<<"The A(shorted) Is:"<<endl;
  outMatr(A);
  cout<<"The B(shorted) Is:"<<endl;
  outVector(B);
  cout<<"The duo(shorted) Is:"<<endl;
  outVector(duo);
  cout<<"The opt(shorted) Is:"<<endl;
  outVector(optTau);
  cout<<"The first appr Is:"<<endl;
  outVector(tempAppr);
  cout<<"The last approximation Is:"<<endl;
  outVector(firstAppr);
  // cout<<"The Vector of ownValues:"<<endl;
  // outReal1Array(wr);
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
  // unsigned int end_time = clock(); // конечное время
  // unsigned int search_time = end_time - start_time; // искомое время
  cout<<"The time is:"<<endl;
  cout<< dsecnd() - t0 <<" s"<<endl;
  return 0;
}
