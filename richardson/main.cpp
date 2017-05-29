// Copyright 2015-2017 Illia Olenchenko

#include <math.h>
#include <iostream>
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
#include "../utils/out.h"
#include "../utils/functions.h"
#include "../utils/init.h"
#include "../utils/transform.h"
#include "../utils/richardson.h"
#include "../utils/tools.h"
#include <string>
#include <ctime>
#include <mkl.h>

using namespace std;
using namespace alglib;
using namespace alglib_impl;

// few definations
#ifndef N
#define N 4
#endif

int main() {
  // time aggregation
  double t0 = dsecnd();

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
  double eps = 0.01;
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
  int maxIter = findMaxIter(eps, ksi);
  maxIter = maxIter * 2;
  vector<double> optTau(1, 1);
  vector<double> duo(0);
  decToDuo(duo, maxIter);
  calculateOptTau(optTau, duo);
  for (int i = 1; i < maxIter + 1; ++i) Tau.push_back(nextTau(Tau, ro0, maxIter, optTau));
  /*
  *main loop here
  */
  firstApprSet(tempAppr);
  double timechecker = dsecnd();
  for (int i = 1; i < maxIter + 1; ++i) {
    // cout << "The " << i << " iter" <<endl;
    // cout << "The temp is" << endl;
    for (int j = 1; j < N - 1; ++j) {
      for (int k = 1; k < N - 1; k++) {
        // cout << (
        //   firstAppr[j][k + 1] + firstAppr[j][k - 1] + firstAppr[j + 1][k] + firstAppr[j - 1][k] -
        //   4 * firstAppr[j][k])
        // << " ";
        tempAppr[j][k] = (
          -B[j][k] + (firstAppr[j][k + 1] +
            firstAppr[j][k - 1] + firstAppr[j + 1][k] + firstAppr[j - 1][k] - 4 * firstAppr[j][k]))
          * Tau[i] + firstAppr[j][k];
      }
    }
    // cout << endl;
    firstAppr = tempAppr;
    // outMatr(firstAppr);
    // cout << endl;
  }
  double tMain = dsecnd() - timechecker;
  /*
  * outing
  */
  firstApprSet(tempAppr);
  cout <<  "The N is : " << N << endl;
  // cout << "The A(shorted) Is:" << endl;
  // outMatr(A);
  // cout << "The B(shorted) Is:" << endl;
  // outMatr(B);
  // cout << "The duo(shorted) Is:" << endl;
  // outVector(duo);
  // cout << "The opt(shorted) Is:" << endl;
  // outVector(optTau);
  // cout << "The first appr Is:" << endl;
  // outMatr(tempAppr);
  // cout << "The last approximation Is:" << endl;
  // outMatr(firstAppr);
  // cout << "The Max alpha Is:" << endl;
  // cout << AlphaMax << endl;
  // cout << "The Min alpha Is:" << endl;
  // cout << AlphaMin << endl;
  // cout << "The Tau is:" << endl;
  // outVector(Tau);
  // cout << "The ksi is:" << endl;
  // cout << ksi << endl;
  // cout << "The ro0 is:" << endl;
  // cout << ro0 << endl;
  // cout << "The ro1 is:" << endl;
  // cout << ro1 << endl;
  cout << "The maxIter is:" << endl;
  cout << maxIter << endl;
  cout << "The time is:" << endl;
  cout <<  dsecnd() - t0 << " s" << endl;
  cout << "The time of main is:" << endl;
  cout <<  tMain << " s" << endl;
  cout << "The 1 1 is:" << endl;
  cout <<  firstAppr[1][1] << endl;
  cout << "The 2 2 is:" << endl;
  cout <<  firstAppr[2][2] << endl;
  cout << "The N - 2 N - 2 is:" << endl;
  cout <<  firstAppr[firstAppr.size() - 2][firstAppr.size() - 2] << endl;
  cout << "The N - 3 N - 3 is:" << endl;
  cout <<  firstAppr[firstAppr.size() - 3][firstAppr.size() - 3] << endl;
  return 0;
}
