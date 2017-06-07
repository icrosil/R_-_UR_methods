// Copyright 2015-2017 Illia Olenchenko

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
#include "../utils/out.h"
#include "../utils/functions.h"
#include "../utils/init.h"
#include "../utils/transform.h"
#include "../utils/upper_relaxation.h"
#include "../utils/tools.h"
#include <string>
#include <ctime>
#include <mkl.h>

using namespace std;
using namespace alglib;
using namespace alglib_impl;

#ifndef N
#define N 4
#endif

int main() {
  double t0 = dsecnd();
  /*
   * TODO: add elliptic diffequations
   * TODO: add CUDA improvements
   * эта часть задачи решает по матрице и правой части итерационный процесс верхних релаксаций
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
    cout << "The " << k << " iter" << endl;
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
        // firstAppr[j + 1][i] + firstAppr[j - 1][i] - 4 * firstAppr[j][i])) * wOpt / (DwL(A, i, wOpt)); +
        // firstAppr[j][i];
        firstAppr[j][i] = (-B[j][i] + firstAppr[j + 1][i] + firstAppr[j - 1][i] + firstAppr[j][i - 1] +
          firstAppr[j][i + 1] - 4 * (1 - 1. / wOpt) * firstAppr[j][i]) * wOpt / 4.;
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
  cout << "The iter is:" << endl;
  cout << k << endl;
  firstApprSet(changeAppr);
  //   /*
  //   * outing
  //   */
  // cout<<"The Matr Is:"<<endl;
  // outMatr(A);
  cout << "The Vector Is:" << endl;
  outMatr(B);
  cout << "The first approximation Is:" << endl;
  outMatr(changeAppr);
  cout << "The epsilon Is:" << endl;
  cout << eps << endl;
  cout << "The Vector of ownValues:" << endl;
  outReal1Array(wr);
  cout << "The Spectr Is:" << endl;
  cout << spectr << endl;
  cout << "The wOpt Is:" << endl;
  cout << wOpt << endl;
  cout << "The result Is:" << endl;
  outMatr(firstAppr);
  cout << "The time is:" << endl;
  cout <<  dsecnd() - t0  << " s" << endl;
  cout << "The time of main is:" << endl;
  cout <<  timeChecker  << " s" << endl;
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
