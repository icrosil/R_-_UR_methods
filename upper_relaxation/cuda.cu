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

/**
 * CUDA functions
 */
#ifndef N
#define N 5
#endif

__device__ int blocks = N - 2;

__global__ void mykernel(double *rightSide, double wOpt, double *fa, double *diff, int n, double *all, int i, int j,
  int litN) {
  // tempAppr[j] = (B[j] - aMulX(A, firstAppr, j)) * Tau[i - 1] + firstAppr[j];

  int index = threadIdx.x;  //+ blockIdx.x * blockDim.x;
  int row = (i - j + 1) / 2;
  int index1 = row * litN + (((j - i) == 1)? j : litN) - 1 + index * (litN - 1);
  int lindex = index1 + N + 1 + 2 * (int) (index1 / (N - 2));
  if (index1 < n) {
    //  fa[index1] = index1;
    fa[index1] = (-rightSide[index1] + all[lindex - N] + all[lindex + N] + all[lindex - 1] + all[lindex + 1] -
      4 * (1 - 1. / wOpt) * all[lindex]) * wOpt / 4.;
    diff[index1] = fa[index1] - all[lindex];
    all[lindex] = fa[index1];
  }
}
__global__ void my_red_black_kernel(double *rightSide, double wOpt, double *fa, double *diff, int n, double *all,
  int litN, int first) {
  // tempAppr[j] = (B[j] - aMulX(A, firstAppr, j)) * Tau[i - 1] + firstAppr[j];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = index + N + 1 + 2 * (int) (index / (N - 2));
  int row = (int)(index / litN);
  int str = index % litN;
  if (index < n && ((row + str) % 2 == first)) {
    // fa[index] = index;
    fa[index] = (-rightSide[index] + all[lindex - N] + all[lindex + N] + all[lindex - 1] + all[lindex + 1] -
      4 * (1 - 1. / wOpt) * all[lindex]) * wOpt / 4.;
    diff[index] = fa[index] - all[lindex];
    all[lindex] = fa[index];
  }
}

int main() {
  double t0 = dsecnd();
  /*
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
  double maxDiff = 0;
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
  // char aber;
  double *rightSide = new double[N * N - 4 * N + 4];
  double *fa = new double[N * N - 4 * N + 4];
  double *all = new double[N * N];
  double *diff = new double[N * N - 4 * N + 4];
  double *d_rs, *d_fa, *d_all, *d_diff;
  int size = sizeof(double);

  cudaMalloc((void **)&d_rs, size * (N * N - 4 * N + 4));
  cudaMalloc((void **)&d_all, size * (N * N));
  cudaMalloc((void **)&d_fa, size * (N * N - 4 * N + 4));
  cudaMalloc((void **)&d_diff, size * (N * N - 4 * N + 4));
  for (int j = 1; j < N - 1; j++) {
    for (int k = 1; k < N - 1; k++) {
      rightSide[(j - 1) * (N - 2) + (k - 1)] = B[j][k];
      fa[(j - 1) * (N - 2) + (k - 1)] = firstAppr[j][k];
      diff[(j - 1) * (N - 2) + (k - 1)] = 1;
    }
  }
  for (int j = 0; j < N ; j++) {
    for (int k = 0; k < N; k++) {
      all[j * N + k] = firstAppr[j][k];
    }
  }
  cudaMemcpy(d_rs, rightSide, size * (N * N - 4 * N + 4), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fa, fa, size * (N * N - 4 * N + 4), cudaMemcpyHostToDevice);
  cudaMemcpy(d_all, all, size * (N * N), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diff, diff, size * (N * N - 4 * N + 4), cudaMemcpyHostToDevice);
  double timeChecker = dsecnd();
  int n = N - 2;
  for (int k = n; k > 0; --k) {
    for (int j = 0, i = 0; i < 2 * k - 1; i++) {
      if (i < n) {
        ++j;
      } else {
        --j;
      }
      mykernel<<<1, j>>>(d_rs, wOpt, d_fa, d_diff, N * N - 4 * N + 4, d_all, i, j, n);
    }
  }
  cudaMemcpy(fa, d_fa, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
  outVector(fa, N * N - 4 * N + 4);
  // cin>>aber;
  cudaMemcpy(diff, d_diff, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
  do {
    //   outVector(diff, N * N - 4 * N + 4);
    // diff[0] = 0;
    // cudaMemcpy(d_diff, diff, size, cudaMemcpyHostToDevice);
    cout << "The " << k << " iter" << endl;
    // copyVectors(firstAppr, changeAppr);
    // cout<<"change: "<<endl;
    // outMatr(changeAppr);
    // cout<<"fa: "<<endl;
    // outMatr(firstAppr);
    // cin>>aber;
    // for (int i = 0; i < A.size(); i++) {
    //     firstAppr[i] = firstAppr[i] + (B[i] - aMulX(A, firstAppr, i)) * wOpt / (DwL(A, i, wOpt));
    // }
    // for (int j = 1; j < N - 1; ++j) {
    //     for (int i = 1; i < N - 1; i++) {
    //               firstAppr[j][i] = (B[j][i] - (firstAppr[j][i + 1] + firstAppr[j][i - 1] +
    // firstAppr[j + 1][i] + firstAppr[j - 1][i] - 4 * firstAppr[j][i])) * wOpt / (DwL(A, i, wOpt)); + firstAppr[j][i];
    my_red_black_kernel<<<N - 2, N - 2>>>(d_rs, wOpt, d_fa, d_diff, N * N - 4 * N + 4, d_all, n, 1);
    // cudaMemcpy(fa, d_fa, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
    // cout<<"Halfstep"<<endl;
    // outVector(fa, N * N - 4 * N + 4);
    my_red_black_kernel<<<N - 2, N - 2>>>(d_rs, wOpt, d_fa, d_diff, N * N - 4 * N + 4, d_all, n, 0);
    // for (int j = 0,i = 0; i < 2 * n - 1; i++) {
    //     if (i < n) {
    //         ++j;
    //     } else {
    //         --j;
    //     }
    //     mykernel<<<1, j>>>(d_rs, wOpt, d_fa, d_diff, N * N - 4 * N + 4, d_all, i, j, n);
    //   //   cudaMemcpy(fa, d_fa, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
    //   //   cout<<"The fa is "<<endl;
    //   //   outVector(fa, N * N - 4 * N + 4);
    //   //   cin>>aber;
    // }
    // firstAppr[j][i] = (-B[j][i] + firstAppr[j + 1][i] + firstAppr[j - 1][i] + firstAppr[j][i - 1] +
    //   firstAppr[j][i + 1] - 4 * (1 - 1. / wOpt) * firstAppr[j][i]) * wOpt / 4.;
    //     }
    // }
    // for (int j = 1; j < N - 1; ++j) {
    //     for (int i = 1; i < N - 1; i++) {
    //         changeAppr[j][i] = fabs(firstAppr[j][i] - changeAppr[j][i]);
    //     }
    // }
    // cudaMemcpy(fa, d_fa, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
    cudaMemcpy(diff, d_diff, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
    // cout<<"The fa is "<<endl;
    // outVector(fa, N * N - 4 * N + 4);
    // cout<<endl;
    // outVector(diff, N * N - 4 * N + 4);
    maxDiff = findMaxInVector(diff, N * N - 4 * N + 4);
    // outVector(changeAppr);
    // cout<<findMaxInVector(changeAppr)<<endl;
    // maxDiff = findMaxInVector(changeAppr);
    // system("pause");
    ++k;
    cout << "Maxdiff is " << maxDiff << endl;
    // cin>>aber;
    if (maxDiff > 1000) {
      break;
    }
  } while (maxDiff > eps);
  timeChecker = dsecnd() - timeChecker;
  cout << "The iter is:" << endl;
  cout << k << endl;
  firstApprSet(changeAppr);
  cudaMemcpy(fa, d_fa, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
  for (int j = 1; j < N - 1; j++) {
    for (int k = 1; k < N - 1; k++) {
      firstAppr[j][k] = fa[(j - 1) * (N - 2) + (k - 1)];
    }
  }
  cudaFree(d_rs);
  cudaFree(d_fa);
  // cudaFree(d_c);
  cudaFree(d_all);
  cudaFree(d_diff);
  //   /*
  //   * outing
  //   */
  cout << "The Matr Is:" << endl;
  outMatr(A);
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
