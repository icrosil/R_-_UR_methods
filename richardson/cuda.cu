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
#include <stdio.h>

using namespace std;
using namespace alglib;
using namespace alglib_impl;

/**
 * CUDA functions
 */

#ifndef N
#define N 4
#endif

__device__ int barrier = N - 2;
__device__ int blocks = N - 2;

__global__ void myshab(double *temp, int n, double *all) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = index + N + 1 + 2 * (int) (index / (N - 2));
  if (index < n) {
    temp[index] = -4 * all[lindex] + all[lindex - N] + all[lindex + N] + all[lindex - 1] + all[lindex + 1];
  }
}

// B, Shablon, Tau, firstAppr, iteration number
__global__ void mykernel(double *a, double *b, double *c, double *d, int n, int i, double *all) {
    // TODO(me) syncthreads will work, so pass needed elements, let them for and sync on every iteration
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = index + N + 1 + 2 * (int) (index / (N - 2));
    if (index < n) {
        d[index] = (-a[index] + b[index]) * c[i] + d[index];
        all[lindex] = d[index];
    }
    // if i ever need an block sync (time expensive), or create own block sync with a vector of 00,
    // fill them with 1 when block is done and go next if all 1
    /* Do whatever it is that this block does. */


  //  /* Make sure all threads in this block are actually here. */
  //  __syncthreads();
  //  /* Once we're done, decrease the value of the barrier. */
  // if ( threadIdx.x == 0 )
  //     atomicSub( &barrier , 1 );
  //
  // /* Now wait for the barrier to be zero. */
  // if ( threadIdx.x == 0 )
  //     while ( atomicCAS( &barrier , 0 , 0 ) != 0 );
  //
  // /* Make sure everybody has waited for the barrier. */
  // __syncthreads();
  //
  // /* Carry on with whatever else you wanted to do. */
  // barrier = N - 2;
}



int main() {
  /**
   * t0 is for documenting whole processing time
   * @type double
   */
  double t0 = dsecnd();
  vector<int> cudas(2);
  cudas[0] = 0;
  cudas[1] = 1;
  /**
   * N is for number of points of SLAU
   * @type int
   */

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
  double *temp = new double[N * N - 4 * N + 4];
  double *all = new double[N * N];
  double *b = new double[N * N - 4 * N + 4];
  double *fa = new double[N * N - 4 * N + 4];
  double *taum = new double[maxIter + 1];
  double *d_a, *d_b, *d_c, *d_d, *d_g;
  int size = sizeof(double);

  cudaMalloc((void **)&d_a, size * (N * N - 4 * N + 4));
  cudaMalloc((void **)&d_b, size * (N * N - 4 * N + 4));
  cudaMalloc((void **)&d_g, size * (N * N));
  cudaMalloc((void **)&d_c, size * (maxIter + 1));
  cudaMalloc((void **)&d_d, size * (N * N - 4 * N + 4));

  for (int j = 1; j < N - 1; j++) {
    for (int k = 1; k < N - 1; k++) {
      temp[(j - 1) * (N - 2) + (k - 1)] = 0;
      b[(j - 1) * (N - 2) + (k - 1)] = B[j][k];
      fa[(j - 1) * (N - 2) + (k - 1)] = firstAppr[j][k];
    }
  }
  for (int j = 0; j < N ; j++) {
    for (int k = 0; k < N; k++) {
      all[j * N + k] = firstAppr[j][k];
    }
  }
  for (int i = 0; i < maxIter + 1; i++) {
    taum[i] = Tau[i];
  }

  outVector(temp, N * N - 4 * N + 4);
  outVector(b, N * N - 4 * N + 4);
  outVector(fa, N * N - 4 * N + 4);

  cudaMemcpy(d_a, b, size * (N * N - 4 * N + 4), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, fa, size * (N * N - 4 * N + 4), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, taum, size * (maxIter + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g, all, size * (N * N), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, temp, size * (N * N - 4 * N + 4), cudaMemcpyHostToDevice);
  double timeChecker = dsecnd();
  char aster;
  for (int i = 1; i < maxIter + 1; ++i) {
    // for (size_t c = 0; c < cudas.size(); c++) {
      // cudaSetDevice(cudas[c]);
    myshab <<<N - 2, N - 2>>>(d_b, N * N - 4 * N + 4, d_g);
    mykernel <<<N - 2, N - 2>>>(d_a, d_b, d_c, d_d, N * N - 4 * N + 4, i, d_g);
    // }
    // sync
    // for (size_t c = 0; c < cudas.size(); c++) {
    //   cudaSetDevice(cudas[c]);
    //   cudaDeviceSynchronize();
    // }
    cout <<"The " <<i <<" iter" <<endl;
    cudaMemcpy(temp, d_b, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
    cout <<endl <<"The temp from GPU is" <<endl;
    outVector(temp, N * N - 4 * N + 4);
    Shablon(firstAppr, temp);
    cout <<"The temp is" <<endl;
    outVector(temp, N * N - 4 * N + 4);
    cin>>aster;
    cudaMemcpy(d_b, temp, size * (N * N - 4 * N + 4), cudaMemcpyHostToDevice);
    cudaMemcpy(fa, d_d, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
    cudaMemcpy(all, d_g, size * (N * N), cudaMemcpyDeviceToHost);
    for (int j = 1; j < N - 1; j++) {
        for (int k = 1; k < N - 1; k++) {
            firstAppr[j][k] = fa[(j - 1) * (N - 2) + (k - 1)];
        }
    }
    cout <<endl <<"fa" <<endl;
    outMatr(firstAppr);
    cout <<"ALLL" <<endl;
    outVector(all, N * N);
    cout <<endl;
  }
  double tMain = dsecnd() - timeChecker;
  cudaMemcpy(fa, d_d, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
  for (int j = 1; j < N - 1; j++) {
    for (int k = 1; k < N - 1; k++) {
      firstAppr[j][k] = fa[(j - 1) * (N - 2) + (k - 1)];
    }
  }

  // for (int i = 0; i < firstAppr.size(); i++) {
  //     firstAppr[i] /= ((firstAppr.size() - 1) * (firstAppr.size() - 1));
  // }
  cudaFree(d_a);
  cudaFree(d_b);
  // cudaFree(d_c);
  cudaFree(d_d);
  /*
  * outing
  */
  firstApprSet(tempAppr);
  cout << "The N is : " << N << endl;
  // cout <<"The A(shorted) Is:" <<endl;
  // outMatr(A);
  // cout <<"The B(shorted) Is:" <<endl;
  // outMatr(B);
  // cout <<"The duo(shorted) Is:" <<endl;
  // outVector(duo);
  // cout <<"The opt(shorted) Is:" <<endl;
  // outVector(optTau);
  // cout <<"The first appr Is:" <<endl;
  // outMatr(tempAppr);
  // cout <<"The last approximation Is:" <<endl;
  // outMatr(firstAppr);
  // cout <<"The Max alpha Is:" <<endl;
  // cout <<AlphaMax <<endl;
  // cout <<"The Min alpha Is:" <<endl;
  // cout <<AlphaMin <<endl;
  // cout <<"The Tau is:" <<endl;
  // outVector(Tau);
  // cout <<"The ksi is:" <<endl;
  // cout <<ksi <<endl;
  // cout <<"The ro0 is:" <<endl;
  // cout <<ro0 <<endl;
  // cout <<"The ro1 is:" <<endl;
  // cout <<ro1 <<endl;
  cout <<"The maxIter is:" <<endl;
  cout <<maxIter <<endl;
  cout <<"The time is:" <<endl;
  cout << dsecnd() - t0 <<" s" <<endl;
  cout <<"The time of main is:" <<endl;
  cout << tMain <<" s" <<endl;
  cout <<"The 1 1 is:" <<endl;
  cout << firstAppr[1][1] <<endl;
  cout <<"The 2 2 is:" <<endl;
  cout << firstAppr[2][2] <<endl;
  cout <<"The N - 2 N - 2 is:" <<endl;
  cout << firstAppr[firstAppr.size() - 2][firstAppr.size() - 2] <<endl;
  cout <<"The N - 3 N - 3 is:" <<endl;
  cout << firstAppr[firstAppr.size() - 3][firstAppr.size() - 3] <<endl;
  return 0;
}
