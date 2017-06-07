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

/**
 * N is for number of points of SLAU
 * should be N % 2 == 0 for correct split
 * @type int
 */
#ifndef N
#define N 10
#endif
#ifndef GPU
#define GPU 2
#endif

__device__ int barrier = N - 2;
__device__ int blocks = N - 2;

__global__ void myshab(double *temp, int n_row, int n_col, int plus, double *all) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int index = row * n_col + col;
  if (index >= n_row * n_col) return;
  int lindex = index + N + 1 + 2 * (int) (index / (N - 2)) + (N * plus);
  // #if __CUDA_ARCH__ >= 200
  //   printf("%d index \n", index);
  //   printf("%d lindex \n", lindex);
  // #endif
  temp[index] = -4 * all[lindex] + all[lindex - N] + all[lindex + N] + all[lindex - 1] + all[lindex + 1];
}

// B, Shablon, Tau, firstAppr, iteration number
__global__ void mykernel(double *a, double *b, double *c, double *d, int n_row, int n_col, int plus, int i,
  double *all
) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int index = row * n_col + col;
    if (index >= n_row * n_col) return;
    int lindex = index + N + 1 + 2 * (int) (index / (N - 2)) + (N * plus);
    d[index] = (-a[index] + b[index]) * c[i] + d[index];
    all[lindex] = d[index];
}


int main() {
  /**
   * t0 is for documenting whole processing time
   * @type double
   */
  double t0 = dsecnd();
  // multi GPUs
  vector<int> cudas(GPU);
  for (size_t i = 0; i < GPU; i++) {
    cudas[i] = i;
  }
  // inner Ns
  int n_inner = (N - 2) * (N - 2);
  int n_splitted_inner = n_inner / GPU;

  /*
  * Getting inputs A and B
  */
  vector<vector<double> > A(n_inner, vector<double>(n_inner, 0));
  readMatr(A);
  vector<vector<double> > B(N, vector<double>(N, 0));
  vector<double> Tau(1, 0);
  vector<vector<double> > firstAppr(N, vector<double>(N, 0));
  vector<vector<double> > tempAppr(N, vector<double>(N, 0));
  firstApprSet(firstAppr);
  readVector(B);
  alglib::real_2d_array matrix;
  matrix.setcontent(n_inner, n_inner, arrToRealArr(A));
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
  alglib::smatrixevd(matrix, N - 2, 0, true, wr, vl);
  double AlphaMax = findMaxRealArr(wr);
  double AlphaMin = findMinRealArr(wr);
  Tau[0] = 2. / (AlphaMax + AlphaMin);
  double ksi = AlphaMin / AlphaMax;
  std::cout << ksi << "ksi" << '\n';  // is it important to calculate n*n alphas?
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
  double **temp = new double*[(int)GPU];
  double *all = new double[N * N];
  double **b = new double*[(int)GPU];
  double **fa = new double*[(int)GPU];
  double *taum = new double[maxIter + 1];
  for (size_t i = 0; i < GPU; i++) {
    temp[i] = new double[(int)n_splitted_inner];
    b[i] = new double[(int)n_splitted_inner];
    fa[i] = new double[(int)n_splitted_inner];
  }

  double *d_a[GPU],
    *d_b[GPU],
    *d_c[GPU],
    *d_d[GPU],
    *d_g[GPU];
  int size = sizeof(double);

  for (size_t i = 0; i < GPU; i++) {
    cudaSetDevice(cudas[i]);
    cudaDeviceEnablePeerAccess(cudas[i], 0);
  }

  for (size_t i = 0; i < GPU; i++) {
    cudaSetDevice(cudas[i]);
    cudaMalloc((void **)&d_a[i], size * (n_splitted_inner));
    cudaMalloc((void **)&d_b[i], size * (n_splitted_inner));
    cudaMalloc((void **)&d_d[i], size * (n_splitted_inner));
    cudaMalloc((void **)&d_g[i], size * (N * N));
    cudaMalloc((void **)&d_c[i], size * (maxIter + 1));
  }

  for (size_t i = 0; i < GPU; i++) {
    int plus = i * ((int)(N / 2) - 1);
    for (int j = 1; j < (int)(N / 2); j++) {
      for (int k = 1; k < N - 1; k++) {
        temp[i][(j - 1) * (int)((N - 2) / 2) + (k - 1)] = 0;
        b[i][(j - 1) * (int)((N - 2) / 2) + (k - 1)] = B[j + plus][k];
        fa[i][(j - 1) * (int)((N - 2) / 2) + (k - 1)] = firstAppr[j + plus][k];
      }
    }
  }
  for (int j = 0; j < N ; j++) {
    for (int k = 0; k < N; k++) {
      all[j * N + k] = firstAppr[j][k];
    }
  }
  // outVector(all, N * N);
  for (int i = 0; i < maxIter + 1; i++) {
    taum[i] = Tau[i];
  }
  for (int i = 0; i < GPU; i++) {
    cudaSetDevice(cudas[i]);
    cudaMemcpy(d_a[i], b[i], size * (n_splitted_inner), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d[i], fa[i], size * (n_splitted_inner), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c[i], taum, size * (maxIter + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g[i], all, size * (N * N), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b[i], temp[i], size * (n_splitted_inner), cudaMemcpyHostToDevice);
  }
  double timeChecker = dsecnd();
  // char aster;
  dim3 threadsPerBlock(16, 16);
  int n_row = (int)((N - 2) / 2);
  int n_col = (N - 2);
  dim3 numBlocks(max(n_row / threadsPerBlock.x, 1), max(n_col / threadsPerBlock.y, 1));
  for (int j = 1; j < maxIter + 1; ++j) {
    for (size_t i = 0; i < GPU; i++) {
      cudaSetDevice(cudas[i]);
      int plus = i * ((int)(N / 2) - 1);
      myshab <<<numBlocks, threadsPerBlock>>>(d_b[i], n_row, n_col, plus, d_g[i]);
      mykernel <<<numBlocks, threadsPerBlock>>>(d_a[i], d_b[i], d_c[i], d_d[i], n_row, n_col, plus, j, d_g[i]);

      // cout <<"The " <<j <<" iter" <<endl;
      // cudaMemcpy(temp[i], d_b[i], size * (n_splitted_inner), cudaMemcpyDeviceToHost);
      // cout <<endl <<"The temp from GPU is" <<endl;
      // outVector(temp[i], n_splitted_inner);
      // Shablon(firstAppr, temp);
      // cout <<"The temp is" <<endl;
      // outVector(temp, N * N - 4 * N + 4);
      // cin >> aster;
      // cudaMemcpy(d_b, temp, size * (N * N - 4 * N + 4), cudaMemcpyHostToDevice);
      // cudaMemcpy(fa, d_d, size * (N * N - 4 * N + 4), cudaMemcpyDeviceToHost);
      // cudaMemcpy(all, d_g, size * (N * N), cudaMemcpyDeviceToHost);
      // for (int j = 1; j < N - 1; j++) {
      //   for (int k = 1; k < N - 1; k++) {
      //     firstAppr[j][k] = fa[(j - 1) * (N - 2) + (k - 1)];
      //   }
      // }
      // cout <<endl <<"fa" <<endl;
      // outMatr(firstAppr);
      // cout <<"ALLL" <<endl;
      // outVector(all, N * N);
      // cout <<endl;
    }
    for (size_t i = 0; i < GPU; i++) {
      int index = N * ((int)(N / 2) + (i == 0 ? -1 : 0));
      // copy all to other.
      cudaMemcpy(
        &d_g[i == 0 ? 1 : 0][index],
        &d_g[i][index],
        size * N,
        cudaMemcpyDefault);
    }
  }
  double tMain = dsecnd() - timeChecker;
  for (size_t i = 0; i < GPU; i++) {
    cudaMemcpy(fa[i], d_d[i], size * (n_splitted_inner), cudaMemcpyDeviceToHost);
  }
  for (size_t i = 0; i < GPU; i++) {
    int plus = i * ((int)(N / 2) - 1);
    for (int j = 1; j < (int)(N / 2); j++) {
      for (int k = 1; k < N - 1; k++) {
        firstAppr[j + plus][k] = fa[i][(j - 1) * (int)((N - 2) / 2) + (k - 1)];
      }
    }
  }
  for (size_t i = 0; i < GPU; i++) {
    cudaFree(d_a[i]);
    cudaFree(d_b[i]);
    cudaFree(d_c[i]);
    cudaFree(d_d[i]);
    cudaFree(d_g[i]);
  }
  /*
  * outing
  */
  // firstApprSet(tempAppr);
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
  // cout <<"The maxIter is:" <<endl;
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
