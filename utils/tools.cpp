// Copyright 2015-2017 Illia Olenchenko

#include <math.h>
#include <iostream>
#include "vector"
#include <string>

#include "../lib/alglib/src/ap.h"

using namespace std;
using namespace alglib;

// multiplication of A[j]X
double aMulX(vector<vector<double> > A, vector<double> X, int j) {
  double res = 0;
  for (int i = 0; i < A.size(); ++i) {
    res += A[j][i] * X[i];
  }
  return res;
}

// util A mult X Vector
double* aMulXVector(vector<vector<double> > A, vector<double> X) {
  double *res = new double[X.size()];
  for (int j = 0; j < X.size(); j++) {
    res[j] = 0;
  }
  for (int j = 0; j < A.size(); ++j) {
    for (int i = 0; i < A.size(); ++i) {
      res[j] += A[j][i] * X[i];
    }
  }
  return res;
}

/*
 * works only for good square matricies
 */
int mulMatricies(vector<vector<double> > A, vector<vector<double> > B, vector<vector<double> > &temp) {
  for (int i = 0; i < A.size(); ++i) {
    for (int j = 0; j < A.size(); ++j) {
      for (int k = 0; k < A.size(); ++k) {
        temp[i][j] += A[k][j] * B[i][k];
      }
    }
  }
  return 0;
}

// simple max finder
double findMaxRealArr(alglib::real_1d_array const wr) {
  double max = fabs(wr[0]);
  for (int i = 1; i < wr.length(); ++i) {
    if (fabs(wr[i]) > max) max = fabs(wr[i]);
  }
  return max;
}

// simple min finder
double findMinRealArr(alglib::real_1d_array const wr) {
  double min = fabs(wr[0]);
  for (int i = 1; i < wr.length(); ++i) {
    if (fabs(wr[i]) < min) min = fabs(wr[i]);
  }
  return min;
}

double findMaxInVector(vector<vector<double> > a) {
  double max = a[1][1];
  for (int i = 2; i < a.size() - 1; i++) {
    for (int j = 2; j < a.size() - 1; j++) {
      if (a[i][j] > max) max = a[i][j];
    }
  }
  return max;
}

int copyVectors(vector<vector<double> > in, vector<vector<double> > &out) {
  for (int i = 0; i < in.size(); i++) {
    for (int j = 0; j < in[i].size(); j++) {
      out[i][j] = in[i][j];
    }
  }
  return 0;
}
