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
