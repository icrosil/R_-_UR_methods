// Copyright 2015-2017 Illia Olenchenko

#include <math.h>
#include <iostream>
#include "vector"
#include <string>

using namespace std;

// transformer of matr to flat array
double* arrToRealArr(vector<vector<double> >const &A) {
  double * local;
  local = new double[A.size() * A.size()];
  for (int i = 0; i < A.size(); ++i) {
    for (int j = 0; j < A[i].size(); ++j) {
      local[i * A.size() + j] = A[i][j];
    }
  }
  return local;
}
