// Copyright 2015-2017 Illia Olenchenko

#include <math.h>
#include "vector"

using namespace std;

// result of F function
double F(double x, double y, double N) {
  return (2 * sin(y) - x * x * sin(y)) / N;
}

// result of U function
double U(double x, double y) {
  return x * x * sin(y) + 1;
}

// Differentiate net
int Shablon(vector<vector<double> > X, double * &res) {
  for (int j = 1; j < X.size() - 1; ++j) {
    for (int i = 1; i < X.size() - 1; ++i) {
      res[(i - 1) * (X.size() - 2) + (j - 1) ] = X[i + 1][j] + X[i - 1][j] + X[i][j + 1] + X[i][j - 1] - 4. * X[i][j];
    }
  }
  return 0;
}
