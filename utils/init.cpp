// Copyright 2015-2017 Illia Olenchenko

#include <math.h>
#include "vector"

#include "./functions.h"

using namespace std;

// initializing of A matr
void readMatr(vector<vector<double> > &A) {
  int sizer = (int)sqrt(A.size());
  A[0][0] = -4;
  A[A.size() - 1][A.size() - 1] = -4;
  A[0][1] = 1;
  A[A.size() - 1][A.size() - 2] = 1;
  for (int i = 1; i < A.size() - 1; i++) {
    A[i][i] = -4;
    if (!((i % sizer) == 0)) {
      A[i][i - 1] = 1;
    }
    if (!(((i - 1) % sizer) == 0)) {
      A[i][i + 1] = 1;
    }
  }
  for (int i = 0; i < A.size() - sizer; i++) {
    A[i][i + sizer] = 1;
    A[i + sizer][i] = 1;
  }
}

// initializing of B vector
void readVector(vector<vector<double> >& B) {
  for (int i = 0; i < B.size(); i++) {
    B[i][0] = U(i / (double) (B.size() - 1), 0);
    B[0][i] = U(0, i / (double) (B.size() - 1));
    B[B.size() - 1][i] = U(1, i / (double) (B.size() - 1));
    B[i][B.size() - 1] = U(i / (double) (B.size() - 1), 1);
  }
  for (int i = 1; i < B.size() - 1; ++i) {
    for (int j = 1; j < B.size() - 1; ++j) {
      B[i][j] = F(i / (double) (B.size() - 1), j / (double) (B.size() - 1), (B.size() - 1) * (B.size() - 1));
    }
  }
}

// setter of first approximation
void firstApprSet(vector<vector<double> >& B) {
  for (int i = 0; i < B.size(); i++) {
    B[i][0] = U(i / (double) (B.size() - 1), 0);
    B[0][i] = U(0, i / (double) (B.size() - 1));
    B[B.size() - 1][i] = U(1, i / (double) (B.size() - 1));
    B[i][B.size() - 1] = U(i / (double) (B.size() - 1), 1);
  }
  for (int i = 1; i < B.size() - 1; ++i) {
    for (int j = 1; j < B.size() - 1; ++j) {
      B[i][j] = F(i / (double) (B.size() - 1), j / (double) (B.size() - 1), (B.size() - 1) * (B.size() - 1)) / 2.;
    }
  }
}
