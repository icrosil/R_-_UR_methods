// Copyright 2015-2017 Illia Olenchenko

#include <iostream>
#include "vector"

#include "../lib/alglib/src/ap.h"

using namespace std;
using namespace alglib;

// simple out vector to console (you can increase additions)
void outVector(vector<double> B) {
  int additions = 1;
  cout << B[0] << " ";
  for (int i = additions; i < B.size() - 1; i += additions) {
    cout << B[i] << " ";
  }
  cout << B[B.size() - 1] << " ";
  cout << endl;
}

// out of Vector with margin of N
void outVector(double* B, int N) {
  int additions = 1;
  cout << B[0] << " ";
  for (int i = additions; i < N - 1; i += additions) {
    cout << B[i] << " ";
  }
  cout << B[N - 1] << " ";
  cout << endl;
  cout << "out outVector" << endl;
}

// simple out matr to console (you can increase additions)
void outMatr(vector<vector<double> > A) {
  int additions = 1;
  outVector(A[0]);
  for (int i = additions; i < A.size() - 1; i += additions) {
    outVector(A[i]);
  }
  outVector(A[A.size() - 1]);
}

// simple out of special vector
void outReal1Array(alglib::real_1d_array wr) {
  for (int i = 0; i < wr.length(); ++i) {
    cout << wr[i] << " ";
  }
  cout << endl;
}
