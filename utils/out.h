// Copyright 2015-2017 Illia Olenchenko

#include "vector"

#include "../lib/alglib/src/ap.h"

using namespace std;
using namespace alglib;

#ifndef OUT_H
#define OUT_H

void outVector(vector<double> B);
void outVector(double* B, int N);
void outMatr(vector<vector<double> > A);
void outReal1Array(alglib::real_1d_array wr);
int outReal2Array(alglib::real_2d_array wr, int size);

#endif /* OUT_H */
