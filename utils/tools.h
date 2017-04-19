// Copyright 2015-2017 Illia Olenchenko

#include "vector"

#include "../lib/alglib/src/ap.h"

using namespace std;
using namespace alglib;

#ifndef TOOLS_H
#define TOOLS_H

double aMulX(vector<vector<double> > A, vector<double> X, int j);
double* aMulXVector(vector<vector<double> > A, vector<double> X);
double findMaxRealArr(alglib::real_1d_array const wr);
double findMinRealArr(alglib::real_1d_array const wr);

#endif /* TOOLS_H */
