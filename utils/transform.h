// Copyright 2015-2017 Illia Olenchenko

#include "vector"

#include "../lib/alglib/src/ap.h"

using namespace std;
using namespace alglib;

#ifndef TRANSFORM_H
#define TRANSFORM_H

double* arrToRealArr(vector<vector<double> >const &A);
int realArr2dToVectorMatr(alglib::real_2d_array matrix, vector<vector<double> > &A);

#endif /* TRANSFORM_H */
