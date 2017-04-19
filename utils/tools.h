// Copyright 2015-2017 Illia Olenchenko

#include "vector"

#include "../lib/alglib/src/ap.h"

using namespace std;
using namespace alglib;

#ifndef TOOLS_H
#define TOOLS_H

double aMulX(vector<vector<double> > A, vector<double> X, int j);
double* aMulXVector(vector<vector<double> > A, vector<double> X);
int mulMatricies(vector<vector<double> > A, vector<vector<double> > B, vector<vector<double> > &temp);
double findMaxRealArr(alglib::real_1d_array const wr);
double findMinRealArr(alglib::real_1d_array const wr);
double findMaxInVector(vector<vector<double> > a);
double findMaxInVector(double *a, int size);
int copyVectors(vector<vector<double> > in, vector<vector<double> > &out);


#endif /* TOOLS_H */
