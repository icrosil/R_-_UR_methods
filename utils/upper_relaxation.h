// Copyright 2015-2017 Illia Olenchenko

#include "vector"

using namespace std;

#ifndef UPPER_RELAXATION_H
#define UPPER_RELAXATION_H

double wOptSet(vector<vector<double> > A, double spectr, double oh);
double DwL(vector<vector<double> > A, int k, double w);

#endif /* UPPER_RELAXATION_H */
