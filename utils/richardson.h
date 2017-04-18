// Copyright 2015-2017 Illia Olenchenko

#ifndef RICHARDSON_H
#define RICHARDSON_H

double nextTau(vector<double> Tau, double ro0, int n, vector<double> optTau);
int findMaxIter(double eps, double ksi, int N);
void decToDuo(vector<double> &duo, int maxIter);
void calculateOptTau(vector<double> &optTau, vector<double> duo);

#endif /* RICHARDSON_H */
