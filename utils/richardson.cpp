// Copyright 2015-2017 Illia Olenchenko

#include <math.h>
#include <iostream>
#include "vector"
#include <string>

#include "./out.h"

using namespace std;

// next tau calculation
double nextTau(vector<double> Tau, double ro0, int n, vector<double> optTau) {
  double tk = cos((optTau[Tau.size() - 1]) * M_PI / (2 * n));
  return Tau[0] / (1 + ro0 * tk);
}

// maxIter claculation
int findMaxIter(double eps, double ksi) {
  return (int)ceil(log(2. / eps) / (2. * sqrt(ksi)));
}

// ?
void decToDuo(vector<double> &duo, int maxIter) {
  vector<double> temp(0);
  while (maxIter > 1) {
    if (maxIter % 2 != 0) {
      temp.push_back(1);
      maxIter--;
    } else {
      temp.push_back(0);
      maxIter /= 2;
    }
  }
  for (int j = 0; j < temp.size(); j++) {
    /* code */
    duo.push_back(temp[temp.size() - j - 1]);
  }
}

// opt tau calculation
void calculateOptTau(vector<double> &optTau, vector<double> duo) {
  vector<double> temp(0);
  double coord = 0;
  double number = 1;
  for (int i = 0; i < duo.size(); i++) {
    temp.clear();
    coord = 0;
    if (duo[i] == 0) {
      number *= 2;
      if ((i + 1 == duo.size()) || (duo[i + 1] == 0)) {
        /*normal*/
        coord = 0;
      } else {
        /*4m+2*/
        coord = 2;
      }
      for (int j = 0; j < optTau.size(); j++) {
        temp.push_back(optTau[j]);
        temp.push_back(4 * optTau.size() + coord - optTau[j]);
      }
    } else {
      number += 1;
      /*+1*/
      for (int j = 0; j < optTau.size(); j++) {
          temp.push_back(optTau[j]);
      }
      temp.push_back(number);
    }
    optTau.clear();
    for (int j = 0; j < temp.size(); j++) {
      optTau.push_back(temp[j]);
    }
    outVector(optTau);
  }
}
