// Copyright 2015-2017 Illia Olenchenko

#include <math.h>

using namespace std;

// result of F function
double F(double x, double y, double N) {
  return (2 * sin(y) - x * x * sin(y)) / N;
}

// result of U function
double U(double x, double y) {
  return x * x * sin(y) + 1;
}
