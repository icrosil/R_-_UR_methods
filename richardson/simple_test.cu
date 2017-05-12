// Copyright 2015-2017 Illia Olenchenko

#include <stdio.h>
#include <iostream>
#include <mkl.h>

using namespace std;

__global__ void mykernel() {}

int main(void) {
  mykernel<<<1, 1>>>();
  double t0 = dsecnd();
  mykernel<<<1, 1>>>();
  cout << dsecnd() - t0 <<" s" <<endl;
  // what time of call cuda function after connection to gpu ~ 1.6e-05
  cout << "HELLO CUDA!/n";
  return 0;
}
