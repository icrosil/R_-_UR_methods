#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void mykernel (void) {}

int main(void) {
  mykernel<<<1, 1>>>();
  cout << "HELLO CUDA!/n";
  return 0;
}
