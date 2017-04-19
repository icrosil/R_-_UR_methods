// Copyright 2015-2017 Illia Olenchenko

#include <math.h>
#include <iostream>
#include "vector"
#include <string>

#include "../lib/alglib/src/linalg.h"
#include "./tools.h"
#include "./transform.h"

/*
 * можно найти оптимальный w как 2 - O(h) но это же слишком просто, по этому пишем эту фиготень
 * совсем не факт что это штука с детерминантом оптимальная но другого у меня нет.
 */
double wOptSet(vector<vector<double> > A, double spectr, double oh) {
  vector<vector<double> > D(A.size(), vector<double>(A.size(), 0));
  vector<vector<double> > LR(A.size(), vector<double>(A.size(), 0));
  vector<vector<double> > det(A.size(), vector<double>(A.size(), 0));
  double rDet;
  double wOpt;
  alglib::ae_int_t info;
  alglib::matinvreport rep;
  alglib::real_2d_array matrixD;
  alglib::real_2d_array matrixLR;
  alglib::real_2d_array matrixDet;
  for (int i = 0; i < A.size(); ++i) {
    D[i][i] = A[i][i];
  }
  for (int i = 0; i < A.size(); ++i) {
    for (int j = 0; j < A.size(); ++j) {
      if (i != j) LR[i][j] = A[i][j];
    }
  }
  matrixD.setcontent(A.size(), A.size(), arrToRealArr(D));
  matrixLR.setcontent(A.size(), A.size(), arrToRealArr(LR));
  alglib::rmatrixinverse(matrixD, A.size(), info, rep);
  realArr2dToVectorMatr(matrixD, D);
  mulMatricies(D, LR, det);
  matrixDet.setcontent(A.size(), A.size(), arrToRealArr(det));
  // outMatr(det);
  // outReal2Array(matrixDet, det.size());
  rDet = alglib::rmatrixdet(matrixDet);
  wOpt = 2 / (1 + sqrt(1 - spectr * spectr));  //* rDet));
  if (wOpt < 1 && wOpt > 0) {
    wOpt +=1;
  } else if (wOpt < 2 && wOpt > 1) {
    wOpt = wOpt;
  } else {
    wOpt = 2 - oh;
  }
  // cout<<"The determ is "<<rDet<<endl;
  // cout<<"The wopt is "<<wOpt<<endl;
  // checking inversed
  // cout<<"The D inverse Is:"<<endl;
  // to find determenant use RMatrixDet

  return wOpt;
}

double DwL(vector<vector<double> > A, int k, double w) {
  double sum = 0;
  for (int i = 0; i < k; ++i) {
    sum += A[k][i] * w;
  }
  sum += A[k][k];
  return sum;
}
