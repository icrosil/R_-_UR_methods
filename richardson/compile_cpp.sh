g++ main.cpp -I /opt/intel/composer_xe_2013/mkl/include/ -L /opt/intel/composer_xe_2013/mkl/lib/intel64/ -L /opt/intel/lib/intel64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -lmkl_intel_lp64 -I /apps/Installed/nvidia/cuda5.0/include -L /apps/Installed/nvidia/cuda5.0/lib64 -lcusparse -lcudart -lcuda -lcublas -fopenmp -lgomp ../utils/functions.cpp ../utils/init.cpp ../utils/out.cpp ../utils/richardson.cpp ../utils/tools.cpp ../utils/transform.cpp ../lib/alglib/src/ap.cpp ../lib/alglib/src/alglibmisc.cpp ../lib/alglib/src/alglibinternal.cpp ../lib/alglib/src/linalg.cpp ../lib/alglib/src/statistics.cpp ../lib/alglib/src/dataanalysis.cpp ../lib/alglib/src/specialfunctions.cpp ../lib/alglib/src/solvers.cpp ../lib/alglib/src/optimization.cpp ../lib/alglib/src/diffequations.cpp ../lib/alglib/src/fasttransforms.cpp ../lib/alglib/src/integration.cpp ../lib/alglib/src/interpolation.cpp
