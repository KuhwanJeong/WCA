#define K_MAX		500
#define M_PI        3.141592653589793238462643383279502884
#define ITER1		50
#define ITER2		50
#define ITER3		10
#define SEED		1234
#define EPS			1e-5

#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>
#include <boost/math/special_functions.hpp>
#include <time.h>
#include <string>

using namespace std;
using namespace Eigen;
using namespace boost::math;
using Eigen::MatrixXd;

double max(double* vec, int length);
int sample(double* prob, int length, mt19937& r, bool normalized=false);
double rgamma(double a, double b, mt19937& r);
double rbeta(double a, double b, mt19937& r);
void rdirichlet(double* p, double* a, int K, mt19937& r);
