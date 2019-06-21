#define K_MAX		1000
#define THIN		1000
#define M_PI        3.141592653589793238462643383279502884
#define SEED		100

#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>
#include <boost/math/special_functions.hpp>

using namespace std;
using namespace Eigen;
using namespace boost::math;
using Eigen::MatrixXd;

double max(double* vec, int length);
int sample(double* prob, int length, mt19937& r, bool normalized=false);
double rgamma(double a, double b, mt19937& r);
