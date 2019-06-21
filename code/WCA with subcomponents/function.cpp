#include "function.h"

double max(double* vec, int length){
	double M=vec[0];
	for(int n=1; n<length; n++)
		if(M < vec[n])	M = vec[n];
	return M;
}

int sample(double* prob, int length, mt19937& r, bool normalized){
	uniform_real_distribution<double> runif;
	double p=runif(r), cum=0.0, p_sum=0.0;

	if(normalized==false){
		for(int k=0; k<length; k++)
			p_sum += prob[k];
		for(int k=0; k<length; k++)
			prob[k] = prob[k]/p_sum;
	}
	
	for(int k=0; k<length-1; k++){
		cum += prob[k];
		if(p < cum)
			return k;
	}
	return length-1;
}

double rgamma(double a, double b, mt19937& r){
	gamma_distribution<double> rgamma(a,1.0/b);
	return rgamma(r);
}

double rbeta(double a, double b, mt19937& r){
	gamma_distribution<double> rgamma(a,1.0);
	double x = rgamma(r);
	rgamma=gamma_distribution<double>(b,1.0);
	return x/(x+rgamma(r));
}

void rdirichlet(double* p, double* a, int K, mt19937& r){
	gamma_distribution<double> rgamma;
	double sum = 0.0;
	int k;
	for(k=0; k<K; k++){
		rgamma=gamma_distribution<double>(a[k],1.0);
		p[k] = rgamma(r);
		sum += p[k];
	}
	for(k=0; k<K; k++)  p[k] /= sum;

	return;
}