#include "function.h"

int main(int argc, char* argv[])
{
	// AA : alpha (the precision parameter of DP)
	// BB0 : b0 (the rate parameter for the prior distribution of lambda)
	string dir = string(argv[1]);
	dir = "/data/home/kuhemian/DPM/" + dir + "/"; 
	double AA;
	AA = atoi(argv[2])/10.0;
	double BB0;
	BB0 = atoi(argv[3])/10.0;
	double EPS;
	EPS = pow(0.1,atoi(argv[4]));
	int PP;
	PP = atoi(argv[5]);

	// n : total # of observations in the training set
	// n_te : total # of observations in the test set
	// D : dimension
	// K : # of occupied components
	int n, n_te, D, K=1, K0;
	double kappa0=0.1, kappai, alpha0=1.0, alphai, beta0=BB0, betai, ll, post, sum, p_max, n_sum = 1.0, n_del, El, Vm, Vl;
	int i, j, d, k;

	string loc;
	loc = dir + "info.txt";
	ifstream in1(loc);
	in1 >> n >> n_te >> D;
	in1.close();
	
	// x_te : test observations; x_post : observations sampled from the posterior predictive distribution
	MatrixXd x_te(n_te,D), mu(K_MAX,D), mu2(K_MAX,D), x_post(2000,D);
	VectorXd x(D), mui(D), kappa(K_MAX), alpha(K_MAX), beta(K_MAX), n_k(K_MAX), q(K_MAX);

	loc = dir + "train" + to_string((long long)PP) + ".txt";
	ifstream in2(loc);

	loc = dir + "test.txt";
	ifstream in3(loc);
	for(i=0; i<n_te; i++)
		for(d=0; d<D; d++)
			in3 >> x_te(i,d);
	in3.close();

	loc = dir + "post" + to_string((long long)(AA*10.0)) + ".txt";
	ifstream in4(loc);
	for(i=0; i<2000; i++){
		for(d=0; d<D; d++)
			in4 >> x_post(i,d);
	}
	in4.close();

	for(d=0; d<D; d++)	in2 >> x(d);
	n_k(0) = 1.0;
	kappa(0) = kappa0 + 1.0;
	mu.row(0) = x/kappa(0);
	alpha(0) = alpha0 + D * 0.5;
	beta(0) = beta0 + x.dot(x) * 0.5 * kappa0 / kappa(0);

	double *p =  (double*)malloc((K_MAX+1)*sizeof(double));

	dir = dir + "result/WCA0/";
	loc = dir + "K" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out1(loc);
	loc = dir + "ll" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out2(loc);
	loc = dir + "post" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out3(loc);

	mt19937 r((unsigned int)SEED);
	normal_distribution<double> rnorm(0,1);

	for(i=1; i<n; i++){
		for(d=0; d<D; d++)	in2 >> x(d);

		// Calculate Eq (8)
		p_max = -DBL_MAX;
		for(k=0; k<K; k++){
			kappai = kappa(k) + 1.0;
			mui = (kappa(k)*mu.row(k).transpose()+x)/kappai;
			alphai = alpha(k) + D * 0.5;
			betai = beta(k) + (mu.row(k).transpose()-x).dot(mu.row(k)-x.transpose()) * 0.5 * kappa(k) / kappai;

			p[k] = log(n_k(k)) + lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
					- lgamma(alpha(k)) + alpha(k) * log(beta(k)) + 0.5 * D * log(kappa(k));
			
			if(p_max < p[k])	p_max = p[k];
		}
		kappai = kappa0 + 1.0;
		mui = x/kappai;
		alphai = alpha0 + D * 0.5;
		betai = beta0 + x.dot(x) * 0.5 * kappa0 / kappai;
		p[K] = log(AA) + lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
				- lgamma(alpha0) + alpha0 * log(beta0) + 0.5 * D * log(kappa0);
		
		if(p_max < p[K])	p_max = p[K];
		K++;
		
		sum = 0.0;
		for(k=0; k<K; k++){
			p[k] = exp(p[k] - p_max);
			sum += p[k];
		}
		
		n_k(K-1) = 0.0;
		beta(K-1) = beta0;
		alpha(K-1) = alpha0;
		kappa(K-1) = kappa0;
		mu.row(K-1).setZero();

		for(k=0; k<K; k++){
			p[k] /= sum;
			n_k(k) += p[k]; // Update of gamma_k^(i) in page 19
			
			kappai = kappa(k) + 1.0;
			mui = (kappa(k)*mu.row(k).transpose()+x)/kappai;
			alphai = alpha(k) + D * 0.5;
			betai = beta(k) + (mu.row(k).transpose()-x).dot(mu.row(k)-x.transpose()) * 0.5 * kappa(k) / kappai;

			El = p[k] * alphai / betai + (1-p[k]) * alpha(k) / beta(k);
			Vl = p[k] * alphai / (betai*betai) + (1-p[k]) * alpha(k) / (beta(k)*beta(k)) 
				+ p[k] * (1-p[k]) * (alphai/betai - alpha(k)/beta(k)) * (alphai/betai - alpha(k)/beta(k));
			Vm = p[k] * betai / (kappai * (alphai-1)) + (1-p[k]) * beta(k) / (kappa(k) * (alpha(k)-1))
				+ p[k] * (1-p[k]) * (mu.row(k)-mui.transpose()).dot(mu.row(k)-mui.transpose()) / D;
			
			// Update of eta_k^(i) in Eq (26)
			beta(k) = El / Vl;
			alpha(k) = El * beta(k);
			kappa(k) = max(beta(k) / (Vm * (alpha(k)-1)), kappa0);
			mu.row(k) = p[k] * mui.transpose() + (1-p[k]) * mu.row(k);
		}

		n_sum += 1.0;
		n_del = 0.0;
		K0 = K-1;
		if(n_k(K-1) < AA * EPS){
			n_del += n_k(K-1);
			K--;
		}
		for(k=K0-1; k>=0; k--){
			// Deletion step
			if(n_k(k)/n_sum < EPS){
				n_del += n_k(k);

				if(k != K-1){
					n_k(k) = n_k(K-1);
					kappa(k) = kappa(K-1);
					alpha(k) = alpha(K-1);
					beta(k) = beta(K-1);
					mu.row(k) = mu.row(K-1);
				}

				K--;
			}
		}
		n_sum -= n_del;

		if(i % THIN == THIN-1){
			// test loglik
			for(k=0; k<K; k++)
				p[k] = n_k(k)/n_sum;
			
			ll = 0.0;
			for(j=0; j<n_te; j++){
				sum = 0.0;
				for(k=0; k<K; k++){
					kappai = kappa(k) + 1.0;
					mui = (kappa(k)*mu.row(k)+x_te.row(j))/kappai;
					alphai = alpha(k) + D * 0.5;
					betai = beta(k) + (mu.row(k)-x_te.row(j)).dot(mu.row(k)-x_te.row(j)) * 0.5 * kappa(k) / kappai;
				
					sum += p[k] * exp(lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
							- lgamma(alpha(k)) + alpha(k) * log(beta(k)) + 0.5 * D * log(kappa(k)/(2 * M_PI)));
				}
				if(sum < DBL_MIN)
					sum = DBL_MIN;
				ll += log(sum);
			}
			ll /= n_te;

			// post loglik
			post = 0.0;
			for(j=0; j<2000; j++){
				sum = 0.0;
				for(k=0; k<K; k++){
					kappai = kappa(k) + 1.0;
					mui = (kappa(k)*mu.row(k)+x_post.row(j))/kappai;
					alphai = alpha(k) + D * 0.5;
					betai = beta(k) + (mu.row(k)-x_post.row(j)).dot(mu.row(k)-x_post.row(j)) * 0.5 * kappa(k) / kappai;
				
					sum += p[k] * exp(lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
							- lgamma(alpha(k)) + alpha(k) * log(beta(k)) + 0.5 * D * log(kappa(k)/(2 * M_PI)));
				}
				if(sum < DBL_MIN)
					sum = DBL_MIN;
				post += log(sum);
			}
			post /= 2000;

			out1 << K << endl;
			out2 << ll << endl;
			out3 << post << endl;
			cout << i+1 << " : " << K << " " << ll << " " << post << endl;
		}
	}
	in2.close();
	out1.close();
	out2.close();
	out3.close();
	free(p);

	return 0;
}
