#include "function.h"

int main(int argc, char* argv[])
{
	string dir = string(argv[1]), dir2 = string(argv[8]), dir3 = string(argv[5]);
	dir = "/data/home/kuhemian/DPM/" + dir + "/"; 
	
	// AA : alpha (the precision parameter of DP)
	// BB0 : b0 (the rate parameter for the prior distribution of lambda)
	// EPS2 : eps (deletion step)
	// B : # of observations in each mini-batch
	double AA;
	AA = atoi(argv[2])/10.0;
	double BB0;
	BB0 = atoi(argv[3])/10.0;
	double EPS2;
	EPS2 = pow(0.1,atoi(argv[4]));
	int ITER, BURNIN, THIN;
	BURNIN = atoi(argv[5]);
	THIN = atoi(argv[6]);
	ITER = BURNIN + THIN * atoi(argv[7]);
	int B;
	B = atoi(argv[8]);
	int PP;
	PP = atoi(argv[9]);

	// n : total # of observations in the training set
	// n_te : total # of observations in the test set
	// D : dimension
	// K : # of occupied components
	// nBatch : # of mini-batches
	// Q : # of MCMC samples
	// T : # of subcomponents
	int n, n_te, D, K, K0, K2, nBatch, Q=(ITER-BURNIN)/THIN, Q2, T=2, idx, cnt, cnt2, cnt3;
	double kappa00=0.1, kappai, alpha00=1.0, alphai, beta00=BB0, betai, ll, ll_prev, ll_best, ll_post, sum, sum2, p_max, m_sum, m_sum_p, m_del, mm, a;
	int i, j, d, k, z_p, t_p, iter, iter2, iter3, iter4, n_sum, restart, K_max, t;
	double s, ds, ds2, y, digam = -digamma(1.0), aa=AA;
	clock_t start, finish;
	cout << aa << " " << BB0 << " " << BURNIN << " " << THIN << " " << PP << " " << EPS2 << endl;

	string loc;
	loc = dir + "info.txt";
	ifstream in1(loc);
	in1 >> n >> n_te >> D;
	in1.close();
	nBatch = n/B;

	// x : training observations; x_te : test observations; x_post : observations sampled from the posterior predictive distribution
	MatrixXd x(B,D), x_te(n_te,D), x_post(2000,D), mu_e(K_MAX,D), phi2(Q,D), lambda1(Q,K_MAX),
		x_sum2(K_MAX,T), kappa(K_MAX,T), alpha(K_MAX,T), beta(K_MAX,T), kappa0(K_MAX,T), alpha0(K_MAX,T), beta0(K_MAX,T);
	VectorXd m(K_MAX), mui(D), kappa_e(K_MAX), alpha_e(K_MAX), beta_e(K_MAX), pi_e(K_MAX), x_bar(D), lambda2(Q), p1(K_MAX), p2(K_MAX);
	VectorXd Ak(K_MAX), Bk(K_MAX), Dk(K_MAX);
	MatrixXd Ck(K_MAX,D), gamma(Q,K_MAX), m_kt(K_MAX,T);
	VectorXi n_k(K_MAX), z2(Q), p_k(K_MAX);
	MatrixXi n_kt(K_MAX,T), z(B,2);
	MatrixXd *phi1 = new MatrixXd[Q];
	for(i=0; i<Q; i++)	phi1[i] = MatrixXd(K_MAX,D);

	double **p_t =  (double**)malloc(K_MAX*sizeof(double*));
	MatrixXd *x_sum = new MatrixXd[K_MAX], *mu = new MatrixXd[K_MAX], *mu0 = new MatrixXd[K_MAX];
	for(k=0; k<K_MAX; k++){
		p_t[k] = (double*)malloc(T*sizeof(double));
		x_sum[k] = MatrixXd(T,D);
		mu[k] = MatrixXd(T,D);
		mu0[k] = MatrixXd(T,D);
	}

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

	double *p = (double*)malloc(K_MAX*sizeof(double));
	double *q = (double*)malloc(K_MAX*sizeof(double));
	
	dir = dir + "result/WCA_mix/" + dir2 + "/" + dir3 + "/";
	loc = dir + "K" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out1(loc);
	loc = dir + "ll" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out2(loc);
	loc = dir + "a" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out3(loc);
	loc = dir + "test" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out4(loc);
	loc = dir + "phi" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out5(loc);
	loc = dir + "post" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out6(loc);
	loc = dir + "log" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out7(loc);
	loc = dir + "mu" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out8(loc);
	loc = dir + "sig" + to_string((long long)(AA*10.0)) + "_" + to_string((long long)PP) + ".txt";
	ofstream out9(loc);

	mt19937 r((unsigned int)SEED);
	normal_distribution<double> rnorm(0.0,1.0);
	uniform_int_distribution<int> runif_int;
	uniform_real_distribution<double> runif;

	K0 = 0;
	m_sum = 0.0;
	for(int batch=0; batch<nBatch; batch++){
		start = clock();

		K = K0;
		for(i=0; i<B; i++){
			for(d=0; d<D; d++)		
				in2 >> x(i,d);
			
			// Initialize z_i using p(z_i=(k,t)|-) in page 17
			for(k=0; k<K0; k++){
				p[k] = 0.0;
				for(t=0; t<T; t++){
					kappai = kappa(k,t) + 1.0;
					mui = (kappa(k,t)*mu[k].row(t)+x.row(i))/kappai;
					alphai = alpha(k,t) + D * 0.5;
					betai = beta(k,t) + (mu[k].row(t)-x.row(i)).dot(mu[k].row(t)-x.row(i)) * 0.5 * kappa(k,t) / kappai;

					p_t[k][t] = (m_kt(k,t)/m(k)) * exp(lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
												- lgamma(alpha(k,t)) + alpha(k,t) * log(beta(k,t)) + 0.5 * D * log(kappa(k,t)/(2 * M_PI)));
					p[k] += p_t[k][t];
				}
				p[k] = log(p[k]);
			}
			
			t = 0;
			for(k=K0; k<K; k++){
				kappai = kappa(k,t) + 1.0;
				mui = (kappa(k,t)*mu[k].row(t)+x.row(i))/kappai;
				alphai = alpha(k,t) + D * 0.5;
				betai = beta(k,t) + (mu[k].row(t)-x.row(i)).dot(mu[k].row(t)-x.row(i)) * 0.5 * kappa(k,t) / kappai;

				p[k] = lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
						- lgamma(alpha(k,t)) + alpha(k,t) * log(beta(k,t)) + 0.5 * D * log(kappa(k,t)/(2 * M_PI));
			}

			kappai = kappa00 + 1.0;
			mui = x.row(i)/kappai;
			alphai = alpha00 + D * 0.5;
			betai = beta00 + x.row(i).dot(x.row(i)) * 0.5 * kappa00 / kappai;

			p[K] = lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
					- lgamma(alpha00) + alpha00 * log(beta00) + 0.5 * D * log(kappa00/(2 * M_PI));

			p_max = max(p,K+1);
			for(k=0; k<=K; k++)	p[k] = exp(p[k]-p_max);
			
			// Sample z_i
			z(i,0) = sample(p, K+1, r);
			if(z(i,0) == K){	
				// new component
				z(i,1) = 0;

				kappa(K,0) = kappai;
				mu[K].row(0) = mui;
				alpha(K,0) = alphai;
				beta(K,0) = betai;

				kappa0(K,0) = kappa00;
				mu0[K].row(0).setZero();
				alpha0(K,0) = alpha00;
				beta0(K,0) = beta00;
				
				x_sum[K].row(0) = x.row(i);
				x_sum2(K,0) = x.row(i).dot(x.row(i));
				n_k(K) = 1; n_kt(K,0) = 1;
				K++;
			}
			else{				
				// existing components
				k = z(i,0);
				if(k >= K0)		t = z(i,1) = 0;
				else			t = z(i,1) = sample(p_t[k], T, r);
				
				kappai = kappa(k,t) + 1.0;
				mui = (kappa(k,t)*mu[k].row(t)+x.row(i))/kappai;
				alphai = alpha(k,t) + D * 0.5;
				betai = beta(k,t) + (mu[k].row(t)-x.row(i)).dot(mu[k].row(t)-x.row(i)) * 0.5 * kappa(k,t) / kappai;

				kappa(k,t) = kappai;
				mu[k].row(t) = mui;
				alpha(k,t) = alphai;
				beta(k,t) = betai;

				x_sum[k].row(t) += x.row(i);
				x_sum2(k,t) += x.row(i).dot(x.row(i));
				n_k(k)++; n_kt(k,t)++;
			}
		}

		p_k.setZero();
		p1.setZero();
		p2.setZero();
		K_max = K0;
		
		// MCMC
		for(iter=0; iter<ITER; iter++){
			for(i=0; i<B; i++){
				z_p = z(i,0); t_p = z(i,1);

				x_sum[z_p].row(t_p) -= x.row(i);
				x_sum2(z_p,t_p) -= x.row(i).dot(x.row(i));
				n_k(z_p)--; n_kt(z_p,t_p)--;
				
				if(n_k(z_p) == 0){
					// Remove the unoccupied component
					if(z_p >= K0){
						K--;
						if(z_p != K){
							x_sum[z_p].row(0) = x_sum[K].row(0);
							x_sum2(z_p,0) = x_sum2(K,0);
							n_k(z_p) = n_k(K);
							n_kt(z_p,0) = n_kt(K,0);

							for(j=0; j<B; j++)
								if(z(j,0) == K) z(j,0) = z_p;

							kappa(z_p,0) = kappa(K,0);
							mu[z_p].row(0) = mu[K].row(0);
							alpha(z_p,0) = alpha(K,0);
							beta(z_p,0) = beta(K,0);
						}
					}
					else{
						kappa(z_p,t_p) = kappa0(z_p,t_p);
						mu[z_p].row(t_p) = mu0[z_p].row(t_p);
						alpha(z_p,t_p) = alpha0(z_p,t_p);
						beta(z_p,t_p) = beta0(z_p,t_p);
					}
				}
				else{
					if(n_kt(z_p,t_p) == 0){
						kappa(z_p,t_p) = kappa0(z_p,t_p);
						mu[z_p].row(t_p) = mu0[z_p].row(t_p);
						alpha(z_p,t_p) = alpha0(z_p,t_p);
						beta(z_p,t_p) = beta0(z_p,t_p);
					}
					else{
						kappa(z_p,t_p) = kappa0(z_p,t_p) + (double)n_kt(z_p,t_p);
						mu[z_p].row(t_p) = (kappa0(z_p,t_p)*mu0[z_p].row(t_p)+x_sum[z_p].row(t_p))/kappa(z_p,t_p);
						alpha(z_p,t_p) = alpha0(z_p,t_p) + n_kt(z_p,t_p) * D * 0.5;
						x_bar = x_sum[z_p].row(t_p) / n_kt(z_p,t_p);
						beta(z_p,t_p) = beta0(z_p,t_p) + (mu0[z_p].row(t_p).transpose()-x_bar).dot(mu0[z_p].row(t_p).transpose()-x_bar) * 0.5 * kappa0(z_p,t_p) * n_kt(z_p,t_p) / kappa(z_p,t_p) 
									+ (x_sum2(z_p,t_p) - n_kt(z_p,t_p) * x_bar.dot(x_bar)) * 0.5;
					}
				}

				// Calculate p(z_i=(k,t)|-) in page 17
				for(k=0; k<K0; k++){
					p[k] = 0.0;
					for(t=0; t<T; t++){
						kappai = kappa(k,t) + 1.0;
						mui = (kappa(k,t)*mu[k].row(t)+x.row(i))/kappai;
						alphai = alpha(k,t) + D * 0.5;
						betai = beta(k,t) + (mu[k].row(t)-x.row(i)).dot(mu[k].row(t)-x.row(i)) * 0.5 * kappa(k,t) / kappai;

						p_t[k][t] = (m_kt(k,t)+n_kt(k,t))/(m(k)+n_k(k)) * exp(lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
													- lgamma(alpha(k,t)) + alpha(k,t) * log(beta(k,t)) + 0.5 * D * log(kappa(k,t)/(2 * M_PI)));
						p[k] += p_t[k][t];
					}
					p[k] = log(m(k)+n_k(k)) + log(p[k]);
				}
			
				t = 0;
				for(k=K0; k<K; k++){
					kappai = kappa(k,t) + 1.0;
					mui = (kappa(k,t)*mu[k].row(t)+x.row(i))/kappai;
					alphai = alpha(k,t) + D * 0.5;
					betai = beta(k,t) + (mu[k].row(t)-x.row(i)).dot(mu[k].row(t)-x.row(i)) * 0.5 * kappa(k,t) / kappai;

					p[k] = log((double)n_k(k)) + lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
							- lgamma(alpha(k,t)) + alpha(k,t) * log(beta(k,t)) + 0.5 * D * log(kappa(k,t)/(2 * M_PI));
				}

				kappai = kappa00 + 1.0;
				mui = x.row(i)/kappai;
				alphai = alpha00 + D * 0.5;
				betai = beta00 + x.row(i).dot(x.row(i)) * 0.5 * kappa00 / kappai;

				p[K] = log(aa) + lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
						- lgamma(alpha00) + alpha00 * log(beta00) + 0.5 * D * log(kappa00/(2 * M_PI));

				p_max = max(p,K+1);
				for(k=0; k<=K; k++)	p[k] = exp(p[k]-p_max);
				
				// Sample z_i
				z(i,0) = sample(p, K+1, r);
				if(z(i,0) == K){
					// new component
					z(i,1) = 0;

					kappa(K,0) = kappai;
					mu[K].row(0) = mui;
					alpha(K,0) = alphai;
					beta(K,0) = betai;

					kappa0(K,0) = kappa00;
					mu0[K].row(0).setZero();
					alpha0(K,0) = alpha00;
					beta0(K,0) = beta00;
				
					x_sum[K].row(0) = x.row(i);
					x_sum2(K,0) = x.row(i).dot(x.row(i));
					n_k(K) = 1; n_kt(K,0) = 1;
					K++;
				}
				else{
					// existing components
					k = z(i,0); 
					if(k >= K0)		t = z(i,1) = 0;
					else			t = z(i,1) = sample(p_t[k], T, r);

					kappai = kappa(k,t) + 1.0;
					mui = (kappa(k,t)*mu[k].row(t)+x.row(i))/kappai;
					alphai = alpha(k,t) + D * 0.5;
					betai = beta(k,t) + (mu[k].row(t)-x.row(i)).dot(mu[k].row(t)-x.row(i)) * 0.5 * kappa(k,t) / kappai;

					kappa(k,t) = kappai;
					mu[k].row(t) = mui;
					alpha(k,t) = alphai;
					beta(k,t) = betai;

					x_sum[k].row(t) += x.row(i);
					x_sum2(k,t) += x.row(i).dot(x.row(i));
					n_k(k)++; n_kt(k,t)++;
				}
			}

			if(iter > BURNIN && (iter-BURNIN)%THIN == THIN-1){
				// save  MCMC samples
				p_k(K-K0)++;
				if(K > K_max)	K_max = K;
				j = (iter-BURNIN)/THIN;

				n_sum = B;
				for(k=0; k<K0; k++){
					// Sample from Eq (23)
					for(t=0; t<T; t++)	p[t] = (m_kt(k,t)+n_kt(k,t))/(m(k)+n_k(k));
					t = sample(p, T, r, true);
					lambda1(j,k) = rgamma(alpha(k,t), beta(k,t), r);
					for(d=0; d<D; d++)	phi1[j](k,d) = mu[k](t,d) + rnorm(r) / sqrt(kappa(k,t) * lambda1(j,k));
					
					// Save sufficient statistics for Eq (22)
					p1(k) += digamma(n_k(k) + m(k)) - digamma(B + m_sum + aa);
					p2(k) += (n_k(k) + m(k)) / (B + m_sum + aa);
					n_sum -= n_k(k);
				}
				p1(K0) += digamma(n_sum + aa) - digamma(B + m_sum + aa);
				p2(K0) += (n_sum + aa) / (B + m_sum + aa);

				for(k=K0; k<K; k++)
					p[k-K0] = n_k(k) / (aa + n_sum);
				p[K-K0] = aa / (aa + n_sum);

				// Sample from Eq (24)
				k = sample(p, K-K0+1, r, true) + K0;
				if(k == K){
					lambda2(j) = rgamma(alpha00, beta00, r);
					for(d=0; d<D; d++) phi2(j,d) = rnorm(r) / sqrt(kappa00 * lambda2(j));
				}
				else{
					lambda2(j) = rgamma(alpha(k,0), beta(k,0), r);
					for(d=0; d<D; d++)	phi2(j,d) = mu[k](0,d) + rnorm(r) / sqrt(kappa(k,0) * lambda2(j));
				}
			}
		}

		// Estimation of proxy parameters
		for(k=0; k<K0; k++){
			p[k] = p2(k) / Q;
			p1(k) /= Q;
		}
		p[K0] = p2(K0) / Q;
		p1(K0) /= Q;

		s = m_sum_p = m_sum + B + aa;
		if(K0 > 0){
			// 4.2.2 Estimation of gamma_k^new for k=1,...,K0
			for(iter2=0; iter2<ITER2; iter2++){
				// Eq (40) of [Minka, 2000]
				for(iter3=0; iter3<ITER3; iter3++){
					ds = digamma(s);
					ds2 = trigamma(s);
					for(k=0; k<=K0; k++){
						ds += p[k] * (p1(k) - digamma(s*p[k]));
						ds2 -= p[k] * p[k] * trigamma(s*p[k]);
					}
					s = 1.0/(1.0/s + ds/(ds2 * s * s));
				}

				// Eq (46), (47) of [Minka, 2000]
				for(iter3=0; iter3<ITER3; iter3++){
					sum = 0.0;
					for(k=0; k<=K0; k++)
						sum += p[k] * (p1(k) - digamma(s*p[k]));

					sum2 = 0.0;
					for(k=0; k<=K0; k++){
						y = p1(k) - sum;
						if(y>=-2.22)
							q[k] = exp(y) + 0.5;
						else
							q[k] = -1.0/(y+digam);

						for(iter4=0; iter4<ITER3; iter4++)	
							q[k] -= (digamma(q[k])-y)/trigamma(q[k]);

						sum2 += q[k];
					}

					for(k=0; k<=K0; k++)
						p[k] = q[k]/sum2;
				}
			}
		}
		s = min(m_sum_p, s);

		mm = s * p[K0];
		m_sum = 0.0;

		for(k=0; k<K0; k++){
			m(k) = s * p[k];
			m_sum += m(k);

			// 4.2.3 Estimation of eta_k^new for k=1,...,K0
			cnt = cnt2 = cnt3 = 0;
			ll_best = -DBL_MAX;
			runif_int=uniform_int_distribution<int>(0,T-1);
			for(iter3=0; iter3<ITER3; iter3++){
				for(t=0; t<T; t++){
					mu_e.row(t) = mu[k].row(t);
					alpha_e(t) = alpha(k,t);
					beta_e(t) = beta(k,t);
					kappa_e(t) = kappa(k,t);

					Ak(t) = Bk(t) = Dk(t) = 0.0;
					Ck.row(t).setZero();
				}

				for(j=0; j<Q; j++){
					if(cnt < 5 && cnt2 < 5 && cnt3 < 5){
						p_max = -DBL_MAX;
						for(t=0; t<T; t++){
							q[t] = -lgamma(alpha_e(t)) + alpha_e(t) * log(beta_e(t)) - 0.5 * D * log(2*M_PI/kappa_e(t))
								+ (alpha_e(t) + 0.5 * D - 1) * log(lambda1(j,k)) - lambda1(j,k) * beta_e(t) - 0.5 * kappa_e(t) * lambda1(j,k) * (phi1[j].row(k)-mu_e.row(t)).dot(phi1[j].row(k)-mu_e.row(t));
							if(q[t] > p_max)		p_max = q[t];
						}
						for(t=0; t<T; t++) q[t] = exp(q[t]-p_max);
						t = z2(j) = sample(q,T,r);
					}
					else{
						if(j==0){ 
							cout << "cnt : " << cnt << " " << cnt2 << " " << cnt3 << endl; 
							out7 << "cnt : " << cnt << " " << cnt2 << " " << cnt3 << endl;
						}
						t = z2(j) = runif_int(r);
					}

					Ak(t) += log(lambda1(j,k));
					Bk(t) += lambda1(j,k);
					Ck.row(t) += lambda1(j,k) * phi1[j].row(k);
					Dk(t) += 1.0;
				}
				if(cnt == 5)	cnt = 0;
				if(cnt2 == 5)	cnt2 = 0;
				if(cnt3 == 5)	cnt3 = 0;
				
				restart = 0;
				for(t=0; t<T; t++){
					if(Dk(t) < 0.5){
						restart = 1;
						cout << "restart 1-1 : initial Dk is equal to zero, " << t << " : " << Dk(t) << " cnt : " << cnt << " " << cnt2 << " " << cnt3 << endl;						out7 << "restart 1-1 : initial Dk is equal to zero, " << t << " : " << Dk(t) << endl;
						mu_e.row(t).setZero();
						alpha_e(t) = alpha00;
						beta_e(t) = beta00;
						kappa_e(t) = kappa00;
						cnt++;
						break;
					}
				}
				if(restart == 1){
					iter3--;
					continue;
				}
				else
					cnt = 0;

				for(t=0; t<T; t++){
					pi_e(t) = Dk(t) / Q;
					Ak(t) /= Dk(t);
					Bk(t) /= Dk(t);
					Ck.row(t) /= Dk(t);
					
					if(fabs(log(Bk(t))-Ak(t)) < EPS)
							Bk(t) = exp(Ak(t)+EPS);

					a = 0.5/(log(Bk(t))-Ak(t));
					for(iter2=0; iter2<ITER3; iter2++)
						a = 1.0/(1.0/a + (digamma(a)-log(a)+log(Bk(t))-Ak(t))/(a*a*(trigamma(a)-1.0/a)));
					
					alpha_e(t) = a;
					beta_e(t) = a / Bk(t);
					mu_e.row(t) = Ck.row(t) / Bk(t);

					sum = 0.0;
					for(j=0; j<Q; j++){
						if(z2(j) == t)
							sum += lambda1(j,k) * (mu_e.row(t)-phi1[j].row(k)).dot(mu_e.row(t)-phi1[j].row(k));
					}
					kappa_e(t) = D * Dk(t) / sum;
				}

				ll_prev = -DBL_MAX;
				for(iter=0; iter<ITER2; iter++){
					for(t=0; t<T; t++){
						Ak(t) = Bk(t) = Dk(t) = 0.0;
						Ck.row(t).setZero();
					}

					for(j=0; j<Q; j++){
						p_max = -DBL_MAX;
						for(t=0; t<T; t++){
							gamma(j,t) = log(pi_e(t)) - lgamma(alpha_e(t)) + alpha_e(t) * log(beta_e(t)) - 0.5 * D * log(2*M_PI/kappa_e(t))
								+ (alpha_e(t) + 0.5 * D - 1) * log(lambda1(j,k)) - lambda1(j,k) * beta_e(t) - 0.5 * kappa_e(t) * lambda1(j,k) * (phi1[j].row(k)-mu_e.row(t)).dot(phi1[j].row(k)-mu_e.row(t));
							if(gamma(j,t) > p_max)		p_max = gamma(j,t);
						}

						sum = 0.0;
						for(t=0; t<T; t++){
							gamma(j,t) = exp(gamma(j,t) - p_max);
							sum += gamma(j,t);
						}
					
						for(t=0; t<T; t++){
							gamma(j,t) /= sum;

							Ak(t) += gamma(j,t) * log(lambda1(j,k));
							Bk(t) += gamma(j,t) * lambda1(j,k);
							Ck.row(t) += gamma(j,t) * lambda1(j,k) * phi1[j].row(k);
							Dk(t) += gamma(j,t);
						}
					}
					
					for(t=0; t<T; t++){
						if(Dk(t) < Q * EPS){
							restart = 1;
							cout << "restart 1-2 : Dk is too small, " << t << " : " << Dk(t) << " cnt : " << cnt << " " << cnt2 << " " << cnt3 << endl;
							out7 << "restart 1-2 : Dk is too small, " << t << " : " << Dk(t) << " cnt : " << cnt << " " << cnt2 << " " << cnt3 << endl;
							cnt3++;
							break;
						}
						pi_e(t) = Dk(t) / Q;
						Ak(t) /= Dk(t);
						Bk(t) /= Dk(t);
						Ck.row(t) /= Dk(t);

						if(Dk(t) < 1.5){
							mu_e.row(t).setZero();
							alpha_e(t) = alpha00;
							beta_e(t) = beta00;
							kappa_e(t) = kappa00;
						}
						else{
							if(fabs(log(Bk(t))-Ak(t)) < EPS)
								Bk(t) = exp(Ak(t)+EPS);

							a = 0.5/(log(Bk(t))-Ak(t));
							for(iter2=0; iter2<ITER3; iter2++)
								a = 1.0/(1.0/a + (digamma(a)-log(a)+log(Bk(t))-Ak(t))/(a*a*(trigamma(a)-1.0/a)));

							mu_e.row(t) = Ck.row(t) / Bk(t);
							alpha_e(t) = a;
							beta_e(t) = a / Bk(t);
							sum = 0.0;
							for(j=0; j<Q; j++)
								sum += gamma(j,t) * lambda1(j,k) * (mu_e.row(t)-phi1[j].row(k)).dot(mu_e.row(t)-phi1[j].row(k));
							kappa_e(t) = D * Dk(t) / sum;
						}
					}
					if(restart == 1)	break;

					ll = 0.0;
					for(j=0; j<Q; j++){
						sum = 0.0;
						for(t=0; t<T; t++)
							sum += pi_e(t) * exp(-lgamma(alpha_e(t)) + alpha_e(t) * log(beta_e(t)) - 0.5 * D * log(2*M_PI/kappa_e(t))
								+ (alpha_e(t) + 0.5 * D - 1) * log(lambda1(j,k)) - lambda1(j,k) * beta_e(t) - 0.5 * kappa_e(t) * lambda1(j,k) * (phi1[j].row(k)-mu_e.row(t)).dot(phi1[j].row(k)-mu_e.row(t)));
						
						if(sum < DBL_MIN)	sum = DBL_MIN;
						ll += log(sum);
					}
					ll /= (double)Q;

					//cout << iter3 << " " << iter << " " << ll_best << " " << ll_prev << " " << ll << endl;
					//out7 << iter3 << " " << iter << " " << ll_best << " " << ll_prev << " " << ll << endl;
					
					if(ll > ll_best){
						ll_best = ll;
						for(t=0; t<T; t++){
							alpha(k,t) = alpha0(k,t) = alpha_e(t);
							beta(k,t) = beta0(k,t) = beta_e(t);
							mu[k].row(t) = mu0[k].row(t) = mu_e.row(t);
							kappa(k,t) = kappa0(k,t) = kappa_e(t);
							m_kt(k,t) = m(k) * pi_e(t);
						}
						cnt3 = 0;
					}

					if(fabs(ll_prev-ll) < EPS){
						if(iter == 1)	cnt2++;
						break;
					}
					else
						if(iter > 0)	cnt2 = 0;

					if(ll > ll_prev)
						ll_prev = ll;
				}

				if(restart == 1){
					iter3--;
					continue;
				}
			}
		}

		// 4.2.1 Estimation of K^new
		if(mm < EPS)
			K2 = K0;
		else{
			for(k=K_max-K0; k>=0; k--){
				if(k != K_max-K0) p_k(k) += p_k(k+1);
				if(p_k(k) > 0.01*Q){
					K2 = K0 + k;
					break;
				}
			}
		}
		cout << K0 << " " << K2 << " " << K_max << endl;
		
		// 4.2.4 Estimation of gamma_k^new, eta_k^new for k=K0+1,...,K^new
		if(batch == 0){
			K2 = K;
			for(k=0; k<K; k++){
				m(k) = n_k(k);
				for(t=0; t<T; t++){
					alpha(k,t) = alpha0(k,t) = alpha(k,0);
					beta(k,t) = beta0(k,t) = beta(k,0);
					mu[k].row(t) = mu0[k].row(t) = mu[k].row(0);
					kappa(k,t) = kappa0(k,t) = kappa(k,0);
					m_kt(k,t) = m(k) / T;
				}
			}
		}
		else if(K2 > K0){
			if(K2 > K){
				for(k=K; k<K2; k++){
					mu[k].row(0).setZero();
					alpha(k,0) = alpha00;
					beta(k,0) = beta00;
					kappa(k,0) = kappa00;
				}
			}

			cnt = cnt2 = cnt3 = 0;
			ll_best = -DBL_MAX;
			runif_int=uniform_int_distribution<int>(0,K2-K0);
			for(iter3=0; iter3<ITER1; iter3++){
				mu_e.row(0).setZero();
				alpha_e(0) = alpha00; beta_e(0) = beta00; kappa_e(0) = kappa00;
				Ak(0) = Bk(0) = Dk(0) = 0.0;
				Ck.row(0).setZero();

				for(k=1; k<=K2-K0; k++){
					mu_e.row(k) = mu[K0+k-1].row(0);
					alpha_e(k) = alpha(K0+k-1,0);
					beta_e(k) = beta(K0+k-1,0);
					kappa_e(k) = kappa(K0+k-1,0);

					Ak(k) = Bk(k) = Dk(k) = 0.0;
					Ck.row(k).setZero();
				}

				for(j=0; j<Q; j++){
					if(cnt < 5 && cnt2 < 5 && cnt3 < 5){
						p_max = -DBL_MAX;
						for(k=0; k<=K2-K0; k++){
							q[k] = -lgamma(alpha_e(k)) + alpha_e(k) * log(beta_e(k)) - 0.5 * D * log(2*M_PI/kappa_e(k))
								+ (alpha_e(k) + 0.5 * D - 1) * log(lambda2(j)) - lambda2(j) * beta_e(k) - 0.5 * kappa_e(k) * lambda2(j) * (phi2.row(j)-mu_e.row(k)).dot(phi2.row(j)-mu_e.row(k));
							if(q[k] > p_max)		p_max = q[k];
						}
						for(k=0; k<=K2-K0; k++) q[k] = exp(q[k]-p_max);
						z2(j) = sample(q,K2-K0+1,r);
					}
					else{
						if(j==0) cout << "cnt : " << cnt << " " << cnt2 << " " << cnt3 << endl;
						z2(j) = runif_int(r);
					}

					Ak(z2(j)) += log(lambda2(j));
					Bk(z2(j)) += lambda2(j);
					Ck.row(z2(j)) += lambda2(j) * phi2.row(j);
					Dk(z2(j)) += 1.0;
				}
				if(cnt == 5)	cnt = 0;
				if(cnt2 == 5)	cnt2 = 0;
				if(cnt3 == 5)	cnt3 = 0;

				restart = 0;
				for(k=0; k<=K2-K0; k++){
					if(Dk(k) < 0.5){
						restart = 1;
						cout << "restart 2-1 : initial Dk is equal to zero, " << k << " : " << Dk(k) << endl;
						out7 << "restart 2-1 : initial Dk is equal to zero, " << k << " : " << Dk(k) << endl;
						mu_e.row(k).setZero();
						alpha_e(k) = alpha00;
						beta_e(k) = beta00;
						kappa_e(k) = kappa00;
						cnt++;
						break;
					}
				}
				if(restart == 1){
					iter3--;
					continue;
				}
				else
					cnt = 0;

				pi_e(0) = Dk(0) / Q;
				for(k=1; k<=K2-K0; k++){
					pi_e(k) = Dk(k) / Q;
					Ak(k) /= Dk(k);
					Bk(k) /= Dk(k);
					Ck.row(k) /= Dk(k);
					
					if(fabs(log(Bk(k))-Ak(k)) < EPS)
							Bk(k) = exp(Ak(k)+EPS);

					a = 0.5/(log(Bk(k))-Ak(k));
					for(iter2=0; iter2<ITER3; iter2++)
						a = 1.0/(1.0/a + (digamma(a)-log(a)+log(Bk(k))-Ak(k))/(a*a*(trigamma(a)-1.0/a)));
					
					alpha_e(k) = a;
					beta_e(k) = a / Bk(k);
					mu_e.row(k) = Ck.row(k) / Bk(k);

					sum = 0.0;
					for(j=0; j<Q; j++){
						if(z2(j) == k)
							sum += lambda2(j) * (mu_e.row(k)-phi2.row(j)).dot(mu_e.row(k)-phi2.row(j));
					}
					kappa_e(k) = D * Dk(k) / sum;
				}

				ll_prev = -DBL_MAX;
				for(iter=0; iter<ITER2; iter++){
					for(k=0; k<=K2-K0; k++){
						Ak(k) = Bk(k) = Dk(k) = 0.0;
						Ck.row(k).setZero();
					}

					for(j=0; j<Q; j++){
						p_max = -DBL_MAX;
						for(k=0; k<=K2-K0; k++){
							gamma(j,k) = log(pi_e(k)) - lgamma(alpha_e(k)) + alpha_e(k) * log(beta_e(k)) - 0.5 * D * log(2*M_PI/kappa_e(k))
								+ (alpha_e(k) + 0.5 * D - 1) * log(lambda2(j)) - lambda2(j) * beta_e(k) - 0.5 * kappa_e(k) * lambda2(j) * (phi2.row(j)-mu_e.row(k)).dot(phi2.row(j)-mu_e.row(k));
							if(gamma(j,k) > p_max)		p_max = gamma(j,k);
						}

						sum = 0.0;
						for(k=0; k<=K2-K0; k++){
							gamma(j,k) = exp(gamma(j,k) - p_max);
							sum += gamma(j,k);
						}
					
						for(k=0; k<=K2-K0; k++){
							gamma(j,k) /= sum;

							Ak(k) += gamma(j,k) * log(lambda2(j));
							Bk(k) += gamma(j,k) * lambda2(j);
							Ck.row(k) += gamma(j,k) * lambda2(j) * phi2.row(j);
							Dk(k) += gamma(j,k);
						}
					}

					pi_e(0) = Dk(0) / Q;
					for(k=1; k<=K2-K0; k++){
						if(Dk(k) < Q * EPS){
							restart = 1;
							cout << "restart 2-2 : Dk is too small, " << k << " : " << Dk(k) << endl;
							out7 << "restart 2-2 : Dk is too small, " << k << " : " << Dk(k) << endl;
							cnt3++;
							break;
						}
						pi_e(k) = Dk(k) / Q;
						Ak(k) /= Dk(k);
						Bk(k) /= Dk(k);
						Ck.row(k) /= Dk(k);

						if(Dk(k) < 1.5){
							mu_e.row(k).setZero();
							alpha_e(k) = alpha00;
							beta_e(k) = beta00;
							kappa_e(k) = kappa00;
						}
						else{
							if(fabs(log(Bk(k))-Ak(k)) < EPS)
								Bk(k) = exp(Ak(k)+EPS);

							a = 0.5/(log(Bk(k))-Ak(k));
							for(iter2=0; iter2<ITER3; iter2++)
								a = 1.0/(1.0/a + (digamma(a)-log(a)+log(Bk(k))-Ak(k))/(a*a*(trigamma(a)-1.0/a)));

							mu_e.row(k) = Ck.row(k) / Bk(k);
							alpha_e(k) = a;
							beta_e(k) = a / Bk(k);
							sum = 0.0;
							for(j=0; j<Q; j++)
								sum += gamma(j,k) * lambda2(j) * (mu_e.row(k)-phi2.row(j)).dot(mu_e.row(k)-phi2.row(j));
							kappa_e(k) = D * Dk(k) / sum;
						}
					}
					if(restart == 1)	break;
					else 
						cnt3 = 0;

					ll = 0.0;
					for(j=0; j<Q; j++){
						sum = 0.0;
						for(k=0; k<=K2-K0; k++)
							sum += pi_e(k) * exp(-lgamma(alpha_e(k)) + alpha_e(k) * log(beta_e(k)) - 0.5 * D * log(2*M_PI/kappa_e(k))
								+ (alpha_e(k) + 0.5 * D - 1) * log(lambda2(j)) - lambda2(j) * beta_e(k) - 0.5 * kappa_e(k) * lambda2(j) * (phi2.row(j)-mu_e.row(k)).dot(phi2.row(j)-mu_e.row(k)));
						
						if(sum < DBL_MIN)	sum = DBL_MIN;
						ll += log(sum);
					}
					ll /= (double)Q;
					
					//cout << iter3 << " " << iter << " " << ll_best << " " << ll_prev << " " << ll << endl;
					//out7 << iter3 << " " << iter << " " << ll_best << " " << ll_prev << " " << ll << endl;
					
					if(ll > ll_best){
						ll_best = ll;
						for(k=1; k<=K2-K0; k++){
							m(K0+k-1) = mm * pi_e(k);
							for(t=0; t<T; t++){
								alpha(K0+k-1,t) = alpha0(K0+k-1,t) = alpha_e(k);
								beta(K0+k-1,t) = beta0(K0+k-1,t) = beta_e(k);
								mu[K0+k-1].row(t) = mu0[K0+k-1].row(t) = mu_e.row(k);
								kappa(K0+k-1,t) = kappa0(K0+k-1,t) = kappa_e(k);
								m_kt(K0+k-1,t) = m(K0+k-1) / T;
							}
						}
					}

					if(fabs(ll_prev-ll) < EPS){
						if(iter == 1)	cnt2++;
						break;
					}
					else
						if(iter > 0)	cnt2 = 0;

					if(ll > ll_prev)
						ll_prev = ll;
				}

				if(restart == 1){
					iter3--;
					continue;
				}
			}
		}

		cout << mm << endl;
		for(k=K2-1; k>=K0; k--){
			m_sum += m(k);
			cout << k << " " << m(k) << " " << kappa(k) << " " << alpha(k) << " " << beta(k) << endl;
			out7 << k << " " << m(k) << " " << kappa(k) << " " << alpha(k) << " " << beta(k) << endl;
		}

		// Deletion step
		m_del = 0.0;
		for(k=K0-1; k>=0; k--){
			if(m(k) < m_sum * EPS2){
				cout << "delete : " << k << " " << m(k) << endl;
				out7 << "delete : " << k << " " << m(k) << endl;
				m_del += m(k);
				
				K2--;
				if(k != K2){
					m(k) = m(K2); m_kt.row(k) = m_kt.row(K2);
					alpha.row(k) = alpha0.row(k) = alpha0.row(K2);
					beta.row(k) = beta0.row(k) = beta0.row(K2);
					mu[k] = mu0[k] = mu0[K2];
					kappa.row(k) = kappa0.row(k) = kappa.row(K2);
				}
			}
		}
		m_sum -= m_del;

		if(m_del > 0)
			cout << "m_del : " << m_del << endl;

		cout << m_sum_p << " " << s << " " << m_sum << endl;

		// Initialization
		K0 = K2;
		for(k=0; k<K0; k++){
			x_sum[k].setZero();
			x_sum2.row(k).setZero();
			n_k(k) = 0; n_kt.row(k).setZero();
		}

		// test loglik
		ll = 0.0;
		for(j=0; j<n_te; j++){
			sum = 0.0;
			for(k=0; k<K0; k++){
				for(t=0; t<T; t++){
					kappai = kappa(k,t) + 1.0;
					mui = (kappa(k,t)*mu[k].row(t)+x_te.row(j))/kappai;
					alphai = alpha(k,t) + D * 0.5;
					betai = beta(k,t) + (mu[k].row(t)-x_te.row(j)).dot(mu[k].row(t)-x_te.row(j)) * 0.5 * kappa(k,t) / kappai;

					sum += (m_kt(k,t)/m_sum) * exp(lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
												- lgamma(alpha(k,t)) + alpha(k,t) * log(beta(k,t)) + 0.5 * D * log(kappa(k,t)/(2 * M_PI)));
				}
			}
			if(sum < DBL_MIN)
				sum = DBL_MIN;
			ll += log(sum);
		}
		ll /= n_te;

		// post loglik
		ll_post = 0.0;
		for(j=0; j<2000; j++){
			sum = 0.0;
			for(k=0; k<K0; k++){
				for(t=0; t<T; t++){
					kappai = kappa(k,t) + 1.0;
					mui = (kappa(k,t)*mu[k].row(t)+x_post.row(j))/kappai;
					alphai = alpha(k,t) + D * 0.5;
					betai = beta(k,t) + (mu[k].row(t)-x_post.row(j)).dot(mu[k].row(t)-x_post.row(j)) * 0.5 * kappa(k,t) / kappai;

					sum += (m_kt(k,t)/m_sum) * exp(lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
												- lgamma(alpha(k,t)) + alpha(k,t) * log(beta(k,t)) + 0.5 * D * log(kappa(k,t)/(2 * M_PI)));
				}
			}
			if(sum < DBL_MIN)
				sum = DBL_MIN;
			ll_post += log(sum);
		}
		ll_post /= 2000;
		finish = clock();

		out1 << K0 << endl;
		out2 << ll << endl;
		out3 << aa << endl; 
		out6 << ll_post << endl;
		cout << batch+1 << " : " << K0 << " " << ll << " " << ll_post << " " << aa << " " << m_sum << " " << (double)(finish - start) / CLOCKS_PER_SEC << endl;
		out7 << batch+1 << " : " << K0 << " " << ll << " " << ll_post << " " << aa << " " << m_sum << " " << (double)(finish - start) / CLOCKS_PER_SEC << endl;
	}
	
	// Allocation of test observations to calculate misclassification errors
	for(j=0; j<n_te; j++){
		p_max = -DBL_MAX;
		for(k=0; k<K0; k++){
			p[k] = 0.0;
			for(t=0; t<T; t++){
				kappai = kappa(k,t) + 1.0;
				mui = (kappa(k,t)*mu[k].row(t)+x_te.row(j))/kappai;
				alphai = alpha(k,t) + D * 0.5;
				betai = beta(k,t) + (mu[k].row(t)-x_te.row(j)).dot(mu[k].row(t)-x_te.row(j)) * 0.5 * kappa(k,t) / kappai;

				p[k] += (m_kt(k,t)/m_sum) * exp(lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
											- lgamma(alpha(k,t)) + alpha(k,t) * log(beta(k,t)) + 0.5 * D * log(kappa(k,t)/(2 * M_PI)));
			}
			p[k] = log(m(k)) + log(p[k]);

			if(p[k] > p_max){
				p_max = p[k];
				idx = k;
			}
		}
		out4 << idx << endl;
	}

	for(k=0; k<K; k++){
		out8 << m_kt.row(k) << endl;
		out9 << kappa.row(k) << " " << alpha.row(k) << " " << beta.row(k) << endl;
		for(t=0; t<T; t++)
			out5 << mu[k].row(t) << endl;
	}

	in2.close();
	out1.close();
	out2.close();
	out3.close();
	out4.close();
	out5.close();
	out6.close();
	out7.close();
	out8.close();
	out9.close();
	free(p);
	free(q);
	for(k=0; k<K_MAX; k++)	free(p_t[k]);
	free(p_t);

	return 0;
}
