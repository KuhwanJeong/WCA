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
	// T : # of mini-batches
	// Q : # of MCMC samples
	int n, n_te, D, K, K0, K2, T, Q=(ITER-BURNIN)/THIN, Q2, cnt, cnt2, cnt3;
	double kappa00=0.1, kappai, alpha00=1.0, alphai, beta00=BB0, betai, ll, ll_prev, ll_best, ll_post, sum, sum2, p_max, m_sum, m_sum_p, m_del, mm, a;
	int i, j, d, k, z_p, iter, iter2, iter3, iter4, n_sum, restart, K_max;
	double s, ds, ds2, y, digam = -digamma(1.0), aa=AA;
	clock_t start, finish;
	cout << aa << " " << BB0 << " " << BURNIN << " " << THIN << " " << PP << " " << EPS2 << endl;

	string loc;
	loc = dir + "info.txt";
	ifstream in1(loc);
	in1 >> n >> n_te >> D;
	in1.close();
	T = n/B;

	// x : training observations; x_te : test observations; x_post : observations sampled from the posterior predictive distribution
	MatrixXd x(B,D), x_te(n_te,D), x_post(2000,D), mu(K_MAX,D), mu0(K_MAX,D), mu_e(K_MAX,D), x_sum(K_MAX,D), phi2(Q,D), lambda1(Q,K_MAX);
	VectorXd m(K_MAX), mui(D), kappa(K_MAX), kappa0(K_MAX), kappa_e(K_MAX), alpha(K_MAX), alpha0(K_MAX), alpha_e(K_MAX), 
		beta(K_MAX), beta0(K_MAX), beta_e(K_MAX), pi_e(K_MAX), x_bar(D), x_sum2(K_MAX), lambda2(Q), p1(K_MAX), p2(K_MAX);
	VectorXd Ak(K_MAX), Bk(K_MAX), Dk(K_MAX), L(K_MAX);
	MatrixXd Ck(K_MAX,D), gamma(Q,K_MAX);
	VectorXi n_k(K_MAX), z(B), z2(Q), p_k(K_MAX);
	MatrixXd *phi1 = new MatrixXd[Q];
	for(i=0; i<Q; i++)	phi1[i] = MatrixXd(K_MAX,D);

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
	
	dir = dir + "result/WCA/" + dir2 + "/" + dir3 + "/";
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

	mu_e.row(0).setZero(); kappa_e(0) = kappa00; alpha_e(0) = alpha00; beta_e(0) = beta00;

	K0 = 0;
	m_sum = 0.0;
	for(int t=0; t<T; t++){
		start = clock();

		K = K0;
		for(i=0; i<B; i++){
			for(d=0; d<D; d++)		
				in2 >> x(i,d);
			
			// Initialize z_i using Eq (16)
			for(k=0; k<K; k++){
				kappai = kappa(k) + 1.0;
				mui = (kappa(k)*mu.row(k)+x.row(i))/kappai;
				alphai = alpha(k) + D * 0.5;
				betai = beta(k) + (mu.row(k)-x.row(i)).dot(mu.row(k)-x.row(i)) * 0.5 * kappa(k) / kappai;

				p[k] = lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
						- lgamma(alpha(k)) + alpha(k) * log(beta(k)) + 0.5 * D * log(kappa(k));
			}

			kappai = kappa00 + 1.0;
			mui = x.row(i)/kappai;
			alphai = alpha00 + D * 0.5;
			betai = beta00 + x.row(i).dot(x.row(i)) * 0.5 * kappa00 / kappai;

			p[K] = lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
					- lgamma(alpha00) + alpha00 * log(beta00) + 0.5 * D * log(kappa00);

			p_max = max(p,K+1);
			for(k=0; k<=K; k++)	p[k] = exp(p[k]-p_max);
				
			// Sample z_i
			z(i) = sample(p, K+1, r);
			if(z(i) == K){
				// new component
				kappa(K) = kappai;
				mu.row(K) = mui;
				alpha(K) = alphai;
				beta(K) = betai;

				kappa0(K) = kappa00;
				mu0.row(K).setZero();
				alpha0(K) = alpha00;
				beta0(K) = beta00;
				
				x_sum.row(K) = x.row(i);
				x_sum2(K) = x.row(i).dot(x.row(i));
				n_k(K) = 1;
				m(K) = 0.0;

				K++;
			}
			else{
				// existing components
				kappai = kappa(z(i)) + 1.0;
				mui = (kappa(z(i))*mu.row(z(i))+x.row(i))/kappai;
				alphai = alpha(z(i)) + D * 0.5;
				betai = beta(z(i)) + (mu.row(z(i))-x.row(i)).dot(mu.row(z(i))-x.row(i)) * 0.5 * kappa(z(i)) / kappai;

				kappa(z(i)) = kappai;
				mu.row(z(i)) = mui;
				alpha(z(i)) = alphai;
				beta(z(i)) = betai;

				x_sum.row(z(i)) += x.row(i);
				x_sum2(z(i)) += x.row(i).dot(x.row(i));
				n_k(z(i))++;
			}
		}

		p_k.setZero();
		p1.setZero();
		p2.setZero();
		K_max = K0;

		// MCMC
		for(iter=0; iter<ITER; iter++){
			for(i=0; i<B; i++){
				z_p = z(i);
				x_sum.row(z_p) -= x.row(i);
				x_sum2(z_p) -= x.row(i).dot(x.row(i));
				n_k(z_p)--;

				if(n_k(z_p) == 0 && z_p >= K0){
					// Remove the unoccupied component
					K--;
					if(z_p != K){
						x_sum.row(z_p) = x_sum.row(K);
						x_sum2(z_p) = x_sum2(K);
						n_k(z_p) = n_k(K);

						for(j=0; j<B; j++)
							if(z(j) == K) z(j) = z_p;

						kappa(z_p) = kappa(K);
						mu.row(z_p) = mu.row(K);
						alpha(z_p) = alpha(K);
						beta(z_p) = beta(K);
					}
				}
				else if(n_k(z_p) != 0){
					kappa(z_p) = kappa0(z_p) + (double)n_k(z_p);
					mu.row(z_p) = (kappa0(z_p)*mu0.row(z_p)+x_sum.row(z_p))/kappa(z_p);
					alpha(z_p) = alpha0(z_p) + n_k(z_p) * D * 0.5;
					x_bar = x_sum.row(z_p) / n_k(z_p);
					beta(z_p) = beta0(z_p) + (mu0.row(z_p).transpose()-x_bar).dot(mu0.row(z_p).transpose()-x_bar) * 0.5 * kappa0(z_p) * n_k(z_p) / kappa(z_p) 
								+ (x_sum2(z_p) - n_k(z_p) * x_bar.dot(x_bar)) * 0.5;
				}
				else{
					kappa(z_p) = kappa0(z_p);
					mu.row(z_p) = mu0.row(z_p);
					alpha(z_p) = alpha0(z_p);
					beta(z_p) = beta0(z_p);
				}

				// Calculate Eq (16)
				for(k=0; k<K; k++){
					kappai = kappa(k) + 1.0;
					mui = (kappa(k)*mu.row(k)+x.row(i))/kappai;
					alphai = alpha(k) + D * 0.5;
					betai = beta(k) + (mu.row(k)-x.row(i)).dot(mu.row(k)-x.row(i)) * 0.5 * kappa(k) / kappai;

					p[k] = log((double)n_k(k)+m(k)) + lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
							- lgamma(alpha(k)) + alpha(k) * log(beta(k)) + 0.5 * D * log(kappa(k));
				}

				kappai = kappa00 + 1.0;
				mui = x.row(i)/kappai;
				alphai = alpha00 + D * 0.5;
				betai = beta00 + x.row(i).dot(x.row(i)) * 0.5 * kappa00 / kappai;

				p[K] = log(aa) + lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
						- lgamma(alpha00) + alpha00 * log(beta00) + 0.5 * D * log(kappa00);

				p_max = max(p,K+1);
				for(k=0; k<=K; k++)	p[k] = exp(p[k]-p_max);
				
				// Sample z_i
				z(i) = sample(p, K+1, r);
				if(z(i) == K){
					// new component
					kappa(K) = kappai;
					mu.row(K) = mui;
					alpha(K) = alphai;
					beta(K) = betai;

					kappa0(K) = kappa00;
					mu0.row(K).setZero();
					alpha0(K) = alpha00;
					beta0(K) = beta00;
				
					x_sum.row(K) = x.row(i);
					x_sum2(K) = x.row(i).dot(x.row(i));
					n_k(K) = 1;
					m(K) = 0.0;

					K++;
				}
				else{
					// existing components
					kappai = kappa(z(i)) + 1.0;
					mui = (kappa(z(i))*mu.row(z(i))+x.row(i))/kappai;
					alphai = alpha(z(i)) + D * 0.5;
					betai = beta(z(i)) + (mu.row(z(i))-x.row(i)).dot(mu.row(z(i))-x.row(i)) * 0.5 * kappa(z(i)) / kappai;

					kappa(z(i)) = kappai;
					mu.row(z(i)) = mui;
					alpha(z(i)) = alphai;
					beta(z(i)) = betai;

					x_sum.row(z(i)) += x.row(i);
					x_sum2(z(i)) += x.row(i).dot(x.row(i));
					n_k(z(i))++;
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
					lambda1(j,k) = rgamma(alpha(k), beta(k), r);
					for(d=0; d<D; d++)	phi1[j](k,d) = mu(k,d) + rnorm(r) / sqrt(kappa(k) * lambda1(j,k));

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
					lambda2(j) = rgamma(alpha(k), beta(k), r);
					for(d=0; d<D; d++)	phi2(j,d) = mu(k,d) + rnorm(r) / sqrt(kappa(k) * lambda2(j));
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
			Ak(k) = Bk(k) = 0.0;
			Ck.row(k).setZero();

			for(j=0; j<Q; j++){
				Ak(k) += log(lambda1(j,k));
				Bk(k) += lambda1(j,k);
				Ck.row(k) += lambda1(j,k) * phi1[j].row(k);
			}
			Ak(k) /= Q;
			Bk(k) /= Q;
			Ck.row(k) /= Q;

			a = 0.5/(log(Bk(k))-Ak(k));
			for(iter2=0; iter2<ITER2; iter2++)
				a = 1/(1/a + (digamma(a)-log(a)+log(Bk(k))-Ak(k))/(a*a*(trigamma(a)-1/a)));

			alpha(k) = alpha0(k) = a;
			beta(k) = beta0(k) = a / Bk(k);
			mu.row(k) = mu0.row(k) = Ck.row(k) / Bk(k);

			sum = 0.0;
			for(j=0; j<Q; j++)
				sum += lambda1(j,k) * (mu.row(k)-phi1[j].row(k)).dot(mu.row(k)-phi1[j].row(k));
			kappa(k) = kappa0(k) = D * Q / sum;
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
		if(t == 0){
			K2 = K;
			for(k=0; k<K; k++){
				m(k) = n_k(k);
				alpha0(k) = alpha(k);
				beta0(k) = beta(k);
				mu0.row(k) = mu.row(k);
				kappa0(k) = kappa(k);
			}
		}
		else if(K2 > K0){
			if(K2 > K){
				for(k=K; k<K2; k++){
					mu.row(k).setZero();
					alpha(k) = alpha00;
					beta(k) = beta00;
					kappa(k) = kappa00;
				}
			}

			cnt = cnt2 = cnt3 = 0;
			ll_best = -DBL_MAX;
			runif_int=uniform_int_distribution<int>(0,K2-K0);
			for(iter3=0; iter3<ITER1; iter3++){
				Ak(0) = Bk(0) = Dk(0) = 0.0;
				Ck.row(0).setZero();
				for(k=1; k<=K2-K0; k++){
					mu_e.row(k) = mu.row(K0+k-1);
					alpha_e(k) = alpha(K0+k-1);
					beta_e(k) = beta(K0+k-1);
					kappa_e(k) = kappa(K0+k-1);

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
						cout << "restart 1 : initial Dk is equal to zero, " << k << " : " << Dk(k) << endl;
						out7 << "restart 1 : initial Dk is equal to zero, " << k << " : " << Dk(k) << endl;
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
							cout << "restart 2 : Dk is too small, " << k << " : " <<  Dk(k) << endl;
							out7 << "restart 2 : Dk is too small, " << k << " : " <<  Dk(k) << endl;
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
							alpha(K0+k-1) = alpha0(K0+k-1) = alpha_e(k);
							beta(K0+k-1) = beta0(K0+k-1) = beta_e(k);
							mu.row(K0+k-1) = mu0.row(K0+k-1) = mu_e.row(k);
							kappa(K0+k-1) = kappa0(K0+k-1) = kappa_e(k);
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
					m(k) = m(K2);
					alpha(k) = alpha0(k) = alpha0(K2);
					beta(k) = beta0(k) = beta0(K2);
					mu.row(k) = mu0.row(k) = mu0.row(K2);
					kappa(k) = kappa0(k) = kappa0(K2);
				}
			}
		}
		m_sum -= m_del;

		if(m_del > 0)
			cout << "m_del : " << m_del << endl;

		cout << m_sum_p << " " << s << " " << m_sum << endl;

		// initialization
		K0 = K2;
		for(k=0; k<K0; k++){
			x_sum.row(k).setZero();
			x_sum2(k) = 0.0;
			n_k(k) = 0;
		}

		// test loglik
		ll = 0.0;
		for(j=0; j<n_te; j++){
			sum = 0.0;
			for(k=0; k<K0; k++){
				kappai = kappa(k) + 1.0;
				mui = (kappa(k)*mu.row(k)+x_te.row(j))/kappai;
				alphai = alpha(k) + D * 0.5;
				betai = beta(k) + (mu.row(k)-x_te.row(j)).dot(mu.row(k)-x_te.row(j)) * 0.5 * kappa(k) / kappai;
				
				sum += (m(k)/m_sum) * exp(lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
						- lgamma(alpha(k)) + alpha(k) * log(beta(k)) + 0.5 * D * log(kappa(k)/(2 * M_PI)));
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
				kappai = kappa(k) + 1.0;
				mui = (kappa(k)*mu.row(k)+x_post.row(j))/kappai;
				alphai = alpha(k) + D * 0.5;
				betai = beta(k) + (mu.row(k)-x_post.row(j)).dot(mu.row(k)-x_post.row(j)) * 0.5 * kappa(k) / kappai;
				
				sum += (m(k)/m_sum) * exp(lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
						- lgamma(alpha(k)) + alpha(k) * log(beta(k)) + 0.5 * D * log(kappa(k)/(2 * M_PI)));
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
		cout << t+1 << " : " << K0 << " " << ll << " " << ll_post << " " << aa << " " << m_sum << " " << (double)(finish - start) / CLOCKS_PER_SEC << endl;
		out7 << t+1 << " : " << K0 << " " << ll << " " << ll_post << " " << aa << " " << m_sum << " " << (double)(finish - start) / CLOCKS_PER_SEC << endl;
	}

	// Allocation of test observations to calculate misclassification errors
	for(j=0; j<n_te; j++){
		sum = -DBL_MAX;
		for(k=0; k<K0; k++){
			kappai = kappa(k) + 1.0;
			mui = (kappa(k)*mu.row(k)+x_te.row(j))/kappai;
			alphai = alpha(k) + D * 0.5;
			betai = beta(k) + (mu.row(k)-x_te.row(j)).dot(mu.row(k)-x_te.row(j)) * 0.5 * kappa(k) / kappai;
				
			sum2 = log(m(k)) + lgamma(alphai) - alphai * log(betai) - 0.5 * D * log(kappai)
					- lgamma(alpha(k)) + alpha(k) * log(beta(k)) + 0.5 * D * log(kappa(k)/(2 * M_PI));
			
			if(sum < sum2){
				p_max = k;
				sum = sum2;
			}
		}
		out4 << p_max << endl;
	}

	for(k=0; k<K0; k++){
		out5 << mu.row(k) << endl;
		out8 << m(k) << endl;
		out9 << kappa(k) << " " << alpha(k) << " " << beta(k) << endl;
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

	return 0;
}
