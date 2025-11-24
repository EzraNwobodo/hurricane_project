data {
  int<lower=1> N_years;         // number of seasons/years
  int<lower=0> N_obs[N_years];  // total low-intensity cyclone counts per year (N1,i)
  int<lower=0> L_obs[N_years];  // number of damaging/landfalling cyclones per year (L1,i)
  vector[N_years] D_obs;        // observed annual damage (0 for zero years, positive otherwise)
  int<lower=1> P;               // number of covariates (including intercept)
  matrix[N_years, P] X;         // covariate matrix (already scaled/with intercept)
}
parameters {
  vector[P] beta;               // regression for log(lambda)
  real<lower=0> r;              // NB dispersion (phi) > 0
  real<lower=0, upper=1> theta; // probability a cyclone is damaging/lands
  real mu;                      // log-mean of positive seasonal damages
  real<lower=0> sigma;          // log-sd of positive seasonal damages
}
transformed parameters {
  vector[N_years] lambda;
  for (i in 1:N_years)
    lambda[i] = exp(X[i] * beta);
}
model {
  // Priors (matching paper)
  beta ~ normal(0, 1e5);
  theta ~ beta(1, 1);
  mu ~ normal(0, 1e5);
  // prior for sigma: paper uses 1/sigma^2 ~ Gamma(1,1).
  // Implement by placing prior on tau = 1/sigma^2 implicitly:
  {
    real tau = 1.0 / (sigma * sigma);
    tau ~ gamma(1, 1);
  }
  r ~ uniform(0, 70);

  // Likelihood:
  for (i in 1:N_years) {
    // Negative binomial for total counts
    target += neg_binomial_2_lpmf(N_obs[i] | lambda[i], r);

    // Binomial for L given N
    if (N_obs[i] > 0)
      L_obs[i] ~ binomial(N_obs[i], theta);
    else
      // if N_obs is zero, L_obs must be zero (dataset should reflect that)
      L_obs[i] ~ binomial(0, theta); // harmless: forces L_obs[i]==0

    // Seasonal damage mixture:
    if (D_obs[i] <= 0) {
      // zero damage: probability = (1 - theta)^{N_obs[i]}
      // log-prob:
      target += N_obs[i] * bernoulli_lpmf(0 | theta); 
      // but bernoulli_lpmf with 0|theta gives log(1-theta), so multiply by N_obs
      // This is equivalent to log((1-theta)^{N_obs})
      // Note: if N_obs[i]==0, (1-theta)^0 = 1 -> log prob 0 (ok)
    } else {
      // positive damage: probability of at least one damaging cyclone * density of lognormal
      // probability of being >0 is 1 - (1-theta)^{N_obs}
      real log_prob_pos;
      real p_pos = 1 - pow(1 - theta, N_obs[i]);
      if (p_pos <= 0)
        target += negative_infinity(); // impossible: observed positive but model says zero prob
      else {
        // mixture component: log(p_pos) + log density of lognormal at D_obs[i]
        log_prob_pos = log(p_pos) + lognormal_lpdf(D_obs[i] | mu, sigma);
        target += log_prob_pos;
      }
    }
  }
}
generated quantities {
  vector[N_years] log_lik; // per-year log-likelihood for loo/waic if desired
  for (i in 1:N_years) {
    // Compute same log-likelihood contributions
    real lp = 0;
    lp += neg_binomial_2_lpmf(N_obs[i] | lambda[i], r);
    if (N_obs[i] > 0)
      lp += binomial_lpmf(L_obs[i] | N_obs[i], theta);
    if (D_obs[i] <= 0)
      lp += N_obs[i] * bernoulli_lpmf(0 | theta);
    else {
      real p_pos = 1 - pow(1 - theta, N_obs[i]);
      lp += log(p_pos) + lognormal_lpdf(D_obs[i] | mu, sigma);
    }
    log_lik[i] = lp;
  }
}
