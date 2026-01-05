functions {
  // GEV log density for one observation
  real gev1_lpdf(real x, real mu, real sigma, real xi) {
    real z;
    if (fabs(xi) < 1e-12) {
      // Gumbel limit
      real t = (x - mu) / sigma;
      return -log(sigma) - t - exp(-t);
    } else {
      z = 1 + xi * ((x - mu) / sigma);
      if (z <= 0) return negative_infinity();
      return -log(sigma) + (-1 - inv(xi)) * log(z) - pow(z, -inv(xi));
    }
  }
}

data {
  int<lower=1> N;
  int<lower=1> p;          // includes intercept; X is centred & scaled except intercept
  matrix[N, p] X;
  vector[N] Z1;            // log(minCP), centred (not scaled)
  vector[N] Y1;            // log(maxWS), centred
  vector[N] Y2;            // log(damages), centred
}

parameters {
  // regression coefficients
  vector[p]   alpha;       // for mu_z1 = X * alpha
  vector[p+1] beta;        // for mu_y1 = [Z1, X] * beta  (beta[1] is Z1)
  vector[p+2] gamma;       // for mu_y2 = [Z1, Y1, X] * gamma (gamma[1]: Z1, gamma[2]: Y1)

  // log scales (instead of inverse-gamma)
  real log_sigma_z1;
  real log_sigma_y1;
  real log_sigma_y2;

  // unconstrained shape parameters
  real xi_raw_z1;
  real xi_raw_y1;
  real xi_raw_y2;
}

transformed parameters {
  real<lower=0> sigma_z1 = exp(log_sigma_z1);
  real<lower=0> sigma_y1 = exp(log_sigma_y1);
  real<lower=0> sigma_y2 = exp(log_sigma_y2);

  // map R -> (-b,b) for some b < 1; you can tune b if needed
  real xi_z1 = 0.7 * tanh(xi_raw_z1);   // approx (-0.7, 0.7)
  real xi_y1 = 0.5 * tanh(xi_raw_y1);   // approx (-0.5, 0.5)
  real xi_y2 = 0.5 * tanh(xi_raw_y2);   // approx (-0.5, 0.5)
}

model {
  // --- PRIORS REVISED TO GUARANTEE VALID INITIALIZATION ---

  // regression coefficients: keep μ near 0 at initialization
  alpha ~ normal(0, 0.3);
  beta  ~ normal(0, 0.3);
  gamma ~ normal(0, 0.3);

  // sigma priors: avoid tiny scales during initialization
  log_sigma_z1 ~ normal(log(3), 0.2);  // sigma ≈ 3
  log_sigma_y1 ~ normal(log(3), 0.2);
  log_sigma_y2 ~ normal(log(3), 0.2);


  // shape parameters: start extremely close to 0 to avoid support boundary
  xi_raw_z1 ~ normal(0, 0.1);
  xi_raw_y1 ~ normal(0, 0.1);
  xi_raw_y2 ~ normal(0, 0.1);


  // --- LIKELIHOOD ---
  for (i in 1:N) {
    real mu_z1 = dot_product(row(X, i), alpha);

    real mu_y1 = beta[1] * Z1[i]
                 + dot_product(row(X, i), beta[2:p+1]);

    real mu_y2 = gamma[1] * Z1[i]
                 + gamma[2] * Y1[i]
                 + dot_product(row(X, i), gamma[3:p+2]);

    target += gev1_lpdf(Z1[i] | mu_z1, sigma_z1, xi_z1);
    target += gev1_lpdf(Y1[i] | mu_y1, sigma_y1, xi_y1);
    target += gev1_lpdf(Y2[i] | mu_y2, sigma_y2, xi_y2);
  }
}

generated quantities {
  real sigma_z1_out = sigma_z1;
  real sigma_y1_out = sigma_y1;
  real sigma_y2_out = sigma_y2;

  // NEW: pointwise log-likelihood for damages (Y2) only
  vector[N] log_lik;
  for (i in 1:N) {
    real mu_z1 = dot_product(row(X, i), alpha);
    real mu_y1 = beta[1] * Z1[i]
                 + dot_product(row(X, i), beta[2:p+1]);
    real mu_y2 = gamma[1] * Z1[i]
                 + gamma[2] * Y1[i]
                 + dot_product(row(X, i), gamma[3:p+2]);

    // Only store the log-lik for Y2, since we're comparing DAMAGE predictions
    log_lik[i] = gev1_lpdf(Y2[i] | mu_y2, sigma_y2, xi_y2);
  }
}
