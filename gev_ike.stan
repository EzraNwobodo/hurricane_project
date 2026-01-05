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

  // log(IKE) observed for some storms (e.g. 2004+)
  int<lower=0> N_obs_z2;           // number of storms with observed log(IKE)
  int<lower=0> N_mis_z2;           // number of storms with missing log(IKE)
  int idx_obs_z2[N_obs_z2];        // indices (1..N) where log(IKE) is observed
  int idx_mis_z2[N_mis_z2];        // indices (1..N) where log(IKE) is missing
  vector[N_obs_z2] Z2_obs;         // observed log(IKE), centred
}

parameters {
  // regression coefficients
  vector[p]   alpha;       // for mu_z1 = X * alpha
  vector[p+1] beta;        // for mu_y1 = [Z1, X] * beta  (beta[1] is Z1)

  // NOTE: gamma now has length p+3: [Z1, Y1, Z2, X]
  vector[p+3] gamma;       // for mu_y2 = [Z1, Y1, Z2, X] * gamma

  // log scales for GEV components
  real log_sigma_z1;
  real log_sigma_y1;
  real log_sigma_y2;

  // unconstrained shape parameters for GEV
  real xi_raw_z1;
  real xi_raw_y1;
  real xi_raw_y2;

  // --- lognormal model for Z2 = log(IKE) ---
  // mu_z2 = delta[1]*Z1 + delta[2]*Y1 + X * delta[3:(p+2)]
  vector[p+2] delta;
  real log_sigma_z2;               // std dev of log(IKE)

  // latent log(IKE) for storms without observations (pre-2004)
  vector[N_mis_z2] Z2_mis;
}

transformed parameters {
  // GEV scales
  real<lower=0> sigma_z1 = exp(log_sigma_z1);
  real<lower=0> sigma_y1 = exp(log_sigma_y1);
  real<lower=0> sigma_y2 = exp(log_sigma_y2);

  // map R -> (-b,b) for GEV shape
  real xi_z1 = 0.7 * tanh(xi_raw_z1);   // approx (-0.7, 0.7)
  real xi_y1 = 0.5 * tanh(xi_raw_y1);   // approx (-0.5, 0.5)
  real xi_y2 = 0.5 * tanh(xi_raw_y2);   // approx (-0.5, 0.5)

  // lognormal (Normal on log IKE)
  real<lower=0> sigma_z2 = exp(log_sigma_z2);

  // full log(IKE) vector for all storms
  vector[N] Z2;
  {
    // fill observed values
    for (n in 1:N_obs_z2)
      Z2[idx_obs_z2[n]] = Z2_obs[n];

    // fill latent values for missing storms
    for (m in 1:N_mis_z2)
      Z2[idx_mis_z2[m]] = Z2_mis[m];
  }
}

model {
  // --- PRIORS ---

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

  // Z2 (log IKE) regression and scale
  delta ~ normal(0, 0.2);              // slightly tighter prior
  log_sigma_z2 ~ normal(log(1), 0.3);  // log(IKE) sd ~ 1 on prior

  // prior on latent log(IKE) to keep them reasonable
  Z2_mis ~ normal(0, 1);

  // --- LIKELIHOOD ---

  // 1) Z1 = log(minCP) GEV
  for (i in 1:N) {
    real mu_z1 = dot_product(row(X, i), alpha);
    target += gev1_lpdf(Z1[i] | mu_z1, sigma_z1, xi_z1);
  }

  // 2) Y1 = log(maxWS) GEV
  for (i in 1:N) {
    real mu_y1 = beta[1] * Z1[i]
                 + dot_product(row(X, i), beta[2:p+1]);
    target += gev1_lpdf(Y1[i] | mu_y1, sigma_y1, xi_y1);
  }

  // 3) Z2 = log(IKE) ~ Normal(mu_z2, sigma_z2)
  // observed part
  for (n in 1:N_obs_z2) {
    int i = idx_obs_z2[n];
    real mu_z2 = delta[1] * Z1[i]
               + delta[2] * Y1[i]
               + dot_product(row(X, i), delta[3:p+2]);
    Z2_obs[n] ~ normal(mu_z2, sigma_z2);
  }
  // missing / latent part
  for (m in 1:N_mis_z2) {
    int i = idx_mis_z2[m];
    real mu_z2 = delta[1] * Z1[i]
               + delta[2] * Y1[i]
               + dot_product(row(X, i), delta[3:p+2]);
    Z2_mis[m] ~ normal(mu_z2, sigma_z2);
  }

  // 4) Y2 = log(damages), GEV, now including Z2
  for (i in 1:N) {
    real mu_y2 = gamma[1] * Z1[i]
               + gamma[2] * Y1[i]
               + gamma[3] * Z2[i]            // NEW: effect of log(IKE)
               + dot_product(row(X, i), gamma[4:p+3]);

    target += gev1_lpdf(Y2[i] | mu_y2, sigma_y2, xi_y2);
  }
}

generated quantities {
  real sigma_z1_out = sigma_z1;
  real sigma_z2_out = sigma_z2;   // log(IKE) sd
  real sigma_y1_out = sigma_y1;
  real sigma_y2_out = sigma_y2;

  // NEW: pointwise log-likelihood for damages (Y2)
  vector[N] log_lik;
  for (i in 1:N) {
    real mu_y2 = gamma[1] * Z1[i]
               + gamma[2] * Y1[i]
               + gamma[3] * Z2[i]
               + dot_product(row(X, i), gamma[4:p+3]);

    log_lik[i] = gev1_lpdf(Y2[i] | mu_y2, sigma_y2, xi_y2);
  }
}

