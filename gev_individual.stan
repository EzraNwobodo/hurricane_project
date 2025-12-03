functions {
  // GEV log density for one observation
  real gev1_lpdf(real x, real mu, real sigma, real xi) {
    real z;
    if (abs(xi) < 1e-12) {
      // Gumbel
      real t = (x - mu) / sigma;
      return -log(sigma) - t - exp(-t);
    } 
    else {
      z = 1 + xi * ( (x - mu) / sigma );
      if (z <= 0) {
        return negative_infinity();
      }
      return -log(sigma) + (-1 - 1.0/xi) * log(z) - pow(z, -1.0/xi);
    }
  }
}

data {
  int<lower=1> N;             // number of cyclones / observations
  int<lower=1> p;             // number of covariates (including intercept) for X
  matrix[N, p] X;             // covariate matrix (intercept + 10 covariates)
  vector[N] Z1;               // log(minCP)
  vector[N] Y1;               // log(maxWS)
  vector[N] Y2;               // log(damages)
}
parameters {
  vector[p] alpha;            // for mu_z1 = X * alpha
  vector[p+1] beta;           // for mu_y1 = [Z1, X] * beta  (first entry for Z1)
  vector[p+2] gamma;          // for mu_y2 = [Z1, Y1, X] * gamma (first two: Z1, Y1)

  // inverse-gamma prior via inv_sigma2 ~ gamma(1,3)
  real<lower=0> inv_sigma2_z1;
  real<lower=0> inv_sigma2_y1;
  real<lower=0> inv_sigma2_y2;

  // shape parameters with bounds
  real<lower=-1, upper=1> xi_z1;
  real<lower=-0.55, upper=0.5> xi_y1;
  real<lower=-0.55, upper=0.5> xi_y2;
}
transformed parameters {
  real<lower=0> sigma_z1 = sqrt(1.0 / inv_sigma2_z1);
  real<lower=0> sigma_y1 = sqrt(1.0 / inv_sigma2_y1);
  real<lower=0> sigma_y2 = sqrt(1.0 / inv_sigma2_y2);
}

model {
  // Priors
  alpha ~ normal(0, 1e2);       // N(0, 10^2)
  beta  ~ normal(0, 1e3);       // N(0, 10^3)
  gamma ~ normal(0, 1e2);       // N(0, 10^2)

  inv_sigma2_z1 ~ gamma(1, 3);
  inv_sigma2_y1 ~ gamma(1, 3);
  inv_sigma2_y2 ~ gamma(1, 3);

  // xi priors: uniform via constrained parameters (implicit)

  // Likelihood: product over i of GEV(Z1_i | X*alpha) * GEV(Y1_i | [Z1,X]*beta) * GEV(Y2_i | [Z1,Y1,X]*gamma)
  for (i in 1:N) {
    real mu_z1 = dot_product(row(X, i), alpha);
    real mu_y1 = beta[1] * Z1[i] + dot_product(row(X, i), beta[2:p+1]); // beta[1] is coef for Z1
    real mu_y2 = gamma[1] * Z1[i] + gamma[2] * Y1[i] + dot_product(row(X, i), gamma[3:p+2]);

    target += gev1_lpdf(Z1[i] | mu_z1, sigma_z1, xi_z1);
    target += gev1_lpdf(Y1[i] | mu_y1, sigma_y1, xi_y1);
    target += gev1_lpdf(Y2[i] | mu_y2, sigma_y2, xi_y2);
  }
}
generated quantities {
  // for convenience: expose sigma values and full coefficient vectors
  real sigma_z1_out = sigma_z1;
  real sigma_y1_out = sigma_y1;
  real sigma_y2_out = sigma_y2;
}

