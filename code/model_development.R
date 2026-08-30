
library(caret)
library(mvtnorm)


####### Functions for kernel calculation

# Function to calculate the angle needed for kernel calculations of each layer
# Utilizes the kernel functions between the inputs and for each input
theta_arc_cos <- function(nngp_kernel, nngp_kernel_x1, nngp_kernel_x2) {
  
  angle = acos(nngp_kernel / sqrt(nngp_kernel_x1 * nngp_kernel_x2))
  
  return(angle)
  
}

# Calculates the initial kernel function for a nn gp using two vectors and
# weight and bias variances
nngp_kernel_init <- function(x1, x2, sigma2_w, sigma2_b) {
  
  d_in = length(x1)
  
  k_init = sigma2_b + sigma2_w * (as.numeric(crossprod(x1, x2)) / d_in)
  
  return(k_init)
}

# Calculates the kernel function corresponding to the l-th layer using
# the kernel functions of the previous layer between the inputs and for each input
# and the weight and bias variances
nngp_kernel_l <- function(nngp_kernel_prev, nngp_kernel_prev_x1, nngp_kernel_prev_x2,
                          sigma2_w, sigma2_b) {
  
  theta_prev = theta_arc_cos(nngp_kernel = nngp_kernel_prev, nngp_kernel_x1 = nngp_kernel_prev_x1,
                             nngp_kernel_x2 = nngp_kernel_prev_x2)
  
  kernel_l = (sigma2_b + (sigma2_w / (2 * pi)) * sqrt(nngp_kernel_prev_x1 * nngp_kernel_prev_x2)
              * (sin(theta_prev) + (pi - theta_prev) * cos(theta_prev)))
  
  return(kernel_l)
  
  
}

# Calculates the kernel function between two vectors for the last layer L of the DNN
# Also requires the weight and bias variances
nngp_kernel_layer_L <- function(L, x1, x2, sigma2_w, sigma2_b) {
  
  # Store the kernel functions between the inputs in a vector
  # of size L + 1 reflecting the initial kernel and those for the L layers
  nngp_kernel_list = rep(NA, L + 1)
  
  # Store the kernel functions for each input in vectors of size L 
  # to obtain the initial kernel function and those for the first L - 1 layers
  # which are the only ones necessary for calculating the kernel between vectors
  nngp_kernel_x1_list = rep(NA, L)
  nngp_kernel_x2_list = rep(NA, L)
  
  # Set the initial kernel functions
  nngp_kernel_list[1] = nngp_kernel_init(x1 = x1, x2 = x2, sigma2_w = sigma2_w, sigma2_b = sigma2_b)
  nngp_kernel_x1_list[1] = nngp_kernel_init(x1 = x1, x2 = x1, sigma2_w = sigma2_w, sigma2_b = sigma2_b)
  nngp_kernel_x2_list[1] = nngp_kernel_init(x1 = x2, x2 = x2, sigma2_w = sigma2_w, sigma2_b = sigma2_b)
  
  # For l = 1,...,L calculate the l-th kernel function using the previous one
  for (l in 2:(L + 1)) {
    
    nngp_kernel_list[l] = nngp_kernel_l(nngp_kernel_prev = nngp_kernel_list[l - 1],
                                        nngp_kernel_prev_x1 = nngp_kernel_x1_list[l - 1],
                                        nngp_kernel_prev_x2 = nngp_kernel_x2_list[l - 1],
                                        sigma2_w = sigma2_w, sigma2_b = sigma2_b)
    
    # Kernel functions for each input vector with itself
    if (l < L + 1) {
      nngp_kernel_x1_list[l] = sigma2_b + (sigma2_w / 2) * nngp_kernel_x1_list[l - 1]
      
      nngp_kernel_x2_list[l] = sigma2_b + (sigma2_w / 2) * nngp_kernel_x2_list[l - 1]
    }
    
  }
  
  return(nngp_kernel_list[L + 1])
  
}

# Calculates the kernel function for a given vector with itself for the last layer L of the DNN
# Also requires the weight and bias variances
nngp_kernel_layer_L_same_vec <- function(L, x1, sigma2_w, sigma2_b) {
  
  # Store the kernel functions for the input with itself in a vector
  # of size L + 1 reflecting the initial kernel and those for the L layers
  nngp_kernel_x1_list = rep(NA, L + 1)
  
  # Set the initial kernel functions
  nngp_kernel_x1_list[1] = nngp_kernel_init(x1 = x1, x2 = x1, sigma2_w = sigma2_w, sigma2_b = sigma2_b)
  
  # For l = 1,...,L calculate the l-th kernel function using the previous one
  for (l in 2:(L + 1)) {
    
    nngp_kernel_x1_list[l] = sigma2_b + (sigma2_w / 2) * nngp_kernel_x1_list[l - 1]
    
  }
  
  return(nngp_kernel_x1_list[L + 1])
  
  
}

full_nngp_kernel_time_series_L <- function(time_grid, sigma2_w, sigma2_b, nngp_L) {
  
  T_len = length(time_grid)
  
  nngp_kernel = matrix(NA, nrow = T_len, ncol = T_len)
  
  # Calculate kernel with given parameters
  for (t1 in 1:T_len) {
    
    for (t2 in t1:T_len) {
      
      if (t1 == t2) {
        
        nngp_kernel[t1, t2] = nngp_kernel_layer_L_same_vec(L = nngp_L, x1 = time_grid[t1], 
                                                           sigma2_w = sigma2_w,
                                                           sigma2_b = sigma2_b)
        
      } 
      
      else {
        
        nngp_kernel[t1, t2] = nngp_kernel_layer_L(L = nngp_L, x1 = time_grid[t1], 
                                                  x2 = time_grid[t2],
                                                  sigma2_w = sigma2_w,
                                                  sigma2_b = sigma2_b)
        
        nngp_kernel[t2, t1] = nngp_kernel[t1, t2]
        
      }
    }
  }
  
  return(nngp_kernel)
  
}


###################################################################################################


###### Functions for calculating likelihoods in MCMC iterations
nngp_log_likelihood <- function(series_vals, time_grid, sigma2_w, sigma2_b, nngp_L) {
  
  T_len = length(time_grid)
  
  # Get time series length
  
  nngp_kernel = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w, 
                                               sigma2_b = sigma2_b, nngp_L = nngp_L)
  
  log_likelihood = dmvnorm(series_vals, mean = rep(0, T_len), sigma = nngp_kernel,
                           log = TRUE)
  
  return(log_likelihood)
  
}


nngp_log_likelihood_mu <- function(mu_series, zeta_series, xi_series, omega_mats, y_adj_mats,
                                   time_grid, 
                                   sigma2_w, sigma2_b, nngp_L) {
  
  T_len = length(time_grid)
  
  J = dim(xi_series)[1]
  L = dim(xi_series)[4]
  
  # Get time series length
  
  nngp_kernel = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w, 
                                               sigma2_b = sigma2_b, nngp_L = nngp_L)
  
  
  # Terms for mu update
  
  
  log_likelihood = dmvnorm(mu_series, mean = rep(0, T_len), sigma = nngp_kernel,
                           log = TRUE)
  
  
  
  
  
  return(log_likelihood)
  
}

nngp_log_likelihood_zeta <- function(mu_series, zeta_series, xi_series, omega_mats, y_adj_mats,
                                     time_grid, 
                                     sigma2_w, sigma2_b, nngp_L) {
  
  T_len = length(time_grid)
  
  J = dim(zeta_series)[1]
  R = dim(zeta_series)[2]
  L = dim(xi_series)[4]
  
  log_likelihood = 0
  
  # Get time series length
  
  nngp_kernel = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w, 
                                               sigma2_b = sigma2_b, nngp_L = nngp_L)
  
  for (j in 1:J) {
    for (r in 1:R) {
      
      log_likelihood = log_likelihood + dmvnorm(zeta_series[j, r, ], mean = rep(0, T_len), sigma = nngp_kernel,
                                                log = TRUE)
      
    }
    
  }  
  
  return(log_likelihood)
  
}

nngp_log_likelihood_xi <- function(mu_series, zeta_series, xi_series, omega_mats, y_adj_mats,
                                   time_grid, x_arr, eta_series, alpha_series, sigma2_k,
                                   sigma2_w, sigma2_b, nngp_L) {
  
  T_len = length(time_grid)
  
  J = dim(xi_series)[1]
  R = dim(xi_series)[2]
  
  L = dim(xi_series)[4]
  
  m = dim(x_arr)[2]
  
  log_likelihood = 0
  
  # Get time series length
  
  nngp_kernel = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w, 
                                               sigma2_b = sigma2_b, nngp_L = nngp_L)
  
  
  
  
  for (j in 1:J) {
    for (l in 1:L) {
      for (r in 1:R) {
        
        log_likelihood = log_likelihood + dmvnorm(xi_series[j, r, , l], mean = rep(0, T_len), sigma = nngp_kernel,
                                                  log = TRUE)
        
      }
    }
  }  
  
  return(log_likelihood)
  
}



nngp_log_likelihood_xi_net <- function(mu_series, zeta_series, xi_series, omega_mats, y_adj_mats,
                                       time_grid, x_arr, eta_series, alpha_series, sigma2_k,
                                       sigma2_w, sigma2_b, nngp_L) {
  
  T_len = length(time_grid)
  
  J = dim(xi_series)[1]
  R = dim(xi_series)[2]
  
  L = dim(xi_series)[4]
  
  m = dim(x_arr)[2]
  
  log_likelihood = 0
  
  # Get time series length
  
  nngp_kernel = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w, 
                                               sigma2_b = sigma2_b, nngp_L = nngp_L)
  
  
  
  
  for (j in 1:J) {
    for (l in 1:L) {
      for (r in 1:R) {
        
        log_likelihood = log_likelihood + dmvnorm(xi_series[j, r, , l], mean = rep(0, T_len), sigma = nngp_kernel,
                                                  log = TRUE)
        
      }
    }
  }  
  
  return(log_likelihood)
  
}


nngp_log_likelihood_xi_node <- function(mu_series, zeta_series, xi_series, omega_mats, y_adj_mats,
                                        time_grid, x_arr, eta_series, alpha_series, sigma2_k,
                                        sigma2_w, sigma2_b, nngp_L) {
  
  T_len = length(time_grid)
  
  J = dim(xi_series)[1]
  R = dim(xi_series)[2]
  
  L = dim(xi_series)[4]
  
  m = dim(x_arr)[2]
  
  log_likelihood = 0
  
  # Get time series length
  
  nngp_kernel = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w, 
                                               sigma2_b = sigma2_b, nngp_L = nngp_L)
  
  
  
  
  for (j in 1:J) {
    for (l in 1:L) {
      for (r in 1:R) {
        
        log_likelihood = log_likelihood + dmvnorm(xi_series[j, r, , l], mean = rep(0, T_len), sigma = nngp_kernel,
                                                  log = TRUE)
        
      }
    }
  }  
  
  return(log_likelihood)
  
}


nngp_log_likelihood_eta <- function(xi_series,
                                    time_grid, x_arr, eta_series, alpha_series, sigma2_k,
                                    sigma2_w, sigma2_b, nngp_L) {
  
  T_len = length(time_grid)
  
  m = dim(x_arr)[2]
  
  L = dim(xi_series)[4]
  
  R = dim(xi_series)[2]
  J = dim(xi_series)[1]
  
  log_likelihood = 0
  
  # Get time series length
  
  nngp_kernel = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w, 
                                               sigma2_b = sigma2_b, nngp_L = nngp_L)
  
  
  for (k in 1:m) {
    
    log_likelihood = log_likelihood + dmvnorm(eta_series[k, ], mean = rep(0, T_len), sigma = nngp_kernel,
                                              log = TRUE)
    
    
  }  
  
  return(log_likelihood)
  
}

nngp_log_likelihood_alpha <- function(xi_series,
                                      time_grid, x_arr, eta_series, alpha_series, sigma2_k,
                                      sigma2_w, sigma2_b, nngp_L) {
  
  T_len = length(time_grid)
  
  m = dim(x_arr)[2]
  
  L = dim(xi_series)[4]
  
  R = dim(xi_series)[2]
  J = dim(xi_series)[1]
  
  log_likelihood = 0
  
  # Get time series length
  
  nngp_kernel = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w, 
                                               sigma2_b = sigma2_b, nngp_L = nngp_L)
  
  
  
  for (k in 1:m) {
    for (l in 1:L) {
      for (r in 1:R) {
        
        log_likelihood = log_likelihood + dmvnorm(alpha_series[k, r, , l], mean = rep(0, T_len), sigma = nngp_kernel,
                                                  log = TRUE)
        
      }
    }
  }  
  
  return(log_likelihood)
  
}

# Optimize the log likelihood over a grid for the nngp weight or bias variance

optimize_sigma2_mu_log_likelihood <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                              time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L) {
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_log_likelihoods = matrix(0, nrow = sigma2_b_len, ncol = sigma2_w_len)
  
  for (b in sigma2_b_len) {
    for (w in sigma2_w_len) { 
      sigma2_log_likelihoods[b, w] = nngp_log_likelihood_mu(mu_series = mu_series, 
                                                            zeta_series = zeta_series,
                                                            xi_series = xi_series,
                                                            omega_mats = omega_mats,
                                                            y_adj_mats = y_adj_mats,
                                                            time_grid = time_grid, 
                                                            sigma2_w = sigma2_w_grid[w],
                                                            sigma2_b = sigma2_b_grid[b],
                                                            nngp_L = nngp_L)
    }
  }
  
  best_sigma2_idx = arrayInd(which.max(sigma2_log_likelihoods), dim(sigma2_log_likelihoods))
  
  

  
  out_list = list(sigma2_b = sigma2_b_grid[best_sigma2_idx[1, 1]],
                  sigma2_w = sigma2_w_grid[best_sigma2_idx[1, 2]])
  
  return(out_list)
  
}

sample_sigma2_mu <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                             time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L, sigma2_b_old, sigma2_w_old) {
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_b_log_likelihoods = rep(0, sigma2_b_len)
  
  for (b in sigma2_b_len) {
    sigma2_b_log_likelihoods[b] = nngp_log_likelihood_mu(mu_series = mu_series, 
                                                         zeta_series = zeta_series,
                                                         xi_series = xi_series,
                                                         omega_mats = omega_mats,
                                                         y_adj_mats = y_adj_mats,
                                                         time_grid = time_grid, 
                                                         sigma2_w = sigma2_w_old,
                                                         sigma2_b = sigma2_b_grid[b],
                                                         nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_b_log_likelihoods)) == Inf) {
    
    sigma2_b_new = sigma2_b_grid[which.max(sigma2_b_log_likelihoods)]
    
  }
  
  else {
    sigma2_b_new = sample(sigma2_b_grid, size = 1, replace = FALSE, prob = exp(sigma2_b_log_likelihoods))
  }
  
  
  sigma2_w_log_likelihoods = rep(0, sigma2_w_len)
  
  for (w in sigma2_w_len) {
    sigma2_w_log_likelihoods[w] = nngp_log_likelihood_mu(mu_series = mu_series, 
                                                         zeta_series = zeta_series,
                                                         xi_series = xi_series,
                                                         omega_mats = omega_mats,
                                                         y_adj_mats = y_adj_mats,
                                                         time_grid = time_grid, 
                                                         sigma2_w = sigma2_w_grid[w],
                                                         sigma2_b = sigma2_b_new,
                                                         nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_w_log_likelihoods)) == Inf) {
    
    sigma2_w_new = sigma2_w_grid[which.max(sigma2_w_log_likelihoods)]
    
  }
  
  else {
    sigma2_w_new = sample(sigma2_w_grid, size = 1, replace = FALSE, prob = exp(sigma2_w_log_likelihoods))
  }
  
  

  
  out_list = list(sigma2_b = sigma2_b_new,
                  sigma2_w = sigma2_w_new)
  
  return(out_list)
  
}



optimize_sigma2_xi_log_likelihood <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                              x_arr, eta_series, alpha_series, sigma2_k,
                                              time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L) {
  
  xi_series_dim = dim(xi_series)
  
  J = xi_series_dim[1]
  R = xi_series_dim[2]
  
  L = xi_series_dim[4]
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_log_likelihoods = matrix(0, nrow = sigma2_b_len, ncol = sigma2_w_len)
  
  for (b in sigma2_b_len) {
    for (w in sigma2_w_len) { 
      
      sigma2_log_likelihoods[b, w] = (sigma2_log_likelihoods[b, w] + 
                                        nngp_log_likelihood_xi(mu_series = mu_series, 
                                                               zeta_series = zeta_series,
                                                               xi_series = xi_series,
                                                               omega_mats = omega_mats,
                                                               y_adj_mats = y_adj_mats,
                                                               x_arr = x_arr, 
                                                               eta_series = eta_series,
                                                               alpha_series = alpha_series,
                                                               sigma2_k = sigma2_k,
                                                               time_grid = time_grid, 
                                                               sigma2_w = sigma2_w_grid[w],
                                                               sigma2_b = sigma2_b_grid[b],
                                                               nngp_L = nngp_L))
      
      
      
    }
  }
  
  best_sigma2_idx = arrayInd(which.max(sigma2_log_likelihoods), dim(sigma2_log_likelihoods))
  
  out_list = list(sigma2_b = sigma2_b_grid[best_sigma2_idx[1, 1]],
                  sigma2_w = sigma2_w_grid[best_sigma2_idx[1, 2]])
  
  return(out_list)
  
}



sample_sigma2_xi <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                             x_arr, eta_series, alpha_series, sigma2_k,
                             time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L, sigma2_b_old, sigma2_w_old) {
  
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_b_log_likelihoods = rep(0, sigma2_b_len)
  
  for (b in sigma2_b_len) {
    sigma2_b_log_likelihoods[b] = nngp_log_likelihood_xi(mu_series = mu_series, 
                                                         zeta_series = zeta_series,
                                                         xi_series = xi_series,
                                                         omega_mats = omega_mats,
                                                         y_adj_mats = y_adj_mats,
                                                         x_arr = x_arr, 
                                                         eta_series = eta_series,
                                                         alpha_series = alpha_series,
                                                         sigma2_k = sigma2_k,
                                                         time_grid = time_grid, 
                                                         sigma2_w = sigma2_w_old,
                                                         sigma2_b = sigma2_b_grid[b],
                                                         nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_b_log_likelihoods)) == Inf) {
    
    sigma2_b_new = sigma2_b_grid[which.max(sigma2_b_log_likelihoods)]
    
  }
  
  else {
    sigma2_b_new = sample(sigma2_b_grid, size = 1, replace = FALSE, prob = exp(sigma2_b_log_likelihoods))
  }
  
  
  sigma2_w_log_likelihoods = rep(0, sigma2_w_len)
  
  for (w in sigma2_w_len) {
    sigma2_w_log_likelihoods[w] = nngp_log_likelihood_xi(mu_series = mu_series, 
                                                         zeta_series = zeta_series,
                                                         xi_series = xi_series,
                                                         omega_mats = omega_mats,
                                                         y_adj_mats = y_adj_mats,
                                                         x_arr = x_arr, 
                                                         eta_series = eta_series,
                                                         alpha_series = alpha_series,
                                                         sigma2_k = sigma2_k,
                                                         time_grid = time_grid, 
                                                         sigma2_w = sigma2_w_grid[w],
                                                         sigma2_b = sigma2_b_new,
                                                         nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_w_log_likelihoods)) == Inf) {
    
    sigma2_w_new = sigma2_w_grid[which.max(sigma2_w_log_likelihoods)]
    
  }
  
  else {
    sigma2_w_new = sample(sigma2_w_grid, size = 1, replace = FALSE, prob = exp(sigma2_w_log_likelihoods))
  }
  
  

  
  out_list = list(sigma2_b = sigma2_b_new,
                  sigma2_w = sigma2_w_new)
  
  return(out_list)
  
  
  
}




optimize_sigma2_xi_net_log_likelihood <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                                  x_arr, eta_series, alpha_series, sigma2_k,
                                                  time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L) {
  
  xi_series_dim = dim(xi_series)
  
  J = xi_series_dim[1]
  R = xi_series_dim[2]
  
  L = xi_series_dim[4]
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_log_likelihoods = matrix(0, nrow = sigma2_b_len, ncol = sigma2_w_len)
  
  for (b in sigma2_b_len) {
    for (w in sigma2_w_len) { 
      
      sigma2_log_likelihoods[b, w] = (sigma2_log_likelihoods[b, w] + 
                                        nngp_log_likelihood_xi_net(mu_series = mu_series, 
                                                                   zeta_series = zeta_series,
                                                                   xi_series = xi_series,
                                                                   omega_mats = omega_mats,
                                                                   y_adj_mats = y_adj_mats,
                                                                   x_arr = x_arr, 
                                                                   eta_series = eta_series,
                                                                   alpha_series = alpha_series,
                                                                   sigma2_k = sigma2_k,
                                                                   time_grid = time_grid, 
                                                                   sigma2_w = sigma2_w_grid[w],
                                                                   sigma2_b = sigma2_b_grid[b],
                                                                   nngp_L = nngp_L))
      
      
      
    }
  }
  
  best_sigma2_idx = arrayInd(which.max(sigma2_log_likelihoods), dim(sigma2_log_likelihoods))
  
  out_list = list(sigma2_b = sigma2_b_grid[best_sigma2_idx[1, 1]],
                  sigma2_w = sigma2_w_grid[best_sigma2_idx[1, 2]])
  
  return(out_list)
  
}






sample_sigma2_xi_net <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                 x_arr, eta_series, alpha_series, sigma2_k,
                                 time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L, sigma2_b_old, sigma2_w_old) {
  
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_b_log_likelihoods = rep(0, sigma2_b_len)
  
  for (b in sigma2_b_len) {
    sigma2_b_log_likelihoods[b] = nngp_log_likelihood_xi_net(mu_series = mu_series, 
                                                             zeta_series = zeta_series,
                                                             xi_series = xi_series,
                                                             omega_mats = omega_mats,
                                                             y_adj_mats = y_adj_mats,
                                                             x_arr = x_arr, 
                                                             eta_series = eta_series,
                                                             alpha_series = alpha_series,
                                                             sigma2_k = sigma2_k,
                                                             time_grid = time_grid, 
                                                             sigma2_w = sigma2_w_old,
                                                             sigma2_b = sigma2_b_grid[b],
                                                             nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_b_log_likelihoods)) == Inf) {
    
    sigma2_b_new = sigma2_b_grid[which.max(sigma2_b_log_likelihoods)]
    
  }
  
  else {
    sigma2_b_new = sample(sigma2_b_grid, size = 1, replace = FALSE, prob = exp(sigma2_b_log_likelihoods))
  }
  
  
  sigma2_w_log_likelihoods = rep(0, sigma2_w_len)
  
  for (w in sigma2_w_len) {
    sigma2_w_log_likelihoods[w] = nngp_log_likelihood_xi_net(mu_series = mu_series, 
                                                             zeta_series = zeta_series,
                                                             xi_series = xi_series,
                                                             omega_mats = omega_mats,
                                                             y_adj_mats = y_adj_mats,
                                                             x_arr = x_arr, 
                                                             eta_series = eta_series,
                                                             alpha_series = alpha_series,
                                                             sigma2_k = sigma2_k,
                                                             time_grid = time_grid, 
                                                             sigma2_w = sigma2_w_grid[w],
                                                             sigma2_b = sigma2_b_new,
                                                             nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_w_log_likelihoods)) == Inf) {
    
    sigma2_w_new = sigma2_w_grid[which.max(sigma2_w_log_likelihoods)]
    
  }
  
  else {
    sigma2_w_new = sample(sigma2_w_grid, size = 1, replace = FALSE, prob = exp(sigma2_w_log_likelihoods))
  }
  
  

  
  out_list = list(sigma2_b = sigma2_b_new,
                  sigma2_w = sigma2_w_new)
  
  return(out_list)
  
  
  
}











optimize_sigma2_xi_node_log_likelihood <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                                   x_arr, eta_series, alpha_series, sigma2_k,
                                                   time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L) {
  
  xi_series_dim = dim(xi_series)
  
  J = xi_series_dim[1]
  R = xi_series_dim[2]
  
  L = xi_series_dim[4]
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_log_likelihoods = matrix(0, nrow = sigma2_b_len, ncol = sigma2_w_len)
  
  for (b in sigma2_b_len) {
    for (w in sigma2_w_len) { 
      
      sigma2_log_likelihoods[b, w] = (sigma2_log_likelihoods[b, w] + 
                                        nngp_log_likelihood_xi_node(mu_series = mu_series, 
                                                                    zeta_series = zeta_series,
                                                                    xi_series = xi_series,
                                                                    omega_mats = omega_mats,
                                                                    y_adj_mats = y_adj_mats,
                                                                    x_arr = x_arr, 
                                                                    eta_series = eta_series,
                                                                    alpha_series = alpha_series,
                                                                    sigma2_k = sigma2_k,
                                                                    time_grid = time_grid, 
                                                                    sigma2_w = sigma2_w_grid[w],
                                                                    sigma2_b = sigma2_b_grid[b],
                                                                    nngp_L = nngp_L))
      
      
      
    }
  }
  
  best_sigma2_idx = arrayInd(which.max(sigma2_log_likelihoods), dim(sigma2_log_likelihoods))
  
  out_list = list(sigma2_b = sigma2_b_grid[best_sigma2_idx[1, 1]],
                  sigma2_w = sigma2_w_grid[best_sigma2_idx[1, 2]])
  
  return(out_list)
  
}





sample_sigma2_xi_node <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                  x_arr, eta_series, alpha_series, sigma2_k,
                                  time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L, sigma2_b_old, sigma2_w_old) {
  
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_b_log_likelihoods = rep(0, sigma2_b_len)
  
  for (b in sigma2_b_len) {
    sigma2_b_log_likelihoods[b] = nngp_log_likelihood_xi_node(mu_series = mu_series, 
                                                              zeta_series = zeta_series,
                                                              xi_series = xi_series,
                                                              omega_mats = omega_mats,
                                                              y_adj_mats = y_adj_mats,
                                                              x_arr = x_arr, 
                                                              eta_series = eta_series,
                                                              alpha_series = alpha_series,
                                                              sigma2_k = sigma2_k,
                                                              time_grid = time_grid, 
                                                              sigma2_w = sigma2_w_old,
                                                              sigma2_b = sigma2_b_grid[b],
                                                              nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_b_log_likelihoods)) == Inf) {
    
    sigma2_b_new = sigma2_b_grid[which.max(sigma2_b_log_likelihoods)]
    
  }
  
  else {
    sigma2_b_new = sample(sigma2_b_grid, size = 1, replace = FALSE, prob = exp(sigma2_b_log_likelihoods))
  }
  
  
  sigma2_w_log_likelihoods = rep(0, sigma2_w_len)
  
  for (w in sigma2_w_len) {
    sigma2_w_log_likelihoods[w] = nngp_log_likelihood_xi_node(mu_series = mu_series, 
                                                              zeta_series = zeta_series,
                                                              xi_series = xi_series,
                                                              omega_mats = omega_mats,
                                                              y_adj_mats = y_adj_mats,
                                                              x_arr = x_arr, 
                                                              eta_series = eta_series,
                                                              alpha_series = alpha_series,
                                                              sigma2_k = sigma2_k,
                                                              time_grid = time_grid, 
                                                              sigma2_w = sigma2_w_grid[w],
                                                              sigma2_b = sigma2_b_new,
                                                              nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_w_log_likelihoods)) == Inf) {
    
    sigma2_w_new = sigma2_w_grid[which.max(sigma2_w_log_likelihoods)]
    
  }
  
  else {
    sigma2_w_new = sample(sigma2_w_grid, size = 1, replace = FALSE, prob = exp(sigma2_w_log_likelihoods))
  }
  
  

  
  out_list = list(sigma2_b = sigma2_b_new,
                  sigma2_w = sigma2_w_new)
  
  return(out_list)
  
  
  
}





optimize_sigma2_zeta_log_likelihood <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                                time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L) {
  
  zeta_series_dim = dim(zeta_series)
  
  J = zeta_series_dim[1]
  R = zeta_series_dim[2]
  
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_log_likelihoods = matrix(0, nrow = sigma2_b_len, ncol = sigma2_w_len)
  
  for (b in sigma2_b_len) {
    for (w in sigma2_w_len) {   
      
      
      sigma2_log_likelihoods[b, w] = (sigma2_log_likelihoods[b, w] + 
                                        nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                                 zeta_series = zeta_series,
                                                                 xi_series = xi_series,
                                                                 omega_mats = omega_mats,
                                                                 y_adj_mats = y_adj_mats,
                                                                 time_grid = time_grid, 
                                                                 sigma2_w = sigma2_w_grid[w],
                                                                 sigma2_b = sigma2_b_grid[b],
                                                                 nngp_L = nngp_L))
      
    }
  }
  
  best_sigma2_idx = arrayInd(which.max(sigma2_log_likelihoods), dim(sigma2_log_likelihoods))
  
  out_list = list(sigma2_b = sigma2_b_grid[best_sigma2_idx[1, 1]],
                  sigma2_w = sigma2_w_grid[best_sigma2_idx[1, 2]])
  
  return(out_list)
  
}



sample_sigma2_zeta <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                               time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L, sigma2_b_old, sigma2_w_old) {
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_b_log_likelihoods = rep(0, sigma2_b_len)
  
  for (b in sigma2_b_len) {
    sigma2_b_log_likelihoods[b] = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                           zeta_series = zeta_series,
                                                           xi_series = xi_series,
                                                           omega_mats = omega_mats,
                                                           y_adj_mats = y_adj_mats,
                                                           time_grid = time_grid, 
                                                           sigma2_w = sigma2_w_old,
                                                           sigma2_b = sigma2_b_grid[b],
                                                           nngp_L = nngp_L)
  }
  

  
  if (max(exp(sigma2_b_log_likelihoods)) == Inf) {
    
    sigma2_b_new = sigma2_b_grid[which.max(sigma2_b_log_likelihoods)]
    
  }
  
  else {
    sigma2_b_new = sample(sigma2_b_grid, size = 1, replace = FALSE, prob = exp(sigma2_b_log_likelihoods))
  }
  
  sigma2_w_log_likelihoods = rep(0, sigma2_w_len)
  
  for (w in sigma2_w_len) {
    sigma2_w_log_likelihoods[w] = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                           zeta_series = zeta_series,
                                                           xi_series = xi_series,
                                                           omega_mats = omega_mats,
                                                           y_adj_mats = y_adj_mats,
                                                           time_grid = time_grid, 
                                                           sigma2_w = sigma2_w_grid[w],
                                                           sigma2_b = sigma2_b_new,
                                                           nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_w_log_likelihoods)) == Inf) {
    
    sigma2_w_new = sigma2_w_grid[which.max(sigma2_w_log_likelihoods)]
    
  }
  
  else {
    sigma2_w_new = sample(sigma2_w_grid, size = 1, replace = FALSE, prob = exp(sigma2_w_log_likelihoods))
  }
  
  
  

  
  out_list = list(sigma2_b = sigma2_b_new,
                  sigma2_w = sigma2_w_new)
  
  return(out_list)
  
}








optimize_sigma2_alpha_log_likelihood <- function(xi_series, 
                                                 x_arr, eta_series, alpha_series, sigma2_k,
                                                 time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L) {
  
  alpha_series_dim = dim(alpha_series)
  
  m = alpha_series_dim[1]
  R = alpha_series_dim[2]
  
  L = alpha_series_dim[4]
  
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_log_likelihoods = matrix(0, nrow = sigma2_b_len, ncol = sigma2_w_len)
  
  for (b in sigma2_b_len) {
    for (w in sigma2_w_len) { 
      
      
      sigma2_log_likelihoods[b, w] = (sigma2_log_likelihoods[b, w] + 
                                        nngp_log_likelihood_alpha(xi_series = xi_series, 
                                                                  x_arr = x_arr, 
                                                                  eta_series = eta_series,
                                                                  alpha_series = alpha_series,
                                                                  sigma2_k = sigma2_k,
                                                                  time_grid = time_grid, 
                                                                  sigma2_w = sigma2_w_grid[w],
                                                                  sigma2_b = sigma2_b_grid[b],
                                                                  nngp_L = nngp_L))
      
      
    }
  }
  
  best_sigma2_idx = arrayInd(which.max(sigma2_log_likelihoods), dim(sigma2_log_likelihoods))
  
  out_list = list(sigma2_b = sigma2_b_grid[best_sigma2_idx[1, 1]],
                  sigma2_w = sigma2_w_grid[best_sigma2_idx[1, 2]])
  
  return(out_list)
  
}




sample_sigma2_alpha <- function(xi_series, 
                                x_arr, eta_series, alpha_series, sigma2_k,
                                time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L, sigma2_b_old, sigma2_w_old) {
  
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_b_log_likelihoods = rep(0, sigma2_b_len)
  
  for (b in sigma2_b_len) {
    sigma2_b_log_likelihoods[b] = nngp_log_likelihood_alpha(xi_series = xi_series, 
                                                            x_arr = x_arr, 
                                                            eta_series = eta_series,
                                                            alpha_series = alpha_series,
                                                            sigma2_k = sigma2_k,
                                                            time_grid = time_grid, 
                                                            sigma2_w = sigma2_w_old,
                                                            sigma2_b = sigma2_b_grid[b],
                                                            nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_b_log_likelihoods)) == Inf) {
    
    sigma2_b_new = sigma2_b_grid[which.max(sigma2_b_log_likelihoods)]
    
  }
  
  else {
    sigma2_b_new = sample(sigma2_b_grid, size = 1, replace = FALSE, prob = exp(sigma2_b_log_likelihoods))
  }
  
  
  sigma2_w_log_likelihoods = rep(0, sigma2_w_len)
  
  for (w in sigma2_w_len) {
    sigma2_w_log_likelihoods[w] = nngp_log_likelihood_alpha(xi_series = xi_series, 
                                                            x_arr = x_arr, 
                                                            eta_series = eta_series,
                                                            alpha_series = alpha_series,
                                                            sigma2_k = sigma2_k,
                                                            time_grid = time_grid, 
                                                            sigma2_w = sigma2_w_grid[w],
                                                            sigma2_b = sigma2_b_new,
                                                            nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_w_log_likelihoods)) == Inf) {
    
    sigma2_w_new = sigma2_w_grid[which.max(sigma2_w_log_likelihoods)]
    
  }
  
  else {
    sigma2_w_new = sample(sigma2_w_grid, size = 1, replace = FALSE, prob = exp(sigma2_w_log_likelihoods))
  }
  

  
  out_list = list(sigma2_b = sigma2_b_new,
                  sigma2_w = sigma2_w_new)
  
  return(out_list)
  
  
  
}





optimize_sigma2_eta_log_likelihood <- function(xi_series, 
                                               x_arr, eta_series, alpha_series, sigma2_k,
                                               time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L) {
  
  eta_series_dim = dim(eta_series)
  
  m = eta_series_dim[1]
  
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_log_likelihoods = matrix(0, nrow = sigma2_b_len, ncol = sigma2_w_len)
  
  for (b in sigma2_b_len) {
    for (w in sigma2_w_len) { 
      
      
      
      sigma2_log_likelihoods[b, w] = (sigma2_log_likelihoods[b, w] + 
                                        nngp_log_likelihood_eta(xi_series = xi_series, 
                                                                x_arr = x_arr, 
                                                                eta_series = eta_series,
                                                                alpha_series = alpha_series,
                                                                sigma2_k = sigma2_k,
                                                                time_grid = time_grid, 
                                                                sigma2_w = sigma2_w_grid[w],
                                                                sigma2_b = sigma2_b_grid[b],
                                                                nngp_L = nngp_L))
      
      
    }
  }
  
  best_sigma2_idx = arrayInd(which.max(sigma2_log_likelihoods), dim(sigma2_log_likelihoods))
  
  out_list = list(sigma2_b = sigma2_b_grid[best_sigma2_idx[1, 1]],
                  sigma2_w = sigma2_w_grid[best_sigma2_idx[1, 2]])
  
  return(out_list)
  
}



sample_sigma2_eta <- function(xi_series, 
                              x_arr, eta_series, alpha_series, sigma2_k,
                              time_grid, sigma2_b_grid, sigma2_w_grid, nngp_L, sigma2_b_old, sigma2_w_old) {
  
  
  sigma2_b_len = length(sigma2_b_grid)
  sigma2_w_len = length(sigma2_w_grid)
  
  
  sigma2_b_log_likelihoods = rep(0, sigma2_b_len)
  
  for (b in sigma2_b_len) {
    sigma2_b_log_likelihoods[b] = nngp_log_likelihood_eta(xi_series = xi_series, 
                                                          x_arr = x_arr, 
                                                          eta_series = eta_series,
                                                          alpha_series = alpha_series,
                                                          sigma2_k = sigma2_k,
                                                          time_grid = time_grid, 
                                                          sigma2_w = sigma2_w_old,
                                                          sigma2_b = sigma2_b_grid[b],
                                                          nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_b_log_likelihoods)) == Inf) {
    
    sigma2_b_new = sigma2_b_grid[which.max(sigma2_b_log_likelihoods)]
    
  }
  
  else {
    sigma2_b_new = sample(sigma2_b_grid, size = 1, replace = FALSE, prob = exp(sigma2_b_log_likelihoods))
  }
  
  
  sigma2_w_log_likelihoods = rep(0, sigma2_w_len)
  
  for (w in sigma2_w_len) {
    sigma2_w_log_likelihoods[w] = nngp_log_likelihood_eta(xi_series = xi_series, 
                                                          x_arr = x_arr, 
                                                          eta_series = eta_series,
                                                          alpha_series = alpha_series,
                                                          sigma2_k = sigma2_k,
                                                          time_grid = time_grid, 
                                                          sigma2_w = sigma2_w_grid[w],
                                                          sigma2_b = sigma2_b_new,
                                                          nngp_L = nngp_L)
  }
  
  if (max(exp(sigma2_w_log_likelihoods)) == Inf) {
    
    sigma2_w_new = sigma2_w_grid[which.max(sigma2_w_log_likelihoods)]
    
  }
  
  else {
    sigma2_w_new = sample(sigma2_w_grid, size = 1, replace = FALSE, prob = exp(sigma2_w_log_likelihoods))
  }
  
  

  
  out_list = list(sigma2_b = sigma2_b_new,
                  sigma2_w = sigma2_w_new)
  
  return(out_list)
  
  
  
}


####### Functions for Gaussian Process Predictions for the time-varying parameters

gp_param_prediction <- function(t, time_grid, param_series, sigma2_w, sigma2_b, nngp_L = 1) {
  
  
  if (t %in% time_grid) {
    
    return(param_series[which(time_grid == t)])
    
  }
  

  
  T_len = length(time_grid)
  
  kernel_obs = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w,
                                              sigma2_b = sigma2_b, nngp_L = nngp_L)
  
  
  kernel_t = nngp_kernel_layer_L_same_vec(L = nngp_L, x1 = t, sigma2_w = sigma2_w,
                                          sigma2_b = sigma2_b)
  
  kernel_vec_t = rep(NA, T_len)
  
  for (i in 1:T_len) {
    
    kernel_vec_t[i] = nngp_kernel_layer_L(L = nngp_L, x1 = t, x2 = time_grid[i], sigma2_w = sigma2_w,
                                          sigma2_b = sigma2_b)
    
  }
  
  t_mean = as.numeric(t(kernel_vec_t) %*% chol2inv(chol(kernel_obs)) %*% param_series)

  
  t_var = kernel_t - as.numeric(t(kernel_vec_t) %*% chol2inv(chol(kernel_obs)) %*% kernel_vec_t)
  

  
  t_var = round(t_var, digits = 7)
  

  
  if (t_var < 0) {
    
    return(t_mean)
    
  }
  
  else {
    return(rnorm(1, mean = t_mean, sd = sqrt(t_var)))
  }
}







gp_edge_prediction <- function(t, j, j_prime, l, time_grid,
                               mu_series, zeta_series, xi_series) {
  
  R = dim(xi_series)[2]
  R_zeta = dim(zeta_series)[2]
  
  
  t_idx = which(time_grid == t)
  
  # Calculate mu^t
  mu_t = mu_series[t_idx]
  

  
  
  # Calculate xi^t_{j, l, r} for r = 1,...,R
  xi_j_l_t = rep(NA, R) 
  
  for (r in 1:R) {
    
    xi_j_l_t[r] =  xi_series[j, r, t_idx, l]
    
  }
  
  
  # Calculate xi^t_{j', l, r} for r = 1,...,R
  xi_j_prime_l_t = rep(NA, R) 
  
  for (r in 1:R) {
    
    xi_j_prime_l_t[r] = xi_series[j_prime, r, t_idx, l]
    
  }
  
  # Calculate zeta^t_{j, r} for r = 1,...,R
  zeta_j_t = rep(NA, R_zeta) 
  
  for (r in 1:R_zeta) {
    
    zeta_j_t[r] = zeta_series[j, r, t_idx]
    
  }
  
  
  # Calculate zeta^t_{j', r} for r = 1,...,R
  zeta_j_prime_t = rep(NA, R_zeta) 
  
  for (r in 1:R_zeta) {
    
    zeta_j_prime_t[r] = zeta_series[j_prime, r, t_idx]
    
  }
  
  
  linear_term = (mu_t + as.numeric(crossprod(xi_j_l_t, xi_j_prime_l_t)) + 
                   as.numeric(crossprod(zeta_j_t, zeta_j_prime_t)))
  
  probability_term = exp(linear_term) / (1 + exp(linear_term))
  
  
  if (is.nan(probability_term)) {
    
    edge_val = 1
    
  } 
  
  else {
    edge_val = rbinom(1, size = 1, prob = probability_term)
  }
  
  return(edge_val)
  
}


gp_node_covariate_prediction <- function(t, j, k, time_grid, sigma2_k,
                                         eta_series, xi_series, alpha_series) {
  
  
  R = dim(xi_series)[2]
  L = dim(xi_series)[4]
  
  t_idx = which(time_grid == t)
  
  # Calculate eta^t_k
  eta_k_t = eta_series[k, t_idx]
  
  # Calculate xi^t_{j, l, r} for r = 1,...,R
  xi_j_t = matrix(NA, nrow = R, ncol = L) 
  
  # Calculate alpha^t_{k, l, r} for r = 1,...,R
  alpha_k_t = matrix(NA, nrow = R, ncol = L) 
  
  
  
  
  for (r in 1:R) {
    for (l in 1:L) {
      
      xi_j_t[r, l] = xi_series[j, r, t_idx, l]
      
      alpha_k_t[r, l] = alpha_series[k, r, t_idx, l]
      
    }
  }
  
  linear_term = eta_k_t + sum(xi_j_t * alpha_k_t) + rnorm(1, mean = 0, sd = sqrt(sigma2_k[k]))
  
  return(linear_term)
  
}






############# Functions for missing values

### Function: random_edge_removal

### Parameters:
# multilayer_network: An array of dimension L x V(V - 1) / 2 x T, containing the upper triangular portions
#                     of the networks on each layer and for each time point
# prop_missing: A real value between 0 and 1, representing the proportion of values to be made missing on each layer
# obs_years: A vector containing the years that were observed

### Returns:
# missing_multilayer_network: An array of dimension L x V(V - 1) / 2 x T, containing the upper triangular portions
#                             of the networks on each layer and for each time point, now with values missing
# missing_years: A vector of the years that contain missing values
random_edge_removal <- function(multilayer_network,  prop_missing, obs_years, seed = 1234) {
  
  set.seed(seed)
  
  # Extract the dimensions of the input
  L = dim(multilayer_network)[1]
  Q = dim(multilayer_network)[2]
  T_len = dim(multilayer_network)[3]
  
  # Initialize the missing multilayer network to be the original multilayer network
  missing_multilayer_network = multilayer_network
  
  
  # Making a grid of the edge and time indices corresponding to each layer
  edge_time_grid = expand.grid(edges = 1:Q, times = 1:T_len)
  
  n_edge_time_combinations = nrow(edge_time_grid)
  
  
  # Number of edges to be removed on each layer
  n_missing = round(prop_missing * Q * T_len, digits = 0)

  
  # Storing the years that end up containing missing edges 
  missing_years = list()
  
  for (l in 1:L) {
    
    # Sample the edge and time combinations for the missing edges on the current layer
    cur_missing_idxs = sample(c(1:n_edge_time_combinations), size = n_missing, replace = FALSE)
    

    
    # Extract these from the edge and time grid
    cur_missing_edges = edge_time_grid[cur_missing_idxs, ]
    
    for (i in 1:nrow(cur_missing_edges)) {
      
      missing_multilayer_network[l, cur_missing_edges[i, 1], cur_missing_edges[i, 2]] = NA  
      
    }
    
    
    missing_years = append(x = missing_years, values = unique(obs_years[cur_missing_edges[, 2]]))
    
  }
  
  missing_years = unique(unlist(missing_years, use.names = FALSE)) 
  
  
  
  
  out = list(missing_multilayer_network = missing_multilayer_network,
             missing_years = sort(missing_years))
  
}

### Function: random_edge_removal

### Parameters:
# multilayer_network: An array of dimension L x V(V - 1) / 2 x T, containing the upper triangular portions
#                     of the networks on each layer and for each time point
# time_props_missing: A length L vector of values between 0 and 1, representing the proportion of times to be made missing on each layer
# edge_props_missing: A L vector of values between 0 and 1, representing the proportion of edges to be made missing on each layer for each missing time
# obs_years: A vector containing the years that were observed

### Returns:
# missing_multilayer_network: An array of dimension L x V(V - 1) / 2 x T, containing the upper triangular portions
#                             of the networks on each layer and for each time point, now with values missing
# missing_years: A vector of the years that contain missing values
random_edge_removal_diff_time_and_edge_props <- function(multilayer_network, time_props_missing,
                                                         edge_props_missing,
                                                         obs_years, 
                                                         remove_last_point = FALSE,
                                                         seed = 1234) {
  
  set.seed(seed)
  
  # Extract the dimensions of the input
  L = dim(multilayer_network)[1]
  Q = dim(multilayer_network)[2]
  T_len = dim(multilayer_network)[3]
  
  # Initialize the missing multilayer network to be the original multilayer network
  missing_multilayer_network = multilayer_network
  
  # Storing the years that end up containing missing edges 
  missing_years = list()
  
  if (all(time_props_missing == 0) & all(edge_props_missing == 0)) {
    
    
    out = list(missing_multilayer_network = missing_multilayer_network,
               missing_years = missing_years)
    
    return(out)
    
  }
  
  if (remove_last_point) {
    
    missing_years = append(x = missing_years, values = T_len)
    
  }
  
  for (l in 1:L) {
    
    
    # Number of edges to be removed on each layer
    n_times_missing = round(time_props_missing[l] * T_len, digits = 0)
    n_edges_missing = round(edge_props_missing[l] * Q, digits = 0)
    
    
    
    # Sample the edge and time combinations for the missing edges on the current layer
    
    if (remove_last_point) {
      
      cur_missing_time_idxs = sample(c(1:(T_len - 1)), size = n_times_missing - 1, replace = FALSE)
      
    }
    
    else {
      
      cur_missing_time_idxs = sample(c(1:T_len), size = n_times_missing, replace = FALSE)
      
    }
    
    missing_years = append(x = missing_years, values = cur_missing_time_idxs)
    
    for (t in 1:length(cur_missing_time_idxs)) {
      
      cur_missing_edge_idxs = sample(c(1:Q), size = n_edges_missing, replace = FALSE)
      
      # Extract these from the edge and time grid
      for (e in 1:length(cur_missing_edge_idxs)) {
        
        missing_multilayer_network[l, 
                                   cur_missing_edge_idxs[e], 
                                   cur_missing_time_idxs[t]] = NA  
        
      }
      
    }
    
  }
  
  missing_years = unique(unlist(missing_years, use.names = FALSE)) 
  
  
  
  out = list(missing_multilayer_network = missing_multilayer_network,
             missing_years = sort(missing_years))
  
  return(out)
  
}



# Function for standardizing nodal attributes
standardize_nodal_attributes <- function(x_arr, train_times, test_times, cat_features = NULL, binary_features = NULL) {
  
  x_dim = dim(x_arr)
  
  std_x_arr = array(NA, dim = x_dim)
  
  
  # Split the data into the train and test sets
  x_arr_train = x_arr[, , train_times, drop = FALSE]
  x_arr_test = x_arr[, , test_times, drop = FALSE]
  
  # Create stacked versions of the train and test datasets
  stacked_x_arr_train = matrix(NA, nrow = length(train_times) * x_dim[1], x_dim[2])
  stacked_x_arr_test = matrix(NA, nrow = length(test_times) * x_dim[1], x_dim[2])
  
  for (t in 1:length(train_times)) {
    
    stacked_x_arr_train[(((t - 1) * x_dim[1]) + 1):(t * x_dim[1]), ] = x_arr_train[, , t]
    
  }
  
  
  for (t in 1:length(test_times)) {
    
    stacked_x_arr_test[(((t - 1) * x_dim[1]) + 1):(t * x_dim[1]), ] = x_arr_test[, , t]
    
  }
  

  
  # If there are categorical features
  if (!is.null(cat_features)) {
    
    # Extract binary features that do not need processing
    stacked_x_arr_train_bin = stacked_x_arr_train[, binary_features, drop = FALSE]
    stacked_x_arr_test_bin = stacked_x_arr_test[, binary_features, drop = FALSE]
    
    # Extract numeric features and standardize on the train and test sets
    stacked_x_arr_train_num = stacked_x_arr_train[, -c(cat_features, binary_features), drop = FALSE]
    stacked_x_arr_test_num = stacked_x_arr_test[, -c(cat_features, binary_features), drop = FALSE]
    
    
    train_means = colMeans(stacked_x_arr_train_num)
    train_sd = apply(stacked_x_arr_train_num, 2, "sd")
    
    centered_train = t(t(stacked_x_arr_train_num) - train_means)
    centered_test = t(t(stacked_x_arr_test_num) - train_means)
    
    std_stacked_x_arr_train_num = t(t(centered_train) / train_sd)
    std_stacked_x_arr_test_num = t(t(centered_test) / train_sd)
    
    
    
    # Perform one-hot encoding of categorical features on the train and test sets
    stacked_x_arr_train_cat = stacked_x_arr_train[, cat_features, drop = FALSE]
    stacked_x_arr_test_cat = stacked_x_arr_test[, cat_features, drop = FALSE]
    
    
    
    # Convert to factor type for dummyVars function
    stacked_x_arr_train_cat = apply(stacked_x_arr_train_cat, 2, "as.factor")
    stacked_x_arr_test_cat = apply(stacked_x_arr_test_cat, 2, "as.factor")
    
    dummy_fit = dummyVars(~ . , data = stacked_x_arr_train_cat, fullRank = TRUE)
    
    std_stacked_x_arr_train_cat = predict(dummy_fit, newdata = stacked_x_arr_train_cat)
    std_stacked_x_arr_test_cat = predict(dummy_fit, newdata = stacked_x_arr_test_cat)
    
    
    
    # Combine the standardized numerical and categorical train features
    stacked_x_arr_train_std = cbind(std_stacked_x_arr_train_num, std_stacked_x_arr_train_cat, 
                                    stacked_x_arr_train_bin)
    stacked_x_arr_test_std = cbind(std_stacked_x_arr_test_num, std_stacked_x_arr_test_cat, 
                                   stacked_x_arr_test_bin)
    
    new_m = dim(stacked_x_arr_train_std)[2]
    
  }
  
  # If all of the features are numeric
  else if (is.null(cat_features) & is.null(binary_features)) {
    
    train_means = colMeans(stacked_x_arr_train)
    train_sd = apply(stacked_x_arr_train, 2, "sd")
    
    centered_train = t(t(stacked_x_arr_train) - train_means)
    centered_test = t(t(stacked_x_arr_test) - train_means)
    
    stacked_x_arr_train_std = t(t(centered_train) / train_sd)
    stacked_x_arr_test_std = t(t(centered_test) / train_sd)
    
    # Dimension does not change in this case
    new_m = x_dim[2]
    
  }
  
  # Do nothing except train test split (placeholder)
  else {
    
    x_out = list(x_arr_train = x_arr[, , train_times, drop = FALSE],
                 x_arr_test = x_arr[, , test_times, drop = FALSE])
    
    return(x_out)
    
  }
  

  
  # Unstack the standardized data
  std_x_arr_train = array(NA, dim = c(x_dim[1], new_m, length(train_times)))
  std_x_arr_test = array(NA, dim = c(x_dim[1], new_m, length(test_times)))
  
  for (t in 1:length(train_times)) {
    
    std_x_arr_train[, , t] = stacked_x_arr_train_std[(((t - 1) * x_dim[1]) + 1):(t * x_dim[1]), ]
    
  }
  
  for (t in 1:length(test_times)) {
    
    std_x_arr_test[, , t] = stacked_x_arr_test_std[(((t - 1) * x_dim[1]) + 1):(t * x_dim[1]), ]
    
  }
  
  
  
  x_out = list(std_x_arr_train = std_x_arr_train, 
               std_x_arr_test = std_x_arr_test)
  
  return(x_out)
  
  
}





sigma2_mu_mh_update <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                time_grid, sigma2_b_old, sigma2_w_old, nngp_L, b_proposal_var = 1,
                                w_proposal_var = 1) {
  
  sigma2_b_proposal = exp(rnorm(1, mean = log(sigma2_b_old), sd = sqrt(b_proposal_var)))
  sigma2_w_proposal = exp(rnorm(1, mean = log(sigma2_w_old), sd = sqrt(w_proposal_var)))
  
  
  # Update sigma2_mu_b
  sigma2_b_old_likelihood = nngp_log_likelihood_mu(mu_series = mu_series, 
                                                   zeta_series = zeta_series,
                                                   xi_series = xi_series,
                                                   omega_mats = omega_mats,
                                                   y_adj_mats = y_adj_mats,
                                                   time_grid = time_grid, 
                                                   sigma2_w = sigma2_w_old,
                                                   sigma2_b = sigma2_b_old,
                                                   nngp_L = nngp_L)
  
  sigma2_b_proposal_likelihood = nngp_log_likelihood_mu(mu_series = mu_series, 
                                                        zeta_series = zeta_series,
                                                        xi_series = xi_series,
                                                        omega_mats = omega_mats,
                                                        y_adj_mats = y_adj_mats,
                                                        time_grid = time_grid, 
                                                        sigma2_w = sigma2_w_old,
                                                        sigma2_b = sigma2_b_proposal,
                                                        nngp_L = nngp_L)
  
  
  acceptance_ratio_b = exp(sigma2_b_proposal_likelihood) / exp(sigma2_b_old_likelihood) * (sigma2_b_old / sigma2_b_proposal)
  
  prob_val = runif(1)
  
  
  if (is.nan(acceptance_ratio_b)) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else if (acceptance_ratio_b > prob_val) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else {
    
    sigma2_b_updated = sigma2_b_old
    
  }
  
  # Update sigma2_mu_w
  sigma2_w_old_likelihood = nngp_log_likelihood_mu(mu_series = mu_series, 
                                                   zeta_series = zeta_series,
                                                   xi_series = xi_series,
                                                   omega_mats = omega_mats,
                                                   y_adj_mats = y_adj_mats,
                                                   time_grid = time_grid, 
                                                   sigma2_w = sigma2_w_old,
                                                   sigma2_b = sigma2_b_updated,
                                                   nngp_L = nngp_L)
  
  sigma2_w_proposal_likelihood = nngp_log_likelihood_mu(mu_series = mu_series, 
                                                        zeta_series = zeta_series,
                                                        xi_series = xi_series,
                                                        omega_mats = omega_mats,
                                                        y_adj_mats = y_adj_mats,
                                                        time_grid = time_grid, 
                                                        sigma2_w = sigma2_w_proposal,
                                                        sigma2_b = sigma2_b_updated,
                                                        nngp_L = nngp_L)
  
  
  acceptance_ratio_w = exp(sigma2_w_proposal_likelihood) / exp(sigma2_w_old_likelihood) * (sigma2_w_old / sigma2_w_proposal)
  
  prob_val = runif(1)
  
  if (is.nan(acceptance_ratio_w)) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  
  else if (acceptance_ratio_w > prob_val) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else {
    
    sigma2_w_updated = sigma2_w_old
    
  }
  
  
  out_list = list(sigma2_b = sigma2_b_updated,
                  sigma2_w = sigma2_w_updated)
  
  return(out_list)
  
}




sigma2_xi_mh_update <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                x_arr, eta_series, alpha_series, sigma2_k,
                                time_grid, sigma2_b_old, sigma2_w_old, nngp_L, b_proposal_var = 1,
                                w_proposal_var = 1) {
  
  sigma2_b_proposal = exp(rnorm(1, mean = log(sigma2_b_old), sd = sqrt(b_proposal_var)))
  sigma2_w_proposal = exp(rnorm(1, mean = log(sigma2_w_old), sd = sqrt(w_proposal_var)))
  
  
  
  # Update sigma2_xi_b
  sigma2_b_old_likelihood = nngp_log_likelihood_xi(mu_series = mu_series, 
                                                   zeta_series = zeta_series,
                                                   xi_series = xi_series,
                                                   omega_mats = omega_mats,
                                                   y_adj_mats = y_adj_mats,
                                                   x_arr = x_arr, 
                                                   eta_series = eta_series,
                                                   alpha_series = alpha_series,
                                                   sigma2_k = sigma2_k,
                                                   time_grid = time_grid,  
                                                   sigma2_w = sigma2_w_old,
                                                   sigma2_b = sigma2_b_old,
                                                   nngp_L = nngp_L)
  
  sigma2_b_proposal_likelihood = nngp_log_likelihood_xi(mu_series = mu_series, 
                                                        zeta_series = zeta_series,
                                                        xi_series = xi_series,
                                                        omega_mats = omega_mats,
                                                        y_adj_mats = y_adj_mats,
                                                        x_arr = x_arr, 
                                                        eta_series = eta_series,
                                                        alpha_series = alpha_series,
                                                        sigma2_k = sigma2_k,
                                                        time_grid = time_grid, 
                                                        sigma2_w = sigma2_w_old,
                                                        sigma2_b = sigma2_b_proposal,
                                                        nngp_L = nngp_L)
  
  
  acceptance_ratio_b = exp(sigma2_b_proposal_likelihood) / exp(sigma2_b_old_likelihood) * (sigma2_b_old / sigma2_b_proposal)
  
  prob_val = runif(1)
  
  
  if (is.nan(acceptance_ratio_b)) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else if (acceptance_ratio_b > prob_val) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else {
    
    sigma2_b_updated = sigma2_b_old
    
  }
  
  # Update sigma2_xi_w
  sigma2_w_old_likelihood = nngp_log_likelihood_xi(mu_series = mu_series, 
                                                   zeta_series = zeta_series,
                                                   xi_series = xi_series,
                                                   omega_mats = omega_mats,
                                                   y_adj_mats = y_adj_mats,
                                                   x_arr = x_arr, 
                                                   eta_series = eta_series,
                                                   alpha_series = alpha_series,
                                                   sigma2_k = sigma2_k,
                                                   time_grid = time_grid,  
                                                   sigma2_w = sigma2_w_old,
                                                   sigma2_b = sigma2_b_updated,
                                                   nngp_L = nngp_L)
  
  sigma2_w_proposal_likelihood = nngp_log_likelihood_xi(mu_series = mu_series, 
                                                        zeta_series = zeta_series,
                                                        xi_series = xi_series,
                                                        omega_mats = omega_mats,
                                                        y_adj_mats = y_adj_mats,
                                                        x_arr = x_arr, 
                                                        eta_series = eta_series,
                                                        alpha_series = alpha_series,
                                                        sigma2_k = sigma2_k,
                                                        time_grid = time_grid, 
                                                        sigma2_w = sigma2_w_proposal,
                                                        sigma2_b = sigma2_b_updated,
                                                        nngp_L = nngp_L)
  
  
  acceptance_ratio_w = exp(sigma2_w_proposal_likelihood) / exp(sigma2_w_old_likelihood) * (sigma2_w_old / sigma2_w_proposal)
  
  prob_val = runif(1)
  
  if (is.nan(acceptance_ratio_w)) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else if (acceptance_ratio_w > prob_val) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else {
    
    sigma2_w_updated = sigma2_w_old
    
  }
  
  
  out_list = list(sigma2_b = sigma2_b_updated,
                  sigma2_w = sigma2_w_updated)
  
  return(out_list)
  
}

sigma2_xi_net_mh_update <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                    x_arr, eta_series, alpha_series, sigma2_k,
                                    time_grid, sigma2_b_old, sigma2_w_old, nngp_L, b_proposal_var = 1,
                                    w_proposal_var = 1) {
  
  sigma2_b_proposal = exp(rnorm(1, mean = log(sigma2_b_old), sd = sqrt(b_proposal_var)))
  sigma2_w_proposal = exp(rnorm(1, mean = log(sigma2_w_old), sd = sqrt(w_proposal_var)))
  
  
  # Update sigma2_xi_b
  sigma2_b_old_likelihood = nngp_log_likelihood_xi_net(mu_series = mu_series, 
                                                       zeta_series = zeta_series,
                                                       xi_series = xi_series,
                                                       omega_mats = omega_mats,
                                                       y_adj_mats = y_adj_mats,
                                                       x_arr = x_arr, 
                                                       eta_series = eta_series,
                                                       alpha_series = alpha_series,
                                                       sigma2_k = sigma2_k,
                                                       time_grid = time_grid,  
                                                       sigma2_w = sigma2_w_old,
                                                       sigma2_b = sigma2_b_old,
                                                       nngp_L = nngp_L)
  
  sigma2_b_proposal_likelihood = nngp_log_likelihood_xi_net(mu_series = mu_series, 
                                                            zeta_series = zeta_series,
                                                            xi_series = xi_series,
                                                            omega_mats = omega_mats,
                                                            y_adj_mats = y_adj_mats,
                                                            x_arr = x_arr, 
                                                            eta_series = eta_series,
                                                            alpha_series = alpha_series,
                                                            sigma2_k = sigma2_k,
                                                            time_grid = time_grid, 
                                                            sigma2_w = sigma2_w_old,
                                                            sigma2_b = sigma2_b_proposal,
                                                            nngp_L = nngp_L)
  
  
  acceptance_ratio_b = exp(sigma2_b_proposal_likelihood) / exp(sigma2_b_old_likelihood) * (sigma2_b_old / sigma2_b_proposal)
  
  prob_val = runif(1)
  
  
  if (is.nan(acceptance_ratio_b)) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else if (acceptance_ratio_b > prob_val) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else {
    
    sigma2_b_updated = sigma2_b_old
    
  }
  
  # Update sigma2_xi_w
  sigma2_w_old_likelihood = nngp_log_likelihood_xi_net(mu_series = mu_series, 
                                                       zeta_series = zeta_series,
                                                       xi_series = xi_series,
                                                       omega_mats = omega_mats,
                                                       y_adj_mats = y_adj_mats,
                                                       x_arr = x_arr, 
                                                       eta_series = eta_series,
                                                       alpha_series = alpha_series,
                                                       sigma2_k = sigma2_k,
                                                       time_grid = time_grid,  
                                                       sigma2_w = sigma2_w_old,
                                                       sigma2_b = sigma2_b_updated,
                                                       nngp_L = nngp_L)
  
  sigma2_w_proposal_likelihood = nngp_log_likelihood_xi_net(mu_series = mu_series, 
                                                            zeta_series = zeta_series,
                                                            xi_series = xi_series,
                                                            omega_mats = omega_mats,
                                                            y_adj_mats = y_adj_mats,
                                                            x_arr = x_arr, 
                                                            eta_series = eta_series,
                                                            alpha_series = alpha_series,
                                                            sigma2_k = sigma2_k,
                                                            time_grid = time_grid, 
                                                            sigma2_w = sigma2_w_proposal,
                                                            sigma2_b = sigma2_b_updated,
                                                            nngp_L = nngp_L)
  
  
  acceptance_ratio_w = exp(sigma2_w_proposal_likelihood) / exp(sigma2_w_old_likelihood) * (sigma2_w_old / sigma2_w_proposal)
  
  prob_val = runif(1)
  
  if (is.nan(acceptance_ratio_w)) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  
  else if (acceptance_ratio_w > prob_val) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else {
    
    sigma2_w_updated = sigma2_w_old
    
  }
  
  
  out_list = list(sigma2_b = sigma2_b_updated,
                  sigma2_w = sigma2_w_updated)
  
  return(out_list)
  
}


sigma2_xi_node_mh_update <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                     x_arr, eta_series, alpha_series, sigma2_k,
                                     time_grid, sigma2_b_old, sigma2_w_old, nngp_L, b_proposal_var = 1,
                                     w_proposal_var = 1) {
  
  sigma2_b_proposal = exp(rnorm(1, mean = log(sigma2_b_old), sd = sqrt(b_proposal_var)))
  sigma2_w_proposal = exp(rnorm(1, mean = log(sigma2_w_old), sd = sqrt(w_proposal_var)))
  
  
  # Update sigma2_xi_b
  sigma2_b_old_likelihood = nngp_log_likelihood_xi_node(mu_series = mu_series, 
                                                        zeta_series = zeta_series,
                                                        xi_series = xi_series,
                                                        omega_mats = omega_mats,
                                                        y_adj_mats = y_adj_mats,
                                                        x_arr = x_arr, 
                                                        eta_series = eta_series,
                                                        alpha_series = alpha_series,
                                                        sigma2_k = sigma2_k,
                                                        time_grid = time_grid,  
                                                        sigma2_w = sigma2_w_old,
                                                        sigma2_b = sigma2_b_old,
                                                        nngp_L = nngp_L)
  
  sigma2_b_proposal_likelihood = nngp_log_likelihood_xi_node(mu_series = mu_series, 
                                                             zeta_series = zeta_series,
                                                             xi_series = xi_series,
                                                             omega_mats = omega_mats,
                                                             y_adj_mats = y_adj_mats,
                                                             x_arr = x_arr, 
                                                             eta_series = eta_series,
                                                             alpha_series = alpha_series,
                                                             sigma2_k = sigma2_k,
                                                             time_grid = time_grid, 
                                                             sigma2_w = sigma2_w_old,
                                                             sigma2_b = sigma2_b_proposal,
                                                             nngp_L = nngp_L)
  
  
  acceptance_ratio_b = exp(sigma2_b_proposal_likelihood) / exp(sigma2_b_old_likelihood) * (sigma2_b_old / sigma2_b_proposal)

  
  prob_val = runif(1)
  
  
  if (is.nan(acceptance_ratio_b)) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else if (acceptance_ratio_b > prob_val) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else {
    
    sigma2_b_updated = sigma2_b_old
    
  }
  
  # Update sigma2_xi_w
  sigma2_w_old_likelihood = nngp_log_likelihood_xi_node(mu_series = mu_series, 
                                                        zeta_series = zeta_series,
                                                        xi_series = xi_series,
                                                        omega_mats = omega_mats,
                                                        y_adj_mats = y_adj_mats,
                                                        x_arr = x_arr, 
                                                        eta_series = eta_series,
                                                        alpha_series = alpha_series,
                                                        sigma2_k = sigma2_k,
                                                        time_grid = time_grid,  
                                                        sigma2_w = sigma2_w_old,
                                                        sigma2_b = sigma2_b_updated,
                                                        nngp_L = nngp_L)
  
  sigma2_w_proposal_likelihood = nngp_log_likelihood_xi_node(mu_series = mu_series, 
                                                             zeta_series = zeta_series,
                                                             xi_series = xi_series,
                                                             omega_mats = omega_mats,
                                                             y_adj_mats = y_adj_mats,
                                                             x_arr = x_arr, 
                                                             eta_series = eta_series,
                                                             alpha_series = alpha_series,
                                                             sigma2_k = sigma2_k,
                                                             time_grid = time_grid, 
                                                             sigma2_w = sigma2_w_proposal,
                                                             sigma2_b = sigma2_b_updated,
                                                             nngp_L = nngp_L)
  
  
  acceptance_ratio_w = exp(sigma2_w_proposal_likelihood) / exp(sigma2_w_old_likelihood) * (sigma2_w_old / sigma2_w_proposal)
  
  prob_val = runif(1)
  
  if (is.nan(acceptance_ratio_w)) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else if (acceptance_ratio_w > prob_val) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else {
    
    sigma2_w_updated = sigma2_w_old
    
  }
  
  
  out_list = list(sigma2_b = sigma2_b_updated,
                  sigma2_w = sigma2_w_updated)
  
  return(out_list)
  
}


sigma2_zeta_mh_update <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                  time_grid, sigma2_b_old, sigma2_w_old, nngp_L, b_proposal_var = 1,
                                  w_proposal_var = 1) {
  
  sigma2_b_proposal = exp(rnorm(1, mean = log(sigma2_b_old), sd = sqrt(b_proposal_var)))
  sigma2_w_proposal = exp(rnorm(1, mean = log(sigma2_w_old), sd = sqrt(w_proposal_var)))
  
  
  # Update sigma2_zeta_b
  sigma2_b_old_likelihood = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                     zeta_series = zeta_series,
                                                     xi_series = xi_series,
                                                     omega_mats = omega_mats,
                                                     y_adj_mats = y_adj_mats,
                                                     time_grid = time_grid,  
                                                     sigma2_w = sigma2_w_old,
                                                     sigma2_b = sigma2_b_old,
                                                     nngp_L = nngp_L)
  
  sigma2_b_proposal_likelihood = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                          zeta_series = zeta_series,
                                                          xi_series = xi_series,
                                                          omega_mats = omega_mats,
                                                          y_adj_mats = y_adj_mats,
                                                          time_grid = time_grid, 
                                                          sigma2_w = sigma2_w_old,
                                                          sigma2_b = sigma2_b_proposal,
                                                          nngp_L = nngp_L)
  
  
  acceptance_ratio_b = exp(sigma2_b_proposal_likelihood) / exp(sigma2_b_old_likelihood) * (sigma2_b_old / sigma2_b_proposal)
  
  prob_val = runif(1)
  
  
  
  if (is.nan(acceptance_ratio_b)) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else if (acceptance_ratio_b > prob_val) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else {
    
    sigma2_b_updated = sigma2_b_old
    
  }
  
  # Update sigma2_zeta_w
  sigma2_w_old_likelihood = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                     zeta_series = zeta_series,
                                                     xi_series = xi_series,
                                                     omega_mats = omega_mats,
                                                     y_adj_mats = y_adj_mats,
                                                     time_grid = time_grid,  
                                                     sigma2_w = sigma2_w_old,
                                                     sigma2_b = sigma2_b_updated,
                                                     nngp_L = nngp_L)
  
  sigma2_w_proposal_likelihood = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                          zeta_series = zeta_series,
                                                          xi_series = xi_series,
                                                          omega_mats = omega_mats,
                                                          y_adj_mats = y_adj_mats,
                                                          time_grid = time_grid, 
                                                          sigma2_w = sigma2_w_proposal,
                                                          sigma2_b = sigma2_b_updated,
                                                          nngp_L = nngp_L)
  
  
  acceptance_ratio_w = exp(sigma2_w_proposal_likelihood) / exp(sigma2_w_old_likelihood) * (sigma2_w_old / sigma2_w_proposal)
  
  prob_val = runif(1)
  
  if (is.nan(acceptance_ratio_w)) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  
  else if (acceptance_ratio_w > prob_val) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else {
    
    sigma2_w_updated = sigma2_w_old
    
  }
  
  
  out_list = list(sigma2_b = sigma2_b_updated,
                  sigma2_w = sigma2_w_updated)
  
  return(out_list)
  
}



sigma2_zeta_mh_update <- function(mu_series, xi_series, zeta_series, omega_mats, y_adj_mats,
                                  time_grid, sigma2_b_old, sigma2_w_old, nngp_L, b_proposal_var = 1,
                                  w_proposal_var = 1) {
  
  sigma2_b_proposal = exp(rnorm(1, mean = log(sigma2_b_old), sd = sqrt(b_proposal_var)))
  sigma2_w_proposal = exp(rnorm(1, mean = log(sigma2_w_old), sd = sqrt(w_proposal_var)))
  
  
  # Update sigma2_zeta_b
  sigma2_b_old_likelihood = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                     zeta_series = zeta_series,
                                                     xi_series = xi_series,
                                                     omega_mats = omega_mats,
                                                     y_adj_mats = y_adj_mats,
                                                     time_grid = time_grid,  
                                                     sigma2_w = sigma2_w_old,
                                                     sigma2_b = sigma2_b_old,
                                                     nngp_L = nngp_L)
  
  sigma2_b_proposal_likelihood = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                          zeta_series = zeta_series,
                                                          xi_series = xi_series,
                                                          omega_mats = omega_mats,
                                                          y_adj_mats = y_adj_mats,
                                                          time_grid = time_grid, 
                                                          sigma2_w = sigma2_w_old,
                                                          sigma2_b = sigma2_b_proposal,
                                                          nngp_L = nngp_L)
  
  
  acceptance_ratio_b = exp(sigma2_b_proposal_likelihood) / exp(sigma2_b_old_likelihood) * (sigma2_b_old / sigma2_b_proposal)
  
  prob_val = runif(1)
  
  
  
  if (is.nan(acceptance_ratio_b)) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else if (acceptance_ratio_b > prob_val) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else {
    
    sigma2_b_updated = sigma2_b_old
    
  }
  
  # Update sigma2_zeta_w
  sigma2_w_old_likelihood = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                     zeta_series = zeta_series,
                                                     xi_series = xi_series,
                                                     omega_mats = omega_mats,
                                                     y_adj_mats = y_adj_mats,
                                                     time_grid = time_grid,  
                                                     sigma2_w = sigma2_w_old,
                                                     sigma2_b = sigma2_b_updated,
                                                     nngp_L = nngp_L)
  
  sigma2_w_proposal_likelihood = nngp_log_likelihood_zeta(mu_series = mu_series, 
                                                          zeta_series = zeta_series,
                                                          xi_series = xi_series,
                                                          omega_mats = omega_mats,
                                                          y_adj_mats = y_adj_mats,
                                                          time_grid = time_grid, 
                                                          sigma2_w = sigma2_w_proposal,
                                                          sigma2_b = sigma2_b_updated,
                                                          nngp_L = nngp_L)
  
  
  acceptance_ratio_w = exp(sigma2_w_proposal_likelihood) / exp(sigma2_w_old_likelihood) * (sigma2_w_old / sigma2_w_proposal)
  
  prob_val = runif(1)
  
  
  if (is.nan(acceptance_ratio_w)) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else if (acceptance_ratio_w > prob_val) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else {
    
    sigma2_w_updated = sigma2_w_old
    
  }
  
  
  out_list = list(sigma2_b = sigma2_b_updated,
                  sigma2_w = sigma2_w_updated)
  
  return(out_list)
  
}



sigma2_eta_mh_update <- function(xi_series, x_arr, eta_series, alpha_series, sigma2_k,
                                 time_grid, sigma2_b_old, sigma2_w_old, nngp_L, b_proposal_var = 1,
                                 w_proposal_var = 1) {
  
  sigma2_b_proposal = exp(rnorm(1, mean = log(sigma2_b_old), sd = sqrt(b_proposal_var)))
  sigma2_w_proposal = exp(rnorm(1, mean = log(sigma2_w_old), sd = sqrt(w_proposal_var)))
  
  
  # Update sigma2_eta_b
  sigma2_b_old_likelihood = nngp_log_likelihood_eta(xi_series = xi_series,
                                                    x_arr = x_arr, 
                                                    eta_series = eta_series,
                                                    alpha_series = alpha_series,
                                                    sigma2_k = sigma2_k,
                                                    time_grid = time_grid,  
                                                    sigma2_w = sigma2_w_old,
                                                    sigma2_b = sigma2_b_old,
                                                    nngp_L = nngp_L)
  
  sigma2_b_proposal_likelihood = nngp_log_likelihood_eta(xi_series = xi_series,
                                                         x_arr = x_arr, 
                                                         eta_series = eta_series,
                                                         alpha_series = alpha_series,
                                                         sigma2_k = sigma2_k,
                                                         time_grid = time_grid, 
                                                         sigma2_w = sigma2_w_old,
                                                         sigma2_b = sigma2_b_proposal,
                                                         nngp_L = nngp_L)
  
  
  acceptance_ratio_b = exp(sigma2_b_proposal_likelihood) / exp(sigma2_b_old_likelihood) * (sigma2_b_old / sigma2_b_proposal)
  
  prob_val = runif(1)
  
  
  if (is.nan(acceptance_ratio_b)) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else if (acceptance_ratio_b > prob_val) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else {
    
    sigma2_b_updated = sigma2_b_old
    
  }
  
  # Update sigma2_eta_w
  sigma2_w_old_likelihood = nngp_log_likelihood_eta(xi_series = xi_series,
                                                    x_arr = x_arr, 
                                                    eta_series = eta_series,
                                                    alpha_series = alpha_series,
                                                    sigma2_k = sigma2_k,
                                                    time_grid = time_grid,  
                                                    sigma2_w = sigma2_w_old,
                                                    sigma2_b = sigma2_b_updated,
                                                    nngp_L = nngp_L)
  
  sigma2_w_proposal_likelihood = nngp_log_likelihood_eta(xi_series = xi_series,
                                                         x_arr = x_arr, 
                                                         eta_series = eta_series,
                                                         alpha_series = alpha_series,
                                                         sigma2_k = sigma2_k,
                                                         time_grid = time_grid, 
                                                         sigma2_w = sigma2_w_proposal,
                                                         sigma2_b = sigma2_b_updated,
                                                         nngp_L = nngp_L)
  
  
  acceptance_ratio_w = exp(sigma2_w_proposal_likelihood) / exp(sigma2_w_old_likelihood) * (sigma2_w_old / sigma2_w_proposal)
  
  prob_val = runif(1)
  
  
  if (is.nan(acceptance_ratio_w)) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else if (acceptance_ratio_w > prob_val) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else {
    
    sigma2_w_updated = sigma2_w_old
    
  }
  
  
  out_list = list(sigma2_b = sigma2_b_updated,
                  sigma2_w = sigma2_w_updated)
  
  return(out_list)
  
}



sigma2_alpha_mh_update <- function(xi_series, x_arr, eta_series, alpha_series, sigma2_k,
                                   time_grid, sigma2_b_old, sigma2_w_old, nngp_L, b_proposal_var = 1,
                                   w_proposal_var = 1) {
  
  sigma2_b_proposal = exp(rnorm(1, mean = log(sigma2_b_old), sd = sqrt(b_proposal_var)))
  sigma2_w_proposal = exp(rnorm(1, mean = log(sigma2_w_old), sd = sqrt(w_proposal_var)))
  
  
  # Update sigma2_alpha_b
  sigma2_b_old_likelihood = nngp_log_likelihood_alpha(xi_series = xi_series,
                                                      x_arr = x_arr, 
                                                      eta_series = eta_series,
                                                      alpha_series = alpha_series,
                                                      sigma2_k = sigma2_k,
                                                      time_grid = time_grid,  
                                                      sigma2_w = sigma2_w_old,
                                                      sigma2_b = sigma2_b_old,
                                                      nngp_L = nngp_L)
  
  sigma2_b_proposal_likelihood = nngp_log_likelihood_alpha(xi_series = xi_series,
                                                           x_arr = x_arr, 
                                                           eta_series = eta_series,
                                                           alpha_series = alpha_series,
                                                           sigma2_k = sigma2_k,
                                                           time_grid = time_grid, 
                                                           sigma2_w = sigma2_w_old,
                                                           sigma2_b = sigma2_b_proposal,
                                                           nngp_L = nngp_L)
  
  
  acceptance_ratio_b = exp(sigma2_b_proposal_likelihood) / exp(sigma2_b_old_likelihood) * (sigma2_b_old / sigma2_b_proposal)
  
  prob_val = runif(1)
  
  
  
  if (is.nan(acceptance_ratio_b)) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else if (acceptance_ratio_b > prob_val) {
    
    sigma2_b_updated = sigma2_b_proposal
    
  }
  
  else {
    
    sigma2_b_updated = sigma2_b_old
    
  }
  
  # Update sigma2_alpha_w
  sigma2_w_old_likelihood = nngp_log_likelihood_alpha(xi_series = xi_series,
                                                      x_arr = x_arr, 
                                                      eta_series = eta_series,
                                                      alpha_series = alpha_series,
                                                      sigma2_k = sigma2_k,
                                                      time_grid = time_grid,  
                                                      sigma2_w = sigma2_w_old,
                                                      sigma2_b = sigma2_b_updated,
                                                      nngp_L = nngp_L)
  
  sigma2_w_proposal_likelihood = nngp_log_likelihood_alpha(xi_series = xi_series,
                                                           x_arr = x_arr, 
                                                           eta_series = eta_series,
                                                           alpha_series = alpha_series,
                                                           sigma2_k = sigma2_k,
                                                           time_grid = time_grid, 
                                                           sigma2_w = sigma2_w_proposal,
                                                           sigma2_b = sigma2_b_updated,
                                                           nngp_L = nngp_L)
  
  
  acceptance_ratio_w = exp(sigma2_w_proposal_likelihood) / exp(sigma2_w_old_likelihood) * (sigma2_w_old / sigma2_w_proposal)
  
  prob_val = runif(1)
  
  if (is.nan(acceptance_ratio_w)) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  
  else if (acceptance_ratio_w > prob_val) {
    
    sigma2_w_updated = sigma2_w_proposal
    
  }
  
  else {
    
    sigma2_w_updated = sigma2_w_old
    
  }
  
  
  out_list = list(sigma2_b = sigma2_b_updated,
                  sigma2_w = sigma2_w_updated)
  
  return(out_list)
  
}









