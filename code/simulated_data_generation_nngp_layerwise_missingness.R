
library(MCMCpack)
library(gdata)
library(mvtnorm)

source("model_development.R")

# Function to generate linear time series for the simulation data
# gen_linear_time_series <- function(T_len, phi, epsilon) {
#   
#   series_vals = arima.sim(model = list(ar = phi), sd = sqrt(epsilon), n = T_len)
#   
#   return(series_vals)
# }

gen_nngp_time_series <- function(time_grid, nngp_L = 1, sigma2_w = 1, sigma2_b = 1) {
  
  T_len = length(time_grid)
  
  # Initialize the covariance kernel for the parameters
  cov_kernel = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w,
                                              sigma2_b = sigma2_b, nngp_L = nngp_L)

  
  param_series = rmvnorm(1, mean = rep(0, T_len), sigma = cov_kernel)
  
    
}



inverse_logit <- function(x) {
  
  output = exp(x) / (1 + exp(x))
  
  return(output)
  
}





# Function to generate incomplete graphs
# 
# Arguments - 
# L: Number of layers
# J: Number of nodes on each layer
# m: Number of node covariates
# T_len: Number of time points
# R: True dimension of the latent effects
# seed: random seed to be used
# 
#
#
# Returns -
# C_arr: L by J(J-1)/2 by T_len matrix containing the upper-triangular portions of the adjacency matrices
# on each layer for each time point
# masked_C_arr: C_arr with values missing randomly across each layer
# x_arr: J by m by T_len array of node covariates at each time

incomplete_graphs_sim_data_gen_nngp_layerwise_missingness <- function(L, J, m, T_len, R,
                                                                      time_grid = NULL,
                                                                      time_props_missing,
                                                                      edge_props_missing, 
                                                                      nngp_L = 1,
                                                                      sigma2_w = 1, sigma2_b = 1,
                                                                      remove_last_point = FALSE,
                                           seed = 1234) {
  
  # sigma2_b_mu = 0.1, sigma2_w_mu = 1.1, 
  # sigma2_b_xi = 0.1, sigma2_w_xi = 1.1,
  # sigma2_b_zeta = 0.1, sigma2_w_zeta = 1.1,
  # sigma2_b_eta = 0.1, sigma2_w_eta = 1.1,
  # sigma2_b_alpha = 0.1, sigma2_w_alpha = 1.1,
  
  set.seed(seed)
  
  if (is.null(time_grid)) {
    time_grid = c(1:T_len)
    
    #time_grid = seq(from = 0, to = 1, length.out = T_len)
    
  }
  
  else {
    
    T_len = length(time_grid)
    
  }
  
  # Generate the latent effects and other parameters from an NNGP using the fixed weight and bias variances
  
  # For each set of parameters, generate it as a linear time series (TENTATIVE may generate in another way) 
  
  # mu
  mu_series = gen_nngp_time_series(time_grid = time_grid, nngp_L = nngp_L, sigma2_w = sigma2_w,
                                   sigma2_b = sigma2_b)
  
  # xi
  xi_series = array(NA, dim = c(J, R, T_len, L))
  
  for (j in 1:J) {
    for (r in 1:R) {
      for (l in 1:L) {
        
        xi_series[j, r, , l] = gen_nngp_time_series(time_grid = time_grid, nngp_L = nngp_L, sigma2_w = sigma2_w,
                                                    sigma2_b = sigma2_b)
      }
    }
  }
  
  # zeta
  zeta_series = array(NA, dim = c(J, R, T_len))
  
  for (j in 1:J) {
    for (r in 1:R) {
      
      
      zeta_series[j, r, ] = gen_nngp_time_series(time_grid = time_grid, nngp_L = nngp_L, sigma2_w = sigma2_w,
                                                 sigma2_b = sigma2_b)
      
    }
  }
  
  # eta
  eta_series = matrix(NA, nrow = m, ncol = T_len)
  
  for (k in 1:m) {
    
    
    eta_series[k, ] = gen_nngp_time_series(time_grid = time_grid, nngp_L = nngp_L, sigma2_w = sigma2_w,
                                           sigma2_b = sigma2_b)
    
  }
  
  # alpha
  alpha_series = array(NA, dim = c(m, R, T_len, L))
  
  for (k in 1:m) {
    for (r in 1:R) {
      for (l in 1:L) {
        
        
        alpha_series[k, r, , l] = gen_nngp_time_series(time_grid = time_grid, nngp_L = nngp_L, sigma2_w = sigma2_w,
                                                       sigma2_b = sigma2_b)
        
      }
    }
  }
  
  
  # Generate variances for node covariate from IG(1, 1)
  sigma2_k_vals = rinvgamma(m, 1, 1)
  
  # Array to store the upper triangular portion of the adjacency matrices for each layer at each time
  C_probs_adj = array(NA, dim = c(L, J * (J - 1) / 2, T_len))
  C_arr = array(NA, dim = c(L, J * (J - 1) / 2, T_len))
  
  for (l in 1:L) {
    
    for (t in 1:T_len) {
      
      placeholder_adj_mat = matrix(NA, nrow = J, ncol = J)
      placeholder_C_mat = matrix(NA, nrow = J, ncol = J)
      
      for (j in 1:(J-1)) {
        
        for (j_prime in (j+1):J) {
          
          linear_fun = (mu_series[t] + 
                          as.numeric(crossprod(xi_series[j, , t, l], xi_series[j_prime, , t, l])) + 
                          as.numeric(crossprod(zeta_series[j, , t], zeta_series[j_prime, , t])))
          
          placeholder_adj_mat[j, j_prime] = inverse_logit(linear_fun)
          placeholder_C_mat[j, j_prime] = rbinom(n = 1, size = 1, prob = placeholder_adj_mat[j, j_prime])
          
          #print(placeholder_adj_mat)
        }
      }
      
      C_probs_adj[l, , t] = upperTriangle(placeholder_adj_mat, diag = FALSE, byrow = TRUE)
      
      C_arr[l, , t] = upperTriangle(placeholder_C_mat, diag = FALSE, byrow = TRUE)
      
    }
  }
  
  
  # Array to store the node covariates for each time point
  x_arr = array(NA, dim = c(J, m, T_len))
  
  for (j in 1:J) {
    
    for (k in 1:m) {
      
      for (t in 1:T_len) {
        
        latent_effect_sum = 0
        
        for(l in 1:L) {
          
          latent_effect_sum = latent_effect_sum + as.numeric(crossprod(xi_series[j, , t, l], 
                                                                       alpha_series[k, , t, l]))
          
        }
        
        x_arr[j, k, t] = eta_series[m, t] + latent_effect_sum + rnorm(1, sd = sqrt(sigma2_k_vals[k]))
        
      }
      
    }
    
  }
  
  # Adding random missingness to adjacency matrices for each layer across time
  random_edge_removal_output = random_edge_removal_diff_time_and_edge_props(multilayer_network = C_arr,
                                                                            time_props_missing = time_props_missing,
                                                                            edge_props_missing = edge_props_missing, 
                                                   obs_years = time_grid,
                                                   remove_last_point = remove_last_point,
                                                   seed = seed)
  
  
  masked_C_arr = random_edge_removal_output$missing_multilayer_network
  
  missing_times = random_edge_removal_output$missing_years
  
  
  
  out = list(C_arr = C_arr, C_probs_adj = C_probs_adj,
             x_arr = x_arr, masked_C_arr = masked_C_arr, missing_times = missing_times)
  
}



# out = incomplete_graphs_sim_data_gen_nngp_layerwise_missingness(L = 2, J = 20, m = 2, T_len = 20, R = 4,
#                                                                 time_props_missing = c(0.1, 0.1),
#                                                                 edge_props_missing = c(0.25, 0.25),
#                                                                 nngp_L = 1, sigma2_w = 0.1,
#                                                                 sigma2_b = 0.01,
#                                                                 remove_last_point = TRUE)

# plot(out$x_arr[7, 1, ])
# 
# plot(out$C_probs_adj[1, 5, ])
# plot(out$C_arr[1, 5, ])
# 
# out$C_arr[1, 2, ]
# out$masked_C_arr[1, 2, ]
# 
# sum(is.na(out$masked_C_arr[1, , ]))
# sum(is.na(out$masked_C_arr[2, , ]))
# 
# length(out$missing_times)
# out$missing_times
