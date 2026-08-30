
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

incomplete_graphs_sim_data_gen_tergm_layerwise_missingness <- function(L, J, m, T_len, R,
                                                                       time_grid = NULL,
                                                                      time_props_missing,
                                                                      edge_props_missing, 
                                                                      nngp_L = 1,
                                                                      sigma2_w = 1, sigma2_b = 1,
                                                                      remove_last_point = FALSE,
                                                                      markov_iter = 500,
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
  # mu_series = gen_nngp_time_series(time_grid = time_grid, nngp_L = nngp_L, sigma2_w = sigma2_w,
  #                                  sigma2_b = sigma2_b)
  
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
  # zeta_series = array(NA, dim = c(J, R, T_len))
  # 
  # for (j in 1:J) {
  #   for (r in 1:R) {
  #     
  #     
  #     zeta_series[j, r, ] = gen_nngp_time_series(time_grid = time_grid, nngp_L = nngp_L, sigma2_w = sigma2_w,
  #                                                sigma2_b = sigma2_b)
  #     
  #   }
  # }
  
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
  
  
  # Series of values to multiply to make a_{jj', l}
  a_node_series = array(NA, dim = c(J, m, T_len, L))
  
  
  for (j in 1:J) {
    for (k in 1:m) {
      for (l in 1:L) {
        
        a_node_series[j, k, , l] = gen_nngp_time_series(time_grid = time_grid, nngp_L = nngp_L, sigma2_w = sigma2_w,
                                                    sigma2_b = sigma2_b)
      } 
    }
  }
  
  
  # Generate variances for node covariate from IG(1, 1)
  sigma2_k_vals = rinvgamma(m, 1, 1)
  
  
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
  

  
  # Array to store the upper triangular portion of the adjacency matrices for each layer at each time
  layer_and_time_probs = matrix(NA, nrow = L, ncol = T_len)
  
  theta_1_vals = c(1:T_len) / T_len
  
  theta_2 = 0.5
  
  
  C_arr = array(NA, dim = c(L, J * (J - 1) / 2, T_len))
  
  
  #markov_iter = 100
  
  
  
    
  
  
  for (l in 1:L) {
    
    for (t in 1:T_len) {
      
      # Initialize the graph and calculate its probability
      prev_C_adj_mat_upper_tri = array(rbinom(n = L * J * (J - 1)* T_len / 2 , size = 1, prob = 0.5),
                                      dim = c(L, J * (J - 1) / 2, T_len))
      
      prev_adj_mat = matrix(0, nrow = J, ncol = J)
      
      upperTriangle(prev_adj_mat, diag = FALSE, byrow = TRUE) = prev_C_adj_mat_upper_tri[l, , t]
      lowerTriangle(prev_adj_mat, diag = FALSE, byrow = FALSE) = prev_C_adj_mat_upper_tri[l, , t]
      
      
      S1_sum_term = 0
      S2_sum_term = 0
      
      
      for (j in 1:(J-1)) {
        for (j_prime in j:J) {
          for (k in 1:m) {  
          
            S1_sum_term = S1_sum_term + (prev_adj_mat[j, j_prime] * x_arr[j, k, t] * x_arr[j_prime, k, t]) / ((J ^ 2) * m)  # (J * (J - 1) * m / 2)
            
           
          }
          
          S2_sum_term = S2_sum_term + prev_adj_mat[j, j_prime] / (J ^ 2) # (J * (J - 1) / 2)
          
        }
      }
      
      linear_term = theta_1_vals[t] * S1_sum_term + theta_2 * S2_sum_term
      
      prev_prob_val = exp(linear_term)
      
      for (i in 1:markov_iter) {
      
        # Generate Candidate
        candidate_adj_mat = matrix(0, nrow = J, ncol = J)
  
        row_idx = sample(c(1:J), size = 1, replace = FALSE)
        col_idx = sample(c(1:J), size = 1, replace = FALSE)
        
        #print(row_idx)
        
        while (row_idx == col_idx) {
          
          col_idx = sample(c(1:J), size = 1, replace = FALSE)
          
        }

        candidate_adj_mat = prev_adj_mat
        
        if (candidate_adj_mat[row_idx, col_idx] == 1) {
          
          candidate_adj_mat[row_idx, col_idx] = 0
        }
        
        else {
          candidate_adj_mat[row_idx, col_idx] = 0
        }
        
        candidate_adj_mat[col_idx, row_idx] = candidate_adj_mat[row_idx, col_idx] 
        
        
        
        # Calculate Probability of Candidate
        S1_sum_term = 0
        S2_sum_term = 0
        

        for (j in 1:(J-1)) {
          for (j_prime in j:J) {
            for (k in 1:m) {
              
              S1_sum_term = S1_sum_term + (candidate_adj_mat[j, j_prime] * x_arr[j, k, t] * x_arr[j_prime, k, t]) / ((J ^ 2) * m) # (J * (J - 1) * m / 2)
              
              
            }
            
            S2_sum_term = S2_sum_term + candidate_adj_mat[j, j_prime] / (J ^ 2)  # (J * (J - 1) / 2)
            
          }
        }
        
        linear_term = theta_1_vals[t] * S1_sum_term + theta_2 * S2_sum_term
        
        candidate_prob_val = exp(linear_term)
        
        #print(prev_prob_val)
        #print(candidate_prob_val)
        
        # Accept or Reject Candidate
        acceptance_prob = candidate_prob_val / (candidate_prob_val + prev_prob_val)
        
        #print(acceptance_prob)
        # 
        # if (is.nan(acceptance_prob)) {
        #   
        #   acceptance_indicator = 0
        #   
        # }
        
        #else {
        acceptance_indicator = rbinom(1, size = 1, prob = acceptance_prob)
        #}
        
        if (acceptance_indicator == 1) {
         
          prev_adj_mat = candidate_adj_mat 
          prev_prob_val = candidate_prob_val
          
        }
        
      }
      
      C_arr[l, , t] = upperTriangle(prev_adj_mat, diag = FALSE, byrow = TRUE)
      
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
  
  #print(C_arr)
  
  
  out = list(C_arr = C_arr, 
             x_arr = x_arr, masked_C_arr = masked_C_arr, missing_times = missing_times)
  
  
  
}



# out = incomplete_graphs_sim_data_gen_tergm_layerwise_missingness(L = 2, J = 20, m = 8, T_len = 21, R = 4,
#                                                                 time_props_missing = c(0.1, 0.1),
#                                                                 edge_props_missing = c(0.25, 0.25),
#                                                                 nngp_L = 1, sigma2_w = 0.001,
#                                                                 sigma2_b = 0.0001,
#                                                                 remove_last_point = TRUE)

# plot(out$x_arr[7, 1, ])
# 
# #plot(out$C_probs_adj[1, 5, ])
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
