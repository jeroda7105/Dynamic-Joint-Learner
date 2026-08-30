library(MCMCpack)
library(gdata)
library(BayesLogit)
library(mvtnorm)

library(doParallel)
library(foreach)


source("model_development.R")


#### Function: dyna_hidden_graph_model_complete: Fits the model on given
#              dynamic multiplex graph C_arr, and nodal attributes x_arr
#
#### Arguments: 
#
# C_arr: Array representing the dynamic multiplex graph. Only the upper-triangular
# entries of the graph on each layer are stored here for each layer and time.
# It has dimension L x Q x T,  where Q = J * (J - 1) / 2. 
# Here J is the number of nodes composing the graph on each layer,
# L is the number of graph layers, and T is the number of time points.
# 
# x_arr: Array representing the dynamic nodal attributes. It has dimension
# J x m x T, J is the number of nodes composing the graph on each layer,
# m is the number of attributes observed for each node, and T is the
# number of time points.
# 
# time_grid: A vector containing the observed time values in the same order
# as the values corresponding to the times in C_arr and x_arr
# 
# R: Latent dimension for the layer-specific latent variables, xi and alpha 
# 
# R_zeta: Latent dimension for the latent variables shared across layers, zeta 
# 
# nngp_L: Represents F, the number of NN-GP layers used to fit DJL  
# 
# niter: The number of MCMC iterations to run the Gibbs sampler of DJL for 
# 
# seed: A random seed for fitting the model 
# 
# phi: Optional parameter used to identify results from simulations in scenario 2 
# 
# epsilon: Another optional parameter used to identify results from simulations in scenario 2 
# 
# True_R: Optional parameter used to store the true value of R for identification purposes
# 
# True_F: Optional parameter used to store the true value of R for identification purposes
# 
# nngp_gen: Optional parameter used to indicate whether scenario 1 is being used
# 
# tergm_gen: Optional parameter used to indicate whether scenario 3 is being used
# 
# real_data: Optional parameter used to indicate whether the BAAD2 real data is being used
# 
# time_props_missing = NULL: 
# 
# edge_props_missing = NULL: 
# 
# save_outputs: Boolean to indicate whether to save all of the MCMC outputs to
# a .RData file
# 
# run_parallel: Boolean to indicate whether to run updates to eta in parallel
# 
# n_workers: Number of cores to use if running in parallel
# 
# sigma2_b_grid: Vector of values representing potential values for the variances
# of the bias terms in the NN-GPs used to fit DJL
# 
# sigma2_w_grid: Vector of values representing potential values for the variances
# of the weight terms in the NN-GPs used to fit DJL
# 
# use_mh: Optional boolean to use Metropolis-Hastings to sample NN-GP parameters 
# instead of a uniform grid
# 
# b_proposal_var: Variance for the proposal distribution of the NN-GP bias terms if use_mh is true
# 
# w_proposal_var: Variance for the proposal distribution of the NN-GP weight terms if use_mh is true
#
#
#### Outputs
# model_outputs: List object containing MCMC samples for the DJL model parameters
# 


dyna_hidden_graph_model_complete <- function(C_arr, x_arr, time_grid, R = 4, R_zeta = R, nngp_L = 1, 
                                             niter = 100, seed = 1234,
                                             phi = NULL,
                                             epsilon = NULL,
                                             True_R = NULL,
                                             True_F = NULL,
                                             nngp_gen = NULL,
                                             tergm_gen = NULL,
                                             real_data = NULL,
                                             time_props_missing = NULL,
                                             edge_props_missing = NULL,
                                             save_outputs = FALSE,
                                             run_parallel = FALSE,
                                             n_workers = NULL,
                                             sigma2_b_grid = seq(0.01, 0.1, by = 0.01),
                                             sigma2_w_grid = seq(0.01, 0.1, by = 0.01),
                                             use_mh = FALSE,
                                             b_proposal_var = 0.1,
                                             w_proposal_var = 0.1) {
  
  
  
  set.seed(seed)
  
  
  # Setup for running node attribute updates in parallel if run_parallel is true
  if (run_parallel) {
    
    if (is.null(n_workers)) {
      n_workers = detectCores()
    }
    
    cl <- makeCluster(n_workers) 
    registerDoParallel(cl) 
  }
  
  # Get number of nodes 
  J = dim(x_arr)[1]

  
  # Get number of layers
  L = dim(C_arr)[1]

  
  # Get number of time points
  T_len = length(time_grid)

  
  
  # Get number of node covariates
  m = dim(x_arr)[2]

  
  
  
  # Get array of indicators of whether each edge is observed
  C_obs_indicators = array(as.numeric(!is.na(C_arr)), dim = dim(C_arr))
  
  
  # Initialize parameters
  
  # sigma2_k for k = 1,...,m
  a_sigma = 1
  b_sigma = 1
  
  sigma2_k = rinvgamma(m, shape = a_sigma, scale = b_sigma)
  
  # Sample the omega value corresponding to each upper-triangular edge
  omega = array(rpg(num = L * J * (J - 1) * T_len / 2, h = 1, z = 0.0),
                dim = dim(C_arr))
  
  # Sample the time-varying mean vector
  mu_series = rnorm(T_len, mean = 0, sd = 1)
  
  # Sample the latent effects for each node
  xi_series = array(0,
                    dim = c(J, R, T_len, L))
  
  zeta_series = array(0,
                      dim = c(J, R_zeta, T_len))
  
  alpha_series = array(0,
                       dim = c(m, R, T_len, L))
  
  
  # Sample the time varying intercept for each node covariate
  eta_series = matrix(0,
                      nrow = m, ncol = T_len)
  
  

  
  if (use_mh) {
    
    sigma2_b_mu = exp(rnorm(1, sd = sqrt(b_proposal_var)))
    sigma2_w_mu = exp(rnorm(1, sd = sqrt(w_proposal_var)))
    
    sigma2_b_xi = exp(rnorm(1, sd = sqrt(b_proposal_var)))
    sigma2_w_xi = exp(rnorm(1, sd = sqrt(w_proposal_var)))
    
    sigma2_b_zeta = exp(rnorm(1, sd = sqrt(b_proposal_var)))
    sigma2_w_zeta = exp(rnorm(1, sd = sqrt(w_proposal_var)))
    
    
    sigma2_b_eta = exp(rnorm(1, sd = sqrt(b_proposal_var)))
    sigma2_w_eta = exp(rnorm(1, sd = sqrt(w_proposal_var)))
    
    sigma2_b_alpha = exp(rnorm(1, sd = sqrt(b_proposal_var)))
    sigma2_w_alpha = exp(rnorm(1, sd = sqrt(w_proposal_var)))
    
    
  }
  
  else {
    sigma2_b_mu_grid = sigma2_b_grid
    sigma2_w_mu_grid = sigma2_w_grid 
    
    sigma2_b_xi_grid = sigma2_b_grid
    sigma2_w_xi_grid = sigma2_w_grid
    
    sigma2_b_zeta_grid = sigma2_b_grid
    sigma2_w_zeta_grid = sigma2_w_grid
    
    
    sigma2_b_eta_grid = sigma2_b_grid
    sigma2_w_eta_grid = sigma2_w_grid
    
    sigma2_b_alpha_grid = sigma2_b_grid
    sigma2_w_alpha_grid = sigma2_w_grid
    
    
    
    
    sigma2_b_mu = sample(sigma2_b_mu_grid, size = 1, replace = FALSE)
    sigma2_w_mu = sample(sigma2_w_mu_grid, size = 1, replace = FALSE)
    
    
    sigma2_b_xi = sample(sigma2_b_xi_grid, size = 1, replace = FALSE)
    sigma2_w_xi = sample(sigma2_w_xi_grid, size = 1, replace = FALSE)
    
    print(sigma2_b_xi)
    print(sigma2_w_xi)
    
    sigma2_b_zeta = sample(sigma2_b_zeta_grid, size = 1, replace = FALSE)
    sigma2_w_zeta = sample(sigma2_w_zeta_grid, size = 1, replace = FALSE)
    
    
    sigma2_b_eta = sample(sigma2_b_eta_grid, size = 1, replace = FALSE)
    sigma2_w_eta = sample(sigma2_w_eta_grid, size = 1, replace = FALSE)
    
    sigma2_b_alpha = sample(sigma2_b_alpha_grid, size = 1, replace = FALSE)
    sigma2_w_alpha = sample(sigma2_b_alpha_grid, size = 1, replace = FALSE)
  }
  
  # Parameter storage for MCMC
  sigma2_k_store = matrix(NA, nrow = niter, ncol = m)
  
  omega_store = list()
  
  mu_series_store = matrix(NA, nrow = niter, ncol = T_len)
  
  xi_series_store = list()
  
  zeta_series_store = list()
  
  alpha_series_store = list()
  
  eta_series_store = list()
  
  sigma2_b_mu_store = rep(NA, niter)
  sigma2_w_mu_store = rep(NA, niter)
  
  sigma2_b_xi_store = rep(NA, niter)
  sigma2_w_xi_store = rep(NA, niter)
  
  sigma2_b_zeta_store = rep(NA, niter)
  sigma2_w_zeta_store = rep(NA, niter)
  
  sigma2_b_eta_store = rep(NA, niter)
  sigma2_w_eta_store = rep(NA, niter)
  
  sigma2_b_alpha_store = rep(NA, niter)
  sigma2_w_alpha_store = rep(NA, niter)
  
  
  
  # Process C_arr_for Polya-Gamma data augmentation
  
  kappa_arr = C_arr - (1 / 2)
  
  
  # Make adjacency matrices from the upper-triangular portions in C_arr
  C_adj_mats = array(data = NA, dim = c(J, J, L, T_len))
  
  # Make matrices for the omega values corresponding to each edge 
  omega_mats = array(data = NA, dim = c(J, J, L, T_len))
  
  for (l in 1:L) {
    
    for (t in 1:T_len) {
      
      upperTriangle(C_adj_mats[, , l, t], diag = FALSE, byrow = TRUE) =  C_arr[l, , t, drop = FALSE]
      diag(C_adj_mats[, , l, t]) = 0
      lowerTriangle(C_adj_mats[, , l, t], diag = FALSE, byrow = FALSE) = C_arr[l, , t, drop = FALSE]
      
      upperTriangle(omega_mats[, , l, t], diag = FALSE, byrow = TRUE) =  omega[l, , t, drop = FALSE]
      diag(omega_mats[, , l, t]) = 0
      lowerTriangle(omega_mats[, , l, t], diag = FALSE, byrow = FALSE) = omega[l, , t, drop = FALSE]
      
      
    }
  }
  
  kappa_adj_mats = C_adj_mats - (1 / 2)
  
  y_adj_mats = kappa_adj_mats / omega_mats
  
  
  # Calculate current kernel for each parameter 
  nngp_kernel_mu = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_mu, 
                                                  sigma2_b = sigma2_b_mu, nngp_L = nngp_L)
  
  nngp_kernel_xi = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_xi, 
                                                  sigma2_b = sigma2_b_xi, nngp_L = nngp_L)
  
  nngp_kernel_zeta = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_zeta, 
                                                    sigma2_b = sigma2_b_zeta, nngp_L = nngp_L)
  
  nngp_kernel_alpha = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_alpha, 
                                                     sigma2_b = sigma2_b_alpha, nngp_L = nngp_L)
  
  nngp_kernel_eta = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_eta, 
                                                   sigma2_b = sigma2_b_eta, nngp_L = nngp_L)
  
  
  # Calculate the inverses of the kernels to be used in the updates
  nngp_kernel_mu_inv = chol2inv(chol(nngp_kernel_mu))
  
  nngp_kernel_xi_inv = chol2inv(chol(nngp_kernel_xi))
  
  nngp_kernel_zeta_inv = chol2inv(chol(nngp_kernel_zeta))
  
  nngp_kernel_alpha_inv = chol2inv(chol(nngp_kernel_alpha))
  
  nngp_kernel_eta_inv = chol2inv(chol(nngp_kernel_eta))
  
  

  
  for (i in 1:niter) {
    
    print(i)
    
    ### Updates to network-only parameters
    
    # Terms for mu update
    omega_diag_sum_mu = matrix(0, nrow = T_len, ncol = T_len)
    
    m_sum_term_mu = rep(0, T_len)
    
    
 
    
    # Update mu
    for (l in 1:L) {
      for (j in 1:(J-1)) {
        for (j_prime in (j+1):J) {
          
          y_tilde = rep(NA, T_len)
          
          for (t in 1:T_len) {  
            y_tilde[t] = (y_adj_mats[j, j_prime, l, t] -
                            as.numeric(crossprod(xi_series[j, , t, l], xi_series[j_prime, , t, l])) - 
                            as.numeric(crossprod(zeta_series[j, , t], zeta_series[j_prime, , t])))
          }
          
          # Calculating diagonal matrices of omega values
          cur_omega_diag = diag(omega_mats[j, j_prime, l, ])
          omega_diag_sum_mu = omega_diag_sum_mu + cur_omega_diag
          
          # Calculating sum involved in the mean term
          
          # cur_omega mat is diagonal so can 
          m_sum_term_mu = m_sum_term_mu + cur_omega_diag%*%y_tilde
          
        }
      }
    }
    
    S_mu = chol2inv(chol(omega_diag_sum_mu + nngp_kernel_mu_inv))
    
    
    mu_series = rmvnorm(1, mean = S_mu %*% m_sum_term_mu, sigma = S_mu)
    

    
    # Update zeta_{j, r} for j = 1,...,J; r = 1,...,R
    

    
    for (r in 1:R_zeta) {
      for (j in 1:J) {
        
        S_sum_term_zeta = matrix(0, nrow = T_len, ncol = T_len)
        
        mu_sum_term_zeta = rep(0, T_len)
        
        for (l in 1:L) {
          for (j_prime in 1:J) {
            
            if (j_prime != j) {
              
              y_zeta = rep(NA, T_len)
              
              for (t in 1:T_len) {
                
                y_zeta[t] = (y_adj_mats[j, j_prime, l, t] - mu_series[t] -
                               as.numeric(crossprod(xi_series[j, , t, l], 
                                                    xi_series[j_prime, , t, l])) - 
                               as.numeric(crossprod(zeta_series[j, -r, t], 
                                                    zeta_series[j_prime, -r, t])))  
                
              }
              
              H_zeta = diag(zeta_series[j_prime, r, ])
              
              cur_omega_diag = diag(omega_mats[j, j_prime, l, ])
              
              S_sum_term_zeta = S_sum_term_zeta + t(H_zeta) %*% cur_omega_diag %*% H_zeta
              
              mu_sum_term_zeta = mu_sum_term_zeta + t(H_zeta) %*% cur_omega_diag %*% y_zeta
              
            }
          }
        }
        
        S_zeta = chol2inv(chol(S_sum_term_zeta + nngp_kernel_zeta_inv))
        
        
        zeta_series[j, r, ] = rmvnorm(1, mean = S_zeta %*% mu_sum_term_zeta, sigma = S_zeta)
        
      }
    }
    


    
    ### Update xi's which are related to both the network and the node covariates
    
    # Update xi_{j, l, r} for j = 1,...,J; l = l,...,L; r = 1,...,R
    

    
    for (j in 1:J) {
      for (l in 1:L) {
        for (r in 1:R) {
          
          S_xi_network_sum_term = matrix(0, nrow = T_len, ncol = T_len)
          m_xi_network_sum_term = rep(0, T_len)
          
          S_xi_node_sum_term = matrix(0, nrow = T_len, ncol = T_len)
          m_xi_node_sum_term = rep(0, T_len)
          
          for (j_prime in 1:J) {
            
            if (j_prime != j) {
              
              y_tilde_xi = rep(0, T_len)
              
              
              
              for (t in 1:T_len) {
                
                y_tilde_xi[t] = (y_adj_mats[j, j_prime, l, t] - mu_series[t] -
                                   as.numeric(crossprod(zeta_series[j, , t], 
                                                        zeta_series[j_prime, , t])) -
                                   as.numeric(crossprod(xi_series[j, -r, t, l], 
                                                        xi_series[j_prime, -r, t, l])))
                
                
              }
              
              cur_omega_diag = diag(omega_mats[j, j_prime, l, ])
              cur_xi_diag = diag(xi_series[j_prime, r, , l])
              
              S_xi_network_sum_term = (S_xi_network_sum_term + 
                                         t(cur_xi_diag) %*% cur_omega_diag %*% cur_xi_diag)
              
              m_xi_network_sum_term = (m_xi_network_sum_term + 
                                         t(cur_xi_diag) %*% cur_omega_diag %*% y_tilde_xi)
            }
            
          }
          
          for (k in 1:m) {
            
            x_tilde_xi = rep(0, T_len)
            
            for (t in 1:T_len) {
              
              x_tilde_xi[t] = (x_arr[j, m, t] - eta_series[k, t] -
                                 as.numeric(crossprod(alpha_series[k, -r, t, l],
                                                      xi_series[j, -r, t, l])))
              
              for (l_prime in 1:L) {
                if (l_prime != L) {
                  
                  x_tilde_xi[t] = x_tilde_xi[t] - as.numeric(crossprod(alpha_series[k, , t, l_prime], 
                                                                       xi_series[j, , t, l_prime]))
                  
                }
              }
              
            }
            
            cur_alpha_diag = diag(alpha_series[k, r, , l])
            
            S_xi_node_sum_term = (S_xi_node_sum_term + 
                                    crossprod(cur_alpha_diag) / sigma2_k[k])
            
            m_xi_node_sum_term = (m_xi_node_sum_term +
                                    crossprod(cur_alpha_diag, x_tilde_xi) / sigma2_k[k])
            
            
          }
          
          S_xi = chol2inv(chol(S_xi_network_sum_term + S_xi_node_sum_term +
                                 nngp_kernel_xi_inv))
          
          
          m_xi = m_xi_network_sum_term + m_xi_node_sum_term
          
          xi_series[j, r, , l] = rmvnorm(1, mean = S_xi %*% m_xi, sigma = S_xi)
          
        }
      }
    }
    

    
    
    # Update omega^t_{jj', l, r} for t = 1,...,T_len; j != j' \in 1,...,J; l = l,...,L
    

    
    for (t in 1:T_len) {
      
      for (l in 1:L) {
        
        for (j in 1:J) {
          
          for (j_prime in 1:J) {
            
            if (j_prime != J) {
              
              fitted_val = (mu_series[t] + 
                              as.numeric(crossprod(zeta_series[j, , t], zeta_series[j_prime, , t])) + 
                              as.numeric(crossprod(xi_series[j, , t, l], xi_series[j_prime, , t, l])))
              
              omega_mats[j, j_prime, l, t] = rpg(num = 1, h = 1,
                                                 z = fitted_val)
              
              omega_mats[j_prime, j, l, t] = omega_mats[j, j_prime, l, t]
              
            }
          }
        }
        
        omega[l, , t] = upperTriangle(omega_mats[, , l, t], diag = FALSE, byrow = TRUE)
        
      }
    }
    

    
    y_adj_mats = kappa_adj_mats / omega_mats
    
    ### Updates to parameters related only to node covariates
    

    
    # Update eta_k for k = 1,...,m
    
    little_s_j_k = array(0, dim = c(J, m, T_len))
    
    for (k in 1:m) {
      for (j in 1:J) {
        for (t in 1:T_len) {
          for (l in 1:L) {
            
            little_s_j_k[j, k, t] = little_s_j_k[j, k, t] + as.numeric(crossprod(xi_series[j, , t, l], 
                                                                                 alpha_series[k, , t, l]))
            
          }
        }
      }
    }
    
    # Node covariate parameter update timing
    # Can be done in parallel or sequentially
    if (run_parallel) {
      
      eta_update <- foreach (k = 1:m) %dopar% {
        
        library(mvtnorm)
        
        eta_resid_var_k = (J / sigma2_k[k]) * diag(rep(1, T_len))
        
        S_eta_k = chol2inv(chol(eta_resid_var_k + nngp_kernel_eta_inv)) 
        
        m_eta_k = rep(0, T_len)
        
        
        for (j in 1:J) {
          
          m_eta_k = m_eta_k + (x_arr[j, k, ] - little_s_j_k[j, k, ]) 
          
        }
        
        m_eta_k = m_eta_k / sigma2_k[k]
        
        
        # Update eta using mean and variance

        return(rmvnorm(1, mean = S_eta_k %*% m_eta_k, sigma = S_eta_k))
        
      }
      
      eta_series = matrix(unlist(eta_update), nrow = nrow(eta_series), ncol = ncol(eta_series),
                          byrow = TRUE)
      
    }
    
    
    else {
      
      for (k in 1:m) {
        
        eta_resid_var_k = (J / sigma2_k[k]) * diag(rep(1, T_len))
        
        S_eta_k = chol2inv(chol(eta_resid_var_k + nngp_kernel_eta_inv)) 
        
        m_eta_k = rep(0, T_len)
        
        
        for (j in 1:J) {
          
          m_eta_k = m_eta_k + (x_arr[j, k, ] - little_s_j_k[j, k, ]) 
          
        }
        
        m_eta_k = m_eta_k / sigma2_k[k]
        
        
        # Update eta using mean and variance
        eta_series[k, ] = rmvnorm(1, mean = S_eta_k %*% m_eta_k, sigma = S_eta_k)
        
      }
      
    }
    
    
    
    # Update sigma^2_k for k = 1,...,m
    for (k in 1:m) {
      
      sigma2_resid_sum = 0
      
      for (j in 1:J) {
        
        cur_resid = x_arr[j, k, ] - eta_series[m, ] - little_s_j_k[j, k, ] 
        
        sigma2_resid_sum = sigma2_resid_sum + as.numeric(crossprod(cur_resid))
        
      }
      
      sigma2_k[k] = rinvgamma(1, shape = a_sigma + J * T_len / 2, 
                              scale = b_sigma + sigma2_resid_sum / 2)
      
    }
    
    
    # Update alpha_{k, l, r} for k = 1,...,m; l = l,...,L; r = 1,...,R
    
    for (k in 1:m) {
      for (l in 1:L) {
        for (r in 1:R) {
          
          S_alpha_sum_term = matrix(0, nrow = T_len, ncol = T_len)
          m_alpha = rep(0, T_len)
          
          for (j in 1:J) {
            
            x_tilde = rep(NA, T_len)
            
            for (t in 1:T_len) {
              x_tilde[t] = (x_arr[j, k, t] - eta_series[k, t] -
                              as.numeric(crossprod(alpha_series[k, -r, t, l],
                                                   xi_series[j, -r, t, l])))
              
              for (l_prime in 1:L) {
                if (l_prime != L) {
                  
                  x_tilde[t] = x_tilde[t] - as.numeric(crossprod(alpha_series[k, , t, l_prime], 
                                                                 xi_series[j, , t, l_prime]))
                  
                }
              }
              
            }
            
            # Calculate sums
            A_alpha_diag = diag(xi_series[j, r, , l])
            
            S_alpha_sum_term = S_alpha_sum_term + crossprod(A_alpha_diag)
            
            m_alpha = m_alpha + crossprod(A_alpha_diag, x_tilde)
            
          }
          
          m_alpha = m_alpha / sigma2_k[k]
          
          S_alpha = chol2inv(chol(S_alpha_sum_term / sigma2_k[k] + nngp_kernel_alpha_inv))
          
          alpha_series[k, r, , l] = rmvnorm(1, mean = S_alpha%*%m_alpha, sigma = S_alpha)
          
        } 
      }
    }
    
    

  
    
    
    
    ### Update the NNGP weight and bias variances for each time-varying parameter
    if (use_mh) {
      
      
      # Get new theta_mu
      theta_mu = sigma2_mu_mh_update(mu_series = mu_series,
                                     xi_series = xi_series, 
                                     zeta_series = zeta_series,
                                     omega_mats = omega_mats,
                                     y_adj_mats = y_adj_mats,
                                     time_grid = time_grid, sigma2_b_old = sigma2_b_mu, 
                                     sigma2_w_old = sigma2_w_mu,
                                     nngp_L = nngp_L, b_proposal_var = b_proposal_var,
                                     w_proposal_var = w_proposal_var)
      
      # update sigma2_b_mu
      sigma2_b_mu = theta_mu$sigma2_b
      #print(sigma2_b_mu)
      
      # update sigma2_w_mu
      sigma2_w_mu = theta_mu$sigma2_w
      #print(sigma2_w_mu)
      
      
      # Get new theta_zeta
      theta_zeta = sigma2_zeta_mh_update (mu_series = mu_series,
                                          xi_series = xi_series, 
                                          zeta_series = zeta_series,
                                          omega_mats = omega_mats,
                                          y_adj_mats = y_adj_mats,
                                          time_grid = time_grid, sigma2_b_old = sigma2_b_zeta, 
                                          sigma2_w_old= sigma2_w_zeta,
                                          nngp_L = nngp_L, b_proposal_var = b_proposal_var,
                                          w_proposal_var = w_proposal_var)
      
      # update sigma2_b_zeta
      sigma2_b_zeta = theta_zeta$sigma2_b
      print(paste("sigma2_b_zeta:", sigma2_b_zeta))
      
      # update sigma2_w_zeta
      sigma2_w_zeta = theta_zeta$sigma2_w
      print(paste("sigma2_w_zeta:", sigma2_w_zeta))
      
      
      # Get new theta_xi
      theta_xi = sigma2_xi_mh_update(mu_series = mu_series,
                                     xi_series = xi_series, 
                                     zeta_series = zeta_series,
                                     omega_mats = omega_mats,
                                     y_adj_mats = y_adj_mats,
                                     x_arr = x_arr, 
                                     eta_series = eta_series,
                                     alpha_series = alpha_series,
                                     sigma2_k = sigma2_k,
                                     time_grid = time_grid, sigma2_b_old = sigma2_b_xi, 
                                     sigma2_w_old = sigma2_w_xi,
                                     nngp_L = nngp_L, b_proposal_var = b_proposal_var,
                                     w_proposal_var = w_proposal_var)
      
      # update sigma2_b_xi
      sigma2_b_xi = theta_xi$sigma2_b
      
      # update sigma2_w_xi
      sigma2_w_xi = theta_xi$sigma2_w
      
      # update sigma2_b_xi
      sigma2_b_xi = theta_xi$sigma2_b
      print(paste("sigma2_b_xi:", sigma2_b_xi))
      
      # update sigma2_w_xi
      sigma2_w_xi = theta_xi$sigma2_w
      print(paste("sigma2_w_xi:", sigma2_w_xi))
      
      
      # Get new theta_alpha
      theta_alpha = sigma2_alpha_mh_update(xi_series = xi_series, 
                                           x_arr = x_arr, 
                                           eta_series = eta_series,
                                           alpha_series = alpha_series,
                                           sigma2_k = sigma2_k,
                                           time_grid = time_grid, 
                                           sigma2_b_old = sigma2_b_alpha, 
                                           sigma2_w_old = sigma2_w_alpha,
                                           nngp_L = nngp_L, b_proposal_var = b_proposal_var,
                                           w_proposal_var = w_proposal_var)
      
      # update sigma2_b_alpha
      sigma2_b_alpha = theta_alpha$sigma2_b
      
      # update sigma2_w_alpha
      sigma2_w_alpha = theta_alpha$sigma2_w
      
      
      # Get new theta_eta
      theta_eta = sigma2_eta_mh_update(xi_series = xi_series, 
                                       x_arr = x_arr, 
                                       eta_series = eta_series,
                                       alpha_series = alpha_series,
                                       sigma2_k = sigma2_k,
                                       time_grid = time_grid, 
                                       sigma2_b_old = sigma2_b_eta, 
                                       sigma2_w_old = sigma2_w_eta,
                                       nngp_L = nngp_L, b_proposal_var = b_proposal_var,
                                       w_proposal_var = w_proposal_var)
      
      # update sigma2_b_eta
      sigma2_b_eta = theta_eta$sigma2_b
      
      # update sigma2_w_zeta
      sigma2_w_eta = theta_eta$sigma2_w
      
      
    }
    
    else {
      # Get new theta_mu
      theta_mu = optimize_sigma2_mu_log_likelihood(mu_series = mu_series,
                                                   xi_series = xi_series,
                                                   zeta_series = zeta_series,
                                                   omega_mats = omega_mats,
                                                   y_adj_mats = y_adj_mats,
                                                   time_grid = time_grid, sigma2_b_grid = sigma2_b_mu_grid,
                                                   sigma2_w_grid = sigma2_w_mu_grid,
                                                   nngp_L = nngp_L)
      

      
      # update sigma2_b_mu
      sigma2_b_mu = theta_mu$sigma2_b
 
      
      # update sigma2_w_mu
      sigma2_w_mu = theta_mu$sigma2_w

      
      
      # Get new theta_zeta
      theta_zeta = optimize_sigma2_zeta_log_likelihood(mu_series = mu_series,
                                                       xi_series = xi_series,
                                                       zeta_series = zeta_series,
                                                       omega_mats = omega_mats,
                                                       y_adj_mats = y_adj_mats,
                                                       time_grid = time_grid, sigma2_b_grid = sigma2_b_zeta_grid,
                                                       sigma2_w_grid = sigma2_w_zeta_grid,
                                                       nngp_L = nngp_L)
      

      
      # update sigma2_b_zeta
      sigma2_b_zeta = theta_zeta$sigma2_b
      
      # update sigma2_w_zeta
      sigma2_w_zeta = theta_zeta$sigma2_w
      
      
      # Get new theta_xi
      theta_xi = optimize_sigma2_xi_log_likelihood(mu_series = mu_series,
                                                   xi_series = xi_series,
                                                   zeta_series = zeta_series,
                                                   omega_mats = omega_mats,
                                                   y_adj_mats = y_adj_mats,
                                                   x_arr = x_arr,
                                                   eta_series = eta_series,
                                                   alpha_series = alpha_series,
                                                   sigma2_k = sigma2_k,
                                                   time_grid = time_grid, sigma2_b_grid = sigma2_b_xi_grid,
                                                   sigma2_w_grid = sigma2_w_xi_grid,
                                                   nngp_L = nngp_L)
      

      
      # update sigma2_b_xi
      sigma2_b_xi = theta_xi$sigma2_b
      
      # update sigma2_w_xi
      sigma2_w_xi = theta_xi$sigma2_w
      
      # update sigma2_b_xi
      sigma2_b_xi = theta_xi$sigma2_b
      print(paste("sigma2_b_xi:", sigma2_b_xi))
      
      # update sigma2_w_xi
      sigma2_w_xi = theta_xi$sigma2_w
      print(paste("sigma2_w_xi:", sigma2_w_xi))
      
      
      # Get new theta_alpha
      theta_alpha = optimize_sigma2_alpha_log_likelihood(xi_series = xi_series,
                                                         x_arr = x_arr,
                                                         eta_series = eta_series,
                                                         alpha_series = alpha_series,
                                                         sigma2_k = sigma2_k,
                                                         time_grid = time_grid,
                                                         sigma2_b_grid = sigma2_b_alpha_grid,
                                                         sigma2_w_grid = sigma2_w_alpha_grid,
                                                         nngp_L = nngp_L)
      

      
      # update sigma2_b_alpha
      sigma2_b_alpha = theta_alpha$sigma2_b
      
      # update sigma2_w_alpha
      sigma2_w_alpha = theta_alpha$sigma2_w
      
      
      # Get new theta_eta
      theta_eta = optimize_sigma2_eta_log_likelihood(xi_series = xi_series,
                                                     x_arr = x_arr,
                                                     eta_series = eta_series,
                                                     alpha_series = alpha_series,
                                                     sigma2_k = sigma2_k,
                                                     time_grid = time_grid,
                                                     sigma2_b_grid = sigma2_b_eta_grid,
                                                     sigma2_w_grid = sigma2_w_eta_grid,
                                                     nngp_L = nngp_L)
      

      
      # update sigma2_b_eta
      sigma2_b_eta = theta_eta$sigma2_b
      
      # update sigma2_w_zeta
      sigma2_w_eta = theta_eta$sigma2_w
      
    }  
    
    
    # calculate updated kernel for each parameter
    nngp_kernel_mu = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_mu, 
                                                    sigma2_b = sigma2_b_mu, nngp_L = nngp_L)
    
    nngp_kernel_xi = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_xi, 
                                                    sigma2_b = sigma2_b_xi, nngp_L = nngp_L)
    
    nngp_kernel_zeta = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_zeta, 
                                                      sigma2_b = sigma2_b_zeta, nngp_L = nngp_L)
    
    nngp_kernel_alpha = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_alpha, 
                                                       sigma2_b = sigma2_b_alpha, nngp_L = nngp_L)
    
    nngp_kernel_eta = full_nngp_kernel_time_series_L(time_grid = time_grid, sigma2_w = sigma2_w_eta, 
                                                     sigma2_b = sigma2_b_eta, nngp_L = nngp_L)
    
    
    # Calculate the updated inverses of the kernels to be used in the updates
    nngp_kernel_mu_inv = chol2inv(chol(nngp_kernel_mu))
    
    nngp_kernel_xi_inv = chol2inv(chol(nngp_kernel_xi))
    
    nngp_kernel_zeta_inv = chol2inv(chol(nngp_kernel_zeta))
    
    nngp_kernel_alpha_inv = chol2inv(chol(nngp_kernel_alpha))
    
    nngp_kernel_eta_inv = chol2inv(chol(nngp_kernel_eta))
    
    
    # Store updated parameters  
    sigma2_k_store[i, ] = sigma2_k
    
    omega_store[[i]] = omega
    
    mu_series_store[i, ] = mu_series
    
    xi_series_store[[i]] = xi_series
    
    zeta_series_store[[i]] = zeta_series
    
    alpha_series_store[[i]] = alpha_series
    
    eta_series_store[[i]] = eta_series
    
    sigma2_b_mu_store[i] = sigma2_b_mu
    sigma2_w_mu_store[i] = sigma2_w_mu
    
    sigma2_b_xi_store[i] = sigma2_b_xi
    sigma2_w_xi_store[i] = sigma2_w_xi
    
    sigma2_b_zeta_store[i] = sigma2_b_zeta
    sigma2_w_zeta_store[i] = sigma2_w_zeta
    
    sigma2_b_eta_store[i] = sigma2_b_eta
    sigma2_w_eta_store[i] = sigma2_w_eta
    
    sigma2_b_alpha_store[i] = sigma2_b_alpha
    sigma2_w_alpha_store[i] = sigma2_w_alpha
    
    
  }
  
  
  
  
  
  filename = paste("outputs/mcmc_out_J_", 
                   J,
                   "_L_", L, "_R_", R, "_R_zeta_", R_zeta, "_m_", m, "_T_len_", T_len,
                   "_phi_", phi, "_epsilon_", epsilon, "_TrueR_", True_R, 
                   "_TrueF_", True_F,
                   "_niter_", niter, "_nngpL_", nngp_L,
                   "_seed_", seed, "_nngp_gen_", nngp_gen,
                   "_tergm_", tergm_gen, 
                   "_real_data_", real_data,
                   ".RData", sep = "")
  
  

  
  model_outputs = list(C_arr = C_arr, x_arr = x_arr, time_grid = time_grid,
                       sigma2_k_store = sigma2_k_store,
                       mu_series_store = mu_series_store,
                       xi_series_store = xi_series_store,
                       zeta_series_store = zeta_series_store,
                       alpha_series_store = alpha_series_store,
                       eta_series_store = eta_series_store,
                       sigma2_b_mu_store = sigma2_b_mu_store,
                       sigma2_w_mu_store = sigma2_w_mu_store,
                       sigma2_b_xi_store = sigma2_b_xi_store,
                       sigma2_w_xi_store = sigma2_w_xi_store,
                       sigma2_b_zeta_store = sigma2_b_zeta_store,
                       sigma2_w_zeta_store = sigma2_w_zeta_store,
                       sigma2_b_eta_store = sigma2_b_eta_store,
                       sigma2_w_eta_store = sigma2_w_eta_store,
                       sigma2_b_alpha_store = sigma2_b_alpha_store,
                       sigma2_w_alpha_store = sigma2_w_alpha_store)
  
  
  if (save_outputs) {
    save(model_outputs, file = filename)
  }
  
  
  if (run_parallel) {
    stopCluster(cl)
  }
  
  return(model_outputs)
  
}



