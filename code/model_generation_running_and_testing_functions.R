
library(mvtnorm)
library(pROC)
library(caret)
library(BayesLogit)

source("model_development.R")
source("simulated_data_generation_layerwise_missingness.R")
source("simulated_data_generation_nngp_layerwise_missingness.R")
source("simulated_data_generation_tergm_layerwise_missingness.R")
source("incomplete_graphs_model.R")





########### Functions for model data generation, running, and testing


model_in_sample_predictions <- function(revised_missing_simulation_dataset,
                                          #data_filename, # mcmc_filename,
                                                         model_outputs,
                                               L, J, m, T_len, R = NULL, sim_gen_nngp_L = NULL,
                                               phi = NULL, epsilon = NULL, time_props_missing,
                                               edge_props_missing, nngp_gen = NULL,
                                               tergm_gen = NULL,
                                               nngp_L = 1, 
                                               niter, fitted_R, 
                                              fitted_R_zeta,
                                               seed = 1234,
                                               n_burn_in, n_thinning,
                                               real_data = FALSE,
                                               full_real_data = NULL) {
  
  #load(data_filename)
  #load(mcmc_filename)
  
  
  iter_seq = seq(from = n_burn_in + 1, to = niter, by = n_thinning)
  n_samples = length(iter_seq)
  
  J = dim(revised_missing_simulation_dataset$x_arr_missing)[1]
  m = dim(revised_missing_simulation_dataset$x_arr_missing)[2]
  
  L = dim(revised_missing_simulation_dataset$C_arr_missing)[1]
  
  T_len = length(model_outputs$time_grid)
  
  

  C_arr_t_preds = array(NA, dim = c(L, J * (J - 1) / 2, T_len, n_samples))
  
  
  print(model_outputs$time_grid)
  
  
  
  # Network predictions
  for (i in 1:n_samples) {
    for (l in 1:L) {
    
      edge_counter = 1
    
      for (j in 1:(J - 1)) {
      
        for (j_prime in (j + 1):J) {
          for(t in 1:T_len) {
            
              C_arr_t_preds[l, edge_counter, t, i] = gp_edge_prediction(model_outputs$time_grid[t], 
                                 j = j, j_prime = j_prime,
                                 l = l,
                                 time_grid = model_outputs$time_grid,
                                 mu_series = model_outputs$mu_series_store[iter_seq[i], ],
                                 xi_series = model_outputs$xi_series_store[[iter_seq[i]]],
                                 zeta_series = model_outputs$zeta_series_store[[iter_seq[i]]])
              
              
              
          }
  
          
          edge_counter = edge_counter + 1
          
        }
      }
    }
  }
  
  C_arr_t_pred_means = apply(C_arr_t_preds, c(1, 2, 3), mean, na.rm=TRUE)
  
  
  C_arr_t_pred_means_vectorized = as.numeric(C_arr_t_pred_means)
  C_arr_t_true_values_vectorized = as.numeric(model_outputs$C_arr)
  
  C_arr_t_roc = roc(response = C_arr_t_true_values_vectorized, C_arr_t_pred_means_vectorized)
  
  C_arr_obs_auc = auc(C_arr_t_roc)
  C_arr_obs_auc
  
  C_predictions = as.factor(as.numeric(C_arr_t_pred_means_vectorized > 0.5))
  C_true_vals = as.factor(C_arr_t_true_values_vectorized )
  
  
  precision = posPredValue(C_predictions, C_true_vals, positive="1")
  recall = sensitivity(C_predictions, C_true_vals, positive="1")
  specificity_val = specificity(C_predictions, C_true_vals, negative="0")
  
  f1_score = (2 * precision * recall) / (precision + recall)
  
  
  #x_arr_t_preds = array(NA, dim = c(J, m, niter))
  x_arr_t_preds = array(NA, dim = c(J, m, T_len, n_samples))
  
  
  # Node attribute predictions
  
    
  for (i in 1:n_samples) {   
    for (j in 1:J) {
      for (k in 1:m) {
        for (t in 1:T_len) {
          

          x_arr_t_preds[j, k, t, i] = gp_node_covariate_prediction(model_outputs$time_grid[t],
                                         j = j, k = k,
                                         time_grid = model_outputs$time_grid,
                                         sigma2_k = model_outputs$sigma2_k_store[iter_seq[i], ],
                                         eta_series = model_outputs$eta_series_store[[iter_seq[i]]],
                                         xi_series = model_outputs$xi_series_store[[iter_seq[i]]],
                                         alpha_series = model_outputs$alpha_series_store[[iter_seq[i]]])
          
          
          
        
        }
      }
    }
  }
  
  
  x_arr_t_preds_means = apply(x_arr_t_preds, c(1, 2, 3), mean)
  
  true_x_arr_vals = model_outputs$x_arr
  
  x_arr_obs_mse = sum((x_arr_t_preds_means - true_x_arr_vals) ^ 2) / sum(true_x_arr_vals ^ 2)
  x_arr_obs_mse
  
  
  results = list(C_arr_obs_auc = C_arr_obs_auc, 
                 C_arr_obs_f1_score = f1_score,
                 C_arr_obs_precision = precision,
                 C_arr_obs_sensitivity = recall,
                 C_arr_obs_specificity = specificity_val,
                 x_arr_obs_mse = x_arr_obs_mse)
  
  
  # Prediction Intervals for x
  x_arr_pred_coverages = array(NA, dim = c(J, m, T_len))
  
  x_arr_pred_lengths = array(NA, dim = c(J, m, T_len))
  
  for (k in 1:m) {
    for (j in 1:J) {
      for (t in 1:T_len) {
        
        cur_interval = quantile(x_arr_t_preds[j, k, t, ], probs = c(0.025, 0.975))
        
        cur_x_lower = cur_interval[1]
        cur_x_upper = cur_interval[2]
        
        
        cur_x = true_x_arr_vals[j, k, t]
        
        if (cur_x >= cur_x_lower && cur_x <= cur_x_upper) {
          x_arr_pred_coverages[j, k, t] = 1
        }
        
        else {
          x_arr_pred_coverages[j, k, t] = 0
        }
        
        x_arr_pred_lengths[j, k, t] = cur_x_upper - cur_x_lower
        
      }
    }
  }
  
  mean_coverage = mean(as.numeric(x_arr_pred_coverages))
  
  mean_length = mean(as.numeric(x_arr_pred_lengths))
  
  mean_coverage_by_attr = apply(x_arr_pred_coverages, 2, mean)
  
  mean_length_by_attr = apply(x_arr_pred_lengths, 2, mean)
  
  results$mean_coverage = mean_coverage
  
  results$mean_length = mean_length
  
  results$mean_coverage_by_attr = mean_coverage_by_attr
  
  results$mean_length_by_attr = mean_length_by_attr
  
  results_filename = paste("outputs/in_samp_pred_J_", 
                           J,
                           "_L", L,
                           "_R", fitted_R,
                           "_R_zeta", fitted_R_zeta,
                           "_m", m,
                           "_T_len", T_len,
                           "_phi", phi, 
                           "_epsilon", epsilon,
                           "_TrueR", R, 
                           "_TrueF", sim_gen_nngp_L,
                           "_niter", niter, "_n_burnin", n_burn_in,
                           "_nngpL", nngp_L, "_nngp_gen", nngp_gen,
                           "_tergm_", tergm_gen,
                           "_seed", seed, "_real_data", real_data,
                           #"_T_prop", paste(time_props_missing, collapse = "_"),
                           #"_C_prop", paste(edge_props_missing, collapse = "_"),
                           "_full_real_", full_real_data,
                           ".RData", sep = "")
  
  print(results_filename)
  
  save(results, file = results_filename)
  
}



model_out_of_sample_predictions <- function(revised_missing_simulation_dataset,
                                              #data_filename, mcmc_filename,
                                                             model_outputs,
                                                             L, J, m, T_len, R = NULL, sim_gen_nngp_L = NULL,
                                                             phi = NULL, epsilon = NULL, time_props_missing,
                                                             edge_props_missing, nngp_gen = NULL,
                                                             tergm_gen = NULL,
                                                             nngp_L = 1, 
                                                             niter, fitted_R, 
                                                            fitted_R_zeta,
                                                             seed = 1234,
                                                             n_burn_in, n_thinning,
                                                             real_data = FALSE,
                                                             only_last = FALSE,
                                                             all_but_last = FALSE,
                                                             only_use_last = FALSE) {
  
  #load(data_filename)
  # load(mcmc_filename)
  
  
  iter_seq = seq(from = n_burn_in + 1, to = niter, by = n_thinning)
  n_samples = length(iter_seq)
  
  J = dim(revised_missing_simulation_dataset$x_arr_missing)[1]
  m = dim(revised_missing_simulation_dataset$x_arr_missing)[2]
  
  
  
  L = dim(revised_missing_simulation_dataset$C_arr_missing)[1]
  
  T_len = length(model_outputs$time_grid)
  
  
  # Missing Time Points
  discarded_time_points_idxs = revised_missing_simulation_dataset$discarded_time_points_idxs
  
  if (only_use_last) {
    
    discarded_time_points_idxs = discarded_time_points_idxs[which.max(discarded_time_points_idxs), drop = FALSE]
    
  }
  
  T_len_missing = length(discarded_time_points_idxs)
  
  
  # Initialize the time varying parameters
  # mu predictions
  cur_mu_predictions = rnorm(T_len_missing)
  
  
  # zeta predictions
  cur_zeta_predictions = array(rnorm(J * fitted_R_zeta * T_len_missing), dim = c(J, fitted_R_zeta, T_len_missing))
  
  # xi predictions
  cur_xi_predictions = array(rnorm(J * fitted_R * T_len_missing * L), dim = c(J, fitted_R, T_len_missing, L))
  
  # eta predictions
  cur_eta_predictions = matrix(rnorm(m * T_len_missing), nrow = m, ncol = T_len_missing)
  
  
  # alpha predictions
  cur_alpha_predictions = array(rnorm(m * fitted_R * T_len_missing * L), dim = c(m, fitted_R, T_len_missing, L))
  
  
  # The current omega matrix
  # cur_omega_preds = array(rpg(num = L * J * (J - 1) * T_len_missing / 2), dim = c(L, J * (J - 1) / 2, T_len_missing))
  # 
  # cur_omega_mats = array(data = NA, dim = c(J, J, L, T_len_missing))
  # 
  
  
  
  # Storing 
  C_arr_t_preds_missing = array(NA, dim = c(L, J * (J - 1) / 2, T_len_missing, n_samples))
  
  x_arr_t_preds_missing = array(NA, dim = c(J, m, T_len_missing, n_samples))
  
  
  # Make adjacency matrices from the upper-triangular portions in C_arr
  C_adj_mats = array(data = NA, dim = c(J, J, L, T_len))

  
  for (l in 1:L) {
    
    for (t in 1:T_len) {
      
      upperTriangle(C_adj_mats[, , l, t], diag = FALSE, byrow = TRUE) =  model_outputs$C_arr[l, , t, drop = FALSE]
      diag(C_adj_mats[, , l, t]) = 0
      lowerTriangle(C_adj_mats[, , l, t], diag = FALSE, byrow = FALSE) = model_outputs$C_arr[l, , t, drop = FALSE]
      
      
    }
    
    
  }
  
  
  
  
  # Network predictions
  for (i in 1:n_samples) {
    
    
    # Make matrices for the omega values corresponding to each edge 
    # omega_mats = array(data = NA, dim = c(J, J, L, T_len))
    # 
    # for (l in 1:L) {
    #   
    #   for (t in 1:T_len) {
    #     
    #     upperTriangle(omega_mats[, , l, t], diag = FALSE, byrow = TRUE) =  model_outputs$omega_store[[iter_seq[i]]][l, , t, drop = FALSE]
    #     diag(omega_mats[, , l, t]) = 0
    #     lowerTriangle(omega_mats[, , l, t], diag = FALSE, byrow = FALSE) = model_outputs$omega_store[[iter_seq[i]]][l, , t, drop = FALSE]
    #     
    #   }
    #   
    #   for (t in 1:T_len_missing) {
    #     
    #     
    #     upperTriangle(cur_omega_mats[, , l, t], diag = FALSE, byrow = TRUE) = cur_omega_preds[l, , t, drop = FALSE]
    #     diag(cur_omega_mats[, , l, t]) = 0
    #     lowerTriangle(cur_omega_mats[, , l, t], diag = FALSE, byrow = FALSE) = cur_omega_preds[l, , t, drop = FALSE]
    #     
    #   }
    #   
    # }
    
    # print(dim(cur_omega_mats))
    
    # mu predictions
    for (t in 1:T_len_missing) {
    
      cur_mu_predictions[t] = gp_param_prediction(t = discarded_time_points_idxs[t],
                          time_grid = model_outputs$time_grid, param_series = model_outputs$mu_series_store[iter_seq[i], ],
                          sigma2_w = model_outputs$sigma2_w_mu_store[iter_seq[i]],
                          sigma2_b = model_outputs$sigma2_b_mu_store[iter_seq[i]],
                          nngp_L = nngp_L)
        
    }
    
    print(cur_mu_predictions)
    
    
    # zeta predictions
    for (t in 1:T_len_missing) {
      for (j in 1:J) {
        for (r in 1:fitted_R_zeta) {
          
          
          
          cur_zeta_predictions[j, r, t] = gp_param_prediction(t = discarded_time_points_idxs[t],
                              time_grid = model_outputs$time_grid, param_series = model_outputs$zeta_series_store[[iter_seq[i]]][j, r, ],
                              sigma2_w = model_outputs$sigma2_w_zeta_store[iter_seq[i]],
                              sigma2_b = model_outputs$sigma2_b_zeta_store[iter_seq[i]],
                              nngp_L = nngp_L)
          
          
        }
      }
    }
    
    
    # xi predictions
    for (t in 1:T_len_missing) {
      for (j in 1:J) {
        for (r in 1:fitted_R) {
          for(l in 1:L) {
            
            cur_xi_predictions[j, r, t, l] = gp_param_prediction(t = discarded_time_points_idxs[t],
                                                                 time_grid = model_outputs$time_grid, param_series = model_outputs$xi_series_store[[iter_seq[i]]][j, r, , l],
                                                                 sigma2_w = model_outputs$sigma2_w_xi_store[iter_seq[i]],
                                                                 sigma2_b = model_outputs$sigma2_b_xi_store[iter_seq[i]],
                                                                 nngp_L = nngp_L)
            
          }
        }
      }
    }
    
    
    

    
    # eta predictions
    for (t in 1:T_len_missing) {
      for (k in 1:m) {
        
        cur_eta_predictions[k, t] = gp_param_prediction(t = discarded_time_points_idxs[t],
                                                        time_grid = model_outputs$time_grid, param_series = model_outputs$eta_series_store[[iter_seq[i]]][k, ],
                                                        sigma2_w = model_outputs$sigma2_w_eta_store[iter_seq[i]],
                                                        sigma2_b = model_outputs$sigma2_b_eta_store[iter_seq[i]],
                                                        nngp_L = nngp_L)
        
        
      }
    }
    
    
    
    # alpha predictions
    for (t in 1:T_len_missing) {
      for (k in 1:m) {
        for (r in 1:fitted_R) {
          for(l in 1:L) {
            
            cur_alpha_predictions[k, r, t, l] = gp_param_prediction(t = discarded_time_points_idxs[t],
                                                                    time_grid = model_outputs$time_grid, param_series = model_outputs$alpha_series_store[[iter_seq[i]]][k, r, , l],
                                                                    sigma2_w = model_outputs$sigma2_w_alpha_store[iter_seq[i]],
                                                                    sigma2_b = model_outputs$sigma2_b_alpha_store[iter_seq[i]],
                                                                    nngp_L = nngp_L)
            
          }
        }
      }
    }
    
    
    
    # Edge predictions
    
    for (l in 1:L) {
      
      edge_counter = 1
      
      for (j in 1:(J - 1)) {
        
        for (j_prime in (j + 1):J) {
          for(t in 1:T_len_missing) {
            
            
            C_arr_t_preds_missing[l, edge_counter, t, i]  = gp_edge_prediction(discarded_time_points_idxs[t], 
                                                                  j = j, j_prime = j_prime,
                                                                  l = l,
                                                                  time_grid = discarded_time_points_idxs,
                                                                  mu_series = cur_mu_predictions,
                                                                  xi_series = cur_xi_predictions,
                                                                  zeta_series = cur_zeta_predictions)
            
            # Update omega
            fitted_val = (cur_mu_predictions[t] + 
                            as.numeric(crossprod(cur_zeta_predictions[j, , t], cur_zeta_predictions[j_prime, , t])) + 
                            as.numeric(crossprod(cur_xi_predictions[j, , t, l], cur_xi_predictions[j_prime, , t, l])))
            
            #print(fitted_val)
            
            # cur_omega_preds[l, edge_counter, t] = rpg(num = 1, h = 1, z = fitted_val)
            
            
            
            
          }
          
          edge_counter = edge_counter + 1
        }
        
        
        
      }
    }
    
    # Node attribute predictions
    
    for (j in 1:J) {
      
      for (k in 1:m) {
        for (t in 1:T_len_missing) {
          
          
          x_arr_t_preds_missing[j, k, t, i] = gp_node_covariate_prediction(discarded_time_points_idxs[t],
                                                                   j = j, k = k,
                                                                   time_grid = discarded_time_points_idxs,
                                                                   sigma2_k = model_outputs$sigma2_k_store[iter_seq[i], ],
                                                                   eta_series = cur_eta_predictions,
                                                                   xi_series = cur_xi_predictions,
                                                                   alpha_series = cur_alpha_predictions)
          
        
        }
      }
    }
    
    
    
  }
  
  # Extract true missing values
  
  if (real_data) {
    
    C_arr_missing = revised_missing_simulation_dataset$C_arr[, , discarded_time_points_idxs, drop = FALSE]
    
    print(dim(C_arr_missing))
    
    
  }
  
  else {
    
    C_arr_missing = revised_missing_simulation_dataset$out$C_arr[, , discarded_time_points_idxs, drop = FALSE]
    
    sum(is.na(C_arr_missing))
    
  }
  
  C_arr_t_pred_means = apply(C_arr_t_preds_missing, c(1, 2, 3), mean, na.rm=TRUE)
  
  #C_arr_t_pred_classifications = (C_arr_t_pred_means > 0.5)
  
  C_arr_t_pred_means_vectorized = as.numeric(C_arr_t_pred_means)
  C_arr_t_true_values_vectorized = as.numeric(C_arr_missing)
  
  print("first")
  print(unique(C_arr_t_true_values_vectorized))
  
  
  C_arr_t_roc = roc(response = C_arr_t_true_values_vectorized, C_arr_t_pred_means_vectorized)
  
  C_arr_obs_auc = auc(C_arr_t_roc)
  C_arr_obs_auc
  
  C_predictions = as.factor(as.numeric(C_arr_t_pred_means_vectorized > 0.5))
  C_true_vals = as.factor(C_arr_t_true_values_vectorized )
  
  
  precision = posPredValue(C_predictions, C_true_vals, positive="1")
  recall = sensitivity(C_predictions, C_true_vals, positive="1")
  specificity_val = specificity(C_predictions, C_true_vals, negative="0")
  
  f1_score = (2 * precision * recall) / (precision + recall)
  
  
  #x_arr_t_preds = array(NA, dim = c(J, m, niter))
  
  # Node attribute predictions

  
  
  x_arr_t_preds_missing_means = apply(x_arr_t_preds_missing, c(1, 2, 3), mean)
  
  
  if (real_data) {
    true_x_arr_missing_vals = revised_missing_simulation_dataset$x_arr[, , discarded_time_points_idxs, drop = FALSE]
  }
  
  else {
    
    true_x_arr_missing_vals = revised_missing_simulation_dataset$out$x_arr[, , discarded_time_points_idxs, drop = FALSE]
    
  }
  
  
  x_arr_obs_mse = sum((x_arr_t_preds_missing_means - true_x_arr_missing_vals) ^ 2) / sum(true_x_arr_missing_vals ^ 2)
  x_arr_obs_mse
  
  
  results = list(C_arr_obs_auc = C_arr_obs_auc, 
                 C_arr_obs_f1_score = f1_score,
                 C_arr_obs_precision = precision,
                 C_arr_obs_sensitivity = recall,
                 C_arr_obs_specificity = specificity_val,
                 x_arr_obs_mse = x_arr_obs_mse)
  
  
  # Prediction Intervals for x
  x_arr_pred_coverages = array(NA, dim = c(J, m, T_len_missing))
  
  x_arr_pred_lengths = array(NA, dim = c(J, m, T_len_missing))
  
  for (k in 1:m) {
    for (j in 1:J) {
      for (t in 1:T_len_missing) {
        
        cur_interval = quantile(x_arr_t_preds_missing[j, k, t, ], probs = c(0.025, 0.975))
        
        cur_x_lower = cur_interval[1]
        cur_x_upper = cur_interval[2]
        
        
        cur_x = true_x_arr_missing_vals[j, k, t]
        
        if (cur_x >= cur_x_lower && cur_x <= cur_x_upper) {
          x_arr_pred_coverages[j, k, t] = 1
        }
        
        else {
          x_arr_pred_coverages[j, k, t] = 0
        }
        
        x_arr_pred_lengths[j, k, t] = cur_x_upper - cur_x_lower
        
      }
    }
  }
  
  mean_coverage = mean(as.numeric(x_arr_pred_coverages))
  
  mean_length = mean(as.numeric(x_arr_pred_lengths))
  
  mean_coverage_by_attr = apply(x_arr_pred_coverages, 2, mean)
  
  mean_length_by_attr = apply(x_arr_pred_lengths, 2, mean)
  
  results$mean_coverage = mean_coverage
  
  results$mean_length = mean_length
  
  results$mean_coverage_by_attr = mean_coverage_by_attr
  
  results$mean_length_by_attr = mean_length_by_attr
  
  
  if (only_last) {
    
    if (real_data) {
      
      
      C_arr_missing_last = revised_missing_simulation_dataset$C_arr[, , max(discarded_time_points_idxs), drop = FALSE]
    }
    
    else {
      
      C_arr_missing_last = revised_missing_simulation_dataset$out$C_arr[, , max(discarded_time_points_idxs), drop = FALSE]
      
    }
    
    
    C_arr_t_pred_means = apply(C_arr_t_preds_missing[, , which.max(discarded_time_points_idxs), , drop = FALSE],
                               c(1, 2, 3), mean, na.rm=TRUE)
    
    
    C_arr_t_pred_means_vectorized = as.numeric(C_arr_t_pred_means)
    C_arr_t_true_values_vectorized = as.numeric(C_arr_missing_last)
    
    print("second")
    print(unique(C_arr_t_true_values_vectorized))
    
    C_arr_t_roc = roc(response = C_arr_t_true_values_vectorized, C_arr_t_pred_means_vectorized)
    
    C_arr_obs_auc_last = auc(C_arr_t_roc)
    C_arr_obs_auc_last
    
    C_predictions = as.factor(as.numeric(C_arr_t_pred_means_vectorized > 0.5))
    C_true_vals = as.factor(C_arr_t_true_values_vectorized )
    
    
    precision_last = posPredValue(C_predictions, C_true_vals, positive="1")
    recall_last = sensitivity(C_predictions, C_true_vals, positive="1")
    specificity_val_last = specificity(C_predictions, C_true_vals, negative="0")
    
    f1_score_last = (2 * precision_last * recall_last) / (precision_last + recall_last)
    
    results$C_arr_obs_auc_last = C_arr_obs_auc_last
    
    results$C_arr_obs_f1_score_last = f1_score_last
    
    results$C_arr_obs_precision_last = precision_last
    
    results$C_arr_obs_sensitivity_last = recall_last
    
    results$C_arr_obs_specificity_last = specificity_val_last
    
    # Node attribute predictions
    x_arr_t_preds_missing_means = apply(x_arr_t_preds_missing[, , which.max(discarded_time_points_idxs), , drop = FALSE], 
                                        c(1, 2, 3), mean)
    
    if (real_data) {
      true_x_arr_missing_vals = revised_missing_simulation_dataset$x_arr[, , max(discarded_time_points_idxs), drop = FALSE]
    }
    
    else {
      
      true_x_arr_missing_vals = revised_missing_simulation_dataset$out$x_arr[, , max(discarded_time_points_idxs), drop = FALSE]
      
    }
    
    x_arr_obs_mse_last = sum((x_arr_t_preds_missing_means - true_x_arr_missing_vals) ^ 2) / sum(true_x_arr_missing_vals ^ 2)
    x_arr_obs_mse_last
    
    results$x_arr_obs_mse_last = x_arr_obs_mse_last
    
    # Prediction Intervals for x
    x_arr_pred_coverages = array(NA, dim = c(J, m))
    
    x_arr_pred_lengths = array(NA, dim = c(J, m))
    
    print(dim(true_x_arr_missing_vals))
    
    for (k in 1:m) {
      for (j in 1:J) {
          
        cur_interval = quantile(x_arr_t_preds_missing[j, k, which.max(discarded_time_points_idxs), , drop = FALSE],
                                probs = c(0.025, 0.975))
        
        cur_x_lower = cur_interval[1]
        cur_x_upper = cur_interval[2]
        
        
        cur_x = true_x_arr_missing_vals[j, k, 1]
        
        if (cur_x >= cur_x_lower && cur_x <= cur_x_upper) {
          x_arr_pred_coverages[j, k] = 1
        }
        
        else {
          x_arr_pred_coverages[j, k] = 0
        }
        
        x_arr_pred_lengths[j, k] = cur_x_upper - cur_x_lower
          
      }
    }
    
    mean_coverage = mean(as.numeric(x_arr_pred_coverages))
    
    mean_length = mean(as.numeric(x_arr_pred_lengths))
    
    mean_coverage_by_attr = apply(x_arr_pred_coverages, 2, mean)
    
    mean_length_by_attr = apply(x_arr_pred_lengths, 2, mean)
    
    results$mean_coverage_last = mean_coverage
    
    results$mean_length_last = mean_length
    
    results$mean_coverage_by_attr_last = mean_coverage_by_attr
    
    results$mean_length_by_attr_last = mean_length_by_attr
    
  }
  
  
  if (all_but_last & length(discarded_time_points_idxs[-which.max(discarded_time_points_idxs)]) >= 1) {
    
    if (real_data) {
      C_arr_missing_all_but_last = revised_missing_simulation_dataset$C_arr[, , discarded_time_points_idxs[-which.max(discarded_time_points_idxs)],
                                                                    drop = FALSE]
    }
    
    else {
      
      C_arr_missing_all_but_last = revised_missing_simulation_dataset$out$C_arr[, , discarded_time_points_idxs[-which.max(discarded_time_points_idxs)],
                                                                   drop = FALSE]
      
    }
    
    C_arr_t_pred_means = apply(C_arr_t_preds_missing[, , -which.max(discarded_time_points_idxs), , drop = FALSE], 
                               c(1, 2, 3), mean, na.rm=TRUE)
    
    
    C_arr_t_pred_means_vectorized = as.numeric(C_arr_t_pred_means)
    C_arr_t_true_values_vectorized = as.numeric(C_arr_missing_all_but_last)
    
    print("third")
    print(unique(C_arr_t_true_values_vectorized))
    
    
    C_arr_t_roc = roc(response = C_arr_t_true_values_vectorized, C_arr_t_pred_means_vectorized)
    
    C_arr_obs_auc_all_but_last = auc(C_arr_t_roc)
    C_arr_obs_auc_all_but_last
    
    C_predictions = as.factor(as.numeric(C_arr_t_pred_means_vectorized > 0.5))
    C_true_vals = as.factor(C_arr_t_true_values_vectorized )
    
    
    precision_all_but_last = posPredValue(C_predictions, C_true_vals, positive="1")
    recall_all_but_last = sensitivity(C_predictions, C_true_vals, positive="1")
    specificity_val_all_but_last = specificity(C_predictions, C_true_vals, negative="0")
    
    f1_score_all_but_last = (2 * precision_all_but_last * recall_all_but_last) / (precision_all_but_last + recall_all_but_last)
    
    
    
    results$C_arr_obs_auc_all_but_last = C_arr_obs_auc_all_but_last
    
    results$C_arr_obs_f1_score_all_but_last = f1_score_all_but_last
    
    results$C_arr_obs_precision_all_but_last = precision_all_but_last
    
    results$C_arr_obs_sensitivity_all_but_last = recall_all_but_last
    
    results$C_arr_obs_specificity_all_but_last = specificity_val_all_but_last
    
    # Node attribute predictions
    x_arr_t_preds_missing = x_arr_t_preds_missing[, , -which.max(discarded_time_points_idxs), , drop = FALSE]
    
    x_arr_t_preds_missing_means = apply(x_arr_t_preds_missing, c(1, 2, 3), mean)
    
    
    if (real_data) {
      true_x_arr_missing_vals = revised_missing_simulation_dataset$x_arr[, , discarded_time_points_idxs[-which.max(discarded_time_points_idxs)],
                                                                         drop = FALSE]
    }
    
    else {
      
      true_x_arr_missing_vals = revised_missing_simulation_dataset$out$x_arr[, , discarded_time_points_idxs[-which.max(discarded_time_points_idxs)],
                                                                             drop = FALSE]
      
    }
    
    x_arr_obs_mse_all_but_last = sum((x_arr_t_preds_missing_means - true_x_arr_missing_vals) ^ 2) / sum(true_x_arr_missing_vals ^ 2)
    x_arr_obs_mse_all_but_last
    
    results$x_arr_obs_mse_all_but_last = x_arr_obs_mse_all_but_last
    
    
    
    # Prediction Intervals for x
    x_arr_pred_coverages = array(NA, dim = c(J, m, T_len_missing - 1))
    
    x_arr_pred_lengths = array(NA, dim = c(J, m, T_len_missing - 1))
    
    for (k in 1:m) {
      for (j in 1:J) {
        for (t in 1:(T_len_missing - 1)) {
          
          cur_interval = quantile(x_arr_t_preds_missing[j, k, t, ], probs = c(0.025, 0.975))
          
          cur_x_lower = cur_interval[1]
          cur_x_upper = cur_interval[2]
          
          
          cur_x = true_x_arr_missing_vals[j, k, t]
          
          if (cur_x >= cur_x_lower && cur_x <= cur_x_upper) {
            x_arr_pred_coverages[j, k, t] = 1
          }
          
          else {
            x_arr_pred_coverages[j, k, t] = 0
          }
          
          x_arr_pred_lengths[j, k, t] = cur_x_upper - cur_x_lower
          
        }
      }
    }
    
    mean_coverage = mean(as.numeric(x_arr_pred_coverages))
    
    mean_length = mean(as.numeric(x_arr_pred_lengths))
    
    mean_coverage_by_attr = apply(x_arr_pred_coverages, 2, mean)
    
    mean_length_by_attr = apply(x_arr_pred_lengths, 2, mean)
    
    results$mean_coverage_all_but_last = mean_coverage
    
    results$mean_length_all_but_last = mean_length
    
    results$mean_coverage_by_attr_all_but_last = mean_coverage_by_attr
    
    results$mean_length_by_attr_all_but_last = mean_length_by_attr
    
    
    
    
    
    # Get Metrics for edges that are missing
    
    if (real_data) {
      C_arr_discarded_times = revised_missing_simulation_dataset$out$missing_multilayer_network[, , discarded_time_points_idxs[-which.max(discarded_time_points_idxs)],
                                                                            drop = FALSE]
      

      
    }
    
    else {
      
      C_arr_discarded_times = revised_missing_simulation_dataset$out$masked_C_arr[, , discarded_time_points_idxs[-which.max(discarded_time_points_idxs)],
                                                                                drop = FALSE]
      

    }
    
    
    missing_edge_indices = which(is.na(C_arr_discarded_times), arr.ind = TRUE)
    
    true_C_arr_missing_edges = C_arr_missing_all_but_last[missing_edge_indices]
    
    
    pred_C_arr_missing_edges = C_arr_t_pred_means[missing_edge_indices]
    
    
    
    
    C_arr_t_pred_means_vectorized = as.numeric(pred_C_arr_missing_edges)
    C_arr_t_true_values_vectorized = as.numeric(true_C_arr_missing_edges)
    
    print("fourth")
    print(unique(C_arr_t_true_values_vectorized))
    
    C_arr_t_roc = roc(response = C_arr_t_true_values_vectorized, C_arr_t_pred_means_vectorized)
    
    C_arr_obs_auc_missing_edges = auc(C_arr_t_roc)
    C_arr_obs_auc_missing_edges
    
    C_predictions = as.factor(as.numeric(C_arr_t_pred_means_vectorized > 0.5))
    C_true_vals = as.factor(C_arr_t_true_values_vectorized)
    
    
    precision_missing_edges = posPredValue(C_predictions, C_true_vals, positive="1")
    recall_missing_edges = sensitivity(C_predictions, C_true_vals, positive="1")
    specificity_val_missing_edges = specificity(C_predictions, C_true_vals, negative="0")
    
    f1_score_missing_edges = (2 * precision_missing_edges * recall_missing_edges) / (precision_missing_edges + recall_missing_edges)
    
    
    
    results$C_arr_obs_auc_missing_edges = C_arr_obs_auc_missing_edges
    
    results$C_arr_obs_f1_score_missing_edges = f1_score_missing_edges
    
    results$C_arr_obs_precision_missing_edges = precision_missing_edges
    
    results$C_arr_obs_sensitivity_missing_edges = recall_missing_edges
    
    results$C_arr_obs_specificity_missing_edges = specificity_val_missing_edges
    
    
    
  }
  
  results_filename = paste("outputs/out_samp_pred_J_", 
                           J,
                           "_L", L,
                           "_R", fitted_R,
                           "_R_zeta", fitted_R_zeta,
                           "_m", m,
                           "_T_len", T_len,
                           "_phi", phi, 
                           "_epsilon", epsilon,
                           "_TrueR", R, 
                           "_TrueF", sim_gen_nngp_L,
                           "_niter", niter, "_n_burnin", n_burn_in,
                           "_nngpL", nngp_L, "_nngp_gen", nngp_gen,
                           "_tergm_", tergm_gen,
                           "_seed", seed, "_real_data", real_data,
                           #"_T_prop", paste(time_props_missing, collapse = "_"),
                           #"_C_prop", paste(edge_props_missing, collapse = "_"),
                           ".RData", sep = "")
  
  print(results_filename)
  
  save(results, file = results_filename)
  
}







# For simulations
model_generation_running_and_testing <- function(L, J, m, T_len = NULL,
                                                 time_grid,
                                                 R, 
                                     phi = NULL, epsilon = NULL, time_props_missing,
                                     edge_props_missing, nngp_gen = FALSE,
                                     tergm_gen = FALSE,
                                     nngp_L = 1, 
                                     sim_nngp_L = NULL,
                                     sigma2_w = 1,
                                     sigma2_b = 1,
                                     niter, fitted_R, fitted_R_zeta=fitted_R,
                                     seed = 1234,
                                     n_burn_in, n_thinning,
                                     remove_last_point = FALSE,
                                     only_use_last = FALSE,
                                     complexity = FALSE,
                                     save_outputs = FALSE,
                                     run_parallel = FALSE,
                                     n_workers = NULL,
                                     sigma2_b_grid = seq(0.01, 0.1, by = 0.01),
                                     sigma2_w_grid = seq(0.01, 0.1, by = 0.01),
                                     use_mh = FALSE,
                                     b_proposal_var = 0.1,
                                     w_proposal_var = 0.1) {
  
  
  T_len = length(time_grid)
  
  
  # Generate Data
  
  if (nngp_gen) {
    
    out = incomplete_graphs_sim_data_gen_nngp_layerwise_missingness(L = L, J = J, m = m,
                                                                    time_grid = time_grid,
                                                                    T_len = T_len, R = R,
                                                                    time_props_missing = time_props_missing,
                                                                    edge_props_missing = edge_props_missing, 
                                                                    remove_last_point = remove_last_point,
                                                                    nngp_L = sim_nngp_L,
                                                                    sigma2_w = sigma2_w,
                                                                    sigma2_b = sigma2_b,
                                                                    seed = seed)
    
    
  }
  
  else if (tergm_gen) {
    
    out = incomplete_graphs_sim_data_gen_tergm_layerwise_missingness(L = L, J = J, m = m,
                                                                     time_grid = time_grid,
                                                                     T_len = T_len, R = R,
                                                                     time_props_missing = time_props_missing,
                                                                     edge_props_missing = edge_props_missing, 
                                                                     nngp_L = sim_nngp_L, sigma2_w = sigma2_w,
                                                                     sigma2_b = sigma2_b,
                                                                     remove_last_point = remove_last_point,
                                                                     seed = seed)
    
  }
  
  else {
    out = incomplete_graphs_sim_data_gen_layerwise_missingness(L = L, 
                                         J = J,
                                         m = m,
                                         time_grid = time_grid,
                                         T_len = T_len, 
                                         R = R,
                                         phi = phi, 
                                         epsilon = epsilon,
                                         time_props_missing = time_props_missing,
                                         edge_props_missing = edge_props_missing, 
                                         remove_last_point = remove_last_point,
                                         seed = seed)
  
  }
  
  
  # Standardize the nodal attributes
  full_T_len = dim(out$x_arr)[3]
  
  
  x_out = standardize_nodal_attributes(x_arr = out$x_arr, train_times = c(1:(full_T_len - 1)),
                                       test_times = c(full_T_len))
  
  # Concactenate standardized arrays of nodal attributes
  std_x_arr = array(NA, dim = c(J, m, full_T_len))
  
  std_x_arr[, , c(1:(full_T_len - 1))] = x_out$std_x_arr_train
  std_x_arr[, , full_T_len] = x_out$std_x_arr_test
    
  out$x_arr = std_x_arr
  
  
  discarded_time_points_idxs = out$missing_times
  
  
  C_arr_missing = out$C_arr[, , -discarded_time_points_idxs, drop = FALSE]
  x_arr_missing = out$x_arr[, , -discarded_time_points_idxs, drop = FALSE]
  
  obs_time_grid = time_grid[-discarded_time_points_idxs]
  missing_time_grid = time_grid[discarded_time_points_idxs]
  
  
  revised_missing_simulation_dataset = list()
  
  revised_missing_simulation_dataset$out = out
  revised_missing_simulation_dataset$C_arr_missing = C_arr_missing
  revised_missing_simulation_dataset$x_arr_missing = x_arr_missing
  revised_missing_simulation_dataset$obs_time_grid = obs_time_grid
  revised_missing_simulation_dataset$missing_time_grid = missing_time_grid
  revised_missing_simulation_dataset$discarded_time_points_idxs = discarded_time_points_idxs
  

  
  data_filename = paste("../data/revised_missing_simulation_dataset_J_", J, 
                        "_T_", T_len,
                        "_L_", L,
                        "_m_", m,
                        "_R_", R,
                        "_phi_", phi, 
                        "_epsilon_", epsilon,  
                        "_seed_", seed,
                        "_nngp_gen_", nngp_gen,
                        "_tergm_", tergm_gen,
                        "_sim_nngp_L_", sim_nngp_L,
                        #"_time_props_missing_", paste(time_props_missing, collapse = "_"),
                        #"_edge_props_missing_", paste(edge_props_missing, collapse = "_"),
                        ".RData", sep = "")
  
  print(data_filename) 
  
  # Only save if the data does not already exist 
  
  if (!file.exists(data_filename)) {
    save(revised_missing_simulation_dataset, file = data_filename)
  }
  
  # Otherwise, load the already created data
  else {
    load(data_filename)
  }
  
  # Run model
  
  # Measure elapsed time for simulated data model run
  
  start_time = proc.time()
  
  model_outputs = dyna_hidden_graph_model_complete(C_arr = revised_missing_simulation_dataset$C_arr_missing,
                                   x_arr = revised_missing_simulation_dataset$x_arr_missing,
                                   time_grid = revised_missing_simulation_dataset$obs_time_grid,
                                   R = fitted_R,
                                   R_zeta = fitted_R_zeta,
                                   nngp_L = nngp_L,
                                   phi = phi,
                                   epsilon = epsilon,
                                   True_R = R,
                                   True_F = sim_nngp_L,
                                   niter = niter, 
                                   nngp_gen = nngp_gen,
                                   tergm_gen = tergm_gen,
                                   real_data = FALSE,
                                   time_props_missing = time_props_missing,
                                   edge_props_missing = edge_props_missing, seed = seed,
                                   save_outputs = save_outputs,
                                   run_parallel = run_parallel,
                                   n_workers = n_workers,
                                   sigma2_b_grid = sigma2_b_grid,
                                   sigma2_w_grid = sigma2_w_grid,
                                   use_mh = use_mh,
                                   b_proposal_var = b_proposal_var,
                                   w_proposal_var = w_proposal_var)
  
  elapsed_time = proc.time() - start_time
  
  elapsed_time_results = list(elapsed_time = elapsed_time)
  
  elapsed_time_results$elapsed_time_per_iter = elapsed_time_results$elapsed_time / niter
  
  # mcmc_filename = paste("outputs/mcmc_out_J_", 
  #                       J,
  #                       "_L_", L,
  #                       "_R_", fitted_R,
  #                       "_m_", m,
  #                       "_T_len_", T_len - length(discarded_time_points_idxs),
  #                       "_phi_", phi, 
  #                       "_epsilon_", epsilon,
  #                       "_TrueR_", R, 
  #                       "_TrueF_", sim_nngp_L,
  #                       "_niter_", niter, "_nngpL_", nngp_L,
  #                       "_seed_", seed, "_nngp_gen_", nngp_gen,
  #                       "_tergm_", tergm_gen,
  #                       "_real_data_", FALSE,
  #                       #"_T_prop_", paste(time_props_missing, collapse = "_"),
  #                       #"_C_prop_", paste(edge_props_missing, collapse = "_"),
  #                       ".RData", sep = "")
  # 
  # print(mcmc_filename)
  
  #load(mcmc_filename)
  
  
  elapsed_time_filename = paste("outputs/mcmc_elapsed_t_J_", 
                        J,
                        "_L_", L,
                        "_R_", fitted_R,
                        "_R_zeta_", fitted_R_zeta,
                        "_m_", m,
                        "_T_len_", T_len - length(discarded_time_points_idxs),
                        "_phi_", phi, 
                        "_epsilon_", epsilon,
                        "_TrueR_", R, 
                        "_TrueF_", sim_nngp_L,
                        "_niter_", niter, "_nngpL_", nngp_L,
                        "_seed_", seed, "_nngp_gen_", nngp_gen,
                        "_tergm_", tergm_gen,
                        "_real_data_", FALSE,
                        #"_T_prop_", paste(time_props_missing, collapse = "_"),
                        #"_C_prop_", paste(edge_props_missing, collapse = "_"),
                        ".RData", sep = "")
  
  
  #save(elapsed_time_results, file = elapsed_time_filename)
  
  if (complexity) {

    print(elapsed_time_filename)

    return(elapsed_time_results)

  }
  
  
  # Get prediction results
  model_in_sample_predictions(revised_missing_simulation_dataset = revised_missing_simulation_dataset,
                                  #data_filename = data_filename,
                                     model_outputs = model_outputs,
                                     #mcmc_filename = mcmc_filename,
                                     L = L, J = J, m = m, T_len = T_len,
                                     R = R, sim_gen_nngp_L = sim_nngp_L,
                                     phi = phi, epsilon = epsilon,
                                     time_props_missing = time_props_missing,
                                     edge_props_missing = edge_props_missing, 
                                     nngp_gen = nngp_gen,
                                     tergm_gen = tergm_gen,
                                     nngp_L = nngp_L, 
                                     niter = niter, fitted_R = fitted_R, 
                                    fitted_R_zeta = fitted_R_zeta,
                                     seed = seed,
                                     n_burn_in = n_burn_in, n_thinning = n_thinning,
                                     real_data = FALSE)

  model_out_of_sample_predictions(revised_missing_simulation_dataset = revised_missing_simulation_dataset,
                                                #data_filename = data_filename,
                                                model_outputs = model_outputs,
                                               #mcmc_filename = mcmc_filename,
                                               L = L, J = J, m = m, T_len = T_len,
                                               R = R, sim_gen_nngp_L = sim_nngp_L,
                                               phi = phi, epsilon = epsilon,
                                               time_props_missing = time_props_missing,
                                               edge_props_missing = edge_props_missing,
                                               nngp_gen = nngp_gen,
                                               tergm_gen = tergm_gen,
                                               nngp_L = nngp_L,
                                               niter = niter, fitted_R = fitted_R,
                                              fitted_R_zeta = fitted_R_zeta,
                                               seed = seed,
                                               n_burn_in = n_burn_in, n_thinning = n_thinning,
                                               real_data = FALSE,
                                               only_last = TRUE,
                                               all_but_last = TRUE,
                                               only_use_last = only_use_last)

  
}







# For real data
model_real_data_running_and_testing <- function(C, x_arr, obs_years,
                                                node_id_list,
                                                time_props_missing,
                                                edge_props_missing,
                                                nngp_L = 1, 
                                                niter, fitted_R, 
                                                fitted_R_zeta = fitted_R,
                                                seed = 1234,
                                                n_burn_in, n_thinning,
                                                remove_last_point = FALSE,
                                                only_use_last = FALSE,
                                                full_real_data = FALSE,
                                                save_outputs = FALSE,
                                                run_parallel = FALSE,
                                                n_workers = NULL,
                                                sigma2_b_grid = seq(0.01, 0.1, by = 0.01),
                                                sigma2_w_grid = seq(0.01, 0.1, by = 0.01),
                                                use_mh = FALSE,
                                                b_proposal_var = 0.1,
                                                w_proposal_var = 0.1) {
  

  T_len = length(obs_years)
  
  J = length(node_id_list)
  
  L = dim(C)[1]
  
  m = dim(x_arr)[2]
  
  time_grid = real_data$obs_years - min(real_data$obs_years) + 1
  
  
  # Standardize Nodal Attributes
  
  cat_features = c(1)
  binary_features = c(3:13)
  
  x_out = standardize_nodal_attributes(x_arr = x_arr, train_times = c(1:(T_len - 1)), test_times = c(T_len),
                                       cat_features = cat_features, binary_features = binary_features)
  
  m = dim(x_out$std_x_arr_train)[2]
  
  # Concactenate x_out arrays
  std_x_arr = array(NA, dim = c(J, m, T_len))
  
  std_x_arr[, , c(1:(T_len - 1))] = x_out$std_x_arr_train
  std_x_arr[, , T_len] = x_out$std_x_arr_test
  
  # Standardize x_arr using standard scaling and one-hot encoding
  
  
  
  if (full_real_data) {  
    
  
    
    discarded_time_points_idxs = c()
    
    #print(discarded_time_points_idxs)
    
    C_arr_missing = C
    x_arr_missing = std_x_arr
    
    time_grid = c(1:dim(C)[3])
    
    
    revised_missing_simulation_dataset = list()
    
    revised_missing_simulation_dataset$C_arr = C
    revised_missing_simulation_dataset$x_arr = std_x_arr
    
    revised_missing_simulation_dataset$out = out
    revised_missing_simulation_dataset$C_arr_missing = C_arr_missing
    revised_missing_simulation_dataset$x_arr_missing = x_arr_missing
    revised_missing_simulation_dataset$time_grid = time_grid
    revised_missing_simulation_dataset$discarded_time_points_idxs = discarded_time_points_idxs 
    
  }
  
  else {
    
    print("here")
    
    out = random_edge_removal_diff_time_and_edge_props(multilayer_network = C,
                                                       time_props_missing = time_props_missing,
                                                       edge_props_missing = edge_props_missing,
                                                       obs_years = time_grid, 
                                                       remove_last_point = remove_last_point,
                                                       seed = seed)  
    
    
    discarded_time_points_idxs = out$missing_years
    
    
    #print(discarded_time_points_idxs)
    
    C_arr_missing = C[, , -discarded_time_points_idxs, drop = FALSE]
    x_arr_missing = std_x_arr[, , -discarded_time_points_idxs, drop = FALSE]
    
    time_grid = c(1:dim(C)[3])[-discarded_time_points_idxs]
    
    
    revised_missing_simulation_dataset = list()
    
    revised_missing_simulation_dataset$C_arr = C
    revised_missing_simulation_dataset$x_arr = std_x_arr
    
    revised_missing_simulation_dataset$out = out
    revised_missing_simulation_dataset$C_arr_missing = C_arr_missing
    revised_missing_simulation_dataset$x_arr_missing = x_arr_missing
    revised_missing_simulation_dataset$time_grid = time_grid
    revised_missing_simulation_dataset$discarded_time_points_idxs = discarded_time_points_idxs 
    
  }
  
  data_filename = paste("outputs/revised_missing_real_dataset_J_", J, 
                        "_T_", T_len,
                        "_L_", L,
                        "_m_", m,
                        "_seed_", seed,
                        #"_time_props_missing_", paste(time_props_missing, collapse = "_"),
                        #"_edge_props_missing_", paste(edge_props_missing, collapse = "_"),
                        ".RData", sep = "")
  
  print(data_filename) 
  
  if (!file.exists(data_filename)) {
    save(revised_missing_simulation_dataset, file = data_filename)
  }
  
  #load(data_filename)

  
  
  # Run model
  
  print("here2")
  
  model_outputs = dyna_hidden_graph_model_complete(C_arr = revised_missing_simulation_dataset$C_arr_missing,
                                   x_arr = revised_missing_simulation_dataset$x_arr_missing,
                                   time_grid = revised_missing_simulation_dataset$time_grid,
                                   R = fitted_R,
                                   R_zeta = fitted_R_zeta,
                                   nngp_L = nngp_L,
                                   niter = niter,
                                   real_data = TRUE,
                                   time_props_missing = time_props_missing,
                                   edge_props_missing = edge_props_missing, seed = seed,
                                   save_outputs = save_outputs,
                                   run_parallel = run_parallel,
                                   n_workers = n_workers,
                                   sigma2_b_grid = sigma2_b_grid,
                                   sigma2_w_grid = sigma2_w_grid,
                                   use_mh = use_mh,
                                   b_proposal_var = b_proposal_var,
                                   w_proposal_var = w_proposal_var)
  
  # mcmc_filename = paste("outputs/mcmc_out_J_", 
  #                       J,
  #                       "_L_", L,
  #                       "_R_", fitted_R,
  #                       "_m_", m,
  #                       "_T_len_", T_len - length(discarded_time_points_idxs),
  #                       "_phi_", NULL, 
  #                       "_epsilon_", NULL,
  #                       "_TrueR_", NULL, 
  #                       "_TrueF_", NULL,
  #                       "_niter_", niter, "_nngpL_", nngp_L,
  #                       "_seed_", seed, "_nngp_gen_", NULL,
  #                       "_tergm_", NULL,
  #                       "_real_data_", TRUE,
  #                       #"_T_prop_", paste(time_props_missing, collapse = "_"),
  #                       #"_C_prop_", paste(edge_props_missing, collapse = "_"),
  #                       ".RData", sep = "")
  # 
  # print(mcmc_filename)
  
  #load(mcmc_filename)
  
  
  # Get prediction results
  model_in_sample_predictions(revised_missing_simulation_dataset = revised_missing_simulation_dataset,
                                               model_outputs = model_outputs,
                                               #mcmc_filename = mcmc_filename,
                                               L = L, J = J, m = m, T_len = T_len,
                                               time_props_missing = time_props_missing,
                                               edge_props_missing = edge_props_missing,
                                               nngp_L = nngp_L, 
                                               niter = niter, fitted_R = fitted_R, 
                                              fitted_R_zeta = fitted_R_zeta,
                                               seed = seed,
                                               n_burn_in = n_burn_in, n_thinning = n_thinning,
                                               real_data = TRUE,
                                               full_real_data = full_real_data)
  
  
  if (!full_real_data) {  
   
  
    model_out_of_sample_predictions(revised_missing_simulation_dataset = revised_missing_simulation_dataset,
                                                 model_outputs = model_outputs,
                                                 #mcmc_filename = mcmc_filename,
                                                 L = L, J = J, m = m, T_len = T_len,
                                                 time_props_missing = time_props_missing,
                                                 edge_props_missing = edge_props_missing,
                                                 nngp_L = nngp_L,
                                                 niter = niter, fitted_R = fitted_R,
                                                fitted_R_zeta = fitted_R_zeta,
                                                 seed = seed,
                                                 n_burn_in = n_burn_in, n_thinning = n_thinning,
                                                 real_data = TRUE,
                                                 only_last = TRUE,
                                                 all_but_last = TRUE,
                                                 only_use_last = only_use_last)

  }
    
}




