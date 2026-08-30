

###### Function
# F_sensitivity_results: Aggregates results from different replications 
# of the analysis for sensitivity to F under given parameters

###### Arguments
# fitted_F_vals: Vector of values of F (number of NN-GP layers) used to fit DJL
# fitted_R: Value of R (dimension of latent vectors) used to fit DJL
# seed_vals: Vector of random seeds used for replications
# L: The number of graph layers for data generation
# J: The number of nodes in each graph layer for data generation
# m: The number of nodal attributes for data generation
# R: The true dimension of simulated latent vectors
# sim_F: The true number of NN-GP layers used for generating data
# T_len: The total number of time points (including out-of-sample) for data generation
# niter: The number of MCMC iterations used to fit DJL
# n_burn_in: The number of MCMC iterations discarded as burn-in after fitting DJL

###### Returns
# results: List of dataframes containing AUC, MSPE, 95% PI Coverages, and 95% PI Lengths

F_sensitivity_results <- function(fitted_F_vals, fitted_R, seed_vals, L, J, m, R, sim_F,
                                  T_len, niter, n_burn_in) {
  

  
  # Initialize matrices to store results
  # Rows correspond to replications
  # Columns correspond to different values for F, 
  # the number of NN-GP layers used to fit DJL
  in_sample_auc_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  missing_auc_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  out_sample_auc_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  
  in_sample_mspe_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  missing_mspe_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  out_sample_mspe_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  
  in_sample_cov_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  missing_cov_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  out_sample_cov_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  
  in_sample_len_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  missing_len_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  out_sample_len_vals = matrix(NA, nrow = length(seed_vals), ncol = length(fitted_F_vals))
  
  
  
  # Indicators for the simulation scenario
  nngp_gen = TRUE
  tergm_gen = FALSE
  
  # These values are not used in NN-GP simulated data generation, so they are left null
  phi = NULL
  epsilon = NULL
  
  # Values for layerwise missingness and proportion of missing edges
  # time_props_missing = rep(0.1, L)
  # edge_props_missing = rep(0.25, L)
  

  
  
  for (f in 1:length(seed_vals)) {
    
    

    
    data_filename = paste("../data/revised_missing_simulation_dataset_J_", J,
                          "_T_", T_len,
                          "_L_", L,
                          "_m_", m,
                          "_R_", R,
                          "_phi_", phi, 
                          "_epsilon_", epsilon,  
                          "_seed_", seed_vals[f],
                          "_nngp_gen_", nngp_gen,
                          "_tergm_", tergm_gen,
                          "_sim_nngp_L_", sim_F,
                          ".RData", sep = "")
    
    
    load(data_filename)
    discarded_time_points = revised_missing_simulation_dataset$discarded_time_points
    
    
    
    
    for (i in 1:length(fitted_F_vals)) {
      
      # Current number of NN-GP layers for model fitting
      nngp_L = fitted_F_vals[i]
      
      
      in_sample_filename = paste("outputs/in_samp_pred_J_", 
                                 J,
                                 "_L", L,
                                 "_R", fitted_R,
                                 "_R_zeta", fitted_R,
                                 "_m", m,
                                 "_T_len", T_len - length(discarded_time_points),
                                 "_phi", phi, 
                                 "_epsilon", epsilon,
                                 "_TrueR", R, 
                                 "_TrueF", sim_F,
                                 "_niter", niter, "_n_burnin", n_burn_in,
                                 "_nngpL", nngp_L, "_nngp_gen", nngp_gen,
                                 "_tergm_", tergm_gen,
                                 "_seed", seed_vals[f], "_real_data", FALSE,
                                 "_full_real_",
                                 ".RData", sep = "")
      
      
      
      
      load(in_sample_filename)
      
      in_sample_auc_vals[f, i] = results$C_arr_obs_auc
      
      in_sample_mspe_vals[f, i] = results$x_arr_obs_mse
      
      in_sample_cov_vals[f, i] = results$mean_coverage
      in_sample_len_vals[f, i] = results$mean_length
      

      
      out_sample_filename = paste("outputs/out_samp_pred_J_", J,
                                  "_L", L,
                                  "_R", fitted_R,
                                  "_R_zeta", fitted_R,
                                  "_m", m,
                                  "_T_len", T_len - length(discarded_time_points),
                                  "_phi", phi, 
                                  "_epsilon", epsilon,
                                  "_TrueR", R, 
                                  "_TrueF", sim_F,
                                  "_niter", niter, "_n_burnin", n_burn_in,
                                  "_nngpL", nngp_L, "_nngp_gen", nngp_gen,
                                  "_tergm_", tergm_gen,
                                  "_seed", seed_vals[f], "_real_data", FALSE,
                                  ".RData", sep = "")
      
      
      
      load(out_sample_filename)
      
      
      missing_auc_vals[f, i] = results$C_arr_obs_auc_missing_edges
      out_sample_auc_vals[f, i] = results$C_arr_obs_auc_last
      
      missing_mspe_vals[f, i] = results$x_arr_obs_mse_all_but_last
      out_sample_mspe_vals[f, i] = results$x_arr_obs_mse_last
      
      missing_cov_vals[f, i] = results$mean_coverage_all_but_last
      out_sample_cov_vals[f, i] = results$mean_coverage_last
      
      missing_len_vals[f, i] = results$mean_length_all_but_last
      out_sample_len_vals[f, i] = results$mean_length_last
      
    }
    
  }
  
  
  # Average the results over replications
  
  mean_in_sample_auc_vals = colMeans(in_sample_auc_vals)
  mean_missing_auc_vals = colMeans(missing_auc_vals)
  mean_out_sample_auc_vals = colMeans(out_sample_auc_vals)
  
  mean_in_sample_mspe_vals = colMeans(in_sample_mspe_vals)
  mean_missing_mspe_vals = colMeans(missing_mspe_vals)
  mean_out_sample_mspe_vals = colMeans(out_sample_mspe_vals)
  
  mean_in_sample_cov_vals = colMeans(in_sample_cov_vals)
  mean_missing_cov_vals = colMeans(missing_cov_vals)
  mean_out_sample_cov_vals = colMeans(out_sample_cov_vals)
  
  mean_in_sample_len_vals = colMeans(in_sample_len_vals)
  mean_missing_len_vals = colMeans(missing_len_vals)
  mean_out_sample_len_vals = colMeans(out_sample_len_vals)
  
  
  # AUC Results
  in_auc = round(mean_in_sample_auc_vals, 4)
  miss_auc = round(mean_missing_auc_vals, 4)
  out_auc = round(mean_out_sample_auc_vals, 4) 
  
  auc_df = data.frame(rbind(in_auc, miss_auc, out_auc))
  rownames(auc_df) = c("in", "mis", "out")
  colnames(auc_df) = fitted_F_vals
  
  
  # MSPE Results
  in_mspe = round(mean_in_sample_mspe_vals, 4)
  miss_mspe = round(mean_missing_mspe_vals, 4)
  out_mspe = round(mean_out_sample_mspe_vals, 4)
  
  mspe_df = data.frame(rbind(in_mspe, miss_mspe, out_mspe))
  rownames(mspe_df) = c("in", "mis", "out")
  colnames(mspe_df) = fitted_F_vals
  
  
  # 95% Prediction Interval Coverage Results
  in_cov = round(mean_in_sample_cov_vals, 4)
  miss_cov = round(mean_missing_cov_vals, 4)
  out_cov = round(mean_out_sample_cov_vals, 4) 
  
  cov_df = data.frame(rbind(in_cov, miss_cov, out_cov))
  rownames(cov_df) = c("in", "mis", "out")
  colnames(cov_df) = fitted_F_vals
  
  
  # 95% Prediction Interval Length Results
  in_len = round(mean_in_sample_len_vals, 4)
  miss_len = round(mean_missing_len_vals, 4)
  out_len = round(mean_out_sample_len_vals, 4)
  
  len_df = data.frame(rbind(in_len, miss_len, out_len))
  rownames(len_df) = c("in", "mis", "out")
  colnames(len_df) = fitted_F_vals
  
  # Store results
  results = list(auc_df = auc_df,
                  mspe_df = mspe_df,
                  cov_df = cov_df,
                  len_df = len_df)
  
  return(results)

}