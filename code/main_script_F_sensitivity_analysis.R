
# Libraries for running in parallel
library(doParallel)
library(foreach)


# Import the functions for generating data, running the model, and obtaining results
source("model_generation_running_and_testing_functions.R")
source("F_sensitivity_results_function.R")




# NN-GP Data Generation Simulation Values used in each of the three cases

# Number of Graph Layers
L = 2

# Number of nodes in the graphs on each layer
J = 20

# Number of attributes for each node
m = 2

# Number of time points (with one being out-of-sample)
T_len = 21


# Hyperparameters for NN-GP Data Generation in Cases 1,2, and 3 

# True dimensions of latent vectors used for data generation, R*
R_vals = c(4, 3, 2)

# True number of NN-GP layers used for data generation, F*
sim_F_vals = c(1, 2, 3)

# Concatenate the corresponding true R and F values into a list of vectors
R_and_sim_F_vals = I(list(c(R_vals[1], sim_F_vals[1]), 
                          c(R_vals[2], sim_F_vals[2]), 
                          c(R_vals[3], sim_F_vals[3])))


# Values for F, the number of NN-GP layers, used to fit DJL in each case
# of the sensitivity analysis
fitted_F_vals = c(1:4)

# Random seeds for the replications of each case
seed_vals = c(22, 25, 26)


# Create a grid using the true R and F for each case, the value of F used to fit DJL,
# and the values for the seeds in each replication
param_grid = expand.grid(R_and_sim_F_vals = R_and_sim_F_vals,
                   fitted_F = fitted_F_vals,
                   seed = seed_vals)


# Number of MCMC iterations used to fit DJL
niter = 3000

# Number of iterations to discard as burn-in after fitting DJL
n_burn_in = 2000


# Dimension of the latent vectors used for fitting DJL
fitted_R = 4



# Create a directory named "outputs" to store results if it does not already exist
if (!dir.exists("outputs")) {
  dir.create("outputs")
}


# Configure parallel R environment

# Number of cores to use, if left equal to NULL, all available cores will be used
n_workers = 6

if (is.null(n_workers)) {
  n_workers = detectCores()
}

print(paste("Number of cores: ", n_workers, sep = ""))

cl <- makeCluster(n_workers) 
registerDoParallel(cl) 

on.exit(stopCluster(cl), add = TRUE)


parallel_runs <- foreach (case = 1:nrow(param_grid)) %dopar% {
  
  source("model_generation_running_and_testing_functions.R")
  

  cur_params = param_grid[case, ]
  

  
  model_generation_running_and_testing(L = L, J = J, m = m, T_len = T_len,
                                       R = cur_params$R_and_sim_F_vals[[1]][1],
                                       time_grid = seq(from = 1, to = T_len, length.out = T_len),
                                       time_props_missing = rep(0.1, L),
                                       edge_props_missing = rep(0.25, L), nngp_gen = TRUE,
                                       tergm_gen = FALSE,
                                       nngp_L = cur_params$fitted_F,
                                       sim_nngp_L = cur_params$R_and_sim_F_vals[[1]][2], 
                                       sigma2_w = 0.4,
                                       sigma2_b = 0.01,
                                       niter = niter, fitted_R = fitted_R,
                                       fitted_R_zeta = fitted_R,
                                       seed = cur_params$seed,
                                       n_burn_in = n_burn_in, n_thinning = 1,
                                       remove_last_point = TRUE)

}
  
  
stopCluster(cl)


# Aggregated results for the first case with R* = 4, F* = 1
results_case_1 = F_sensitivity_results(fitted_F_vals = fitted_F_vals,
                                       fitted_R = fitted_R,
                                       seed_vals = seed_vals,
                                       L = L, J = J, m = m,
                                       R = R_vals[1],
                                       sim_F = sim_F_vals[1],
                                      T_len = T_len, niter = niter, n_burn_in = n_burn_in) 

# Aggregated results for the second case with R* = 3, F* = 2
results_case_2 = F_sensitivity_results(fitted_F_vals = fitted_F_vals,
                                       fitted_R = fitted_R,
                                       seed_vals = seed_vals,
                                       L = L, J = J, m = m,
                                       R = R_vals[2],
                                       sim_F = sim_F_vals[2],
                                       T_len = T_len, niter = niter, n_burn_in = n_burn_in) 

# Aggregated results for the third case with R* = 2, F* = 3
results_case_3 = F_sensitivity_results(fitted_F_vals = fitted_F_vals,
                                       fitted_R = fitted_R,
                                       seed_vals = seed_vals,
                                       L = L, J = J, m = m,
                                       R = R_vals[3],
                                       sim_F = sim_F_vals[3],
                                       T_len = T_len, niter = niter, n_burn_in = n_burn_in) 

# Print the resulting tables

# Case 1: R* = 4, F* = 1
results_case_1

# Case 2: R* = 3, F* = 2
results_case_2

# Case 3: R* = 2, F* = 3
results_case_3

