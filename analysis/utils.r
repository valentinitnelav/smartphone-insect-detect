#' Permutation Test
#'
#' Performs a permutation test to compare the difference in means or medians
#' between two samples.
#'
#' @param sample1 Numeric vector with the first sample.
#' @param sample2 Numeric vector with the second sample.
#' @param n_sim Integer. Number of permutations to perform. Default is 1000.
#' @param stat Character. The statistic to compare: either "mean" or "median". Default is "mean".
#'
#' @return A list with the following components:
#' \describe{
#'   \item{observed_difference}{The observed difference ("sample2 - sample 1") in the statistic between the two samples.}
#'   \item{expected_difference}{The expected difference under the null hypothesis. This is the metric of the null distribution.}
#'   \item{null_distribution}{The entire null distribution of permuted differences.}
#'   \item{confidence_interval}{The 95% confidence interval (quantile based) of the null distribution.}
#'   \item{p_value}{The two-tailed p-value indicating if the observed difference is statistically significant.}
#' }
#'
#' @examples
#' library(data.table)
#' library(coin)
#' library(magrittr)
#' 
#' set.seed(123)
#' sample1 <- rnorm(30, mean = 5, sd = 2)
#' sample2 <- rnorm(30, mean = 6, sd = 2)
#' result <- permutation_test(sample1, sample2, n_sim = 1000, stat = "mean")
#' print(result)
permutation_test <- function(sample1, sample2, n_sim = 1000, stat = "mean") {
  if (!stat %in% c("mean", "median")) {
    stop("Invalid statistic. Use 'mean' or 'median'.")
  }
  
  # For double checking, uses also the approximative (Monte Carlo) Fisher-Pitman test.
  # It wasn't clear to me if this test allows the comparison of means and
  # medians too, but its results turn out to align well with the results of the
  # hard coded permutations further implemented.
  # Call it here before altering dat_obs with permutations.
  dat_obs <- data.table(
    value = c(sample1, sample2),
    group = factor(c(rep("s1", length(sample1)), 
                     rep("s2", length(sample2))))
  )
  set.seed(123)
  test_fp <- oneway_test(value ~ group, 
                         data = dat_obs, 
                         distribution = approximate(nresample = 1000))
  p_value_fp <- pvalue(test_fp) %>% as.numeric()
  z_value_fp <- test_fp@statistic@teststatistic
  
  
  # Continue with permutation implementation.
  
  # Place samples into data frame for processing.
  dat_obs <- data.table(
    value = c(sample1, sample2),
    group = c(rep("s1", length(sample1)), rep("s2", length(sample2)))
  )
  
  # Compute the observed difference, sample2 - sample 1 with diff().
  # Computes the mean of value for each group in dat_obs$group with tapply().
  # Then takes the difference between the means.
  obs_stat <- if (stat == "mean") {
    diff(tapply(dat_obs$value, dat_obs$group, mean, na.rm = TRUE))
  } else {
    diff(tapply(dat_obs$value, dat_obs$group, median, na.rm = TRUE))
  }
  names(obs_stat) <- NULL
  
  # Run permutations by shuffling group labels.
  null_distribution <- numeric(n_sim)
  set.seed(123)
  for (i in seq_len(n_sim)) {
    dat_obs$group <- sample(dat_obs$group)
    null_distribution[i] <- if (stat == "mean") {
      diff(tapply(dat_obs$value, dat_obs$group, mean, na.rm = TRUE))
    } else {
      diff(tapply(dat_obs$value, dat_obs$group, median, na.rm = TRUE))
    }
  }
  
  # Compute expected difference from the null distribution = it's mean 
  # (the null distribution is expected to be normal).
  # Expected difference under H0 (null hypothesis: no difference between groups).
  # expected_dif <- mean(null_distribution, na.rm = TRUE)
  expected_dif <- 0
  
  # Two-tailed p-value & quantile based CI
  # p-value = the proportion of permuted differences that are at least as 
  # extreme as the observed difference.
  p_value <- mean(abs(null_distribution) >= abs(obs_stat), na.rm = TRUE)
  # CI of the null distribution. Is the observed value within this CI?
  # If yes, then the observed difference could reasonably occur due to chance 
  # under the null hypothesis.
  ci <- quantile(null_distribution, probs = c(0.025, 0.975), na.rm = TRUE)
  
  null_distribution_mean <- mean(null_distribution, na.rm = TRUE)
  
  return(list(
    data = dat_obs,
    observed_difference = obs_stat,
    expected_difference = expected_dif,
    null_distribution = null_distribution,
    null_distribution_mean = null_distribution_mean,
    confidence_interval = ci,
    p_value = p_value,
    p_value_fp = p_value_fp,
    z_value_fp = z_value_fp
  ))
}


#' Permutation test histograms
#' 
#' This function creates a histogram of the null distribution from a permutation
#' test. and overlays lines to show the observed difference and the mean of the
#' permuted differences.
#'
#' @param permutation_results A list containing the results of a permutation test,
#'                            including $null_distribution, $observed_difference, 
#'                            and $null_distribution_mean.
#'
#' @return A histogram plot of the permutation test results.
permutation_test_histogram <- function(permutation_results) {
  # Extract components from the input object.
  null_dist <- permutation_results$null_distribution
  obs_diff <- permutation_results$observed_difference
  
  # Calculate x-axis limits to include null distribution and symmetric observed
  # differences. This ensures that both tails of the distribution are visible.
  xlims <- range(c(null_dist, obs_diff, -obs_diff))
  
  # Create histogram of the null distribution.
  hist(null_dist, 
       col = "grey", 
       xlim = xlims, 
       main = "Permutation Test Null Distribution", 
       xlab = "Differences", 
       ylab = "Frequency")
  
  # Add vertical lines for the observed difference (both positive and negative
  # tails). The absolute (abs) value is used to ensure symmetry in the plot.
  abline(v = -abs(obs_diff), col = "red", lwd = 2)  # at negative (left) tail
  abline(v = abs(obs_diff), col = "red", lwd = 2)   # at positive (right) tail
  
  # Add a vertical line for the mean of the permuted differences.
  perm_dif_mean <- permutation_results$null_distribution_mean
  abline(v = perm_dif_mean, col = "blue", lwd = 2)
  
  # Add a legend.
  legend("topright", 
         legend = c("Null Distribution Mean", "Observed Difference"),
         col = c("blue", "red"), 
         lwd = 2)
}


#' Apply permutation tests across categories, variables, and metrics
#'
#' This function iterates over specified categories, variables, and metrics to
#' perform permutation tests and update the comparison data table with results.
#'
#' @param categories A vector of category names to iterate over
#' @param var_vector A vector of variable names to iterate over
#' @param metrics A vector of metrics (e.g., "mean", "median") to use in tests
#' @param dt_max_mis Data table with info for misclassifications
#' @param dt_max_cor Data table with info for correctly classified cases
#' @param dt_compare Data table to store comparison results
#' @param n_sim Number of simulations for permutation test. Default: 1000
#'
#' @return The updated dt_compare data table with permutation test results
apply_permutation_test <- function(categories, var_vector, metrics, dt_max_mis, 
                                   dt_max_cor, dt_compare, n_sim = 1000) {
  # Start timing the function execution
  start_time <- Sys.time()
  
  for (ctg in categories) {
    for (varb in var_vector) {
      for (metric in metrics) {
        
        # Print current test information
        cat("Test for", ctg, varb, metric, "\n")
        
        # Extract samples for current category and variable
        sample_mis <- dt_max_mis[category == ctg, get(varb)]
        sample_cor <- dt_max_cor[category == ctg, get(varb)]
        
        # Perform permutation test with error handling
        perm_res <- tryCatch(
          permutation_test(sample1 = sample_mis, 
                           sample2 = sample_cor, 
                           n_sim = n_sim, 
                           stat = metric),
          error = function(e) {
            # Log error and return NA values
            message(sprintf("Error encountered for Taxa: %s, Variable: %s\nERROR message: %s", 
                            ctg, varb, e$message))
            message("NA assigned for this case")
            return(list(
              observed_difference = NA,
              expected_difference = NA,
              confidence_interval = NA,
              p_value = NA,
              p_value_fp = NA  # Added for Fisher-Pitman (fp) test
            ))
          }
        )
        
        # Calculate and store observed correlation metric
        obs_cor_metric <- if (metric == "mean") {
          round(mean(sample_cor), digits = 4)
        } else {
          round(median(sample_cor), digits = 4)
        }
        col_name <- paste(metric, varb, "cor", sep = "_")
        dt_compare[category == ctg, (col_name) := obs_cor_metric]
        
        # Calculate and store standard deviation if metric is mean
        if (metric == "mean") {
          sd_cor_mean <- round(sd(sample_cor), digits = 4)
          col_name <- paste("sd", metric, varb, "cor", sep = "_")
          dt_compare[category == ctg, (col_name) := sd_cor_mean]  
        }
        
        # Store permutation test results
        col_name <- paste(metric, varb, "dif_cor_mis", sep = "_")
        dt_compare[category == ctg, (col_name) := round(perm_res$observed_difference, digits = 4)]
        
        col_name <- paste(metric, varb, "QuantPermCI", sep = "_")
        dt_compare[category == ctg, (col_name) := sprintf("%.4f:%.4f", 
                                                          perm_res$confidence_interval["2.5%"], 
                                                          perm_res$confidence_interval["97.5%"])]
        
        col_name <- paste(metric, varb, "dif_p_val", sep = "_")
        dt_compare[category == ctg, (col_name) := round(perm_res$p_value, digits = 4)]
        
        col_name <- paste(metric, varb, "dif_p_val_is_singif", sep = "_")
        dt_compare[category == ctg, (col_name) := ifelse(perm_res$p_value < 0.05, "yes", "no")]
        
        # Store Fisher-Pitman test results
        col_name <- paste(varb, "dif_p_val_fp", sep = "_")
        dt_compare[category == ctg, (col_name) := round(perm_res$p_value_fp, digits = 4)]
        
        col_name <- paste(varb, "dif_p_val_fp_is_singif", sep = "_")
        dt_compare[category == ctg, (col_name) := ifelse(perm_res$p_value_fp < 0.05, "yes", "no")]
        
        # Clean up temporary variables
        rm(sample_mis, sample_cor, perm_res, obs_cor_metric, col_name)
      }
    }
  }
  
  # Calculate and print execution time
  end_time <- Sys.time()
  execution_time <- end_time - start_time
  cat("Execution time:", execution_time, "seconds\n")
  
  return(dt_compare)
}
