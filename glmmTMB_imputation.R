# ==============================================================================
# SIMPLE GLMMTMB FUNCTION FOR MICE
# ==============================================================================

# This is the imputation function
# It takes incomplete data and fills in missing values using glmmTMB

mice.impute.glmmTMB <- function(y, ry, x, family = "gaussian", ...) {
  
  # y = variable with missing values
  # ry = TRUE/FALSE for observed/missing
  # x = predictor variables
  # family = distribution type (gaussian, binomial, poisson)
  
  # STEP 1: Check if we have predictors
  if (ncol(x) == 0) {
    stop("No predictors to use for imputation")
  }
  
  # STEP 2: Create dataframe for modeling
  # Combine the outcome variable with all predictors
  data_model <- data.frame(y = y, x)
  
  # STEP 3: Fit glmmTMB model using only observed values
  # This learns the relationship from the data we have
  # Build formula from predictor names (instead of using "." which causes issues)
  predictor_names <- colnames(x)
  formula_str <- paste("y ~", paste(predictor_names, collapse = " + "))
  formula_obj <- as.formula(formula_str)
  
  fit <- glmmTMB::glmmTMB(
    formula_obj,
    data = data_model,
    family = family,
    subset = ry  # Only use rows where y is observed
  )
  
  # STEP 4: Make predictions for missing values
  # Use the fitted model to predict what missing values should be
  pred <- predict(fit, newdata = data_model, type = "link", re.form = NA)
  
  # STEP 5: Draw imputed values from the predictive distribution
  # Different families need different approaches
  imputed <- y
  
  if (family == "gaussian") {
    # For continuous data: prediction + random normal noise
    residual_sd <- sqrt(mean(residuals(fit)^2, na.rm = TRUE))
    imputed[!ry] <- pred[!ry] + rnorm(sum(!ry), 0, residual_sd)
    
  } else if (family == "binomial") {
    # For binary data: predict probability then draw from Bernoulli
    prob <- plogis(pred)
    imputed[!ry] <- rbinom(sum(!ry), size = 1, prob = prob[!ry])
    
  } else if (family == "poisson") {
    # For count data: predict rate then draw from Poisson
    lambda <- exp(pred)
    imputed[!ry] <- rpois(sum(!ry), lambda = lambda[!ry])
  }
  
  return(imputed)
}

# End of function
# ==============================================================================
