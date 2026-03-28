mice.impute.2l.glmmTMB <- function(y,
                                   ry,
                                   x,
                                   type,
                                   wy        = NULL,
                                   intercept = TRUE,
                                   formula   = NULL,
                                   ...) {

  install.on.demand("glmmTMB", ...)
  # first thing i did understand that wy is which rows need to be imputed.
  # so we just say: impute everything that is not observed
  if (is.null(wy)) wy <- !ry

  # I copied this directly from 2l.lmer. We add a column of 1s to x as the
  # intercept, and give it type = 2 so it becomes a random intercept
  # Without this the model would have no intercept at all
  if (intercept) {
    x    <- cbind(1, as.matrix(x))
    type <- c(2, type)
    names(type)[1] <- colnames(x)[1] <- "(Intercept)"
  }

  # again copied from 2l.lmer. we use the type vector to figure out which
  # column is the cluster ID, which columns get a random slope, and which
  # are just fixed effects. from what I did understand 
  # type is a named vector where each name is a column name from x
  clust <- names(type[type == -2])
  rande <- names(type[type == 2])
  fixe  <- names(type[type > 0])

  # we need the unique group labels to loop over later
  lev <- unique(x[, clust])

  # now as far as I understant we split x into two design matrices, same as 2l.lmer code:
  # X has the fixed effect columns, Z has the random effect columns
  # we also pull out just the observed rows (ry == TRUE) for fitting the model.
  X    <- x[,  fixe,  drop = FALSE]
  Z    <- x[,  rande, drop = FALSE]
  xobs <- x[ry, ,     drop = FALSE]
  yobs <- y[ry]
  Xobs <- X[ry, , drop = FALSE]
  Zobs <- Z[ry, , drop = FALSE]

  # here is the one part I changed compared to 2l.lmer.I tried to implement the idea
  # that you told me in the last meeting (not sure if it's the correct way but it makes sense to me)
  # I added that if the user gives their own formula we use that directly
  # that way they can write things like age:SES for an interaction term,
  # or use whatever random effects structure they need.
  # if they don't give a formula we build it automatically, same way as 2l.lmer
  if (!is.null(formula)) {
    randmodel <- as.formula(formula)
  } else {
    # I'm just copying the formula builder from 2l.lmer here
    # it builds something like: yobs ~ age + SES + (1 + age | school_id)
    fr <- ifelse(
      length(rande) > 1,
      paste("+ ( 1 +", paste(rande[-1L], collapse = "+")),
      "+ ( 1 "
    )
    randmodel <- paste(
      "yobs ~",
      paste(fixe[-1L], collapse = "+"),
      fr, "|", clust, ")"
    )
    randmodel <- as.formula(randmodel)
  }

  # we now fit glmmTMB on the observed rows only.
  # I wrap it in try() the same way 2l.lmer does, so if the model fails
  # (e.g. not enough data per group) we don't crash the whole imputation,
  # we just return the current values and show a warning.
  suppressWarnings(
    fit <- try(
      glmmTMB::glmmTMB(
        formula = randmodel,
        data    = data.frame(yobs, xobs),
        ...
      ),
      silent = TRUE # not sure what this does
    )
  )

  if (inherits(fit, "try-error")) {
    warning("glmmTMB did not run. Simplify imputation model.")
    return(y[wy])
  }

# -----------------------------------------------------------------------
# NOTE FOR NEXT MEETING
#
# everything from here until the end of the function is something I did
# not fully understand in the 2l.lmer code and how to translate
# it correctly to glmmTMB. I will need help going through this together
# I also read in Stefan's code something about usage of the Gibbs sampler to draw
# from posteriors, we did study this in Bayesian Statistics course, but I still
# need to understand how it can be implemented in this context.  

  # draw sigma*
  sigmahat <- sigma(fit)
  df <- nrow(fit@frame) - length(fit@beta)
  sigma2star <- df * sigmahat^2 / rchisq(1, df)
  
  # draw beta*
  beta <- lme4::fixef(fit)
  RX <- lme4::getME(fit, "RX")
  
  # cov-matrix, i.e., vcov(fit)
  covmat <- sigma2star * chol2inv(RX)
  rv <- t(chol(covmat))
  beta.star <- beta + rv %*% rnorm(ncol(rv))
  
  # draw psi*
  # applying the standard Wishart prior
  rancoef <- as.matrix(lme4::ranef(fit)[[1]])
  lambda <- t(rancoef) %*% rancoef
  df.psi <- nrow(rancoef)
  temp.psi.star <- stats::rWishart(1, df.psi, diag(nrow(lambda)))[,, 1]
  temp <- MASS::ginv(lambda)
  ev <- eigen(temp)
  if (sum(ev$values > 0) == length(ev$values)) {
    deco <- ev$vectors %*% diag(sqrt(ev$values), nrow = length(ev$values))
    psi.star <- MASS::ginv(deco %*% temp.psi.star %*% t(deco))
  } else {
    try(temp.svd <- svd(lambda))
    if (!inherits(temp.svd, "try-error")) {
      deco <- temp.svd$u %*% diag(sqrt(temp.svd$d), nrow = length(temp.svd$d))
      psi.star <- MASS::ginv(deco %*% temp.psi.star %*% t(deco))
    } else {
      psi.star <- temp
      warning("psi fixed to estimate")
    }
  }
  
  # Calculate myi, vyi and drawing bi per cluster
  for (jj in lev) {
    if (jj %in% unique(xobs[, clust])) {
      Xi <- Xobs[xobs[, clust] == jj, ]
      Zi <- as.matrix(Zobs[xobs[, clust] == jj, ])
      yi <- yobs[xobs[, clust] == jj]
      sigma2 <- diag(sigma2star, nrow = nrow(Zi))
      Mi <- psi.star %*%
        t(Zi) %*%
        MASS::ginv(Zi %*% psi.star %*% t(Zi) + sigma2)
      myi <- Mi %*% (yi - Xi %*% beta.star)
      vyi <- psi.star - Mi %*% Zi %*% psi.star
    } else {
      myi <- matrix(0, nrow = nrow(psi.star), ncol = 1)
      vyi <- psi.star
    }
    
    vyi <- vyi - upper.tri(vyi) * vyi + t(lower.tri(vyi) * vyi)
    # generating bi.star using eigenvalues
    deco1 <- eigen(vyi)
    if (sum(deco1$values > 0) == length(deco1$values)) {
      A <- deco1$vectors %*%
        sqrt(diag(deco1$values, nrow = length(deco1$values)))
      bi.star <- myi + A %*% rnorm(length(myi))
    } else {
      # generating bi.star using svd
      try(deco1 <- svd(vyi))
      if (!inherits(deco1, "try-error")) {
        A <- deco1$u %*% sqrt(diag(deco1$d, nrow = length(deco1$d)))
        bi.star <- myi + A %*% rnorm(length(myi))
      } else {
        bi.star <- myi
        warning("b_", jj, " fixed to estimate")
      }
    }
    
    # imputation
    y[wy & x[, clust] == jj] <- as.vector(
      as.matrix(X[wy & x[, clust] == jj, , drop = FALSE]) %*%
        beta.star +
        as.matrix(Z[wy & x[, clust] == jj, , drop = FALSE]) %*%
        as.matrix(bi.star) +
        rnorm(sum(wy & x[, clust] == jj)) * sqrt(sigma2star)
    )
  }
  y[wy]
}
