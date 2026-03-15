## ============================================================================
## mice.impute.ml.glmmTMB.R
## Multilevel imputation using glmmTMB, a richer alternative to lme4
##
## --------------------
## The standard mice package ships a multilevel imputer built on lme4
## (mice.impute.ml.lmer). lme4 is excellent, but it only handles normal and
## binomial outcomes, and its residual / dispersion model is fixed.
##
## glmmTMB lifts those restrictions:
##   • Zero-inflated and hurdle models   (count data with excess zeros)
##   • Negative-binomial, beta, Tweedie  (overdispersed / bounded outcomes)
##   • A dispersion sub-model            (heteroskedasticity across groups)
##   • Compound-symmetry and AR(1) structures (longitudinal data)
##   • Everything above COMBINED with multilevel random effects
##
## In all other respects the imputation logic is identical to the lme4 version:
##   draw fixed effects → draw random effects → assemble predictor → draw imp.
##
## PIPELINE OVERVIEW (15 steps)
## -----------------------------
##  1.  Prepare y          — factor → integer, keep level labels for later
##  2.  Pull mice context  — variable name, full data, which level y lives at
##  3.  Resolve level IDs  — build cluster vectors, group counts
##  4.  Aggregate (opt.)   — if y lives at level-2 collapse to one row per group
##  5.  Mundlak correction — add group-mean columns so FE absorbs between-variance
##  6.  Build formula      — fixed part + RE part + optional interactions
##  7.  Pick family        — gaussian / binomial / poisson or user-supplied
##  8.  Assemble data      — one tidy data.frame for glmmTMB
##  9.  Fit model          — glmmTMB() on observed rows only
## 10.  Draw fixed effects — perturb b via multivariate normal (Rubin's rules)
## 11.  Draw random effects— perturb BLUPs via their posterior variance
## 12.  Assemble eta       — X*b + sum(Z*u) for every row
## 13.  Draw imputations   — gaussian / binary / count / PMM branch
## 14.  Broadcast back     — expand level-2 imputations to individual rows
## 15.  Restore factor     — integer → factor if y was originally a factor
## ============================================================================


mice.impute.ml.glmmTMB <- function(

    ## ------------------------------------------------------------------
    ## Standard mice arguments
    ## ------------------------------------------------------------------
    ## mice passes these three automatically; you never supply them by hand.

    y,      # Numeric/factor vector of length N (all rows).
            # Observed values are real numbers; missing values are NA.

    ry,     # Logical vector of length N.
            # TRUE  → this row has an observed value of y.
            # FALSE → this row is missing and needs an imputed value.

    x,      # Numeric matrix of length N × P.
            # Contains all predictors selected by the predictor matrix.
            # Includes cluster-ID columns (we strip those out below).

    type,   # Named integer vector of length P (one entry per column of x).
            # Encoding used by mice:
            #   1  = ordinary fixed-effect predictor
            #   2  = cluster-level predictor (lives at a higher level)
            #  -2  = cluster identifier (group membership, not a regressor)
            # We use this mainly to identify which columns are IDs.

    ## ------------------------------------------------------------------
    ## Multilevel bookkeeping
    ## ------------------------------------------------------------------

    levels_id,               # Character vector naming the cluster-ID columns
                             # that appear in the full dataset (not in x!).
                             # Order: finest grouping first → coarsest last.
                             # Example for pupils in classes in schools:
                             #   levels_id = c("class_id", "school_id")
                             # The random intercepts will be:
                             #   (1 | class_id) + (1 | school_id)

    variables_levels = NULL, # Optional named list that says which level each
                             # variable in the dataset lives at.
                             # Example:
                             #   list(school_SES = "school_id",
                             #        pupil_IQ   = "class_id")
                             # Used to decide whether to aggregate (Step 4).

    random_slopes = NULL,    # Optional named list.
                             # Each element is named after a grouping level and
                             # contains a character vector of predictor names
                             # that should also vary randomly across groups at
                             # that level.
                             # Example — let SES slope vary across schools:
                             #   random_slopes = list(school_id = c("SES"))
                             # Produces: (1 + SES | school_id)

    ## ------------------------------------------------------------------
    ## Interaction terms
    ## ------------------------------------------------------------------

    interactions = NULL,     # Optional character vector of interaction terms
                             # to add to the FIXED-effects part of the formula.
                             # Each element should be a valid R interaction
                             # string using ":" (pure interaction) or "*"
                             # (main effects + interaction).
                             #
                             # Examples:
                             #   interactions = c("SES:school_type")
                             #   interactions = c("age*gender", "SES:IQ")
                             #
                             # IMPORTANT: every variable named in an interaction
                             # must already be a column of x (or be added by the
                             # Mundlak step). We check this and warn loudly if
                             # a term references a missing column so you don't
                             # silently lose a predictor.
                             #
                             # Cross-level interactions (level-1 predictor x
                             # level-2 group mean) work naturally here because
                             # the Mundlak step (Step 5) already added the group
                             # mean as a column before we build the formula.
                             # Example:
                             #   interactions = c("pupil_SES:school_SES_grpmn")

    ## ------------------------------------------------------------------
    ## Aggregation / centering options
    ## ------------------------------------------------------------------

    aggregate_automatically = TRUE,
                             # When TRUE (recommended) the function automatically
                             # computes group means for continuous predictors and
                             # adds them as extra columns (Mundlak correction).
                             # Set to FALSE if you have already done this outside
                             # the imputer.

    groupcenter_slope = FALSE,
                             # When TRUE the original predictor x_j is replaced
                             # by (x_j − group_mean(x_j)) so the fixed effect
                             # estimate is a pure within-group slope.
                             # The group mean column is still added separately.

    intercept = TRUE,        # Include a fixed intercept in the model?
                             # Set to FALSE only if you have a very specific
                             # reason (e.g. the outcome is always centred).

    ## ------------------------------------------------------------------
    ## Posterior sampling options
    ## ------------------------------------------------------------------

    draw_fixed = TRUE,       # If TRUE (recommended) we draw the fixed-effect
                             # coefficients from their multivariate normal
                             # posterior approximation. This propagates
                             # coefficient uncertainty into the imputations,
                             # which is what Rubin's multiple-imputation rules
                             # require. Set to FALSE only for debugging.

    re_shrinkage = 1e-6,     # Small ridge value added to every variance/
                             # covariance matrix before drawing. Prevents
                             # Cholesky failures when a variance component is
                             # estimated as exactly zero (a common edge case
                             # with small groups or many levels).

    ## ------------------------------------------------------------------
    ## Model / family specification
    ## ------------------------------------------------------------------

    model = "continuous",    # High-level shortcut for common outcome types:
                             #   "continuous" → gaussian family, identity link
                             #   "binary"     → binomial family, logit link
                             #   "pmm"        → gaussian family + PMM draw step
                             #   "count"      → poisson family, log link
                             # If you need anything else (beta, nbinom2, …)
                             # leave model = "continuous" and supply the family
                             # explicitly via family_glmmTMB below.

    family_glmmTMB = NULL,   # Override the automatic family choice.
                             # Supply any glmmTMB family object, e.g.:
                             #   family_glmmTMB = glmmTMB::nbinom2()
                             #   family_glmmTMB = glmmTMB::beta_family()
                             #   family_glmmTMB = glmmTMB::tweedie()
                             # When NULL the family is derived from `model`.

    ## ------------------------------------------------------------------
    ## glmmTMB-specific sub-models (not available in lme4)
    ## ------------------------------------------------------------------

    zi_formula = ~0,         # Zero-inflation sub-model formula.
                             # ~0 means "no zero-inflation" (default).
                             # ~1 means a single global ZI probability.
                             # ~x1 + x2 means ZI probability depends on
                             #   covariates x1 and x2.
                             # Only meaningful for count / hurdle families.

    disp_formula = ~1,       # Dispersion sub-model formula.
                             # ~1 means constant dispersion (default; same
                             #   behaviour as lme4).
                             # ~group allows each group to have its own sigma.
                             # ~covariate models heteroskedasticity as a
                             #   smooth function of a continuous predictor.

    ## ------------------------------------------------------------------
    ## PMM settings
    ## ------------------------------------------------------------------

    donors = 3,              # Number of donor candidates for predictive mean
                             # matching. The imputed value is sampled
                             # uniformly from the `donors` observed cases
                             # whose predicted value is closest to the
                             # predicted value of the missing case.
                             # 3-10 is the usual range; 3 gives more
                             # variability, 10 gives more stability.

    ## ------------------------------------------------------------------
    ## Verbosity / warnings
    ## ------------------------------------------------------------------

    glmmTMB_warnings = TRUE, # If FALSE all glmmTMB convergence warnings are
                             # suppressed. Useful in large simulation studies
                             # where you know the model is correct and the
                             # warnings are noise. In applied work keep TRUE.

    ## ------------------------------------------------------------------
    ## Pass-through to glmmTMB()
    ## ------------------------------------------------------------------

    ...                      # Any additional arguments are forwarded directly
                             # to glmmTMB::glmmTMB(), e.g. control = glmmTMBControl()
)

{   # ======================================================================
    # BEGIN FUNCTION BODY
    # ======================================================================


    # ========================================================================
    # STEP 0 — Load required packages
    # ========================================================================
    # We check at runtime (not at load time) because these are Suggests, not
    # hard dependencies of mice itself. A clear error message is much more
    # helpful than a cryptic "object not found" buried in the call stack.

    if (!requireNamespace("glmmTMB", quietly = TRUE))
        stop(paste(
            "Package 'glmmTMB' is needed for mice.impute.ml.glmmTMB.",
            "Install it with:  install.packages('glmmTMB')"
        ))

    if (!requireNamespace("lme4", quietly = TRUE))
        stop(paste(
            "Package 'lme4' is needed (glmmTMB uses its ranef/VarCorr conventions).",
            "Install it with:  install.packages('lme4')"
        ))

    if (!requireNamespace("MASS", quietly = TRUE))
        stop(paste(
            "Package 'MASS' is needed for multivariate normal draws.",
            "Install it with:  install.packages('MASS')"
        ))


    # ========================================================================
    # STEP 1 — Prepare the outcome variable y
    # ========================================================================
    # WHY: The imputation arithmetic (linear predictors, residual draws) only
    # makes sense on a numeric scale. If the user's variable is an ordered
    # or unordered factor, we convert it to integer codes 1, 2, 3 ... and store
    # the original levels so we can convert back at the very end (Step 15).
    #
    # NOTE: imputing factors via PMM is the correct approach here — we predict
    # a numeric score and then match to an observed factor level. Imputing
    # factors via logistic regression requires a multinomial model which is
    # outside the scope of this function.

    is_factor <- is.factor(y)
    y_levels  <- NULL             # will hold e.g. c("low","medium","high")

    if (is_factor) {
        y_levels <- levels(y)     # preserve the original level labels
        y        <- as.integer(y) # 1 = first level, 2 = second level, ...
        message("  [Step 1] y is a factor — converted to integer codes for imputation.")
        message(sprintf("           Levels: %s", paste(y_levels, collapse = ", ")))
    }

    # Keep a pristine copy of y before any aggregation in Step 4.
    # We will need this copy when broadcasting imputed values back.
    y_original  <- y
    ry_original <- ry   # same for the missingness indicator


    # ========================================================================
    # STEP 2 — Pull context from the mice call stack
    # ========================================================================
    # HOW mice works internally:
    #   mice() loops over variables. For each variable it calls the imputation
    #   function. The calling environment (two frames up from here) holds the
    #   full dataset, the name of the variable being imputed right now, and
    #   other bookkeeping objects. We climb up to grab them.
    #
    # WHY we need them:
    #   vname     : lets us look up which level this variable lives at
    #   full_data : contains the cluster-ID columns that are NOT in x
    #               (mice strips ID columns from x by design)

    caller_env <- parent.frame(n = 2)

    # Name of the variable currently being imputed
    vname <- tryCatch(
        get("vname", envir = caller_env),
        error = function(e) {
            warning("Could not retrieve variable name from mice call stack. Using 'y'.")
            "y"
        }
    )

    # Full data frame (all variables, all rows)
    full_data <- tryCatch(
        get("data", envir = caller_env),
        error = function(e) {
            warning("Could not retrieve full data from mice call stack. Using x.")
            as.data.frame(x)
        }
    )

    # Determine which hierarchical level this variable lives at.
    # Empty string means "lives at the individual / lowest level" — the common case.
    vname_level <- ""
    if (!is.null(variables_levels) && vname %in% names(variables_levels)) {
        vname_level <- variables_levels[[vname]]
        message(sprintf(
            "  [Step 2] '%s' lives at level '%s' → will aggregate before fitting.",
            vname, vname_level
        ))
    }

    message(sprintf(
        "\n[glmmTMB imputer] Imputing '%s'", vname
    ))
    message(sprintf(
        "  Level: %s   |   Model type: %s   |   N=%d, n_missing=%d",
        ifelse(vname_level == "", "level-1 (individual)", vname_level),
        model, length(y), sum(!ry)
    ))


    # ========================================================================
    # STEP 3 — Resolve cluster identifiers
    # ========================================================================
    # We build three parallel data structures, one entry per grouping level:
    #
    #   clus[[lid]]        : vector of length N giving each row's group label
    #   clus_unique[[lid]] : sorted vector of unique group labels
    #   ngr[[lid]]         : scalar — number of groups at this level
    #
    # These are used later when:
    #   drawing random effects (Step 11) — need to loop over groups
    #   broadcasting imputed values (Step 14) — need row-to-group mapping
    #
    # We verify every level ID exists in the data and give a clear error
    # if something is missing rather than crashing deep inside glmmTMB.

    NL <- length(levels_id)   # number of grouping levels (e.g. 2 for class+school)

    if (NL < 1)
        stop("'levels_id' must name at least one grouping variable (e.g. 'school_id').")

    clus        <- vector("list", NL); names(clus)        <- levels_id
    clus_unique <- vector("list", NL); names(clus_unique) <- levels_id
    ngr         <- setNames(integer(NL), levels_id)

    for (lid in levels_id) {
        if (!lid %in% colnames(full_data))
            stop(sprintf(
                "Cluster ID '%s' (from levels_id) was not found in the data.",
                lid
            ))
        clus[[lid]]        <- full_data[[lid]]
        clus_unique[[lid]] <- sort(unique(full_data[[lid]]))
        ngr[[lid]]         <- length(clus_unique[[lid]])
        message(sprintf("  [Step 3] Level '%s': %d groups.", lid, ngr[[lid]]))
    }


    # ========================================================================
    # STEP 4 — (Optional) Aggregate data to a higher level
    # ========================================================================
    # THE PROBLEM:
    #   Suppose we are imputing school_SES — a variable that takes the same
    #   value for every pupil in a school. The individual-level data has
    #   thousands of rows but only ~100 distinct schools. Fitting a model at
    #   the pupil level would be wasteful and misleading (pseudo-replication).
    #
    # THE SOLUTION:
    #   Collapse to one row per school (the unique-rows trick). Fit the model
    #   on that compact frame. Later (Step 14) broadcast the imputed school
    #   value back to every pupil who belongs to that school.
    #
    # We only activate this when vname_level is non-empty, which means the user
    # told us this variable lives at a higher level via variables_levels.

    if (vname_level != "") {
        # One representative row per higher-level unit.
        # duplicated() returns FALSE for the FIRST occurrence of each unique ID,
        # so this picks exactly one row per group without sorting.
        agg_rows <- !duplicated(full_data[[vname_level]])

        message(sprintf(
            "  [Step 4] Aggregating %d individual rows → %d '%s' units.",
            nrow(full_data), sum(agg_rows), vname_level
        ))

        y  <- y[agg_rows]
        ry <- ry[agg_rows]
        x  <- x[agg_rows, , drop = FALSE]

        # For the aggregated frame we no longer need the level that is now the
        # unit of analysis. Keep only coarser levels (if any).
        levels_id <- levels_id[levels_id != vname_level]
        NL        <- length(levels_id)

        # Rebuild cluster vectors for the slimmer frame
        clus        <- lapply(levels_id, function(lid) full_data[[lid]][agg_rows])
        names(clus) <- levels_id
        clus_unique <- lapply(clus, function(v) sort(unique(v)))
        ngr         <- sapply(clus, function(v) length(unique(v)))

        # Edge case: if there is only one level and it IS the variable's level
        # (e.g. a pure two-level model imputing a level-2 variable), NL is now
        # zero. We still need something in levels_id for formula building.
        # In that case silently restore the original single level.
        if (NL == 0) {
            levels_id <- names(ngr)
            NL        <- 1
        }
    }


    # ========================================================================
    # STEP 5 — Mundlak / cluster-mean correction
    # ========================================================================
    # THE STATISTICAL MOTIVATION:
    #   In a mixed-effects model the random intercept absorbs between-group
    #   differences. But if a predictor x_j is correlated with the random
    #   intercept (endogeneity), the fixed-effect estimate of x_j is biased.
    #   Adding the group mean of x_j as an extra predictor (Mundlak 1978,
    #   Enders & Tofighi 2007) breaks this correlation:
    #     The coefficient on x_j becomes the pure within-group effect.
    #     The coefficient on mean(x_j) captures the between-group effect.
    #   This is especially important in educational data (pupils in schools)
    #   where individual SES and school-mean SES have different interpretations.
    #
    # IMPLEMENTATION:
    #   For each continuous predictor we compute its group mean (using the
    #   finest grouping level) and append a new column named <var>_grpmn.
    #   We skip binary variables (their means are just proportions, less useful)
    #   and columns that already have a _grpmn counterpart.

    x_enriched <- x   # we accumulate extra columns here

    if (aggregate_automatically && NL >= 1) {
        primary_lid <- levels_id[1]          # finest grouping level
        grp_ids     <- clus[[primary_lid]]   # group membership for each row

        new_cols <- list()   # will collect the new mean columns

        for (j in seq_len(ncol(x))) {
            col_name <- colnames(x)[j]
            mn_name  <- paste0(col_name, "_grpmn")

            already_present <- mn_name %in% colnames(x)
            is_id_col       <- col_name %in% levels_id
            is_binary       <- length(unique(na.omit(x[, j]))) <= 2

            if (!already_present && !is_id_col && !is_binary) {
                # tapply computes the mean of x[,j] within each group, then we
                # broadcast back to individual rows via the group membership index
                grp_means           <- tapply(x[, j], grp_ids, mean, na.rm = TRUE)
                new_cols[[mn_name]] <- as.numeric(grp_means[as.character(grp_ids)])

                if (groupcenter_slope) {
                    # Replace the original column with its within-group deviation
                    x_enriched[, j] <- x[, j] - new_cols[[mn_name]]
                }
            }
        }

        if (length(new_cols) > 0) {
            new_mat    <- do.call(cbind, new_cols)
            x_enriched <- cbind(x_enriched, new_mat)
            message(sprintf(
                "  [Step 5] Added %d group-mean (Mundlak) columns: %s",
                length(new_cols), paste(names(new_cols), collapse = ", ")
            ))
        } else {
            message("  [Step 5] No new group-mean columns needed.")
        }
    }


    # ========================================================================
    # STEP 6 — Construct the glmmTMB formula
    # ========================================================================
    # glmmTMB (and lme4) use R's standard mixed-model formula syntax.
    # A typical formula looks like:
    #
    #   y_obs ~ age + SES + SES_grpmn + (1 | class_id) + (1 + age | school_id)
    #
    # We build it in three parts:
    #   A) Fixed-effects part   (the terms before the "|" symbols)
    #   B) Random-effects part  (the "(... | ...)" terms)
    #   C) Interaction terms    (appended to the fixed part)
    #
    # -------------------------------------------------------------------------
    # Part A: Fixed-effects predictors
    # -------------------------------------------------------------------------
    # Every column of x_enriched that is NOT a cluster ID is a candidate fixed
    # predictor. Cluster IDs only enter via the random effects structure; using
    # them as dummy predictors would be wrong and cause rank deficiency.

    fixed_predictors <- setdiff(colnames(x_enriched), levels_id)

    # Build the fixed-effects string.
    # If intercept=FALSE we prepend "0" to suppress the global intercept.
    if (length(fixed_predictors) == 0) {
        # No predictors at all — intercept-only model
        fixed_part <- if (intercept) "1" else "0"
    } else {
        fixed_part <- paste(
            c(if (!intercept) "0" else NULL, fixed_predictors),
            collapse = " + "
        )
    }

    # -------------------------------------------------------------------------
    # Part B: Random-effects terms
    # -------------------------------------------------------------------------
    # We create one RE term per grouping level.
    #   Default: random intercept only  -> (1 | level_id)
    #   With random slopes              -> (1 + slope1 + slope2 | level_id)
    #
    # Random slopes are only added if:
    #   (i)  the user listed them in random_slopes for this level, AND
    #   (ii) the predictor actually exists in x_enriched (safety check)

    re_terms <- character(0)

    for (lid in levels_id) {
        slopes_here <- if (!is.null(random_slopes) && !is.null(random_slopes[[lid]])) {
            valid   <- intersect(random_slopes[[lid]], fixed_predictors)
            invalid <- setdiff(random_slopes[[lid]], fixed_predictors)
            if (length(invalid) > 0)
                warning(sprintf(
                    "Random slope(s) [%s] requested for '%s' but not found in predictors. Skipped.",
                    paste(invalid, collapse = ", "), lid
                ))
            valid
        } else {
            character(0)
        }

        if (length(slopes_here) == 0) {
            re_terms <- c(re_terms, sprintf("(1 | %s)", lid))
        } else {
            slope_str <- paste(slopes_here, collapse = " + ")
            re_terms  <- c(re_terms, sprintf("(1 + %s | %s)", slope_str, lid))
        }
    }

    re_part <- paste(re_terms, collapse = " + ")

    # -------------------------------------------------------------------------
    # Part C: Interaction terms
    # -------------------------------------------------------------------------
    # Interactions go into the FIXED part of the formula only — random-effect
    # interactions (e.g. a random cross-level product) are handled separately
    # via random_slopes and are not supported here.
    #
    # ":" means pure interaction (glmmTMB does NOT add main effects automatically).
    # "*" means main effects + interaction (R expands A*B → A + B + A:B).
    #
    # Why it is safe to use ":" without manually adding main effects:
    #   The main effects are already listed in the fixed_part from Part A.
    #   So writing fixed_part + "SES:school_type" is equivalent to writing
    #   "SES + school_type + SES:school_type", which is exactly what we want.
    #   Using "*" would then double-count the main effects, so ":" is preferred
    #   unless you intentionally want to add a new main effect too.
    #
    # Validation logic:
    #   We parse each interaction string to extract the individual variable names
    #   (splitting on ":", "*", and whitespace), then check that each name is a
    #   column in x_enriched. Missing columns get a loud warning and the whole
    #   interaction term is dropped — better to lose one term than to silently
    #   corrupt the formula with an R parsing error.

    interaction_part <- ""

    if (!is.null(interactions) && length(interactions) > 0) {

        valid_interactions <- character(0)

        for (iterm in interactions) {

            # Extract all variable names from the term string.
            # strsplit on the regex [:\* ]+ splits "SES:school_type" into
            # c("SES", "school_type") and "age*gender" into c("age", "gender").
            var_names_in_term <- unique(
                trimws(unlist(strsplit(iterm, "[:\\*\\s]+")))
            )
            # Drop empty strings that can appear after splitting
            var_names_in_term <- var_names_in_term[nchar(var_names_in_term) > 0]

            # Check that every variable in the interaction exists as a column.
            # We allow both the enriched predictors and the original cluster IDs
            # (though using a cluster ID in an interaction would be unusual).
            all_available <- c(colnames(x_enriched), levels_id)
            missing_vars  <- setdiff(var_names_in_term, all_available)

            if (length(missing_vars) > 0) {
                warning(sprintf(
                    paste(
                        "[glmmTMB imputer] Interaction term '%s' references",
                        "variable(s) not found in predictors: [%s].",
                        "This term will be SKIPPED.",
                        "Available predictors: [%s]"
                    ),
                    iterm,
                    paste(missing_vars,  collapse = ", "),
                    paste(all_available, collapse = ", ")
                ))
            } else {
                valid_interactions <- c(valid_interactions, iterm)
                message(sprintf("  [Step 6] Interaction term accepted: '%s'", iterm))
            }
        }

        if (length(valid_interactions) > 0) {
            # Concatenate all valid interaction strings into one formula chunk
            interaction_part <- paste(valid_interactions, collapse = " + ")
        }
    }

    # -------------------------------------------------------------------------
    # Assemble the full formula string
    # -------------------------------------------------------------------------
    # Final structure:
    #   y_obs ~ [fixed predictors] + [interaction terms] + [random effects]
    #
    # We name the outcome "y_obs" to avoid clashes with any column already in
    # the data (e.g. if someone named their variable literally "y").

    formula_rhs_parts <- c(
        fixed_part,
        if (nchar(interaction_part) > 0) interaction_part else NULL,
        re_part
    )

    full_formula_str <- paste("y_obs ~", paste(formula_rhs_parts, collapse = " + "))
    full_formula     <- as.formula(full_formula_str)

    message(sprintf("  [Step 6] Final formula: %s", deparse(full_formula)))


    # ========================================================================
    # STEP 7 — Pick the glmmTMB family
    # ========================================================================
    # If the user supplied a family object directly (family_glmmTMB != NULL)
    # we use it as-is. Otherwise we map the high-level `model` string to the
    # appropriate glmmTMB family:
    #
    #   "continuous" → gaussian(link="identity")
    #   "binary"     → binomial(link="logit")
    #   "pmm"        → gaussian(link="identity")  [PMM draw happens in Step 13]
    #   "count"      → poisson(link="log")
    #
    # For more exotic families (beta, nbinom2, tweedie, truncated counts) the
    # user should pass the family explicitly, e.g.:
    #   family_glmmTMB = glmmTMB::nbinom2(link="log")

    if (is.null(family_glmmTMB)) {
        family_glmmTMB <- switch(
            model,
            "continuous" = glmmTMB::gaussian(),
            "binary"     = glmmTMB::binomial("logit"),
            "pmm"        = glmmTMB::gaussian(),
            "count"      = glmmTMB::poisson("log"),
            stop(sprintf(
                "Unknown model='%s'. Choose 'continuous','binary','pmm','count' or supply family_glmmTMB.",
                model
            ))
        )
        message(sprintf("  [Step 7] Family: %s (auto-selected from model='%s').",
                        class(family_glmmTMB)[1], model))
    } else {
        message(sprintf("  [Step 7] Family: user-supplied (%s).",
                        class(family_glmmTMB)[1]))
    }


    # ========================================================================
    # STEP 8 — Assemble the model data frame
    # ========================================================================
    # glmmTMB needs a named data frame (not a matrix). We build one that has:
    #   All columns from x_enriched  (predictors + group means)
    #   The cluster ID columns       (looked up from full_data or clus)
    #   y_obs                        (the outcome, NA where missing)
    #
    # WHY keep missing rows in the data frame?
    #   We fit only on observed rows (fit_df), but we PREDICT on all rows
    #   including missing ones (in Step 12). Keeping everything in one frame
    #   makes indexing with ry straightforward.

    model_df <- as.data.frame(x_enriched)

    # Add cluster ID columns so glmmTMB can find the grouping variables
    for (lid in levels_id) {
        model_df[[lid]] <- clus[[lid]]
    }

    # Add the outcome: observed values are real, missing values are NA
    model_df[["y_obs"]]      <- y
    model_df[["y_obs"]][!ry] <- NA

    # Rows used for fitting
    fit_df <- model_df[ry, ]

    message(sprintf(
        "  [Step 8] Model data: %d observed rows for fitting, %d rows for prediction.",
        nrow(fit_df), sum(!ry)
    ))


    # ========================================================================
    # STEP 9 — Fit the glmmTMB model on observed data
    # ========================================================================
    # We call glmmTMB() with:
    #   formula      = the lme4-style mixed-model formula from Step 6
    #   data         = only the observed rows (fit_df)
    #   family       = the family from Step 7
    #   ziformula    = zero-inflation sub-model (default ~0 = no ZI)
    #   dispformula  = dispersion sub-model     (default ~1 = constant)
    #   ...          = any extra arguments the user passed through
    #
    # The warning-catching wrapper keeps the imputation output readable.
    # Convergence issues produce a warning (not a silent failure) so the user
    # knows to investigate.

    fit_call <- function() {
        glmmTMB::glmmTMB(
            formula     = full_formula,
            data        = fit_df,
            family      = family_glmmTMB,
            ziformula   = zi_formula,
            dispformula = disp_formula,
            ...
        )
    }

    message("  [Step 9] Fitting glmmTMB model...")

    fit <- if (glmmTMB_warnings) {
        fit_call()
    } else {
        suppressWarnings(fit_call())
    }

    # Check the convergence flag embedded in the fitted object
    conv_msg <- tryCatch(fit$fit$message, error = function(e) NULL)
    if (!is.null(conv_msg) && grepl("false convergence|convergence failure",
                                    conv_msg, ignore.case = TRUE)) {
        warning(sprintf(
            "[glmmTMB imputer] Possible convergence issue for '%s': %s",
            vname, conv_msg
        ))
    } else {
        message("  [Step 9] Model fitted successfully.")
    }


    # ========================================================================
    # STEP 10 — Draw from the posterior of fixed-effect coefficients
    # ========================================================================
    # THEORY (why we draw rather than use point estimates):
    #   Multiple imputation requires that each imputed dataset reflects not
    #   just uncertainty about missing data but also uncertainty about the model
    #   parameters (Rubin 1987). If we always plug in the same maximum-likelihood
    #   estimates the between-imputation variance will be too small and
    #   confidence intervals from the final analysis will be anticonservative.
    #
    # METHOD (proper Bayesian draw):
    #   We treat the MLE as the mode of an approximate normal posterior and draw
    #   one realisation:
    #       b* ~ Normal( b_MLE , Var(b_MLE) )
    #   The covariance matrix Var(b_MLE) comes from the inverse observed Hessian
    #   (which glmmTMB returns via vcov()).
    #
    # The re_shrinkage ridge makes the draw stable when any eigenvalue of the
    # covariance matrix is close to zero (can happen with sparse data).

    b_point <- glmmTMB::fixef(fit)$cond   # named vector of fixed-effect estimates

    if (draw_fixed) {
        vcov_fixed <- as.matrix(vcov(fit)$cond)

        # Add a small ridge to the diagonal to guarantee positive-definiteness.
        # This is the same strategy used in mice.impute.ml.lmer.
        vcov_fixed <- vcov_fixed + diag(re_shrinkage, nrow(vcov_fixed))

        b_draw <- as.numeric(MASS::mvrnorm(n = 1, mu = b_point, Sigma = vcov_fixed))
        names(b_draw) <- names(b_point)

        message(sprintf(
            "  [Step 10] Drew fixed effects from N(%d-dim posterior).",
            length(b_draw)
        ))
    } else {
        b_draw <- b_point
        message("  [Step 10] Using point estimates for fixed effects (draw_fixed=FALSE).")
    }


    # ========================================================================
    # STEP 11 — Draw from the posterior of random effects
    # ========================================================================
    # THEORY:
    #   The random effects u_j are not estimated as free parameters; instead
    #   lme4/glmmTMB compute the Best Linear Unbiased Predictor (BLUP), which
    #   is the conditional mean E[u_j | data]. This shrinks every group's
    #   effect towards zero proportional to the ratio of within-group noise to
    #   between-group variance (Stein shrinkage).
    #
    #   For imputation we need to capture uncertainty AROUND the BLUP, not just
    #   the BLUP itself. The conditional variance Var(u_j | data) is stored in
    #   the "postVar" attribute of the ranef() output — this is the posterior
    #   variance of each group's random effect. We draw from Normal(BLUP, postVar).
    #
    # STRUCTURE:
    #   re_contribution is a N x NL matrix.
    #   Column ll holds the total random-effects contribution from level ll.
    #   At the end we sum across columns to get the per-row RE contribution.
    #
    # RANDOM SLOPES:
    #   If a variable has a random slope at some level, its contribution is:
    #       u_{intercept,g} + u_{slope,g} * x_{ij}
    #   where g is the group that row i belongs to and x_{ij} is the predictor
    #   value for that row.

    re_contribution <- matrix(0.0, nrow = nrow(model_df), ncol = NL)
    colnames(re_contribution) <- levels_id

    # glmmTMB stores random effects in the same format as lme4:
    #   ranef(fit)$cond is a named list, one element per grouping variable.
    #   Each element is a data frame with one row per group and one column
    #   per random-effect term (e.g. "(Intercept)", "SES").
    re_means   <- lme4::ranef(fit)$cond
    re_condvar <- lme4::ranef(fit, condVar = TRUE)$cond

    for (ll in seq_len(NL)) {
        lid <- levels_id[ll]

        blup_df <- re_means[[lid]]
        if (is.null(blup_df)) {
            message(sprintf("  [Step 11] No random effects found for '%s'. Skipped.", lid))
            next
        }

        grp_names <- rownames(blup_df)   # e.g. c("school1", "school2", ...)
        n_re      <- ncol(blup_df)       # 1 for intercept-only, more with slopes
        n_grp     <- nrow(blup_df)

        # postVar is a 3D array: [n_re x n_re x n_grp]
        # Slice [,, g] is the posterior covariance matrix for group g
        pv_array <- attr(re_condvar[[lid]], "postVar")

        # Draw a new random effect vector for every group
        re_draws <- matrix(NA_real_, nrow = n_grp, ncol = n_re,
                           dimnames = list(grp_names, colnames(blup_df)))

        for (g in seq_len(n_grp)) {
            mu_g <- as.numeric(blup_df[g, ])

            sigma_g <- if (!is.null(pv_array)) {
                # Extract the g-th posterior covariance slice
                matrix(pv_array[, , g], n_re, n_re) +
                    diag(re_shrinkage, n_re)
            } else {
                # Fallback: only the shrinkage ridge (acts like strong shrinkage)
                diag(re_shrinkage, n_re)
            }

            re_draws[g, ] <- if (n_re == 1) {
                rnorm(1, mean = mu_g, sd = sqrt(sigma_g[1, 1]))
            } else {
                as.numeric(MASS::mvrnorm(1, mu = mu_g, Sigma = sigma_g))
            }
        }

        # Map group-level draws back to every individual row.
        # clus[[lid]] is the vector of length N giving each row its group label.
        grp_ids_this_level <- clus[[lid]]

        # Random intercept: each row gets the draw for its group
        if ("(Intercept)" %in% colnames(re_draws)) {
            re_contribution[, ll] <- re_contribution[, ll] +
                re_draws[as.character(grp_ids_this_level), "(Intercept)"]
        }

        # Random slopes: each row gets its group's slope draw * its predictor value
        slope_cols <- setdiff(colnames(re_draws), "(Intercept)")
        for (sc in slope_cols) {
            if (sc %in% colnames(model_df)) {
                re_contribution[, ll] <- re_contribution[, ll] +
                    re_draws[as.character(grp_ids_this_level), sc] * model_df[[sc]]
            } else {
                warning(sprintf(
                    "Random slope '%s' has no matching column in the model data. Skipped.", sc
                ))
            }
        }

        message(sprintf(
            "  [Step 11] Drew random effects for '%s' (%d groups, %d RE term(s)).",
            lid, n_grp, n_re
        ))
    }

    # Sum the contributions from all grouping levels into a single vector
    total_re <- rowSums(re_contribution)


    # ========================================================================
    # STEP 12 — Assemble the full linear predictor (eta) for every row
    # ========================================================================
    # The linear predictor is:
    #
    #   eta_i = X_i * b*  +  sum over levels l of u_{g_l(i)}
    #
    # where:
    #   X_i   = the i-th row of the fixed-effects design matrix
    #           (includes the interaction columns if any were specified)
    #   b*    = the drawn (or point-estimate) fixed-effect coefficients
    #   u_g   = drawn random effect for the group g that row i belongs to
    #
    # IMPORTANT: glmmTMB may have dropped some predictors internally if they
    # caused rank deficiency (perfect collinearity). We therefore align the
    # design matrix columns with the names in b_draw rather than assuming they
    # match column-for-column.
    #
    # For binary and count outcomes eta is on the link scale (log-odds / log).
    # We back-transform to probabilities / rates in Step 13 as needed.
    #
    # NOTE ON INTERACTIONS:
    #   Interaction columns (e.g. SES:school_type) are NOT pre-computed columns
    #   in model_df — they live inside the formula and glmmTMB creates them
    #   when building its own design matrix. When we predict manually here we
    #   need to materialise those product columns ourselves.
    #   We do this via model.matrix() using the fixed part of the formula only,
    #   which is the standard and correct approach.

    # Extract only the fixed-effects part of the formula (drop the RE terms)
    # by re-parsing the formula string without the "(... | ...)" chunks.
    fe_formula_str <- gsub("\\+?\\s*\\([^)]*\\|[^)]*\\)", "",
                           deparse(full_formula))
    fe_formula_str <- trimws(gsub("\\+\\s*$", "", fe_formula_str))
    fe_formula     <- as.formula(fe_formula_str)

    # model.matrix() correctly expands ":" and "*" interactions into columns
    x_design <- tryCatch(
        model.matrix(fe_formula, data = model_df),
        error = function(e) {
            # Fallback: build a simple design matrix without interaction expansion
            warning(paste(
                "[glmmTMB imputer] model.matrix() failed for prediction; falling back",
                "to plain fixed-effects matrix. Interaction terms will be missing.",
                "Error:", conditionMessage(e)
            ))
            fe_cols  <- intersect(colnames(model_df), setdiff(names(b_draw), "(Intercept)"))
            xm       <- as.matrix(model_df[, fe_cols, drop = FALSE])
            if ("(Intercept)" %in% names(b_draw)) cbind(`(Intercept)` = 1, xm) else xm
        }
    )

    # Align the design matrix with b_draw (drop columns not in b_draw, keep order)
    shared_cols <- intersect(colnames(x_design), names(b_draw))
    x_design    <- x_design[, shared_cols, drop = FALSE]
    b_aligned   <- b_draw[shared_cols]

    # Compute the full linear predictor
    eta_all <- as.numeric(x_design %*% b_aligned) + total_re

    message(sprintf(
        "  [Step 12] Linear predictor assembled for all %d rows.", nrow(model_df)
    ))


    # ========================================================================
    # STEP 13 — Draw imputed values for missing rows
    # ========================================================================
    # We extract eta for rows where ry == FALSE (the missing observations)
    # and draw imputed values from the appropriate conditional distribution.
    #
    # Four branches:
    #
    #  A) continuous (Gaussian):
    #     y* ~ Normal( eta_missing, sigma_residual )
    #     sigma_residual is extracted from the glmmTMB fit via sigma().
    #
    #  B) binary (Bernoulli):
    #     p  = logistic(eta_missing)
    #     y* ~ Bernoulli(p)
    #     Returned as 0/1 integer.
    #
    #  C) count (Poisson):
    #     lambda = exp(eta_missing)
    #     y*     ~ Poisson(lambda)
    #
    #  D) PMM (Predictive Mean Matching):
    #     For each missing unit find the `donors` observed units with predicted
    #     values closest to the missing unit's predicted value, then sample one
    #     of those observed values uniformly at random.
    #     This keeps imputations on the observed support — no extrapolation,
    #     no distributional assumptions about the residuals.

    eta_missing  <- eta_all[!ry]
    eta_observed <- eta_all[ry]
    y_observed   <- y[ry]

    imp <- if (model == "continuous") {

        # Branch A: Gaussian draw
        # sigma() returns the residual standard deviation for Gaussian families.
        sigma_resid <- sigma(fit)
        message(sprintf("  [Step 13] Gaussian draw (sigma = %.4f).", sigma_resid))

        rnorm(length(eta_missing), mean = eta_missing, sd = sigma_resid)

    } else if (model == "binary") {

        # Branch B: Binary / Bernoulli draw
        # plogis() is R's built-in inverse-logit: plogis(x) = 1 / (1 + exp(-x))
        probs <- plogis(eta_missing)
        message(sprintf(
            "  [Step 13] Binary draw (predicted P range: [%.3f, %.3f]).",
            min(probs), max(probs)
        ))

        rbinom(length(probs), size = 1, prob = probs)

    } else if (model == "count") {

        # Branch C: Poisson count draw
        rates <- exp(eta_missing)
        message(sprintf(
            "  [Step 13] Poisson draw (predicted lambda range: [%.3f, %.3f]).",
            min(rates), max(rates)
        ))

        rpois(length(rates), lambda = rates)

    } else if (model == "pmm") {

        # Branch D: Predictive Mean Matching
        # For each missing unit i:
        #   1. Compute |eta_obs_j - eta_missing_i| for every observed j
        #   2. Keep the `donors` closest observed units
        #   3. Sample one of their actual y values uniformly
        #
        # PMM is robust to non-normality because we never draw from a
        # parametric distribution — we only ever impute observed values.

        n_missing     <- length(eta_missing)
        n_obs         <- length(eta_observed)
        actual_donors <- min(donors, n_obs)

        if (actual_donors < donors)
            warning(sprintf(
                "[glmmTMB PMM] Only %d observed units available; using %d donors.",
                n_obs, actual_donors
            ))

        message(sprintf(
            "  [Step 13] PMM draw (%d missing, %d donors each).",
            n_missing, actual_donors
        ))

        imp_pmm <- numeric(n_missing)

        for (i in seq_len(n_missing)) {
            dists      <- abs(eta_observed - eta_missing[i])
            donor_idx  <- order(dists)[seq_len(actual_donors)]
            imp_pmm[i] <- sample(y_observed[donor_idx], size = 1)
        }

        imp_pmm
    }


    # ========================================================================
    # STEP 14 — Broadcast imputed values back to individual rows
    # ========================================================================
    # This step is only needed when we imputed a higher-level variable (Step 4
    # aggregated the data). In that case:
    #   `imp` has one entry per MISSING HIGHER-LEVEL UNIT
    #   But we need to return one entry per MISSING INDIVIDUAL ROW
    #
    # We build a lookup table (group ID → imputed value) and use it to copy
    # the school-level imputed value to every pupil in that school whose
    # school variable was missing.

    if (vname_level != "") {
        message(sprintf(
            "  [Step 14] Broadcasting %d group-level imputations to individual rows.",
            length(imp)
        ))

        # Identify which rows in the ORIGINAL data were representative of each
        # higher-level unit AND had a missing value
        agg_rows_all     <- !duplicated(full_data[[vname_level]])
        agg_rows_missing <- agg_rows_all & !ry_original

        # Get the IDs of the higher-level units that were missing
        unit_ids_imp <- full_data[[vname_level]][agg_rows_missing]

        # Build the lookup: unit ID → imputed value
        lookup <- setNames(imp, as.character(unit_ids_imp))

        # Map back to every individual row that was missing this variable
        individual_ids_missing <- full_data[[vname_level]][!ry_original]
        imp <- lookup[as.character(individual_ids_missing)]

        if (any(is.na(imp))) {
            warning(paste(
                sprintf("[glmmTMB imputer] %d individual rows could not be matched", sum(is.na(imp))),
                "back to a group-level imputation.",
                "Check that levels_id and variables_levels are consistent."
            ))
        }
    }


    # ========================================================================
    # STEP 15 — Restore factor encoding (if y was originally a factor)
    # ========================================================================
    # In Step 1 we converted factor → integer. Now we go back.
    # We round the imputed integer codes (PMM always returns exact integers,
    # but Gaussian draws might not) and clip them to [1, n_levels].

    if (is_factor && !is.null(y_levels)) {
        imp_int <- round(as.numeric(imp))
        imp_int <- pmax(1L, pmin(length(y_levels), imp_int))
        imp     <- factor(y_levels[imp_int], levels = y_levels)
        message(sprintf(
            "  [Step 15] Converted %d imputed integer codes back to factor levels.",
            length(imp)
        ))
    }


    # ========================================================================
    # Done — return a vector of length sum(!ry_original)
    # ========================================================================
    # mice expects exactly this: a vector of imputed values, one per missing row,
    # in the same order as the rows in the original data where ry == FALSE.

    message(sprintf(
        "[glmmTMB imputer] Done imputing '%s'. Returning %d values.\n",
        vname, length(imp)
    ))

    return(imp)

}   # END mice.impute.ml.glmmTMB


## ============================================================================
## COMPANION FUNCTION: make_ml_glmmTMB_predMatrix()
## ============================================================================
##
## PURPOSE
## -------
## mice() needs a "predictor matrix" — a square 0/1 matrix where entry [i,j]
## means "variable j is used to predict variable i".
##
## For multilevel data the default quickpred() matrix is wrong because it:
##   1. Includes cluster-ID columns as ordinary fixed-effect predictors
##   2. Lets level-2 variables be "predicted" by level-1 variables
##      (statistically nonsensical — pupil IQ cannot predict school SES)
##   3. Does not know about the hierarchical structure at all
##
## This function builds a predictor matrix that respects the level structure.
##
## ARGUMENTS
## ---------
##   data        : the data frame you will pass to mice()
##   levels_id   : character vector of grouping column names (fine → coarse)
##   level2_vars : names of variables that live at level 2 (e.g. school-level)
##   level3_vars : names of variables that live at level 3 (if applicable)
##
## USAGE EXAMPLE
## -------------
##   pm  <- make_ml_glmmTMB_predMatrix(
##               data,
##               levels_id   = c("class_id", "school_id"),
##               level2_vars = c("school_SES", "school_size")
##           )
##   imp <- mice(data,
##               method          = "ml.glmmTMB",
##               predictorMatrix = pm,
##               levels_id       = c("class_id", "school_id"))
## (if implemented into mice)
## ============================================================================

make_ml_glmmTMB_predMatrix <- function(
    data,
    levels_id,
    level2_vars = NULL,
    level3_vars = NULL
) {
    vars <- colnames(data)
    n    <- length(vars)

    # Start with "everyone predicts everyone" and then remove what is wrong
    pm        <- matrix(1L, nrow = n, ncol = n, dimnames = list(vars, vars))
    diag(pm)  <- 0L   # a variable never predicts itself

    # Rule 1: Cluster IDs are never used as fixed-effect predictors.
    # They enter the model via the random-effects structure (the formula's
    # "(1 | cluster_id)" terms), NOT as a column in X. Using them as
    # predictors would mean fitting a separate dummy for every school,
    # which defeats the purpose of mixed models entirely.
    pm[, levels_id] <- 0L
    pm[levels_id, ] <- 0L   # and we do not try to impute them either

    # Rule 2: Level-2 variables are not predicted by level-1 variables.
    # A school's mean SES is a school-level property; it cannot be caused
    # by individual pupils' scores. Allowing this would add noise and
    # is conceptually wrong in a top-down hierarchy.
    if (!is.null(level2_vars)) {
        level1_vars <- setdiff(vars, c(levels_id, level2_vars,
                                       if (!is.null(level3_vars)) level3_vars))
        pm[level2_vars, level1_vars] <- 0L
    }

    # Rule 3: Same logic one level higher — level-3 variables are not
    # predicted by level-1 or level-2 variables.
    if (!is.null(level3_vars)) {
        non_level3 <- setdiff(vars, c(levels_id, level3_vars))
        pm[level3_vars, non_level3] <- 0L
    }

    return(pm)
}
