# ==============================================================================
# 1. Loading Required Libraries
# ==============================================================================
# install.packages(c("faux", "mice", "glmmTMB", "lme4", "microbenchmark", "ggplot2", "dplyr", "tidyr", "broom.mixed"))

library(faux)
library(mice)
library(glmmTMB)
library(lme4)
library(microbenchmark)
library(ggplot2)
library(dplyr)
library(tidyr)
library(broom.mixed)

set.seed(2026)

# ==============================================================================
# 2. Simulating Data (Both Continuous and Binary Outcomes)
# ==============================================================================
cat("Simulating multilevel data (5,000 rows) for fast benchmarking\n")

# 100 groups x 50 obs = 5,000 rows (Fast but sufficient for benchmarking)
dat <- add_random(group = 100) %>%
  add_random(obs = 50, .nested_in = "group") %>%
  # Random effects: u0 = intercept, u1 = slope
  add_ranef("group", u0 = 0.5, u1 = 0.2, .cors = 0.3) %>% 
  # Residual variance for the continuous model
  add_ranef(sigma = 1) %>% 
  mutate(
    x1_cont = rnorm(n(), mean = 0, sd = 1),
    x2_bin  = rbinom(n(), size = 1, prob = 0.5),
    
    # 1. CONTINUOUS OUTCOME (Target for lmer)
    y_cont = 5 + (2 + u1)*x1_cont + 1.5*x2_bin + u0 + sigma,
    
    # 2. BINARY/LOGISTIC OUTCOME (Target for glmer vs glmmTMB)
    # Use logit link to generate probabilities, then draw 0 or 1
    logit_p = -1 + (1.5 + u1)*x1_cont + 0.8*x2_bin + u0,
    prob    = plogis(logit_p), # Convert log-odds to probabilities
    y_bin   = rbinom(n(), size = 1, prob = prob)
  ) %>%
  select(group, y_cont, y_bin, x1_cont, x2_bin)

# ==============================================================================
# 3. Amputing the Dataset
# ==============================================================================
cat("Amputing data\n")

vars_to_ampute <- dat %>% select(y_cont, y_bin, x1_cont)
amp_result <- ampute(data = vars_to_ampute, prop = 0.20, mech = "MAR")

dat_amp <- dat
dat_amp$y_cont  <- amp_result$amp$y_cont
dat_amp$y_bin   <- amp_result$amp$y_bin
dat_amp$x1_cont <- amp_result$amp$x1_cont

# Complete-case subset for the models
obs_data <- dat_amp %>% drop_na()

# ==============================================================================
# 4. Benchmark: Continuous vs Logistic
# ==============================================================================
cat("Benchmarking Continuous Models\n")
form_cont <- y_cont ~ x1_cont + x2_bin + (1 | group)

bm_cont <- microbenchmark(
  lme4_cont    = lmer(form_cont, data = obs_data, REML = FALSE),
  glmmTMB_cont = glmmTMB(form_cont, data = obs_data, REML = FALSE),
  times = 5
)
summary(bm_cont)

cat("Benchmarking Logistic Models (This may take a moment)...\n")
form_bin <- y_bin ~ x1_cont + x2_bin + (1 | group)

bm_bin <- microbenchmark(
  lme4_bin    = suppressMessages(glmer(form_bin, data = obs_data, family = binomial)),
  glmmTMB_bin = glmmTMB(form_bin, data = obs_data, family = binomial),
  times = 5
)
summary(bm_bin)

# ==============================================================================
# 5. Plot Everything Nicely
# ==============================================================================

# Speed Plots
p_speed_cont <- autoplot(bm_cont) + theme_minimal() + 
  labs(title = "Speed: CONTINUOUS Model (lmer vs glmmTMB)", x = "Engine", y = "Time")
p_speed_bin <- autoplot(bm_bin) + theme_minimal() + 
  labs(title = "Speed: LOGISTIC Model (glmer vs glmmTMB)", x = "Engine", y = "Time")

# Display plots
print(p_speed_cont)
print(p_speed_bin)