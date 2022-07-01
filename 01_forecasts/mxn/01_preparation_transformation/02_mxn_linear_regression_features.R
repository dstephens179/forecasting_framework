# LIBRARIES ----
library(tidymodels)
library(modeltime)
library(timetk)
library(tidyverse)
library(lubridate)
library(bigrquery)
library(skimr)
library(DataExplorer)



# DATA ----
# Gold Price Data
projectid = "source-data-314320"
sql <- "SELECT *
        FROM `source-data-314320.joyeria_dataset.gold_price`
        ORDER BY Date
"


gold_price <- bq_project_query(projectid, sql)
bq_gold_price <- bq_table_download(gold_price)


mxn_tbl <- bq_gold_price[!duplicated(bq_gold_price$date), ] %>%
  select(date, mxn)



# Data Cleaning & Transformation
mxn_transformed_tbl <- mxn_tbl %>%
  summarize_by_time(date, .by = "day", mxn) %>%
  
  #preprocessing
  mutate(mxn_trans = log_interval_vec(mxn, limit_lower = 0)) %>%
  mutate(mxn_trans = standardize_vec(mxn_trans)) %>%
  
  select(-mxn)



mxn_transformed_tbl %>%
  pivot_longer(-date) %>%
  plot_time_series(
    .date_var = date, 
    .value = value, 
    .color_var = name, 
    .smooth = FALSE)




# * Parameters ----
mxn_limit_lower <- 0
mxn_limit_upper <- 26.30372
mxn_offset      <- 0
mxn_std_mean    <- 1.07106922859136
mxn_std_sd      <- 0.349705371745141

mxn_horizon    <- 365
mxn_lag_period <- 31
mxn_rolling_periods <- c(30, 60, 90)


mxn_prepared_full_tbl <- mxn_transformed_tbl %>%
  
  # add future window
  bind_rows(
    future_frame(.data = ., .date_var = date, .length_out = mxn_horizon)
  ) %>%
  
  # add autocorrelated lags -- you're just extending everything by 30 days
  tk_augment_lags(.value = mxn_trans, .lags = mxn_lag_period) %>%
  
  # add rolling features -- this is cool
  tk_augment_slidify(
    .value   = mxn_trans_lag31,
    .f       = mean,
    .period  = mxn_rolling_periods,
    .align   = "center",
    .partial = TRUE) %>%
  
  # format columns
  rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .))



mxn_prepared_full_tbl %>%
  pivot_longer(-date) %>%
  plot_time_series(
    .date_var = date, 
    .value = value, 
    .color_var = name,
    .smooth = FALSE)





# 1.0 TIME-BASED FEATURES ----
# - tk_augment_timeseries_signature()


# * Time Series Signature ----

mxn_prepared_signature_tbl <- mxn_transformed_tbl %>%
  tk_augment_timeseries_signature() %>%
  select(-diff,
         -contains(".iso"),
         -contains(".xts"),
         -matches("(hour)|(minute)|(second)|(am.pm)"))


mxn_prepared_signature_tbl %>%
  plot_acf_diagnostics(.date_var = date,.value = mxn_trans)




# * Model ----

model_formula_mxn <- as.formula(
  mxn_trans ~ splines::ns(x = index.num, 
                          knots = quantile(index.num, probs = c(0.15, 0.30, 0.69, 0.72, 0.79, 0.95)))
  + (as.factor(week4) * month.lbl)
  + month.lbl + wday.lbl
)



# Visualize

mxn_prepared_signature_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = model_formula_mxn,
    .show_summary = TRUE
  )


# Linear Regression Model

model_mxn_best_fit <- lm(model_formula_mxn, data = mxn_prepared_signature_tbl)



# 2.0 STEP 2 - SEPARATE INTO MODELING & FORECAST DATA ----

mxn_prepared_tbl <- mxn_prepared_full_tbl %>%
  filter(!is.na(mxn_trans))


mxn_forecast_tbl <- mxn_prepared_full_tbl %>%
  filter(is.na(mxn_trans))


# 3.0 TRAIN/TEST (MODEL DATASET) ----


mxn_splits <- time_series_split(mxn_prepared_tbl, assess = mxn_horizon, cumulative = TRUE)

mxn_splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(.date_var = date, .value = mxn_trans)


# 4.0 RECIPES ----

model_mxn_best_fit %>% summary()
model_mxn_best_fit$terms %>% formula()


recipe_mxn_base <-  recipe(mxn_trans ~ ., data = training(mxn_splits)) %>%
  
  step_timeseries_signature(date) %>%
  step_rm(matches("(iso)|(xts)|(hour)|(minute)|(second)|(am.pm)")) %>%
  
  step_normalize(matches("(index.num)|(year)|(yday)")) %>%
  
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  
  step_interact(~ matches("week4") * matches("wday.lbl")) %>%
  
  step_fourier(date, period = c(7, 14, 30, 90, 365), K = 2)
  
  


recipe_mxn_base %>%
  prep() %>%
  juice() %>%
  glimpse()






# 5.0 SPLINE MODEL ----

# * LM Model Spec ----

model_mxn_lm <- linear_reg() %>%
  set_engine("lm")



# * Spline Recipe Spec ----

recipe_mxn_base %>% 
  prep() %>% 
  juice() %>%
  glimpse()


recipe_mxn_1 <- recipe_mxn_base %>%
  step_rm(date) %>%
  step_ns(ends_with("index.num"), deg_free = 2) %>%
  step_rm(starts_with("lag_"))
  

recipe_mxn_1 %>% 
  prep() %>% 
  juice() %>%
  glimpse()


# * Spline Workflow  ----

workflow_fit_mxn_lm_1_spline <- workflow() %>%
  add_model(model_mxn_lm) %>%
  add_recipe(recipe_mxn_1) %>%
  fit(training(mxn_splits))

workflow_fit_mxn_lm_1_spline


workflow_fit_mxn_lm_1_spline %>% 
  pull_workflow_fit()  %>%
  pluck("fit") %>%
  summary()




# 6.0 MODELTIME  ----

calibration_mxn_tbl <- modeltime_table(workflow_fit_mxn_lm_1_spline) %>%
  modeltime_calibrate(new_data = testing(mxn_splits))


calibration_mxn_tbl %>%
  modeltime_forecast(new_data = testing(mxn_splits), actual_data = mxn_prepared_tbl) %>%
  plot_modeltime_forecast()


calibration_mxn_tbl %>% 
  modeltime_accuracy()



# 7.0 LAG MODEL ----

# * Lag Recipe ----

# we dropped the lags previously, but now we want to check the recipe WITH the lags.

recipe_mxn_base %>% 
  prep() %>% 
  juice() %>%
  glimpse()


recipe_mxn_2 <- recipe_mxn_base %>% 
  step_rm(date) %>%
  step_naomit(starts_with("lag_"))


recipe_mxn_2 %>% 
  prep() %>% 
  juice() %>%
  glimpse()



# * Lag Workflow ----

workflow_fit_mxn_lm_2_lag <- workflow() %>%
  add_model(model_mxn_lm) %>%
  add_recipe(recipe_mxn_2) %>%
  fit(training(mxn_splits))

workflow_fit_mxn_lm_2_lag


workflow_fit_mxn_lm_2_lag %>% 
  pull_workflow_fit() %>% 
  pluck("fit") %>% 
  summary()



# * Compare the two models with Modeltime -----


calibration_mxn_tbl <- modeltime_table(
  workflow_fit_mxn_lm_1_spline,
  workflow_fit_mxn_lm_2_lag
) %>%
  modeltime_calibrate(new_data = testing(mxn_splits), quiet = FALSE)


calibration_mxn_tbl %>%
  modeltime_forecast(new_data = testing(mxn_splits),
                     actual_data = mxn_prepared_tbl) %>%
  plot_modeltime_forecast()


calibration_mxn_tbl %>%
  modeltime_accuracy()




# 8.0 FUTURE FORECAST ----

# this is where you are retraining the data
refit_mxn_tbl <- calibration_mxn_tbl %>%
  modeltime_refit(data = mxn_prepared_tbl)



# now you pull in the forecast as new data.
refit_mxn_tbl %>%
  modeltime_forecast(new_data = mxn_forecast_tbl,
                     actual_data = mxn_prepared_tbl) %>%
  
  # invert so you can see actual amounts
  mutate(across(.value:.conf_hi, .fns = ~ standardize_inv_vec(
    x    = ., 
    mean = mxn_std_mean, 
    sd   = mxn_std_sd)
  )) %>%
  
  mutate(across(.value:.conf_hi, .fns = ~ log_interval_inv_vec(
    x           = ., 
    limit_lower = mxn_limit_lower, 
    limit_upper = mxn_limit_upper, 
    offset      = mxn_offset))) %>%
  
  plot_modeltime_forecast()










