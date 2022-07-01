# GOAL: Forecast Gold Price in MXN

# OBJECTIVES ----
# - Forecast using 3 models
#   1. Prophet
#   2. LM Spline
#   3. LM Lag Features



# Libraries ----
# Time Series Machine Learning
library(tidymodels)
library(modeltime)


# Exploratory Data Analysis
library(DataExplorer)

# Core
library(tidyverse)
library(timetk)
library(lubridate)
library(bigrquery)



# DATA ----

# * Pull Data ----

projectid = "source-data-314320"
sql_gold <- "SELECT *
             FROM `source-data-314320.joyeria_dataset.gold_price`
             ORDER BY Date
"


# Run the query and store the data in a tibble
gold_price <- bq_project_query(projectid, sql_gold)
bq_gold_price <- bq_table_download(gold_price)





# 1.0 DATA PREP ----

bq_gold_price <- bq_gold_price[!duplicated(bq_gold_price$date), ]


bq_gold_price %>%
  plot_time_series(
    .date_var = date, 
    .value    = mxn_per_gram, 
    .smooth   = FALSE)


bq_gold_price %>%
  plot_acf_diagnostics(
    .date_var = date,
    .value    = mxn_per_gram
  )


# 2.0 TRANSFORMATION ----

# * Log & Standardize ----


gold_trans_daily_tbl <- bq_gold_price %>%
  select(date, mxn_per_gram) %>%
  
  mutate(mxn_per_gram = log(mxn_per_gram)) %>%
  mutate(mxn_per_gram = standardize_vec(mxn_per_gram))



gold_trans_daily_tbl %>%
  plot_time_series(.date_var = date, .value = mxn_per_gram)


gold_trans_daily_tbl %>%
  plot_acf_diagnostics(
    .date_var = date, 
    .value = diff_vec(mxn_per_gram))


# 3.0 CREATE FULL DATASET ----

# * Save Key Params ----

gold_horizon <- 240
gold_lag_periods <- 240
gold_rolling_periods <- c(8, 28, 55, 60, 87, 112, 119, 138, 146)

std_gold_mean <- 6.79741406992952
std_gold_sd <- 0.241322766834133


# * Prepare Full Dataset ----

gold_prepared_full_daily_tbl <- gold_trans_daily_tbl %>%
  # add future window
  bind_rows(
    future_frame(
      .data = .,
      .date_var = date, 
      .length_out = "240 days")
  ) %>%
  
  tk_augment_lags(
    .value = mxn_per_gram, 
    .lags = gold_lag_periods) %>%
  
  tk_augment_slidify(
    .value = mxn_per_gram_lag240, 
    .period = gold_rolling_periods, 
    .f = mean, 
    .align = "center", 
    .partial = TRUE) %>%
  
  rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .))



# * Visualize ----

gold_prepared_full_daily_tbl %>%
  pivot_longer(-date) %>%
  plot_time_series(
    .date_var  = date, 
    .value     = value, 
    .color_var = name, 
    .smooth    = FALSE)



# * Model Data / Forecast Data Split ----

gold_prepared_daily_tbl <- gold_prepared_full_daily_tbl %>%
  filter(!is.na(mxn_per_gram))


gold_forecast_daily_tbl <- gold_prepared_full_daily_tbl %>%
  filter(is.na(mxn_per_gram))



# * Train / Test Split ----

gold_splits <- gold_prepared_daily_tbl %>%
  time_series_split(
  date_var = date, 
  assess = "240 days", 
  cumulative = TRUE
  )


gold_splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(
    .date_var = date, 
    .value = mxn_per_gram)




# 4.0 PROPHET FORECAST ----

# Prophet using Modeltime/Parsnip ----

model_prophet_gold_fit <- prophet_reg(
  seasonality_yearly = TRUE, 
  seasonality_weekly = TRUE, 
  changepoint_range = 0.75
) %>%
  set_engine("prophet") %>%
  fit(mxn_per_gram ~ date, data = training(gold_splits))



# * Modeltime Process ----

model_gold_tbl <- modeltime_table(
  model_prophet_gold_fit
)



# * Calibrate ----

calibration_gold_tbl <- model_gold_tbl %>%
  modeltime_calibrate(new_data = testing(gold_splits))



# * Visualize Forecast ----

calibration_gold_tbl %>%
  modeltime_forecast(actual_data = gold_prepared_daily_tbl) %>%
  plot_modeltime_forecast()



# * Get Accuracy Metrics ----

calibration_gold_tbl %>%
  modeltime_accuracy()




# 4.1 LM FEATURE ENGINEERING ----

# * Identify Possible Features ----

# plot seasonality, use log() transformation in value to see more variation

gold_prepared_daily_tbl %>%
  plot_seasonal_diagnostics(
    .date_var = date, 
    .value = mxn_per_gram)



# * Base Recipe ----


recipe_gold_base <- recipe(mxn_per_gram ~., data = training(gold_splits)) %>%
  
  # Time Series Signature
  step_timeseries_signature(date) %>%
  
  # this next step removes unnecessary columns and cleans up dataframe
  step_rm(matches("(iso)|(xts)|(hour)|(day)|(minute)|(second)|(am.pm)")) %>%
  
  # now we normalize the time and year, because those were big numbers and they look weird.
  step_normalize(matches("(index.num)|(year)|(yday)")) %>%
  
  # now we create dummy variables from categorical data.
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  
  # create fourier series
  step_fourier(date, period = c(8, 28, 55, 60, 87, 112, 119, 138, 146), K = 2)


recipe_gold_base %>%
  prep() %>% 
  juice() %>%
  glimpse()




# Spline Model ----

# * Visualize ----

gold_prepared_daily_tbl %>%
  plot_time_series_regression(.date_var = date, 
                              .formula = mxn_per_gram ~ splines::ns(date, df = 12),
                              .show_summary = TRUE)



# * LM Model Specs ----

model_gold_daily_lm <- linear_reg() %>%
  set_engine("lm")



# * Recipe Spec - Spline ----

recipe_gold_1_spline <- recipe_gold_base %>%
  step_rm(date) %>%
  step_ns(contains("index.num"), deg_free = 12) %>%
  step_rm(starts_with("lag_"))


recipe_gold_1_spline %>%
  prep() %>%
  juice()%>%
  glimpse()



# * Workflow - Spline ----

workflow_fit_gold_lm_1_spline <- workflow() %>%
  add_model(model_gold_daily_lm) %>%
  add_recipe(recipe_gold_1_spline) %>%
  fit(training(gold_splits))


workflow_fit_gold_lm_1_spline %>%
  pull_workflow_fit() %>%
  pluck("fit") %>%
  summary()



# Rolling Lag Model ----

# * Recipe Spec - Lag ----

recipe_gold_2_lag <- recipe_gold_base %>%
  step_rm(date) %>%
  step_naomit(starts_with("lag_"))


recipe_gold_2_lag %>%
  prep() %>%
  juice()%>%
  glimpse()



# * Workflow - Spline ----

workflow_fit_gold_lm_2_lag <- workflow() %>%
  add_model(model_gold_daily_lm) %>%
  add_recipe(recipe_gold_2_lag) %>%
  fit(training(gold_splits))


workflow_fit_gold_lm_2_lag %>%
  pull_workflow_fit() %>%
  pluck("fit") %>%
  summary()



# 5.0 MODELTIME ----

# * Update Modeltime Table ----

model_gold_tbl <- modeltime_table(
  model_prophet_gold_fit,
  workflow_fit_gold_lm_1_spline,
  workflow_fit_gold_lm_2_lag
)



# As a precautionary measure, refit the models using modeltime_refit()
# This prevents models that can go bad over time because of software changes

model_gold_tbl <- model_gold_tbl %>%
  modeltime_refit(training(gold_splits))



# Updated the Calibration Data ----

calibration_gold_tbl <- model_gold_tbl %>% 
  modeltime_calibrate(new_data = testing(gold_splits))



# Check Accuracy ----

calibration_gold_tbl %>% modeltime_accuracy()



# Visualize Models on Testing Data----

calibration_gold_tbl %>%
  modeltime_forecast(new_data = testing(gold_splits), 
                     actual_data = gold_prepared_daily_tbl) %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)




# 6.0 FUTURE FORECAST ----

# * Refit to Prepared Data ----

refit_gold_daily_tbl <- calibration_gold_tbl %>%
  modeltime_refit(data = gold_prepared_daily_tbl)



# * Forecast with Forecast_Data ----

refit_gold_daily_tbl %>%
  modeltime_forecast(new_data = gold_forecast_daily_tbl, 
                     actual_data = gold_prepared_daily_tbl) %>%
  plot_modeltime_forecast()



# * Invert Transformaton ----

refit_gold_daily_tbl %>%
  modeltime_forecast(new_data = gold_forecast_daily_tbl, 
                     actual_data = gold_prepared_daily_tbl) %>%
  
  mutate(across(.cols = .value:.conf_hi, .fns = ~ standardize_inv_vec(x = ., 
                                                                      mean = std_gold_mean, 
                                                                      sd = std_gold_sd))) %>%
  
  mutate(across(.cols = .value:.conf_hi, 
                .fns = exp)) %>%

  plot_modeltime_forecast(.conf_interval_show = FALSE)






# BONUS ----

## GLMNET - Elastic Net ----

workflow_fit_gold_glmnet_2_lag <- workflow_fit_gold_lm_2_lag %>%
  update_model(
    spec = linear_reg(penalty = 0.1, mixture = 0.5) %>%
      set_engine("glmnet")
  ) %>%
  fit(training(gold_splits))


calibration_gold_tbl <- modeltime_table(
  model_prophet_gold_fit,
  workflow_fit_gold_lm_1_spline,
  workflow_fit_gold_lm_2_lag,
  workflow_fit_gold_glmnet_2_lag
) %>%
  update_model_description(1, "prophet") %>%
  update_model_description(2, "lm - spline") %>%
  update_model_description(3, "lm - lag") %>%
  update_model_description(4, "glmnet - lag") %>%
  
  # FIX - NEED TO REFIT TO THE TRAINING DATASET (TO UPDATE THE OLD LM MODELS)
  modeltime_refit(training(gold_splits)) %>%
  
  modeltime_calibrate(testing(gold_splits))



calibration_gold_tbl %>% modeltime_accuracy()


calibration_gold_tbl %>%
  modeltime_forecast(new_data = testing(gold_splits),
                     actual_data = gold_prepared_daily_tbl) %>%
  plot_modeltime_forecast()


refit_gold_daily_tbl <- calibration_gold_tbl %>%
  modeltime_refit(data = gold_prepared_daily_tbl)




refit_gold_daily_tbl %>%
  modeltime_forecast(new_data = gold_forecast_daily_tbl,
                     actual_data = gold_prepared_daily_tbl) %>%
  
  mutate(across(.cols = .value:.conf_hi, 
                .fns = ~ standardize_inv_vec(x = ., 
                                             mean = std_gold_mean, 
                                             sd = std_gold_sd))) %>%
  
  mutate(across(.cols = .value:.conf_hi, 
                .fns = exp)) %>%
  
  plot_modeltime_forecast(.conf_interval_show = FALSE)












