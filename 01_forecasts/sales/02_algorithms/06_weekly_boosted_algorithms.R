# LIBRARIES & SETUP ----

# Time Series ML
library(tidymodels)
library(modeltime)

# Core 
library(tidyverse)
library(lubridate)
library(timetk)



# DATA & ARTIFACTS ----

weekly_artifacts_list  <- read_rds("00_models/weekly_artifacts_list.rds")

prepared_weekly_tbl    <- weekly_artifacts_list$data$prepared_weekly_tbl
forecast_weekly_tbl    <- weekly_artifacts_list$data$forecast_weekly_tbl

recipe_weekly_base     <- weekly_artifacts_list$recipes$recipe_weekly_base
recipe_weekly_1_spline <- weekly_artifacts_list$recipes$recipe_weekly_1_spline
recipe_weekly_2_lag    <- weekly_artifacts_list$recipes$recipe_weekly_2_lag

std_weekly_mean        <- weekly_artifacts_list$standardize$std_weekly_mean
std_weekly_sd          <- weekly_artifacts_list$standardize$std_weekly_sd



# FUNCTIONS & MODELS ----
source("00_scripts/01_calibrate_and_plot.R")

model_fit_best_prophet <- read_rds("00_models/model_fit_best_prophet.rds")
model_fit_best_arima   <- read_rds("00_models/model_fit_best_arima.rds")



# TRAIN / TEST ----

weekly_splits <- prepared_weekly_tbl %>%
  time_series_split(
    date_var = date, 
    assess = "52 weeks",
    cumulative = TRUE)


weekly_splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(.date_var = date, .value = sales)







# 1.0 PROPHET BOOST ----

# * Best Prophet Model ----

model_fit_best_prophet


calibrate_and_plot(
  model_fit_best_prophet
)


model_fit_best_prophet$preproc$terms %>% formula()





# * Boosting Prophet Models ----

# Recipes

recipe_weekly_base_no_lag <- recipe_weekly_base %>%
  step_rm(starts_with("lag"))


recipe_weekly_base_no_lag %>%
  prep() %>%
  juice() %>%
  glimpse()


# Model Spec
model_weekly_prophet_boost <- prophet_boost(
  
  # prophet params
  changepoint_num    = 25,
  changepoint_range  = 0.8,
  seasonality_daily  = FALSE,
  seasonality_weekly = FALSE,
  seasonality_yearly = TRUE,
  
  # xgboost params
  mtry           = 0.1,
  min_n          = 1,
  tree_depth     = 3,
  learn_rate     = 0.1,
  loss_reduction = 0.5,
  trees          = 10
) %>%
  set_engine("prophet_xgboost", counts = FALSE)



# Workflow
set.seed(123)
wflw_fit_prophet_boost <- workflow() %>%
  add_model(model_weekly_prophet_boost) %>%
  add_recipe(recipe_weekly_base_no_lag) %>%
  fit(training(weekly_splits))


calibrate_and_plot(
  model_fit_best_prophet,
  wflw_fit_prophet_boost,
  type = "testing"
)




# check the residuals (should be close to zero)

modeltime_table(
  wflw_fit_prophet_boost
) %>%
  modeltime_residuals(training(weekly_splits)) %>%
  plot_modeltime_residuals()




# 2.0 ARIMA BOOST ----

# * Best ARIMA Model ----

model_fit_best_arima

calibrate_and_plot(
  model_fit_best_arima,
  type = "testing"
)


model_fit_best_arima$preproc$terms %>% formula()



# * Boosting ARIMA ----

model_weekly_arima_boost <- arima_boost(
  # arima params
  seasonal_period = 1,
  
  # xgboost params
  mtry           = 0.7,
  min_n          = 50,
  tree_depth     = 1,
  learn_rate     = 0.75,
  loss_reduction = 0.10,
  trees          = 200
) %>%
  set_engine("auto_arima_xgboost", counts = FALSE)


set.seed(123)
wflw_fit_arima_boost <- wflw_fit_prophet_boost %>%
  update_model(model_weekly_arima_boost) %>%
  fit(training(weekly_splits))


calibrate_and_plot(
  wflw_fit_arima_boost,
  type = "testing"
)




# 3.0 MODELTIME EVALUATION ----

# * Modeltime ----

model_weekly_tbl <- modeltime_table(
  model_fit_best_prophet,
  wflw_fit_prophet_boost,
  
  model_fit_best_arima,
  wflw_fit_arima_boost
)




# * Calibration ----

calibration_tbl <- model_weekly_tbl %>% 
  modeltime_calibrate(testing(weekly_splits))




# * Accuracy Test ----

calibration_tbl %>% modeltime_accuracy()




# * Forecast Test ----

calibration_tbl %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)




# * Refit & Forecast Future ----

refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = prepared_weekly_tbl)


forecast_future_tbl <- refit_tbl %>%
  modeltime_forecast(new_data = forecast_weekly_tbl,
                     actual_data = prepared_weekly_tbl)


forecast_future_tbl %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)




# * Invert ----
forecast_future_tbl <- calibration_tbl %>%
  modeltime_forecast(new_data = forecast_weekly_tbl,
                     actual_data = prepared_weekly_tbl) %>%
  
  mutate(across(.cols = .value:.conf_hi,
                .fns = ~ standardize_inv_vec(x = ., 
                                             mean = std_weekly_mean, 
                                             sd = std_weekly_sd))) %>%
  
  mutate(across(.cols = .value:.conf_hi, 
                .fns = exp))




# Plot Weekly ----
forecast_future_tbl %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)




# Plot Monthly ----
forecast_future_tbl %>%
  group_by(.model_id, .model_desc, .key, .index) %>%
  summarize_by_time(.by = "month", .value = sum(.value)) %>%
  ungroup() %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)





# 4.0 SAVE ARTIFACTS ----

calibration_tbl %>%
  write_rds("00_models/calibration_tbl_boosted_models.rds")





