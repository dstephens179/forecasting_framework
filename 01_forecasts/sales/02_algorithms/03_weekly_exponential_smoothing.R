# GOAL: Understand Exponential Smoothing

# OBJECTIVES ----
# - ETS - Exponential Smoothing
# - TBATS - Multiple Seasonality Models
# - Seasonal Decomposition - Multiple Seasonality Models

# LIBRARIES & SETUP ----

# Time Series ML
library(tidymodels)
library(modeltime)
library(forecast)

# Core 
library(tidyverse)
library(lubridate)
library(timetk)




# DATA & ARTIFACTS ----

weekly_artifacts_list  <- read_rds("00_models/weekly_artifacts_list.rds")

prepared_weekly_tbl    <- weekly_artifacts_list$data$prepared_weekly_tbl
forecast_weekly_tbl    <- weekly_artifacts_list$data$forecast_weekly_tbl

std_weekly_mean        <- weekly_artifacts_list$standardize$std_weekly_mean
std_weekly_sd          <- weekly_artifacts_list$standardize$std_weekly_sd




# TRAIN / TEST ----

weekly_splits <- prepared_weekly_tbl %>%
  time_series_split(
    date_var   = date, 
    assess     = "18 weeks",
    cumulative = TRUE)


weekly_splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(
    .date_var = date,
    .value    = sales)


training(weekly_splits) %>%
  plot_acf_diagnostics(.date_var = date, .value = diff_vec(sales))



# 1.0 EXPONENTIAL SMOOTHING (ETS) -----
# - Error, Trend, Seasonal Model - Holt-Winters Seasonal
# - Automatic forecasting method based on Exponential Smoothing
# - Single Seasonality
# - Cannot use Xregs (purely univariate)


# * ETS Model

model_fit_weekly_ets <- exp_smoothing(
  error = "additive",
  trend = "additive", 
  season = "additive"
) %>%
  set_engine("ets") %>%
  fit(sales ~ date, data = training(weekly_splits))



# * Modeltime

modeltime_table(
  model_fit_weekly_ets
) %>%
  modeltime_calibrate(new_data = testing(weekly_splits)) %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast()




# 2.0 TBATS ----
# - Multiple Seasonality Model
# - Extension of ETS for complex seasonality
# - Automatic
# - Does not support XREGS

# * TBATS Model

model_fit_weekly_tbats <- seasonal_reg(
  seasonal_period_1 = 4,
  seasonal_period_2 = 26,
  seasonal_period_3 = 52
) %>%
  set_engine("tbats") %>%
  fit(sales ~ date, data = training(weekly_splits))



# * Modeltime

modeltime_table(
  model_fit_weekly_ets,
  model_fit_weekly_tbats
) %>%
  modeltime_calibrate(new_data = testing(weekly_splits)) %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)







# 3.0 SEASONAL DECOMPOSITION ----
# - Uses seasonal decomposition to model 
#   trend & seasonality separately
#   - Trend modeled with ARIMA or ETS
#   - Seasonality modeled with Seasonal Naive (SNAIVE)
# - Can handle multiple seasonality
# - ARIMA version accepts XREGS, ETS does not

# * STLM ETS Model

model_fit_weekly_stlm_ets <- seasonal_reg(
  seasonal_period_1 = 4,
  seasonal_period_2 = 26,
  seasonal_period_3 = 52
) %>%
  set_engine("stlm_ets") %>%
  fit(sales ~ date, data = training(weekly_splits))



model_fit_weekly_stlm_ets$fit$models$model_1$stl %>% autoplot()


# * STLM ARIMA + XREGS

model_fit_weekly_stlm_arima <- seasonal_reg(
  seasonal_period_1 = 4,
  seasonal_period_2 = 26,
  seasonal_period_3 = 52
) %>%
  set_engine("stlm_arima") %>%
  fit(sales ~ date, data = training(weekly_splits))



model_fit_weekly_stlm_arima_xregs <- seasonal_reg(
  seasonal_period_1 = 4,
  seasonal_period_2 = 26,
  seasonal_period_3 = 52
) %>%
  set_engine("stlm_arima") %>%
  fit(sales ~ date + event, data = training(weekly_splits))







# 4.0 EVALUATION ----

# * Modeltime

model_weekly_tbl <- modeltime_table(
  model_fit_weekly_ets,
  model_fit_weekly_tbats,
  model_fit_weekly_stlm_ets,
  model_fit_weekly_stlm_arima,
  model_fit_weekly_stlm_arima_xregs
) 



# * Calibration

calibration_weekly_tbl <- model_weekly_tbl %>%
  modeltime_calibrate(new_data = testing(weekly_splits))


# * Forecast Test

calibration_weekly_tbl %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)




# * Accuracy Test

calibration_weekly_tbl %>%
  modeltime_accuracy()



# * Refit

refit_weekly_tbl <- calibration_weekly_tbl %>%
  modeltime_refit(data = prepared_weekly_tbl) %>%
  update_model_description(1, "ets") %>%
  update_model_description(2, "tbats") %>%
  update_model_description(3, "stlm ets") %>%
  update_model_description(4, "stlm arima") %>%
  update_model_description(5, "stlm arima+xregs")


refit_weekly_tbl %>%
  modeltime_forecast(new_data = forecast_weekly_tbl,
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)




# * Invert

forecast_future_weekly_tbl <- refit_weekly_tbl %>% 
  modeltime_forecast(
    new_data = forecast_weekly_tbl, 
    actual_data = prepared_weekly_tbl,
    conf_interval = 0.80
  ) %>%
  
  # standardize_inv_vec
  mutate(across(
    .cols = .value:.conf_hi, 
    .fns = ~ standardize_inv_vec(x = ., 
                                 mean = std_weekly_mean, 
                                 sd = std_weekly_sd))) %>%
  
  # reverse log, which is exp
  mutate(across(
    .cols = .value:.conf_hi, 
    .fns = exp))



# * Visualize Weekly ----
forecast_future_weekly_tbl %>%
  plot_modeltime_forecast(
    .legend_max_width = 25,
    .conf_interval_show = FALSE,
    .title = "Weekly Sales Forecast"
  )


# * Visualize Monthly ----
forecast_future_weekly_tbl %>%
  group_by(.model_id, .model_desc, .key, .index) %>%
  summarize_by_time(.by = "month", .value = sum(.value)) %>%
  ungroup() %>%
  plot_modeltime_forecast(
    .legend_max_width = 25,
    .conf_interval_show = FALSE,
    .title = "Monthly Sales Forecast"
  )







# SAVING ALL MODELS ----

calibration_weekly_tbl %>%
  write_rds("00_models/calibration_tbl_ets_tbats.rds")

read_rds("00_models/calibration_tbl_ets_tbats.rds")









