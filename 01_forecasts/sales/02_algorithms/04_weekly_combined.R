# LIBRARIES ----

# Time Series ML
library(tidymodels)
library(modeltime)

# Core
library(tidyverse)
library(timetk)
library(lubridate)



# DATA & ARTIFACTS ----

weekly_artifacts_list  <- read_rds("00_models/weekly_artifacts_list.rds")

prepared_weekly_tbl    <- weekly_artifacts_list$data$prepared_weekly_tbl
forecast_weekly_tbl    <- weekly_artifacts_list$data$forecast_weekly_tbl

recipe_weekly_1_spline <- weekly_artifacts_list$recipes$recipe_weekly_1_spline

std_weekly_mean        <- weekly_artifacts_list$standardize$std_weekly_mean
std_weekly_sd          <- weekly_artifacts_list$standardize$std_weekly_sd




# TRAIN / TEST ----

weekly_splits <- prepared_weekly_tbl %>%
  time_series_split(
    date_var = date, 
    assess = "52 weeks",
    cumulative = TRUE)


weekly_splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(
    .date_var = date,
    .value = sales)

training(weekly_splits) %>%
  plot_acf_diagnostics(.date_var = date, .value = diff_vec(sales))




# ARIMA ----

# * Model 1 - Basic Auto Arima ----

model_fit_1_weekly_arima <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(sales ~ date, 
      training(weekly_splits))


model_fit_1_weekly_arima



# * Model 2 - Add XRegs ----

model_fit_2_weekly_arima_xregs <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(sales ~ date
      + event, 
      training(weekly_splits))

model_fit_2_weekly_arima_xregs




# * Model 3 - Yearly Seasonality + XRegs ----

model_fit_3_weekly_arima_sarimax <- arima_reg(
  seasonal_period = 52) %>%
  set_engine("auto_arima") %>%
  fit(sales ~ date
      + event, 
      training(weekly_splits))

model_fit_3_weekly_arima_sarimax




# * Model 4 - Quarterly Seasonality + XRegs ----

model_fit_4_weekly_arima_sarimax <- arima_reg(
    seasonal_period = 13, 
    non_seasonal_ar = 2,
    non_seasonal_differences = 1, 
    non_seasonal_ma = 2,
    seasonal_ar = 1,
    seasonal_differences = 0, 
    seasonal_ma = 1
  ) %>%
  set_engine("arima") %>%
  fit(sales ~ date
      + event, 
      training(weekly_splits))

model_fit_4_weekly_arima_sarimax




# * Model 5 - Arima + XRegs + Fourier ----

model_fit_5_weekly_arima_xreg_fourier <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(sales ~ date
      + event
      + fourier_vec(date, period = 5)
      + fourier_vec(date, period = 7)
      + fourier_vec(date, period = 13)
      + fourier_vec(date, period = 20)
      + fourier_vec(date, period = 26)
      + fourier_vec(date, period = 53),
      training(weekly_splits))

model_fit_5_weekly_arima_xreg_fourier






# * * Modeltime: Calibrate & Test ----
model_tbl_arima <- modeltime_table(
  model_fit_1_weekly_arima,
  model_fit_2_weekly_arima_xregs,
  model_fit_3_weekly_arima_sarimax,
  model_fit_4_weekly_arima_sarimax,
  model_fit_5_weekly_arima_xreg_fourier
)

model_tbl_arima


# Calibrate
calibration_tbl <- model_tbl_arima %>%
  modeltime_calibrate(testing(weekly_splits))


# Test Forecast & Accuracy
calibration_tbl %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast()


calibration_tbl %>%
  modeltime_accuracy()





# PROPHET ----

# * Model 6 - Basic Prophet ----

model_fit_6_prophet_basic <- prophet_reg() %>%
  set_engine("prophet") %>%
  fit(sales ~ date, 
      training(weekly_splits))





# * Model 7 - Prophet + Yearly ----

model_fit_7_prophet_yearly <- prophet_reg(
  seasonality_yearly = TRUE
) %>%
  set_engine("prophet") %>%
  fit(sales ~ date, 
      training(weekly_splits))





# * Model 8 - Prophet + Events ----

model_fit_8_prophet_event <- prophet_reg(
  seasonality_yearly = FALSE
) %>%
  set_engine("prophet") %>%
  fit(sales ~ date 
      + event, 
      training(weekly_splits))





# * Model 9 - Prophet + XRegs + Fourier ----

model_fit_9_prophet_fourier <- prophet_reg(
  seasonality_yearly = TRUE
) %>%
  set_engine("prophet") %>%
  fit(sales ~ date 
      + event
      + fourier_vec(date, period = 5)
      + fourier_vec(date, period = 7)
      + fourier_vec(date, period = 13)
      + fourier_vec(date, period = 20)
      + fourier_vec(date, period = 26)
      + fourier_vec(date, period = 53), 
      training(weekly_splits))






# * * Modeltime: Calibrate & Test ----
model_tbl_prophet <-  modeltime_table(
  model_fit_6_prophet_basic,
  model_fit_7_prophet_yearly,
  model_fit_8_prophet_event,
  model_fit_9_prophet_fourier
)


# Calibrate on testing data
calibration_tbl <- model_tbl_prophet %>%
  modeltime_calibrate(testing(weekly_splits))


# Test Forecast & Accuracy
calibration_tbl %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast()



calibration_tbl %>%
  modeltime_accuracy()









# EXPONENTIAL SMOOTHING ----

# * Model 10 - ETS ----

model_fit_10_ets <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales ~ date, training(weekly_splits))

model_fit_10_ets


# * Model 11 - TBATS ----

model_fit_11_tbats <- seasonal_reg(seasonal_period_1 = 5,
                                   seasonal_period_2 = 13, 
                                   seasonal_period_3 = 26) %>%
  set_engine("tbats") %>%
  fit(sales ~ date, training(weekly_splits))

model_fit_11_tbats





# * * Modeltime: Calibrate & Test ----
model_tbl_exp_smooth <- modeltime_table(
  model_fit_10_ets,
  model_fit_11_tbats
)


# Calibrate on testing data
calibration_tbl <-  model_tbl_exp_smooth %>%
  modeltime_calibrate(new_data = testing(weekly_splits))


# Test Forecast & Accuracy
calibration_tbl %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast()


calibration_tbl %>% 
  modeltime_accuracy()






# COMBINE AND COMPARE ----

# * Combine ----

model_tbl <- combine_modeltime_tables(
  model_tbl_arima,
  model_tbl_prophet,
  model_tbl_exp_smooth
) %>%
  update_model_description(1, "arima basic") %>%
  update_model_description(2, "arima xregs") %>%
  update_model_description(3, "sarimax yearly") %>% 
  update_model_description(4, "sarimax quarterly") %>%
  update_model_description(5, "arima xregs fourier") %>%
  update_model_description(6, "prophet basic") %>%
  update_model_description(7, "prophet yearly") %>%
  update_model_description(8, "prophet xregs") %>%
  update_model_description(9, "prophet xregs fourier") %>%
  update_model_description(10, "ets") %>%
  update_model_description(11, "tbats")



# * Refit ----
# Refit models on training data, just to make sure they work over time
model_tbl <- model_tbl %>%
  modeltime_refit(training(weekly_splits))




# * Calibrate ----
calibration_tbl <- model_tbl %>%
  modeltime_calibrate(new_data = testing(weekly_splits))



# * Test ----
calibration_tbl %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)


calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy()








# FORECAST THE FUTURE ----

# * Refit ----
refit_tbl <- calibration_tbl %>%
  modeltime_refit(prepared_weekly_tbl) %>%
  
  update_model_description(1, "arima basic") %>%
  update_model_description(2, "arima xregs") %>%
  update_model_description(3, "sarimax yearly") %>% 
  update_model_description(4, "sarimax quarterly") %>%
  update_model_description(5, "arima xregs fourier") %>%
  update_model_description(6, "prophet basic") %>%
  update_model_description(7, "prophet yearly") %>%
  update_model_description(8, "prophet xregs") %>%
  update_model_description(9, "prophet xregs fourier") %>%
  update_model_description(10, "ets") %>%
  update_model_description(11, "tbats")




# * Forecast Future ----
refit_tbl %>%
  modeltime_forecast(new_data = forecast_weekly_tbl,
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)




# * Invert ----
forecast_future_tbl <- refit_tbl %>%
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
  



