# LIBRARIES ----

# Time Series ML
library(tidymodels)
library(modeltime)
library(prophet)

# Core 
library(tidyverse)
library(lubridate)
library(timetk)
library(plotly)



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
    assess     = "52 weeks",
    cumulative = TRUE)


weekly_splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(
    .date_var = date,
    .value    = sales)


training(weekly_splits) %>%
  plot_acf_diagnostics(.date_var = date, .value = diff_vec(sales))



# 1.0 PROPHET MODEL ----

# * Modeltime Model

model_fit_weekly_prophet <- prophet_reg(
  changepoint_num = 25,
  changepoint_range = 0.8,
  seasonality_yearly = TRUE, 
  seasonality_weekly = FALSE
) %>%
  set_engine("prophet") %>%
  fit(sales ~ date, data = training(weekly_splits))



model_weekly_tbl <- modeltime_table(
  model_fit_weekly_prophet
) %>%
  update_model_description(1, "prophet")



model_weekly_tbl %>%
  modeltime_calibrate(new_data = testing(weekly_splits)) %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast()






# 2.0 PROPHET + XREGS ----

# * Model

model_fit_weekly_prophet_xregs <- prophet_reg(
  changepoint_num = 25,
  changepoint_range = 0.8,
  seasonality_yearly = TRUE, 
  seasonality_weekly = FALSE
) %>%
  set_engine("prophet") %>%
  fit(sales ~ date + event, data = training(weekly_splits))


# * Calibration

calibration_weekly_tbl <- modeltime_table(
  model_fit_weekly_prophet,
  model_fit_weekly_prophet_xregs
) %>%
  modeltime_calibrate(testing(weekly_splits))



# * Forecast Test

calibration_weekly_tbl %>%
  modeltime_forecast(
    new_data = testing(weekly_splits),
    actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)





# 3.0 COMPARE & VISUALIZE ----

# * Accuracy

calibration_weekly_tbl %>%
  modeltime_accuracy()


# Table Modeltime Accuracy

calibration_weekly_tbl %>%
  modeltime_accuracy(
    metric_set = default_forecast_accuracy_metric_set()
  ) %>%
  table_modeltime_accuracy(
    .interactive = TRUE, 
    bordered = TRUE, 
    resizable = TRUE)



# Metric Sets

calibration_weekly_tbl %>%
  modeltime_accuracy(
    metric_set = metric_set(mae, rmse, rsq)
  )



# * Refit

refit_weekly_tbl <- calibration_weekly_tbl %>%
  modeltime_refit(data = prepared_weekly_tbl)


refit_weekly_tbl %>%
  modeltime_forecast(new_data = forecast_weekly_tbl,
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast(
    .title = "Weekly Sales Forecast",
    .conf_interval_show = FALSE)



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




# SAVING ----

model_fit_best_prophet <- calibration_weekly_tbl %>%
  slice(2) %>%
  pluck(".model", 1)


write_rds(model_fit_best_prophet, "00_models/model_fit_best_prophet.rds")





