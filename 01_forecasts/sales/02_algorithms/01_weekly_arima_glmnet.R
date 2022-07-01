# LIBRARIES ----

# Time Series ML
library(tidymodels)
library(modeltime)

# Core
library(tidyverse)
library(timetk)
library(lubridate)
library(glmnet)



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



# 1.0 ARIMA & GLMNET ----

# * Parsnip Model (ARIMA + XREGS)

model_fit_weekly_arima <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(
    sales ~ date
    + fourier_vec(date, period = 4)
    + fourier_vec(date, period = 13)
    + fourier_vec(date, period = 26)
    + fourier_vec(date, period = 52)
    + month(date, label = TRUE)
    + event,
    data = training(weekly_splits)
  )


model_fit_weekly_arima



# * Workflow (ARIMA + Date Features)

model_spec_weekly_arima <- arima_reg() %>%
  set_engine("auto_arima")


recipe_spec_weekly_fourier <- recipe(sales ~ date, data = training(weekly_splits)) %>%
  step_fourier(date, period = c(4, 26, 52), K = 1)


recipe_spec_weekly_fourier %>%  # note: date must be included in fourier, as it normally isn't.
  prep() %>%
  juice() %>%
  glimpse()


workflow_fit_weekly_arima <- workflow() %>%
  add_model(model_spec_weekly_arima) %>%
  add_recipe(recipe_spec_weekly_fourier) %>%
  fit(training(weekly_splits))



# * Workflow (GLMNET + XREGS)

recipe_weekly_1_spline %>%
  prep() %>%
  juice() %>%
  glimpse()


model_spec_weekly_glmnet <- linear_reg(
  penalty = 0.1,
  mixture = 0.5
) %>%
  set_engine("glmnet")


workflow_fit_weekly_glmnet <- workflow() %>%
  add_model(model_spec_weekly_glmnet) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))






# 2.0 MODELTIME TABLE ----
# Organize

model_weekly_tbl <- modeltime_table(
  model_fit_weekly_arima,
  workflow_fit_weekly_arima,
  workflow_fit_weekly_glmnet
) %>%
  
  update_model_description(1, "arima + xregs") %>%
  update_model_description(2, "arima recipe") %>%
  update_model_description(3, "glmnet")

model_weekly_tbl





# 3.0 CALIBRATION ----
# - Calculates residual model errors on test set
# - Gives us a true prediction error estimate when we model with confidence intervals

calibration_weekly_tbl <- model_weekly_tbl %>%
  modeltime_calibrate(new_data = testing(weekly_splits))


calibration_weekly_tbl %>%
  slice(1) %>%
  unnest(.calibration_data)





# 4.0 TEST ACCURACY ----


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




# 5.0 TEST FORECAST ----
# - Visualize the out-of-sample forecast

calibration_weekly_tbl %>%
  modeltime_forecast(
    new_data = testing(weekly_splits), 
    actual_data = prepared_weekly_tbl,
    conf_interval = 0.80
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25,
    .conf_interval_show = TRUE,
    .conf_interval_alpha = 0.1, 
    .conf_interval_fill = "lightblue",
    .title = "Weekly Sales Forecast"
  )





# 6.0 REFIT AND FORECAST ----

# * Refit

# before the model was fit to training(weekly_splits - 80%), now we refit to the entire actuals dataset
refit_weekly_tbl <- calibration_weekly_tbl %>%
  modeltime_refit(data = prepared_weekly_tbl)





# * Final Forecast

refit_weekly_tbl %>%
  modeltime_forecast(
    new_data = forecast_weekly_tbl, 
    actual_data = prepared_weekly_tbl
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25,
    .conf_interval_show = FALSE,
    .title = "Weekly Sales Forecast"
  )


# Invert

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


# Arima w/Fourier
model_fit_best_arima <- calibration_weekly_tbl %>%
  slice(1) %>%
  pluck(".model", 1)


write_rds(model_fit_best_arima, "00_models/model_fit_best_arima.rds")



# GLMNet
model_fit_best_glmnet <- calibration_weekly_tbl %>%
  slice(3) %>%
  pluck(".model", 1)


write_rds(model_fit_best_glmnet, "00_models/model_fit_best_glmnet.rds")



