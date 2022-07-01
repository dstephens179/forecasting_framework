# LIBRARIES & SETUP ----

# Time Series ML
library(tidymodels)
library(modeltime)
library(rules)

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





# PLOTTING UTILITY  ----
# - Calibrate & Plot

calibrate_and_plot <- function(..., type = "testing") {
  
  if (type == "testing") {
    new_data <- testing(weekly_splits)
  } else {
    new_data <- training(weekly_splits) %>% drop_na()
  }
  
  calibration_tbl <- modeltime_table(...) %>%
    modeltime_calibrate(new_data)
  
  print(calibration_tbl %>% modeltime_accuracy())
  
  calibration_tbl %>% modeltime_forecast(new_data = new_data,
                                         actual_data = prepared_weekly_tbl) %>%
    plot_modeltime_forecast(.conf_interval_show = FALSE)
}





# 1.0 ELASTIC NET REGRESSION ----
# - Strengths: Very good for trend
# - Weaknesses: Not as good for complex patterns (i.e. seasonality)

# Spline

model_spec_glmnet <- linear_reg(
  mode = "regression",
  penalty = 0.1,
  mixture = .5
) %>%
  set_engine("glmnet")


# Spline
wflw_fit_glmnet_spline <- workflow() %>%
  add_model(model_spec_glmnet) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))


# Lag
wflw_fit_glmnet_lag <- wflw_fit_glmnet_spline %>%
  update_recipe(recipe_weekly_2_lag) %>%
  fit(training(weekly_splits))


# Calibrate & Plot
calibrate_and_plot(
  wflw_fit_glmnet_spline, 
  wflw_fit_glmnet_lag,
  type = "testing"
)




# 2.0 MARS ----
# Multiple Adaptive Regression Splines
# - Strengths: Best algorithm for modeling trend
# - Weaknesses: 
#   - Not good for complex patterns (i.e. seasonality)
#   - Don't combine with splines! MARS makes splines.
# - Key Concept: Can combine with xgboost (better seasonality detection)
#   - prophet_reg: uses a technique similar to mars for modeling trend component
#   - prophet_boost: Uses prophet for trend, xgboost for features

model_spec_mars <- mars(
  mode = "regression",
  num_terms = 50, prod_degree = 2,
) %>%
  set_engine("earth", endspan = 20)


# Spline
wflw_fit_mars_spline <- workflow() %>%
  add_model(model_spec_mars) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))


# Lag
wflw_fit_mars_lag <- workflow() %>%
  add_model(model_spec_mars) %>%
  add_recipe(recipe_weekly_2_lag) %>%
  fit(training(weekly_splits))



# Calibrate & Plot
calibrate_and_plot(
  wflw_fit_mars_spline,
  wflw_fit_mars_lag
)




# 3.0 SVM POLY ----
# Strengths: Well-rounded algorithm
# Weaknesses: Needs tuned or can overfit

model_spec_svm_poly <- svm_poly(
  mode = "regression",
  cost = 1,
  scale_factor = 1,
  margin = 0.1
) %>%
  set_engine("kernlab")


# Spline
set.seed(123)
wflw_fit_svm_poly_spline <- workflow() %>%
  add_model(model_spec_svm_poly) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))


# Lag
set.seed(123)
wflw_fit_svm_poly_lag <- workflow() %>%
  add_model(model_spec_svm_poly) %>%
  add_recipe(recipe_weekly_2_lag) %>%
  fit(training(weekly_splits))


# Calibrate & Plot
calibrate_and_plot(
  wflw_fit_svm_poly_spline,
  wflw_fit_svm_poly_lag
)



# 4.0 SVM RADIAL BASIS ----
# Strengths: Well-rounded algorithm
# Weaknesses: Needs tuned or can overfit

model_spec_svm_rbf <- svm_rbf(
  mode = "regression",
  cost = 25, 
  margin = 0.2
) %>%
  set_engine("kernlab")



# Spline
set.seed(123)
wflw_fit_svm_rbf_spline <- workflow() %>%
  add_model(model_spec_svm_rbf) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))


# Lag
set.seed(123)
wflw_fit_svm_rbf_lag <- workflow() %>%
  add_model(model_spec_svm_rbf) %>%
  add_recipe(recipe_weekly_2_lag) %>%
  fit(training(weekly_splits))


# Calibrate & Plot
calibrate_and_plot(
  wflw_fit_svm_rbf_spline,
  wflw_fit_svm_rbf_lag
)




# 5.0 K-NEAREST NEIGHBORS ----
# - Strengths: Uses neighboring points to estimate 
# - Weaknesses: Cannot predict beyond the maximum/minimum target (e.g. increasing trend)
# - Solution: Model trend separately (if needed). 
#   - Can combine with ARIMA, Linear Regression, Mars, or Prophet

model_spec_knn <- nearest_neighbor(
  mode = "regression", 
  neighbors = 1,
  dist_power = 2
) %>%
  set_engine("kknn")


# Spline
set.seed(123)
wflw_fit_knn_spline <- workflow() %>%
  add_model(model_spec_knn) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))


# Lag
set.seed(123)
wflw_fit_knn_lag <- workflow() %>%
  add_model(model_spec_knn) %>%
  add_recipe(recipe_weekly_2_lag) %>%
  fit(training(weekly_splits))


# Calibrate & Plot
calibrate_and_plot(
  wflw_fit_knn_spline,
  wflw_fit_knn_lag
)




# 6.0 RANDOM FOREST ----
# - Strengths: Can model seasonality very well
# - Weaknesses: 
#   - Cannot predict beyond the maximum/minimum target (e.g. increasing trend)
# - Solution: Model trend separately (if needed). 
#   - Can combine with ARIMA, Linear Regression, Mars, or Prophet

model_spec_rf <- rand_forest(
  mode = "regression", 
  mtry = 50,
  trees = 50
) %>%
  set_engine("randomForest")


# Spline
set.seed(123)
wflw_fit_rf_spline <- workflow() %>%
  add_model(model_spec_rf) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))


# Lag
set.seed(123)
wflw_fit_rf_lag <- workflow() %>%
  add_model(model_spec_rf) %>%
  add_recipe(recipe_weekly_2_lag) %>%
  fit(training(weekly_splits))


# Calibrate & Plot
calibrate_and_plot(
  wflw_fit_rf_spline,
  wflw_fit_rf_lag
)




# 7.0 XGBOOST ----
# - Strengths: Best for seasonality & complex patterns
# - Weaknesses: 
#   - Cannot predict beyond the maximum/minimum target (e.g. increasing trend)
# - Solution: Model trend separately (if needed). 
#   - Can combine with ARIMA, Linear Regression, Mars, or Prophet
#   - prophet_boost & arima_boost: Do this

model_spec_xgboost <- boost_tree(
  mode = "regression", 
  mtry = 25, 
  trees = 100, 
  min_n = 2,
  tree_depth = 20,
  learn_rate = 0.3
) %>%
  set_engine("xgboost")


# Spline
set.seed(123)
wflw_fit_xgboost_spline <- workflow() %>%
  add_model(model_spec_xgboost) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))


# Lag
set.seed(123)
wflw_fit_xgboost_lag <- workflow() %>%
  add_model(model_spec_xgboost) %>%
  add_recipe(recipe_weekly_2_lag) %>%
  fit(training(weekly_splits))


calibrate_and_plot(
  wflw_fit_xgboost_spline,
  wflw_fit_xgboost_lag
)




# 8.0 NEURAL NET ----
# - Single Layer Multi-layer Perceptron Network
# - Simple network - Like linear regression
# - Can improve learning by adding more hidden units, epochs, etc


model_spec_nnet <- mlp(
  mode = "regression", 
  hidden_units = 2,
  penalty = 0.01,
  epochs = 5
) %>%
  set_engine("nnet")


# Spline
set.seed(123)
wflw_fit_nnet_spline <- workflow() %>%
  add_model(model_spec_nnet) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))


# Lag
set.seed(123)
wflw_fit_nnet_lag <- workflow() %>%
  add_model(model_spec_nnet) %>%
  add_recipe(recipe_weekly_2_lag) %>%
  fit(training(weekly_splits))


# Calibrate & Plot
calibrate_and_plot(
  wflw_fit_nnet_spline,
  wflw_fit_nnet_lag
)




# 9.0 NNETAR ----
# - NNET with Lagged Features (AR)
# - Is a sequential model (comes from the forecast package)
# - Must include date feature

recipe_weekly_base %>%
  prep() %>%
  juice() %>%
  glimpse()


model_spec_nnetar <- nnetar_reg(
  mode = "regression", 
  non_seasonal_ar = 2, 
  seasonal_ar     = 2,
  hidden_units    = 1,
  penalty         = 0.1, 
  num_networks    = 10, 
  epochs          = 500
) %>%
  set_engine("nnetar")


# Base Model
set.seed(123)
wflw_fit_nnetar_base <- workflow() %>%
  add_model(model_spec_nnetar) %>%
  add_recipe(recipe_weekly_base) %>%
  fit(training(weekly_splits) %>% drop_na())


# Calibrate & Plot
calibrate_and_plot(
  wflw_fit_nnetar_base
)



# 10.0 Modeltime Forecasting Workflow -----
# - Compare model performance

# * Modeltime Table ----

model_tbl <- modeltime_table(
  wflw_fit_glmnet_spline,
  wflw_fit_glmnet_lag,
  wflw_fit_mars_spline,
  wflw_fit_mars_lag,
  wflw_fit_svm_poly_spline,
  wflw_fit_svm_poly_lag,
  wflw_fit_svm_rbf_spline,
  wflw_fit_svm_rbf_lag,
  wflw_fit_knn_spline,
  wflw_fit_knn_lag,
  wflw_fit_rf_spline,
  wflw_fit_rf_lag,
  wflw_fit_xgboost_spline,
  wflw_fit_xgboost_lag,
  wflw_fit_nnet_spline,
  wflw_fit_nnet_lag,
  wflw_fit_nnetar_base
) %>%
  mutate(
    .model_desc_2 = str_c(.model_desc, rep_along(.model_desc, c(" - spline", " - lag")))
  ) %>%
  mutate(
    .model_desc = ifelse(.model_id == 17, .model_desc, .model_desc_2)
  ) %>%
  select(-.model_desc_2)



# * Calibration Table ----

calibration_tbl <- model_tbl %>%
  modeltime_calibrate(new_data = testing(weekly_splits))



# * Obtain Test Forecast Accuracy ----

calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy()



# * Visualize Test Forecast ----

forecast_test_tbl <- calibration_tbl %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl)
  
forecast_test_tbl %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)



# * Refit & Forecast Future ----

set.seed(123)
calibration_tbl %>%
  modeltime_refit(data = prepared_weekly_tbl)


forecast_future_tbl <- calibration_tbl %>%
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






# 11.0 SAVING ARTIFACTS ----

calibration_tbl %>%
  write_rds("00_models/machine_learning_calibration_tbl.R")

read_rds("00_models/machine_learning_calibration_tbl.R")


dump(c("calibrate_and_plot"), file = "00_scripts/01_calibrate_and_plot.R")

source("00_scripts/01_calibrate_and_plot.R")




