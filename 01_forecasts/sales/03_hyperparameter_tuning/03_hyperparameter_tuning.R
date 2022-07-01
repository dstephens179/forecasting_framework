# GOAL: Tune Models that Generalize Better

# OBJECTIVES ----
# - Sequential (e.g. ARIMA, ETS, TBATS) vs Non-Sequential Algorithms (e.g. Prophet, ML)
# - Cross-Validation Workflow - K-FOLD vs TSCV
# - Hyperparameter tuning with Prophet Boost
# - Parallel Processing with doFuture for 3X-5X Speedup


# LIBRARIES & SETUP ----

# Time Series ML
library(tidymodels)
library(modeltime)
library(rules)

# Timing & Parallel Processing
library(tictoc)
library(future)
library(doFuture)
library(parallel)

# Core 
library(plotly)
library(tidyverse)
library(lubridate)
library(timetk)


# DATA ----

weekly_artifacts_list <- read_rds("00_models/weekly_artifacts_list.rds")

prepared_weekly_tbl <- weekly_artifacts_list$data$prepared_weekly_tbl


# MODELS ----

source("00_scripts/01_calibrate_and_plot.R")

calibration_ml_tbl      <- read_rds("00_models/machine_learning_calibration_tbl.R")
calibration_boosted_tbl <- read_rds("00_models/calibration_tbl_boosted_models.rds")
calibration_ets_tbl     <- read_rds("00_models/calibration_tbl_ets_tbats.rds")


# TRAIN / TEST SPLITS ----

weekly_splits <- time_series_split(data = prepared_weekly_tbl, 
                                   date_var = date, 
                                   assess = "52 weeks", 
                                   cumulative = TRUE)

weekly_splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(.date_var = date, .value = sales)






# 1.0 REVIEW  ----

# * Combine Tables ----

model_tbl <- combine_modeltime_tables(
  calibration_ml_tbl,
  calibration_boosted_tbl,
  calibration_ets_tbl
)



# * Review Accuracy ----

calibration_tbl <- model_tbl %>% modeltime_calibrate(testing(weekly_splits))

calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(defaultPageSize = 30, bordered = TRUE, resizable = TRUE)



# [SEQUENTIAL MODEL] ----
# 2.0 NNETAR  ----
# - Sequential Model Definition: 
#   - Creates Lags internally
#   - [IMPORTANT DIFFERENTIATION]: Predicts next H observations
#   - All data must be sequential
#   - Cannot use K-Fold Cross Validation / Must use Time Series Cross Validation
# - Examples of Sequential Models:
#   - ARIMA
#   - Exponential Smoothing
#   - NNETAR
#   - Any algorithm from the forecast package


# * Extract Fitted Model ----

wflw_fit_nnetar <- calibration_tbl %>%
  pluck_modeltime_model(.model_id = 17)



# * Cross Validation Plan (TSCV) -----
# - Time Series Cross Validation

resamples_tscv_lag <- time_series_cv(
  date_var = date,
  data = training(weekly_splits) %>% drop_na(), 
  cumulative = TRUE, 
  initial = "1 years",
  assess = "52 weeks", 
  skip = "12 weeks",
  slice_limit = 6
)


resamples_tscv_lag %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(.date_var = date, .value = sales)



# * Recipe ----

recipe_spec_3_lag_date <- wflw_fit_nnetar %>%
  extract_preprocessor() %>%
  step_naomit(starts_with("lag"))


recipe_spec_3_lag_date %>%
  prep() %>%
  juice() %>%
  glimpse()



# * Model Spec ----

wflw_fit_nnetar %>% extract_spec_parsnip() 

model_spec_nnetar <- nnetar_reg(
  seasonal_period = 13,
  non_seasonal_ar = tune(id = "non_seasonal_ar"),
  seasonal_ar     = tune(),
  hidden_units    = tune(),
  num_networks    = 10,
  penalty         = tune(),
  epochs          = 500
) %>%
  set_engine("nnetar")



# Grid Spec - Round 1 ----

set.seed(123)
grid_spec_nnetar_1 <- grid_latin_hypercube(
  hardhat::extract_parameter_set_dials(model_spec_nnetar),
  size = 15
)


# * Tune ----
# - Expensive Operation
# - Parallel Processing is essential

wflw_tune_nnetar <- wflw_fit_nnetar %>%
  update_recipe(recipe_spec_3_lag_date) %>%
  update_model(model_spec_nnetar)



# ** Setup Parallel Processing ----

registerDoFuture()

n_cores <- detectCores()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)



# ** TSCV Cross Validation ----

set.seed(123)
tune_results_nnetar_1 <- wflw_tune_nnetar %>%
  tune_grid(
    resamples = resamples_tscv_lag,
    grid      = grid_spec_nnetar_1,
    metrics   = default_forecast_accuracy_metric_set(),
    control   = control_grid(verbose = TRUE, save_pred = TRUE)
  )



# ** Reset Sequential Plan ----

plan(strategy = sequential)



# Show Results

tune_results_nnetar_1
tune_results_nnetar_1 %>% show_best(metric = "rmse", n = Inf)



# Visualize Results

g1 <- tune_results_nnetar_1 %>% 
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(g1)




# Grid Spec - Round 2 ----
 

set.seed(123)
grid_spec_nnetar_2 <- grid_latin_hypercube(
  non_seasonal_ar(range = c(1, 5)),
  seasonal_ar(range = c(1, 2)),
  hidden_units(range = c(1, 3)),
  penalty(range = c(-2.0, -1.0), trans = scales::log10_trans()),
  size = 15
)



# ** Setup Parallel Processing ----

registerDoFuture()

n_cores <- detectCores()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)



# ** TSCV Cross Validation ----

# Round 2

set.seed(123)
tune_results_nnetar_2 <- wflw_tune_nnetar %>%
  tune_grid(
    resamples = resamples_tscv_lag,
    grid      = grid_spec_nnetar_2,
    metrics   = default_forecast_accuracy_metric_set(),
    control   = control_grid(verbose = TRUE, save_pred = TRUE)
  )


# ** Reset Sequential Plan ----
plan(strategy = sequential)



# Show Results
tune_results_nnetar_2 %>% show_best(metric = "rmse", n = Inf)



# Visualize Results
g2 <- tune_results_nnetar_2 %>% 
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(g2)




# Retrain & Assess -----

set.seed(123)
wflw_fit_nnetar_tscv <- wflw_tune_nnetar %>%
  finalize_workflow(
    tune_results_nnetar_2 %>%
      show_best(metric = "rmse", n = Inf) %>%
      dplyr::slice(1)
  ) %>%
  fit(training(weekly_splits))


calibrate_and_plot(
  wflw_fit_nnetar_tscv
)






# [NON-SEQUENTIAL] ----
# 3.0 PROPHET BOOST   -----
# - Non-Sequential Model:
#   - Uses date features
#   - Lags Created * Externally * (We Provide)
#   - Spline can be modeled with random missing observations
#   - Therefore can be Tuned using K-Fold Cross Validation
#   - IMPORTANT: Experiment to see which gives better results
# - Other Examples:
#   - Machine Learning Algorithms that use Calendar Features (e.g. GLMNet, XGBoost)
#   - Prophet
# - IMPORTANT: Can use time_series_cv() or vfold_cv(). Usually better performance with vfold_cv().


# * Extract Fitted Model ----

wflw_fit_prophet_boost <- calibration_tbl %>%
  pluck_modeltime_model(.model_id = 19)


# * Cross Validation Plans (K-Fold) ----
# - Prophet is a non-sequential model. We can randomize time observations.
# - K-Fold is OK
# - Should set.seed() because random process

set.seed(123)
resamples_kfold <- vfold_cv(training(weekly_splits), v = 10)


resamples_kfold %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(.date_var = date, .value = sales, .facet_ncol = 2)



# * Recipe Spec ----

wflw_fit_prophet_boost %>%
  extract_preprocessor() %>%
  prep() %>%
  juice() %>%
  glimpse()




# * Model Spec ----

wflw_fit_prophet_boost %>% 
  extract_spec_parsnip()


model_spec_prophet_boost <- prophet_boost(
  # prophet params
  changepoint_num    = 25,
  changepoint_range  = 0.8,
  seasonality_yearly = TRUE,
  seasonality_weekly = FALSE,
  seasonality_daily  = FALSE,
  
  # xgboost params
  mtry           = tune(),
  trees          = 10,
  min_n          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune()
) %>%
  set_engine("prophet_xgboost")




# Grid Spec Round 1 ----

set.seed(123)
grid_spec_boost_1 <- grid_latin_hypercube(
  hardhat::extract_parameter_set_dials(model_spec_prophet_boost) %>%
    update(
      mtry = mtry(range = c(1, 106)) # mtry showed error, so updated to max 106, as that's the number of columns.
    ),
  size = 15
)



# * Tune ----

# * Setup Parallel Processing ----
registerDoFuture()

n_cores <- parallel::detectCores()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)



# * K-Fold Cross Validation ----

tic()
set.seed(123)
tune_results_prophet_kfold <- wflw_fit_prophet_boost %>%
  update_model(model_spec_prophet_boost) %>%
  
  tune_grid(
    resamples = resamples_kfold,
    grid      = grid_spec_boost_1,
    metrics   = default_forecast_accuracy_metric_set(),
    control   = control_grid(verbose = FALSE, save_pred = TRUE)
  )
toc()



# * Reset Sequential Plan ----
plan(strategy = sequential)


# Show Best
tune_results_prophet_kfold %>% show_best(metric = "rmse", n = Inf)


# Visualize
g3 <- tune_results_prophet_kfold %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(g3)






# Grid Spec Round 2 ----

set.seed(123)
grid_spec_boost_2 <- grid_latin_hypercube(
  mtry(range = c(96, 106)),
  min_n(range = c(4, 10)),
  tree_depth(range = c(2, 7)),
  learn_rate(range = c(-2.2, -1.5)),
  loss_reduction(range = c(-6.9, -4.1)),
  size = 15
)



# * Tune ----

# * Setup Parallel Processing ----
registerDoFuture()

n_cores <- parallel::detectCores()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)



# * K-Fold Cross Validation ----

tic()
set.seed(123)
tune_results_prophet_kfold <- wflw_fit_prophet_boost %>%
  update_model(model_spec_prophet_boost) %>%
  
  tune_grid(
    resamples = resamples_kfold,
    grid      = grid_spec_boost_2,
    metrics   = default_forecast_accuracy_metric_set(),
    control   = control_grid(verbose = FALSE, save_pred = TRUE)
  )
toc()



# * Reset Sequential Plan ----
plan(strategy = sequential)



# Show Best
tune_results_prophet_kfold %>% show_best(metric = "rmse", n = Inf)



# Visualize
g4 <- tune_results_prophet_kfold %>%
  autoplot() +
  geom_smooth(se = FALSE)


ggplotly(g4)






# Grid Spec Round 3 ----

set.seed(123)
grid_spec_boost_3 <- grid_latin_hypercube(
  mtry(range = c(96, 100)),
  min_n(range = c(8, 10)),
  tree_depth(range = c(2, 4)),
  learn_rate(range = c(-1.9, -1.5)),
  loss_reduction(range = c(-6.6, -5.8)),
  size = 15
)



# * Tune ----

# * Setup Parallel Processing ----
registerDoFuture()

n_cores <- parallel::detectCores()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)



# * K-Fold Cross Validation ----

tic()
set.seed(123)
tune_results_prophet_kfold <- wflw_fit_prophet_boost %>%
  update_model(model_spec_prophet_boost) %>%
  
  tune_grid(
    resamples = resamples_kfold,
    grid      = grid_spec_boost_3,
    metrics   = default_forecast_accuracy_metric_set(),
    control   = control_grid(verbose = FALSE, save_pred = TRUE)
  )
toc()



# * Reset Sequential Plan ----
plan(strategy = sequential)



# Show Best
tune_results_prophet_kfold %>% show_best(metric = "rmse", n = Inf)



# Visualize
g5 <- tune_results_prophet_kfold %>%
  autoplot() +
  geom_smooth(se = FALSE)


ggplotly(g5)




# * Retrain & Assess ----

set.seed(123)
wflw_fit_prophet_boost_kfold <- wflw_fit_prophet_boost %>%
  update_model(model_spec_prophet_boost) %>%
  finalize_workflow(
    tune_results_prophet_kfold %>%
      show_best(metric = "rmse") %>%
      dplyr::slice(1)
  ) %>%
  fit(training(weekly_splits))


calibrate_and_plot(
  wflw_fit_prophet_boost_kfold
)








# 4.0 KNN SPLINE -----

# * Extract Fitted Model ----

wflw_fit_knn_spline <- calibration_tbl %>%
  pluck_modeltime_model(.model_id = 9)


# * Cross Validation Plans (K-Fold) ----

set.seed(123)
resamples_kfold <- vfold_cv(training(weekly_splits), v = 10)


# View KFold
resamples_kfold %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(.date_var = date, .value = sales, .facet_ncol = 2)



# * Recipe Spec ----

wflw_fit_knn_spline %>%
  extract_preprocessor() %>%
  prep() %>%
  juice() %>%
  glimpse()




# * Model Spec ----

wflw_fit_knn_spline %>% 
  extract_spec_parsnip()


model_spec_knn <- nearest_neighbor(
  neighbors  = tune(),
  dist_power = tune()
) %>%
  set_engine("kknn")




# Grid Spec Round 1 ----

set.seed(123)
grid_spec_knn_1 <- grid_latin_hypercube(
  hardhat::extract_parameter_set_dials(model_spec_knn),
  size = 15
)



# * Tune ----

# * Setup Parallel Processing ----
registerDoFuture()

n_cores <- parallel::detectCores()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)



# * K-Fold Cross Validation ----

tic()
set.seed(123)
tune_results_knn <- wflw_fit_knn_spline %>%
  update_model(model_spec_knn) %>%
  
  tune_grid(
    resamples = resamples_kfold,
    grid      = grid_spec_knn_1,
    metrics   = default_forecast_accuracy_metric_set(),
    control   = control_grid(verbose = FALSE, save_pred = TRUE)
  )
toc()



# * Reset Sequential Plan ----
plan(strategy = sequential)


# Show Best
tune_results_knn %>% show_best(metric = "rmse", n = Inf)


# Visualize
g6 <- tune_results_knn %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(g6)






# Grid Spec Round 2 ----

set.seed(123)
grid_spec_knn_2 <- grid_latin_hypercube(
  neighbors(range = c(1, 1)),
  dist_power(range = c(2, 2)),
  size = 15
)



# * Tune ----

# * Setup Parallel Processing ----
registerDoFuture()

n_cores <- parallel::detectCores()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)



# * K-Fold Cross Validation ----

tic()
set.seed(123)
tune_results_knn <- wflw_fit_knn_spline %>%
  update_model(model_spec_knn) %>%
  
  tune_grid(
    resamples = resamples_kfold,
    grid      = grid_spec_knn_2,
    metrics   = default_forecast_accuracy_metric_set(),
    control   = control_grid(verbose = FALSE, save_pred = TRUE)
  )
toc()



# * Reset Sequential Plan ----
plan(strategy = sequential)



# Show Best
tune_results_knn %>% show_best(metric = "rmse", n = Inf)



# Visualize
g7 <- tune_results_knn %>%
  autoplot() +
  geom_smooth(se = FALSE)


ggplotly(g7)







# * Retrain & Assess ----

set.seed(123)
wflw_fit_knn_kfold <- wflw_fit_knn_spline %>%
  update_model(model_spec_knn) %>%
  finalize_workflow(
    tune_results_knn %>%
      show_best(metric = "rmse") %>%
      dplyr::slice(1)
  ) %>%
  fit(training(weekly_splits))


calibrate_and_plot(
  wflw_fit_knn_kfold
)





# 5.0 SAVE ARTIFACTS ----

set.seed(123)
calibration_tbl <- modeltime_table(
  wflw_fit_nnetar_tscv,
  wflw_fit_prophet_boost_kfold,
  wflw_fit_knn_kfold
) %>%
  modeltime_calibrate(testing(weekly_splits))


calibration_tbl %>%
  write_rds("00_models/calibration_tbl_hyperparameter_tuning.rds")


calibration_tbl %>%
  modeltime_accuracy()




