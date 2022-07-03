# OBJECTIVE ----
# * forecast by store, owner, metal type, product type

# External Packages ----
# remotes::install_github("rstudio/rmarkdown")
# remotes::install_github("rstudio/crosstalk")
# remotes::install_github("curso-r/treesnip")


# LIBRARIES ----

# Machine Learning
library(lightgbm)
library(xgboost)
library(tictoc)


# Tidymodels
library(tidymodels)
library(treesnip)
library(modeltime)
library(modeltime.ensemble)



# Timing & Parallel Processing
library(future)
library(doFuture)


# Core

library(tidyverse)
library(skimr)
library(timetk)
library(bigrquery)
library(data.table)



# * Parallel Processing ----

registerDoFuture()
n_cores <- parallel::detectCores()
plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores))




# 1.0  DATA ----

# * Read Data ----

calendar_tbl <- read_csv("00_data/calendar.csv")
calendar_tbl


projectid = "source-data-314320"
sql <- "SELECT 
          tienda,
          may_men,
          date,
          owner,
          metal_type,
          product_type,
          sales
        FROM `source-data-314320.Store_Data.All_Data`
        WHERE date >= '2020-08-01'
        AND sales <> 0
        ORDER BY Date desc
"


# Run the query and store
bq_query <- bq_project_query(projectid, sql)
data_tbl <- bq_table_download(bq_query)




# Prepare original dataset

data_tbl$tienda <- str_replace_all(data_tbl$tienda, "En Linea", "En.Linea")

data_tbl$combined <- str_c(data_tbl$product_type)





# * Transform & Reshape ----

data_long_tbl <- data_tbl %>%
  
  # remove any previously forecasted rows
  filter(!is.na(sales)) %>%
  
  # # summarize by time
  group_by(combined) %>%
  summarize_by_time(.date_var = date, .by = "day", value = sum(sales)) %>%
  
  # filter out "combined" time series with less than 2 instances
  filter(n() >= 2) %>%
  ungroup() %>%
  
  mutate(sales = value) %>%
  select(-value)



data_long_tbl %>% skim()



# * Split into Centro ----
data_long_centro_tbl <- data_long_tbl
  # filter(grepl("Centro", combined))






# PREPARE FULL DATA ----

full_data_centro_tbl <- data_long_centro_tbl %>%
  
  # fill in 0's so all combined lines for every day is filled
  group_by(combined) %>%
  pad_by_time(
    .date_var   = date,
    .by         = "day",
    .pad_value  = 0) %>%
  
  # global features / transformations
  mutate(sales = ifelse(sales <= 0, 0, sales)) %>%
  mutate(sales = log1p(sales)) %>%
  
  # extend by group
  future_frame(.date_var = date, .length_out = 28, .bind_data = TRUE) %>%
  ungroup() %>%
  
  # lags, fourier and rolling features
  mutate(combined = as.factor(combined)) %>%
  group_by(combined) %>%
  group_split() %>%          # this will follow the same order as it was originally, not all over the place
  map(.f = function(df) {
    df %>%
      arrange(date) %>%
      tk_augment_fourier(date, .periods = c(14, 28)) %>%
      tk_augment_lags(sales, .lags = 28) %>%
      tk_augment_slidify(
        sales_lag28,
        .f       = ~ mean(.x, na.rm = TRUE),
        .period  = c(7, 28, 28*2),
        .partial = TRUE,
        .align   = "center"
      )
  }) %>%
  bind_rows() %>%
  
  # add row id
  rowid_to_column(var = "row_id")




full_data_centro_tbl %>% skim()







# * Visualize ----

item_id_sample <- c("Cadena", 
                    "Argolla",
                    "Anillo",
                    "Esclava",
                    "Pulsera",
                    "Arete")


full_data_centro_tbl %>%
  filter(combined %in% item_id_sample) %>%
  group_by(combined) %>%
  plot_time_series(
    date, sales, 
    .smooth        = FALSE,
    .facet_ncol    = 2
  )



# * Data Prepared ----
data_prepared_tbl <- full_data_centro_tbl %>%
  filter(!is.na(sales)) %>%
  drop_na()

data_prepared_tbl



# * Future Data ----

future_tbl <- full_data_centro_tbl %>%
  filter(is.na(sales)) %>%
  filter(date > max(data_long_tbl$date))


# filter for missing values... which will be a problem.
future_tbl %>% filter(is.na(sales_lag28))


# first convert NaN to NA, then take care of NA's
future_tbl <- future_tbl %>%
  mutate(across(.cols = contains("_lag"),
                .fns  = function(x) ifelse(is.nan(x), NA, x))
  ) %>%
  mutate(across(.cols = contains("_lag"),
                .fns  = ~ replace_na(.x, 0)))


# check for NA's. Should be empty.
future_tbl %>% filter(is.na(sales_lag28_roll_28))





# 2.0 TIME SPLIT ----

splits <- data_prepared_tbl %>%
  time_series_split(
    date_var   = date,
    assess     = 28,
    cumulative = TRUE
  )


splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(.date_var = date, .value = sales)





# 3.0 RECIPE ----

# * Clean Training Set ----
# - With Panel Data, need to do this outside of a recipe
# - Transformation happens by group


train_cleaned <- training(splits) %>%
  group_by(combined) %>%
  mutate(sales = ts_clean_vec(sales, period = 7)) %>%
  ungroup()


train_cleaned %>%
  group_by(combined) %>%
  plot_time_series(
    date, sales, 
    .facet_ncol  = 4, 
    .smooth      = FALSE, 
    .interactive = FALSE
  )


# * Recipe Specification ----

train_cleaned

recipe_spec <- recipe(sales ~ ., data = train_cleaned) %>%
  update_role(row_id, new_role = "indicator") %>%
  step_timeseries_signature(date) %>%
  step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(am.pm)")) %>%
  step_normalize(date_index.num, date_year) %>%
  step_other(combined) %>%
  step_dummy(all_nominal(), one_hot = TRUE)


recipe_spec %>% summary()

recipe_spec %>% prep() %>% summary()

recipe_spec %>% prep() %>% juice() %>% glimpse()


# create ML recipe so not rewriting "step_rm(date)" all the time
recipe_spec_ml <- recipe_spec %>% step_rm(date)




# 4.0 MODELS ----
# - REMINDER: Cannot use sequential models

# * PROPHET ----
wflw_fit_prophet <- workflow() %>%
  add_model(spec = prophet_reg() %>% set_engine("prophet")) %>%
  add_recipe(recipe_spec) %>%
  fit(train_cleaned)



# * PROPHET BOOST ----
wflw_fit_prophet_boost <- workflow() %>%
  add_model(
    spec = prophet_boost(
      seasonality_daily = FALSE,
      seasonality_weekly = FALSE,
      seasonality_yearly = FALSE
    ) %>%
      set_engine("prophet_xgboost")
  ) %>%
  add_recipe(recipe_spec) %>%
  fit(train_cleaned)




# * XGBOOST ----
wflw_fit_xgboost <- workflow() %>%
  add_model(spec = boost_tree(mode = "regression") %>% set_engine("xgboost")) %>%
  add_recipe(recipe_spec_ml) %>%
  fit(train_cleaned)




# * RANDOM FOREST ----
wflw_fit_rf <- workflow() %>%
  add_model(
    spec = rand_forest(mode = "regression") %>% set_engine("ranger")
  ) %>%
  add_recipe(recipe_spec_ml) %>%
  fit(train_cleaned)




# * NNET ----
wflw_fit_nnet <- workflow() %>%
  add_model(
    spec = mlp(mode = "regression") %>% set_engine("nnet")
  ) %>%
  add_recipe(recipe_spec_ml) %>%
  fit(train_cleaned)




# * MARS ----
wflw_fit_mars <- workflow() %>%
  add_model(
    spec = mars(mode = "regression") %>% set_engine("earth")
  ) %>%
  add_recipe(recipe_spec_ml) %>%
  fit(train_cleaned)







# ACCURACY CHECK ----

# Modeltime table
submodels_1_tbl <- modeltime_table(
  wflw_fit_prophet,
  wflw_fit_prophet_boost,
  wflw_fit_xgboost,
  wflw_fit_rf,
  wflw_fit_nnet,
  wflw_fit_mars
) 


submodels_1_tbl %>%
  modeltime_accuracy(new_data = testing(splits)) %>%
  arrange(rmse)






# 5.0 HYPER PARAMETER TUNING ---- 

# * RESAMPLES - K-FOLD ----- 

set.seed(123)
resamples_kfold <- train_cleaned %>% vfold_cv(v = 5)

resamples_kfold %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(
    .date_var = date, 
    .value = sales, 
    .facet_ncol = 2
  )




# * XGBOOST TUNE ----

# ** Tunable Specification

model_spec_xgboost_tune <- boost_tree(
  mode           = "regression",
  mtry           = tune(),
  trees          = tune(),
  min_n          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune()
) %>%
  set_engine("xgboost")



wflw_spec_xgboost_tune <- workflow() %>%
  add_model(model_spec_xgboost_tune) %>%
  add_recipe(recipe_spec %>% update_role(date, new_role = "indicator"))




# ** Tuning

tic()
set.seed(123)
tune_results_xgboost <- wflw_spec_xgboost_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    param_info = parameters(wflw_spec_xgboost_tune) %>%
      update(
        learn_rate = learn_rate(range = c(0.001, 0.400), trans = NULL)
      ),
    grid = 10,
    control = control_grid(verbose = TRUE, allow_par = TRUE)
  )
toc()


# ** Results

tune_results_xgboost %>% show_best("rmse", n = Inf)



# ** Finalize

wflw_fit_xgboost_tuned <- wflw_spec_xgboost_tune %>%
  finalize_workflow(select_best(tune_results_xgboost, "rmse")) %>%
  fit(train_cleaned)




# * RANGER TUNE ----

# ** Tunable Specification

model_spec_rf_tune <- rand_forest(
  mode  = "regression",
  mtry  = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger")


wflw_spec_rf_tune <- workflow() %>%
  add_model(model_spec_rf_tune) %>%
  add_recipe(recipe_spec %>% update_role(date, new_role = "indicator"))


# ** Tuning

tic()
set.seed(123)
tune_results_rf <- wflw_spec_rf_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid      = 5,
    control   = control_grid(verbose = TRUE, allow_par = TRUE)
  )
toc()



# ** Results

tune_results_rf %>% show_best("rmse", n = Inf)



# ** Finalize

wflw_fit_rf_tuned <- wflw_spec_rf_tune %>%
  finalize_workflow(select_best(tune_results_rf, "rmse")) %>%
  fit(train_cleaned)






# * EARTH TUNE ----

# ** Tunable Specification

model_spec_earth_tune <- mars(
  mode        = "regression",
  num_terms   = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")


wflw_spec_earth_tune <- workflow() %>%
  add_model(model_spec_earth_tune) %>%
  add_recipe(recipe_spec %>% update_role(date, new_role = "indicator"))



# ** Tuning

tic()
set.seed(123)
tune_results_earth <- wflw_spec_earth_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid      = 10,
    control   = control_grid(verbose = TRUE, allow_par = TRUE)
  )
toc()



# ** Results

tune_results_earth %>% show_best("rmse")


# ** Finalize
wflw_fit_earth_tuned <- wflw_spec_earth_tune %>%
  finalize_workflow(tune_results_earth %>% select_best("rmse")) %>%
  fit(train_cleaned)





# 6.0 EVALUATE PANEL FORECASTS  -----

# * Model Table ----

submodels_2_tbl <- modeltime_table(
  wflw_fit_xgboost_tuned,
  wflw_fit_rf_tuned,
  wflw_fit_earth_tuned
) %>%
  update_model_description(1, "xgboost - tuned") %>%
  update_model_description(2, "ranger - tuned") %>%
  update_model_description(3, "earth - tuned") %>%
  combine_modeltime_tables(submodels_1_tbl)



# * Calibration ----
calibration_tbl <- submodels_2_tbl %>%
  modeltime_calibrate(testing(splits))



# * Accuracy ----
calibration_tbl %>%
  modeltime_accuracy() %>%
  arrange(rmse)



# * Forecast Test ----

calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data   = TRUE
  ) %>%
  group_by(combined) %>%
  plot_modeltime_forecast(
    .conf_interval_show = FALSE,
    .trelliscope        = TRUE
  )




# 7.0 RESAMPLING ----
# - Assess the stability of our models over time
# - Helps us strategize an ensemble approach

# * Time Series CV ----

resamples_tscv <- train_cleaned %>%
  time_series_cv(
    date_var    = date,
    assess      = 28,
    skip        = 28,
    cumulative  = TRUE,
    slice_limit = 4
  )


resamples_tscv %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(
    .date_var = date,
    .value    = sales
  )



# * Fitting Resamples ----

model_tbl_tuned_resamples <- submodels_2_tbl %>%
  modeltime_fit_resamples(
    resamples = resamples_tscv,
    control   = control_resamples(verbose = TRUE, allow_par = TRUE)
  )



# * Resampling Accuracy Table ----

model_tbl_tuned_resamples %>%
  modeltime_resample_accuracy(
    metric_set  = metric_set(rmse, rsq),
    summary_fns = list(mean = mean, sd = sd)
  )



# * Resampling Accuracy Plot ----

model_tbl_tuned_resamples %>%
  plot_modeltime_resamples(
    .metric_set = metric_set(mae, rmse, rsq),
    .point_size = 4, 
    .point_alpha = 0.8, 
    .facet_ncol = 1
  )



# 8.0 ENSEMBLE PANEL MODELS ----

# * Average Ensemble ----

submodels_2_ids_to_keep <- c(1, 2, 3, 4)


ensemble_fit <- submodels_2_tbl %>%
  filter(.model_id %in% submodels_2_ids_to_keep) %>%
  ensemble_average(type = "median")


model_ensemble_tbl <- modeltime_table(
  ensemble_fit
)


# * Accuracy ----

model_ensemble_tbl %>%
  modeltime_accuracy(testing(splits))


# Forecast ----

forecast_ensemble_test_tbl <- model_ensemble_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data   = TRUE
  ) %>%
  mutate(
    across(.cols = c(.value, sales), .fns = expm1)
  )


forecast_ensemble_test_tbl %>%
  group_by(combined) %>%
  filter(combined %in% item_id_sample) %>%
  filter_by_time(
    .date_var = date, 
    .start_date = last(date) %-time% "6 month", 
    .end_date = "end"
  ) %>%
  plot_modeltime_forecast(
    .facet_ncol = 4,
    .conf_interval_show = FALSE
  )


forecast_ensemble_test_tbl %>%
  filter(.key == "prediction") %>%
  select(combined, .value, sales) %>%
  # group_by(combined) %>%
  summarize_accuracy_metrics(
    truth      = sales,
    estimate   = .value,
    metric_set = metric_set(mae, rmse, rsq)
  )


# Refit ----

data_prepared_cleaned_tbl <- data_prepared_tbl %>%
  group_by(combined) %>%
  mutate(sales = ts_clean_vec(sales, period = 7)) %>%
  ungroup()



model_ensemble_refit_tbl <- model_ensemble_tbl %>%
  modeltime_refit(data = data_prepared_cleaned_tbl)



model_ensemble_refit_tbl %>%
  modeltime_forecast(
    new_data    = future_tbl,
    actual_data = data_prepared_tbl,
    keep_data   = TRUE
  ) %>%
  mutate(
    .value = expm1(.value),
    sales  = expm1(sales)
  ) %>%
  group_by(combined) %>%
  plot_modeltime_forecast(
    .y_intercept  = 0,
    .trelliscope  = TRUE
  )














# * Turn OFF Parallel Backend
plan(sequential)



