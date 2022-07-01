# LIBRARIES & DATA ----
# Time Series ML
library(tidymodels)
library(modeltime)

# Base
library(timetk)
library(tidyverse)
library(lubridate)
library(bigrquery)
library(skimr)
library(DataExplorer)



# Centro Data
projectid = "source-data-314320"
sql_Centro <- "SELECT *
        FROM `source-data-314320.Store_Data.All_Data`
        WHERE Owner = 'M&D'
          AND Tienda = 'Centro'
          AND product_type <> 'Aro'
          AND product_type <> 'Bisel'
          AND product_type <> 'Collar'
          AND product_type <> 'Varios'
          AND product_type <> 'Gargantilla'
          AND sales <> 0
        ORDER BY Date desc
"


All_Data_Centro <- bq_project_query(projectid, sql_Centro)
BQ_Table_Centro <- bq_table_download(All_Data_Centro)



# Gold Price Data
projectid = "source-data-314320"
sql <- "SELECT *
        FROM `source-data-314320.joyeria_dataset.gold_price`
        ORDER BY Date
"


gold_price <- bq_project_query(projectid, sql)
bq_gold_price <- bq_table_download(gold_price)



# Data Wrangle & Transform
centro_transformed_tbl <- BQ_Table_Centro %>%
  summarize_by_time(date, .by = "day", sales = sum(sales)) %>%
  pad_by_time(.by = "day", .pad_value = 0) %>%
  mutate(sales = ifelse(sales < 0, 0, sales)) %>%
  
  #preprocessing steps
  mutate(sales_trans = log_interval_vec(sales, limit_lower = 0, offset = 1)) %>%
  mutate(sales_trans = standardize_vec(sales_trans)) %>%
  
  #start after COVID impact
  filter_by_time(.start_date = "2020-05-17") %>%
  
  #clean
  mutate(sales_trans_cleaned = ts_clean_vec(x = sales_trans, period = 7)) %>%
  mutate(sales_trans = sales_trans_cleaned) %>%
  
  select(-sales_trans_cleaned, -sales)



# Save Key Params
c_limit_lower <- 0
c_limit_upper <- 181388.646
c_offset      <- 1
c_std_mean    <- -3.54883287171812
c_std_sd      <- 2.87129633834657



# 1.0 CREATE FULL DATA SET ----
# - Extend to Future Window
# - Add any lags to full dataset
# - Add any external regressors to full dataset


c_horizon    <- 8*7
c_lag_period <- 8*7
c_rolling_periods <- c(30, 60, 90)


centro_prepared_full_tbl <- centro_transformed_tbl %>%
  
  # add future window
  bind_rows(
    future_frame(.data = ., .date_var = date, .length_out = c_horizon)
  ) %>% 
  
  # add autocorrelated lags -- just extending everything by 56 days.
  tk_augment_lags(.value = sales_trans, .lags = c_lag_period) %>%
  
  # add the rolling features
  tk_augment_slidify(
    .value = sales_trans_lag56, 
    .f = mean,
    .period = c_rolling_periods,
    .align = "center", 
    .partial =  TRUE
    ) %>%
  
  # format columns
  rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .))


centro_prepared_full_tbl %>%
  pivot_longer(-date) %>%
  plot_time_series(date, value, .color_var = name, .smooth = FALSE)




# 2.0 STEP 2 - SEPARATE INTO MODELING & FORECAST DATA ----

# split this new tbl for modeling
centro_prepared_tbl <-  centro_prepared_full_tbl %>%
  filter(!is.na(sales_trans))

# and this one for the future forecast
centro_forecast_tbl <- centro_prepared_full_tbl %>%
  filter(is.na(sales_trans))




# 3.0 TRAIN/TEST (MODEL DATASET) ----

centro_prepared_tbl

centro_splits <- time_series_split(
  date_var = date, 
  data = centro_prepared_tbl, 
  assess = c_horizon, 
  cumulative = TRUE)

# this cv_plan is a nice way to see training vs. testing data
centro_splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(
    .date_var = date, 
    .value = sales_trans)




# 4.0 RECIPES ----
# - Time Series Signature - Adds bulk time-based features
# - Spline Transformation to index.num
# - Interaction: wday.lbl:week2
# - Fourier Features

model_fit_centro_lm %>% summary()
model_fit_centro_lm$terms %>% formula()


recipe_centro_base <- recipe(sales_trans ~ ., data = training(centro_splits)) %>%
  
  # time series signature
  step_timeseries_signature(date) %>%
  step_rm(matches("(iso)|(xts)|(hour)|(minute)|(second)|(am.pm)")) %>%
  
  # standardization/normalization
  step_normalize(matches("(index.num)|(year)|(yday)")) %>%
  
  # one-hot encoding (dummy) -- something that is text or lbl
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  
  # interaction
  step_interact(~ matches("week2") * matches("wday.lbl")) %>%
  
  # fourier
  step_fourier(date, period = c(7, 14, 30, 90, 365), K = 2)


recipe_centro_base %>%
  prep() %>%
  juice() %>%
  glimpse()




# 5.0 SPLINE MODEL ----

# * LM Model Spec ----

model_spec_centro_lm <- linear_reg() %>%
  set_engine("lm")


# * Spline Recipe Spec ----

# this is our base.
recipe_centro_base %>%
  prep() %>%
  juice() %>%
  glimpse()


# make a few changes...
recipe_centro_1 <- recipe_centro_base %>%
  step_rm(date) %>%
  step_ns(ends_with("index.num"), deg_free = 2) %>%
  step_rm(starts_with("lag_"))


# and this is our new recipe
recipe_centro_1 %>%
  prep() %>%
  juice() %>%
  glimpse()


# * Spline Workflow  ----

# create workflow, add a model, add a recipe, and fit it.
workflow_centro_fit_lm_1_spline <- workflow() %>%
  add_model(model_spec_centro_lm) %>%
  add_recipe(recipe_centro_1) %>%
  fit(training(centro_splits))


# now you can see a summary of the fit portion of the workflow you created.
workflow_centro_fit_lm_1_spline %>%
  pull_workflow_fit() %>%
  pluck("fit") %>%
  summary()



# 6.0 MODELTIME  ----

# put your fitted workflow into the modeltime table (this is where you compare multiple workflows)
calibration_centro_tbl <- modeltime_table(workflow_centro_fit_lm_1_spline) %>%
  modeltime_calibrate(new_data = testing(centro_splits))

# take the calibrated data, new data has to be the future testing data, and actual is the prepared data from the beginning.
calibration_centro_tbl %>%
  modeltime_forecast(new_data = testing(centro_splits), actual_data = centro_prepared_tbl) %>%
  plot_modeltime_forecast()


calibration_centro_tbl %>% 
  modeltime_accuracy()



# 7.0 LAG MODEL ----

# * Lag Recipe ----
# we dropped the lags previously, but now we want to check the recipe WITH the lags.

recipe_centro_base %>%
  prep() %>% 
  juice() %>%
  glimpse()


recipe_centro_2 <- recipe_centro_base %>%
  step_rm(date) %>%
  step_naomit(starts_with("lag_"))



recipe_centro_2 %>% 
  prep() %>% 
  juice() %>%
  glimpse()


# * Lag Workflow ----

workflow_centro_fit_lm_2_lag <- workflow() %>%
  add_model(model_spec_centro_lm) %>%
  add_recipe(recipe_centro_2) %>%
  fit(training(centro_splits))



workflow_centro_fit_lm_2_lag %>%
  pull_workflow_fit() %>%
  pluck("fit") %>%
  summary()



# * Compare with Modeltime -----

calibration_centro_tbl <- modeltime_table(
  workflow_centro_fit_lm_1_spline,
  workflow_centro_fit_lm_2_lag
) %>%
  modeltime_calibrate(new_data = testing(centro_splits))


calibration_centro_tbl %>%
  modeltime_forecast(new_data = testing(centro_splits),
                     actual_data = centro_prepared_tbl) %>%
  plot_modeltime_forecast()


calibration_centro_tbl %>%
  modeltime_accuracy()




# 8.0 FUTURE FORECAST ----

# this is where you are retraining on the full dataset, not just the 80% from the beginning.
refit_centro_tbl <- calibration_centro_tbl %>%
  modeltime_refit(data = centro_prepared_tbl)


# THIS is where you pull in the forecast_tbl you created at the beginning.
refit_centro_tbl %>%
  modeltime_forecast(new_data = centro_forecast_tbl,
                     actual_data = centro_prepared_tbl) %>%
  
  # invert transformation -- must follow this order, inverting the last one first.
  mutate(across(.value:.conf_hi, .fns = ~ standardize_inv_vec(
    x    = .,
    mean = c_std_mean,
    sd   = c_std_sd
  ))) %>%
  
  mutate(across(.value:.conf_hi, .fns = ~ log_interval_inv_vec(
    x           = .,
    limit_lower = c_limit_lower,
    limit_upper = c_limit_upper,
    offset      = c_offset
  ))) %>%
  
  plot_modeltime_forecast()




