# GOAL: Forecast Gold MXN to USD conversion rate.


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
sql <- "SELECT *
        FROM `source-data-314320.joyeria_dataset.gold_price`
        ORDER BY Date
"

# Run the query and store the data in a tibble
mxn_usd <- bq_project_query(projectid, sql)
bq_mxn_usd <- bq_table_download(mxn_usd)



bq_mxn_usd <- bq_mxn_usd[!duplicated(bq_mxn_usd$date), ]


# * Skim Data ----

skimr::skim(bq_mxn_usd)



# 1.0 EDA & DATA PREP ----

glimpse(bq_mxn_usd)


# * Diagnostics ----

bq_mxn_usd %>%
  tk_summary_diagnostics(
    .date_var = date
  )


# * Pad the Time Series ----

# fills in NA values for missing dates, but we specify .by = "day" with 0
mxn_day_tbl <- bq_mxn_usd %>%
  pad_by_time(
    .date_var = date,
    .by = "day",
    .pad_value = 0
  )


# * Visualization ----
mxn_day_tbl %>%
  plot_time_series(
    .date_var = date,
    .value = mxn
  )



# 2.0 EVALUATION PERIOD ----

# * Transform & Filter ----

evaluation_mxn_tbl <- mxn_day_tbl %>%
  filter_by_time(
    .date_var = date,
    .start_date = "2015-07-01",
    .end_date = "end"
  ) %>%
  select(date, mxn) %>%
  mutate(mxn = log(mxn)) %>%
  mutate(mxn = standardize_vec(mxn))



# * Save Parameters ----
mxn_horizon <- 240
mxn_lag_periods <- 240
mxn_rolling_periods <- c(24, 44, 65, 74, 78, 127)

std_mxn_mean <- 2.96478965060408
std_mxn_sd   <- 0.0780616094829867



# 2.1 CREATE FULL DATASET ----

# * Prepare Full Dataset ----

mxn_full_daily_tbl <- evaluation_mxn_tbl %>%
  # add future window
  bind_rows(
    future_frame(
      .data = .,
      .date_var = date, 
      .length_out = "240 days")
  ) %>%
  
  tk_augment_lags(
    .value = mxn, 
    .lags = mxn_lag_periods) %>%
  
  tk_augment_slidify(
    .value = mxn_lag240, 
    .period = mxn_rolling_periods, 
    .f = mean, 
    .align = "center", 
    .partial = TRUE) %>%
  
  rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .))



# * Visualize ----

mxn_full_daily_tbl %>%
  pivot_longer(-date) %>%
  plot_time_series(
    .date_var = date, 
    .value = value, 
    .color_var = name, 
    .smooth = FALSE)



# * Model Data / Forecast Data Split ----

mxn_prepared_daily_tbl <- mxn_full_daily_tbl %>%
  filter(!is.na(mxn))


mxn_forecast_daily_tbl <- mxn_full_daily_tbl %>%
  filter(is.na(mxn))



mxn_prepared_daily_tbl %>%
  plot_time_series(
    .date_var = date,
    .value = mxn,
    .smooth_period = "1 year"
  )


mxn_prepared_daily_tbl %>%
  plot_acf_diagnostics(.date_var = date, .value = mxn)



# * Train/Test ----

splits_mxn <- mxn_prepared_daily_tbl %>%
  time_series_split(
    date_var = date,
    assess = "240 days",   
    cumulative = TRUE    
  )


splits_mxn %>%
  tk_time_series_cv_plan() %>%    
  plot_time_series_cv_plan(
    .date_var = date,
    .value = mxn
  )







# 3.0 PROPHET FORECASTING ----

# * Prophet Model using Modeltime/Parsnip ----

model_prophet_fit_mxn <- prophet_reg(seasonality_yearly = TRUE, 
                                     seasonality_weekly = TRUE, 
                                     changepoint_range = .75,
                                     changepoint_num = 25) %>%
  set_engine("prophet") %>%
  fit(mxn ~ date, 
      data = training(splits_mxn)) 


model_prophet_fit_mxn

# * Modeltime Process ----

model_mxn_tbl <- modeltime_table(
  model_prophet_fit_mxn
)


# * Calibrate on Testing Data----
calibration_mxn_tbl <- model_mxn_tbl %>%
  modeltime_calibrate(
    new_data = testing(splits_mxn)
  )


# * Visualize Forecast ----
calibration_mxn_tbl %>%
  modeltime_forecast(actual_data = mxn_prepared_daily_tbl) %>%
  plot_modeltime_forecast()


# * Get Accuracy Metrics ----
calibration_mxn_tbl %>%
  modeltime_accuracy()




# 4.0 FORECASTING WITH FEATURE ENGINEERING ----

# * Identify Possible Features ----

# plot seasonality, use log() transformation in value to see more variation
mxn_prepared_daily_tbl %>%
  plot_seasonal_diagnostics(
    .date_var    = date,
    .value       = (mxn),
    .feature_set = c("week", "month.lbl", "year")
  )



# * Recipe Spec ----


recipe_spec_mxn <- recipe(mxn ~ .,            # the . tells recipe() to look at all variables
                      data = training(splits_mxn)) %>%
  
  # Time Series Signature
  step_timeseries_signature(date) %>%  # this builds calendar features from a date field
  
  # this next step removes unnecessary columns and cleans up dataframe
  step_rm(ends_with(".iso"),
          ends_with(".xts"),
          contains("hour"), 
          contains("minute"), 
          contains("second"), 
          contains("am.pm")
  ) %>%
  
  step_normalize(ends_with("index.num"),
                 ends_with("_year")) %>%
  
  step_fourier(date, period = c(24, 44, 65, 74, 78, 127), K = 2) %>%
  
  # now we create dummy variables from categorical data.
  step_dummy(all_nominal())    #all_nominal() selects any features that are character or factor



recipe_spec_mxn %>%
  prep() %>%    # this performs pre-calculations and preprocessing training steps.
  juice() %>%   # returns the trianing data post-transformation so we can see the transformations that were applied.
  glimpse()


# * Machine Learning Specs ----

# when you have a recipe and model, you we create a workflow -- here is linear regression.
model_spec_mxn <- linear_reg() %>%
  set_engine("lm")


# create a workflow
workflow_fit_lm_mxn <- workflow() %>%
  add_model(model_spec_mxn) %>%     # first, add the model 
  add_recipe(recipe_spec_mxn) %>%   # next, add the recipe
  fit(training(splits_mxn))         # last, fit the training(splits_mxn)


workflow_fit_lm_mxn


# * Modeltime Process ----

# Calibrate on Testing
calibration_mxn_tbl <- modeltime_table(
  model_prophet_fit_mxn,
  workflow_fit_lm_mxn
) %>%
  modeltime_calibrate(testing(splits_mxn))



# Check Accuracy & Plot
calibration_mxn_tbl %>%
  modeltime_accuracy()


calibration_mxn_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits_mxn),
    actual_data = mxn_prepared_daily_tbl
  ) %>%
  plot_modeltime_forecast()



# Invert
calibration_mxn_tbl %>%
  modeltime_forecast(new_data = mxn_forecast_daily_tbl,
                     actual_data = mxn_prepared_daily_tbl) %>%
  
  mutate(across(.cols = .value:.conf_hi, .fns = ~ standardize_inv_vec(x = ., 
                                                                      mean = std_mxn_mean, 
                                                                      sd = std_mxn_sd))) %>%
  
  mutate(across(.cols = .value:.conf_hi, .fns = exp)) %>%
  
  plot_modeltime_forecast(.conf_interval_show = FALSE)








# 7.0 SAVE ARTIFACTS ----

mxn_daily_artifacts_list <- list(
  # data
  data = list(
    mxn_prepared_daily_tbl = mxn_prepared_daily_tbl, 
    mxn_forecast_daily_tbl = mxn_forecast_daily_tbl
  ),
  
  # recipes
  recipes = list(
    recipe_spec_mxn     = recipe_spec_mxn
  ),
  
  # models/workflows
  models = list(
    workflow_fit_lm_mxn = workflow_fit_lm_mxn
  ),
  
  # inversion params
  standardize = list(
    std_mxn_mean = std_mxn_mean,
    std_mxn_sd   = std_mxn_sd
  )
)


# save it for later.  You will need it.
mxn_daily_artifacts_list %>% 
  write_rds("00_models/mxn_daily_artifacts_list.rds")






