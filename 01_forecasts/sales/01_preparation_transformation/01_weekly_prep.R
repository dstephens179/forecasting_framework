# GOAL: Forecast Sales

# OBJECTIVES ----
# - Forecast using 3 models
#   1. Prophet
#   2. LM Spline
#   3. LM Lag Features



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
sql <- "SELECT 
          date,
          sales
        FROM `source-data-314320.Store_Data.All_Data`
        WHERE owner = 'M&D'
        AND tienda = 'Centro'
        AND sales <> 0
        ORDER BY Date desc
"

# Run the query and store the data in a tibble
All_Data_Centro <- bq_project_query(projectid, sql)
BQ_Table_Centro <- bq_table_download(All_Data_Centro)



events <- read.csv(file = "00_data/calendar.csv") %>% mutate(event = 1)
events$date <- ymd(events$date)
events <- events %>% 
  summarize_by_time(.date_var = date, .by = "week", event = max(event))





# * Skim Data ----

skimr::skim(BQ_Table_Centro)



# 1.0 DATA PREP ----

glimpse(BQ_Table_Centro)



# Create daily dataset
sales_daily_tbl <- BQ_Table_Centro %>%
  
  summarize_by_time(
    .date_var = date,
    .by = "day",
    sales = sum(sales)
  ) %>%
  
  pad_by_time(
    .date_var = date,
    .by = "day",
    .pad_value = 0)




# * Correct Covid Closure with Lead----

start_date = as.Date('2020-03-01')
end_date = as.Date('2020-12-31')



# create new table where covid date data is overwritten with prior year
sales_daily_covid_tbl <- sales_daily_tbl %>%
  mutate(date_1 = ymd(sales_daily_tbl$date) + years(1)) %>%
  mutate(sales_1 = sales) %>%
  filter(date_1 %within% interval(start_date, end_date)) %>%
  mutate(date = date_1) %>%
  mutate(sales = sales_1) %>%
  select(date, sales)


# create new table without covid dates
sales_daily_drop_tbl <- sales_daily_tbl %>%
  mutate(date_na = ifelse(date %within% interval(start_date, end_date), NA, date)) %>%
  drop_na() %>%
  select(-date_na)


# finally, bind rows
sales_daily_imputed_tbl <- bind_rows(sales_daily_drop_tbl, sales_daily_covid_tbl) %>% arrange(date)



# Create weekly dataset
sales_weekly_imputed_tbl <- sales_daily_imputed_tbl %>%
  
  summarize_by_time(
    .date_var = date,
    .by = "week",
    sales = sum(sales)
  ) %>%
  
  pad_by_time(
    .date_var = date,
    .by = "week",
    .pad_value = 0) %>%
  
  filter_by_time(
    .date_var = date, 
    .start_date = "2015-07-19", 
    .end_date = "2022-04-30")



# * Visualize ----

sales_weekly_imputed_tbl %>%
  plot_time_series(
    .date_var = date,
    .value    = sales
  )



# * Plot ACF ----

sales_weekly_imputed_tbl %>%
  plot_acf_diagnostics(
    .date_var = date,
    .value    = sales
  )





# 2.0 TRANSFORMATION ----

# * Log & Standardize Revenue ----

sales_trans_weekly_tbl <- sales_weekly_imputed_tbl %>%
  mutate(sales = log(sales)) %>%
  mutate(sales = standardize_vec(sales))

std_weekly_mean <- 11.5451308641348
std_weekly_sd   <- 0.53505184655934


# * Visualize ----

sales_trans_weekly_tbl %>%
  plot_time_series(.date_var = date, .value = sales)




# 3.0 CREATE FULL DATASET ----

# * Save Key Params ----

weekly_horizon         <- 52
weekly_lag_period      <- 52
weekly_rolling_periods <- c(5, 12, 20, 21, 25, 26, 32)



# * Prepare Full Dataset ----

prepared_full_weekly_tbl <- sales_trans_weekly_tbl %>%
  # add future window
  bind_rows(
    future_frame(
      .data = .,
      .date_var = date, 
      .length_out = "52 weeks")
  ) %>%
  
  tk_augment_lags(
    .value = sales,
    .lags = weekly_lag_period) %>%
  
  tk_augment_slidify(
    .value = sales_lag52, 
    .period = weekly_rolling_periods,
    .f = mean, 
    .align = "center",
    .partial = TRUE) %>%
  
  # add xregs
  left_join(events, by = "date") %>%
  mutate(event = ifelse(is.na(event),0,event)) %>%
  
  rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .))
  
  
# * Visualize ----

prepared_full_weekly_tbl %>%
  pivot_longer(-date) %>%
  plot_time_series(
    .date_var = date,
    .value = value,
    .color_var = name,
    .smooth = FALSE)



# * Model Data / Forecast Data Split ----

prepared_weekly_tbl <- prepared_full_weekly_tbl %>%
  filter(!is.na(sales))


forecast_weekly_tbl <- prepared_full_weekly_tbl %>%
  filter(is.na(sales))




# * Train / Test Split ----

weekly_splits <- prepared_weekly_tbl %>%
  time_series_split(
    date_var = date,
    assess = "52 weeks",
    cumulative = TRUE
  )


weekly_splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(
    .date_var = date,
    .value = sales
  )





# 4.0 LM FEATURE ENGINEERING ----

# * Identify Possible Features ----

# plot seasonality, use log() transformation in value to see more variation
prepared_weekly_tbl %>%
  plot_seasonal_diagnostics(
    .date_var = date,
    .value = sales
  )



# * Base Recipe ----


recipe_weekly_base <- recipe(sales ~ ., data = training(weekly_splits)) %>%
  
  # Time Series Signature
  step_timeseries_signature(date) %>% 
  
  # this next step removes unnecessary columns and cleans up dataframe
  step_rm(matches("(iso)|(xts)|(hour)|(day)|(minute)|(second)|(am.pm)")) %>%
  
  # now we normalize the time and year, because those were big numbers and they look weird.
  step_normalize(matches("(index.num)|(year)|(yday)")) %>%
  
  # now we create dummy variables from categorical data.
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  
  # interaction
  step_interact(~ matches("month") * matches("week")) %>%
  
  step_fourier(date, period = c(4, 13, 26, 52), K = 2)



recipe_weekly_base %>%
  prep() %>% 
  juice() %>%
  glimpse()




# Spline Model ----


# * Visualize ----

prepared_weekly_tbl %>%
  plot_time_series_regression(.date_var = date, 
                              .formula = sales ~ splines::ns(date, df = 4),
                              .show_summary = TRUE)



# * LM Model Specs ----

model_weekly_lm <- linear_reg() %>%
  set_engine("lm")



# * Recipe Spec - Spline ----
recipe_weekly_1_spline <- recipe_weekly_base %>%
  step_rm(date) %>%
  step_ns(contains("index.num"), deg_free = 4) %>%
  step_rm(starts_with("lag_"))


recipe_weekly_1_spline %>%
  prep() %>%
  juice() %>%
  glimpse()



# * Workflow - Spline ----
workflow_fit_weekly_lm_1_spline <- workflow() %>%
  add_model(model_weekly_lm) %>%
  add_recipe(recipe_weekly_1_spline) %>%
  fit(training(weekly_splits))


workflow_fit_weekly_lm_1_spline %>%
  extract_fit_parsnip() %>%
  pluck("fit") %>%
  summary()




# Rolling Lag Model ----

# * Recipe Spec - Lag ----
recipe_weekly_2_lag <- recipe_weekly_base %>%
  step_rm(date) %>% 
  step_naomit(starts_with("lag_"))


recipe_weekly_2_lag %>%
  prep() %>%
  juice() %>%
  glimpse()




# * Workflow - Lag ----
workflow_fit_weekly_lm_2_lag <- workflow() %>%
  add_model(model_weekly_lm) %>%
  add_recipe(recipe_weekly_2_lag) %>%
  fit(training(weekly_splits))


workflow_fit_weekly_lm_2_lag %>%
  extract_fit_parsnip() %>%
  pluck("fit") %>%
  summary()







# 5.0 MODELTIME ----

# * Make Modeltime Table ----
model_weekly_tbl <- modeltime_table(
  workflow_fit_weekly_lm_1_spline,
  workflow_fit_weekly_lm_2_lag
) %>%
  update_model_description(1, "lm - spline") %>%
  update_model_description(2, "lm - lag")


# As a precautionary measure, refit the models using modeltime_refit()
# This prevents models that can go bad over time because of software changes
model_weekly_tbl <- model_weekly_tbl %>%
  modeltime_refit(training(weekly_splits))



# * Calibrate ----
calibration_weekly_tbl <- model_weekly_tbl %>%
  modeltime_calibrate(testing(weekly_splits))



# * Check Accuracy ----
calibration_weekly_tbl %>%
  modeltime_accuracy()



# * Visualize ----
calibration_weekly_tbl %>%
  modeltime_forecast(new_data = testing(weekly_splits),
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)



# 6.0 FUTURE FORECAST ----

# * Refit to Prepared Data ----
refit_weekly_tbl <- calibration_weekly_tbl %>% 
  modeltime_refit(prepared_weekly_tbl)


refit_weekly_tbl


# * Forecast ----

refit_weekly_tbl %>%
  modeltime_forecast(new_data = forecast_weekly_tbl,
                     actual_data = prepared_weekly_tbl) %>%
  plot_modeltime_forecast()



# * Invert Transformation ----
forecast_future_weekly_tbl <- refit_weekly_tbl %>%
  modeltime_forecast(new_data = forecast_weekly_tbl,
                     actual_data = prepared_weekly_tbl) %>%
  
  mutate(across(.cols = .value:.conf_hi, 
                .fns = ~ standardize_inv_vec(x = ., 
                                             mean = std_weekly_mean, 
                                             sd = std_weekly_sd))) %>%
  
  mutate(across(.cols = .value:.conf_hi, 
                .fns = exp))
  


# * Visualize Weekly ----
forecast_future_weekly_tbl %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)



# Summarize & Visualize by Month ----
forecast_future_weekly_tbl %>%
  group_by(.model_id, .model_desc, .key, .index) %>%
  summarize_by_time(.by = "month", .value = sum(.value)) %>%
  ungroup() %>%
  plot_modeltime_forecast(.conf_interval_show = FALSE)





# 7.0 SAVE ARTIFACTS ----

weekly_artifacts_list <- list(
  # data
  data = list(
    prepared_weekly_tbl = prepared_weekly_tbl, 
    forecast_weekly_tbl = forecast_weekly_tbl
  ),
  
  # recipes
  recipes = list(
    recipe_weekly_base     = recipe_weekly_base,
    recipe_weekly_1_spline = recipe_weekly_1_spline,
    recipe_weekly_2_lag    = recipe_weekly_2_lag
  ),
  
  # models/workflows
  models = list(
    workflow_fit_weekly_lm_1_spline = workflow_fit_weekly_lm_1_spline,
    workflow_fit_weekly_lm_2_lag    = workflow_fit_weekly_lm_2_lag
  ),
  
  # inversion params
  standardize = list(
    std_weekly_mean = std_weekly_mean,
    std_weekly_sd   = std_weekly_sd
  )
)


# save it for later.  You will need it.
weekly_artifacts_list %>% 
  write_rds("00_models/weekly_artifacts_list.rds")


