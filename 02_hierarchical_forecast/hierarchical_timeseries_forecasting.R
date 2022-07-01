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
        ORDER BY Date desc
"


# Run the query and store
bq_query <- bq_project_query(projectid, sql)
data_tbl <- bq_table_download(bq_query)




# Prepare original dataset

data_tbl$tienda <- str_replace_all(data_tbl$tienda, "En Linea", "En.Linea")

data_tbl$combined <- str_c(data_tbl$tienda, "_",
                           data_tbl$metal_type, "_",
                           data_tbl$product_type)





# * Transform & Reshape ----

data_long_tbl <- data_tbl %>%
  
  # remove any previously forecasted rows
  filter(!is.na(sales)) %>%
  
  # # summarize by time
  group_by(combined) %>%
  summarize_by_time(.date_var = date, .by = "day", value = sum(sales)) %>%
  mutate(date = as.Date(date)) %>%
  
  # filter out "combined" time series with less than 2 instances
  filter(n() >= 2) %>%
  ungroup() %>%
  
  mutate(sales = value) %>%
  select(-value)



data_long_tbl %>% skim()



# * Split into Centro ----
data_long_centro_tbl <- data_long_tbl %>% 
  filter(grepl("Centro", combined))






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

item_id_sample <- c("Centro_10_Cadena", 
                    "Centro_14_Cadena",
                    "Centro_10_Argolla",
                    "Centro_14_Argolla",
                    "Centro_10_Anillo",
                    "Centro_14_Anillo")


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

data_prepared_tbl %>% glimpse()



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
model_tbl <- modeltime_table(
  wflw_fit_prophet,
  wflw_fit_prophet_boost,
  wflw_fit_xgboost,
  wflw_fit_rf,
  wflw_fit_nnet,
  wflw_fit_mars
) 


# Calibrate
calibration_tbl <- model_tbl %>%
  modeltime_calibrate(new_data = testing(splits))


# accuracy 
calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = TRUE,
    bordered = TRUE,
    resizable = TRUE
  )



# * Forecast ----

# model forecast on testing(splits)
test_forecast_tbl <- model_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data   = TRUE 
  )


# Visualize Actual vs Test

test_forecast_tbl %>%
  
  # FILTER IDENTIFIERS
  filter(combined %in% item_id_sample) %>%
  
  group_by(combined) %>%
  
  # Focus on end of series
  filter_by_time(
    .date_var = date,
    .start_date = last(date) %-time% "3 month",
    .end_date = "end"
  ) %>%
  
  plot_modeltime_forecast(
    .conf_interval_show = FALSE, 
    .smooth = FALSE, 
    .facet_ncol = 2, 
    .interactive = TRUE
  )



# * Refit ----

data_prepared_tbl_cleaned <- data_prepared_tbl %>%
  group_by(combined) %>%
  mutate(sales = ts_clean_vec(sales, period = 7)) %>%
  ungroup()



model_refit_tbl <- model_tbl %>%
  modeltime_refit(data = data_prepared_tbl_cleaned)



forecast_future_tbl <- calibration_tbl %>%
  modeltime_forecast(
    new_data    = future_tbl,
    actual_data = data_prepared_tbl,
    keep_data   = TRUE
  ) %>%
  mutate(
    .value    = expm1(.value),
    sales = expm1(sales))
  


forecast_future_tbl %>% 
  filter(combined %in% item_id_sample) %>%
  group_by(combined) %>%
  
  plot_modeltime_forecast(
    .facet_ncol = 3,
    .conf_interval_show = FALSE)



# * Turn OFF Parallel Backend
plan(sequential)



