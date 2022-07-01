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
sql_gold <- "SELECT *
             FROM `source-data-314320.joyeria_dataset.gold_price`
             ORDER BY Date
"

# Run the query and store the data in a tibble
gold_price <- bq_project_query(projectid, sql_gold)
bq_gold_price <- bq_table_download(gold_price)




# 1.0 DATA PREP ----

bq_gold_price <- bq_gold_price[!duplicated(bq_gold_price$date), ]


# fills in NA values for missing dates, but we specify .by = "day" with 0
gold_day_prepared_tbl <- bq_gold_price %>%
  pad_by_time(
    .date_var = date,
    .by = "day",
    .pad_value = 0
  )



# * Visualization ----
gold_day_prepared_tbl %>%
  plot_time_series(
    .date_var = date,
    .value = mxn_per_gram
  )




# * Filtering ----

evaluation_gold_tbl <- gold_day_prepared_tbl %>%
  filter_by_time(
    .date_var = date,
    .start_date = "2015-07-01",
    .end_date = "end"
  )



evaluation_gold_tbl %>%
  plot_time_series(
    .date_var = date,
    .value = mxn_per_gram,
    .smooth_period = "1 year"
  )



# 4.0 OUTLIER CLEANING ----


# * + Cleaning (Imputation + Outlier Removal) ----
gold_daily_cleaned_tbl <- evaluation_gold_tbl %>%
  mutate(mxn_per_gram_cleaned = ts_clean_vec(mxn_per_gram)) %>%
  select(date, mxn_per_gram, mxn_per_gram_cleaned)

gold_daily_cleaned_tbl %>%
  pivot_longer(-date) %>%
  plot_time_series(date, value, .color_var = name, .smooth = FALSE)



# Outlier Effect - Before Cleaning
gold_daily_cleaned_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = mxn_per_gram ~ as.numeric(date) +
      wday(date, label = TRUE) +
      month(date, label = TRUE),
    .show_summary = TRUE
  )


# Outlier Effect - After Cleaning
gold_daily_cleaned_tbl %>%
  select(-mxn_per_gram) %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = mxn_per_gram_cleaned ~ date +
      wday(date, label = TRUE) +
      month(date, label = TRUE),
    .show_summary = TRUE
  )



# 5.0 LAGS & DIFFERENCING -----

# * Differencing ----

gold_daily_tbl <- evaluation_gold_tbl %>%
  select(date, mxn_per_gram)


gold_daily_tbl %>%
  mutate(gold_accel = diff_vec(mxn_per_gram, lag = 1, difference = 1)) %>%
  pivot_longer(-date) %>%
  group_by(name) %>%
  plot_time_series(date, value, .color_var = name, .smooth = FALSE)




# Comparing Differences

evaluation_diff_tbl <- evaluation_gold_tbl %>%
  mutate(across(mxn_per_gram:gold_usd_oz, .fns = diff_vec))


evaluation_diff_tbl %>%
  pivot_longer(-date) %>%
  plot_time_series(date, value, name, .smooth = FALSE)


# Inversion
evaluation_diff_tbl %>%
  mutate(mxn_per_gram = diff_inv_vec(mxn_per_gram, initial_values = 592.50461)) %>%
  mutate(mxn = diff_inv_vec(mxn, initial_values = 15.77486)) %>%
  mutate(gold_usd_oz = diff_inv_vec(gold_usd_oz, initial_values = 1168.25))






# 6.0 FOURIER SERIES ----
# - Useful for incorporating seasonality & autocorrelation
# - BENEFIT: Don't need a lag, just need a frequency (based on your time index)

evaluation_gold_tbl %>%
  plot_acf_diagnostics(date, mxn)



# * Augmenting (Multiple Fourier Series) ----

# Gold MXN (gr)
evaluation_gold_tbl %>%
  select(date, mxn_per_gram) %>%
  tk_augment_fourier(.date_var = date, .periods = c(4, 25, 30, 64, 119, 360), .K = 2) %>%
  plot_time_series_regression(
    .date_var = date, .formula = log1p(mxn_per_gram) ~ as.numeric(date) + . - date,
    .show_summary = TRUE
  )


# Gold USD (oz)
evaluation_gold_tbl %>%
  select(date, gold_usd_oz) %>%
  tk_augment_fourier(.date_var = date, .periods = c(4, 25, 30, 64, 119, 360), .K = 2) %>%
  plot_time_series_regression(
    .date_var = date, .formula = log1p(gold_usd_oz) ~ as.numeric(date) + . - date,
    .show_summary = TRUE
  )


# MXN/USD rate
evaluation_gold_tbl %>%
  select(date, mxn) %>%
  tk_augment_fourier(.date_var = date, .periods = c(10, 24, 65, 74, 78, 127), .K = 2) %>%
  plot_time_series_regression(
    .date_var = date, .formula = log1p(mxn) ~ as.numeric(date) + . - date,
    .show_summary = TRUE
  )





# 7.0 CONFINED INTERVAL FORECASTING ----
# - Showcase: log_interval_vec()
# - Transformation used to confine forecasts to a max/min interval



# * Data ----

gold_lower <- 0
gold_upper <- 1581.09352
offset <- 1



evaluation_gold_tbl %>%
  select(date, mxn_per_gram) %>%
  plot_time_series(.date_var = date, log_interval_vec(mxn_per_gram, 
                                                      limit_lower = gold_lower, 
                                                      limit_upper = gold_upper,
                                                      offset = 1))

gold_fourier_periods <- c(4, 25, 30, 64, 119, 360)
gold_fourier_order <- 4

gold_transformed_tbl <- evaluation_gold_tbl %>%
  select(date, mxn_per_gram) %>%
  mutate(mxn_per_gram = log_interval_vec(mxn_per_gram,
                                       limit_lower = gold_lower,
                                       limit_upper = gold_upper,
                                       offset = offset)) %>%
  tk_augment_fourier(.date_var = date, .periods = gold_fourier_periods, .K = gold_fourier_order)

gold_transformed_tbl %>% glimpse()



# * Model ----

gold_model_formula <- as.formula(
  mxn_per_gram ~ as.numeric(date) +
    wday(date, label = TRUE) +
    month(date, label = TRUE) +
    quarter(date) +
    . - date)


gold_transformed_tbl %>%
  plot_time_series_regression(.date_var = date, 
                              .formula = gold_model_formula, 
                              .show_summary = TRUE)


model_fit_gold_lm <- lm(formula = gold_model_formula, data = gold_transformed_tbl)

summary(model_fit_gold_lm)



# * Create Future Data ----

gold_future_tbl <- gold_transformed_tbl[!duplicated(gold_transformed_tbl$date), ] %>%
  future_frame(.length_out = "35 weeks") %>%
  tk_augment_fourier(.date_var = date, .periods = gold_fourier_periods, .K = gold_fourier_order)


# * Predict ----

gold_predictions <- predict(model_fit_gold_lm, newdata = gold_future_tbl) %>% as.vector()


# * Create Alpha, Residuals & Confidence Level ----

gold_conf_int  <- 0.95
gold_residuals <- model_fit_gold_lm$residuals %>% as.vector()

gold_alpha <- (1-gold_conf_int) / 2

qnorm(gold_alpha)
qnorm(1-gold_alpha)

gold_abs_margin_error <- abs(qnorm(gold_alpha) * sd(gold_residuals))



# * Combine data ----

gold_forecast_tbl <- gold_transformed_tbl %>%
  select(date, mxn_per_gram) %>%
  add_column(type = "actual") %>%
  bind_rows(
    gold_future_tbl %>%
      select(date) %>%
      mutate(mxn_per_gram = gold_predictions,
             type = "prediction"
             ) %>%
      mutate(
        conf_lo = mxn_per_gram - gold_abs_margin_error,
        conf_hi = mxn_per_gram + gold_abs_margin_error
      )
  )


gold_forecast_tbl %>% tail(20)

# * Visualize ----

gold_forecast_tbl %>%
  pivot_longer(cols = c(mxn_per_gram, conf_lo, conf_hi)) %>%
  plot_time_series(date, value, .color_var = name, .smooth = FALSE)




# * Invert and Visualize ----

gold_forecast_tbl %>%
  pivot_longer(cols = c(mxn_per_gram, conf_lo, conf_hi)) %>%
  plot_time_series(.date_var = date, 
                   .value = log_interval_inv_vec(x = value, 
                                                 limit_lower = gold_lower, 
                                                 limit_upper = gold_upper,
                                                 offset = offset),
                   .color_var = name, 
                   .smooth = FALSE)






gold_forecast_tbl %>%
  select(date, type, mxn_per_gram) %>%
  plot_time_series(.date_var = date, 
                   .value = log_interval_inv_vec(x = mxn_per_gram, 
                                                 limit_lower = gold_lower, 
                                                 limit_upper = gold_upper,
                                                 offset = offset),
                   .color_var = type, 
                   .smooth = FALSE)



