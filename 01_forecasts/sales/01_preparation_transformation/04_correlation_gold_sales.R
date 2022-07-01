# Libraries ----
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
library(plotly)


# DATA ----

# * Pull Sales Data ----

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

# Run the query and store the data in a tibble
All_Data_Centro <- bq_project_query(projectid, sql_Centro)
BQ_Table_Centro <- bq_table_download(All_Data_Centro)



# * Pull Gold Price data ----

projectid = "source-data-314320"
sql <- "SELECT *
        FROM `source-data-314320.joyeria_dataset.gold_price`
        ORDER BY Date
"

# Run the query and store the data in a tibble
gold_price <- bq_project_query(projectid, sql)
bq_gold_price <- bq_table_download(gold_price)




# WEEKLY DATA ----

# create a weekly structure using the max mxn_per_gram
transactions_weekly_tbl <- BQ_Table_Centro %>%
  select(date, sales, mxn_per_gram) %>%
  summarize_by_time(.date_var    = date, 
                    .by          = "week", 
                    sales        = sum(sales), 
                    mxn_per_gram = max(mxn_per_gram))




# Visualize ----
transactions_weekly_tbl %>%
  pivot_longer(-date) %>%
  group_by(name) %>%
  plot_time_series(
    .date_var = date, 
    .value = value, 
    .facet_nrow = 2)




# Transform ----

log_standardized_transactions_weekly_tbl <- transactions_weekly_tbl %>%
  mutate(across(.cols = sales:mxn_per_gram, .fns = log1p)) %>%
  mutate(across(.cols = sales:mxn_per_gram, .fns = standardize_vec))
  


# Cross Correlations ----

log_standardized_transactions_weekly_tbl %>%
  plot_acf_diagnostics(.date_var = date,
                       .value = sales,
                       .ccf_vars = mxn_per_gram,
                       .show_ccf_vars_only = TRUE)







# DATA PREPARATION ----
# - Apply Preprocessing to Target



centro_prep_tbl <- BQ_Table_Centro %>%
  summarize_by_time(date, .by = "day", sales = sum(sales)) %>%
  pad_by_time(.pad_value = 0) %>%
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
  





# FEATURE INVESTIGATION ----

# 1.0 TIME-BASED FEATURES ----
# - tk_augment_timeseries_signature()


# * Time Series Signature ----

centro_prep_signature_tbl <- centro_prep_tbl %>%
  tk_augment_timeseries_signature() %>%
  select(-diff, 
         -contains(".iso"), 
         -contains(".xts"), 
         -matches("(hour)|(minute)|(second)|(am.pm)"))


# * Trend-Based Features ----

# ** Linear Trend

centro_prep_signature_tbl %>%
  plot_time_series_regression(
    .date_var = date, 
    .formula = sales_trans ~ index.num)


# ** Nonlinear Trend - Basis/Natural Splines

# Basis Spline
centro_prep_signature_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = sales_trans ~ splines::bs(index.num, degree = 3),
    .show_summary = TRUE
  )


#Natural Spline
centro_prep_signature_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = sales_trans ~ splines::ns(index.num, df = 50),
    .show_summary = TRUE
  )


# * Seasonal Features ----

# Weekly Seasonality

centro_prep_signature_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = sales_trans ~ wday.lbl,
    .show_summary = TRUE
  )

# ** Monthly Seasonality

centro_prep_signature_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = sales_trans ~ month.lbl,
    .show_summary = TRUE
  )


# ** The two combined

model_formula_centro_seasonality <- as.formula(
  sales_trans ~ splines::ns(x = index.num, df = 50)
  + wday.lbl + month.lbl + .
)


centro_prep_signature_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = model_formula_centro_seasonality,
    .show_summary = TRUE
  )




# 2.0 INTERACTIONS ----

centro_prep_signature_tbl %>% glimpse()

model_formula_centro_interactions <- as.formula(
  sales_trans ~ splines::ns(x = index.num, df = NULL)
  + .
  + (as.factor(wday.lbl) * month.lbl)
)


centro_prep_signature_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = model_formula_centro_interactions,
    .show_summary = TRUE
  )




# 3.0 FOURIER SERIES ----
# - tk_augment_fourier

# Data Prep

centro_prep_signature_tbl %>%
  plot_acf_diagnostics(.date_var = date, .value = sales_trans)


centro_prep_fourier_tbl <- centro_prep_signature_tbl %>%
  tk_augment_fourier(date, .periods = c(7, 14, 30, 90, 365), .K = 2)

centro_prep_fourier_tbl %>% glimpse()



# Model

model_formula_centro_fourier <- as.formula(
  sales_trans ~ splines::ns(x = index.num, df = 30)
  + .
)



# Visualize

centro_prep_fourier_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = model_formula_centro_fourier,
    .show_summary = TRUE
  )



# 4.0 LAGS ----
# - tk_augment_lags()

# Data Prep, forecasting 8 weeks.

centro_prep_fourier_tbl %>%
  plot_acf_diagnostics(
    date, .value = sales_trans, .lags = (8*7):600
  )


centro_prep_lags_tbl <- centro_prep_fourier_tbl %>%
  tk_augment_lags(.value = sales_trans, .lags = c(56, 58, 62, 63, 64, 69)) %>%
  drop_na()


# Model

model_formula_centro <- as.formula(
  sales_trans ~ splines::ns(x = index.num, df = NULL)
  + .
  + (as.factor(wday.lbl) * month.lbl)
)



# Visualize

centro_prep_lags_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = model_formula_centro,
    .show_summary = TRUE
  )




# 5.0 SPECIAL EVENTS ----

# Data Prep

# prep data for acceleration of gold price

gold_daily_tbl <- bq_gold_price %>%
  mutate(gold_accel = diff_vec(x = mxn_per_gram, lag = 1, difference = 1)) %>%
  
  #start after COVID
  filter_by_time(.date_var = date, .start_date = "2020-05-17") %>%
  
  select(date, gold_accel)


gold_prep_events_tbl <- gold_daily_tbl %>%
  mutate(increase = ifelse(gold_accel >= 20, 1, 0)) %>%
  mutate(decrease = ifelse(gold_accel <= -20, 1, 0)) %>%
  select(-gold_accel) %>%
  filter((increase|decrease) != 0)


# use best model and left join and change NA's to 0

centro_prep_events_tbl <- centro_prep_lags_tbl %>%
  left_join(gold_prep_events_tbl, by = "date") %>%
  mutate(increase = ifelse(is.na(increase), 0, increase)) %>%
  mutate(decrease = ifelse(is.na(decrease), 0, decrease))
  
centro_prep_events_tbl %>% glimpse()




# show the special events in plot. Tip: start with plot_time_series, and then convert to ggplot/plotly
graph <- centro_prep_events_tbl %>%
  plot_time_series(
    date, 
    sales_trans,
    .interactive = FALSE,
    .smooth = FALSE) +
  geom_point(color = "red", data = . %>% filter(increase == 1)) +
  geom_point(color = "green", data = . %>% filter(decrease == 1)) +
  ggtitle(paste("Sales vs Gold Price acceleration"))
  

ggplotly(graph)



# Model

model_formula_centro <- as.formula(
  sales_trans ~ splines::ns(x = index.num, df = NULL)
  + .
)


# Visualize

centro_prep_events_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = model_formula_centro,
    .show_summary = TRUE
  )





# 7.0 RECOMMENDATION ----
# - Best model: 
# - Best Model Formula:

# Model

model_formula_centro <- as.formula(
  sales_trans ~ splines::ns(x = index.num, knots = quantile(index.num, probs = c(0.25, 0.50, 0.84)))
  + .
  + wday.lbl + month.lbl
)


# Visualize

centro_prep_lags_tbl %>%
  plot_time_series_regression(
    .date_var = date,
    .formula = model_formula_centro,
    .show_summary = TRUE
  )




# Linear Regression Model

model_fit_centro_lm <- lm(model_formula_centro, data = centro_prep_lags_tbl)


# automatically recreate the model from what is saved in the environment
model_fit_centro_lm$terms %>%
  formula()


write_rds(model_fit_centro_lm, path = "00_models/model_fit_centro_lm.rds")

read_rds("00_models/model_fit_centro_lm.rds")






