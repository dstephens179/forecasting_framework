# GOAL: Forecast Sales

# OBJECTIVES ----
# - Dive into a time-series analysis project
# - Experience Frameworks: modeltime
# - Experience 2 Algorithms:
#   1. Prophet
#   2. LM w/ Engineered Features



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



# * Skim Data ----

skimr::skim(BQ_Table_Centro)



# 1.0 EDA & DATA PREP ----

glimpse(BQ_Table_Centro)

sales_monthly_tbl <- BQ_Table_Centro %>%
  
  summarize_by_time(
    .date_var = date,
    .by = "month",
    sales = sum(sales)
  ) %>%
  
  pad_by_time(
    .date_var = date,
    .by = "month",
    .pad_value = 0)



# * Diagnostics ----

sales_monthly_tbl %>%
  tk_summary_diagnostics(
    .date_var = date
  )


# * Visualization ----
sales_monthly_tbl %>%
  plot_time_series(
    .date_var = date,
    .value = sales
  )


## * Visualize ACF ----

sales_monthly_tbl %>%
  plot_acf_diagnostics(
    .date_var = date,
    .value = sales
  )





# SEPARATE INTO MODELING & FORECAST DATA ----

prepared_full_tbl <- sales_monthly_tbl %>%
  # add future window
  bind_rows(
    future_frame(.data = ., .date_var = date, .length_out = "1 year")
  ) 




prepared_tbl <- prepared_full_tbl %>%
  filter(!is.na(sales))


forecast_tbl <- prepared_full_tbl %>%
  filter(is.na(sales))


# EVALUATION PERIOD ----

# * Train/Test ----

splits <- sales_monthly_tbl %>%
  time_series_split(
    date_var = date,
    assess = "1 year",
    cumulative = TRUE     # TRUE grabs all prior data for training; you specify FALSE with "initial."
  )


splits %>%
  tk_time_series_cv_plan() %>%     # puts the .id and .key fields into the tibble
  plot_time_series_cv_plan(
    .date_var = date,
    .value = sales
  )




# 3.0 PROPHET FORECASTING ----

# * Prophet Model using Modeltime/Parsnip ----

model_prophet_fit <- prophet_reg(seasonality_yearly = TRUE) %>%
  set_engine("prophet") %>%
  fit(sales ~ date,       # must include date feature, because it works with modeltime
      data = training(splits))


model_prophet_fit

# * Modeltime Process ----

model_tbl <- modeltime_table(    # modeltime_table captures and organizes models, good for many models
  model_prophet_fit
)


# * Calibration ----
calibration_tbl <- model_tbl %>%
  modeltime_calibrate(          #this pulls in testing & residuals
    new_data = testing(splits)
  )


# * Visualize Forecast ----
calibration_tbl %>%
  modeltime_forecast(actual_data = sales_monthly_tbl) %>%  # this is the full dataset that we used at the beginning.
  plot_modeltime_forecast()


# * Get Accuracy Metrics ----
calibration_tbl %>%
  modeltime_accuracy()




# 4.0 FORECASTING WITH FEATURE ENGINEERING ----

# * Identify Possible Features ----

# plot seasonality, use log() transformation in value to see more variation
sales_monthly_tbl %>%
  plot_seasonal_diagnostics(
    .date_var = date,
    .value = log(sales)
  )



# * Recipe Spec ----

training(splits)


recipe_spec <- recipe(sales ~ .,            # the . tells recipe() to look at all variables
                      data = training(splits)) %>%
  
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
  
  # now we normalize the time and year, because those were big numbers and they look weird.
  step_normalize(ends_with("index.num"),
                 ends_with("_year")) %>%
  
  # now we create dummy variables from categorical data.
  step_dummy(all_nominal())    #all_nominal() selects any features that are character or factor



recipe_spec %>%
  prep() %>%    # this performs pre-calculations and preprocessing training steps.
  juice() %>%   # returns the trianing data post-transformation so we can see the transformations that were applied.
  glimpse()




# * Machine Learning Specs ----

# when you have a recipe and model, you we create a workflow -- here is linear regression.
model_spec <- linear_reg() %>%
  set_engine("lm")


# create a workflow
workflow_fit_lm <- workflow() %>%
  add_model(model_spec) %>%     # first, add the model 
  add_recipe(recipe_spec) %>%   # next, add the recipe
  fit(training(splits))         # last, fit the training(splits)


workflow_fit_lm


# * Modeltime Process ----

# now you can check the models against each other
# Add the prophet and the workflow you just created, and then calibrate the testing data.
calibration_tbl <- modeltime_table(
  model_prophet_fit,
  workflow_fit_lm
) %>%
  modeltime_calibrate(testing(splits))   # already been trained on the training data, but now calibrate on testing data.


# now check accuracy. IMPORTANT: lower means better for these metrics.
calibration_tbl %>%
  modeltime_accuracy()


# then pipe in the modeltime_forecast with new testing data, original evaluation data, and plot it.
calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = sales_monthly_tbl
  ) %>%
  plot_modeltime_forecast()


# FUTURE FORECAST ----

# this is where you are retraining the data
refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = prepared_tbl)



# now you pull in the forecast as new data.
refit_tbl %>%
  modeltime_forecast(new_data = forecast_tbl,
                     actual_data = prepared_tbl) %>%
  
  plot_modeltime_forecast()






