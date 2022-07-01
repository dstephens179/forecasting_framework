calibrate_and_plot <-
function(..., type = "testing") {
  
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
