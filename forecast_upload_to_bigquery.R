library(bigrquery)


# store the dataset
datasetid <- "source-data-314320.joyeria_dataset.forecast"



# bind_rows is great. This let's you append the tidy_data created in each store's forecast
tidy_fcast <- bind_rows(tidy_Centro, tidy_Segovia, tidy_Patria, tidy_Lupe, tidy_Vallardo)

# using which() to select only "forecast" rows in the key column
tidy_subset <- tidy_fcast[which(tidy_fcast$key == "forecast"), names(tidy_fcast)]

# increase subset forecast by +0%.
# This is global. If you want to change on the store-level, you will have to create it.
tidy_subset$total_sales <- tidy_subset$total_sales*1.00


# use bigrquery to upload/overwrite the dataset
bq_perform_upload(datasetid,
                  tidy_subset,
                  nskip = 0,
                  source_format = "CSV",
                  create_disposition = "CREATE_IF_NEEDED",
                  write_disposition = "WRITE_TRUNCATE")

