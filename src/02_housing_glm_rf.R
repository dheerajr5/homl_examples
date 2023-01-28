
library('dplyr')
library('caret')
data <- readRDS('data/housing_regr.rds')

data$Sale_Price <- data$Sale_Price / 10e4
data_fact <- mltools::one_hot(data.table::data.table(data))

trim_data_fact_cols <- lapply(data_fact, FUN = function(x) {
  if (sum(unique(x)) == 1)
    if (!(max(table(x)) > nrow(data) * 0.8))
      TRUE
})
trim_col_names <- names(unlist(trim_data_fact_cols) == TRUE)

data <- data_fact %>% as_tibble() %>% 
  select(-trim_col_names)


data_nas <- lapply(data, FUN = function(x) {
  sum(is.na(x))
})

if (all(data_nas == 0)) {
  data_train_index <- createDataPartition(data$Sale_Price, p = 0.7,
                                          list = FALSE)
  data_train_df <- data[data_train_index,]
  data_test_df <- data[-data_train_index,]
  
  train_x <- data_train_df %>% select(-Sale_Price)
  train_y <- data_train_df[['Sale_Price']]
  test_x <- data_test_df %>% select(-Sale_Price)
  test_y <- data_test_df[['Sale_Price']]
  
} else stop('NAs present in column', call. = FALSE)

# GLM ----------

model_glm <- glm(Sale_Price ~ ., 
                 data = data_train_df, 
                 family = "gaussian")


glm_summary <- summary(model_glm)
test_glm <- predict.glm(model_glm, newdata = data_test_df, type = "response")

# GLM cv ----
train_ctl <- trainControl(method = 'cv', number = 5)

glm_tune <- train(train_x, 
                  train_y, 
                  method = "glm",
                  trControl = train_ctl)
glm_tune$results

library(randomForest)

rf_fit <- randomForest(x = train_x, y = train_y,
                       xtest = test_x, ytest = test_y, 
                       ntree = 500, mtry = ceiling(ncol(train_x)/3), 
                       importance = TRUE)

rf_fit

importance(rf_fit)
