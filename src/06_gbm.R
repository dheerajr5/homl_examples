suppressPackageStartupMessages({
  library(dplyr)
  library(gbm)
  library(h2o) # for a java-based implementation of GBM variants
  library(xgboost)
  library(rsample)
  library(recipes)
  library(pdp)
  library(lime)
})


ames <- readRDS('data/housing_regr.rds')

# needs preprocessing

set.seed(123)  # for reproducibility
response <- "Sale_Price"
split <- initial_split(ames, strata = response)
ames_train <- training(split)
ames_test <- testing(split)


h2o.init(max_mem_size = "4g")
train_h2o <- as.h2o(ames_train)
predictors <- setdiff(colnames(ames_train), response)

# Fit a decision tree to the data:  
# We then fit the next decision tree to the residuals of the previous:  
# Add this new tree to our algorithm:  
# Continue this process until some mechanism (i.e. cross validation) tells us to stop.
# The final model here is a stagewise additive model of individual trees.
# # F_1(x) = y
# residual h_1(x) =  y - F_1(x)
# F_2(x) = F_1(x) + h_1(x)
# f_x = F_1(x) + F_2(x) + ... upto number of trees (may stop early)

.start <- Sys.time()
ames_gbm1 <- gbm(
  formula = Sale_Price ~ .,
  data = ames_train, 
  distribution = "gaussian",  # SSE loss function
  n.trees = 5000, 
  shrinkage = 0.1, # learning rate
  interaction.depth = 3, # depth increases
  n.minobsinnode = 10, # 5 to 15 obs. for a tree
  cv.folds = 10 # 10 fold CV 
  
)
.end <- Sys.time()
print(.end - .start) # 5 mins

# Variables Relative influence less than 5%
model_infl_var <- summary(ames_gbm1) %>% 
  filter(rel.inf < 5)

# find index for number trees with minimum CV error
best <- which.min(ames_gbm1$cv.error)

# get MSE and compute RMSE
sqrt(ames_gbm1$cv.error[best])

gbm.perf(ames_gbm1, method = "cv")
# 1165

pred <- predict(ames_gbm1, n.trees = ames_gbm1$n.trees, ames_test)

# results
caret::RMSE(pred, ames_test$Sale_Price)

# plots ----
par(mar = c(5, 8, 1, 1))
summary(
  ames_gbm1, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

# takes time
summary(
  ames_gbm1, 
  cBars = 10,
  method = permutation.test.gbm, # also can use permutation.test.gbm
  las = 2
)

# http://uc-r.github.io/gbm_regression
# apply LIME ----
model_type.gbm <- function(x, ...) {
  return("regression")
}

predict_model.gbm <- function(x, newdata, ...) {
  pred <- predict(x, newdata, n.trees = x$n.trees)
  return(as.data.frame(pred))
}

local_obs <- ames_test[1:3, ]
explainer <- lime(ames_train, ames_gbm1)
explanation <- explain(local_obs, explainer, n_features = 5)
plot_features(explanation)


# xgboost ----
set.seed(123)
xgb_prep <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_integer(all_nominal()) %>%
  prep(training = ames_train, retain = TRUE) %>%
  juice() %>% select(-Sale_Price)

X <- as.matrix(xgb_prep)
Y <- ames_train$Sale_Price

.start <- Sys.time()
# 2 mins
ames_xgb <- xgb.cv(
  data = X,
  label = Y,
  nrounds = 1000,
  objective = "reg:squarederror",
  early_stopping_rounds = 50, 
  nfold = 10,
  params = list(
    eta = 0.1,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.8,
    colsample_bytree = 1.0),
  verbose = 0
)
.end <- Sys.time()
print(.end - .start)

ames_xgb$evaluation_log %>% 
  arrange(test_rmse_mean) %>% 
  dplyr::slice(10)

ames_xgb$evaluation_log %>% 
  arrange(train_rmse_mean) %>% 
  dplyr::slice(10)

ames_xgb$best_ntreelimit %>% print()

ames_xgb$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_rmse_mean == min(train_rmse_mean))[1],
    rmse.train   = min(train_rmse_mean),
    ntrees.test  = which(test_rmse_mean == min(test_rmse_mean))[1],
    rmse.test   = min(test_rmse_mean),
  )


params <- list(
  eta = 0.01,
  max_depth = 5,
  min_child_weight = 5,
  subsample = 0.65,
  colsample_bytree = 1
)

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = X,
  label = Y,
  nrounds = 1576,
  objective = "reg:squarederror",
  verbose = 0
)

importance_matrix <- xgb.importance(model = xgb.fit.final)

# variable importance plot
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")


# partial dependency plot 
xgb.fit.final %>%
  partial(pred.var = "Gr_Liv_Area", n.trees = 1576, grid.resolution = 100, train = X,
          plot = TRUE)



# tuning gbm ----
# 4 diff models
hyper_grid <- expand.grid(
  n.trees = 6000,
  shrinkage = 0.05,
  interaction.depth = c(3, 5),
  n.minobsinnode = c(5, 10)
)

# create model fit function
model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
  set.seed(123)
  m <- gbm(
    formula = Sale_Price ~ .,
    data = ames_train,
    distribution = "gaussian",
    n.trees = n.trees,
    shrinkage = shrinkage,
    interaction.depth = interaction.depth,
    n.minobsinnode = n.minobsinnode,
    cv.folds = 10
  )
  # compute RMSE
  sqrt(min(m$cv.error))
}

# perform search grid, takes time
# hyper_grid$rmse <- purrr::pmap_dbl(
#   hyper_grid,
#   ~ model_fit(
#     n.trees = ..1,
#     shrinkage = ..2,
#     interaction.depth = ..3,
#     n.minobsinnode = ..4
#   )
# )
# 
# # results
# arrange(hyper_grid, rmse)
