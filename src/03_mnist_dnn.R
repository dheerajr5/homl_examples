# Deep Neural network using keras for multinomial classfn.
# of digit recognization

#' requires python interpreter configured and 
#' tensorflow installed for python.
#' Check for python keras
keras::is_keras_available()

.start <- Sys.time()

suppressWarnings({
  suppressPackageStartupMessages({
    library(keras)
    library(tfruns)
    library(dplyr)
  })
})

mnist <- dslabs::read_mnist()

mnist_x <- mnist$train$images #60K * 784
mnist_y <- mnist$train$labels

colnames(mnist_x) <- paste0("V", 1:ncol(mnist_x))
mnist_x <- mnist_x / 255 # standardize feature values

# One-hot encode response
mnist_y <- to_categorical(mnist_y, 10)
mnist_y %>% glimpse()

# param (ncol(mnist_x)+1)*128
# simple keras model ----
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

fit1 <- model %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 25,
    batch_size = 128,
    validation_split = 0.2,
    verbose = FALSE
  )

plot(fit1)

.end <- Sys.time()
print(.end - .start)

# Batch normalization ----
.start <- Sys.time()
model_w_norm <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dense(units = 10, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )

fit2 <- model_w_norm %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 25,
    batch_size = 128,
    validation_split = 0.2,
    verbose = FALSE
  )

plot(fit2)


.end <- Sys.time()
print(.end - .start)

# L2 Ridge regularization ----
model_w_reg <- keras_model_sequential() %>%
  
  # Network architecture with L1 regularization and batch normalization
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x),
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 128, activation = "relu", 
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 64, activation = "relu", 
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 10, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )

fit3 <- model_w_reg %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 25,
    batch_size = 128,
    validation_split = 0.2,
    verbose = FALSE
  )
plot(fit3)


# adjust alpha Line 151 ----
model_w_adj_lrn <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  ) %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 35,
    batch_size = 128,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(factor = 0.05)
    ),
    verbose = FALSE
  )


min(model_w_adj_lrn$metrics$val_loss)
max(model_w_adj_lrn$metrics$val_acc)

# Learning rate
plot(model_w_adj_lrn)



# !Grid search ----
# this may run for more than two hours
# runs <- tuning_run("src/04_mnist_dnn_grid.R", 
#                    flags = list(
#                      nodes1 = c(64, 128, 256),
#                      nodes2 = c(64, 128, 256),
#                      nodes3 = c(64, 128, 256),
#                      dropout1 = c(0.2, 0.3, 0.4),
#                      dropout2 = c(0.2, 0.3, 0.4),
#                      dropout3 = c(0.2, 0.3, 0.4),
#                      optimizer = c("rmsprop", "adam"),
#                      lr_annealing = c(0.1, 0.05)
#                    ),
#                    sample = 0.05
# )
# 
# runs %>% 
#   filter(metric_val_loss == min(metric_val_loss)) %>% 
#   glimpse()

