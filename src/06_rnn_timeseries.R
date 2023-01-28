# dir.create("jena_climate", recursive = TRUE)
# download.file(
#   "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
#   "jena_climate/jena_climate_2009_2016.csv.zip"
# )
# unzip(
#   "jena_climate/jena_climate_2009_2016.csv.zip",
#   exdir = "jena_climate"
# )


library(tibble)
library(readr)

data_dir <- "jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)

data <- data.matrix(data[,-1])
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)


generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] - 1 + delay,2]
    }           
    list(samples, targets)
}

lookback <- 1440 # extract 10 days of data
step <- 6 # one point per 10 min
delay <- 144 # target points 24hr data
batch_size <- 128

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

# How many steps to draw from val_gen in order to see the entire validation set
val_steps <- (300000 - 200001 - lookback) / batch_size

# How many steps to draw from test_gen in order to see the entire test set
test_steps <- (nrow(data) - 300001 - lookback) / batch_size


library(keras)
evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method()

celsius_mae <- 0.3069 * std[[2]] + mean[[2]]


model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

# throws error
# Layer "sequential_1" expects 1 input(s), but it received 2 input tensors. 
history <- model %>% fit(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)


# https://www.appsloveworld.com/coding/python3x/72/valueerror-layer-sequential-expects-1-inputs-but-it-received-211-input-tensors
# x = np.stack([i.tolist() for i in x])


# z1 <- keras::keras_array(x = train_gen)
# z2 <- keras::keras_array(x = val_gen)
# history <- model %>% fit(
#   z1,
#   steps_per_epoch = 500,
#   epochs = 20,
#   validation_data = z2,
#   validation_steps = val_steps
# )

# train_gen[[1]][1,,14] %>% glimpse()
# train_gen[[2]] %>% glimpse()
# val_gen %>% glimpse()
# 
# # np <- reticulate::import("numpy", convert=FALSE)
# # z1 <- np$stack(train_gen)
# 
# history <- model %>% fit(
#   train_gen[[1]][,,2], train_gen[[2]],
#   steps_per_epoch = 500,
#   epochs = 20,
#   validation_data = list(val_gen[[1]][,,2], 
#                          val_gen[[2]]),
#   validation_steps = val_steps
# )
# 
