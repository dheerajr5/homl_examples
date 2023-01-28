library(tensorflow)
library(tfdatasets)
# save and load -----
MyModule(tf$Module) %py_class% {
  initialize <- function(self, value) {
    self$weight <- tf$Variable(value)
  }
  
  multiply <- tf_function(function(self, x) {
    x * self$weight
  })
}

mod <- MyModule(3)
mod$multiply(as_tensor(c(1, 2, 3), "float32"))

save_path <- tempfile()
tf$saved_model$save(mod, save_path)

view_savedmodel(model_dir = save_path)

mod_l <- tf$saved_model$load(export_dir = save_path)
# dtype should be same as line 13
mod_l$multiply(as_tensor(c(1:3), dtype = 'float32'))

# read inputs

titanic_file <- get_file(
  fname = "train.csv", 
  origin = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
)
df <- readr::read_csv(titanic_file)

#mtcars_spec <- csv_record_spec("mtcars.csv", types = "dididddiiii")


titanic_slices <- tfdatasets::tensor_slices_dataset(df)

# 1st row
titanic_slices %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next()

titanic_lines <- text_line_dataset(titanic_file)
epochs <- 3
dataset <- titanic_lines %>% 
  dataset_batch(128)

# model ----
c(train, test) %<-% dataset_fashion_mnist()

fmnist_train_ds <- train %>% 
  tensor_slices_dataset() %>% 
  dataset_map(unname) %>% 
  dataset_shuffle(5000) %>% 
  dataset_batch(32)

model <- keras_model_sequential() %>% 
  layer_rescaling(scale = 1/255) %>% 
  layer_flatten() %>% 
  layer_dense(10)

model %>% compile(
  optimizer = 'adam',
  loss = loss_sparse_categorical_crossentropy(from_logits = TRUE), 
  metrics = 'accuracy'
)
model %>% fit(
  fmnist_train_ds %>% dataset_repeat(), 
  epochs = 2, 
  steps_per_epoch = 20
)

model %>% evaluate(fmnist_train_ds)

predict_ds <- tensor_slices_dataset(train$x) %>% 
  dataset_batch(32)
result <- predict(model, predict_ds, steps = 10)
dim(result)

# visualize runs ----
library(tfruns)
tfruns::training_run("src/tensorflow/mnist_mlp.R",
                     run_dir = 'src/tensorflow/')
latest_run()

# Code tab includes all open files in editor
tfruns::view_run()

# save_run_view(run_dir = 'runs/2023-01-23T11-50-56Z/',
#               filename = 'report.html')

copy_run(ls_runs(eval_acc >= 0.9), to = "best-runs")
clean_runs() # archives all runs in the "runs" directory
purge_runs() # permanently delete


# run random sample (0.3) of dropout1 and dropout2 combinations
runs <- tuning_run("src/tensorflow/mnist_mlp.R", sample = 0.3, flags = list(
  dropout1 = c(0.2, 0.3, 0.4),
  dropout2 = c(0.2, 0.3, 0.4)
))


tensorboard(ls_runs(latest_n = 2))         # last 2 runs
