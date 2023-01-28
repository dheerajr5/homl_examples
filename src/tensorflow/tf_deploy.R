library(keras)
library(plumber)
library(jpeg)
library(httr)


model <- load_model_tf("cnn-mnist/")

#* Predicts the number in an image
#* @param enc a base64  encoded 28x28 image
#* @post /cnn-mnist
function(enc) {
  # decode and read the jpeg image
  img <- jpeg::readJPEG(source = base64enc::base64decode(enc))
  
  # reshape
  img <- img %>% 
    array_reshape(., dim = c(1, dim(.), 1))
  
  # make the prediction
  predict_classes(model, img)
}

# run separately
# p <- plumber::plumb("src/tensorflow/deploy_plumber.R")
# p$run(port = 8000)


# plumber request ----
img <- mnist$test$x[1,,,]
mnist$test$y[1]

encoded_img <- img %>% 
  jpeg::writeJPEG() %>% 
  base64enc::base64encode()


req <- httr::POST("http://localhost:8000/cnn-mnist",
                  body = list(enc = encoded_img), 
                  encode = "json")
httr::content(req)

# other options ----
# https://tensorflow.rstudio.com/guides/deploy/shiny

# https://tensorflow.rstudio.com/guides/deploy/docker

# rsconnect::deployTFModel("cnn-mnist/")