# pkgs for data
# "modeldata", "rsample", "dslabs"

options(warn = -1)
suppressPackageStartupMessages({
  library(dplyr)     # for data manipulation
  library(ggplot2)   # for awesome graphics
  
  # Modeling process packages
  library(rsample)   # for resampling procedures
  library(caret)     # for resampling and model training
  #library(h2o)
})

# h2o set-up 
#h2o.no_progress()  # turn off h2o progress bars
#h2o.init()         # launch h2o
#h2o.shutdown()

# regression
ames <- modeldata::ames

# classification
attrition <- modeldata::attrition

# multinomial classn.
mnist <- dslabs::read_mnist()

# unsupervised classn.
my_basket <- readr::read_csv("https://koalaverse.github.io/homlr/data/my_basket.csv")

saveRDS(object = ames, file = 'data/housing_regr.rds')
saveRDS(object = attrition, file = 'data/attrition_cls.rds')
saveRDS(object = mnist, file = 'data/mnist_digit_recogn_multinml.rds')
saveRDS(object = my_basket, file = 'data/basket_clust.rds')
