suppressPackageStartupMessages({
  library(dplyr)
  library(class)
})

data(iris)

tr_index <- sample(1:nrow(iris), 0.8 * nrow(iris))

scale_01 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

iris_norm <- iris %>% 
  mutate_at(.vars = names(iris)[1:4],
            .funs = scale_01) %>% 
  select(-Species)

iris_train <- iris_norm[tr_index, ]
iris_test <- iris_norm[-tr_index, ]
iris_train_label <- iris %>% slice(tr_index) %>% pull(5)
iris_test_label <- iris %>% slice(-tr_index) %>% pull(5)

#' knn from package class
#' 13 neighbor points considered
class_pred <- knn(iris_train, iris_test, cl = iris_train_label, k = 13)

##create confusion matrix
cfm <- table(class_pred, iris_test_label)

#' Accuracy = (TP+TN)/total
accuracy <- function(x) {
  sum(diag(x) / (sum(rowSums(x)))) * 100
}
cat('Accuracy with 13 neighbor points:', accuracy(cfm))


# 5 neighbors
class_pred1 <- knn(iris_train, iris_test, cl = iris_train_label, k = 5)

cfm1 <- table(class_pred1, iris_test_label)

accuracy <- function(x) {
  sum(diag(x) / (sum(rowSums(x)))) * 100
}
cat('Accuracy with 5 neighbor points:', accuracy(cfm))
