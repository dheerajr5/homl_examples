suppressPackageStartupMessages({
  library(dplyr)    # for data wrangling
  library(ggplot2)  # for awesome graphics
  library(rsample)  # for data splitting
  library(caret)    # for classification and regression training
  library(kernlab)  # for fitting SVMs
  
  # Model interpretability packages
  library(pdp)   
})

attrition <- readRDS('data/attrition_cls.rds')
df <- attrition %>% 
  mutate_if(is.ordered, factor, ordered = FALSE)

# Create training (70%) and test (30%) sets
set.seed(123)  # for reproducibility
churn_split <- initial_split(df, prop = 0.7, strata = "Attrition")
churn_train <- training(churn_split)
churn_test  <- testing(churn_split)


# svm ----
# p=2, this defines a line in 2-D space, and when  
# p=3, it defines a plane in 3-D space

# HMC ----
#' hard margin classifier
#' HMC estimates the coefficients of the hyperplane 
#' by solving a quadratic programming problem with 
#' linear inequality constraints
# \begin{align}
# &\underset{\beta_0, \beta_1, \dots, \beta_p}{\text{maximize}} \quad M \\
# &\text{subject to} \quad \begin{cases} \sum_{j = 1}^p \beta_j^2 = 1,\\ y_i\left(\beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip}\right) \ge M,\quad  i = 1, 2, \dots, n \end{cases} 
# \end{align}
# (beta_0 ... beta_p)^2 = 1
# y_i * (beta_0 ... beta_p) <= M

# does not allow any points to be on the 
# wrong side of the margin

# SMC ----
# soft margin classifier

# \begin{align}
# &\underset{\beta_0, \beta_1, \dots, \beta_p}{\text{maximize}} \quad M \\
# &\text{subject to} \quad \begin{cases} \sum_{j = 1}^p \beta_j^2 = 1,\\ y_i\left(\beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip}\right) \ge M\left(1 - \xi_i\right), \quad i = 1, 2, \dots, n\\ \xi_i \ge 0, \\ \sum_{i = 1}^n \xi_i \le C\end{cases} 
# \end{align}

# By varying C, we allow points to violate 
# the margin which helps make the SVM robust to outliers

# C=0 (the HMC) allow very few points (outlier) inside margin

# kernel functions in SVM
# polynomial
# radial bias
# hyperbolic tangent

# only support two classes
# approaches: one-versus-all (OVA) and one-versus-one (OVO)


class.weights = c("No" = 1, "Yes" = 10)

churn_svm <- train(
  Attrition ~ ., 
  data = churn_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

churn_svm$results

ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary  # also needed for AUC/ROC
)

# Tune an SVM
set.seed(5628)  # for reproducibility
churn_svm_auc <- train(
  Attrition ~ ., 
  data = churn_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  metric = "ROC",  # area under ROC curve (AUC)       
  trControl = ctrl,
  tuneLength = 10
)

churn_svm_auc$modelInfo

confusionMatrix(churn_svm_auc)


features <- c("OverTime", "WorkLifeBalance", 
              "JobSatisfaction", "JobRole")
pdps <- lapply(features, function(x) {
  partial(churn_svm_auc, pred.var = x, which.class = 2,  
          prob = TRUE, plot = TRUE, plot.engine = "ggplot2") +
    coord_flip()
})

pdps

train_svm <- e1071::svm(formula = Attrition ~ ., data = churn_train)

train_svm$SV %>% as_tibble()

# 1 to 256 +ve
train_svm$coefs

train_svm$kernel
