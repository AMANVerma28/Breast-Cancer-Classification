## Descriptive Statistics
library(needs)
needs(readr,
      dplyr,
      ggplot2,
      corrplot,
      gridExtra,
      pROC,
      MASS,
      caTools,
      caret,
      caretEnsemble,
      doMC)
cancer <- read.csv("BCBDdataet.csv")
cancer$Class <- as.factor(ifelse(cancer$Class > 1, 'Malignant', 'Benign'))
str(cancer)
summary(cancer) # No NA values
y <- cancer$Class
cbind(freq=table(y), percentage=prop.table(table(y))*100)
sapply(cancer[,1:9],sd)
stat.desc(cancer[,1:9], basic=TRUE, desc=TRUE, norm=TRUE, p=0.95)
corr_mat <- cor(cancer[,1:9])
corrplot(corr_mat, order = "hclust", tl.cex = 1, addrect = 8)

## Exploratory Analysis
library(caTools)
set.seed(1000)
split = sample.split(cancer$Class, SplitRatio = 0.80)
train = subset(cancer, split==TRUE)
test = subset(cancer, split==FALSE)

# Box Plots
library(ggplot2)
#ggplot(stack(train[,1:9]), aes(x = ind, y = values)) + geom_boxplot() + 
#  theme(axis.text.x = element_text(angle = 60, hjust = 1, vjust=1)) +
#  labs(title = "Boxplots of columns") + labs(x = "", y = "Values") + 
#  scale_y_continuous(breaks = seq(1, 10, by = 1))

ggplot(stack(train[,1:8]), aes(x = ind, y = values)) + geom_boxplot() + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1, vjust=1)) +
  labs(title = "Boxplots of columns") + labs(x = "", y = "Values") + 
  scale_y_continuous(breaks = seq(1, 10, by = 1))

## Modelling
library(MASS)
library(e1071)
library()

# Normalize variables
normalize <- function(x) {
  return ((x - mean(x)) / (sd(x))) }

train_n <- as.data.frame(lapply(train[1:9], normalize))
train_n$Classification <- train$Classification
train_n$Class <- train$Class

# Creates model and uses 10-fold cross validation
create.model <- function(method, data.to.model){
  library(caret)
  cvCtrl <- trainControl("cv", 5, savePred=T) # 5-fold Cross Validation
  set.seed(123)
  model <- train(Class ~ . - Classification,
                 data = data.to.model,
                 method = method,
                 trControl = cvCtrl)
  model
  print(paste0("Method: ", method, " -- 5-fold CV Accuracy = ", round(100*max(model$results$Accuracy), 3), "%"))
  return(model)
}

models <- lapply(c('glm','rpart','lda','qda','svmLinear'), 
                 function(x) {
                   if(x=='glm'){create.model(method=x, train_n)} # function is included in the appendix
                   else{create.model(method=x, train_n)} # function is included in the appendix
                 })

fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)
# KNN
model_knn <- train(Class~.-Classification,
                   train_n,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10,
                   trControl=fitControl)

pred_knn <- predict(model_knn, test)
cm_knn <- confusionMatrix(pred_knn, test$Class)
cm_knn

# LDA transformations
lda_res <- lda(Classification~., data, center = TRUE, scale = TRUE) # Applying LDA on the data
lda_df <- predict(lda_res, data)$x %>% as.data.frame() %>% cbind(Classification=data$Classification)
lda_res

ggplot(lda_df, aes(x=LD1, y=0, col=Classification)) + geom_point(alpha=0.5)
ggplot(lda_df, aes(x=LD1, fill=Classification)) + geom_density(alpha=0.5)
train_data_lda <- lda_df[data_index,]
test_data_lda <- lda_df[-data_index,]