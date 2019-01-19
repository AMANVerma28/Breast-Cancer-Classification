## Descriptive Statistics
# Importing the libraries needed for the study
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

data <- read.csv('Documents/workspace/misc/dataR2.csv') # Reading the data into a dataframe
data$Class <- as.factor(ifelse(data$Classification > 1, 'Malignant', 'Benign')) # Creating a new column for the classification
data$Classification <- data$Class# 1:Benign 2:Malignant
data$Class <- NULL
str(data) # Checking column names in the dataframe
summary(data) # Summary of the dataframe

prop.table(table(data$Classification)) # Checking the no of instances of each class

# Correlation between variables
corr_mat <- cor(data[,1:9]) # Calculating correlation
corrplot(corr_mat, order = "hclust", tl.cex = 1, addrect = 8) # Plotting correlation graph

# Box plots
par(mfrow=c(1,2)) # Plotting 2 boxplots side by side
boxplot(data$Age, xlab = "Age")
boxplot(data$BMI, xlab = "BMI")
boxplot(data$Glucose, xlab = "Glucose")
boxplot(data$Insulin, xlab = "Insulin")
boxplot(data$HOMA, xlab = "HOMA")
boxplot(data$Leptin, xlab = "Leptin")
boxplot(data$Adiponectin, xlab = "Adiponectin")
boxplot(data$Resistin, xlab = "Resistin")

par(mfrow=c(1,1)) # Plotting only 1 boxplot in one frame
boxplot(data$MCP.1, xlab = "MCP-1")

# Train-Test split
library(caTools)
set.seed(1000)
split = sample.split(data$Classification, SplitRatio = 0.80)# Dividing dataframe into train and test sets
train = subset(data, split==TRUE)# 80% of data is for training
test = subset(data, split==FALSE)# 20% of data is for testing

# PCA
library(ggbiplot) # library to plot PCA biplots
pca_res <- prcomp(data[,1:9], center = TRUE, scale = TRUE) # Calculating Principal Components
plot(pca_res, type="l") # Plotting Principal components
summary(pca_res)

ggbiplot(pca_res)

# Control Parameters for training
fitControl <- trainControl(method="cv", # CV stands for Cross-Validaton
                           number = 5, # 5-fold CV
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

## Modeling
# LDA
model_lda <- train(Classification~., 
                   train,
                   method="lda2", # Applying LDA classification
                   #tuneLength = 10,
                   metric="ROC", # ROC for evaluation metric
                   preProc = c("center", "scale"),
                   trControl=fitControl)
pred_lda <- predict(model_lda, test) # Predicting classes using test data
cm_lda <- confusionMatrix(pred_lda, test$Classification) # Creating Confusion matrix
cm_lda
pred_prob_lda <- predict(model_lda, test, type="prob") # Predicting class probabilities using test data
#colAUC(pred_prob_lda, test$Classification, plotROC=TRUE)
roc_lda <- roc(test$Classification, pred_prob_lda$Benign) # Calculating ROC
plot(roc_lda) # Plotting ROC curve

# KNN
model_knn <- train(Classification~.,
                   train,
                   method="knn", # Applying KNN classification
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10,
                   trControl=fitControl)
pred_knn <- predict(model_knn, test) # Predicting classes using test data
cm_knn <- confusionMatrix(pred_knn, test$Classification) # Creating Confusion matrix
cm_knn
pred_prob_knn <- predict(model_knn, test, type="prob") # Predicting class probabilities using test data
roc_knn <- roc(test$Classification, pred_prob_knn$Benign) # Calculating ROC
plot(roc_knn) # Plotting ROC curve

# SVM-Radial
model_svm <- train(Classification~.,
                   train,
                   method="svmRadial", # Applying SVM classification using Radial Kernel
                   metric="ROC",
                   preProcess=c('center', 'scale'),
                   trace=FALSE,
                   trControl=fitControl)
pred_svm <- predict(model_svm, test)
cm_svm <- confusionMatrix(pred_svm, test$Classification)
cm_svm
pred_prob_svm <- predict(model_svm, test, type="prob")
roc_svm <- roc(test$Classification, pred_prob_svm$Benign)
plot(roc_svm)

# Decision tree
model_dt <- train(Classification~.,
                   train,
                   method="rpart", # Applying DT classification
                   metric="ROC",
                   preProcess=c('center', 'scale'),
                   parms=list(split='information'),
                   trControl=fitControl)
pred_dt <- predict(model_dt, test)
cm_dt <- confusionMatrix(pred_dt, test$Classification)
cm_dt
pred_prob_dt <- predict(model_dt, test, type="prob")
roc_dt <- roc(test$Classification, pred_prob_dt$Benign)
plot(roc_dt)

# QDA
model_qda <- train(Classification~.,
                   train,
                   method="qda", # Applying QDA classification
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   #tuneLength=10,
                   trControl=fitControl)
pred_qda <- predict(model_qda, test)
cm_qda <- confusionMatrix(pred_qda, test$Classification)
cm_qda
pred_prob_qda <- predict(model_qda, test, type="prob")
roc_qda <- roc(test$Classification, pred_prob_qda$Benign)
plot(roc_qda)

# Logistic Regression
model_lr <- train(Classification~BMI + Glucose + Resistin,
                   train,
                   method="glm", # Applying LR classification
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   #tuneLength=10,
                   family="binomial",
                   trControl=fitControl)
pred_lr <- predict(model_lr, test)
cm_lr <- confusionMatrix(pred_lr, test$Classification)
cm_lr
pred_prob_lr <- predict(model_lr, test, type="prob")
roc_lr <- roc(test$Classification, pred_prob_lr$Benign)
plot(roc_lr)

# Model Comparison
model_list <- list(KNN=model_knn, LDA=model_lda, SVM=model_svm, DT=model_dt, QDA=model_qda, LR=model_lr)
resamples <- resamples(model_list)
model_cor <- modelCor(resamples) # Calculation correlation between models
corrplot(model_cor) # Plotting correlation between models
# bwplot(resamples, metric="ROC")

cm_list <- list(KNN=cm_knn, LDA=cm_lda, QDA=cm_qda, SVM=cm_svm, DT=cm_dt, LR=cm_lr)
cm_list_results <- sapply(cm_list, function(x) x$byClass) # Calculating Confusion matrix for all models
cm_list_results
cm_results_max <- apply(cm_list_results, 1, which.max) # Calculating maximum of values from the confusion matrix
output_report <- data.frame(metric=names(cm_results_max), 
                            best_model=colnames(cm_list_results)[cm_results_max],
                            value=mapply(function(x,y) {cm_list_results[x,y]}, 
                                         names(cm_results_max), 
                                         cm_results_max))
rownames(output_report) <- NULL
output_report

# Plotting ROC curves of all models
plot(roc_dt)
plot(roc_knn, add=TRUE, col="red")
plot(roc_lda, add=TRUE, col="blue")
plot(roc_lr, add=TRUE, col="green")
plot(roc_qda, add=TRUE, col="purple")
plot(roc_svm, add=TRUE, col="pink")
legend("bottomright", legend=c("DT", "KNN", "LDA", "LR", "QDA", "SVM"),
       col=c("black", "red", "blue", "green", "purple", "pink"), lty=c(1,1,1,1,1,1), lwd=c(2.5,2.5,2.5,2.5,2.5,2.5))