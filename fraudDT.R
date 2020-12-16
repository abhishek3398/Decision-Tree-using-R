library(C50)
library(caret)
library(gmodels)
library(mlbench)
library(moments)
library(psych)
library(DMwR)
library(DMwR2)
library(RANN)
windows()
Fraud_check <- read.csv(choose.files(), stringsAsFactors = TRUE)
View(Fraud_check)
summary(Fraud_check)
str(Fraud_check)

Fraud_check$Taxable.Income <- cut(Fraud_check$Taxable.Income, breaks = c(0,30000,100000), labels = c("risky", "good"))
Fraud_check$Taxable.Income <- as.factor(Fraud_check$Taxable.Income)
str(Fraud_check)

barplot(as.matrix(Fraud_check))
boxplot(Fraud_check)
pairs.panels(Fraud_check)

skewness(Fraud_check$City.Population)
kurtosis(Fraud_check$City.Population)
skewness(Fraud_check$Work.Experience)
kurtosis(Fraud_check$Work.Experience)

model <- train(Taxable.Income~., data = Fraud_check, method = "rpart")
imp <- varImp(model)
print(imp)

set.seed(1234)
pd <- sample(2, nrow(Fraud_check), replace = TRUE, prob = c(0.80, 0.20))
trainfraud <- Fraud_check[pd==1,]
testfraud <- Fraud_check[pd==2,]

fraud_model <- C5.0(trainfraud[,-3], trainfraud$Taxable.Income)
plot(fraud_model)
pred_train <- predict(fraud_model, trainfraud)
confusionMatrix(pred_train, trainfraud$Taxable.Income)

pred_test <- predict(fraud_model, newdata = testfraud)
confusionMatrix(pred_test, testfraud$Taxable.Income)
CrossTable(testfraud$Taxable.Income, pred_test)

sum(is.na(Fraud_check))

preVal <- preProcess(Fraud_check, method = c("medianImpute", "center", "scale"))
data <- predict(preVal, Fraud_check)
sum(is.na(data))

index <- createDataPartition(Fraud_check$Taxable.Income, p=0.75, list = FALSE)
train1 <- Fraud_check[index,]
test1 <- Fraud_check[-index,]

fitControl <- trainControl(method = "cv",  number = 5,  savePredictions = 'final',  classProbs = T)

model_rf <- train(train1[,-3], train1$Taxable.Income, method = 'rf', trControl = fitControl, tuneLength = 3)
pred_rf <- predict(object = model_rf, test1[,-3])
plot(model_rf)
confusionMatrix(test1$Taxable.Income, pred_rf)

model_glm <- train(train1[,-3], train1$Taxable.Income, method = 'glm', trControl = fitControl, tuneLength = 3)
pred_glm <- predict(object = model_glm, test1[,-3])
confusionMatrix(test1$Taxable.Income, pred_glm)

model_gbm <- train(train1[,-3],train1[,3],method='gbm',trControl=fitControl,tuneLength=3)
pred_gbm <- predict(model_gbm, test1[,-3])
confusionMatrix(test1$Taxable.Income, pred_gbm)