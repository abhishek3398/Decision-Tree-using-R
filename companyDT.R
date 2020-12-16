library(C50)
library(caret)
library(gmodels)
library(mlbench)
library(moments)
library(Boruta)
library(psych)
library(DMwR)
library(DMwR2)
library(RANN)
windows()
company_data <- read.csv(choose.files())
View(company_data)
summary(company_data)
str(company_data)

company_data$Sales<-cut(company_data$Sales,breaks = c(0,6,11,17), labels = c("A","B","c"), right = F)
company_data$Sales<-as.factor(as.numeric(company_data$Sales))
str(company_data)

barplot(as.matrix(company_data[-c(7,10,11)]), las = 2)
boxplot(company_data[-c(7,10,11)])
pairs.panels(company_data)

skewness(company_data$Price)
kurtosis(company_data$Price)
skewness(company_data$Population)
kurtosis(company_data$Population)
skewness(company_data$Income)
kurtosis(company_data$Income)
skewness(company_data$Advertising)
kurtosis(company_data$Advertising)

boruta <- Boruta(Sales~., data = company_data, doTrace = 2)
print(boruta)
plot(boruta, las = 2, cex.axis = 0.7)
attStats(boruta)
TentativeRoughFix(boruta)

set.seed(1234)
pd <- sample(2,nrow(company_data), replace = TRUE, prob = c(0.8, 0.2))
company_train <- company_data[pd==1,]
company_test <- company_data[pd==2,]

                     #### preparing model on all variables ####

company_model <- C5.0(company_train,company_train$Sales)
plot(company_model)
pred_train <- predict(company_model, company_train)
confusionMatrix(pred_train, company_train$Sales)

pred_test <- predict(company_model, newdata = company_test)
confusionMatrix(pred_test, company_test$Sales)
CrossTable(company_test$Sales, pred_test)

getConfirmedFormula(boruta)

                 #### preparing model using only important variables ####

company_model_boruta <- C5.0(company_train[-c(5,9,10,11)], company_train$Sales)
plot(company_model_boruta)
pred_boruta_train <- predict(company_model_boruta, company_train)
confusionMatrix(pred_boruta_train, company_train$Sales)

pred_boruta_test <- predict(company_model_boruta, newdata = company_test)
confusionMatrix(pred_boruta_test, company_test$Sales)
CrossTable(company_test$Sales, pred_boruta_test)
