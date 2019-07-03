library(readr)
library(glmnet)

## Helper functions
get_label <- function(x){
  if (x > 0.5){
    return(1)
  }else {
    return(0)
  }  
}

# Load Data
leukemia_big <- read_csv("leukemia_big.csv")
X = t(leukemia_big) 
Y = names(leukemia_big)
Y = gsub("AML.*", 0, Y)
Y = as.integer(gsub("ALL.*", 1, Y))

## 75% of the sample size
## smp_size <- floor(0.75 * nrow(X))

## set the seed to make your partition reproducible
# set.seed(123)
# train_ind <- sample(seq_len(nrow(X)), size = smp_size)

# Split Data into Test and Train
# X_TRAIN=X[train_ind, ]
# Y_TRAIN=Y[train_ind]
# 
# X_TEST=X[-train_ind, ]
# Y_TEST=Y[-train_ind]

X_TRAIN=X[35:72,]
Y_TRAIN=Y[35:72]

X_TEST=X[1:34, ]
Y_TEST=Y[1:34]

# Fit the model and see the inner iterations
fit = glmnet(X_TRAIN, Y_TRAIN, family = "binomial", alpha=0.5)
par(mfrow=c(2,2))
plot(fit, xvar = "dev", label = TRUE)
plot(fit, xvar = "lambda", label = TRUE)
plot(fit)
print(fit)

# Train Model with CV
cv1  = cv.glmnet(X_TRAIN, Y_TRAIN, family = "binomial",type.measure = "class",  nfolds = 10, alpha=1)
cv.5 = cv.glmnet(X_TRAIN, Y_TRAIN, family = "binomial", type.measure = "class", nfolds = 10, alpha=0.5)
# cv0  = cv.glmnet(X_TRAIN, Y_TRAIN, family = "binomial",type.measure = "class",  nfolds = 10, alpha=0)

# PLOT V results
par(mfrow=c(2,2))
plot(cv1);plot(cv.5)
#;plot(cv0)
# plot(log(cv1$lambda),cv1$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cv1$name)
# points(log(cv.5$lambda),cv.5$cvm,pch=19,col="green")
# points(log(cv0$lambda),cv0$cvm,pch=19,col="blue")
# legend("topleft",legend=c("alpha= 1","alpha= .5","alpha 0"),pch=19,col=c("red","grey","blue"))

# Predict on test data and calculate error
# Get the confusion matrix

# Train Error
yhat = predict(cv.5, newx = X_TRAIN, s = "lambda.min", type = "class")
confusionMatrix(as.factor(yhat), as.factor(Y_TRAIN))

# Test Error
yhat = predict(cv.5, newx = X_TEST, s = "lambda.min", type = "class")
confusionMatrix(as.factor(yhat), as.factor(Y_TEST))


yhat = predict(cv.5, newx = X_TEST, s = "lambda.1se", type = "class")
confusionMatrix(as.factor(yhat), as.factor(Y_TEST))


# Know what variables were selected by the model and rank them
# coef(cv.5, s = "lambda.min")
tmp_coeffs <- coef(cv.5, s = "lambda.min")
tmp_coeffs <- data.frame(name = tmp_coeffs@Dimnames[[1]][tmp_coeffs@i + 1], coefficient = tmp_coeffs@x)


##### XGBOOST ######
cols <- sapply(seq(1:dim(X)[2]),function(x) paste("V", toString(x), sep=""))
bst <- xgboost(data = X_TRAIN, 
                  label = Y_TRAIN, 
                  max.depth = 15, 
                  subsample = 0.75,
                  eta = 0.2, 
                  nthread = 2, 
                  nrounds = 10,
                  objective = "binary:logistic")
importance <- xgb.importance(feature_names = cols, model = bst)
xgb.plot.importance(importance)


y_pred <- predict(bst, X_TEST)
yt <- sapply(y_pred, function(x) get_label(x))
confusionMatrix(as.factor(yt), as.factor(Y_TEST))


#### Fit Logistic regression ####
data=data.frame(dna1=X_TRAIN[,2020] ,dna2=X_TRAIN[,1807], label=Y_TRAIN)
m = glm(label ~ dna1 + dna2, data=data, family='binomial')

data=data.frame(dna1=X_TEST[,2020], dna2=X_TEST[,1807], label=Y_TEST)
y = predict(m, data, type = "response")

data=data.frame(dna1=X_TRAIN[,2020], dna2=X_TRAIN[,1807], label=Y_TRAIN)
y = predict(m, data, type = "response")


### Correlations ###
# https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
correlationMatrix <- cor(X)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(highlyCorrelated)
corrplot(correlationMatrix[1000:1005,1000:1005], type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

### Feature Ranking By lvq ###
Yf=as.factor(Y)
df_leukemia=data.frame(X,Yf)
control <- trainControl(method="repeatedcv", number=5, repeats=3)
model <- train(Yf~., data=df_leukemia, method="lvq", preProcess="scale", trControl=control, tuneGrid = data.frame(size = 3, k = 1:2))
importance <- varImp(model, scale=FALSE)
print(importance)
plot(importance)




yt <- sapply(y, function(x) fn(x))
confusionMatrix(as.factor(yt), as.factor(Y_TEST))

confusionMatrix(as.factor(yt), as.factor(Y_TRAIN))
