library(tidyverse)
library(caret)
library(naniar)
library(lubridate)

trainfull <- read.csv("~/Desktop/train5.csv")
testfull <- read.csv("~/Desktop/test5.csv")


trainfull%>% 
  gg_miss_var(show_pct = TRUE)


trainsub <- trainfull[,-c(1:4,6)]

diffdays <- difftime("2021-06-28", trainsub$last_review, units = "days") #thats the date when the episode aired 
trainsub$days_since_review <- as.numeric(diffdays)

library(mice)
set.seed(100)
imp <-mice(trainsub, m=1)
train <- complete(imp)
train1 <- train[,-8]




library(tidymodels)
set.seed(100)

split <- initial_split(train1, prop = 0.8)
##gives us even levels of our response variable in our split 
train <- training(split) #takes in the split and spits out the data 
holdout <- testing(split)

##test data: 


testfull%>% 
  gg_miss_var(show_pct = TRUE)

diffdaystest <- difftime("2021-06-28", testfull$last_review, units = "days")
testfull$days_since_review <- as.numeric(diffdaystest)


set.seed(100)
test1 <- testfull[,-c(1:4,6,12)]


imp <-mice(test1, m=1)
test <- complete(imp)



library(GGally)
##STARTING EDA 

eda <- train %>% subset(select = c( "price", "neighbourhood_group" ,"room_type",  "days_since_review", "availability_365"))

ggpairs(eda) + theme_bw()




ggplot(aes(x =price ), data = train) + geom_density()
#lets zoom in a little bc lets be honest, we arent going to rent a hotel more than 500 a night 

subprice <- subset(train, train$price < 750) 
data <- as.data.frame(subprice)
ggplot(aes(x =price ), data = data) + geom_density(aes(fill = neighbourhood_group), data = data) + facet_wrap(~neighbourhood_group)

ggplot(aes(x =price ), data = data) + geom_density(aes(fill = room_type), data = data) + facet_wrap(~room_type)
ggplot(aes(y =price, x = days_since_review ), data = data) + geom_point(aes(color = room_type), data = data) + geom_smooth() + facet_wrap(~neighbourhood_group)
library(glmnet)
train$neighbourhood_group <- as.numeric(as.factor(train$neighbourhood_group))
train$room_type <- as.numeric(as.factor(train$room_type))

holdout$neighbourhood_group <- as.numeric(as.factor(holdout$neighbourhood_group))
holdout$room_type <- as.numeric(as.factor(holdout$room_type))

trainx <- as.matrix(train[, -5])
holdoutx <- as.matrix(holdout[, -5])
trainy <- train[, 5]
holdouty<- holdout[,5]

set.seed(413)
lassocv <- cv.glmnet(trainx, trainy, alpha = 1, scale = T)
lassocv

library(coefplot)
set.seed(123) 
lasso <- glmnet(trainx, trainy, alpha =1, lambda = lassocv$lambda.min, standardize = T)
(coeffs <- coef(lasso))

#calculated host listing count, days since review, availability 365 are basically 0. 

library(leaps)
set.seed(123)
select <- regsubsets(price ~ ., data = train, method = "backward")
selectsum <- summary(select)
which.min(selectsum$cp)
coefficients(select, id=8) 
#reviews per month and calculated host listing not included 

library(doParallel)

gcontrol <- trainControl(
  method = "repeatedcv",
  number = 10, 
  repeats = 2)

gridc <-  expand.grid(.mtry = seq(3:9))

set.seed(143)
registerDoParallel(cores = 9)
rf_cv <- train(price~., data = train,
               method = "rf",
               trControl = gcontrol,
               tuneGrid = gridc)
rf_cv$bestTune 
#mtry = 3 


library(randomForest)

rfsub <- randomForest(price ~ neighbourhood_group + latitude + longitude + room_type + minimum_nights + number_of_reviews + days_since_review, data = train, importance = T, mtry = 3)
rfsub 
plot(rfsub)
rfsubpred <- predict(rfsub, holdout)

mserfsub  <- sqrt(sum((log(rfsubpred + 1)) - log(holdout$price + 1))^2 / length(holdout))
mserfsub

library(randomForest)
rfsub1 <- randomForest(price ~ neighbourhood_group + latitude + longitude + room_type + minimum_nights + number_of_reviews + days_since_review, data = train1, importance = T, mtry = 3)

finalpredsub <- predict(rfsub1, test)

#write.csv(data.frame(id = testfull$id, price = finalpredsub),
#"~/Desktop/finalpredssliced05_rfsub.csv", row.names = FALSE)
rffull <- randomForest(price ~., data = train, importance = T, mtry = 3)
rffull
varImpPlot(rffull)
plot(rffull)
rfpred <- predict(rffull, holdout)
mserf <- sqrt(sum((log(rfpred + 1)) - log(holdout$price + 1))^2 / length(holdout))
mserf
rffull1 <- randomForest(price ~., data = train1, importance = T, mtry = 3)

finalpredfull <- predict(rffull1, test)
#write.csv(data.frame(id = testfull$id, price = finalpredfull),"~/Desktop/finalpredssliced05_rffull.csv", row.names = FALSE)

preds <- data.frame(finalpredfull, finalpredsub, test$neighbourhood_group)

ggplot(aes(x =finalpredfull ), data = preds) + geom_density(aes(fill = test.neighbourhood_group), data = preds) + facet_wrap(~test.neighbourhood_group) + ggtitle("Final Predictions from Kitchen Sink Random Forest")

ggplot(aes(x =finalpredsub ), data = preds) + geom_density(aes(fill = test.neighbourhood_group), data = preds) + facet_wrap(~test.neighbourhood_group)  + ggtitle("Final Predictions from Selected Random Forest")
#selected 
