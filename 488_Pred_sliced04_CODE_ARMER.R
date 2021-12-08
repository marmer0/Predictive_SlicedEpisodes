##This is all Madeleine Armers code for Predictive F21 sliced 04 episode 

library(tidyverse)
library(caret)
library(mice)
library(naniar)
train <- read.csv("~/Desktop/train4.csv")
test <- read.csv("~/Desktop/test4.csv")

train%>% 
  gg_miss_var(show_pct = TRUE)
trainimp <- train[,-c(1:3, 7,8)] #getting rid of id and other variables that i dont need and evaporation and sunshine as almost 98% of the data is missing 
imp <- mice(trainimp, m=1)
trainfull <- complete(imp)
train <- na.omit(trainfull) #omitting the data that was no able to be imputed by the imputation method

train$rain_tomorrow <- as.factor(train$rain_tomorrow)
train$rain_today <- as.factor(train$rain_today)

test%>% 
  gg_miss_var(show_pct = TRUE)

testimp <-  test[, -c(1:3,7,8)]


imp <- mice(testimp, m=1)
testfull <- complete(imp)
#still missing alot of data 

test1 <- testfull[,-c(4,6,7)]

library(GGally)
##STARTING EDA 


eda <- train %>% 
  subset(select = c("rain_tomorrow", "rainfall", "wind_gust_speed", "wind_speed3pm", "humidity3pm", "cloud3pm", "temp3pm", "pressure3pm")) 

ggpairs(eda, aes(color = rain_tomorrow)) + theme_bw()


#does wind gust speed indicate a storm coming? 
ggplot(aes(y =wind_gust_speed , x =rain_tomorrow, group=rain_tomorrow), data = train) + geom_boxplot(aes(fill = rain_tomorrow), data = train) + ggtitle("Wind gust speed for Rain Tomorrow")
##it does indeed look like higher wind speeds do correlate with higher chances of rain the next day 



mosaicplot(table(raintoday = train$rain_today, raintomorrow = train$rain_tomorrow), main = "Mosaic plot of rain tomorrow  vs. rain today")
##looks like rain today = 0 usually indicates that it will not rain tomorrow 

ggplot(aes(y =humidity3pm , x =rain_tomorrow, group=rain_tomorrow), data = train) + geom_boxplot(aes(fill= rain_tomorrow), data = train) + ggtitle("Humidity at 3pm for Rain Tomorrow")
##wow so humidity@3pm is very important 

library(tidymodels)
set.seed(100)
split <- initial_split(train, prop = 0.8)
##gives us even levels of our response variable in our split 
train1 <- training(split) 
holdout <- testing(split)

test$rain_today <- as.factor(test$rain_today)

library(randomForest)

library(doParallel)
set.seed(123)
library(caret)
#10 folds repeat 2 times
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=2)

set.seed(123)
#Number randomly variable selected is mtry
tunegrid <- expand.grid(.mtry = 3:9)
registerDoParallel(cores = 9)
rf_cv <- train(rain_tomorrow~., 
               data=train1, 
               method='rf', 
               metric='Accuracy', 
               tuneGrid=tunegrid, 
               trControl=control)
rf_cv
### Random Forest 
# 
# 23922 samples
#    18 predictor
#     2 classes: '0', '1' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 2 times) 
# Summary of sample sizes: 21530, 21531, 21530, 21530, 21529, 21531, ... 
# Resampling results across tuning parameters:
# 
#   mtry  Accuracy   Kappa    
#   3     0.8450592  0.4733314
#   4     0.8479644  0.4950357
#   5     0.8494483  0.5042080
#   6     0.8490930  0.5051382
#   7     0.8490929  0.5070882
#   8     0.8491140  0.5078922
#   9     0.8492603  0.5095269
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 5.

rf <- randomForest(rain_tomorrow ~., data = train1, ntree = 1000, mtry = 5)
rf
plot(rf)
varImpPlot(rf)
prednull <- predict(rf, holdout)
(accrf <- sum(prednull == holdout$rain_tomorrow)/length(holdout$rain_tomorrow))
table(prednull, holdout$rain_tomorrow)

library(leaps)
set.seed(123)
select <- regsubsets(rain_tomorrow ~ ., data = train, method = "backward")
selectsum <- summary(select)
which.min(selectsum$cp)
coefficients(select, id=8)

library(randomForest)
rf <- randomForest(rain_tomorrow ~ wind_gust_speed + humidity3pm + pressure9am + cloud3pm + rain_today + wind_speed3pm , data = train, ntree = 800, mtry = 5)
rf
plot(rf)
test1$rain_today <- as.factor(test1$rain_today) 
finalpred <- predict(rf, test1)

write.csv(data.frame(id = test$id, rain_tomorrow = finalpred),
  "~/Desktop/finalpredssliced04.csv", row.names = FALSE)



rffull <- randomForest(rain_tomorrow ~ min_temp+ max_temp+ rainfall+ wind_gust_speed + wind_speed9am + wind_speed3pm+ humidity9am+ humidity3pm + pressure9am + pressure3pm+ cloud9am+ cloud3pm + temp9am + temp3pm + rain_today  , data = train, ntree = 800, mtry = 5)

fullpreds <- predict(rffull, test1)

write.csv(data.frame(id = test$id, rain_tomorrow = fullpreds),
"~/Desktop/finalpredssliced04_fullrf.csv", row.names = FALSE)


preds <- data.frame(fullpreds, finalpred)

ggplot(aes(x = finalpred), data = preds) + geom_density(aes(fill=finalpred), alpha = 0.5)  + ggtitle( " Density of Regsubset RF Model Preds Grouped by Prediction of Rain Tomorrow ")

ggplot(aes(x = fullpreds), data = preds) + geom_density(aes(fill=fullpreds), alpha = 0.5)  + ggtitle( " Density of Full RF Model Preds Grouped by Prediction of Rain Tomorrow ")





