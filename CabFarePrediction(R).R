#clearing all objects
rm(list=ls(all=T))

#setting the working directory
setwd("E:/Data Science_ Edwisor/Cab Fare Prediction")

#checking the path
getwd()

#loading required libraries
x = c("ggplot2", "DMwR", "corrgram", "Hmisc", "rpart", "randomForest", "geosphere")
install.packages(x)
lapply(x,require,character.only=TRUE)
rm(x)

#we will now load the given train data
train = read.csv("train_cab.csv",header=T)[,-2]

#we wil now observe the structure of train data
str(train)

#view the train data
View(train)

#conveting the data type of fare_amount and passenger_count
train$fare_amount = as.numeric(as.character(train$fare_amount))
train$passenger_count= as.integer(train$passenger_count)

#we will now eliminate the cells with same pickup and dropoff location
train = subset(train,!(train$pickup_longitude==train$dropoff_longitude & train$pickup_latitude==train$pickup_latitude))

#substituting the o's present in the train dataset with NA
train[train==0]=NA

#Missing Value Analysis
missingvalue= function(data){
  missing_value = data.frame(apply(data, 2 , function(x){sum(is.na(x))}))
  colnames(missing_value)="Missing_Value_frequency"
  missing_value$percentage=apply(missing_value , 1 , function(x){x/nrow(train)*100})
  missing_value = cbind(row.names(missing_value), missing_value)
  row.names(missing_value)=NULL
  colnames(missing_value)[1]="Variables"
  print(missing_value)
  
  
  #plot Missing Values
  library(ggplot2)
  ggplot(data = missing_value, aes(x=reorder(Variables , -percentage),y = percentage))+
    geom_bar(stat = "identity",fill = "blue")+xlab("Variables")+
    ggtitle("Missing Values") + theme_bw()
}

#calculate the missing values
missingvalue(train)

#we observe that passenger_count is categorical variable , hence we will use mode imputation
#calculate mode-create function
mode= function(data){
  uniq=unique(data)
  as.numeric(as.character(uniq[which.max(tabulate(match(data,uniq)))]))
  #print(mode_d)
}

#calculating mode
mode(train$passenger_count)

#now we will impute using mode
train$passenger_count[is.na(train$passenger_count)]=mode(train$passenger_count)

sum(is.na(train$passenger_count))

#now we will choose suitabke method for imputation of missing values for other variables

#for comparing to actual value
train[40,1]
#train[40,1]= 17.5 
#Mean= 15.12488
#Median= 8.5
#KNN= 15.90051


# Mean method
# train$fare_amount[is.na(train$fare_amount)] = mean(train$fare_amount, na.rm = T)

# Median Method 
# 
# train$fare_amount[is.na(train$fare_amount)] = median(train$fare_amount, na.rm = T)

# KNN Method 
# 
# train = knnImputation(train, k = 5)


#saving the data in df
df = train
train=train[complete.cases(train[,1]),]

#we observe from above imputation of different methods for missing values that KNN is the closest to the actual value
#hence we will use KNN method to impute the missing values of train dataset
train = knnImputation(train,k=5)

#to check the number of missing values now present in the train dataset
missingvalue(train)

#copying the train data in df
df = train

#Outlier Analysis

#we observe that there are outliers present in fare_amount as negative values and we will remove these values now
train$fare_amount = ifelse(train$fare_amount<0,NA,train$fare_amount)
train$fare_amount=ifelse(train$fare_amount>30,NA,train$fare_amount)


#now we will remove the outliers from passenger_Count which is greater than 8
unique(train$passenger_count)

#we will now convert more than 8 passenger_count to NA
for (i in 1:nrow(train)){
  if (as.integer(train$passenger_count[i]) > 8){
    train$passenger_count[i]=NA
  }
}

#we will now observe the range of location points
range(train$pickup_longitude)
range(train$pickup_latitude)
range(train$dropoff_longitude)
range(train$dropoff_latitude)

cnames=colnames(train[,c(2:5)])
library(ggplot2)
for (i in 1:length(cnames)){
  assign(paste0("gn",i),ggplot(aes_string(y=(cnames[i]),x="fare_amount"),data=subset(train))+
           stat_boxplot(geom="errorbar",width=0.5)+geom_boxplot(outlier.color = "red",fill="grey",
                                                                outlier.shape = 18,outlier.size = 1,notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y=cnames[i],x="y")+
           ggtitle(paste("Box  Plot f fare amount",cnames[i])))
}


#plotting the plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)

#now we will replace all outliers with NA and impute
#creating NA on outliers
for(i in cnames){
  val = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
  print(length(val))
  train[,i][train[,i] %in% val] = NA
}

missingvalue(train)

#we will replace the missing value with mode
mode(train$passenger_count)

train$passenger_count[is.na(train$passenger_count)]= mode(train$passenger_count)
train = train[complete.cases(train[,1]), ]


#now we will replace the other missing values with mean
train$fare_amount[is.na(train$fare_amount)] = mean(train$fare_amount, na.rm=T)
train$pickup_longitude[is.na(train$pickup_longitude)] = mean(train$pickup_longitude, na.rm=T)
train$pickup_latitude[is.na(train$pickup_latitude)] = mean(train$pickup_latitude, na.rm=T)
train$dropoff_longitude[is.na(train$dropoff_longitude)] = mean(train$dropoff_longitude, na.rm=T)
train$dropoff_latitude[is.na(train$dropoff_latitude)] = mean(train$dropoff_latitude, na.rm=T)

#checking the missing values
missingvalue(train)

#now we will convert passenger_count into factor
unique(train$passenger_count)

train$passenger_count = as.factor(train$passenger_count)

#checking the structure of train data again
str(train)

#copying the train data into df
df = train

#Feature scaling/engineering
library(geosphere)
#creating new variable
train$dist = distHaversine(cbind(train$pickup_longitude,train$pickup_latitude),cbind(train$dropoff_longitude,train$dropoff_latitude))

#we will change the dist values in kms
train$dist = as.numeric(train$dist)/1000

View(train$dist)

df = train
train = df


#Correlation Analysis
corrgram(train[,-6], order = F,upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#we will now observe correlaton between the numeric variables
num_cor=round(cor(train[,-6]), 3)

#we will now eliminate the pickup and dropoff locations if same

train=subset(train, !(train$pickup_longitude==train$dropoff_longitude & train$pickup_latitude==train$dropoff_latitude))

#checking distribution of data
hist(train$dist)
hist(train$fare_amount)

#we will now remove unnecessary variables
rm(cnames)


#model development
#create sampling and divide data into train and test

set.seed(123)
train_index = sample(1:nrow(train), 0.8 * nrow(train))

train1 = train[train_index,]
test1 = train[-train_index,]

#define mape
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y*100))
}


#Decision Tree
library(rpart)
fit = rpart(fare_amount ~. , data = train1, method = "anova", minsplit=5)

summary(fit)
predictions_DT = predict(fit, test1[,-1])

MAPE(test1[,1], predictions_DT)
#Error 48.62096
#Accuracy 51.37

write.csv(predictions_DT, "DT_Data.csv", row.names = F)

#Random Forest
RF_model = randomForest(fare_amount ~.  , train1, importance = TRUE, ntree=500)
RF_Predictions = predict(RF_model, test1[,-1])

write.csv(RF_Predictions, "DF_Data.csv", row.names = F)

MAPE(test1[,1], RF_Predictions)
#error 38.53504 for n=500
#accuracy = 61.46

importance(RF_model, type = 1)


#linear regression model
lm_model = lm(fare_amount ~. , data = train1)
summary(lm_model)

predictions_LR = predict(lm_model, test1[,-1])
MAPE(test1[,1], predictions_LR)

#error  43.76524
#Accuracy 56.23

#KNN Implementation

library(class)
KNN_Predictions = knn(train1[, 2:7], test1[, 2:7], train1$fare_amount, k = 1)

#converting  the values into numeric
KNN_Predictions=as.numeric(as.character((KNN_Predictions)))

#calculating mape
MAPE(test1[,1],KNN_Predictions)
#error 43.35813
#Accuracy = 56.64

write.csv(KNN_Predictions, "KNN1_Data.csv", row.names = F)

#Model Selection and Final tuning

#random forest  with using mtry=2 which means that we are fixing only two variables to split at each tree node
RF_model = randomForest(fare_amount ~.  , train1, importance = TRUE, ntree=200, mtry=2)

RF_Predictions = predict(RF_model, test1[,-1])

#calculating mape
MAPE(test1[,1], RF_Predictions)
importance(RF_model, type = 1)

#error 38.82566
#accuracy 61.174

rm(num_cor)

#we will now predict values in test data
predict_testdata=read.csv("test.csv", header= T)[,-1]

str(predict_testdata)
View(predict_testdata)

#creating distance variable
predict_testdata=subset(predict_testdata, !(predict_testdata$pickup_longitude==predict_testdata$dropoff_longitude & predict_testdata$pickup_latitude==predict_testdata$dropoff_latitude))
predict_testdata[predict_testdata==0]= NA

#converting the data into proper data types
predict_testdata$passenger_count=as.factor(predict_testdata$passenger_count)

#calculating distance
predict_testdata$dist= distHaversine(cbind(predict_testdata$pickup_longitude, predict_testdata$pickup_latitude), cbind(predict_testdata$dropoff_longitude,predict_testdata$dropoff_latitude))

View(predict_testdata$dist)

#on viewing we notice that the output is in metres so we will convert it into kms
predict_testdata$dist=as.numeric(predict_testdata$dist)/1000

#chcking the values again
View(predict_testdata$dist)

#create the target variable for maching to train data orginally
predict_testdata$fare_amount = 0
predict_testdata=predict_testdata[,c(1,2,3,4,5,6,7)]

#random forest
RF_model = randomForest(fare_amount ~.  , train, importance = TRUE, ntree=200, mtry=2)

predict_testdata$fare_amount = predict(RF_model, predict_testdata[,0:6])

#saving the predicted data into the file location
write.csv(predict_testdata, "Predicted_Data(R).csv", row.names = F)
