# Machine Learning Course Project
Shailesh Patel  
March 22, 2015  

#Background

This project is related to Human Activity Research (HAR) and the use of personal activity data captured via devices such as Jawbone Up, Nike FuelBand, and Fitbit.  The dataset contains data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions. We are provided a data set from:

http://groupware.les.inf.puc-rio.br/har

Our goal is to use the data set to predict the quality of the exercise the user conducted.  The quality outcome is defined as classes as follows:

A - exactly according to the specification
B - throwing the elbows to the front
C - lifting the dumbbell only halfway
D - lowering the dumbbell only halfway
E - throwing the hips to the front

#Reproducibility
We will first load the necessary libraries and set the seed.

```r
# Install and load the libraries we will need - only if needed
list.of.packages <- c("caret", 
                      "kernlab", 
                      "randomForest", 
                      "Hmisc", 
                      "abind", 
                      "arm", 
                      "rpart", 
                      "doParallel")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(kernlab)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(Hmisc)
```

```
## Loading required package: grid
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: Formula
## 
## Attaching package: 'Hmisc'
## 
## The following object is masked from 'package:randomForest':
## 
##     combine
## 
## The following objects are masked from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
library(abind)
library(arm)
```

```
## Loading required package: MASS
## Loading required package: Matrix
## Loading required package: lme4
## Loading required package: Rcpp
## 
## arm (Version 1.7-07, built: 2014-8-27)
## 
## Working directory is /Users/shaileshpatel/Library/Mobile Documents/com~apple~CloudDocs/Coursera/Machine Learning/Course Project/ML-Project
```

```r
library(rpart)
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
set.seed(8666)
```

#Data 
##Load Libraries and Download Data
The training data for this project wre loaded from the following URLs:

Training data:  https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

Test data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Download the data for training and for final testing. We call the testing data "final testing" data to avoid confusion with the test set we'll create to validate our model.  Final testing is used for the submission of the results of this work. 


```r
trainingcsv <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingcsv <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainingcsv, "pml-training.csv", method="curl")
download.file(testingcsv, "pml-testing.csv", method="curl")

date() # Log the date and time when the file is downloaded
```

```
## [1] "Mon Mar 16 23:35:47 2015"
```

```r
trainingData <- read.csv("pml-training.csv")
finalTestingData <- read.csv("pml-testing.csv")
```

##Cleaning Data
Make sure all the column names between the training and final testing data sets are identical. We know the final testing data does not contain the "classe" column.   


```r
all.equal(colnames(finalTestingData)[1:length(colnames(finalTestingData))-1], colnames(trainingData)[1:length(colnames(trainingData))-1])
```

```
## [1] TRUE
```
To clean the data, we are first removing the near zero variance predictors. Many variables have a high degree of correlation. 

```r
nzvValues <- nearZeroVar(trainingData)
trainingData <- trainingData[, -nzvValues]
dim(trainingData)
```

```
## [1] 19622   100
```
We then remove the first five predictors/columns of the training data id, user names, and time stamps are not very useful with models.  

```r
trainingData <- trainingData[, -(1:5)]
dim(trainingData)
```

```
## [1] 19622    95
```
Lastly, we remove columns with too many NA's. We want to reduce predictors with more than 60% missing data.  

```r
remove <- sapply(colnames(trainingData), function(x) if(sum(is.na(trainingData[, x])) 
                                                        > 0.60*nrow(trainingData))
  {return(TRUE)
} else {
return(FALSE)})
trainingData <- trainingData[, !remove]
dim(trainingData)
```

```
## [1] 19622    54
```
After cleaning the data the following predictors are included in the models.

```r
names(trainingData)
```

```
##  [1] "num_window"           "roll_belt"            "pitch_belt"          
##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [52] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

##Partitioning Data
We'll first partition the data so that 60% of the data will be used for training and 40% will be used for testing the models. 

```r
inTrain <- createDataPartition(y=trainingData$classe, p=0.6, list=FALSE)
modelTraining <- trainingData[inTrain, ]
modelTesting <- trainingData[-inTrain, ]
dim(modelTraining)
```

```
## [1] 11776    54
```

```r
dim(modelTesting)
```

```
## [1] 7846   54
```

#Model Development
##Selecting a Model
We can now create models based on the pre-processed data set. In order to avoid overfitting and to reduce out of sample errors, we use TrainControl to perform 5-fold cross validation instead of the default 10. The six models estimated are: Random forest, Support Vector Machine (both radial and linear), a Neural net, a Bayes Generalized linear model and a Logit Boosted model. We will also use the parallel library to maximize multiple cores on the host machine.  


```r
library(parallel); library(doParallel)
registerDoParallel(clust <- makeForkCluster(detectCores()))

# Set up the control for the models
modelCTRL <- trainControl(method = "cv", number = 5, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)

# Run the six models
randomForest <- train(classe ~ ., data = modelTraining, method = "rf", trControl= modelCTRL, ntree=50)
svmr <- train(classe ~ ., data = modelTraining, method = "svmRadial", trControl= modelCTRL)
neural <- train(classe ~ ., data = modelTraining, method = "nnet", trControl= modelCTRL)
```

```
## Loading required package: nnet
```

```
## # weights:  300
## initial  value 23503.680401 
## iter  10 value 18281.619047
## iter  20 value 17962.866579
## iter  30 value 17154.108235
## iter  40 value 16969.304281
## iter  50 value 16803.429215
## iter  60 value 16650.033797
## iter  70 value 16504.447574
## iter  80 value 16480.448570
## iter  90 value 16375.929941
## iter 100 value 16315.117373
## final  value 16315.117373 
## stopped after 100 iterations
```

```r
svml <- train(classe ~ ., data = modelTraining, method = "svmLinear", trControl= modelCTRL)
bayesglm <- train(classe ~ ., data = modelTraining, method = "bayesglm", trControl= modelCTRL)
logitboost <- train(classe ~ ., data = modelTraining, method = "LogitBoost", trControl= modelCTRL)
```

```
## Loading required package: caTools
```

```r
stopCluster(clust)
```

Let's compare the accuracy of the models. Random Forest, LogitBoost, and SVM (radial) have the best accuracy of all size models.  


```r
model <- c("Random Forest","LogitBoost","SVM (radial)", "SVM (linear)", "Neural Net", "Bayes GLM")
Accuracy <- c(max(randomForest$results$Accuracy),
         max(logitboost$results$Accuracy),
         max(svmr$results$Accuracy),
         max(svml$results$Accuracy),
         max(neural$results$Accuracy),
         max(bayesglm$results$Accuracy))
        
Kappa <- c(max(randomForest$results$Kappa),
         max(logitboost$results$Kappa),
         max(svmr$results$Kappa),
         max(svml$results$Kappa),
         max(neural$results$Kappa),
         max(bayesglm$results$Kappa))  

performance <- cbind(model, Accuracy, Kappa)
knitr::kable(performance)
```



model           Accuracy            Kappa             
--------------  ------------------  ------------------
Random Forest   0.994735472091063   0.993340443450489 
LogitBoost      0.92972348753086    0.910549154246006 
SVM (radial)    0.917714570664534   0.895782596238381 
SVM (linear)    0.787109308208236   0.729396971598701 
Neural Net      0.431975847961046   0.280604261183444 
Bayes GLM       0.401239518617301   0.234750696831903 

##Cross Validation and Out of Sample Error
We now predict new values within the test set that we created for random forest, LogitBoost, and SVM(radial) models.  


```r
randomForestPrediction <- predict(randomForest, modelTesting)
logitboostPrediction <- predict(logitboost, modelTesting)
svmrPrediction <- predict(svmr, modelTesting)
```

Let's check to see if the models give same predictions. Random Forest and LogitBoost provide the greatest number of matches.  


```r
firstPrediction <- data.frame(cbind(randomForestPrediction, logitboostPrediction))
firstPrediction$same <- with(firstPrediction, randomForestPrediction == logitboostPrediction)
colnames(firstPrediction) <- c("Random Forest", "LogitBoost", "SamePrediction")

# Number of matches between Random Forest and LogitBoost
dim(firstPrediction[firstPrediction$SamePrediction==TRUE,])
```

```
## [1] 7407    3
```

```r
secondPrediction <- data.frame(cbind(randomForestPrediction, svmrPrediction))
secondPrediction$same <- with(secondPrediction, randomForestPrediction == svmrPrediction)
colnames(secondPrediction) <- c("Random Forest", "SVM (radial)", "SamePrediction")

# Number of matches between Random Forest and SVM (radial)
dim(secondPrediction[secondPrediction$SamePrediction==TRUE,])
```

```
## [1] 7306    3
```

We can calculate the expected out of sample error based on the test set that we created for cross-validation -- using the Random Forest model.  The accuracy of the Random Forest model is 99.4735472%. So I expect the out of sample error estimate to be less than 41.305486, where 7,846 is the number of rows in the modelTesting dataset. The prediction results with random forest are encouraging in looking at the confusion matrix.  The confusionMatrix shows that 24 predictions are inaccurate, which is better than my expected out of sample error estimate.  


```r
myMatrix <- confusionMatrix(randomForestPrediction, modelTesting[, "classe"])
myMatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    8    0    0    0
##          B    0 1508    5    0    0
##          C    0    2 1363    7    0
##          D    0    0    0 1278    1
##          E    1    0    0    1 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9968          
##                  95% CI : (0.9953, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.996           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9934   0.9963   0.9938   0.9993
## Specificity            0.9986   0.9992   0.9986   0.9998   0.9997
## Pos Pred Value         0.9964   0.9967   0.9934   0.9992   0.9986
## Neg Pred Value         0.9998   0.9984   0.9992   0.9988   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1922   0.1737   0.1629   0.1837
## Detection Prevalence   0.2854   0.1928   0.1749   0.1630   0.1839
## Balanced Accuracy      0.9991   0.9963   0.9975   0.9968   0.9995
```

#Prediction Submission
Generate the files for submission using the final test data provided.

```r
answers <- predict(randomForest, finalTestingData)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}
pml_write_files(answers)
```


