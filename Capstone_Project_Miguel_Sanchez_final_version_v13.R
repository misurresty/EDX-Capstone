#######################################################################################
# Fraud Detection system
# September 29, 2020
# Miguel Sanchez
#    
# Deploy a Machine learning platform to identify risk transactions for a Bank/Financial institution. 
# The System uses synthetic data for the train, test and validation. The goal is to identify risky 
# patterns and flag the transaction as a possible fraud.
#
# Naive Bayes, Key Nearest Neighbour (KNN) and Random Forest models being used for 
# prediction over test and training data sets. The model with best accuracy will be used 
# to process against validation data set. 
#
# Base data set as well as the R script are saved on GITHUB repository
#
#
#######################################################################################

#######################################################################################
# load libraries 
#######################################################################################


if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(sqldf)) install.packages("sqldf", repos = "http://cran.us.r-project.org")
if(!require(mlbench)) install.packages("mlbench", repos = "http://cran.us.r-project.org")
if(!require(rsample)) install.packages("rsample", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

library(sqldf)
library(caret)
library(randomForest)
library(mlbench)
library(rsample)
library(e1071)
library(reshape2)
library(ggplot2)
library(dplyr)

#######################################################################################
# Loading the DATA SET 
#######################################################################################

githubURL <- "https://github.com/misurresty/EDX-Capstone/blob/master/sampledata-edx-finalcapstonev2.RData?raw=true"
load(url(githubURL))

#######################################################################################
# CHECKING AND EXPLORING THE DATA SET  
#######################################################################################

nrow(base)
str(base, list.len=ncol(base))
head(base)
#######################################################################################
# Base Data Set: 6.681.203 Records
# Base Data Set: 117 variables
# 
# Data set has been created using synthetic methods. Real/Transactional data used as a seed, 
# coming from a Banking legacy/core platform.
#
# Most or the variables are categorical as data is coming from a tranasactional system, only
# a few of them are continuous (i.e. balance, deposit)
#
# Additional variables with the prediction will be created over the test data set, depending 
# on the used algorithm
# 
# CLASS /TARGET variable used for training and prediction
# --> 0 for regular txn큦
# --> 1 for suspicious txn큦 that could lead on a FRAUD 
#
# Values for the CLASS/target variable were assigned based on real occurrences of 
# suspicious vs not suspicious transactions, using a Data Engineering process.
#######################################################################################

#######################################################################################
# Feature Selection & Data Wrangling
#######################################################################################
# Initial analysis will be performed over the original data set to determine relevant 
# variables. Machine Learning algorithms perform better if highly correlated attributes
# are removed.
#######################################################################################
#
# DATA WRANGLING - deleting logs and SIMDEFAULT variables as those are systemic.
# Setting TARGET variable to CLASS and deleting TARGET
#######################################################################################
base$SIMDEFAULT <- NULL
base$LOG <- NULL

base$class <- base$target
base$target <- NULL
#######################################################################################
# Variable Redundancy
#######################################################################################
# Option 1 
#######################################################################################
# The Caret R package provides the findCorrelation which will analyze a correlation matrix 
# of my data's attributes report on attributes that can be removed.
# I want to remove attributes with an absolute correlation of (ideally >0.75).
#######################################################################################
set.seed(7)
# Using a sample dataset to determine variable redundancy;only 1.000.000 records will be 
# processed because of memory limitations
basep <- base[sample(1:6340852,1000000),4:107] 
basep[] <- lapply (basep, function (x) as.numeric (as.character (x)))
correlationMatrix <- cor(basep)
# Summarize the correlation matrix
options(max.print=100000)
print(correlationMatrix)
# Find attributes that are highly correlated
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# Print indexes of highly correlated attributes
print(highlyCorrelated)
# Variables that will be left out due to its high correlation
summary(basep[,highlyCorrelated])

rm(basep)

#######################################################################################
# Option 2 
#######################################################################################
# Building a Learning Vector Quantization (LVQ) model. The varImp is then used to estimate 
# the variable importance, which is printed and plotted
#######################################################################################

# ensure results are repeatable
set.seed(7)
# Sample Dataset
basep <- base[sample(1:6340852,1000),-112]
positivos <- base[base$class == 1,-112]
basep <- rbind(positivos,basep)
# Excluding the class variable
basepclass <- basep$class
basep$class <- NULL
# Converting the variables to numeric
basep[] <- lapply (basep, function (x) as.numeric (as.character (x)))
# Adding the class variable
basepclass -> basep$class
# Factorizing the class variable 
basep$class <- as.factor(basep$class)
# Wrangling for NA큦
basep <- na.omit(basep)
# Several variables with zero variances will be removed from the Dataset
basep$MODFOINT <-NULL
basep$CPP <- NULL
basep$CHECK <-NULL
basep$MODEMAILOTR <- NULL
basep$EFECTI <- NULL
basep$MODDIRCOM <- NULL
basep$MODDIROTR <- NULL
basep$MODFONCOM <- NULL
basep$MODFONINT <- NULL
basep$RESCLASEGD <- NULL
basep$MAILPAGPENS <- NULL
basep$CLADINCONF <-  NULL
basep$CLADINMOD1 <- NULL
basep$CLADINMOD2 <-  NULL
basep$WVPA <- NULL
basep$SOLRETCCV <- NULL
# Prepare the training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# Train the model
model <- train(class~., data=basep, method="lvq", preProcess="scale", trControl=control)
# Estimate variable importance
importance <- varImp(model, scale=FALSE)
# Summarize importance
print(importance, top = 50)
# Plot importance
plot(importance, top = 50)


#######################################################################################
# There is a manual check as there are variables with high correlation but due to 
# business requirements, those need to be included in the data set.
# 
# Based on the results on both variable redundace methods and business requirements, a
# new data set will be created.
#######################################################################################
rm(basep)


#######################################################################################
# New Data Set with relevant variables, based on variable redundace methods applied
#######################################################################################
#
base <- base[,c('ACANXCLADIN',
                'APVP2',
                'APVP3',
                'AVCETRAM',
                'CAND2',
                'CAVP2AUT',
                'CDMODCEL',
                'CDREVDESENR',
                'CLADINCONF',
                'CLADINDESENR',
                'CLADINMOD1',
                'CLADINMOD2',
                'CLADINP1',
                'CLADINREV1',
                'CLADINREV2',
                'CONSCLASEG',
                'CONSTRAM',
                'CONSVINLAB',
                'DEPOSIT',
                'CUSTOMER_AGE',
                'EMAILMODDAT',
                'ENTCLASEGAD',
                'GENCLAVDIN',
                'IPRODSAL',
                'MANDATE',
                'MCLAACCFOR',
                'MCLASATFOR',
                'MODANTCLI',
                'MODEMAILCOM',
                'OPECLADIN',
                'qxtotp',
                'RECACCWEB',
                'RECUPCLACCE',
                'REPAVTRAM',
                'BALANCE',
                'CUST_PROFIT_SCORE',
                'CUST_SERVICE_SCORE',
                'SECLSEG',
                'SMSMODCEL1',
                'SMSMODCEL2',
                'REQVP',
                'REQVPA',
                'WVP',
                'TOTPEMAIL',
                'dominio',
                'class')]

# Delete records without selected transactions
base$borrar <- ifelse(
  base$ACANXCLADIN == 0 &
    base$APVP2 == 0 &
    base$APVP3 == 0 &
    base$AVCETRAM == 0 &
    base$CAND2 == 0 &
    base$CAVP2AUT == 0 &
    base$CDMODCEL == 0 &
    base$CDREVDESENR == 0 &
    base$CLADINCONF == 0 &
    base$CLADINDESENR == 0 &
    base$CLADINMOD1 == 0 &
    base$CLADINMOD2 == 0 &
    base$CLADINP1 == 0 &
    base$CLADINREV1 == 0 &
    base$CLADINREV2 == 0 &
    base$CONSCLASEG == 0 &
    base$CONSTRAM == 0 &
    base$CONSVINLAB == 0 &
    base$DEPOSIT == 0 &
    #base$edad_cliente == 0 &
    base$EMAILMODDAT == 0 &
    base$ENTCLASEGAD == 0 &
    base$GENCLAVDIN == 0 &
    base$IPRODSAL == 0 &
    base$MANDATE == 0 &
    base$MCLAACCFOR == 0 &
    base$MCLASATFOR == 0 &
    base$MODANTCLI == 0 &
    base$MODEMAILCOM == 0 &
    base$OPECLADIN == 0 &
    #base$qxtotp == 0 &
    base$RECACCWEB == 0 &
    base$RECUPCLACCE == 0 &
    base$REPAVTRAM == 0 &
    base$BALANCE == 0 &
    #base$score_rentabilidad == 0 &
    #base$score_servicio == 0 &
    base$SECLSEG == 0 &
    base$SMSMODCEL1 == 0 &
    base$SMSMODCEL2 == 0 &
    base$REQVP == 0 &
    base$REQVPA == 0 &
    base$WVP == 0 &
    base$TOTPEMAIL == 0 ,1,0)

table(base$borrar)

## Base with selected variables and transactions with movements
base <- base[base$borrar == 0,]
# Clean up (NA큦) and save the dataset
base <- na.omit(base)

table(base$class)

## Additional wrangling to include risky web domain (business requirement)
base$dominioriesgoso <- ifelse(base$dominio == 'vtr.net' | base$dominio == 'mi.cl',1,0)
base$dominioriesgoso <- as.factor(base$dominioriesgoso)

table(base$dominioriesgoso)

base$dominio <- NULL

#######################################################################################
# Additional analysis over the data set (base), once relevant variables have been 
# selected
#######################################################################################
cat("\nBase set dimension :",dim(base))
cat("\nNumber of unique ages :",base$CUSTOMER_AGE %>% unique() %>% length())
cat("\nNumber of unique profitability score  :",base$CUST_PROFIT_SCORE %>% unique() %>% length())
cat("\nNumber of unique service score  :",base$CUST_SERVICE_SCORE %>% unique() %>% length())

#######################################################################################
# New data set with relevant variables
# Number of records: 4.645.972
# Number of variables: 47
#######################################################################################
# Number of transactions by Age Range 
#######################################################################################
base %>%
  group_by(CUSTOMER_AGE) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = CUSTOMER_AGE, y = count)) +
  geom_line() +
  ggtitle("Number of transactions by Age Range")

#######################################################################################
# Histogram of number of transactions for each service score 
#######################################################################################
base %>%
  group_by(CUSTOMER_AGE) %>%
  summarise(CUST_SERVICE_SCORE=n()) %>%
  ggplot(aes(CUST_SERVICE_SCORE)) +
  geom_histogram(color="black", binwidth = 50) +
  
  ggtitle("Histogram of number of service score by Customer Age")

#######################################################################################
# CLASS variable Analysis - Variable used to identify negative == 0 & positive == 1 fraud 
# transactions 
#######################################################################################
table(base$class)
#######################################################################################
# Only 16 transactions with suspicious (fraud) activity
#######################################################################################


#######################################################################################
# DATA WRANGLING  
#######################################################################################
# Split for training and test; training with 80% of the initial data set
#######################################################################################

set.seed(123)
v <- c(1:(nrow(base)*1))
variables <- c(4:ncol(base))
train_test_split <- initial_split(base[v,variables], prop = 0.80)
train_test_split

#######################################################################################
# Functions training() and testing() used to create train and test data sets
#######################################################################################

train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split)

nrow(train_tbl)
nrow(test_tbl)
#######################################################################################
# Train Data set : 3.716.778 records
# Test Data set  : 929.194 records
#######################################################################################

table(train_tbl$class)
table(test_tbl$class)

#######################################################################################
# Suspicious transaction in the train data set : 13
# Suspicious transaction in the test data set  : 3
#######################################################################################
 
#######################################################################################
# Split for validation Data Set. 50% of the test data set will be used for validation
#######################################################################################

set.seed(123)
porcvalidac <- nrow(test_tbl) * 0.5
filasaletorias <- sample(1:nrow(test_tbl),porcvalidac)
tbl_validacion <- test_tbl[filasaletorias,]
table(tbl_validacion$class)
test_tbl <- test_tbl[-filasaletorias,]

#######################################################################################
# Positive transactions (fraud) proportion in Data Sets 
#######################################################################################
table(train_tbl$class)
table(test_tbl$class)
table(tbl_validacion$class)
#######################################################################################
# Suspicious transaction in the train data set       : 13
# Suspicious transaction in the test data set        : 2
# Suspicious transaction in the validation data set  : 1
#
# Based on the fraud proportions, it is clear we have a data sampling issue that needs 
# to be addressed using data balancing techniques
#######################################################################################

#######################################################################################
# DATA BALANCE
# To train the models we should have 20% on suspicious (positive) and 80% on negative 
# transactions. A new training Data Set will be created
# 1) Undersampling - Decrease negative (not suspicious) transactions 
#######################################################################################
set.seed(123456)
# Positive cases for training 
qx <- 13
qxn <- qx * 4
# Training data set assembly 
negativos <- train_tbl[train_tbl$class == 0, ]
tbl_negativos <- negativos[sample(1:nrow(negativos), qxn),]
tbl_positivos <- train_tbl[train_tbl$class == 1, ]

train_tbl_manual <- rbind(tbl_negativos, tbl_positivos) 

# Checking for the new Training data set
nrow(train_tbl_manual)
head(train_tbl_manual)
table(train_tbl_manual$class)
#######################################################################################
# Number of negative (not fraud) transactions : 52
# Number of positive (fraud) transactions     : 13
#######################################################################################
# 2) Oversampling - Increase positive (suspicious) transactions 
#######################################################################################
positivos <- train_tbl[train_tbl$class == 1, ]

# Increasing positives

n <- 5

for(i in 1:(n-1)) {
  
  positivos <- rbind(positivos, positivos)
  
}

negativos <- train_tbl[train_tbl$class == 0, ]
indnegativos <- sample(1:nrow(negativos), (nrow(positivos)*4))
tbl_negativos <- negativos[indnegativos,]
train_tbl_manual <- rbind(tbl_negativos, positivos)

table(train_tbl_manual$class)
#######################################################################################
# Number of negative (not fraud) transactions : 832
# Number of positive (fraud) transactions     : 208
#######################################################################################

#######################################################################################
# Saving the data sets 
#######################################################################################

save(train_tbl_manual,file="train_tbl_manual.RData")
save(tbl_validacion,file="validacion_tbl.RData")
save(train_tbl,file="train_tb_completa.RData")
save(test_tbl,file="test_tb_completa.RData")

# Deleting objects to release memory
rm(list = ls())

#######################################################################################
# DATA WRANGLING
#######################################################################################
# Variable treatment
#######################################################################################

load("train_tbl_manual.RData")
load("test_tb_completa.RData")



train_tbl_manual$DELETE <- NULL
test_tbl$DELETE <- NULL

#######################################################################################
# Factorizing the class variable (target variable to train the algorithm)
#######################################################################################

train_tbl_manual$class <- as.factor(train_tbl_manual$class)
test_tbl$class <- as.factor(test_tbl$class)

#######################################################################################
# Cleaning the train data set
#######################################################################################


p <- as.data.frame(summary(train_tbl_manual))
p <- na.omit(p) 

#######################################################################################
# Additional wrangling for special transactions (deprecated transactions 
# based on business definition)
#######################################################################################

p1 <- sqldf("select Var2 as q from p where Freq not like '%1: 0%' group by Var2 having count(Var2) > 1")


#######################################################################################
# Wrangling - Removing blankspaces from the names and adding to the data frame
#######################################################################################

p1$q <- gsub(pattern = "\\s",   
             replacement = "",
             x = p1$q)


incluir <- p1$q

train_tbl_manual <- train_tbl_manual[,incluir]


#######################################################################################
# Moving the target/class variable to the end of the table
#######################################################################################

target<- train_tbl_manual$class
train_tbl_manual$class <- NULL
target -> train_tbl_manual$class


#######################################################################################
# Excluding the target variable (class) from the test data set for prediction
#######################################################################################
x<-test_tbl[,-42] 

#######################################################################################
# Wrangling -excluding NA큦 from the train data set
#######################################################################################

train_tbl_manual<- na.omit(train_tbl_manual)


#######################################################################################
# DATA MODELING 
#######################################################################################
# Training with several Machine Learning Models
#######################################################################################

#######################################################################################
# Naive Bayes Algorithm
#######################################################################################

#Build the model
modelBayes<-naiveBayes(class~.,data=train_tbl_manual)

#Summarize the model
summary(modelBayes)

#Predict using the model
test_tbl$pred_Bayes<-predict(modelBayes,x)

#Accuracy of the model
mtab1<-table(test_tbl$pred_Bayes,test_tbl$class, dnn = c("prediccion", "real"))
confusionMatrix(mtab1, positive = '1')

#Saving model큦 accuracy 
cm1<- confusionMatrix(mtab1, positive = '1')
overall.accuracy1<-cm1$overall['Accuracy']

#Saving the model
save(modelBayes, file = "modelBayes.rda")



#######################################################################################
# Random Forest Algorithm
#######################################################################################

#Build the model
model15<-randomForest(class ~ ., data=train_tbl_manual[,-1], ntree=600) 

#Summarize the model
summary(model15)

#Predict using the model
test_tbl$pred_randomforest<-predict(model15,x)


#Accuracy of the model
mtab2<-table(test_tbl$pred_randomforest,test_tbl$class, dnn = c("prediction", "real"))
confusionMatrix(mtab2, positive = '1')

#Saving model큦 accuracy 
cm2<- confusionMatrix(mtab2, positive = '1')
overall.accuracy2<-cm2$overall['Accuracy']

#Saving the model
save(model15, file = "model15_RF.rda")


#######################################################################################
# KNN Algorithm
#######################################################################################

#Build the model
model9<-knn3(class ~ .,data=train_tbl_manual,k=14)
summary(model9)

#Predict using the model
test_tbl$pred_knn<-predict(model9,x,type="class")

#Accuracy of the model
mtab3<-table(test_tbl$pred_knn,test_tbl$class, dnn = c("prediccion", "real"))
confusionMatrix(mtab3, positive = '1')

#Saving model큦 accuracy
cm3<- confusionMatrix(mtab3, positive = '1')
overall.accuracy3<-cm3$overall['Accuracy']

#Saving the model
save(model9, file = "modeloknn2020.rda")

#######################################################################################
# DATA MODELING 
#######################################################################################
# Machine Learning Model Validation
#######################################################################################

MODEL_EVALUATED<- c("Bayes Model", "RF Model", "KNN Model")
MODEL_ACCURACY<- c(overall.accuracy1, overall.accuracy2, overall.accuracy3)
EVALUATION_RESULT<- data.frame(MODEL_EVALUATED, MODEL_ACCURACY)
EVALUATION_RESULT

#######################################################################################
# Based on the results processing over training and test data sets, the Random forest 
# Algorithm is providing the best accuracy. The RF algorithm will be used to process 
# against the validation data set.
#######################################################################################

#######################################################################################
# DATA MODELING 
#######################################################################################
# Processing against validation data set
#######################################################################################

#Loading validation data set
load("validacion_tbl.RData")

#Checking for fraud (positive == 1) transactions
table(tbl_validacion$class)

# Excluding the target variable (class) from the validation data set for prediction
x_final<-tbl_validacion[,-42]

#Predict using the model
tbl_validacion$pred_randomforest<-predict(model15,x_final)

#Accuracy of the model
mtabfinal<-table(tbl_validacion$pred_randomforest,tbl_validacion$class, dnn = c("prediccion", "real"))
confusionMatrix(mtabfinal, positive = '1')

#Getting model큦 accuracy
cmfinal<- confusionMatrix(mtabfinal, positive = '1')
overall.accuracyfinal<-cmfinal$overall['Accuracy']
overall.accuracyfinal

#Plotting the model
plot(model15)

#######################################################################################
# Observation 
#######################################################################################
# After 50 iterations (trees) real vs prediction trend to have the same 
# values
# Accuracy : 0.9998902
#######################################################################################
# Conclusion / Recomendation / Next Steps
#######################################################################################
# * Automate the training and then model/algorithm selection process, 
#   based on its accuracy.
# * Increase data processing capacity using a platform like DATABRICKS.
# * Deploy a real time application using the trained model; the application 
#   should be calling a Rest API.
# * Further investigation on API큦 deployment (Shiny vs Plumber) must be 
#   performed
#######################################################################################
