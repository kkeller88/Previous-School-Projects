#########################################################
#                                                       #
#           ---------------------------------           #        
#              PREPARE WORKSPACE AND DATA               #
#           ---------------------------------           #        
#                                                       #
#########################################################

# working directory
setwd("~/Documents/UCLA/CLASSES/273 CART/final")

require(rpart)
require(randomForest)
require(corrplot)
require(caTools)
require(e1071)

### load data
data1 <- read.csv("/Users/kristenkeller/Documents/UCLA/CLASSES/273 CART/final/student-mat.csv",sep = ";")
data_levels <- apply(data1,2,unique)
data_dim <- dim(data1) # 395 x 33

### double check continuous variables
data1$age<-as.numeric(data1$age)
data1$Medu<-as.numeric(data1$Medu)
data1$Fedu<-as.numeric(data1$Fedu)
data1$traveltime<-as.numeric(data1$traveltime)
data1$studytime<-as.numeric(data1$studytime)
data1$failures<-as.numeric(data1$failures)
data1$famrel<-as.numeric(data1$famrel)
data1$freetime<-as.numeric(data1$freetime)
data1$goout<-as.numeric(data1$goout)
data1$Dalc<-as.numeric(data1$Dalc)
data1$Walc<-as.numeric(data1$Walc)
data1$health<-as.numeric(data1$health)
data1$absences<-as.numeric(data1$absences)
data1$G1<-as.numeric(data1$G1)
data1$G2<-as.numeric(data1$G2)
data1$G3<-as.numeric(data1$G3)


#########################################################
#                                                       #
#         SUMMARY OF DATA AND TEST/TRAIN SPLIT          #
#                                                       #
#########################################################

### CORRELATION: separate continuous and categorical
cont <- c("age","Medu","Fedu","traveltime","studytime","failures","famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2","G3")
data1_cont <- data1[,cont]
data1_cat <- data1[,!names(data1) %in% cont]

correlation_mat <- cor(data1_cont,method="spearman") # correltion using SPEARMAN
corrplot(correlation_mat,tl.col=1) # Figure 1

### split data
set.seed(9383)
sample <- sample.split(data1$G3, SplitRatio = .8)
train <- subset(data1,sample == TRUE)
test <- subset(data1,sample == FALSE)

train_cont <- train[,cont]
train_cat <- train[,!names(train) %in% cont]
test_cont <- test[,cont]
test_cat <- test[,!names(test) %in% cont]

# summary of continuous variables
sumCont = function(invector){
  c(mean(invector),median(invector),sd(invector))
}

# summary of categorical variables
propOne = function(invector){
  propLevel <- c()
  # calculate proportion in each category
  for (levelNum in unique(invector)){
    propLev <- length(which(invector==levelNum))/length(invector)
    propLevel <- c(propLevel,propLev)
  }
  # make all same length: change 5 to fn oaram to generalize
  if (length(propLevel) < 5){
    propLevel <- c(propLevel,rep(0,5-length(propLevel)))
  }
  return(propLevel)
}

# summary of categorical variables
sumCatTrain <- apply(train_cat,2,propOne); sumCatTrain <- t(round(sumCatTrain,2))
sumCatTest <- apply(test_cat,2,propOne); sumCatTest <- t(round(sumCatTest,2))
sumCat <- cbind(sumCatTrain,sumCatTest); colnames(sumCat) <- c("Test: Category 1","Category 2","Category 3","Category 4","Category 5","Train: Category 1","Category 2","Category 3","Category 4","Category 5")
 
# summary of continuous table
sumContTrain <- apply(train_cont,2,sumCont); sumContTrain <- t(round(sumContTrain,2))
sumContTest <- apply(test_cont,2,sumCont); sumContTest <- t(round(sumContTest,2))
sumCont <- cbind(sumContTrain,sumContTest); colnames(sumCont)<-c("Test: Mean","Test: Median","Test: SD","Train: Mean","Test: Median","Test: SD")

# write tables
write.csv(sumCont,"~/Documents/UCLA/CLASSES/273 CART/final/contTable.csv")
write.csv(sumCat,"~/Documents/UCLA/CLASSES/273 CART/final/catTable.csv")

# recode outcome variable
test$G3_I <- ifelse(test$G3>=10,1,2)
train$G3_I <- ifelse(train$G3>=10,1,2)




#########################################################
#                                                       #
#           ---------------------------------           #        
#              PREPARE WORKSPACE AND DATA               #
#           ---------------------------------           #        
#                                                       #
#########################################################

### basic model
set.seed(343)
sampRF <- 0.70*length(train$G3_I)
sizeNode <- 1
forest0 <- randomForest(y=as.factor(train$G3_I),x=train[,1:30],
              # ytest=as.factor(test$G3_I),xtest=test[,1:30],
              ntree=1000,mtry=15,importance=TRUE,sampsize = sampRF,nodesize=sizeNode)
forest0.response <- predict(forest0,newdata=test[,1:30])

# prediction error
length(which(forest0.response==test$G3_I))/length(forest0.response)

### Choosing parameters

# find optimal values for mtree and size of last node
err.rates <- c()
set.seed(43433)

for (nodeSz in c(1,3,5,8)){
  for (mtrySz in c(5,10,15,20,25)){
    for.est <- randomForest(y=as.factor(train$G3_I),x=train[,1:30],ntree=1000,mtry=mtrySz,sampsize = sampRF,nodesize=nodeSz)
    err.rate <- mean(for.est$err.rate[,1])
    err.rates <- c(err.rates,err.rate)
  }
}

# names(err.rates) <- c("nodes1_mtry5","nodes1_mtry10","nodes1_mtry15","nodes1_mtry20","nodes3_mtry5","nodes3_mtry10","nodes3_mtry15","nodes3_mtry20","nodes5_mtry5","nodes5_mtry10","nodes5_mtry15","nodes5_mtry20","nodes8_mtry5","nodes8_mtry10","nodes8_mtry15","nodes8_mtry20")
err.rates.mat <- matrix(err.rates,nrow=4,byrow=TRUE); colnames(err.rates.mat)<-c("mtry = 5","mtry = 10","mtry = 15","mtry = 20","mtry=25"); rownames(err.rates.mat)<-c("node size = 1","node size = 3","node size = 5","node size = 8"); err.rates.mat <- round(err.rates.mat,4)
# write.csv(err.rates.mat,"~/Documents/UCLA/CLASSES/273 CART/final/RFmrtyNodsGridSearch2.csv")

# optimal values for parameters
set.seed(3663)
forest1 <- randomForest(y=as.factor(train$G3_I),x=train[,1:30],ntree=1000,mtry=20,sampsize = sampRF,nodesize=1,keep.forest=TRUE,importance=TRUE)
forest1.response <- predict(forest1,newdata=test[,1:30])

# original model sensitivity, specificity, etc... 2 = fail
out.true<-test$G3_I
out.predict<-forest1.response

SensSpec <- function(out.true,out.predict){
  # true and false positives
  OverallPrediction<-(1-length(which(out.true==out.predict))/length(out.true))*100
  TruePositive<-length(which(out.true==1 & out.predict==1))
  FalsePositive<-length(which(out.true==2 & out.predict==1))
  FalseNegative<-length(which(out.true==1 & out.predict==2))
  TrueNegative<-length(which(out.true==2 & out.predict==2))
  # values
  Sensitivity<-TruePositive/(TruePositive+FalseNegative)
  Specificity<-TrueNegative/(TrueNegative+FalsePositive)
  PPV <- TruePositive/(TruePositive+FalsePositive)
  NPV <- TrueNegative/(TrueNegative+FalseNegative)
  #return
  return(list("Prediction Error"=OverallPrediction,"Sensitivity"=Sensitivity,"Specificity"=Specificity,"PPV"=PPV,"NPV"=NPV))
}

# results for first model
forest1.sens <- SensSpec(test$G3_I,forest1.response) 
forest1.sens.mat<-matrix(forest1.sens,nrow=5);
rownames(forest1.sens.mat)<-names(forest1.sens); 

# proportion 1 and proportion 2 in predictions
predict.pass <- length(which(out.predict==1))/length(out.predict)
predict.fail <- length(which(out.predict==2))/length(out.predict)
true.pass <- length(which(test$G3_I==1))/length(test$G3_I)
true.fail <- length(which(test$G3_I==2))/length(test$G3_I)

# response & predictions for all individual trees
forest1.response.all <- predict(forest1,newdata=test[,1:30],predict.all=TRUE)
forest1.predictions.all<-forest1.response.all$individual

### look for a better cutoff than 50-50
senz <- c()
set.seed(3663)
for (cutP in seq(from=0.35,to=0.75,by=0.025)){ 
  forest3 <- randomForest(y=as.factor(train$G3_I),x=train[,1:30],ntree=1000,mtry=20,sampsize = sampRF,nodesize=1,cutoff=c(cutP,1-cutP))
  sen <- c(forest3$confusion[,3])
  senz<-rbind(senz,sen)
}

senz <- 1-senz
colnames(senz)<-c("Sensitivity","Specificity")
rownames(senz)<-seq(from=0.35,to=0.75,by=0.025)

# ROC CURVE
plot(1-senz[,2],senz[,1],type="step",xlab="1-Specificity",ylab="Sensitivity",col=2)
# write.csv(senz,"~/Documents/UCLA/CLASSES/273 CART/final/RF_ROC_DATA.csv")

# forrests with different cutoffs
set.seed(3663)
forest1 <- randomForest(y=as.factor(train$G3_I),x=train[,1:30],ntree=1000,mtry=20,sampsize = sampRF,nodesize=1,keep.forest=TRUE,importance=TRUE,cutoff = c(0.5,0.5))
forest2 <- randomForest(y=as.factor(train$G3_I),x=train[,1:30],ntree=1000,mtry=20,sampsize = sampRF,nodesize=1,keep.forest=TRUE,importance=TRUE,cutoff = c(0.55,0.45))
forest3 <- randomForest(y=as.factor(train$G3_I),x=train[,1:30],ntree=1000,mtry=20,sampsize = sampRF,nodesize=1,keep.forest=TRUE,importance=TRUE,cutoff = c(0.6,0.4))

forest1.response <- predict(forest1,newdata=test[,1:30])
forest2.response <- predict(forest2,newdata=test[,1:30])
forest3.response <- predict(forest3,newdata=test[,1:30])

forest1.sens <- SensSpec(test$G3_I,forest1.response)
forest2.sens <- SensSpec(test$G3_I,forest2.response)
forest3.sens <- SensSpec(test$G3_I,forest3.response)

cutoff.sens<-cbind(unlist(forest1.sens),unlist(forest2.sens),unlist(forest3.sens))
cutoff.sens<-round(cutoff.sens,4)
cutoff.sens[1,]<-round(cutoff.sens[1,],2)
# write.csv(cutoff.sens,"~/Documents/UCLA/CLASSES/273 CART/final/test_sensitivity_cutoffs.csv")

# precent positive and negative
length(which(forest1.response==1))/length(forest1.response) 
length(which(forest1.response==2))/length(forest1.response) 

### Variable importance
forest1$importance[,4]
forest1.imp <- cbind(names(forest1$importance[,4]),unname(forest1$importance[,4]))
forest1.var.imp <- forest1.imp[order(as.numeric(forest1.imp[,2]),decreasing=TRUE),]
forest1.var.imp[,2]<-round((as.numeric(forest1.var.imp[,2])),2)
# write.csv(forest1.var.imp,"~/Documents/UCLA/CLASSES/273 CART/final/variable_importance.csv")


#########################################################
#                                                       #
#           ---------------------------------           #        
#              SUPPORT VECTOR MACHINES                  #
#           ---------------------------------           #        
#                                                       #
#########################################################

# data for svm models
data2_train <- cbind(as.factor(train$G3_I),x=train[,1:30]); colnames(data2_train)[1]<-"G3"
data2_test <- cbind(as.factor(test$G3_I),x=test[,1:30]); colnames(data2_test)[1]<-"G3"
set.seed(3422)

# polynomial (2)
svm_tune1 <- tune(svm,train.x = G3~.,data=data2_train,ranges=list(gamma=c(1/16,1/8,1/4,1/2,1,2,4),cost=c(1/16,1/8,1/4,1/2,1,2,4)),tunecontrol = tune_options, kernel="polynomial",degree=2)
svm1 <- svm(G3~.,data=data2_train,type="C-classification",kernel="polynomial",degree=2,gamma=0.1250,cost=0.0625)
  
# polynomial (3)
svm_tune2 <- tune(svm,train.x = G3~.,data=data2_train,ranges=list(gamma=c(1/16,1/8,1/4,1/2,1,2,4),cost=c(1/16,1/8,1/4,1/2,1,2,4)),tunecontrol = tune_options, kernel="polynomial",degree=3)
svm2 <- svm(G3~.,data=data2_train,type="C-classification",kernel="polynomial",degree=3,gamma=0.1250,cost=0.0625)

# radial
svm_tune3 <- tune(svm,train.x = G3~.,data=data2_train,ranges=list(gamma=c(1/16,1/8,1/4,1/2,1,2,4),cost=c(1/16,1/8,1/4,1/2,1,2,4)),tunecontrol = tune_options, kernel="radial")
svm3 <- svm(G3~.,data=data2_train,type="C-classification",kernel="radial",gamma=0.0625,cost=1)

# sigmoid
svm_tune4 <- tune(svm,train.x = G3~.,data=data2_train,ranges=list(gamma=c(1/16,1/8,1/4,1/2,1,2,4),cost=c(1/16,1/8,1/4,1/2,1,2,4)),tunecontrol = tune_options, kernel="sigmoid")
svm4 <- svm(G3~.,data=data2_train,type="C-classification",kernel="sigmoid",gamma=0.0625,cost=0.2500)

# optimal cost and gamma values
svm.cost.gamma<-rbind(svm_tune1$best.parameters,svm_tune2$best.parameters,svm_tune3$best.parameters,svm_tune4$best.parameters)
rownames(svm.cost.gamma)<-c("Quadradic Polynomial","Cubic Polynomial","Radial","Sigmoid")
colnames(svm.cost.gamma)<-c("Gamma","Cost")
# write.csv(svm.cost.gamma,"~/Documents/UCLA/CLASSES/273 CART/final/svm_cost_gamma_opt.csv")

# prediction error
svm1.pred <- predict(svm1, data2_test[2:31])
svm2.pred <- predict(svm2, data2_test[2:31])
svm3.pred <- predict(svm3, data2_test[2:31])
svm4.pred <- predict(svm4, data2_test[2:31])

svm1.sens<-SensSpec(test$G3_I,svm1.pred)
svm2.sens<-SensSpec(test$G3_I,svm2.pred)
svm3.sens<-SensSpec(test$G3_I,svm3.pred)
svm4.sens<-SensSpec(test$G3_I,svm4.pred)

svm.sens<-rbind(unlist(svm1.sens),unlist(svm2.sens),unlist(svm3.sens),unlist(svm4.sens))
rownames(svm.sens)<-c("Quadradic Polynomial","Cubic Polynomial","Radial","Sigmoid")
svm.sens[,1]<-round(svm.sens[,1],2); svm.sens[,2:5]<-round(svm.sens[,2:5],4)
# write.csv(svm.sens,"~/Documents/UCLA/CLASSES/273 CART/final/svm_sens.csv")





