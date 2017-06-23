
### library
require(plyr)
require(data.table)
require(xgboost)
require(randomForest)
require(caret)

### load data
test_imputed <- data.frame(fread("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/imputed_datasets/test_imputed.csv"))
test_complete <- data.frame(fread("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/imputed_datasets/test_complete.csv"))
train_imputed <- data.frame(fread("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/imputed_datasets/train_imputed.csv"))
train_complete <- data.frame(fread("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/imputed_datasets/train_complete.csv"))
macro_data <- data.frame(fread("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Data/macro.csv"))

##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                             Modify the data                                #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

### remove ID and timestamp

# function
id_drop <- function(thedata) {
  thedata <- data.frame(thedata) # make sure df for indexing 
  thedata <- thedata[,-which(colnames(thedata) %in% c('id','timestamp',"V1"))] # drop
  return(thedata)
}

# apply id_drop
test_imputed_ <- id_drop(test_imputed)
train_imputed_ <- id_drop(train_imputed)
test_complete_ <- id_drop(test_complete)
train_complete_ <- id_drop(train_complete)

### seperate x and y 
train_imputed_x <- train_imputed_[,-grep("price_doc",colnames(train_imputed_))]
train_imputed_y <- train_imputed_$price_doc
train_complete_x <- train_complete_[,-grep("price_doc",colnames(train_complete_))]
train_complete_y <- train_complete_$price_doc

##############################################################################
#                                                                            #
#                   Split train into train and validation                    #
#                                                                            #
##############################################################################

### decide which date to use
hist(as.Date(train_imputed$timestamp),breaks=1000)
length(which(train_imputed$timestamp>as.Date("2015-1-1")))

### extract validation set - data from end
seperate_validation_data <- function(thedata) {
  
  thedata <- data.frame(thedata)
  thedata$timestamp <- as.Date(thedata$timestamp)
  split_date <- as.Date("2015-1-1")
  
  # date to split on
  validation <- id_drop(thedata[which(thedata$timestamp>split_date),])
  train <- id_drop(thedata[which(thedata$timestamp<split_date),])
  
  # split x and y
  validation_x <- validation[,-grep("price_doc",colnames(validation))]
  validation_y <- validation$price_doc
  train_x <- train[,-grep("price_doc",colnames(train))]
  train_y <- train$price_doc
  
  # return
  return_list <- list(validation_x,validation_y,train_x,train_y)
  names(return_list) <- c("validation_x","validation_y","train_x","train_y")
  return(return_list)
}

### create new dataframes

# fully imputed
train_list_imputed <- seperate_validation_data(train_imputed)
imputed_train_x <- train_list_imputed[["train_x"]]
imputed_train_y <- train_list_imputed[["train_y"]]
imputed_validation_x <- train_list_imputed[["validation_x"]]
imputed_validation_y <- train_list_imputed[["validation_y"]]
  
# partially imputed
train_list_complete <- seperate_validation_data(train_complete)
complete_train_x <- train_list_complete[["train_x"]]
complete_train_y <- train_list_complete[["train_y"]]
complete_validation_x <- train_list_complete[["validation_x"]]
complete_validation_y <- train_list_complete[["validation_y"]]

##############################################################################
#                                                                            #
#           Use only data from 2014-2015 - with validation set               #
#                                                                            #
##############################################################################

### create new dataframes

# partially imputed
train_list_complete_14 <- seperate_validation_data(  train_complete[which(as.Date(train_complete$timestamp) > as.Date("2014-01-01")),]  )
complete_train_x_14 <- train_list_complete_14[["train_x"]]
complete_train_y_14 <- train_list_complete_14[["train_y"]]
complete_validation_x_14 <- train_list_complete_14[["validation_x"]]
complete_validation_y_14 <- train_list_complete_14[["validation_y"]]

##############################################################################
#                                                                            #
#           Use weighted dataset - with validation set                       #
#                                                                            #
##############################################################################

### create datasets
start_date <- min(as.Date(train_complete$timestamp))
train_complete$timestamp <- as.Date(train_complete$timestamp)

train_complete_w1 <- train_complete
train_complete_w2 <- train_complete
train_complete_w3 <- train_complete

### create data weights - cubic, linear, quadratic
yearsfromstart <- round((train_complete$timestamp-start_date)/365) +1
train_complete_w1$weight <- as.numeric(yearsfromstart)
train_complete_w2$weight <- as.numeric(yearsfromstart)^2
train_complete_w3$weight <- as.numeric(yearsfromstart)^3

### seperate validation and train dataset
train_list_complete_w1 <- seperate_validation_data(train_complete_w1)
train_list_complete_w1 <- lapply(train_list_complete_w1,as.matrix)
complete_train_x_w1 <- train_list_complete_w1[["train_x"]]
complete_train_y_w1 <- train_list_complete_w1[["train_y"]]
complete_validation_x_w1 <- train_list_complete_w1[["validation_x"]]
complete_validation_y_w1 <- train_list_complete_w1[["validation_y"]]

train_list_complete_w2 <- seperate_validation_data(train_complete_w2)
train_list_complete_w2 <- lapply(train_list_complete_w2,as.matrix)
complete_train_x_w2 <- train_list_complete_w2[["train_x"]]
complete_train_y_w2 <- train_list_complete_w2[["train_y"]]
complete_validation_x_w2 <- train_list_complete_w2[["validation_x"]]
complete_validation_y_w2 <- train_list_complete_w2[["validation_y"]]

train_list_complete_w3 <- seperate_validation_data(train_complete_w3)
train_list_complete_w3 <- lapply(train_list_complete_w3,as.matrix)
complete_train_x_w3 <- train_list_complete_w3[["train_x"]]
complete_train_y_w3 <- train_list_complete_w3[["train_y"]]
complete_validation_x_w3 <- train_list_complete_w3[["validation_x"]]
complete_validation_y_w3 <- train_list_complete_w3[["validation_y"]]

##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                             Modify the data                                #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

# remove characters
macro_data$child_on_acc_pre_school <- gsub(",","",macro_data$child_on_acc_pre_school)
macro_data$child_on_acc_pre_school <- as.numeric(macro_data$child_on_acc_pre_school)
macro_data$modern_education_share <- gsub(",","",macro_data$modern_education_share)
macro_data$modern_education_share <- as.numeric(macro_data$modern_education_share)
macro_data$old_education_build_share <- gsub(",","",macro_data$old_education_build_share)

# transform
train_complete$timestamp <- as.Date(train_complete$timestamp)
macro_data$timestamp <- as.Date(macro_data$timestamp)
macro_train_complete <- merge(train_complete,macro_data,by="timestamp")
macro_train_list_complete <- seperate_validation_data(macro_train_complete)
macro_train_list_complete <- lapply(macro_train_list_complete,data.matrix)
macro_train_x <- macro_train_list_complete[["train_x"]]
macro_train_y <- macro_train_list_complete[["train_y"]]
macro_validation_x <- macro_train_list_complete[["validation_x"]]
macro_validation_y <- macro_train_list_complete[["validation_y"]]

##############################################################################
#                                                                            #
#                               few macro variables                          #
#                                                                            #
##############################################################################

#### 5 vars

# get right data
vars_keep <- c(names(train_complete),"mortgage_rate","salary","cpi","gdp_quart","rent_price_2room_eco")
macro_train_complete2 <- macro_train_complete[,which(names(macro_train_complete) %in% vars_keep)]

# run models
macro_train_list_complete2 <- seperate_validation_data(macro_train_complete2)
macro_train_list_complete2 <- lapply(macro_train_list_complete2,data.matrix)
macro_train_x2 <- macro_train_list_complete2[["train_x"]]
macro_train_y2 <- macro_train_list_complete2[["train_y"]]
macro_validation_x2 <- macro_train_list_complete2[["validation_x"]]
macro_validation_y2 <- macro_train_list_complete2[["validation_y"]]

#### 10 vars

# get right data
vars_keep <- c(names(train_complete),"mortgage_rate","salary","cpi","gdp_quart","rent_price_2room_eco","ppi","apartment_build","balance_trade","usdrub","rent_price_2room_bus")
macro_train_complete3 <- macro_train_complete[,which(names(macro_train_complete) %in% vars_keep)]

# run models
macro_train_list_complete3 <- seperate_validation_data(macro_train_complete3)
macro_train_list_complete3 <- lapply(macro_train_list_complete3,data.matrix)
macro_train_x3 <- macro_train_list_complete3[["train_x"]]
macro_train_y3 <- macro_train_list_complete3[["train_y"]]
macro_validation_x3 <- macro_train_list_complete3[["validation_x"]]
macro_validation_y3 <- macro_train_list_complete3[["validation_y"]]

##############################################################################
#                                                                            #
#                        macro variables AND 2014                            #
#                                                                            #
##############################################################################

vars_keep <- c(names(train_complete),"mortgage_rate","salary","cpi","gdp_quart","rent_price_2room_eco")
macro_train_complete_14 <- macro_train_complete[,which(names(macro_train_complete) %in% vars_keep)]
macro_train_complete_14 <- macro_train_complete_14[which(as.Date(macro_train_complete_14$timestamp)>as.Date("2014-01-01")),]

# run models
macro_train_list_complete_14 <- seperate_validation_data(macro_train_complete_14)
macro_train_list_complete_14 <- lapply(macro_train_list_complete_14,data.matrix)
macro_train_x_14 <- macro_train_list_complete_14[["train_x"]]
macro_train_y_14 <- macro_train_list_complete_14[["train_y"]]
macro_validation_x_14 <- macro_train_list_complete_14[["validation_x"]]
macro_validation_y_14 <- macro_train_list_complete_14[["validation_y"]]


##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                             functions                                      #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

rmsle <- function(data,lev=NULL,model=NULL) {
  out <- sqrt((1/length(data$pred))*sum((log(data$pred+1)-log(data$obs+1))^2))
  c("RMSLE"=out)
}

run_xbg_general <- function(x,train_x,train_y,validation_x,validation_y,weights=NULL,weight_lab=NULL) {
  
  # save value
  nrounds <- data.frame(x)$nrounds
  
  # run model
  set.seed(15) ############ was not used in firest validation run
  xgb <- xgboost(data=train_x,label=train_y, weight=weights,
                 params = as.list(as.vector(x[-grep("nrounds",names(x))])), 
                 nrounds=nrounds,verbose=F)
  
  # make predictions using xbg and calculate rmsle
  xgb_predict <- predict(xgb,validation_x)
  thedata <- data.frame(pred=xgb_predict,obs=validation_y)
  thedata$pred[which(thedata$pred<0)] <- median(thedata$pred)
  rmsle_ <- rmsle(thedata)
  
  # print status update
  print("Tree Complete")
  
  # make object to return
  obj_out <- cbind(x,"RMSLE"=rmsle_)
  
  if (!is.null(weights)){
    obj_out <- data.frame(obj_out)
    obj_out$weights = weight_lab
  }
  
  return(obj_out)
  
}
















##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                              CV + all data                                 #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

##############################################################################
#                                                                            #
#                   Random forests : fully imputed data  (RMSE)              #
#                                                                            #
##############################################################################

### basic random forest
rf_1 <- randomForest(price_doc~., ntree=200, mtry=50,
             nodesize=200, do.trace=25, train_imputed_,rand_seed=70)
rf_2 <- randomForest(price_doc~., ntree=200, mtry=50,
                     nodesize=100, do.trace=25, train_imputed_,rand_seed=70) # reduce nodesize

### tune parameters using RMSLE
rmsle <- function(data,lev=NULL,model=NULL) {
  out <- sqrt((1/length(data$pred))*sum((log(data$pred+1)-log(data$obs+1))^2))
  c("RMSLE"=out)
}

### first set of models
control <- trainControl(summaryFunction=rmsle,method="cv",number=5)
mtry <- c(50, 100, 150, 200)  
tunegrid <- expand.grid('mtry'=mtry)
set.seed(75)
rf_node_500 <- train(price_doc~.,data=train_imputed_,method='rf',metric="RMSLE",trControl=control,tuneGrid=tunegrid,
                     ntree=150,nodesize=500,rand_seed=75,do.trace=50) 
rf_node_200 <- train(price_doc~.,data=train_imputed_,method='rf',metric="RMSLE",trControl=control,tuneGrid=tunegrid,
                     ntree=150,nodesize=200,rand_seed=75,do.trace=50) 
rf_node_100 <- train(price_doc~.,data=train_imputed_,method='rf',metric="RMSLE",trControl=control,tuneGrid=tunegrid,
                   ntree=150,nodesize=100,rand_seed=75,do.trace=50) 
rf_node_50 <- train(price_doc~.,data=train_imputed_,method='rf',metric="RMSLE",trControl=control,tuneGrid=tunegrid,
                     ntree=150,nodesize=50,rand_seed=75,do.trace=50) 
rf_node_10 <- train(price_doc~.,data=train_imputed_,method='rf',metric="RMSLE",trControl=control,tuneGrid=tunegrid,
                     ntree=150,nodesize=10,rand_seed=75,do.trace=50) 


### look at results 

# make df
r500 <- rf_node_500$results; r500$nodeSize <- 500
r200 <- rf_node_200$results; r200$nodeSize <- 200
r100 <- rf_node_100$results; r100$nodeSize <- 100
r50 <- rf_node_50$results; r50$nodeSize <- 50
r10 <- rf_node_10$results; r10$nodeSize <- 10
r_all <- rbind(r500,r200,r100,r50,r10)

# plot
ggplot(r_all,aes(x=mtry,y=RMSLE,color=factor(nodeSize)))+geom_line()+
  labs(color="Node Size")+xlab("Mtry")+scale_y_continuous(limits = c(0.23,0.34))

# save
# write.csv(r_all,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/rf_basic_tune.csv")
r_all <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/rf_basic_tune.csv")

### run model for predictions
rf1_p <- randomForest(price_doc~.,data=train_imputed_,mtry=150,ntree=150,nodesize=10,rand_seed=75,do.trace=50) 
rf1_predict <- predict(rf1_p,test_imputed_x)

##############################################################################
#                                                                            #
#                   XGBoost partially imputed dataset                        #
#                                                                            #
##############################################################################

# tune parameters
set.seed(43)
control <- trainControl(summaryFunction=rmsle,method="cv",number=5,verboseIter=T)
tunegrid <- expand.grid(nrounds = c(300,500), # number of trees
                        eta = c(0.01,0.1,0.3), # learning rate
                        max_depth = c(2,5,10), # depth of trees
                        gamma = 0,
                        colsample_bytree = 1, 
                        min_child_weight = 1,
                        subsample = 1
                        )
xgb_train <- train(x=train_complete_x,y=train_complete_y,
                   method='xgbTree',trControl=control,
                   tuneGrid=tunegrid, metric="RMSLE",
                   verbose=1)

# save results
# write.csv(xgb_train$results,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_basic_tune.csv")
xbg_basic <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_basic_tune.csv")

# plot
ggplot(xbg_basic,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))

### predict 
set.seed(75)
xg1_p <- xgboost(x=train_complete_x,y=train_complete_y,nrounds = c(300,500),eta = c(0.01,0.1,0.3), 
                 max_depth = c(2,5,10), gamma = 0,colsample_bytree = 1,min_child_weight = 1,subsample = 1)
xg1_predict <- predict(xg1_p,test_complete_x)

##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                          Validation + all data                             #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

##############################################################################
#                                                                            #
#                   XGBoost partially imputed dataset                        #
#                                                                            #
##############################################################################

# rmsle function
rmsle <- function(data,lev=NULL,model=NULL) {
  out <- sqrt((1/length(data$pred))*sum((log(data$pred+1)-log(data$obs+1))^2))
  c("RMSLE"=out)
}

# parameter grid --> list
parmgrid <- expand.grid(nrounds = c(300,500), # number of trees
                        eta = c(0.01,0.1,0.3), # learning rate
                        max_depth = c(2,5,10), # depth of trees
                        gamma = 0,
                        colsample_bytree = 1, 
                        min_child_weight = 1,
                        subsample = 1)

parmlist <- setNames(split(parmgrid, seq(nrow(parmgrid))), rownames(parmgrid))

### make sure these are matricies as matricies
complete_train_x <- as.matrix(complete_train_x)
complete_train_y <- as.matrix(complete_train_y)
complete_validation_x <- as.matrix(complete_validation_x)
complete_validation_y <- as.matrix(complete_validation_y)

### function to run models
run_xbg <- function(x) {
  
  # save value
  nrounds <- data.frame(x)$nrounds
  
  # run model
  xgb <- xgboost(data=complete_train_x,label=complete_train_y,
          params = as.list(as.vector(x[-grep("nrounds",names(x))])), 
          nrounds=nrounds,verbose=F)
  
  # make predictions using xbg and calculate rmsle
  xgb_predict <- predict(xgb,complete_validation_x)
  thedata <- data.frame(pred=xgb_predict,obs=complete_validation_y)
  thedata$pred[which(thedata$pred<0)] <- median(thedata$pred)
  rmsle_ <- rmsle(thedata)
  
  # print status update
  print("Tree Complete")
  
  # make object to return
  obj_out <- cbind(x,"RMSLE"=rmsle_)
  return(obj_out)
  
}

### apply function
set.seed(322)
xgb_list <- lapply(parmlist,run_xbg)
xgb_validation_results <- ldply(xgb_list,data.frame)

### save results
# write.csv(xgb_validation_results,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_validation_tune.csv")
xgb_validation_results <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_validation_tune.csv")

### plot
ggplot(xgb_validation_results,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))

### make predictions
set.seed(75)
xg2_p <- xgboost(data=complete_train_x,label=complete_train_y,nrounds = 500,eta = 0.1, 
                 max_depth = 5, gamma = 0,colsample_bytree = 1,min_child_weight = 1,subsample = 1)
xg2_predict <- predict(xg2_p,test_complete_x)

##############################################################################
#                                                                            #
#                                 Random Forest                              #
#                                                                            #
##############################################################################


##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                    Models + validation + only 2015 - 2014 data             #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

##############################################################################
#                                                                            #
#                   XGBoost partially imputed dataset                        #
#                                                                            #
##############################################################################

# parameter grid --> list
parmgrid <- expand.grid(nrounds = c(300,500), # number of trees
                        eta = c(0.01,0.1,0.3), # learning rate
                        max_depth = c(2,5,10), # depth of trees
                        gamma = 0,
                        colsample_bytree = 1, 
                        min_child_weight = 1,
                        subsample = 1)

parmlist <- setNames(split(parmgrid, seq(nrow(parmgrid))), rownames(parmgrid))

### make sure these are matricies as matricies
complete_train_x_14 <- as.matrix(complete_train_x_14)
complete_train_y_14 <- as.matrix(complete_train_y_14)
complete_validation_x_14 <- as.matrix(complete_validation_x_14)
complete_validation_y_14 <- as.matrix(complete_validation_y_14)

### function to run models

### apply function
set.seed(322)
xgb_list_14 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=complete_train_x_14,train_y=complete_train_y_14,validation_x=complete_validation_x_14,validation_y=complete_validation_y_14))
xgb_validation_results_14 <- ldply(xgb_list_14,data.frame)

### save results
# write.csv(xgb_validation_results_14,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_validation_tune_14.csv")
xgb_validation_results_14 <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_validation_tune_14.csv")

### plot
ggplot(xgb_validation_results_14,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))


#### importance 

# 14 
set.seed(50)
xgb <- xgboost(data=data.matrix(complete_train_x_14),label=complete_train_y_14, weight=weights,
               params = as.list(as.vector(x[-grep("nrounds",names(x))])), 
               nrounds=x$nrounds,verbose=F)
importance_14 <- xgb.importance(feature_names = colnames(complete_train_x),xgb)
importance_14 <- importance_14[order(importance_14$Frequency,decreasing = T),] 
importance_14_short <- importance_14[1:10,]

# 15
set.seed(50)
xgb <- xgboost(data=data.matrix(complete_train_x),label=complete_train_y, weight=weights,
               params = as.list(as.vector(x[-grep("nrounds",names(x))])), 
               nrounds=x$nrounds,verbose=F)
importance_all <- xgb.importance(feature_names = colnames(complete_train_x),xgb)
importance_all <- importance_all[order(importance_all$Frequency,decreasing = T),] 
importance_all_short <- importance_all[1:10,]

# combine 
importance_14_short$model <- "2014-2015"
importance_all_short$model <- "All Years"
importance_all <- rbind(importance_14_short,importance_all_short)

### fix weird ggplot fat bars
nas <- rep(0,2*length(unique(importance_all$Feature)))
fix <- cbind(Feature=c(unique(importance_all$Feature),unique(importance_all$Feature)),Gain=nas,Cover=nas,Frequency=nas,model=c(rep("2014-2015",length(unique(importance_all$Feature))),rep("All Years",length(unique(importance_all$Feature)))))
fix <- rbind(importance_all,fix)
fix <- fix[which(!duplicated(fix[,c(1,5)])),]

# numeric
fix$Frequency <- as.numeric(as.character(fix$Frequency))
fix$Gain <- as.numeric(as.character(fix$Gain))

# order
fix$Feature2 <- factor(fix$Feature,levels=importance_all$Feature)
fix$Feature3 <- factor(fix$Feature,levels=fix$Feature[order(fix$Gain,decreasing=T)])
fix$Feature4 <- factor(fix$Feature,levels=fix$Feature[order(fix$Frequency,decreasing=T)])

ggplot(fix,aes(x=Feature4,y=Frequency,fill=model))+geom_bar(stat="identity",position="dodge")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+xlab("")

#### best model

x<- parmlist[[10]]
set.seed(50)
xgb <- xgboost(data=data.matrix(complete_train_x_14),label=complete_train_y_14, weight=weights,
               params = as.list(as.vector(x[-grep("nrounds",names(x))])), 
               nrounds=x$nrounds,verbose=F)

##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                   Weighted models + validation + all data                  #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

##############################################################################
#                                                                            #
#                   XGBoost partially imputed dataset                        #
#                                                                            #
##############################################################################

### RUN PREVIOUS CHUNK 

### seperate weights 
weights1 <- unname(complete_train_x_w1[,'weight'])
weights2 <- unname(complete_train_x_w2[,'weight'])
weights3 <- unname(complete_train_x_w3[,'weight'])

complete_train_x_w1 <- complete_train_x_w1[,-grep("weight",colnames(complete_train_x_w1))]
complete_validation_x_w1 <- complete_validation_x_w1[,-grep("weight",colnames(complete_validation_x_w1))]
  
### run models
xgb_list_w1 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=complete_train_x_w1,train_y=complete_train_y_w1,validation_x=complete_validation_x_w1,validation_y=complete_validation_y_w1,weights = weights1,weight_lab = 1))
xgb_validation_results_w1 <- ldply(xgb_list_w1,data.frame)
    
xgb_list_w2 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=complete_train_x_w1,train_y=complete_train_y_w1,validation_x=complete_validation_x_w1,validation_y=complete_validation_y_w1,weights = weights2,weight_lab = 2))
xgb_validation_results_w2 <- ldply(xgb_list_w2,data.frame)

xgb_list_w3 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=complete_train_x_w1,train_y=complete_train_y_w1,validation_x=complete_validation_x_w1,validation_y=complete_validation_y_w1,weights = weights3,weight_lab = 3))
xgb_validation_results_w3 <- ldply(xgb_list_w3,data.frame)

xbg_weights_comb <- rbind(xgb_validation_results_w1,xgb_validation_results_w2,xgb_validation_results_w3)
# write.csv(xbg_weights_comb,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_weight_tune.csv")
xbg_weights_comb <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_weight_tune.csv")

ggplot(xbg_weights_comb,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(weights~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))


### up the weights
weights4 <- weights1^4
weights5 <- weights1^5
weights6 <- weights1^6

xgb_list_w4 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=complete_train_x_w1,train_y=complete_train_y_w1,validation_x=complete_validation_x_w1,validation_y=complete_validation_y_w1,weights = weights4,weight_lab = 4))
xgb_validation_results_w4 <- ldply(xgb_list_w4,data.frame)

xgb_list_w5 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=complete_train_x_w1,train_y=complete_train_y_w1,validation_x=complete_validation_x_w1,validation_y=complete_validation_y_w1,weights = weights5,weight_lab = 5))
xgb_validation_results_w5 <- ldply(xgb_list_w5,data.frame)

xgb_list_w6 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=complete_train_x_w1,train_y=complete_train_y_w1,validation_x=complete_validation_x_w1,validation_y=complete_validation_y_w1,weights = weights6,weight_lab = 6))
xgb_validation_results_w6 <- ldply(xgb_list_w6,data.frame)

xbg_weights_comb_2 <- rbind(xgb_validation_results_w4,xgb_validation_results_w5,xgb_validation_results_w6)
# write.csv(xbg_weights_comb_2,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_weight_tune_2.csv")
xbg_weights_comb_2 <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_weight_tune_2.csv")

ggplot(xbg_weights_comb_2,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(weights~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))

##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                         Macro + Validation + All                           #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

# parameters to check
parmgrid <- expand.grid(nrounds = c(300,500), # number of trees
                        eta = c(0.01,0.1,0.3), # learning rate
                        max_depth = c(2,5,10), # depth of trees
                        gamma = 0,
                        colsample_bytree = 1, 
                        min_child_weight = 1,
                        subsample = 1)
parmlist <- setNames(split(parmgrid, seq(nrow(parmgrid))), rownames(parmgrid))

# run models
xgb_list_macro <- lapply(parmlist,function(x) run_xbg_general(x,train_x=macro_train_x,train_y=macro_train_y,validation_x=macro_validation_x,validation_y=macro_validation_y))
xgb_validation_macro <- ldply(xgb_list_macro,data.frame)

# write.csv(xgb_validation_macro,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_all.csv")
xgb_validation_macro <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_all.csv")

# plot
ggplot(xgb_validation_macro,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))

### get variable importance

# run model
x <- parmlist[[10]]
set.seed(50)
xgb <- xgboost(data=train_x,label=train_y, weight=weights,
               params = as.list(as.vector(x[-grep("nrounds",names(x))])), 
               nrounds=nrounds,verbose=F)
importance <- xgb.importance(feature_names = colnames(train_x),xgb)

# alter importance table
importance$feature_order <- factor(importance$Feature,ordered = T)
importance$macro <- ifelse((importance$Feature %in% names(train_complete)),"House","Macro")

# plot
ggplot(importance,aes(x=feature_order,y=Gain,fill=macro))+geom_bar(stat="identity")+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

##############################################################################
#                                                                            #
#                             Examine Data                                   #
#                                                                            #
##############################################################################

##### correlations using all entries

# get data
rel_rows <- c(names(macro_data),"price_doc"); rel_rows <- rel_rows[-1]
macro_only <- macro_train_complete[,which(names(macro_train_complete) %in% rel_rows)]
macro_only <- data.matrix(macro_only)

# cor with price
macro_cor <- cor(macro_only,method="spearman",use="pairwise.complete.obs")
macro_cor_price <- macro_cor[1,]
macro_cor_price[order(macro_cor_price)]

##### average over month
rel_rows <- c(names(macro_data),"price_doc")
macro_only <- macro_train_complete[,which(names(macro_train_complete) %in% rel_rows)]
macro_only$timestamp <- format(macro_only$timestamp,format="%m/%Y")

# cor by month
macro_month <- aggregate(data.matrix(macro_only[,2:101]),list(macro_only$timestamp),function(x) mean(x,na.rm = T))
macro_cor_month <- cor(macro_month[,2:101],use="pairwise.complete.obs",method="spearman")
macro_corr_month_price <- macro_cor_month[1,]
macro_corr_month_price[order(macro_corr_month_price)]

##############################################################################
#                                                                            #
#                            Include only 5                                  #
#                                                                            #
##############################################################################

# parameters to check
parmgrid <- expand.grid(nrounds = c(300,500), # number of trees
                        eta = c(0.01,0.1,0.3), # learning rate
                        max_depth = c(2,5,10), # depth of trees
                        gamma = 0,
                        colsample_bytree = 1, 
                        min_child_weight = 1,
                        subsample = 1)
parmlist <- setNames(split(parmgrid, seq(nrow(parmgrid))), rownames(parmgrid))

# run models
xgb_list_macro2 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=macro_train_x2,train_y=macro_train_y2,validation_x=macro_validation_x2,validation_y=macro_validation_y2))
xgb_validation_macro2 <- ldply(xgb_list_macro2,data.frame)

# save
# write.csv(xgb_validation_macro2,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_5.csv")
xgb_validation_macro2 <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_5.csv")

# plot
ggplot(xgb_validation_macro2,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))

#### importance

# run model
x <- parmlist[[15]]
set.seed(50)
xgb <- xgboost(data=macro_train_x2,label=macro_train_y2, weight=weights,
               params = as.list(as.vector(x[-grep("nrounds",names(x))])), 
               nrounds=x$nrounds,verbose=F)
importance <- xgb.importance(feature_names = colnames(macro_train_x2),xgb)

# alter importance table
importance$feature_order <- factor(importance$Feature,ordered = T)
importance$macro <- ifelse((importance$Feature %in% names(train_complete)),"House","Macro")

# plot
ggplot(importance,aes(x=feature_order,y=Gain,fill=macro))+geom_bar(stat="identity")+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

##############################################################################
#                                                                            #
#                            Include only 10                                 #
#                                                                            #
##############################################################################

# parameters to check
parmgrid <- expand.grid(nrounds = c(300,500), # number of trees
                        eta = c(0.01,0.1,0.3), # learning rate
                        max_depth = c(2,5,10), # depth of trees
                        gamma = 0,
                        colsample_bytree = 1, 
                        min_child_weight = 1,
                        subsample = 1)
parmlist <- setNames(split(parmgrid, seq(nrow(parmgrid))), rownames(parmgrid))

# run models
xgb_list_macro3 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=macro_train_x3,train_y=macro_train_y3,validation_x=macro_validation_x3,validation_y=macro_validation_y3))
xgb_validation_macro3 <- ldply(xgb_list_macro3,data.frame)

# plot
ggplot(xgb_validation_macro3,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))


##############################################################################
#                                                                            #
#                        Include 5 and only 2014 data                        #
#                                                                            #
##############################################################################

# parameters to check
parmgrid <- expand.grid(nrounds = c(300,500), # number of trees
                        eta = c(0.01,0.1,0.3), # learning rate
                        max_depth = c(2,5,10), # depth of trees
                        gamma = 0,
                        colsample_bytree = 1, 
                        min_child_weight = 1,
                        subsample = 1)
parmlist <- setNames(split(parmgrid, seq(nrow(parmgrid))), rownames(parmgrid))

# run models
xgb_list_macro_14 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=macro_train_x_14,train_y=macro_train_y_14,validation_x=macro_validation_x_14,validation_y=macro_validation_y_14))
xgb_validation_macro_14 <- ldply(xgb_list_macro_14,data.frame)

# write.csv(xgb_validation_macro_14,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_14.csv")
xgb_validation_macro_14 <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_14.csv")

# plot
ggplot(xgb_validation_macro_14,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))

##############################################################################
#                                                                            #
#                        Include 5 and MORE parameter values                 #
#                                                                            #
##############################################################################

# parameters to check
parmgrid <- expand.grid(nrounds = c(300,500,700), # number of trees
                        eta = c(0.005,0.01,0.05,0.1,0.15), # learning rate
                        max_depth = c(8,10,12,14), # depth of trees
                        gamma = 0,
                        colsample_bytree = 1, 
                        min_child_weight = 1,
                        subsample = 1)
parmlist <- setNames(split(parmgrid, seq(nrow(parmgrid))), rownames(parmgrid))

# run models
xgb_list_macro4 <- lapply(parmlist,function(x) run_xbg_general(x,train_x=macro_train_x2,train_y=macro_train_y2,validation_x=macro_validation_x2,validation_y=macro_validation_y2))
xgb_validation_macro4 <- ldply(xgb_list_macro4,data.frame)

#write.csv(xgb_validation_macro4,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_extra_params.csv")
xgb_validation_macro4 <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_extra_params.csv")

# plot
ggplot(xgb_validation_macro4,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))

xgb_validation_macro4[which(xgb_validation_macro4$RMSLE==min(xgb_validation_macro4$RMSLE)),]


##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                         Tuning other Parameters                            #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

###### colsample_bytree

# parameters to check
parmgrid2 <- expand.grid(nrounds = c(300,500), # number of trees
                         eta = c(0.01,0.1,0.3), # learning rate
                         max_depth = c(2,5,10), # depth of trees
                         gamma = 0,
                         colsample_bytree = c(0.25,0.5,0.75), 
                         min_child_weight = 1,
                         subsample = 1)
parmlist2 <- setNames(split(parmgrid2, seq(nrow(parmgrid2))), rownames(parmgrid2))

# run models
xgb_list_macro_parm <- lapply(parmlist2,function(x) run_xbg_general(x,train_x=macro_train_x2,train_y=macro_train_y2,validation_x=macro_validation_x2,validation_y=macro_validation_y2))
xgb_validation_macro_parm <- ldply(xgb_list_macro_parm,data.frame)

# save
# write.csv(xgb_validation_macro_parm,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_tune_parm.csv")

# plot
ggplot(xgb_validation_macro_parm,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(colsample_bytree~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))

###### subsample

# parameters to check
parmgrid3 <- expand.grid(nrounds = c(300,500), # number of trees
                         eta = c(0.01,0.1,0.3), # learning rate
                         max_depth = c(2,5,10), # depth of trees
                         gamma = 0,
                         colsample_bytree = 1, 
                         min_child_weight = 1,
                         subsample = c(0.5,0.75))
parmlist3 <- setNames(split(parmgrid3, seq(nrow(parmgrid3))), rownames(parmgrid3))

# run models
xgb_list_macro_parm <- lapply(parmlist3,function(x) run_xbg_general(x,train_x=macro_train_x2,train_y=macro_train_y2,validation_x=macro_validation_x2,validation_y=macro_validation_y2))
xgb_validation_macro_parm <- ldply(xgb_list_macro_parm,data.frame)

# save
# write.csv(xgb_validation_macro_parm,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_tune_parm.csv")
xgb_validation_macro_parm <- read.csv("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_tune_parm.csv")


ggplot(xgb_validation_macro_parm,aes(y=RMSLE,x=max_depth,color=factor(eta)))+
  geom_line()+facet_grid(subsample~nrounds)+labs(color="Shrinkage\nParameter")+xlab("Tree Depth")+
  scale_y_continuous(limits = c(0.23,0.34))

#write.csv(xgb_validation_macro_parm,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Backup_results/xgb_macro_tune_parm_subsample.csv")




##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                         Examining best models                              #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

parmgrid <- expand.grid(nrounds = c(300,500), # number of trees
                        eta = c(0.01,0.1,0.3), # learning rate
                        max_depth = c(2,5,10), # depth of trees
                        gamma = 0,
                        colsample_bytree = 1, 
                        min_child_weight = 1,
                        subsample = 1)
parmlist <- setNames(split(parmgrid, seq(nrow(parmgrid))), rownames(parmgrid))
weights=NULL

#### best model (NO MACRO)

x<- parmlist[[10]]
set.seed(50)
xgb1 <- xgboost(data=data.matrix(complete_train_x_14),label=complete_train_y_14, weight=weights,
               params = as.list(as.vector(x[-grep("nrounds",names(x))])), 
               nrounds=x$nrounds,verbose=F)

#### best model (MACRO)
x<- parmlist[[15]]
set.seed(50)
xgb2 <- xgboost(data=macro_train_x2,label=macro_train_y2, weight=weights,
                params = as.list(as.vector(x[-grep("nrounds",names(x))])), 
                nrounds=x$nrounds,verbose=F)

#### make predictions
pred1 <- predict(xgb1,data.matrix(complete_validation_x)) ##### train on 14 predict on all
pred2 <- predict(xgb2,macro_validation_x2)

# data 
pred_df <- data.frame(macro_validation_y2,pred1,pred2)
pred_df$pred_Higher_1 <- ifelse(pred_df$pred1>pred_df$macro_validation_y2,1,0) 
pred_df$pred_Higher_2 <- ifelse(pred_df$pred2>pred_df$macro_validation_y2,1,0)
pred_df$pred_off_same_dir <- ifelse(sign(pred_df$pred2-pred_df$macro_validation_y2)==sign(pred_df$pred1-pred_df$macro_validation_y2),1,0) 
pred_df$pred_2_closer <- ifelse(abs(pred_df$pred2-pred_df$macro_validation_y2)<abs(pred_df$pred1-pred_df$macro_validation_y2),1,0) 


# look at rmlse
data2 = data.frame(obs=(pred_df$pred1),pred=pred_df$macro_validation_y2)
data1 = data.frame(obs=(pred_df$pred1+pred_df$pred2)/2,pred=pred_df$macro_validation_y2)
rmsle(data1)
rmsle(data2)

# make quick table
Situation <- c("Both Models Off in Same Direction","Model 1 Prediction Overestimates","Model 2 Prediction Overestimates","Model 2 Prediction Closer")
Proportion <- c(0.8433076,0.3288288,0.3870656,0.5563063)
df1 <- data.frame("Situation"=Situation,"Proportion"=round(Proportion,3))
kable(df1,row.names = F)











