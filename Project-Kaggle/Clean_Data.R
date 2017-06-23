### NOTES
### 
###   - make sure categorical variables coded correctly 
###

require(data.table)
require(knitr)
require(ggplot2)
require(gridExtra)
require(mice)
require(dplyr)
require(mice)
require(imputeMissings)
require(lazyeval)


##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                           Make changes to data *                           #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

### read data and examine
train_data <- fread("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Data/train.csv")
test_data <- fread("/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/Data/test.csv")
train_data <- data.frame(train_data)
test_data <- data.frame(test_data)

### create log transformed price
#train_data$log_price <- log(train_data$price_doc)

##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                         Basic Data Characteristics                         #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

# dimension
data_dim <- dim(train_data) # 30471 rows
data_dim_2 <- dim(test_data) 

# missing values - should prob just make fn out of this
data_na <- apply(train_data,2,function(x) length(which(is.na(x))))
data_na <- data_na[which(data_na>0)]
data_na_percentage <- round(data_na/data_dim[1],3)

data_na_2 <- apply(test_data,2,function(x) length(which(is.na(x))))
data_na_2 <- data_na_2[which(data_na_2>0)]
data_na_percentage_2 <- round(data_na_2/data_dim[1],3)

# missing values - df
missing_df <- data.frame(names(data_na_percentage),data_na_percentage)
missing_df_2 <- data.frame(names(data_na_percentage_2),data_na_percentage_2)
colnames(missing_df) <- c("Variable","Percent Missing (Train)")
colnames(missing_df_2) <- c("Variable","Percent Missing (Test)")
missing_df <- merge(missing_df,missing_df_2,all=T,by="Variable")
missing_df <- missing_df[order(missing_df$"Percent Missing (Train)",decreasing = T),]
missing_df[is.na(missing_df)] <- "-" # turn missing into "-"
kable(missing_df,align="c",row.names = FALSE)

# missing values - split df for display
missing_df_1 <- missing_df[1:round(dim(missing_df)[1]/2),]
missing_df_2 <- missing_df[(round(dim(missing_df)[1]/2)+1):dim(missing_df)[1],]
kable(missing_df_1,align="c",row.names = F); kable(missing_df_2,align="c",row.names = F)

# are house features less likely to be missing toward the end
missing_buildyear <- ifelse(is.na(train_data$build_year),1,0)
a <- summarize(group_by(train_data,year(as.Date(train_data$timestamp))),missing_by = length(which(is.na(build_year)))/length(build_year))

### correlation - numeric
cor(train_data[sapply(train_data,is.numeric)],use="pairwise.complete.obs")

### outcome variable

# plot outcome variable
p1 <- ggplot(train_data,aes(x=price_doc))+geom_density(color="dodgerblue4",fill="dodgerblue4")+
  theme_bw()+theme(legend.position="none")+xlab("Price")+ylab("Density")+geom_rug(color="gray45")
p2 <- ggplot(train_data,aes(x=log(price_doc)))+geom_density(color="dodgerblue4",fill="dodgerblue4")+
  theme_bw()+theme(legend.position="none")+xlab("Log Price")+ylab("Density")+geom_rug(color="gray45")
grid.arrange(p1,p2,ncol=2)

### sales by date

# get date range
train_data$timestamp <- as.Date(train_data$timestamp,"%Y-%m-%d")
min_date <- min(train_data$timestamp)
max_date <- max(train_data$timestamp)
num_days <- max_date-min_date

# plot number of sales per week
ggplot(train_data,aes(x=timestamp))+geom_histogram(bins=num_days/7)+
  xlab("Date (Week)")+ylab("Count")+theme_bw()

# plot price of sales by day
ggplot(train_data,aes(x=timestamp,y=log_price))+geom_point()+
  xlab("Date")+ylab("Log Price")+geom_smooth(method="lm",color="dodgerblue4")
ggplot(train_data,aes(x=timestamp,y=price_doc))+geom_point()+
  xlab("Date")+ylab("Price")+geom_smooth(method="lm",color="dodgerblue4")

### sales by weekday

### time variables
train_data$weekday <- weekdays(train_data$timestamp)

# number of sales by weekday
ggplot(train_data,aes(x=weekday))+geom_bar()+xlab("")+ylab("Count")

# average price of sales by weekday
ggplot(train_data,aes(x=weekday,y=log_price))+geom_boxplot()+ xlab("")+
  ylab("Log Price")+theme(axis.text.x=element_text(angle=30, hjust=1))

### sales by month

# month variable
train_data$month <- months(train_data$timestamp)
train_data$month <- ordered(train_data$month, month.name)

# number of sales by weekday
ggplot(train_data,aes(x=month))+geom_bar()+xlab("")+ylab("Count")

# average price of sales by month
ggplot(train_data,aes(x=month,y=log_price))+geom_boxplot()+
  theme(axis.text.x=element_text(angle=30, hjust=1))+xlab("Month")+
  ylab("Log Price")

##############################################################################
#                                                                            #
#                     plot missingness by time                               #
#                                                                            #
##############################################################################

### plot of missing data by date - all vars
train_temp <- train_data
train_temp$na_obs <- apply(train_temp,1,function(x) length(which(is.na(x)))/length(x)) # percent missing 
mean_perc_miss_date <- summarize(group_by(train_temp,timestamp),avg_perc_miss = mean(na_obs))

# plot by day - all values
mean_perc_miss_date$timestamp <- as.Date(mean_perc_miss_date$timestamp)
p1 <- ggplot(mean_perc_miss_date,aes(x=timestamp,y=avg_perc_miss))+geom_line()+
  xlab("")+ylab("Proportion Missing (All)")

### missing data by date - individual vars

#individual apartment vars
train_temp <- train_data[,c(2:12)]
train_temp$na_obs <- apply(train_temp,1,function(x) length(which(is.na(x)))/length(x)) # percent missing 
mean_perc_miss_date <- summarize(group_by(train_temp,timestamp),avg_perc_miss = mean(na_obs))
individual_apt <- data.frame(mean_perc_miss_date); individual_apt$Variable <- "House Characteristics"

mean_perc_miss_date$timestamp <- as.Date(mean_perc_miss_date$timestamp)
p2 <- ggplot(mean_perc_miss_date,aes(x=timestamp,y=avg_perc_miss))+geom_line()+
  xlab("")+ylab("Proportion Missing (House)")

# cafe vars 
train_temp <- train_data[,c(2,grep("cafe",colnames(train_data)))]
train_temp$na_obs <- apply(train_temp,1,function(x) length(which(is.na(x)))/length(x)) # percent missing 
mean_perc_miss_date <- summarize(group_by(train_temp,timestamp),avg_perc_miss = mean(na_obs))
cafe <- data.frame(mean_perc_miss_date); cafe$Variable <- "Cafe Variables"

mean_perc_miss_date$timestamp <- as.Date(mean_perc_miss_date$timestamp)
p3 <- ggplot(mean_perc_miss_date,aes(x=timestamp,y=avg_perc_miss))+geom_line()+
  xlab("")+ylab("Proportion Missing (Cafe)")

### plot all
grid.arrange(p1,p2,p3)

##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#             Variable transformation and anomylous observations             #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

##### outcome variable

### plot price_doc
hist(train_data$price_doc,breaks=1000)
sort(train_data$price_doc)
sort(train_data$price_doc,decreasing = T)

# look at top obs - are they large houses
low_price <- train_data[which(log(train_data$price_doc)<14.55),c('price_doc','full_sq','build_year','num_room')]
high_price <- train_data[which(train_data$price_doc>50000000),c('price_doc','full_sq','build_year','num_room')]
mean_sq <- mean(train_data$full_sq) # 54

# !! property type
ggplot(train_data,aes(x=log_price))+geom_histogram(bins=200)+facet_grid(~product_type)+
  xlab("Log Price")+ylab("Count")

# histogram of log values
hist1 <- ggplot(train_data,aes(x=log(price_doc)))+geom_histogram(bins=200)+xlab("Log Price")+
  ylab("Count")+geom_vline(xintercept=14.55,color="red",linetype="dotted")+
  scale_x_continuous(limits=c(12,18))

### identify anomylous observations
dim(train_data[which(log(train_data$price_doc)<14.55),]) # 2097 with low price
dim(train_data[which(log(train_data$price_doc)<14.55 & train_data$full_sq > 
                       quantile(train_data$full_sq,0.05)),]) # 1877 above 10th percentil square footage
 
# remove anomolies and plot
train_data$price_doc[which(log(train_data$price_doc)<14.55 & train_data$full_sq > quantile(train_data$full_sq,0.05))] <- NA
hist2 <- ggplot(train_data,aes(x=log(price_doc)))+geom_histogram(bins=200)+xlab("Log Price")+
  ylab("Count")+geom_vline(xintercept=14.55,color="red",linetype="dotted")+
  scale_x_continuous(limits=c(12,18))

# plots before and after removing
grid.arrange(hist1,hist2)

##############################################################################
#                                                                            #
#                     examine predictor distributions                        #
#                                                                            #
##############################################################################

### full_sq
hist(train_data$full_sq,breaks=1000)
sort(train_data$full_sq,decreasing=TRUE) # one way too high, two zeros
sort(train_data$full_sq) # one way too high, two zeros

### life_sq
hist(train_data$life_sq,breaks=1000)
sort(train_data$life_sq,decreasing=TRUE) # one way too high, 45 zeros

### full_sq - life_sq
train_data$full_sq[which(train_data$full_sq>1000)] <- NA
train_data$full_sq[which(train_data$full_sq<=1)] <- NA
train_data$life_sq[which(train_data$life_sq>1000|train_data$life_sq<=1)] <- NA

# plot and look at extreme values
hist(train_data$full_sq-train_data$life_sq,breaks=200)
sort(train_data$full_sq-train_data$life_sq)
sort(train_data$full_sq-train_data$life_sq,decreasing = T)

# look at prices, values of life_sq and full_sq
sq_df <- data.frame(cbind(train_data$full_sq,train_data$life_sq,train_data$full_sq-train_data$life_sq,train_data$price_doc,train_data$num_room))
names(sq_df) <- c("full_sq","life_sq","diff_sq","price","num_rooms")
sq_df[order(sq_df$diff_sq),]
sq_df[order(sq_df$diff_sq,decreasing = T),]

### floor - those with max floor < floor mostly have 0 or 1 for max floor
floor_anom <- cbind(train_data$floor[which(train_data$floor > train_data$max_floor)]
      ,train_data$max_floor[which(train_data$floor > train_data$max_floor)])

### material - what does this mean - should we make these indicators!!!
train_data$material <- as.factor(train_data$material)

### build_year
hist(train_data$build_year)
sort(train_data$build_year,decreasing=T)
sort(train_data$build_year)

### kitchen area
hist(train_data$kitch_sq,breaks=1500)
sort(train_data$kitch_sq)
sort(train_data$kitch_sq[which(train_data$kitch_sq>1)])
sort(train_data$kitch_sq,decreasing = T)

### state - do we keep this as continuous
hist(train_data$state,breaks=100)
sort(train_data$state,decreasing = T)

### full_sq, life_sq, and kitch_sq - relationship in complete cases
sq_comp <- train_data[which(train_data$full_sq < train_data$life_sq | train_data$life_sq < train_data$kitch_sq),c(1:12,dim(train_data)[2])]

##############################################################################
#                                                                            #
#                            transform variables *                           #
#                                                                            #
##############################################################################

anomalous_values <- function(train_data) {
  
  ### compare full_sq, life_sq, and kitch_sq
  
  # set 0 and 1 values to NA
  train_data$full_sq[which(train_data$full_sq<=1)] <- NA
  train_data$life_sq[which(train_data$life_sq<=1)] <- NA
  train_data$kitch_sq[which(train_data$kitch_sq<=1)] <- NA
  
  ### full_sq
  train_data$full_sq[which(train_data$full_sq<10)] <- NA
  
  ### life_sq > full_sq
  train_data$life_sq <- ifelse(train_data$life_sq>train_data$full_sq & train_data$life_sq-train_data$full_sq<5,train_data$full_sq,train_data$life_sq)
  train_data$life_sq <- ifelse(train_data$life_sq>train_data$full_sq & train_data$life_sq-train_data$full_sq>=5,NA,train_data$life_sq)
  
  ### kitch_sq > life_sq
  train_data$life_sq <- ifelse((train_data$kitch_sq>train_data$life_sq) & ((train_data$full_sq-train_data$life_sq)>train_data$life_sq),NA,train_data$life_sq)
  train_data$kitch_sq <- ifelse((train_data$kitch_sq>train_data$life_sq) & ((train_data$full_sq-train_data$life_sq)<=train_data$life_sq),NA,train_data$kitch_sq)
  
  ### remaining sq features
  train_data$full_sq[which(train_data$full_sq>1000)] <- NA
  
  ### build_year
  train_data$build_year[which(train_data$build_year<1600|train_data$build_year>2018)] <- NA
  
  ### max_floor
  train_data$max_floor[which(train_data$max_floor<train_data$floor)] <- NA
  
  ### state
  train_data$state[which(train_data$state>4)] <- NA
  
  ### num_room
  train_data$num_room[which(train_data$num_room==0)] <- NA
  
  return(train_data)
  
}

##############################################################################
#                                                                            #
#                             rescale variables *                            #
#                                                                            #
##############################################################################

rescale_variables <- function(train_data) {
  
  ### build materials and years
  
  # get names
  build_names <- colnames(train_data)[grep("build_count",colnames(train_data))]
  build_names_material <- build_names[which(!grepl("1",build_names) & !grepl("with",build_names))]
  build_names_year <- build_names[which(grepl("1",build_names) & !grepl("with",build_names))]
  
  # change variables
  train_data[,build_names_material] <- apply(train_data[,build_names_material],2,function(x) x/train_data$raion_build_count_with_material_info)
  train_data[,build_names_year] <- apply(train_data[,build_names_year],2,function(x) x/train_data$raion_build_count_with_builddate_info)
  
  ### express ranion counts as a ratio of total population
  raion_names <- c('preschool_education_centers_raion','school_education_centers_raion','school_education_centers_top_20_raion','hospital_beds_raion','healthcare_centers_raion','university_top_20_raion','sport_objects_raion','additional_education_raion','culture_objects_top_25_raion','shopping_centers_raion','office_raion')
  train_data[,raion_names] <- apply(train_data[,raion_names],2,function(x) (x/train_data$raion_popul)*1000)
  
  ### recode 'yes' / 'no'
  train_data$thermal_power_plant_raion <- ifelse(train_data$thermal_power_plant_raion=="yes",1,0)
  train_data$incineration_raion <- ifelse(train_data$incineration_raion=="yes",1,0)
  train_data$oil_chemistry_raion <- ifelse(train_data$oil_chemistry_raion=="yes",1,0)
  train_data$radiation_raion <- ifelse(train_data$radiation_raion=="yes",1,0)
  train_data$railroad_terminal_raion <- ifelse(train_data$railroad_terminal_raion=="yes",1,0)
  train_data$big_market_raion <- ifelse(train_data$big_market_raion=="yes",1,0)
  train_data$nuclear_reactor_raion <- ifelse(train_data$nuclear_reactor_raion=="yes",1,0)
  train_data$detention_facility_raion <- ifelse(train_data$detention_facility_raion=="yes",1,0)
  train_data$product_type <- ifelse(train_data$product_type=='Investment',1,0)
  train_data$culture_objects_top_25 <- ifelse(train_data$culture_objects_top_25=="yes",1,0)
  train_data$water_1line <- ifelse(train_data$water_1line=="yes",1,0)
  train_data$railroad_1line <- ifelse(train_data$railroad_1line=="yes",1,0)
  
  ### sub_area
  area_names <- table(train_data$sub_area)
  area_names_10 <- names(area_names[which(area_names<=10)]) 
  area_names_100 <- names(area_names[which(area_names>10 & area_names<=100)])
  area_names_250 <- names(area_names[which(area_names>101 & area_names<250)])
  
  # recode
  train_data$sub_area[which(train_data$sub_area %in% area_names_10)] <- "less_10"
  train_data$sub_area[which(train_data$sub_area %in% area_names_100)] <- "less_100"
  train_data$sub_area[which(train_data$sub_area %in% area_names_250)] <- "less_250"
  
  # expand 
  area_expand <- model.matrix(~sub_area-1,train_data)
  train_data <- data.frame(cbind(train_data,area_expand))
  
  #### rescale age variables
  male_names <- colnames(train_data)[which(grepl("male",colnames(train_data)) & !grepl("f",colnames(train_data)))]
  female_names <- colnames(train_data)[which(grepl("female",colnames(train_data)) & !grepl("female_f",colnames(train_data)))]
  all_names <- colnames(train_data)[which(grepl("_all",colnames(train_data)) & !grepl("full",colnames(train_data)))]
  
  # recalculate total population 
  train_data$all_new <- train_data$young_all + train_data$work_all + train_data$ekder_all
  train_data$female_new <- train_data$young_female + train_data$work_female + train_data$ekder_female
  train_data$male_new <- train_data$young_male + train_data$work_male + train_data$ekder_male
  
  # replace
  train_data[,male_names] <- apply(train_data[,male_names],2,function(x) x/train_data$male_new)
  train_data[,female_names] <- apply(train_data[,female_names],2,function(x) x/train_data$female_new)
  train_data[,all_names] <- apply(train_data[,all_names],2,function(x) x/train_data$all_new)
  
  return(train_data)
}

##############################################################################
#                                                                            #
#                         individual apartments *                            #
#                                                                            #
##############################################################################

individual_apartment <- function(train_data) {
    
  # create apt_id
  train_data$apt_id <- paste(train_data$sub_area,"_",train_data$metro_km_avto,sep="")
  train_data <- train_data[order(train_data$apt_id),]
  train_lim <- train_data[order(train_data$apt_id),c(2:10,dim(train_data)[2])]
  
  # fill in build year - should make a fn to do this
  train_data <- train_data %>%
    group_by(apt_id) %>%
    mutate(build_year=ifelse(
      is.na(build_year) & # only look for missing values
        length(apt_id)>1 & # more than one record
        length(setdiff(unique(build_year),NA))==1, # one unique value (aside from NA)
      setdiff(unique(build_year),NA),build_year))
  
  # fill in max_floor - should make a fn to do this
  train_data <- train_data %>%
    group_by(apt_id) %>%
    mutate(max_floor=ifelse(
      is.na(max_floor) & # only look for missing values
        length(apt_id)>1 & # more than one record
        length(setdiff(unique(max_floor),NA))==1, # one unique value (aside from NA)
      setdiff(unique(max_floor),NA), max_floor))
  
  # fill in material - should make a fn to do this
  train_data <- train_data %>%
    group_by(apt_id) %>%
    mutate(material=ifelse(
      is.na(material) & # only look for missing values
        length(apt_id)>1 & # more than one record
        length(setdiff(unique(material),NA))==1, # one unique value (aside from NA)
      setdiff(unique(material),NA), material))
  
  return(train_data)
}
  
##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                          Dealing with Missing Data *                       #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

##############################################################################
#                                                                            #
#                        handle individual values *                          #
#                        imputed in all datasets                             #
#                                                                            #
##############################################################################

cafe_variables <- function(train_data) {
  
  ### cafe variables
  cafe_ind <- grep("cafe",colnames(train_data))
  colnames(train_data)[cafe_ind]
  cafe_data <- train_data[,cafe_ind]
  
  # impute
  cafe_imputed <- mice(cafe_data,m=1,method='pmm',seed=500)
  cafe_complete <- complete(cafe_imputed)
  apply(cafe_complete,2,function(x) length(unique(x))) # number of unique values 
  
  train_data[,cafe_ind] <- cafe_complete
  
  return(train_data)
}

##############################################################################
#                                                                            #
#                           variable reduction *                             #
#                                                                            #
##############################################################################

reduce_variables <- function(train_data) {
  
  #### cafe vars
  cafe_ind <- grep("cafe",colnames(train_data))
  cafe_complete <- train_data[,cafe_ind]
  
  # prepare for PCA
  cafe_comp_l <- log(cafe_complete+1) # LOG
  cafe_comp_l <- impute(cafe_comp_l)
  
  # PCA - data not preprocessed
  cafe_principal <- prcomp(cafe_comp_l,scale=T,center=T)
  plot(cafe_principal,type='l') # plot variances - 3 to 4 looks okay
  summary(cafe_principal) # prop var explained
  cafe_complete_prcomp <- predict(cafe_principal,cafe_comp_l) # predict principal components for all
  
  # remove training data cafe variables
  train_data$cafe_PC1 <- cafe_complete_prcomp[,1]
  train_data$cafe_PC2 <- cafe_complete_prcomp[,2]
  train_data$cafe_PC3 <- cafe_complete_prcomp[,3]
  train_data$cafe_PC4 <- cafe_complete_prcomp[,4]
  train_data$cafe_PC5 <- cafe_complete_prcomp[,5]
  
  #### distance vars
  
  # get data
  distance_names <- colnames(train_data)[which(grepl("00",colnames(train_data)) & !grepl("cafe",colnames(train_data)) )]
  train_data_distance <- train_data[,distance_names]
  
  # prepare data
  distance_l <- log(train_data_distance+1)
  distance_l <- impute(distance_l) 
  
  # run principal components
  distance_principal <- prcomp(distance_l,scale=T,center=T)
  plot(distance_principal,type='l') # 3 to 4 looks okay
  summary(distance_principal)
  distance_prcomp <- predict(distance_principal,distance_l)
  
  # create vars
  train_data$distance_PC1 <- distance_prcomp[,1]
  train_data$distance_PC2 <- distance_prcomp[,2]  
  train_data$distance_PC3 <- distance_prcomp[,3]  
  train_data$distance_PC4 <- distance_prcomp[,4]  
  train_data$distance_PC5 <- distance_prcomp[,5]
  
  ### return
  return(train_data)
  
}


##############################################################################
#                                                                            #
#                         imputations continues *                            #
#                 variables only imputed in complete dataset                 #
#                                                                            #
##############################################################################

school_variables <- function(train_data) {
  
  ### school variables
  school_vars <- train_data[,18:24]
  school_vars <- mice(school_vars,m=1,method='pmm',seed=500)
  school_vars <- complete(school_vars)
  train_data[,18:24] <- school_vars
  
  return(train_data)
}

build_variables <- function(train_data){

  # impute - build year and material
  build_names <- c("full_sq","life_sq","floor","max_floor","material","build_year","num_room","kitch_sq","state",colnames(train_data)[grep("build_count",colnames(train_data))])
  build_year_material <- train_data[,c(build_names)]
  build_year_material <- mice(build_year_material,m=1,method='pmm',seed=500)
  build_year_material <- complete(build_year_material)
  train_data[,c(build_names)] <- build_year_material
  
  return(train_data)
}

##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                             Make New Variables *                           #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

create_variables <- function(train_data) {
  
  ############################################################################## 
  #######################################                 Created              #
  ##############################################################################
  
  ### top floor or top half of all floors
  train_data$top_floor <- ifelse(is.na(train_data$max_floor) | is.na(train_data$floor) | train_data$floor != train_data$max_floor,0,1)
  train_data$bottom_floor <- ifelse(is.na(train_data$floor) | train_data$floor != 1,0,1)
  train_data$building_less_5_floors <-ifelse(is.na(train_data$max_floor) | train_data$max_floor > 5,0,1)
  train_data$building_more_15_floors <-ifelse(is.na(train_data$max_floor) | train_data$max_floor < 15,0,1)
  
  ### balcony, etc.
  train_data$balcony_etc <- train_data$full_sq-train_data$life_sq 
  
  ### living area per room
  train_data$sq_per_room <- train_data$life_sq/train_data$num_room
  
  ### % of floors above apartment
  #train_data$per_floor_above <- (train_data$max_floor-train_data$floor)/train_data$max_floor
  #train_data$per_floor_above[which(train_data$per_floor_above<0)] <- NA
  
  ### number of sales that month 
  train_data$temp_date <- format(train_data$timestamp,format="%m/%Y")
  monthly_sales_data <- summarize(group_by(train_data,temp_date), num_sales_month=length(id))
  train_data <- merge(train_data,monthly_sales_data,by="temp_date",all=T)
  train_data <- train_data[,which(colnames(train_data) != "temp_date")]
  
  ### ratio of school aged children to seats
  preschool_ratio <- train_data$children_preschool/train_data$preschool_quota
  school_ratio <- train_data$children_school/train_data$school_quota

  ### traffic
  train_data$traffic_metro <- train_data$metro_min_avto/train_data$metro_km_avto
  
  ### counts within apartment
  train_data <- train_data %>%
    group_by(apt_id,build_year,max_floor) %>%
    mutate(complex_num_sales = length(apt_id),complex_avg_sq=median(as.numeric(full_sq),na.rm=T))
  train_data$complex_num_sales_stand <- train_data$complex_num_sales/(train_data$max_floor+1)
  
  ### walk score to essentials
  train_data$prox_score_essentials <- ifelse(train_data$park_km<2,1,0)+ 
    ifelse(train_data$school_km<2,1,0)+ 
    ifelse(train_data$market_shop_km<2,1,0)+ 
    ifelse(train_data$market_shop_km<2,1,0)+ 
    ifelse(train_data$university_km<2,1,0)+ 
    ifelse(train_data$workplaces_km<2,1,0)+ 
    ifelse(train_data$public_healthcare_km<2,1,0)+ 
    ifelse(train_data$big_church_km<2,1,0)
  
  ### walk score to additional recreational venues
  train_data$prox_score_recreation <- ifelse(train_data$swim_pool_km<2,1,0)+
    ifelse(train_data$ice_rink_km<2,1,0)+
    ifelse(train_data$basketball_km<2,1,0)+
    ifelse(train_data$theater_km<2,1,0)+
    ifelse(train_data$museum_km<2,1,0)+
    ifelse(train_data$exhibition_km<2,1,0)+
    ifelse(train_data$fitness_km<2,1,0)
  
  ### distance to environmental hazards
  train_data$env_hazard_score <- ifelse(train_data$incineration_km<5,1,0)+
    ifelse(train_data$oil_chemistry_km<5,1,0)+
    ifelse(train_data$nuclear_reactor_km<5,1,0)+
    ifelse(train_data$radiation_km<5,1,0)
  
  ### houses that were pre-purchased before building completed
  train_data$pre_purchase <- ifelse(year(as.Date(train_data$timestamp))<train_data$build_year,1,0)
  
  ### recode material - only categories with 1000+ - wood (1) and concrete+brick (803) ref
  train_data$material_panel <- ifelse(train_data$material==1,1,0)
  train_data$material_brick <-  ifelse(train_data$material==2,1,0)
  train_data$material_concrete <- ifelse(train_data$material==4,1,0)
  train_data$material_breezeblock <- ifelse(train_data$material==5,1,0)
  
  ### is house made out of one of the more popular materials in raion !!! only brick/panel - hard to translate between material vars
  popular_material <- ifelse((train_data$material_panel==1 & train_data$build_count_panel>0.4) |
                               (train_data$material_brick==1 & train_data$build_count_brick>0.4),1,0)
  
  ### population density
  train_data$population_dens <- train_data$raion_popul/train_data$area_m
  
  ### recode ecology
  train_data$good_ecology <- ifelse(train_data$ecology=="good" | train_data$ecology=="excellent"| train_data$ecology=="satisfactory",1,0)
  train_data$poor_ecology <- ifelse(train_data$ecology=="poor",1,0)
  
  ### age of building 
  train_data$building_age <-  year(train_data$timestamp) - train_data$build_year
  train_data$building_age[which(train_data$building_age<0)] <- 0

  ### percent male
  train_data$prop_male <- train_data$female_new/train_data$all_new
  
  return(train_data)
}


##############################################################################
#                                                                            #
#                         lists of variables to drop *                       #
#                                                                            #
##############################################################################

drop_vars <- function(train_data) {
  
  # lists of vars to drop
  drop_list <- c('max_floor','material',"raion_build_count_with_builddate_info","raion_build_count_with_material_info",
                 "big_road1_1line","big_road1_2line","apt_id","ecology","ID_metro","ID_railroad_station_walk",
                 "ID_railroad_station_avto","ID_big_road1","ID_big_road2","ID_railroad_terminal","ID_bus_terminal",
                 'sub_area','full_all','female_f','male_f') 
  
  # create dataframes
  train_data <- data.frame(train_data)
  train_data <- train_data[,which(!(names(train_data) %in% drop_list))]
  
  # remove cafe variables
  cafe_ind <- which(grepl("cafe",colnames(train_data))&!grepl("PC",colnames(train_data)))
  train_data <- train_data[,which(!colnames(train_data) %in% colnames(train_data)[cafe_ind])]
  
  # remove distance variables
  distance_names <- colnames(train_data)[which(grepl("00",colnames(train_data)) & !grepl("cafe",colnames(train_data)) )]
  train_data <- train_data[,which(!colnames(train_data) %in% distance_names)]
  
  return(train_data)
}


##############################################################################
#                                                                            #
#                              clean up outcome *                            #
#                                                                            #
##############################################################################

clean_outcome <- function(train_data) {

  ### outcome variable 
  train_data$price_doc[which(log(train_data$price_doc)<14.55 & train_data$full_sq > quantile(train_data$full_sq,0.05,na.rm = T))] <- NA
  
  ### drop variables with missing outcome
  train_data_cout <- train_data[which(!is.na(train_data$price_doc)),]
  
  return(train_data_cout)
}










##############################################################################
#                                                                            #
#                            __________________                              # 
#                 __________________________________________                 #
#                              Process Data                                  #
#                 __________________________________________                 #
#                            __________________                              # 
#                                                                            #
##############################################################################

### combined test and train for cleaning 
test_data_id <- test_data; test_data_id$group_id <- 1
train_data_id <- train_data; train_data_id$group_id <- 2; 
train_data_id <- train_data_id[,-which(colnames(train_data_id)=="price_doc")]
complete_data <- data.frame(rbind(train_data_id,test_data_id))

### Clean up data
complete_data_clean <- anomalous_values(complete_data)
complete_data_rescale <- rescale_variables(complete_data_clean)
complete_data_apt <- individual_apartment(complete_data_rescale)
complete_data_cafe <- cafe_variables(complete_data_apt)
complete_data_reduce <- reduce_variables(complete_data_cafe)
complete_data_school <- school_variables(complete_data_reduce)
complete_data_build <- build_variables(complete_data_school)
complete_data_impute <- impute(complete_data_build)
complete_data_new_vars <- create_variables(complete_data_impute)
complete_data_drop_vars <- drop_vars(complete_data_new_vars)

### Seperate data frames - imputed 

# seperate test and train
complete_data_drop_vars <- complete_data_drop_vars[complete.cases(complete_data_drop_vars),]
train_imputed <- complete_data_drop_vars[which(complete_data_drop_vars$group_id==2),]
test_imputed <- complete_data_drop_vars[which(complete_data_drop_vars$group_id==1),]
 
# get price data and merge
price_data <- train_data[,c('id','price_doc')]
train_imputed <- merge(train_imputed,price_data,by="id",all=F)
train_imputed_full <- train_imputed # save
train_imputed <- clean_outcome(train_imputed)

### Seperate data frames - not imputed 

# seperate test and train
test_comp <- complete_data_reduce[which(complete_data_reduce$group_id==1),]
train_comp <- complete_data_reduce[which(complete_data_reduce$group_id==2),]

# merge price data
train_complete_full <- merge(train_comp,price_data,by="id",all=F)
train_complete <- clean_outcome(train_complete_full)

### make sure that new variables are created for both
test_complete <- drop_vars(create_variables(test_comp))
train_complete <- drop_vars(create_variables(train_complete))


##############################################################################
#                                                                            #
#                               save_data_frames                             #
#                                                                            #
##############################################################################

### Test

# write
write.csv(test_imputed,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/imputed_datasets/test_imputed_c.csv")
write.csv(test_complete,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/imputed_datasets/test_complete_c.csv")

### Train 
train_imputed <- train_imputed
train_complete <- train_complete

# write
write.csv(train_imputed,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/imputed_datasets/train_imputed_c.csv")
write.csv(train_complete,"/Users/kristenkeller/Documents/UCLA CLASSES/Masters Report/Kaggle/imputed_datasets/train_complete_c.csv")




























