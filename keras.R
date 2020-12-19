library(fields)
library(keras)
library(xgboost)
library(ggplot2)
library(readr)

library(stringr)

library(caret)

library(car)


pkgs <- c("keras", "lime", "tidyquant", "rsample", "recipes", "yardstick", "corrr")
install.packages(pkgs)
install.packages("labeling")

#################################################################
##############  load X_train ##############
#################################################################
set.seed(1)
data_raw <- read.csv("/Users/kelly/Desktop/9530/midterm_project/data/xgboostX.csv")
str(data_raw)

unique(data_raw$tfa_year)
unique(data_raw$tfa_month)
unique(data_raw$tfa_day)

data_raw$tfa_year <- data_raw$tfa_month <- data_raw$tfa_day <-data_raw$id <- NULL

str(data_raw)

obs=c(38,2011,5,25,rep(1,137))
obs

data_new=rbind(data_raw,obs)
str(data_new)



summary(data_raw$signup_methodweibo )
unique(data_raw$language.unknown.)
unique(data_raw$first_browserNintendo.Browser)
unique(data_raw$first_browserUC.Browser  )
summary(data_raw$first_browserIBrowse)

##data_raw$signup_methodweibo<- data_raw$language.unknown.<-data_raw$first_browserNintendo.Browser<-data_raw$first_browserUC.Browser<-data_raw$first_browserIBrowse<- NULL


X_train=scale(data_new)

colMeans(X_train)  # faster version of apply(scaled.dat, 2, mean)
apply(X_train, 2, sd)

str(X_train)



##########generate 10% test data


smp_size <- floor(0.9 * nrow(X_train))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(X_train)), size = smp_size)

X_train2 <- X_train[train_ind, ]
X_test2 <- X_train[-train_ind, ]

str(X_train2)
str(X_test2)











###load X_test

set.seed(1)
data_test <- read.csv("/Users/kelly/Desktop/9530/midterm_project/data/xgboostX_test.csv")
str(data_test)

id=data_test$id
id[1:30]


unique(data_test$tfa_year)
unique(data_test$tfa_month)
unique(data_test$tfa_day)

data_test$tfa_year <- data_test$tfa_month <- data_test$tfa_day <-data_test$id <- NULL

str(data_test)


data_testnew=rbind(data_test,obs)
str(data_testnew)


unique(data_test$signup_methodweibo )
unique(data_test$language.unknown.)
unique(data_test$first_browserNintendo.Browser)
unique(data_test$first_browserUC.Browser  )
unique(data_test$first_browserIBrowse)


###data_test$signup_methodweibo<- data_test$language.unknown.<-data_test$first_browserNintendo.Browser<-data_test$first_browserUC.Browser<-data_test$first_browserIBrowse<- NULL



X_test=scale(data_testnew)

colMeans(X_test)  # faster version of apply(scaled.dat, 2, mean)
apply(X_test, 2, sd)


str(X_test)











###load y_train


data_rawy <- read.csv("/Users/kelly/Desktop/9530/midterm_project/data/xgboosty.csv")
str(data_rawy)
unique(data_rawy)

data_rawy=rbind(data_rawy,4)



y_train <- to_categorical(data_rawy$y, num_classes = 12)
str(y_train)
y_train[1:30,]



####10% test data

y_train2 <- y_train[train_ind, ]
y_test2 <- data_rawy$y[-train_ind ]

str(y_train2)
str(y_test2)












#################################################################
######################## Model building  ########################
#################################################################
#library(keras)
model <- keras_model_sequential() 
########## build the model 
model %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(141)) %>%
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 12, activation = "softmax")

#this shows details on the number of parameters for each layer and in total
#summary(model)  

# loss defines the objective function for optimization
# For different optimizers, see the bottom of ?optimizer_adam
# metrics (I think) is more for monitoring.  For example, if you were doing
# a classification problem, you might have cross entropy loss and accuracy as a metric
#   -see MNIST example
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adagrad",
  metrics = c("accuracy")
)

# epochs = number of passes through the ENTIRE data set.
# batch_size = Model is trained(i.e. weights are updated) on small subsets of the data to
# prevent overfitting.  The lower it is, the noisier the training signal is going to be,
# the higher it is, the longer it will take to compute the gradient for each step.
start <- Sys.time()
history <- model %>% fit(
  X_train2, y_train2, 
  epochs = 20, batch_size = 256, 
  validation_split = 0.4
)
ItTook <- difftime(Sys.time(),start)
sprintf("DNN took %s %s to run",round(ItTook,digits=1),attr(ItTook,"units"))

#################################################################
######################## Reconstruction  ########################
#################################################################
## This will make a grid and let us see how well we can reconstruct the function F(x1,x2)




new <- model %>% predict(as.matrix(X_test2)) #warning is OK
str(new)
new[1:50,]

new1=apply(new, 1, sum)
new1[1:50]


predictions=as.data.frame(t(new))
str(predictions)


rownames(predictions) <- c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')


predictions[,1:20]

predictions_top5 <- as.vector(apply(predictions, 2, function(x) names(sort(x)[12:8])))


str(predictions_top5)


########ndcg



pred <- matrix(predictions_top5, nrow = 5)
pred[,1:30]

str(pred)
str(y_test2)


y_test22 <- recode(y_test2,"0='NDF'; 1='US'; 2='other'; 3='FR'; 4='CA'; 5='GB'; 6='ES'; 7='IT'; 8='PT'; 9='NL'; 10='DE'; 11='AU'")

str(y_test22)






f=function(pred,y){
  pos=which(pred == y)
  return(pos)
}

testn=ncol(pred)
poss=rep(NA,testn)

for (i in 1:testn){
  if (y_test22[i] %in% pred[,i]){poss[i]=f(pred[,i],y_test22[i])}
  else {poss[i]=6}
}
str(poss)

ndcg=mean(1/(log2(poss+1)))
ndcg

###[1] 0.8386203


summary(y_test22)

(prop.table(table(y_test22)))

#AU           CA           DE           ES           FR           GB           IT          NDF 
#0.0024829008 0.0059495924 0.0055748150 0.0113838658 0.0232362035 0.0105874637 0.0118991849 0.5827789750 
#NL        other           PT           US 
#0.0039351635 0.0489084606 0.0009369437 0.2923264312 
 


######output


ids <- NULL

#for (i in 1:NROW(X_test)) {
  
#  idx <- id[i]
  
 # ids <- append(ids, rep(,5))
  
#}

ids<- rep(id, each = 5)
ids[1:30]

submission <- NULL

submission$id <- ids

submission$country <- predictions_top5[1:310480]



str(X_test)


str(submission)
write.csv(submission, "/Users/kelly/Desktop/9530/midterm_project/data/submissionkeras.csv", quote=FALSE, row.names = FALSE)



######plot ndcg





newdata <- mydata[ which(mydata$gender=='F'& mydata$age > 65), ]






y_test2[1:250]

index=which(y_test2==11)
lst=list(index)
#lst

for (i in 1:11){
  index=which(y_test2==(11-i))
  lst=c(list(index), lst)
}

lst



lst[[12]]






f=function(pred,y){
  pos=which(pred == y)
  return(pos)
}

ndcg=rep(NA,12)





for (j in 1:12){
  
  pred1=pred[,lst[[j]]]
  y_test23=y_test22[lst[[j]]]
  
  testn=ncol(pred1)
  poss=rep(NA,testn)
  
  for (i in 1:testn){
    if (y_test23[i] %in% pred1[,i]){poss[i]=f(pred1[,i],y_test23[i])}
    else {poss[i]=12}
  }
  
  ndcg[j]=mean(1/(log2(poss+1)))

}



ndcg


#[1] 0.9507511 0.7899730 0.5000000 0.4303531 0.2702382 0.2707541 0.2702382 0.3863937 0.2702382 0.2702382
#[11] 0.2702382 0.2702382



names=rownames(predictions)
names


library(labeling)

df <- data.frame(country=names,
                 ndcg=ndcg)
head(df)

p<-ggplot(data=df, aes(x=country, y=ndcg)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_minimal()
p



######only top one pred

ndcg1=rep(NA,12)


pred11=pred[1,]


for (j in 1:12){
  
  pred1=pred11[lst[[j]]]
  y_test23=y_test22[lst[[j]]]
  
  testn=length(pred1)
  poss=rep(NA,testn)
  
  for (i in 1:testn){
    if (y_test23[i] %in% pred1[i]){poss[i]=f(pred1[i],y_test23[i])}
    else {poss[i]=12}
  }
  
  ndcg1[j]=mean(1/(log2(poss+1)))
  
}



ndcg1


#[1] 0.9026202 0.5847141 0.2702382 0.2702382 0.2702382 0.2702382 0.2702382 0.2702382 0.2702382 0.2702382 0.2702382 0.2702382



names=rownames(predictions)
names


library(labeling)

df <- data.frame(country=names,
                 ndcg=ndcg1)
head(df)

p<-ggplot(data=df, aes(x=country, y=ndcg)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_minimal()
p





######only top two pred

ndcg2=rep(NA,12)


pred12=pred[1:2,]


for (j in 1:12){
  
  pred1=pred12[lst[[j]]]
  y_test23=y_test22[lst[[j]]]
  
  testn=length(pred1)
  poss=rep(NA,testn)
  
  for (i in 1:testn){
    if (y_test23[i] %in% pred1[i]){poss[i]=f(pred1[i],y_test23[i])}
    else {poss[i]=12}
  }
  
  ndcg2[j]=mean(1/(log2(poss+1)))
  
}



ndcg2


#[1] 0.6361750 0.6362886 0.2702382 0.2702382 0.2702382 0.2702382 0.2702382 0.2702382
#[9] 0.2702382 0.2702382 0.2702382 0.2702382

names=rownames(predictions)
names


library(labeling)

df <- data.frame(country=names,
                 ndcg=ndcg2)
head(df)

p<-ggplot(data=df, aes(x=country, y=ndcg)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_minimal()
p







######only top two pred

ndcg3=rep(NA,12)


pred13=pred[1:3,]


for (j in 1:12){
  
  pred1=pred13[lst[[j]]]
  y_test23=y_test22[lst[[j]]]
  
  testn=length(pred1)
  poss=rep(NA,testn)
  
  for (i in 1:testn){
    if (y_test23[i] %in% pred1[i]){poss[i]=f(pred1[i],y_test23[i])}
    else {poss[i]=12}
  }
  
  ndcg3[j]=mean(1/(log2(poss+1)))
  
}



ndcg3


#[1] 0.5138050 0.5146616 0.5428503 0.2702382 0.2702382 0.2702382 0.2702382 0.2702382
#[9] 0.2702382 0.2702382 0.2702382 0.2702382

names=rownames(predictions)
names


library(labeling)

df <- data.frame(country=names,
                 ndcg=ndcg3)
head(df)

p<-ggplot(data=df, aes(x=country, y=ndcg)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_minimal()
p








