library(randomForest)
library(tm)
library(RTextTools)
library(dummies)
library(mlr)
library(caret)
library(e1071)
library(DMwR)

# reading data and summary
dr <- read.csv("Reactive data after missing imputation1.csv", sep=",", header = TRUE)
dr$Injury_risk = NULL
dr$X = NULL

xr = as.factor(dr[,13])
dr = createDummyFeatures(dr[,-13])
dr$Injury.Type = xr

xm = as.factor(dr[,16])
dr = createDummyFeatures(dr[,-16])
dr$Injury.Type = xm
summary(dr)
colnames(dr)
colnames(dr)[100] <- "y"
colnames(dr)[138] <- "y"
#,  "I",  "J", "K", "L", "M", "N", "O", "P"
#, "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37", "V38", "V39", "V40", "V41"
table(dr$y)
dr$y = as.factor(dr$y)


#Different class imbalance techniques
#smote
data_smote = SMOTE(y~.,dr, perc.over = 200, perc.under = 200)
dr = SMOTE(y~.,data_smote, perc.over = 200, perc.under = 300)
table(dr$y)

#rose
library(ROSE)
dr = ovun.sample(y~.,data = dr, method = "both")$data
table(dr$y)

#MWMOTE
library(imbalance)
dr$y = as.numeric(dr$y)
add = mwmote(dr, 500, kNoisy = 5, kMajority = 3, kMinority=5, threshold = 5, cmax = 2, cclustering = 3, classAttr = "y")
dr = rbind(dr, add)
table(dr$y)
dr$y = as.factor(dr$y)


#BLSMOTE
library(smotefamily)
dr$class = NULL
dr$y = as.numeric(dr$y)
BLS = BLSMOTE(dr, dr$y,K=10,C=5,dupSize=0,method = "type2")
dr=BLS[[1]]
write.csv(dr, file = "missing values.csv")
library(Amelia)
colnames(dr)
table(dr$y)

#Machine learning classification algorithms
#crosss-validation
set.seed(75)

#SVM
model_svm = train(y~.,dr, method = "svmLinear", trControl = trainControl("cv",10))

#KNN
model_knn = train(y~.,dr, method = "knn", trControl = trainControl(method = "cv", number = 10), tuneGrid=expand.grid(.k = 10))

#CART
rpart.grid <- expand.grid(.cp=0.07)
model_cart <- train(y~.,dr, method="rpart",trControl=trainControl("cv",10),tuneGrid=rpart.grid)

#RF
library(randomForest)
model_rf = train(y~.,dr, method = "rf", trControl = trainControl(method = "cv", number = 10), tuneGrid=expand.grid(.mtry = 92))

ac_mat = matrix(0, nrow = 1, ncol = 4)
spec = matrix(0, nrow = 4, ncol = 3)
sens = matrix(0, nrow = 4, ncol = 3)
fm = matrix(0, nrow = 4, ncol = 3)
prec = matrix(0, nrow = 4, ncol = 3)
gm = matrix(0, nrow = 4, ncol = 3)
kappa = matrix(0, nrow = 4, ncol = 1)


# calculate performance measures for all algorithms 
for(l in 1:4){
  if(l==1)  
    model = model_svm  # just change the name of model for the particular algorithm
  if(l==2)  
    model = model_rf  # just change the name of model for the particular algorithm
  if(l==3)  
    model = model_knn  # just change the name of model for the particular algorithm
  if(l==4)  
    model = model_cart  # just change the name of model for the particular algorithm
  cm = model$resampledCM #confusion_matrix of that model
  cm[,10] = NULL
  cm[,11] = NULL
  
  Accuracy = matrix(0,nrow=10,ncol=1)
  sensitivity = matrix(0, nrow=11, ncol=3)
  sensitivity[1,1] = "High"
  sensitivity[1,2] = "Low"
  sensitivity[1,3] = "Medium"
  specificity = matrix(0, nrow=11, ncol=3)
  specificity[1,1] = "High"
  specificity[1,2] = "Low"
  specificity[1,3] = "Medium"
  precision = matrix(0, nrow=11, ncol=3)
  precision[1,1] = "High"
  precision[1,2] = "Low"
  precision[1,3] = "Medium"
  F_measure = matrix(0, nrow=11, ncol=3)
  F_measure[1,1] = "High"
  F_measure[1,2] = "Low"
  F_measure[1,3] = "Medium"
  G_mean = matrix(0, nrow=11, ncol=3)
  G_mean[1,1] = "High"
  G_mean[1,2] = "Low"
  G_mean[1,3] = "Medium"
  
  sum_prec = matrix(0,nrow =4, ncol = 3)
  sum_spec = matrix(0,nrow =4, ncol = 3)
  sum_sens = matrix(0,nrow =4, ncol = 3)
  sum_fm = matrix(0,nrow =4, ncol = 3)
  sum_gm = matrix(0,nrow =4, ncol = 3)
  weight = matrix(c(0,2,1,4,0,2,2,1,0),nrow = 3, ncol = 3)
  kap = matrix(0, nrow=10,ncol=3)
  
  for (i in 1:10)
  {
    # Accuracy calculation for each fold
    Accuracy[i,1] = (cm[i,1]+cm[i,4])/(cm[i,1]+cm[i,2]+cm[i,3]+(cm[i,4]))
    # sensitivity/Recall caalculation for each class in each fold 
    sensitivity[i+1,1] = cm[i,1]/(cm[i,1]+cm[i,4]+cm[i,7]) 
    sensitivity[i+1,2] = cm[i,5]/(cm[i,2]+cm[i,5]+cm[i,8])
    sensitivity[i+1,3] = cm[i,9]/(cm[i,3]+cm[i,6]+cm[i,9])
    
    # specificity caalculation for each class in each fold  
    specificity[i+1,1] = (cm[i,5]+cm[i,9])/(cm[i,5]+cm[i,9]+cm[i,2]+cm[i,3]) 
    specificity[i+1,2] = (cm[i,1]+cm[i,9])/(cm[i,1]+cm[i,9]+cm[i,4]+cm[i,6])
    specificity[i+1,3] = (cm[i,1]+cm[i,5])/(cm[i,1]+cm[i,5]+cm[i,7]+cm[i,8])     
    
    # Precision caalculation for each class in each fold 
    precision[i+1,1] = cm[i,1]/(cm[i,1]+cm[i,2]+cm[i,3]) 
    precision[i+1,2] = cm[i,5]/(cm[i,4]+cm[i,5]+cm[i,6])
    precision[i+1,3] = cm[i,9]/(cm[i,7]+cm[i,8]+cm[i,9])     
    
    # F-measure caalculation for each class in each fold 
    F_measure[i+1,1] = 2*as.numeric(precision[i+1,1])*as.numeric(sensitivity[i+1,1])/(as.numeric(precision[i+1,1])+as.numeric(sensitivity[i+1,1])) 
    F_measure[i+1,2] = 2*as.numeric(precision[i+1,2])*as.numeric(sensitivity[i+1,2])/(as.numeric(precision[i+1,2])+as.numeric(sensitivity[i+1,2]))
    F_measure[i+1,3] = 2*as.numeric(precision[i+1,3])*as.numeric(sensitivity[i+1,3])/(as.numeric(precision[i+1,3])+as.numeric(sensitivity[i+1,3])) 
    
    # G-mean caalculation for each class in each fold 
    G_mean[i+1,1] = sqrt(as.numeric(precision[i+1,1])*as.numeric(sensitivity[i+1,1]))
    G_mean[i+1,2] = sqrt(as.numeric(precision[i+1,2])*as.numeric(sensitivity[i+1,2]))
    G_mean[i+1,3] = sqrt(as.numeric(precision[i+1,3])*as.numeric(sensitivity[i+1,3])) 
    
    #kappa
      mat = matrix(cm[i,],nrow=3,ncol=3, byrow = TRUE)
      mat = apply(mat, 1, as.numeric)
      kap[i,1] = cohen.kappa(mat,w=weight, n.obs = 100)[[2]]
      kappa[l,1] = kappa[l,1]+kap[i,1]/10
    
  }
  for (i in 1:10)
  {
    sum_prec[l,1] = sum_prec[l,1] + as.numeric(precision[i+1,1])
    sum_prec[l,2] = sum_prec[l,2] + as.numeric(precision[i+1,2])
    sum_prec[l,3] = sum_prec[l,3] + as.numeric(precision[i+1,3])
    
    sum_sens[l,1] = sum_sens[l,1]+ as.numeric(sensitivity[i+1,1])
    sum_sens[l,2] = sum_sens[l,2]+ as.numeric(sensitivity[i+1,2])
    sum_sens[l,3] = sum_sens[l,3]+ as.numeric(sensitivity[i+1,3])
    
    sum_spec[l,1] = sum_spec[l,1]+ as.numeric(specificity[i+1,1])
    sum_spec[l,2] = sum_spec[l,2]+ as.numeric(specificity[i+1,2])
    sum_spec[l,3] = sum_spec[l,3]+ as.numeric(specificity[i+1,3])
    
    sum_fm[l,1] = sum_fm[l,1] + as.numeric(F_measure[i+1,1])
    sum_fm[l,2] = sum_fm[l,2] + as.numeric(F_measure[i+1,2])
    sum_fm[l,3] = sum_fm[l,3] + as.numeric(F_measure[i+1,3])
    
    sum_gm[l,1]  =sum_gm[l,1]+ as.numeric(G_mean[i+1,1])
    sum_gm[l,2]  =sum_gm[l,2]+ as.numeric(G_mean[i+1,2])
    sum_gm[l,3]  =sum_gm[l,3]+ as.numeric(G_mean[i+1,3])
  }

  ac_mat[1,l] = mean(Accuracy)
  prec[l,] = sum_prec[l,]/10
  sens[l,] = sum_sens[l,]/10
  spec[l,] = sum_spec[l,]/10
  fm[l,] = sum_fm[l,]/10
  gm[l,] = sum_gm[l,]/10
}

result = matrix(0, nrow = 4, ncol = 17)
for (i in 1:4)
{
  result[i,1] = ac_mat[i]
  result[i,2] = prec[i,1]
  result[i,3] = prec[i,2]
  result[i,4] = prec[i,3]
  result[i,5] = sens[i,1]
  result[i,6] = sens[i,2]
  result[i,7] = sens[i,3]
  result[i,8] = spec[i,1]
  result[i,9] = spec[i,2]
  result[i,10] = spec[i,3]
  result[i,11] = fm[i,1]
  result[i,12] = fm[i,2]
  result[i,13] = fm[i,3]
  result[i,14] = gm[i,1]
  result[i,15] = gm[i,2]
  result[i,16] = gm[i,3]
  result[i,17] = kappa[i,1]
}
print(result)
write.csv(result, file = "results.csv")

print(ac_mat)
print(prec)
print(sens)
print(spec)
print(fm)
print(gm)




write.csv(ac_mat, file = "ac_mat_smote.csv")
write.csv(prec, file = "prec_smote.csv")
write.csv(sens, file = "sens_smote.csv")
write.csv(spec, file = "spec_smote.csv")
write.csv(fm, file = "fm_smote.csv")
write.csv(gm, file = "gm_smote.csv")



