
library(glmnet)
library(glmnetUtils)

#loading data
load(file="C:/Users/1gugl/Documents/homeworkSL/amazon_review_clean.RData")

#dataset ridotto
dim.tr<-10000
set.seed(179)
sel.tr <- sample(1:dim(X_tr)[1], dim.tr, replace = F)
X_rid_te=X_tr[-sel.tr,]
X_rid_tr=X_tr[sel.tr,]
y_rid_te=y_tr[-sel.tr]
y_rid_tr=y_tr[sel.tr]

?cva.glmnet #cross-validation for lambda and alpha
cv.elasticnet<-cva.glmnet(X_rid_tr,y_rid_tr,family="binomial",nfolds=3,trace.it=1)
cv.elasticnet$alpha

mean_cross_error<-cv.elasticnet[["modlist"]][[1]][["cvm"]][100]
o<-1
for (i in 2:10){
  if (mean_cross_error>min(cv.elasticnet[["modlist"]][[i]][["cvm"]])){
    mean_cross_error<-min(cv.elasticnet[["modlist"]][[i]][["cvm"]])
    o<-i
  }
}

alpha.opt<-cv.elasticnet$alpha[o]
lambda.opt<-cv.elasticnet[["modlist"]][[o]][["lambda.min"]]

saveRDS(object=cv.elasticnet,file="ENcrossval.rds")

cv.elasticnet<-readRDS(file="ENcrossval.rds")
elastic.fit<-cv.elasticnet[["modlist"]][[o]][["glmnet.fit"]] #elastic-net fitted after CV
#cv.elastic.fit<-cv.elasticnet[["modlist"]][[o]] #elastic-net con alpha-opt
plot(elastic.fit,xvar="lambda",label=T) #lasso-path

performance<-assess.glmnet(elastic.fit,newx=X_te,newy=y_te,family="binomial",s=lambda.opt)
performance$class #misclassification (proportion of misclassified points)

#training e testing sul dataset completo:
elastic.fit.Tot<-glmnet(X_tr, y_tr, family="binomial", lambda=lambda.opt, alpha=alpha.opt)

performanceTot<-assess.glmnet(elastic.fit.Tot ,newx=X_te, newy=y_te, family="binomial", s=lambda.opt)
performanceTot$class



#last thing to do: check what the most important words are
wordsTot<-glmnet(X_tr,y_tr,lambda=0.01,alpha=1,family="binomial")
subset(row.names(as.matrix(wordsTot$beta)),abs(as.matrix(wordsTot$beta)[,1])>0)



###################### 3
memory.limit(size = 6000000) #increase memory limit = 600gb. As we are using a 64-bit version of Rstudio in a 64-bit windows 
#environment the limit is 8tb
M<-6
n<-dim(X_rid_tr)[1]
p<-dim(X_rid_tr)[2]
X_drop<-matrix(0,nrow=M*n,ncol=p)
Y_drop<-numeric(M*n)
Csi<-matrix(0,M*n,p)

delta<-seq(from=0.1,to=0.95,by=0.1)
d<-length(delta)
dropdelta<-numeric(d)

for(j in 1:M){
  ni1<-(j-1)*n+1
  ni2<-j*n
  Y_drop[ni1:ni2]<-y_rid_tr
}

Y_drop<-as.factor(Y_drop)
y_rid_te<-as.factor(y_rid_te)

for(i in 1:d) {
  delta1<-delta[i]
  
  Csi=matrix(rbinom(M*n*p,size=1,prob=1-delta1)/(1-delta1),M*n,p)
  
  for (j in 1:M){
  ni1<-(j-1)*n+1
  ni2<-j*n
  X_drop[ni1:ni2,]<-X_rid_tr*Csi[ni1:ni2,]
  }
  
  logisticDrop<-bigGlm(x=X_drop,y=Y_drop,family="binomial",trace.it=1,maxit=1000,path=TRUE)
  y.hat<-predict(logisticDrop,newx=X_rid_te,family="binomial",s=0,type="class") 
  
  #https://bookdown.org/egarpor/PM-UC3M/glm-deviance.html
  dropdelta[i]<-1-mean(y.hat==y_rid_te)
  }

plot(delta,dropdelta,xlab=bquote(delta),ylab="MISSClass Error",ylim=c(0.02,0.06))
lines(delta,dropdelta)

abline(v=delta[which.min(dropdelta)])
dropdelta
deltastar<-delta[which.min(dropdelta)]
deltastar
missclass.drop<-dropdelta[which.min(dropdelta)]

#cross-validation sul lasso
cv.lasso<-cv.glmnet(X_rid_tr,y_rid_tr,alpha=1,family="binomial",nfolds=10,trace.it=1)
lasso.fit<-cv.lasso[["glmnet.fit"]]
missclass.lasso<-1-mean(predict(object=lasso.fit,newx=X_rid_te,family="binomial",s=cv.lasso$lambda1.se,alpha=1,type="class")==y_rid_te)
missclass.drop<missclass.lasso

#
missclass.elasticnet<-1-mean(predict(object=elastic.fit,
                                     newx=X_rid_te,family="binomial",s=lambda.opt,alpha=alpha.opt,type="class")==y_rid_te)$class
missclass.drop<missclass.elasticnet



###################### 4
#minimizzare -l(p)+R(beta)

funz.obb <- function(beta, X, Y,lambda=1,delta=0.7,n=1){
  p<-numeric(n)
  for (i in 1:n){
    p[i]<- (1 + exp(-as.numeric(X[i,]) %*% beta))^-1
  }
  p.coeff <- p*(1 - p)
  penb <- as.numeric(0.5 * delta / (1 - delta) * (p.coeff %*% (X^2 %*% beta^2)))
  loglik <- as.numeric(n*Y %*% log(p) + (1 - Y) %*% log(1 - p))
  loss <- (-(loglik) + (lambda)*penb)
  return(loss)
}

logit = function(X, beta) {
  return(exp(X %*% beta)/(1+ exp(X %*% beta)) )
}


loss.gradient = function(beta, X, Y, lambda=1,delta=.7,n=1) {
  p<-numeric(n)
  for (i in 1:n){
    p[i]<- (1 + exp(-as.numeric(X[i,]) %*% beta))^-1
  }
  p.coeff <- p*(1 - p)
  penalty.grad<-numeric(length(beta))
  for (j in 1:length(beta)){
    penalty.grad[j]<-(delta)/(1-delta)*beta[j]*(p.coeff%*%(X[,j]^2))
  }
  loglik.score<-t(X) %*% (logit(X, beta) - Y)
  return(as.numeric(loglik.score+(lambda)*penalty.grad))
}

#Prima cosa: prendere data set

#https://archive.ics.uci.edu/ml/datasets/banknote+authentication (forgery)
dba<-data_banknote_authentication
X_tr.2<-as.matrix(x=dba[,1:4])
Y_tr.2<-dba[,5]
n<-dim(X_tr.2)[1]
d<-dim(X_tr.2)[2]
?optim
beta.0<-rep(3,d)
beta.opt.0<-optim(par=beta.0,fn=funz.obb,X=X_tr.2,Y=Y_tr.2,n=length(Y_tr.2),lambda=1,
                  method="BFGS")$par
pi<-numeric(n)
for (i in 1:n){
  pi[i]<- (1 + exp(-X_tr.2[i,] %*% beta.opt.0))^-1
}

y.hat<-numeric(n)
y.hat[pi>0.5]=1
1-mean(y.hat==Y_tr.2)


########
#Dropout Approx Cross-val
dim.tr<-5000
p<-dim(X_tr)[2]
set.seed(123)
sel.tr <- sample(1:dim(X_tr)[1], dim.tr, replace = F)
X_rid_te=X_tr[-sel.tr,]
X_rid_tr=X_tr[sel.tr,]
y_rid_te=y_tr[-sel.tr]
y_rid_tr=y_tr[sel.tr]
y_bin_tr<-numeric(dim.tr)
y_bin_tr[y_rid_tr=="movies"]<-1

#test dimension
dim.te<-length(y_rid_te)
y_bin_te<-numeric(dim.te)
y_bin_te[y_rid_te=="movies"]<-1

#cross-validating parameters
delta<-seq(0.1,0.9,by=0.1)
lambda<-seq(0.1,5.1,by=1)
grid.dl<-expand.grid(delta,lambda)
mce<-numeric(dim(grid.dl)[1])
beta.opt.0<-rep(0,p)
beta.0<-beta.opt.0
pi2<-numeric(dim.te)

for (i in 1:dim(grid.dl)[1]){
beta.opt.0<-optim(par=beta.0,fn=funz.obb,gr=loss.gradient,X=X_rid_tr,Y=y_bin_tr,n=dim.tr,lambda=grid.dl[i,2],
                   delta=grid.dl[i,1],method="BFGS",control=list(trace=1))$par

#missclass on test set
for (j in 1:dim.te){
  pi2[j]<- (1 + exp(-X_rid_te[j,] %*% beta.opt.0))^-1
}
y.hat<-numeric(length(y_rid_te))
y.hat[pi2>0.5]=1
mce[i]<-1-mean(y.hat==y_bin_te)
print(mce[i])
rm(y.hat)
}
rm(X_rid_te)
rm(y_rid_te)
rm(X_rid_tr)
rm(y_rid_tr)
rm(y_bin_tr)
rm(y_bin_te)

#saveRDS(mce,file="missclasscv.rds")
i.star<-which.min(mce)
opt.dl<-grid.dl[i.star,]
delta.opt<-grid.dl[i.star,1]
lambda.opt<-grid.dl[i.star,2]

#optimization over the whole dataset
y_tr.bin<-numeric(length(y_tr))
y_tr.bin[y_tr=="movies"]=1

beta.opt<-optim(par=beta.0,fn=funz.obb,gr=loss.gradient,X=X_tr,Y=y_tr.bin,n=dim(X_tr)[1]
                ,lambda=lambda.opt,delta=delta.opt,method="BFGS",control=list(trace=1))$par
#saveRDS(beta.opt,file="betaopt.rds")
#missclass using the whole dataset
pi2<-numeric(length(y_te))
for (j in 1:length(y_te)){
  pi2[j]<- (1 + exp(-X_te[j,] %*% beta.opt))^-1
}
y.hat<-numeric(length(y_te))
y.hat[pi2>0.5]=1
y_bin_te<-numeric(length(y_te))
y_bin_te[y_te=="movies"]=1
mean(y.hat==y_bin_te)

library(plot3D)
scatter3D(grid.dl$Var1, grid.dl$Var2, mce,
       xlab="Delta",ylab="Lambda",zlab="Cross-validated Missclass. Error",
       phi = 0, bty = "g",
       pch = 20, cex = 2, ticktype = "detailed")
