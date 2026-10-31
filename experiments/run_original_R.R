# Original R OSEM frontend with guarded temporal parent scores.
# Run from the code directory: Rscript run_original_R.R input.json output.json
# Full R/BiDAG compatibility remains unverified in this environment.
# Exported parameters use the shared Python latent-interval integral for OCE.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2) stop('Need input.json output.json')
required <- c('jsonlite','BiDAG','pcalg','dplyr','leaps','glmnet','psych','tmvmixnorm','mvtnorm','ggm')
missing <- required[!vapply(required,requireNamespace,quietly=TRUE,FUN.VALUE=logical(1))]
if(length(missing)) stop(paste('Install required R packages:',paste(missing,collapse=', ')))
config <- jsonlite::fromJSON(normalizePath(args[1]));dest <- normalizePath(args[2],mustWork=FALSE)
base <- dirname(normalizePath(sub('^--file=','',commandArgs()[grep('^--file=',commandArgs())][1])))
setwd(file.path(base,'..','ContinueEffectsRcode'));library(BiDAG);source('OSEMSource/R/OrdinalScore.R')
for(f in c('spacefns.R','usrscorefns.R','initpar.R','scoreagainstdag.R')) insertSource(file.path('OSEMSource/R',f),package='BiDAG')
lag_id <- rep(0:config$L,each=config$p)
set.seed(config$seed)
C <- as.matrix(config$C);storage.mode(C)<-'integer'
fit <- ordinalStructEM(ncol(C),C,usrpar=list(penType='other',L=config$R,lambda=config$lambda,preLevels=as.integer(config$levels),node_lags=if(config$temporal) lag_id else NULL,allow_contemporaneous=config$contemporaneous),regsubsets=FALSE)
A <- as.matrix(fit$DAG);storage.mode(A)<-'numeric'
if(config$temporal && any(A[outer(lag_id,lag_id,'<')]!=0)) stop('Temporal score guard failed: reject output, do not prune arrows after fitting')
if(!config$contemporaneous && any(A[outer(lag_id,lag_id,'==')]!=0)) stop('Contemporaneous score guard failed')
S <- fit$param$Sigma_hat;n<-nrow(S);B<-matrix(0,n,n);v<-numeric(n)
for(j in seq_len(n)){
 pa <- which(A[,j]!=0)
 if(length(pa)) B[j,pa] <- solve(S[pa,pa,drop=FALSE],S[pa,j,drop=FALSE])
 v[j]<-max(1e-8,S[j,j]-sum(B[j,pa]*S[pa,j]))
}
W<-solve(diag(n)-B);Sigma<-W%*%diag(v)%*%t(W)
# Export the actual model parameters. OCE then uses the same latent distributional integral.
jsonlite::write_json(list(A=unname(A),B=unname(B),v=v,Sigma=unname(Sigma),cuts=lapply(fit$param$cuts,function(x) ifelse(is.infinite(x),ifelse(x<0,'-Inf','Inf'),format(x,digits=17))),trace=list(),converged=NA),dest,auto_unbox=FALSE,matrix='rowmajor',digits=16,pretty=TRUE)
saveRDS(fit,paste0(dest,'.rds'));writeLines(capture.output(sessionInfo()),paste0(dest,'.sessionInfo.txt'))
