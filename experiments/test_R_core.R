# Test pure functions from the real source without pretending BiDAG is installed.
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)) args[1] else ".."
expr <- parse(file.path(root,"ContinueEffectsRcode/OSEMSource/R/OrdinalScore.R"))
env <- new.env(parent=globalenv())
for(e in expr) {
  if(is.call(e) && identical(e[[1]], as.name("<-")) &&
     as.character(e[[2]]) %in% c("learnCuts","ordinalCoreScore","ordinalUpdateParam")) eval(e,env)
}
p <- list(n=2L,ordinalLevels=c(4L,2L),data=cbind(c(0L,3L,3L),c(0L,1L,0L)))
c <- env$learnCuts(p)
stopifnot(length(c[[1]])==5L,all(diff(c[[1]])>0),is.infinite(c[[1]][1]),
          c[[1]][1]<0,is.infinite(tail(c[[1]],1)),tail(c[[1]],1)>0)
p$data[1,1] <- 4L
stopifnot(inherits(try(env$learnCuts(p),silent=TRUE),"try-error"))
q <- list(N=20L,Sigma_hat=diag(4),lambda=6,penType="other",
          node_lags=c(0L,0L,1L,1L),allow_contemporaneous=TRUE)
stopifnot(is.infinite(env$ordinalCoreScore(3,1,4,q)),
          env$ordinalCoreScore(3,1,4,q)<0,
          is.finite(env$ordinalCoreScore(1,3,4,q)))
q$allow_contemporaneous <- FALSE
stopifnot(is.infinite(env$ordinalCoreScore(1,2,4,q)))
stopifnot(inherits(try(env$ordinalUpdateParam(q,diag(4),TRUE),silent=TRUE),"try-error"))
for(f in list.files(root,pattern="\\.[Rr]$",recursive=TRUE,full.names=TRUE)) parse(f)
cat("PASS: R thresholds including absent categories, range checks, temporal score guards, pseudo-replication rejection, R syntax\n")
