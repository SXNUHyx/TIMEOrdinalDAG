library(parallel)
load("../data/discdata.RData")

N <- 712

set.seed(1234)

boot_indices <- list()
for (i in 1:500) {
  boot_indices[[i]] <- sample(N, N, replace = TRUE)
}
save(boot_indices, file = "../data/Boot_List_Heart500.RData")


library(BiDAG)
source("../OSEMSource/R/OrdinalScore.R")
insertSource("../OSEMSource/R/spacefns.R", package = "BiDAG")
insertSource("../OSEMSource/R/usrscorefns.R", package = "BiDAG")
insertSource("../OSEMSource/R/initpar.R", package = "BiDAG")
insertSource("../OSEMSource/R/scoreagainstdag.R", package = "BiDAG")

# 4. 定义核心计算函数
sim_once_by_index <- function(idx) {
  n <- ncol(discdata)
  sub_data <- discdata[idx, ] 
  
  OSEMfit <- ordinalStructEM(n, sub_data,
                             usrpar = list(penType = "other",
                                           L = 5,
                                           lambda = 6))
  return(OSEMfit)
}


num_cores <- 4 
cat("开始 500 次重抽样分叉并行计算，核心数：", num_cores, "...\n")


system.time(
  results <- mclapply(boot_indices, sim_once_by_index, mc.cores = num_cores)
)


save(results, file = "../data/Results_Boot_heart_500.RData")
cat("全部计算完成，结果已安全落盘！\n")