library(readr)
library(dplyr)
# =============================
# 0. 初始化
# =============================
rm(list = ls())

library(rpart)
library(mgcv)
library(ggplot2)
library(readr)
# 读数据
data <- read_csv("data/heart_data.csv")
data <- data %>% select(-6)
data <- as.data.frame(scale(data))
# 目标变量（Y）
y_raw <- data$`HR(t)` 

# 保存结果
disc_data <- data

# =============================
# 1. 定义分箱函数
# =============================
discretize_var <- function(x_raw, y_raw, max_bins = 3){
  
  # 标准化 X 和 Y
  x_scaled <- scale(x_raw)
  y_scaled <- scale(y_raw)
  
  x <- x_scaled[,1]
  y <- y_scaled[,1]
  
  # 保存参数
  x_mean <- attr(x_scaled, "scaled:center")
  x_sd   <- attr(x_scaled, "scaled:scale")
  
  # 用 rpart 找切点
  fit <- rpart(y ~ x,
               method = "anova",
               control = rpart.control(maxdepth = max_bins - 1, cp = 0.001))
  
  # 如果没有切分（变量没用）
  if(is.null(fit$splits)){
    return(rep(0, length(x_raw)))
  }
  
  # 提取切点（标准化空间）
  cut_std <- sort(unique(fit$splits[, "index"]))
  
  # 还原到原始空间
  cut_raw <- cut_std * x_sd + x_mean
  
  # 分箱（用原始数据）
  x_disc <- cut(x_raw,
                breaks = c(-Inf, cut_raw, Inf),
                labels = FALSE)
  
  return(x_disc - 1)  # 转成0/1/2
}

# =============================
# 2. 自动处理所有列
# =============================

for(col in names(data)){
  
  # 跳过时间戳和目标变量
  if(col %in% c("Timestamp", "`HR(t)`")) next
  
  cat("正在处理:", col, "\n")
  
  x_raw <- data[[col]]
  
  # 跳过非数值列
  if(!is.numeric(x_raw)) next
  
  # 分箱
  disc_col <- discretize_var(x_raw, y_raw)
  
  # 保存
  disc_data[[paste0(col, "_disc")]] <- disc_col
}

#  保存数据 ---
write_csv(disc_data, "data/disc_data.csv")
discdata <- disc_data[, 16:ncol(disc_data)]

# ----------------------------
# 保存为 RData
# ----------------------------
save(discdata, file = "data/discdata.RData")

