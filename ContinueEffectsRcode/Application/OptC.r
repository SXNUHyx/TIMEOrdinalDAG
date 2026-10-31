# =============================
# Visualize CART optimal split points
# (including HR(t))
# =============================

rm(list = ls())

library(readr)
library(dplyr)
library(rpart)
library(mgcv)
library(ggplot2)

# 读取数据
data <- read_csv("data/heart_data.csv")

# 删除第6列
data <- data %>% select(-6)

# 标准化
data <- as.data.frame(scale(data))

plot_data <- data.frame()
cut_data  <- data.frame()


for(col in names(data)){

  x <- data[[col]]

  # 目标变量仍然使用 HR(t)
  y <- data$`HR(t)`

  temp_df <- data.frame(
    Variable = col,
    x = x,
    y = y
  )

  plot_data <- rbind(plot_data, temp_df)

  fit <- rpart(
    y ~ x,
    method = "anova",
    control = rpart.control(
      maxdepth = 2,
      cp = 0.001
    )
  )

  if(!is.null(fit$splits)){

    cuts <- sort(unique(fit$splits[, "index"]))

    cut_df <- data.frame(
      Variable = rep(col,length(cuts)),
      cut = cuts
    )

    cut_data <- rbind(cut_data,cut_df)
  }
}

# 绘图
p <- ggplot(plot_data,
       aes(x = x,
           y = y)) +

  geom_point(
    alpha = 0.08,
    size = 0.4,
    color = "#A6761D"
  ) +

  geom_smooth(
    method = "gam",
    formula = y ~ s(x),
    se = FALSE,
    linewidth = 1,
    color = "#1B9E77"
  ) +

  geom_vline(
    data = cut_data,
    aes(xintercept = cut),
    linetype = "dashed",
    linewidth = 0.6
  ) +

  facet_wrap(
    ~Variable,
    scales = "free_x",
    ncol = 5
  ) +

  theme_bw(base_size = 12) +

  labs(
    x = "Standardized Variable Value",
    y = "HR(t)",
    title = "Optimal CART-Based Supervised Discretization Boundaries"
  )
print(p)

# 保存图片
ggsave(
  "CART_Discretization_All_Variables.png",
  p,
  width = 15,
  height = 10,
  dpi = 600
)