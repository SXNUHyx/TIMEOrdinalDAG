# --- 步骤 0: 加载库并设置参数 ---
library(dplyr)
library(lubridate)
library(tidyr)
library(purrr)
library(readr)

set.seed(42)

variables_to_lag <- c("HeartRate", "Steps", "Calories", "Distance")
short_names <- c("HR", "Steps", "Cal", "Dist")
MAX_LAG <- 2

# --- 1. 生成带因果关系的模拟数据 ---
generate_causal_data <- function(id, n_minutes = 600) {
  
  # 时间序列
  Timestamp <- ymd_hms("2025-01-01 00:00:00") + minutes(0:(n_minutes - 1))
  
  # 初始化变量
  Steps <- numeric(n_minutes)
  Calories <- numeric(n_minutes)
  HeartRate <- numeric(n_minutes)
  Distance <- numeric(n_minutes)
  
  # 初始值
  HeartRate[1] <- 70
  
  for (t in 1:n_minutes) {
    
    # 步数：基础水平 + 随机活动（白噪声 + 周期波动）
    Steps[t] <- max(0, round(20 + 10 * sin(t / 100) + rnorm(1, 0, 5)))
    
    # 卡路里消耗（因果关系：随步数上升）
    Calories[t] <- 1 + 0.03 * Steps[t] + rnorm(1, 0, 0.2)
    
    # 距离（随步数线性增长）
    Distance[t] <- 0.0008 * Steps[t] + rnorm(1, 0, 0.005)
    
    # 心率：受步数和前一分钟心率影响
    if (t > 1) {
      HeartRate[t] <- 60 + 0.2 * Steps[t] + 0.5 * HeartRate[t - 1] + rnorm(1, 0, 2)
    }
  }
  
  tibble(
    Id = id,
    Timestamp,
    HeartRate,
    Steps,
    Calories,
    Distance
  )
}

# 生成4个ID的数据（每个1440分钟）
demo_data <- map_dfr(1:4, ~generate_causal_data(.x))

cat("--- 1. 模拟的原始数据 (含因果关系) ---\n")
print(head(demo_data, 10))

# --- 2. 嵌套数据 ---
nested_data <- demo_data %>%
  group_by(Id) %>%
  nest()

cat("\n--- 2. 嵌套后的数据 (4 行) ---\n")
print(nested_data)

# --- 3. 创建滞后数据的函数 ---
create_complete_lagged_data <- function(df, max_lag) {
  
  time_grid <- tibble(
    Timestamp = seq(
      min(df$Timestamp),
      max(df$Timestamp),
      by = "1 min"
    )
  )
  
  aligned_data <- df %>%
    right_join(time_grid, by = "Timestamp") %>%
    arrange(Timestamp)
  
  for (lag in 1:max_lag) {
    lag_cols <- aligned_data %>%
      select(all_of(variables_to_lag)) %>%
      mutate(across(
        .cols = everything(),
        .fns = ~dplyr::lag(., n = lag),
        .names = "{.col}_t_{lag}"
      )) %>%
      select(ends_with(paste0("_t_", lag)))
    
    aligned_data <- bind_cols(aligned_data, lag_cols)
  }
  
  aligned_data
}

# --- 4. 应用滞后函数 ---
mapped_data <- nested_data %>%
  mutate(lagged = map(data, ~create_complete_lagged_data(., max_lag = MAX_LAG)))

cat("\n--- 3. 映射后的数据 (查看ID=1) ---\n")
print(head(mapped_data$lagged[[1]], 10))

# --- 5. 展开、清理、重命名 ---
simulated_data <- mapped_data %>%
  select(Id, lagged) %>% 
  unnest(lagged) %>%   
  ungroup() %>%
  drop_na() %>%
  rename_with(
    .cols = all_of(variables_to_lag),
    .fn = ~paste0(short_names[match(., variables_to_lag)], "(t)")
  ) %>%
  rename_with(
    .cols = ends_with(c("_t_1", "_t_2")),
    .fn = function(cols) {
      new_cols <- cols
      for (i in seq_along(variables_to_lag)) {
        new_cols <- gsub(variables_to_lag[i], short_names[i], new_cols, fixed = TRUE)
      }
      new_cols <- gsub("_t_(\\d)", "(t-\\1)", new_cols)
      return(new_cols)
    }
  ) %>%
  select(-Id)

cat("\n--- 4. 最终用于ICD的矩阵 (Z_final) ---\n")
print(head(simulated_data, 10))

# --- 6. 保存 ---
dir.create("data", showWarnings = FALSE)
write_csv(demo_data, "data/demo_data.csv")
write_csv(simulated_data, "data/simulated_data.csv")
cat("\n--- 数据已保存到 data/demo_data.csv 和 data/simulated_data.csv ---\n")