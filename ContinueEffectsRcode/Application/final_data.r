# 步骤 0: 安装和加载必要的库
# 如果您还没有安装它们，请取消下面这行的注释来运行
# install.packages(c("dplyr", "readr", "lubridate", "tidyr", "purrr"))

library(readr)      # 用于快速读取CSV
library(dplyr)      # 用于数据操作 (group_by, mutate, lag)
library(lubridate)  # 用于处理日期和时间
library(tidyr)      # 用于数据整理 (nest, unnest, drop_na)
library(purrr)      # 用于在嵌套数据上迭代 (map)

# --- 步骤 1: 加载您的完整数据集 ---
# 假设您的文件名为 "merged_minute_level_data.csv"
cat("正在加载数据...\n")
tryCatch({
  full_data <- read_csv("data/merged_minute_level_data.csv")
}, error = function(e) {
  stop("错误: 无法找到 'merged_minute_level_data.csv'。请确保文件在您的R工作目录中。\n", e)
})

# --- 步骤 2: 定义参数和预处理 ---
# 定义我们关心的变量和最大滞后阶数
#Avg_HeartRate	Calories	Intensity	METs	Steps
variables_to_lag <- c("Avg_HeartRate", "Calories", "Intensity", "METs" , "Steps")
short_names <- c("HR", "Cal", "Inte", "METs" , "Steps")
MAX_LAG <- 2  # 我们将创建 t-1 和 t-2

# 假设您的时间戳列名为 "Time" 并且 "Id" 列存在
# 将时间戳列转换为 R 的 datetime 对象
# format = "%m/%d/%Y %I:%M:%S %p" 是 "4/12/2016 12:00:00 AM" 这种格式
cat("正在预处理数据 (转换时间戳)...\n")
prepped_data <- full_data %>%
  select(Id, Time, all_of(variables_to_lag)) %>%
  mutate(
    Timestamp = mdy_hm(Time) # !! 如果您的时间格式不同，请在此处更改 !!
  ) %>%
  select(-Time) # 移除旧的时间戳列

# --- 步骤 3: 核心功能 - 为单个ID创建完整并滞后的数据 ---
# 这个函数是实现您需求的核心
# 它为单个用户的数据框执行三个操作：
# 1. 创建一个从该用户min(time)到max(time)的完整分钟网格
# 2. 将原始数据连接到这个网格上，这会正确地为丢失的分钟创建NA
# 3. 在这个对齐后的数据上执行滞后操作
create_complete_lagged_data <- function(df, max_lag) {
  
  # 1. 创建完整的时间网格
  time_grid <- tibble(
    Timestamp = seq(
      min(df$Timestamp),
      max(df$Timestamp),
      by = "1 min"
    )
  )
  
  # 2. 将数据连接到网格上，以填充缺失的分钟
  aligned_data <- df %>%
    right_join(time_grid, by = "Timestamp") %>%
    arrange(Timestamp) # 必须按时间排序
  
  # 3. 创建滞后列
  for (lag in 1:max_lag) {
    # setNames 动态创建新列名
    # across() 对所有我们关心的变量应用 lag() 函数
    lag_cols <- aligned_data %>%
      select(all_of(variables_to_lag)) %>%
      mutate(across(
        .cols = everything(),
        .fns = ~lag(., n = lag),
        .names = "{.col}_t_{lag}" # 创建如 "Steps_t_1" 这样的列名
      )) %>%
      select(ends_with(paste0("_t_", lag))) # 只保留新创建的滞后列
    
    # 将新列绑定回我们的数据框
    aligned_data <- bind_cols(aligned_data, lag_cols)
  }
  
  return(aligned_data)
}

# --- 步骤 4: 按ID应用滞后功能 ---
# 这是处理30个ID的关键：
# 1. group_by(Id): 将数据分成30个小的数据框
# 2. nest(): 将这30个数据框“折叠”到一个名为'data'的列表列中
# 3. mutate(map(...)): 对'data'列中的每个数据框应用我们的自定义函数
# 4. unnest(): 将结果重新“展开”回一个大的数据框
cat("正在按ID分组并应用滞后操作 (这可能需要一些时间)...\n")
final_lagged_data <- prepped_data %>%
  group_by(Id) %>%
  nest() %>%  # 此时, 'data' 列包含了每个ID的tibble
  mutate(
    lagged = map(data, ~create_complete_lagged_data(., max_lag = MAX_LAG))
  ) %>%
  select(Id, lagged) %>% # 只保留ID和新的滞后数据
  unnest(lagged) %>%   # 展开结果
  ungroup()            # 移除分组

# --- 步骤 5: 清理和重命名 ---
# 1. 移除任何包含NA的行。这会移除：
#    a) 每个ID开头的几行 (因为没有 t-1, t-2)
#    b) 我们在步骤3中识别出的所有时间间隙
# 2. 重命名 t=0 的列 (例如 "HeartRate" -> "HR(t)")
cat("正在清理数据 (移除NA并重命名)...\n")
final_dataset_for_icd <- final_lagged_data %>%
  drop_na() %>% # 关键一步：移除所有不完整的行
  rename_with(
    .cols = all_of(variables_to_lag),
    .fn = ~paste0(short_names[match(., variables_to_lag)], "(t)")
  ) %>%
  # 重新命名滞后列以匹配Python示例 (已修复向量化问题)
  rename_with(
    .cols = ends_with(c("_t_1", "_t_2")),
    .fn = function(cols) {
      # 'cols' 是一个包含所有匹配列名的向量
      
      # 步骤 1: 循环遍历您的“长-短”名称对，
      # 在 *所有* 列名上执行替换。gsub是向量化的。
      new_cols <- cols
      for (i in seq_along(variables_to_lag)) {
        new_cols <- gsub(
          variables_to_lag[i], 
          short_names[i], 
          new_cols, 
          fixed = TRUE
        )
      }
      
      # 步骤 2: 向量化地替换 "_t_1" 为 "(t-1)" 等
      # 我们使用正则表达式：
      # "_t_" 匹配文字 "_t_"
      # (\\d) 匹配一个数字 (例如 '1' 或 '2') 并将其捕获为第1组
      # "(t-\\1)" 是替换字符串，\\1 会被替换为捕获的数字
      new_cols <- gsub("_t_(\\d)", "(t-\\1)", new_cols)
      
      # 返回已转换的、等长的向量
      return(new_cols)
    }
  ) %>%
  select(-Id) # ICD算法通常不需要ID列

# --- 步骤 6: 检查结果 ---
cat("完成！\n\n")
cat("最终数据集的前几行：\n")
print(head(final_dataset_for_icd))

cat(paste0("\n最终数据集维度: ", 
           nrow(final_dataset_for_icd), " 行 x ", 
           ncol(final_dataset_for_icd), " 列\n"))

cat("\n最终的列名 (节点):\n")
print(colnames(final_dataset_for_icd))

#保存这个数据集，以便在ICD算法中使用
write_csv(final_dataset_for_icd, "data/final_lagged_dataset_for_icd.csv")
