library(dplyr)
library(lubridate)

# === 1. 读取所有分钟级数据 ===
heartrate <- read.csv("data/heartrate_minutes_merged.csv", stringsAsFactors = FALSE)
calories  <- read.csv("data/minuteCaloriesNarrow_merged.csv", stringsAsFactors = FALSE)
intensity <- read.csv("data/minuteIntensitiesNarrow_merged.csv", stringsAsFactors = FALSE)
mets      <- read.csv("data/minuteMETsNarrow_merged.csv", stringsAsFactors = FALSE)
steps     <- read.csv("data/minuteStepsNarrow_merged.csv", stringsAsFactors = FALSE)

# === 2. 确保时间格式一致（转为POSIXct）===
# 心率数据的时间列名是 Time_minute，其他一般是 ActivityMinute
heartrate$Time <- mdy_hms(heartrate$Time_minute)
calories$Time  <- mdy_hms(calories$ActivityMinute)
intensity$Time <- mdy_hms(intensity$ActivityMinute)
mets$Time      <- mdy_hms(mets$ActivityMinute)
steps$Time     <- mdy_hms(steps$ActivityMinute)

# === 3. 选取必要列并统一命名 ===
heartrate <- heartrate %>% select(Id, Time, Avg_HeartRate)
calories  <- calories  %>% select(Id, Time, Calories)
intensity <- intensity %>% select(Id, Time, Intensity)
mets      <- mets      %>% select(Id, Time, METs)
steps     <- steps     %>% select(Id, Time, Steps)

# === 4. 逐步内连接 (inner join) 仅保留相同Id & Time的数据 ===
merged_data <- heartrate %>%
  inner_join(calories,  by = c("Id", "Time")) %>%
  inner_join(intensity, by = c("Id", "Time")) %>%
  inner_join(mets,      by = c("Id", "Time")) %>%
  inner_join(steps,     by = c("Id", "Time"))

#5 格式化时间为带AM/PM的字符串 ===
Sys.setlocale("LC_TIME", "C")
merged_data$Time <- format(merged_data$Time, "%m/%d/%Y %I:%M %p")

# === 6. 查看结果并导出 ===
head(merged_data)
write.csv(merged_data, "data/merged_minute_level_data.csv", row.names = FALSE)
