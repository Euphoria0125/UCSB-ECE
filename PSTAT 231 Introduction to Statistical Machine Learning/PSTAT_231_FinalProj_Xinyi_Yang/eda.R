library(tidyverse)
library(lubridate)
if (!requireNamespace("corrplot", quietly = TRUE)) install.packages("corrplot")
library(corrplot)
if (!requireNamespace("patchwork", quietly = TRUE)) install.packages("patchwork")
library(patchwork)

set.seed(131)

# 1. Load Data
bmw <- read_csv("bmw_advanced_features.csv") %>%
  mutate(
    Target    = factor(Target, levels = c("1", "0"),
                       labels = c("Up", "Down")),
    Date      = as.Date(Date),
    DayOfWeek = factor(DayOfWeek,
                       labels = c("Mon","Tue","Wed","Thu","Fri"))
  )

cat("Rows:", nrow(bmw), "\n")
cat("Cols:", ncol(bmw), "\n")
glimpse(bmw)
summary(bmw)

# 2. Outcome Distribution
bmw %>%
  count(Target) %>%
  mutate(pct = n / sum(n) * 100)

# 3. Closing Price Over Time
ggplot(bmw, aes(x = Date, y = `Adj Close`)) +
  geom_line(color = "#2166ac", alpha = 0.7, linewidth = 0.4) +
  geom_smooth(method = "loess", span = 0.15,
              color = "#d73027", se = FALSE) +
  labs(title = "BMW Adjusted Close Price 1997–2026",
       x = "Date", y = "Adj Close (EUR)") +
  theme_minimal()

# 4. Daily Return Distribution by Target
bmw %>%
  ggplot(aes(x = Return, fill = Target)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("Up" = "#2166ac", "Down" = "#d73027")) +
  labs(title = "Daily Return by Next-Day Direction") +
  theme_minimal()

bmw %>%
  group_by(Target) %>%
  summarise(
    mean_return = mean(Return, na.rm = TRUE),
    sd_return   = sd(Return,   na.rm = TRUE),
    n           = n()
  )

# 5. Volatility Over Time
ggplot(bmw, aes(x = Date, y = Volatility_30, color = Target)) +
  geom_line(alpha = 0.4, linewidth = 0.3) +
  scale_color_manual(values = c("Up" = "#2166ac", "Down" = "#d73027")) +
  labs(title = "30-Day Rolling Volatility Over Time") +
  theme_minimal()

# Cluster analysis: average volatility by period
bmw %>%
  mutate(period = cut(Year,
                      breaks = c(1996, 2002, 2009, 2016, 2020, 2026),
                      labels = c("1997-2002","2003-2009",
                                 "2010-2016","2017-2020","2021-2026"))) %>%
  group_by(period) %>%
  summarise(mean_vol = mean(Volatility_30, na.rm = TRUE))

# 6. Day-of-Week Effect
bmw %>%
  group_by(DayOfWeek) %>%
  summarise(
    pct_up = mean(Target == "Up"),
    n      = n()
  )

# 7. Momentum vs Target
bmw %>%
  ggplot(aes(x = Target, y = Momentum_7, fill = Target)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("Up" = "#2166ac", "Down" = "#d73027")) +
  labs(title = "7-Day Momentum by Next-Day Direction") +
  theme_minimal()

# 8. Correlation Matrix
bmw %>%
  select(Return, Momentum_7, Momentum_30,
         Volatility_7, Volatility_30,
         High_Low_Spread, Open_Close_Spread,
         Lag1, Lag2, Lag3) %>%
  cor(use = "complete.obs") %>%
  corrplot(method = "circle", type = "lower", diag = FALSE,
           tl.cex = 0.8, addCoef.col = "black", number.cex = 0.65)

# 9. Feature-Target Correlation (point-biserial)
target_num <- as.numeric(bmw$Target == "Up")
features   <- c("Return","Momentum_7","Momentum_30",
                "Volatility_7","Volatility_30",
                "High_Low_Spread","Open_Close_Spread","Lag1")

feature_cors <- tibble(
  feature = features,
  cor_with_target = map_dbl(features, function(f) {
    cor(bmw[[f]], target_num, use = "complete.obs")
  })
) %>% arrange(desc(abs(cor_with_target)))

print(feature_cors)

# 10. Annual "Up" Rate
bmw %>%
  group_by(Year) %>%
  summarise(pct_up = mean(Target == "Up"), n = n()) %>%
  ggplot(aes(x = Year, y = pct_up)) +
  geom_col(fill = "#2166ac", alpha = 0.8) +
  geom_hline(yintercept = 0.5, lty = 2, color = "red") +
  labs(title = "Annual Proportion of Up Days",
       y = "Proportion Up", x = "Year") +
  theme_minimal()
