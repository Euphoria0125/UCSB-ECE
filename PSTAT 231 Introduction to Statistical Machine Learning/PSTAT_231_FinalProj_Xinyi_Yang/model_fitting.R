pkgs <- c("tidyverse", "tidymodels", "discrim", "kknn",
          "vip", "ranger", "glmnet", "MASS")
install.packages(setdiff(pkgs, rownames(installed.packages())))

library(tidyverse)
if (!requireNamespace("tidymodels", quietly = TRUE)) install.packages("tidymodels")
library(tidymodels)
if (!requireNamespace("discrim", quietly = TRUE)) install.packages("discrim")
library(discrim)
if (!requireNamespace("ranger", quietly = TRUE)) install.packages("ranger")
library(ranger)
if (!requireNamespace("vip", quietly = TRUE)) install.packages("vip")
library(vip)
tidymodels_prefer()
set.seed(131)

# 1. Load and prepare data

bmw <- read_csv("bmw_advanced_features.csv") %>%
  mutate(
    Target    = factor(Target, levels = c("1","0"),
                       labels = c("Up","Down")),
    Date      = as.Date(Date),
    DayOfWeek = factor(DayOfWeek,
                       labels = c("Mon","Tue","Wed","Thu","Fri"))
  )

# 2. Temporal train/test split (80/20)
n       <- nrow(bmw)
cutoff  <- floor(0.80 * n)
bmw_train <- bmw[1:cutoff, ]
bmw_test  <- bmw[(cutoff + 1):n, ]

cat("Train:", nrow(bmw_train), "rows |",
    format(min(bmw_train$Date)), "to", format(max(bmw_train$Date)), "\n")
cat("Test: ", nrow(bmw_test),  "rows |",
    format(min(bmw_test$Date)), "to", format(max(bmw_test$Date)), "\n")

# 3. Rolling-origin cross-validation
bmw_folds <- rolling_origin(
  bmw_train,
  initial    = 4000,
  assess     = 252,
  skip       = 300,
  cumulative = FALSE
)
cat("CV folds:", nrow(bmw_folds), "\n")

# 3.5 Drop Volume_Change (Inf/-Inf when previous volume = 0)
bmw_train <- bmw_train %>% select(-Volume_Change)
bmw_test  <- bmw_test  %>% select(-Volume_Change)

# 4. Recipe
bmw_recipe <- recipe(
  Target ~ Return + Log_Return + Momentum_7 + Momentum_30 +
    Volatility_7 + Volatility_30 + High_Low_Spread +
    Open_Close_Spread + Price_Range_Pct +
    Lag1 + Lag2 + Lag3 + Volume_MA7 +
    DayOfWeek + Month,
  data = bmw_train
) %>%
  step_rm(Log_Return) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# 5. Model specifications

# 5a. Logistic Regression (elastic net, tuned penalty)
log_spec <- logistic_reg(penalty = tune(), mixture = 0.5) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

log_wf   <- workflow() %>% add_recipe(bmw_recipe) %>% add_model(log_spec)
log_grid <- grid_regular(penalty(range = c(-4, 0)), levels = 15)

log_res <- tune_grid(log_wf, resamples = bmw_folds,
                     grid = log_grid, metrics = metric_set(roc_auc))
best_log   <- select_best(log_res, metric = "roc_auc")
log_final  <- finalize_workflow(log_wf, best_log)
log_fit    <- fit(log_final, data = bmw_train)
cat("Best log penalty:", best_log$penalty, "\n")

# 5b. LDA
lda_spec <- discrim_linear() %>%
  set_engine("MASS") %>% set_mode("classification")
lda_wf  <- workflow() %>% add_recipe(bmw_recipe) %>% add_model(lda_spec)
lda_fit <- fit(lda_wf, data = bmw_train)

# 5c. QDA
qda_spec <- discrim_quad() %>%
  set_engine("MASS") %>% set_mode("classification")
qda_wf  <- workflow() %>% add_recipe(bmw_recipe) %>% add_model(qda_spec)
qda_fit <- fit(qda_wf, data = bmw_train)

# 5d. Random Forest (tuned mtry and min_n)
rf_spec <- rand_forest(mtry = tune(), trees = 200, min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wf   <- workflow() %>% add_recipe(bmw_recipe) %>% add_model(rf_spec)
rf_grid <- grid_regular(mtry(range = c(3, 8)),
                        min_n(range = c(10, 30)), levels = 2)

rf_res <- tune_grid(rf_wf, resamples = bmw_folds,
                    grid = rf_grid, metrics = metric_set(roc_auc))
best_rf  <- select_best(rf_res, metric = "roc_auc")
rf_final <- finalize_workflow(rf_wf, best_rf)
rf_fit   <- fit(rf_final, data = bmw_train)
cat("Best RF mtry:", best_rf$mtry, "| min_n:", best_rf$min_n, "\n")

# 6. Test-set evaluation
get_auc <- function(fit_obj, label) {
  predict(fit_obj, bmw_test, type = "prob") %>%
    bind_cols(bmw_test %>% select(Target)) %>%
    roc_auc(truth = Target, .pred_Up) %>%
    mutate(Model = label)
}

results <- bind_rows(
  get_auc(log_fit, "Logistic Regression"),
  get_auc(lda_fit, "LDA"),
  get_auc(qda_fit, "QDA"),
  get_auc(rf_fit,  "Random Forest")
) %>%
  select(Model, .estimate) %>%
  rename(`Test ROC-AUC` = .estimate) %>%
  arrange(desc(`Test ROC-AUC`))

print(results)

# 7. Best model: Confusion matrix + Variable importance
rf_class <- predict(rf_fit, bmw_test, type = "class") %>%
  bind_cols(bmw_test %>% select(Target))

cm <- conf_mat(rf_class, truth = Target, estimate = .pred_class)
print(cm)
autoplot(cm, type = "heatmap")

# Variable importance
rf_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 12)

# ROC curve
predict(rf_fit, bmw_test, type = "prob") %>%
  bind_cols(bmw_test %>% select(Target)) %>%
  roc_curve(Target, .pred_Up) %>%
  autoplot() +
  labs(title = "ROC Curve — Random Forest (Test Set)")
