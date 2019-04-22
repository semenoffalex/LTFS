# Модель для сравнения train & test для поиска ликов
tr <- read_csv("train.csv") %>% 
  select(-UniqueID, -loan_default) %>% 
  mutate_if(is.character, as.factor)

val <- read_csv("test_bqCt9Pv.csv") %>% 
  select(-UniqueID) %>% 
  mutate_if(is.character, as.factor)

df <- bind_rows("1" = tr, "0" = val, .id = "target") %>% 
  mutate_if(is.character, as.factor) %>% 
  feat_engin()

smp_size <- floor(0.7 * NROW(df))
set.seed(699)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

train <- df[train_ind, ]
test <- df[-train_ind, ]

# Сохраняем отдельно вектора с лейблами для таргета
train_lab <- train$target
test_lab <- test$target

# Удаляем колонки с таргетом из датафреймов
train$target <- NULL
test$target <- NULL

# OHE
previous_na_action <- options('na.action')
options(na.action = 'na.pass')
train_smm <- Matrix::sparse.model.matrix(~ . -1, data = train)
test_smm <- Matrix::sparse.model.matrix(~ . -1, data = test)
options(na.action = previous_na_action$na.action)

# Готовим dense matrices для xgboost
train_dm <- xgb.DMatrix(train_smm, 
                        label = as.numeric(as.character(train_lab)), 
                        missing = NA)
test_dm <- xgb.DMatrix(test_smm, 
                       label = as.numeric(as.character(test_lab)), 
                       missing = NA)

watchlist = list(train = train_dm, test = test_dm)

xgb_cv <- xgb.cv(data = train_dm,
                 objective = "binary:logistic",
                 metrics = "auc",
                 nround = 2000,
                 eta = 0.2,
                 early_stopping_rounds = 15,
                 #nthread = 40,
                 nfold = 10,
                 print_every_n = 50,
                 max_depth = 3,
                 subsample = 1,
                 colsample_bytree = 0.7,
                 verbose = 1,
                 #scale_pos_weight = 99,
                 showsd = TRUE,
                 maximize = TRUE,
                 stratify = TRUE)

nround <- xgb_cv$best_ntreelimit

lal_fit <- xgb.train(data = train_dm,
                     objective = "binary:logistic",
                     metrics = "auc",
                     #watchlist = watchlist,
                     nround = nround,
                     eta = 0.2,
                     #early_stopping_rounds = 15,
                     #nthread = 8,
                     nfold = 10,
                     max_depth = 3,
                     subsample = 1,
                     colsample_bytree = 0.7,
                     verbose = 1,
                     #scale_pos_weight = 99,
                     showsd = TRUE,
                     maximize = TRUE,
                     stratify = TRUE)


test$target <- test_lab
lal_pred <- predict(lal_fit, test_dm)
lal_rocr <- ROCR::prediction(lal_pred, test$target)

lal_auc <- ROCR::performance(lal_rocr, 'auc')
lal_auc@y.values[[1]] # AUC

lal_fi <- xgb.importance(model = lal_fit)
xgb.plot.importance(importance_matrix = lal_fi[1:20], main = "Feature Importance")
View(lal_fi[1:20])

lal_roc <- ROCR::performance(lal_rocr, measure = 'tpr', x.measure = 'fpr')
plot(lal_roc, main = "ROC")
abline(a = 0, b = 1)

lal_precrec <- ROCR::performance(lal_rocr, 'prec', x.measure = 'rec')
plot(lal_precrec, main = 'Precision & Recall', colorize = TRUE)

lal_senspec <- performance(lal_rocr, measure = 'sens', x.measure = 'spec')
plot(lal_senspec, main = 'Sensitivity & Specificity', colorize = TRUE)

df %>% ggplot() +
  geom_boxplot(aes(x = target, y = age))
