# Установка библиотек
install.packages('tidyverse')
install.packages('caret')
install.packages('xgboost')
install.packages('ROCR')
install.packages('Hmisc')

# Загрузка библиотек
library(tidyverse)
library(xgboost)
library(ROCR)
library(caret)

# Helper Functions --------------------------------------------------------
# Преобразование фичей
feat_engin <- function(df) {
  foo <- df %>% 
    mutate(birthdate = dmy(Date.of.Birth),
           event = dmy(DisbursalDate),
           age = interval(start = birthdate, end = event) / duration(num = 1, units = "years")
    )
  
  bar <- data.frame(str_split_fixed(foo$AVERAGE.ACCT.AGE, " ", 2))
  bar$X1 <- as.numeric(str_remove(bar$X1, "yrs"))
  bar$X2 <- as.numeric(str_remove(bar$X2, "mon"))
  
  baz <- data.frame(str_split_fixed(foo$CREDIT.HISTORY.LENGTH, " ", 2))
  baz$X1 <- as.numeric(str_remove(baz$X1, "yrs"))
  baz$X2 <- as.numeric(str_remove(baz$X2, "mon"))
  
  
  foo$avg_acct_agge <- bar$X1*12 + bar$X2
  foo$credit_history_len <- baz$X1*12 + baz$X2
  
  foo$CREDIT.HISTORY.LENGTH <- NULL
  foo$AVERAGE.ACCT.AGE <- NULL
  
  foo <- foo %>% 
    mutate(branch_id = as.factor(branch_id),
           manufacturer_id = as.factor(manufacturer_id),
           State_ID = as.factor(State_ID),
           pri_act_tot_ratio = PRI.ACTIVE.ACCTS/(PRI.NO.OF.ACCTS+1),
           pri_over_tot_ratio = PRI.OVERDUE.ACCTS/(PRI.NO.OF.ACCTS+1),
           sec_act_tot_ratio = SEC.ACTIVE.ACCTS/(SEC.NO.OF.ACCTS+1),
           sec_over_tot_ratio = SEC.OVERDUE.ACCTS/(SEC.NO.OF.ACCTS+1)
    ) %>% 
    select(-birthdate, -event, -Date.of.Birth, -DisbursalDate, 
           -MobileNo_Avl_Flag)
}


# Чтение датасета
df <- read_csv("train.csv") %>% 
  select(-UniqueID) %>% 
  mutate_if(is.character, as.factor)

val <- read_csv("test_bqCt9Pv.csv") %>% 
  select(-UniqueID) %>% 
  mutate_if(is.character, as.factor)

val_ids <- read_csv("test_bqCt9Pv.csv") %>% 
  select(UniqueID)

df <- feat_engin(df)
val <- feat_engin(val)

smp_size <- floor(0.7 * NROW(df))
set.seed(699)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

train <- df[train_ind, ]
test <- df[-train_ind, ]

# Сохраняем отдельно вектора с лейблами для таргета
train_lab <- train$loan_default
test_lab <- test$loan_default

# Удаляем колонки с таргетом из датафреймов
train$loan_default <- NULL
test$loan_default <- NULL

# OHE
previous_na_action <- options('na.action')
options(na.action = 'na.pass')
train_smm <- Matrix::sparse.model.matrix(~ . -1, data = train)
test_smm <- Matrix::sparse.model.matrix(~ . -1, data = test)
val_smm<- Matrix::sparse.model.matrix(~ . -1, data = val)
options(na.action = previous_na_action$na.action)

# Готовим dense matrices для xgboost
train_dm <- xgb.DMatrix(train_smm, 
                        label = train_lab, 
                        missing = NA)
test_dm <- xgb.DMatrix(test_smm, 
                       label = test_lab, 
                       missing = NA)
val_dm <- xgb.DMatrix(val_smm, 
                      missing = NA)

# Кросс-валидация в caret
xgb_trcontrol <- trainControl(
  method = "cv",
  number = 10,  
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

# Сетка гиперпараметров
xgbGrid <- expand.grid(nrounds = c(100,200),
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5), 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)

# Обучение модели с перебором гиперпараметров
set.seed(0) 
xgb_model <- train(
  train_dm, 
  train_lab,  
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  method = "xgbTree",
  verbose = TRUE
)

# Оптимальный набор гиперпараметров
xgb_model$bestTune
xgb_params <- list(xgb_model$bestTune) 

# Оценка качества модели
test$target <- test_lab # добавляем таргет обратно
pred <- predict(xgb_model, test_dm) # предсказываем новые наблюдения
rocr <- prediction(pred, test$target) # готовим объект для оценки качества модели
auc <- performance(rocr, 'auc') # 
auc@y.values[[1]] # значение AUC
auc_plot <- performance(rocr, measure = "tpr", x.measure = "fpr") # График
plot(auc_plot) # Отрисовка графика
        
# Скоринг валидационного множества
valpred <- predict(xgb_model, val_dm)
View(valpred)
scores <- ifelse(valpred >0.5, 1, 0)
sub <- bind_cols(list(val$UniqueID), list(scores))
View(sub)
colnames(sub) <- c("UniqueID", "loan_default")
write_csv(sub, "as_sub_v01.csv")



# MLBayesOpt --------------------------------------------------------------

# install.packages("MlBayesOpt")
library(MlBayesOpt)

res0 <- xgb_cv_opt(data = train_smm,
                   label = train_lab,
                   objectfun = 'binary:logistic',
                   evalmetric = 'auc',
                   n_folds = 10,
                   classes = 2,
                   init_points = 3,
                   n_iter = 30)

xgb_trcontrol <- trainControl(
  method = "cv",
  number = 10,  
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

# Подставляем параметры из MLBayesOpt
# Почему-то при подстановке параметров напрямую в train происходит ошибка
# "nrounds" matched by multiple actual arguments и ничего не считается
xgbGrid <- expand.grid(eta = 0.1,
                       max_depth = 5,
                       nrounds = 139,
                       subsample = 0.9306,
                       colsample_bytree = 1, 
                       gamma=0,
                       min_child_weight = 1
)

xgb_model <- train(
  train_dm, 
  as.factor(train_lab),  
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  method = "xgbTree",
  verbose = TRUE
)

valpred <- predict(xgb_model, val_dm)
sub <- bind_cols(list(val_ids$UniqueID), list(valpred))
colnames(sub) <- c("UniqueID", "loan_default")
write_csv(sub, "as_sub_v13.csv")
