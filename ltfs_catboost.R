devtools::install_url('https://github.com/catboost/catboost/releases/download/v0.14.1/catboost-R-Darwin-0.14.1.tgz', args = c("--no-multiarch"))
library(catboost)
library(tidyverse)
library(caret)
library(lubridate)

# Функция для предобработки данных
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
    select(-birthdate, -event, -Date.of.Birth, -DisbursalDate)
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

labels <- unlist(df$loan_default)
features <- df %>% select(-loan_default)

new_features <- feat_engin(features)
new_val <- feat_engin(val)

# cat_feats <- c(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,36,37)
cat_feats <- c(3,5,7,8,17)

train_pool <- catboost.load_pool(data = new_features, 
                                 label = labels,
                                 cat_features = cat_feats)
real_pool <- catboost.load_pool(new_val, 
                                cat_features = cat_feats)


model <- catboost.train(train_pool,  NULL,
                        params = list(loss_function = 'Logloss',
                                      eval_metric = 'AUC', 
                                      learning_rate = 0.1,
                                      iterations = 1000, 
                                      metric_period = 10,
                                      border_count = 64,
                                      depth = 8,
                                      learning_rate = 0.1,
                                      l2_leaf_reg = 0.1,
                                      rsm = 0.95,
                                      logging_level = 'Verbose'
                                      ))


prediction <- catboost.predict(model, 
                               real_pool,
                               prediction_type = "Class")
sub <- bind_cols(list(val_ids), list(prediction))
colnames(sub) <- c("UniqueID", "loan_default")
write_csv(sub, "as_sub_v10.csv")


# Tuning in caret ---------------------------------------------------------
fit_control <- trainControl(method = "cv",
                            number = 10,
                            classProbs = FALSE)

grid <- expand.grid(depth = c(4, 6, 8),
                    learning_rate = 0.1,
                    iterations = 100,
                    l2_leaf_reg = 0.1,
                    rsm = 0.95,
                    border_count = 64)

model <- train(new_features, 
               as.factor(make.names(labels)),
               method = catboost.caret,
               logging_level = 'Verbose', 
               preProc = NULL,
               tuneGrid = grid, 
               trControl = fit_control,
               metric = 'Kappa')

print(model)

importance <- varImp(model, scale = FALSE)
print(importance)
prd <- predict(model, newdata = new_val, type = 'raw')
prediction <- ifelse(as.numeric(prd)==1,0,1)
sub <- bind_cols(list(val_ids), list(prediction))
colnames(sub) <- c("UniqueID", "loan_default")
write_csv(sub, "as_sub_v09.csv")
