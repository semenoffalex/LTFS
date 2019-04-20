feat_engin <- function(df) {
  foo <- df %>% 
    mutate(birthdate = dmy(Date.of.Birth),
           event = dmy(DisbursalDate),
           age = interval(start = birthdate, end = event) / duration(num = 1, 
                                                                     units = "years"),
           birth_day = day(birthdate),
           birth_mon = month(birthdate),
           birth_year = year(birthdate),
           event_day = day(event),
           event_mon = month(event),
           event_year = year(event)
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
    select(-Date.of.Birth, -DisbursalDate, 
           -MobileNo_Avl_Flag, -birthdate, -event, -event_mon)
}