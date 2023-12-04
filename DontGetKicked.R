##
## Don't Get Kicked ##
##

library(tidymodels)
library(vroom)
library(embed) # target encoding
library(ranger) # random forests

kick_train <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/DontGetKicked/training.csv") %>%
  select(IsBadBuy,RefId,VehYear, VehicleAge, Transmission, VehOdo, IsOnlineSale) %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))
kick_test <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/DontGetKicked/test.csv") %>%
  select(RefId,VehYear, VehicleAge, Transmission, VehOdo, IsOnlineSale)
summary(kick_train)


##### Recipe #####
# Variables I think are important: VehYear, VehicleAge, Transmission, VehOdo, IsOnlineSale
my_recipe <- recipe(IsBadBuy~., data=kick_train) %>%
  step_mutate(Transmission = factor(Transmission)) %>%
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%# turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors())
  
  
prepped_recipe <- prep(my_recipe)
baked_recipe <- bake(prepped_recipe, kick_test)  


##### Random Forests #####

classForest_model <- rand_forest(mtry = tune(), # how many var are considered
                                 min_n=tune(), # how many observations per leaf
                                 trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Set Workflow

classForest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(classForest_model)

## Grid of values to tune over

tuning_grid <- grid_regular(mtry(range =c(1,1)),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV

folds <- vfold_cv(kick_train, v = 3, repeats=1)


CV_results <- classForest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #f_meas,sens, recall,spec, precision, accuracy


bestTune <- CV_results %>%
  select_best("roc_auc")


final_wf <- classForest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_train)


classForest_preds <- predict(final_wf, new_data=kick_test,type="prob") %>%
  bind_cols(., kick_test) %>% #Bind predictions with test data
  select(RefId, .pred_1) %>% #Just keep resource and predictions
  rename(IsBadBuy=.pred_1)

vroom_write(x=classForest_preds, file="./kick_classForest.csv", delim=",")
