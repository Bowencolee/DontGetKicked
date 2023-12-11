##
## Don't Get Kicked ##
##

library(tidymodels)
library(vroom)
library(embed) # target encoding
library(ranger) # random forests

train <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/DontGetKicked/training.csv",
                      na = c("", "NA", "NULL", "NOT AVAIL"))
test <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/DontGetKicked/test.csv",
                     na = c("", "NA", "NULL", "NOT AVAIL"))

##### Recipe #####
# Variables I think are important: VehYear, VehicleAge, Transmission, VehOdo, IsOnlineSale
my_recipe <- recipe(IsBadBuy ~., data = train) %>%
  update_role(RefId, new_role = 'ID') %>%
  update_role_requirements("ID", bake = FALSE) %>%
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) %>%
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  #step_mutate_at(contains("MMR"), fn = numeric) %>%
  step_rm(contains('MMR')) %>%
  step_rm(BYRNO, WheelTypeID, VehYear, VNST, VNZIP1, PurchDate, # these variables don't seem very informative, or are repetitive
          AUCGUART, PRIMEUNIT, # these variables have a lot of missing values
          Model, SubModel, Trim) %>% # these variables have a lot of levels - could also try step_other()
  step_corr(all_numeric_predictors(), threshold = .7) %>%
  step_other(all_nominal_predictors(), threshold = .0001) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_median(all_numeric_predictors())
  
  
  
prepped_recipe <- prep(my_recipe)
baked_recipe <- bake(prepped_recipe, test)  


##### Random Forests #####

classForest_model <- rand_forest(mtry = tune(), # how many var are considered
                                 min_n=tune(), # how many observations per leaf
                                 trees=250) %>% #Type of model
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

folds <- vfold_cv(train, v = 3, repeats=1)


CV_results <- classForest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #f_meas,sens, recall,spec, precision, accuracy


bestTune <- CV_results %>%
  select_best("gain_capture")


final_wf <- classForest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)


classForest_preds <- predict(final_wf, new_data=test,type="prob") %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(RefId, .pred_1) %>% #Just keep resource and predictions
  rename(IsBadBuy=.pred_1)

vroom_write(x=classForest_preds, file="./kick_classForest.csv", delim=",")

##### Boosting #####
boosted_model <- boost_tree(tree_depth=4, #Determined by random store-item combos
                            trees=1500,
                            learn_rate=0.01) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boosted_model) %>%
  fit(data=train)

boost_preds <- predict(boost_wf, new_data=test,type="prob") %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(RefId, .pred_1) %>% #Just keep resource and predictions
  rename(IsBadBuy=.pred_1)

##### Penalized Regression #####
penLog_mod <- logistic_reg(mixture = tune(),
                           penalty = tune()) %>% #Type of model
  set_engine("glmnet")

penLog_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(penLog_mod) %>%
  fit(data = train)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 3)

folds <- vfold_cv(train, v = 3, repeats = 1)

CV_results <- penLog_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #f_meas,sens, recall,spec, precision, accuracy

bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <- penLog_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

penLog_preds <- predict(final_wf, new_data=test,type="prob") %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(RefId, .pred_1) %>% #Just keep resource and predictions
  rename(IsBadBuy=.pred_1)

##### Stacking #####
folds <- vfold_cv(train, v = 5, repeats=1)

untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

# Boosting
boosted_model <- boost_tree(tree_depth=4, #Determined by random store-item combos
                            trees=1500,
                            learn_rate=0.01) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boosted_model) %>%
  fit(data=train)

boost_models <- fit_resamples(boost_wf,
                              resamples = folds,
                              metrics = metric_set(roc_auc),
                              control = tunedModel)

# Random forest
classForest_model <- rand_forest(mtry = tune(), # how many var are considered
                                 min_n=tune(), # how many observations per leaf
                                 trees=1000) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Set Workflow

classForest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(classForest_model)

forest_tuning_grid <- grid_regular(mtry(range =c(1,1)),
                                   min_n(),
                                   levels = 5)

forest_models <- classForest_wf %>%
  tune_grid(resamples=folds,
            grid=forest_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)

my_stack <- stacks() %>%
  add_candidates(forest_models) %>%
  add_candidates(boost_models)

stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members()

stack_preds <- stack_mod %>%
  predict(new_data=test,type="prob") %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(RefId, .pred_1) %>% #Just keep resource and predictions
  rename(IsBadBuy=.pred_1)
