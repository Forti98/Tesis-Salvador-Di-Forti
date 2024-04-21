#Modelado de Datos de NBA

#Paquetes

pacman::p_load(readxl,janitor,dplyr,skimr,corrplot,tidymodels,magrittr,vip,ggplot2,randomForest)

#Importar 
data <- read_excel("data.xlsx", sheet = "final")%>%
  clean_names()%>%
  mutate(conf = case_when(tm %in% c("WAS","CHO","TOT","PHI","DET","ORL","CLE","IND","TOR","MIA",
                                    "ATL","BOS","BRK","MIL","CHI","NYK") ~ "EAST",
                          TRUE ~ "WEST"),
         across(player_year, as.factor),
         salary = salary/1000000)%>%
  select(c(-player,-tm,-age,-g))

#Exploracion de los datos

data %>%
  glimpse()

skim(data)

#Evaluar las correlaciones

data_cor <- cor(data %>% dplyr::select_if(is.numeric))
corrplot::corrplot(data_cor, tl.cex = 0.5)

#Dividir los datos

set.seed(1234)
data_split <- rsample::initial_split(data = data, prop = 2/3)
data_split
train_data <-rsample::training(data_split)
test_data <-rsample::testing(data_split)

#Elaboracion de una recta

simple_rec <- train_data %>%
  recipes::recipe(salary ~ .)%>%#aqui se colocan todas las varibles como predictoras
  update_role(player_year, new_role = "id_player")%>%#cambiamos player_year como id
  step_dummy(pos,conf, one_hot = TRUE)%>%#declaramos como dummy la posicion y el equipo
  step_corr(all_predictors(), -ws)%>%#filtramos las variables que estan muy correlacionadas
  step_nzv(all_predictors(), -ws )#filtramos las variables con varianza cercana 0

simple_rec

#Ejecucion del procesamiento

prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE)#guardamos la receta
names(prepped_rec)
preproc_train <- bake(prepped_rec, new_data = NULL)#vemos que es lo que hace la receta con bake
glimpse(preproc_train)

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data)#aplicamos la receta a los datos de prueba
glimpse(baked_test_data)

#Especificacion del modelo

data_model <- parsnip::linear_reg() #especificamos que se usara una regresion lineal
data_model

lm_data_model <- 
  data_model  %>%
  parsnip::set_engine("lm")%>%#aqui decimos que queremos ajustar la regresion a los minimos cuadrados
  set_mode("regression")#le decimos que queremos predecir
lm_data_model

data_wflow <-workflows::workflow() %>%#aqui generamos un worflow
  workflows::add_recipe(simple_rec) %>%#agregamos la receta
  workflows::add_model(lm_data_model)#aqui agregamos el modelo de prediccion
data_wflow

data_wflow_fit <- parsnip::fit(data_wflow, data = train_data)#estimamos los parametros 
data_wflow_fit

wflowoutput <- data_wflow_fit %>% 
  pull_workflow_fit() %>% 
  broom::tidy() #Aqui ordenamos los resultados de forma ordenada
wflowoutput

data_wflow_fit %>% 
  pull_workflow_fit() %>%#creamos un grafico donde nos indica cuales son las variables que mas contribuyen 
  vip(num_features = 10)

wf_fit <- data_wflow_fit %>% 
  pull_workflow_fit()

wf_fitted_values <- wf_fit$fit$fitted.values
head(wf_fitted_values)

wf_fitted_values <- 
  broom::augment(wf_fit$fit, data = preproc_train) %>% 
  select(salary, .fitted:.std.resid)

head(wf_fitted_values)

values_pred_train <- 
  predict(data_wflow_fit, train_data)
values_pred_train

wf_fitted_values %>% 
  ggplot(aes(x =  salary, y = .fitted)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values")
yardstick::metrics(wf_fitted_values, 
                   truth = salary, estimate = .fitted)
#Validacion Cruzada
set.seed(1234)

vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)
set.seed(122)
resample_fit <- tune::fit_resamples(data_wflow, vfold_data)
resample_fit
collect_metrics(resample_fit)

#Random Forest
datatree_model <- 
  parsnip::rand_forest(mtry = 10, min_n = 4)
datatree_model

RF_data_model <- #especificamos el modelo de randomforest
  datatree_model %>%
  set_engine("randomForest") %>%
  set_mode("regression")

RF_data_model

RF_wflow <- workflows::workflow() %>%#lo agregamos todo en un worflow
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(RF_data_model)
RF_wflow

RF_wflow_fit <- parsnip::fit(RF_wflow, data = train_data)#se ajusta a los datos del modelo
RF_wflow_fit

RF_wflow_fit %>% #vemos la importancia de las variables
  pull_workflow_fit() %>% 
  vip(num_features = 10)

set.seed(456)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data)#se ve el rendimiento del modelo con la validacion cruzada
collect_metrics(resample_RF_fit)

tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression")#con tune haces que los hiperparametros se sintonicen


tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)
RF_tune_wflow

library(parallel)
parallel::detectCores()

doParallel::registerDoParallel(cores=2)
set.seed(123)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20)
tune_RF_results

tune_RF_results%>%
  collect_metrics() %>%
  head()
show_best(tune_RF_results, metric = "rmse", n =1)
tuned_RF_values<- select_best(tune_RF_results, "rmse")
tuned_RF_values
RF_tuned_wflow <-RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values)
overallfit <-tune::last_fit(RF_tuned_wflow, data_split)
collect_metrics(overallfit)
test_predictions <-collect_predictions(overallfit)
head(test_predictions)
test_predictions %>% 
  ggplot(aes(x =  salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm")
