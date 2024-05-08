#Modelado de Datos de NBA

#Paquetes
pacman::p_load(readxl,janitor,dplyr,skimr,corrplot,tidymodels,magrittr,vip,ggplot2,randomForest)

#Importar y limpiar datos
data <- read_excel("data.xlsx", sheet = "final") %>%
  clean_names() %>%
  mutate(
    conf = case_when(
      tm %in% c("WAS","CHO","TOT","PHI","DET","ORL","CLE","IND","TOR","MIA","ATL","BOS","BRK","MIL","CHI","NYK") ~ "EAST",
      TRUE ~ "WEST"
    ),
    across(player_year, as.factor), # Convertir a factor
    salary = salary/1000000 # Dividir el salario para manejarlo en millones
  ) %>%
  select(c(-player,-tm,-age,-g)) # Eliminar columnas innecesarias

#Exploración de los datos
data %>%
  glimpse() # Ver la estructura de los datos

skim(data) # Resumen estadístico de los datos

#Evaluar las correlaciones
data_cor <- cor(data %>% dplyr::select_if(is.numeric))
corrplot::corrplot(data_cor, tl.cex = 0.5) # Visualizar correlaciones

#Dividir los datos en conjunto de entrenamiento y prueba
set.seed(1234)
data_split <- rsample::initial_split(data = data, prop = 2/3)
data_split
train_data <- rsample::training(data_split)
test_data <- rsample::testing(data_split)

#Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(pos, conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

#Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados

#Especificación del modelo de regresión lineal
data_model <- parsnip::linear_reg() # Especificar el modelo de regresión lineal
data_model

lm_data_model <- 
  data_model %>%
  parsnip::set_engine("lm") %>% # Configurar el motor del modelo
  set_mode("regression") # Establecer modo de regresión

lm_data_model

#Crear un flujo de trabajo
data_wflow <- workflows::workflow() %>% # Crear un flujo de trabajo
  workflows::add_recipe(simple_rec) %>% # Agregar la receta
  workflows::add_model(lm_data_model) # Agregar el modelo de regresión lineal

data_wflow

#Ajustar el modelo a los datos de entrenamiento
data_wflow_fit <- parsnip::fit(data_wflow, data = train_data) # Ajustar el modelo a los datos de entrenamiento
data_wflow_fit # Ver el modelo ajustado

#Ordenar los resultados del modelo
wflowoutput <- data_wflow_fit %>% 
  pull_workflow_fit() %>% 
  broom::tidy() # Organizar los resultados de forma ordenada
wflowoutput

#Ver la importancia de las variables
data_wflow_fit %>% 
  pull_workflow_fit() %>% 
  vip(num_features = 10) # Crear un gráfico de la importancia de las variables

wf_fit <- data_wflow_fit %>% 
  pull_workflow_fit()

#Predecir valores para los datos de entrenamiento
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

#Validación Cruzada
set.seed(1234)

vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)
set.seed(122)
resample_fit <- tune::fit_resamples(data_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
resample_fit
collect_metrics(resample_fit)

#Modelo de Bosque Aleatorio
datatree_model <- 
  parsnip::rand_forest(mtry = 10, min_n = 4) # Especificar modelo de bosque aleatorio
datatree_model

RF_data_model <- 
  datatree_model %>%
  set_engine("randomForest") %>% # Configurar el motor del modelo
  set_mode("regression") # Establecer modo de regresión

RF_data_model

RF_wflow <- workflows::workflow() %>% # Crear un flujo de trabajo
  workflows::add_recipe(simple_rec) %>% # Agregar la receta
  workflows::add_model(RF_data_model) # Agregar el modelo de bosque aleatorio

RF_wflow

RF_wflow_fit <- parsnip::fit(RF_wflow, data = train_data) # Ajustar el modelo a los datos de entrenamiento
RF_wflow_fit

RF_wflow_fit %>% # Ver la importancia de las variables
  pull_workflow_fit() %>% 
  vip(num_features = 10)

#Validación Cruzada con modelo de Bosque Aleatorio
set.seed(456)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

#Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

library(parallel)
parallel::detectCores()

doParallel::registerDoParallel(cores=2) # Usar 2 cores para la sintonización
set.seed(123)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, metric = "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados

overallfit <- tune::last_fit(RF_tuned_wflow, data_split) # Ajustar modelo con hiperparámetros sintonizados
collect_metrics(overallfit)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)

#Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x =  salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión
