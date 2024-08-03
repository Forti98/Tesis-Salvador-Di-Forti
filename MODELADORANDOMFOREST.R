# Modelado de Datos de NBA 2019-2020

# Paquetes
pacman::p_load(readxl, janitor, dplyr, skimr, corrplot, tidymodels, magrittr, vip, ggplot2, randomForest,stringr,parallel,openxlsx)

# Importar y limpiar datos
data <- read_excel("data.xlsx", sheet = "final") %>%
  clean_names() %>%
  mutate(
    conf = case_when(
      tm %in% c("WAS", "CHO", "TOT", "PHI", "DET", "ORL", "CLE", "IND", "TOR", "MIA", "ATL", "BOS", "BRK", "MIL", "CHI", "NYK") ~ "EAST",
      TRUE ~ "WEST"
    ),
    across(player_year, as.factor), # Convertir a factor
    salary = round(salary / 1000000, 2) # Dividir el salario para manejarlo en millones
  ) %>%
  select(c(-player, -tm, -age, -g)) # Eliminar columnas innecesarias

dataPG <- data %>% 
  filter(pos == 'PG') %>% 
  select(-pos)
dataSG <- data %>% 
  filter(pos == 'SG')%>% 
  select(-pos)
dataSF <- data %>% 
  filter(pos == 'SF')%>% 
  select(-pos)
dataPF <- data %>% 
  filter(pos == 'PF')%>% 
  select(-pos)
dataC <- data %>% 
  filter(pos == 'C')%>% 
  select(-pos)

# Exploración de los datos
data %>%
  glimpse() # Ver la estructura de los datos

skim(data) # Resumen estadístico de los datos

# Evaluar las correlaciones
data_cor <- cor(data %>% dplyr::select_if(is.numeric))
corrplot::corrplot(data_cor, tl.cex = 0.5) # Visualizar correlaciones

#POS PG
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataPG %>% filter(str_detect(player_year, '2019|2020')) # Datos de entrenamiento
test_data <- dataPG %>% filter(str_detect(player_year, '2021')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataPG$player_year, '2019|2020'))
assessment_idx <- which(str_detect(dataPG$player_year, '2021'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('PG'),
  ano = ('2021'),
  rmse = round(final_metrics[1,3],2)
)
tablapg <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión



#POS SG
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal SG
train_data <- dataSG %>% filter(str_detect(player_year, '2019|2020')) # Datos de entrenamiento
test_data <- dataSG %>% filter(str_detect(player_year, '2021')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataSG$player_year, '2019|2020'))
assessment_idx <- which(str_detect(dataSG$player_year, '2021'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataSG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('SG'),
  ano = ('2021'),
  rmse = round(final_metrics[1,3],2)
)
tablaSG <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS SF
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataSF %>% filter(str_detect(player_year, '2019|2020')) # Datos de entrenamiento
test_data <- dataSF %>% filter(str_detect(player_year, '2021')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataSF$player_year, '2019|2020'))
assessment_idx <- which(str_detect(dataSF$player_year, '2021'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('SF'),
  ano = ('2021'),
  rmse = round(final_metrics[1,3],2)
)
tablaSF <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS PF
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataPF %>% filter(str_detect(player_year, '2019|2020')) # Datos de entrenamiento
test_data <- dataPF %>% filter(str_detect(player_year, '2021')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataPF$player_year, '2019|2020'))
assessment_idx <- which(str_detect(dataPF$player_year, '2021'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPF)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('PF'),
  ano = ('2021'),
  rmse = round(final_metrics[1,3],2)
)
tablapf <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS C
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataC %>% filter(str_detect(player_year, '2019|2020')) # Datos de entrenamiento
test_data <- dataC %>% filter(str_detect(player_year, '2021')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataC$player_year, '2019|2020'))
assessment_idx <- which(str_detect(dataC$player_year, '2021'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataC)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('C'),
  ano = ('2021'),
  rmse = round(final_metrics[1,3],2)
)
tablaC <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión


tabla2021 <- rbind(tablapg,tablaSG,tablaSF,tablapf,tablaC)

# Modelado de Datos de NBA 2020-2021

#POS PG
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataPG %>% filter(str_detect(player_year, '2020|2021')) # Datos de entrenamiento
test_data <- dataPG %>% filter(str_detect(player_year, '2022')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataPG$player_year, '2020|2021'))
assessment_idx <- which(str_detect(dataPG$player_year, '2022'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('PG'),
  ano = ('2022'),
  rmse = round(final_metrics[1,3],2)
)
tablapg <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión



#POS SG
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal SG
train_data <- dataSG %>% filter(str_detect(player_year, '2020|2021')) # Datos de entrenamiento
test_data <- dataSG %>% filter(str_detect(player_year, '2022')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataSG$player_year, '2020|2021'))
assessment_idx <- which(str_detect(dataSG$player_year, '2022'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataSG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('SG'),
  ano = ('2022'),
  rmse = round(final_metrics[1,3],2)
)
tablaSG <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS SF
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataSF %>% filter(str_detect(player_year, '2020|2021')) # Datos de entrenamiento
test_data <- dataSF %>% filter(str_detect(player_year, '2022')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataSF$player_year, '2020|2021'))
assessment_idx <- which(str_detect(dataSF$player_year, '2022'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('SF'),
  ano = ('2022'),
  rmse = round(final_metrics[1,3],2)
)
tablaSF <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS PF
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataPF %>% filter(str_detect(player_year, '2020|2021')) # Datos de entrenamiento
test_data <- dataPF %>% filter(str_detect(player_year, '2022')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataPF$player_year, '2020|2021'))
assessment_idx <- which(str_detect(dataPF$player_year, '2022'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPF)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('PF'),
  ano = ('2022'),
  rmse = round(final_metrics[1,3],2)
)
tablapf <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS C
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataC %>% filter(str_detect(player_year, '2020|2021')) # Datos de entrenamiento
test_data <- dataC %>% filter(str_detect(player_year, '2022')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataC$player_year, '2020|2021'))
assessment_idx <- which(str_detect(dataC$player_year, '2022'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataC)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('C'),
  ano = ('2022'),
  rmse = round(final_metrics[1,3],2)
)
tablaC <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión


tabla2022 <- rbind(tablapg,tablaSG,tablaSF,tablapf,tablaC)

# Modelado de Datos de NBA 2021-2022

#POS PG
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataPG %>% filter(str_detect(player_year, '2021|2022')) # Datos de entrenamiento
test_data <- dataPG %>% filter(str_detect(player_year, '2023')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataPG$player_year, '2021|2022'))
assessment_idx <- which(str_detect(dataPG$player_year, '2023'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('PG'),
  ano = ('2023'),
  rmse = round(final_metrics[1,3],2)
)
tablapg <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión



#POS SG
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal SG
train_data <- dataSG %>% filter(str_detect(player_year, '2021|2022')) # Datos de entrenamiento
test_data <- dataSG %>% filter(str_detect(player_year, '2023')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataSG$player_year, '2021|2022'))
assessment_idx <- which(str_detect(dataSG$player_year, '2023'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataSG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('SG'),
  ano = ('2023'),
  rmse = round(final_metrics[1,3],2)
)
tablaSG <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS SF
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataSF %>% filter(str_detect(player_year, '2021|2022')) # Datos de entrenamiento
test_data <- dataSF %>% filter(str_detect(player_year, '2023')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataSF$player_year, '2021|2022'))
assessment_idx <- which(str_detect(dataSF$player_year, '2023'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('SF'),
  ano = ('2023'),
  rmse = round(final_metrics[1,3],2)
)
tablaSF <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS PF
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataPF %>% filter(str_detect(player_year, '2021|2022')) # Datos de entrenamiento
test_data <- dataPF %>% filter(str_detect(player_year, '2023')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataPF$player_year, '2021|2022'))
assessment_idx <- which(str_detect(dataPF$player_year, '2023'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPF)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('PF'),
  ano = ('2023'),
  rmse = round(final_metrics[1,3],2)
)
tablapf <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS C
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataC %>% filter(str_detect(player_year, '2021|2022')) # Datos de entrenamiento
test_data <- dataC %>% filter(str_detect(player_year, '2023')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataC$player_year, '2021|2022'))
assessment_idx <- which(str_detect(dataC$player_year, '2023'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataC)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('C'),
  ano = ('2023'),
  rmse = round(final_metrics[1,3],2)
)
tablaC <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión


tabla2023 <- rbind(tablapg,tablaSG,tablaSF,tablapf,tablaC)

# Modelado de Datos de NBA 2022-2023

#POS PG
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataPG %>% filter(str_detect(player_year, '2022|2023')) # Datos de entrenamiento
test_data <- dataPG %>% filter(str_detect(player_year, '2024')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataPG$player_year, '2022|2023'))
assessment_idx <- which(str_detect(dataPG$player_year, '2024'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('PG'),
  ano = ('2024'),
  rmse = round(final_metrics[1,3],2)
)
tablapg <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión



#POS SG
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal SG
train_data <- dataSG %>% filter(str_detect(player_year, '2022|2023')) # Datos de entrenamiento
test_data <- dataSG %>% filter(str_detect(player_year, '2024')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataSG$player_year, '2022|2023'))
assessment_idx <- which(str_detect(dataSG$player_year, '2024'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataSG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('SG'),
  ano = ('2024'),
  rmse = round(final_metrics[1,3],2)
)
tablaSG <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS SF
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataSF %>% filter(str_detect(player_year, '2022|2023')) # Datos de entrenamiento
test_data <- dataSF %>% filter(str_detect(player_year, '2024')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataSF$player_year, '2022|2023'))
assessment_idx <- which(str_detect(dataSF$player_year, '2024'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPG)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('SF'),
  ano = ('2024'),
  rmse = round(final_metrics[1,3],2)
)
tablaSF <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS PF
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataPF %>% filter(str_detect(player_year, '2022|2023')) # Datos de entrenamiento
test_data <- dataPF %>% filter(str_detect(player_year, '2024')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataPF$player_year, '2022|2023'))
assessment_idx <- which(str_detect(dataPF$player_year, '2024'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataPF)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('PF'),
  ano = ('2024'),
  rmse = round(final_metrics[1,3],2)
)
tablapf <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión




#POS C
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- dataC %>% filter(str_detect(player_year, '2022|2023')) # Datos de entrenamiento
test_data <- dataC %>% filter(str_detect(player_year, '2024')) # Datos de prueba

# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(dataC$player_year, '2022|2023'))
assessment_idx <- which(str_detect(dataC$player_year, '2024'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = dataC)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
  step_corr(all_predictors(), -ws) %>% # Filtrar variables correlacionadas
  step_nzv(all_predictors(), -ws) # Filtrar variables con varianza cercana a 0

# Ejecución del procesamiento de la receta
prepped_rec <- prep(simple_rec, verbose = TRUE, retain = TRUE) # Preparar la receta
names(prepped_rec) # Nombres de las transformaciones
preproc_train <- bake(prepped_rec, new_data = NULL) # Aplicar la receta a los datos de entrenamiento
glimpse(preproc_train) # Visualizar los datos preparados

baked_test_data <- recipes::bake(prepped_rec, new_data = test_data) # Aplicar la receta a los datos de prueba
glimpse(baked_test_data) # Visualizar los datos de prueba preparados


# Validación Cruzada
set.seed(1234)
vfold_data <- rsample::vfold_cv(data = train_data, v = 10)
vfold_data
pull(vfold_data, splits)


# Modelo de Bosque Aleatorio
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

# Validación Cruzada con modelo de Bosque Aleatorio
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, vfold_data) # Ver el rendimiento del modelo con validación cruzada
collect_metrics(resample_RF_fit)

# Sintonización de Hiperparámetros
tune_RF_model <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression") # Especificar modelo con hiperparámetros sintonizados

tune_RF_model
RF_tune_wflow <- workflows::workflow() %>%
  workflows::add_recipe(simple_rec) %>%
  workflows::add_model(tune_RF_model)

RF_tune_wflow

parallel::detectCores()

doParallel::registerDoParallel(cores = 2) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics() %>%
  head()

show_best(tune_RF_results, metric = "rmse", n = 1) # Mostrar el mejor modelo según RMSE

tuned_RF_values <- select_best(tune_RF_results, "rmse") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados



overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla <- data.frame(
  pos = ('C'),
  ano = ('2024'),
  rmse = round(final_metrics[1,3],2)
)
tablaC <- tabla %>% 
  pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
train_data_with_predictions <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary,diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión


tabla2024 <- rbind(tablapg,tablaSG,tablaSF,tablapf,tablaC)

tablarmse <- tabla2021 %>% 
  left_join(tabla2022, by = 'pos') %>% 
  left_join(tabla2023, by = 'pos') %>% 
  left_join(tabla2024, by = 'pos')

wb <- createWorkbook()
addWorksheet(wb, 'tabla')
writeData(wb = wb, sheet = 1, x = tablarmse, startCol = 1, startRow = 7, colNames = T)
saveWorkbook(wb, 'tablarmse.xlsx', overwrite = TRUE)
