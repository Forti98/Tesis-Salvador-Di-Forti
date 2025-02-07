# Modelado de Datos de NBA 2019-2020

# Paquetes
pacman::p_load(readxl, janitor, dplyr, skimr, corrplot, tidymodels, magrittr, vip, ggplot2, randomForest,stringr,parallel,openxlsx)

# Importar y limpiar datos
data <- read_excel("data.xlsx", sheet = "data") %>%
  clean_names() %>%
  mutate(
    conf = case_when(
      tm %in% c("WAS", "CHO", "TOT", "PHI", "DET", "ORL", "CLE", "IND", "TOR", "MIA", "ATL", "BOS", "BRK", "MIL", "CHI", "NYK") ~ "EAST",
      TRUE ~ "WEST"
    ),
    across(player_year, as.factor), # Convertir a factor
    salary = round(salary/1000000,4),
    type_player = case_when(
      pos %in% c("PG","SG","SF") ~ "EXT", TRUE ~ "INT"
    )) %>% 
  select(c(-player, -tm, -pos, -g)) %>% # Eliminar columnas innecesarias 
  filter(salary <= 40)

ggplot() +
  geom_density(aes(data$salary))

# dataExt <- data %>% 
#   filter(pos %in% c('PG','SG','SF')) %>% 
#   select(-pos)
# 
# dataInt <- data %>% 
#   filter(pos %in% c('PF','C')) %>% 
#   select(-pos)

# Exploración de los datos
data %>%
  glimpse() # Ver la estructura de los datos

skim(data) # Resumen estadístico de los datos

# Evaluar las correlaciones
data_cor <- cor(data %>% dplyr::select_if(is.numeric))
corrplot::corrplot(data_cor, tl.cex = 0.5) # Visualizar correlaciones


#Jugadores Exteriores
# Dividir los datos en conjunto de entrenamiento y prueba respetando la secuencia temporal PG
train_data <- data %>% filter(str_detect(player_year, '2019|2020|2021|2022|2023')) # Datos de entrenamiento
test_data <- data %>% filter(str_detect(player_year, '2024')) # Datos de prueba




# Crear índices para el análisis y evaluación
analysis_idx <- which(str_detect(data$player_year, '2019|2020|2021|2022|2023'))
assessment_idx <- which(str_detect(data$player_year, '2024'))

# Convertir a rsplit
train_test_split <- rsample::make_splits(list(analysis = analysis_idx, assessment = assessment_idx), data = data)

# Elaboración de una receta
simple_rec <- train_data %>%
  recipes::recipe(salary ~ .) %>% # Definir la receta para predecir el salario
  update_role(player_year, new_role = "id_player") %>% # Cambiar el rol de la variable
  step_dummy(conf,type_player, one_hot = TRUE) %>% # Convertir variables categóricas en dummy
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
mae_metrics <- yardstick::metric_set(mae,rsq,rmse)
set.seed(1234)
resample_RF_fit <- tune::fit_resamples(RF_wflow, resamples = vfold_data, metrics = mae_metrics) # Ajusta el modelo usando validación cruzada
results <- collect_metrics(resample_RF_fit) # Obtener métricas

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

doParallel::registerDoParallel(cores = 6) # Usar 2 cores para la sintonización
set.seed(1234)
tune_RF_results <- tune_grid(object = RF_tune_wflow, resamples = vfold_data, grid = 20, metrics = mae_metrics) # Sintonizar hiperparámetros
tune_RF_results

tune_RF_results %>%
  collect_metrics()

show_best(tune_RF_results, metric = "mae", n = 1) # Mostrar el mejor modelo según MAE

tuned_RF_values <- select_best(tune_RF_results, metric = "mae") # Seleccionar los mejores hiperparámetros
tuned_RF_values

RF_tuned_wflow <- RF_tune_wflow %>%
  tune::finalize_workflow(tuned_RF_values) # Finalizar el flujo de trabajo con hiperparámetros sintonizados

overallfit <- tune::last_fit(RF_tuned_wflow, split = train_test_split, metrics = mae_metrics) # Ajustar modelo con hiperparámetros sintonizados
final_metrics <- collect_metrics(overallfit)

#Guardar estos valores en una tabla
tabla2024 <- data.frame(
  ano = ('2024'),
  mae = round(final_metrics[1,3],2),
  rsq = round(final_metrics[2,3],2)# Asegurarse de que se está usando MAE
)
#tablaext <- tabla %>% 
  #pivot_wider(names_from = ano, values_from = .estimate)

test_predictions <- collect_predictions(overallfit) # Predecir valores para datos de prueba
head(test_predictions)
predicted_values <- test_predictions$.pred
# Agregar las predicciones al conjunto de datos de entrenamiento
tablapredicciones <- test_data %>%
  mutate(predicted_salary = round(predicted_values,2)) %>% 
  mutate(diferencias = (salary - predicted_salary)) %>% 
  arrange(diferencias) %>% 
  select(player_year, salary, predicted_salary, diferencias)

# Gráfico de predicciones
test_predictions %>% 
  ggplot(aes(x = salary, y = .pred)) + 
  geom_point() + 
  xlab("actual outcome values") + 
  ylab("predicted outcome values") +
  geom_smooth(method = "lm") # Agregar línea de regresión

# Supongamos que ya tienes los valores de MAE y R² calculados
mae <- tabla2024[1,2]  # Reemplaza con tu valor real de MAE
rsq <- tabla2024[1,3]

ggplot() +
  geom_density(aes(x = tablapredicciones$salary, color = "Real Salary"), fill = "blue", alpha = 0.2, size = 0.8) +
  geom_density(aes(x = tablapredicciones$predicted_salary, color = "Predicted Salary"), fill = "red", alpha = 0.2, size = 0.8) +
  labs(title = "NBA Perimeter Players' Salary Prediction for 2024",
       subtitle = "Real vs Predicted Salaries using Random Forest",
       x = "Salary (in USD)",
       y = "Density",
       color = "Legend") +
  scale_color_manual(values = c("Real Salary" = "#1f77b4", "Predicted Salary" = "#ff7f0e")) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.title.x = element_text(face = "bold", size = 10),
    axis.title.y = element_text(face = "bold", size = 10),
    axis.text = element_text(size = 8),
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 8)
  ) +
  annotate("text", x = max(tablapredicciones$salary), y = 0.05,  # Ajusta la posición vertical
           label = paste("Estimating 2024 salaries for NBA perimeter players\n",
                         "MAE:", mae, "\n",
                         "R²:", rsq), 
           hjust = 1, color = "black", size = 4)  # Aumenta el tamaño del texto











