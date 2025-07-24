# Lab 2

---

## Configuraciones para modelos LSTM

Estas configuraciones definen distintos modelos LSTM que se utilizaron en el laboratorio, y corresponden a las imagenes en el directorio *imagenes/*.

### Lista de configuraciones

#### Config 1

* `n_steps`: 12
* `lstm_units`: 50
* `dropout_rate`: 0.2
* `learning_rate`: 0.001
* `layers`: 1
* `epochs`: 100
* `batch_size`: 32

#### Config 2

* `n_steps`: 24
* `lstm_units`: 100
* `dropout_rate`: 0.3
* `learning_rate`: 0.0005
* `layers`: 2
* `epochs`: 150
* `batch_size`: 16

#### Config 3

* `n_steps`: 18
* `lstm_units`: 75
* `dropout_rate`: 0.25
* `learning_rate`: 0.002
* `layers`: 3
* `epochs`: 120
* `batch_size`: 24

#### Config 4

* `n_steps`: 6
* `lstm_units`: 25
* `dropout_rate`: 0.1
* `learning_rate`: 0.003
* `layers`: 1
* `epochs`: 80
* `batch_size`: 64

### Función de cada parámetro

* **`n_steps`**: Cantidad de pasos de tiempo que el modelo usará como entrada para predecir el siguiente valor. 
* **`lstm_units`**: Número de neuronas en cada capa LSTM. 
* **`dropout_rate`**: Porcentaje de unidades que se apagan aleatoriamente durante el entrenamiento.
* **`learning_rate`**: Tasa de aprendizaje del optimizador. Controla qué tanto se ajustan los pesos del modelo con cada iteración.
* **`layers`**: Número de capas LSTM apiladas.
* **`epochs`**: Número de veces que el modelo verá todos los datos de entrenamiento.
* **`batch_size`**: Tamaño del lote de datos que se usa para actualizar los pesos en cada paso de entrenamiento.

