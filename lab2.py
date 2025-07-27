import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)

def detectar_header(df):
    for i in range(10):
        row = df.iloc[i].astype(str).str.lower()
        if any(word in row.to_string() for word in ['regular', 'super', 'súper', 'diesel', 'gas']):
            return i
    return 0

def cargar_con_encabezado(path):
    preview = pd.read_excel(path, header=None, nrows=15)
    header_row = detectar_header(preview)
    df = pd.read_excel(path, header=header_row)
    df.columns = df.columns.str.lower().str.strip()
    return df

def preparar_df(df, fuente):
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.dropna(subset=['fecha'])
    for col in ['diesel bajo azufre', 'diesel ultra bajo azufre', 'diesel alto azufre']:
        if col not in df.columns:
            df[col] = 0
    df_filtrado = pd.DataFrame({
        'fecha': df['fecha'],
        'gasolina_regular': df.get('gasolina regular', 0),
        'gasolina_superior': df.get('gasolina superior', 0),
        'gas_licuado': df.get('gas licuado de petróleo', 0),
        'diesel': df['diesel bajo azufre'] + df['diesel ultra bajo azufre'] + df['diesel alto azufre'],
        'fuente': fuente
    })
    return df_filtrado

# cargar datos
files = {
    "consumo_2024": "CONSUMO-HIDROCARBUROS-2024-12.xlsx",
    "ventas_2025": "VENTAS-HIDROCARBUROS-2025-05.xlsx",
    "import_2024": "IMPORTACION-HIDROCARBUROS-VOLUMEN-2024-12.xlsx",
    "import_2025": "IMPORTACION-HIDROCARBUROS-VOLUMEN-2025-05.xlsx"
}

dfs = {k: cargar_con_encabezado(path) for k, path in files.items()}

consumo_total = pd.concat([
    preparar_df(dfs["consumo_2024"], "consumo"),
    preparar_df(dfs["ventas_2025"], "ventas")
], ignore_index=True)

importacion_total = pd.concat([
    preparar_df(dfs["import_2024"], "importacion"),
    preparar_df(dfs["import_2025"], "importacion")
], ignore_index=True)

# crear series
serie_importacion_diesel = importacion_total.set_index('fecha')['diesel'].resample('ME').sum()
serie_importacion_diesel = serie_importacion_diesel.dropna()

serie_consumo_gasolina = consumo_total.set_index('fecha')['gasolina_superior'].resample('ME').sum()
serie_consumo_gasolina = serie_consumo_gasolina.dropna()

def crear_conjuntos_entrenamiento_prueba(serie, test_years=3, min_train_obs=24):
    if serie.index.freq is None:
        serie = serie.asfreq('MS')
    
    meses_test = test_years * 12
    total = len(serie)
    if total - meses_test < min_train_obs:
        meses_test = max(total - min_train_obs, int(total * 0.2))
        meses_test = max(meses_test, 0)
    fecha_corte = serie.index[-meses_test] if meses_test > 0 else serie.index[0]
    train = serie[serie.index <= fecha_corte]
    test = serie[serie.index > fecha_corte]
    return train, test, fecha_corte

def supervisada(serie, retrasos=1):
    serie_x = []
    serie_y = []
    for i in range(len(serie) - retrasos):
        valor = serie[i:(i + retrasos), 0]
        valor_sig = serie[i + retrasos, 0]
        serie_x.append(valor)
        serie_y.append(valor_sig)
    return np.array(serie_x), np.array(serie_y)

def crear_modelo_lstm(n_steps, n_features, lstm_units, dropout_rate, learning_rate, layers):
    model = Sequential()
    
    if layers == 1:
        model.add(LSTM(lstm_units, input_shape=(n_steps, n_features)))
    else:
        model.add(LSTM(lstm_units, return_sequences=True, input_shape=(n_steps, n_features)))
        for i in range(layers - 2):
            model.add(LSTM(lstm_units, return_sequences=True))
        model.add(LSTM(lstm_units))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def evaluar_modelo_lstm(y_true, y_pred, nombre_modelo):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan
    
    print(f"métricas para {nombre_modelo}:")
    print(f"   mae: {mae:.4f}")
    print(f"   rmse: {rmse:.4f}")
    print(f"   mape: {mape:.2f}%" if not np.isnan(mape) else "   mape: n/a")
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def prediccion_fun(data, modelo, batch_size, scaler, serie_original=None, inicio_idx=0):
    prediccion = [0] * len(data)
    for i, X in enumerate(data):
        X = np.reshape(X, (1, len(X), 1))
        yhat = modelo.predict(X, batch_size=batch_size, verbose=0)
        # invertir escalado
        yhat = scaler.inverse_transform(yhat)
        # almacenar
        prediccion[i] = yhat[0][0]
    return np.array(prediccion)

def entrenar_y_evaluar_lstm(train_data, test_data, config, nombre_serie, config_name):
    scaler = MinMaxScaler()
    
    # preparar datos de entrenamiento
    train_values = train_data.values.reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_values)
    
    # preparar datos de prueba
    test_values = test_data.values.reshape(-1, 1)
    test_scaled = scaler.transform(test_values)
    
    # crear series supervisadas
    x_train, y_train = supervisada(train_scaled, config['n_steps'])
    x_test, y_test = supervisada(test_scaled, config['n_steps'])
    
    if len(x_train) == 0 or len(x_test) == 0:
        print(f"no hay suficientes datos para crear secuencias con n_steps={config['n_steps']}")
        return None, None
    
    # reshape para lstm
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    model = crear_modelo_lstm(
        n_steps=config['n_steps'],
        n_features=1,
        lstm_units=config['lstm_units'],
        dropout_rate=config['dropout_rate'],
        learning_rate=config['learning_rate'],
        layers=config['layers']
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        x_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # hacer predicciones usando la función adaptada
    predicciones = prediccion_fun(x_test, model, config['batch_size'], scaler)
    
    # obtener valores reales para comparación
    test_values_for_comparison = test_data.iloc[config['n_steps']:].values
    
    metricas = evaluar_modelo_lstm(test_values_for_comparison, predicciones, f"{nombre_serie} - {config_name}")
    
    # crear fechas para las predicciones
    fechas_pred = test_data.index[config['n_steps']:]
    
    return model, {'predicciones': predicciones, 
                   'fechas': fechas_pred,
                   'reales': test_values_for_comparison,
                   'metricas': metricas,
                   'history': history,
                   'scaler': scaler}

def predecir_futuro(model, scaler, serie_completa, n_steps, periodos_futuros, batch_size=1):
    predicciones_futuras = []
    
    # obtener últimos datos y escalar
    ultimos_valores = serie_completa.values[-n_steps:].reshape(-1, 1)
    ultimos_escalados = scaler.transform(ultimos_valores)
    
    # usar los últimos valores como punto de partida
    datos_actuales = ultimos_escalados.flatten()
    
    for _ in range(periodos_futuros):
        # tomar los últimos n_steps
        X = datos_actuales[-n_steps:].reshape(1, n_steps, 1)
        
        # predecir
        yhat = model.predict(X, batch_size=batch_size, verbose=0)
        yhat_real = scaler.inverse_transform(yhat)[0][0]
        
        predicciones_futuras.append(yhat_real)
        
        # actualizar datos actuales con la predicción escalada
        yhat_escalado = scaler.transform([[yhat_real]])[0][0]
        datos_actuales = np.append(datos_actuales, yhat_escalado)
    
    return np.array(predicciones_futuras)

# series a analizar
series_analizar = {
    'Importacion_Diesel': serie_importacion_diesel,
    'Consumo_Gasolina_Superior': serie_consumo_gasolina
}

# configuraciones lstm adaptadas
configuraciones_lstm = {
    'Importacion_Diesel': {
        'LSTM_Diesel_1': {
            'n_steps': 16, 
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'layers': 3,
            'epochs': 500,
            'batch_size': 8
        },
        'LSTM_Diesel_2': {
            'n_steps':24, 
            'lstm_units': 300,
            'dropout_rate': 0.15,
            'learning_rate': 0.0007,
            'layers': 6,
            'epochs': 350,
            'batch_size': 6
        }
    },
    'Consumo_Gasolina_Superior': {
        'LSTM_Gasolina_1': {
            'n_steps': 3, 
            'lstm_units': 128,
            'dropout_rate': 0.5,
            'learning_rate': 0.0005,
            'layers': 4,
            'epochs': 450,
            'batch_size': 3
        },
        'LSTM_Gasolina_2': {
            'n_steps': 1, 
            'lstm_units': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'layers': 2,
            'epochs': 300,
            'batch_size': 2
        }
    }
}

print("laboratorio 2 - modelos lstm para series de tiempo")
print("objetivo: crear 2 modelos lstm por cada serie de tiempo")
print("series analizadas:")
print("1. importación de diesel (mensual)")
print("2. consumo de gasolina superior (mensual)")

resultados_lstm = {}
mejores_modelos = {}

for nombre_serie, serie in series_analizar.items():
    configs_serie = configuraciones_lstm.get(nombre_serie, {})
    print(f"\nanálisis lstm: {nombre_serie.replace('_', ' ')}")
    
    train, test, fecha_corte = crear_conjuntos_entrenamiento_prueba(serie)
    
    print(f"división de datos:")
    print(f"   entrenamiento: {train.index.min().strftime('%Y-%m')} a {train.index.max().strftime('%Y-%m')} ({len(train)} obs)")
    print(f"   prueba: {test.index.min().strftime('%Y-%m')} a {test.index.max().strftime('%Y-%m')} ({len(test)} obs)")
    
    if len(test) == 0 or len(train) < 10:
        print("no hay suficientes datos para lstm")
        continue
    
    resultados_serie = {}
    mejor_mae = float('inf')
    mejor_config_nombre = None
    
    for config_name, config in configs_serie.items():
        print(f"\nentrenando: {config_name}")
        print(f"parámetros:")
        print(f"   ventana temporal (n_steps): {config['n_steps']}")
        print(f"   unidades lstm: {config['lstm_units']}")
        print(f"   capas lstm: {config['layers']}")
        print(f"   dropout: {config['dropout_rate']}")
        print(f"   learning rate: {config['learning_rate']}")
        print(f"   épocas máximas: {config['epochs']}")
        print(f"   batch size: {config['batch_size']}")
        
        try:
            modelo, resultados = entrenar_y_evaluar_lstm(train, test, config, nombre_serie, config_name)
            if modelo is not None and resultados is not None:
                resultados_serie[config_name] = {
                    'modelo': modelo,
                    'resultados': resultados,
                    'config': config
                }
                
                # rastrear el mejor modelo
                mae_actual = resultados['metricas']['MAE']
                if mae_actual < mejor_mae:
                    mejor_mae = mae_actual
                    mejor_config_nombre = config_name
                    
                print(f"{config_name} entrenado exitosamente")
            else:
                print(f"falló la configuración {config_name}")
        except Exception as e:
            print(f"error en {config_name}: {str(e)}")
    
    if resultados_serie:
        # guardar mejor modelo
        mejores_modelos[nombre_serie] = {
            'nombre': mejor_config_nombre,
            'mae': mejor_mae,
            'modelo_data': resultados_serie[mejor_config_nombre]
        }
        
        print(f"mejor modelo para {nombre_serie.replace('_', ' ')}: {mejor_config_nombre} (mae: {mejor_mae:.4f})")
        
        # generar predicciones futuras con el mejor modelo
        print(f"generando predicciones futuras...")
        mejor_modelo_data = resultados_serie[mejor_config_nombre]
        modelo = mejor_modelo_data['modelo']
        scaler = mejor_modelo_data['resultados']['scaler']
        config = mejor_modelo_data['config']
        
        # predecir 6 meses futuros
        predicciones_6_meses = predecir_futuro(modelo, scaler, serie, config['n_steps'], 6, config['batch_size'])
        
        # crear fechas futuras
        ultima_fecha = serie.index[-1]
        fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1), periods=6, freq='ME')
        
        print(f"predicciones para los próximos 6 meses:")
        for fecha, pred in zip(fechas_futuras, predicciones_6_meses):
            print(f"   {fecha.strftime('%Y-%m')}: {pred:.2f}")
        
        # visualización
        plt.figure(figsize=(16, 12))
        
        # subplot 1: comparación de predicciones en test
        plt.subplot(3, 1, 1)
        plt.plot(serie.index, serie.values, 'k-', label='Serie Original', linewidth=2)
        plt.axvline(x=fecha_corte, color='red', linestyle='--', alpha=0.7, label='División Train/Test')
        
        colors = ['blue', 'green']
        for i, (config_name, datos) in enumerate(resultados_serie.items()):
            resultados = datos['resultados']
            plt.plot(resultados['fechas'], resultados['predicciones'], 
                    color=colors[i], 
                    label=f'{config_name} (MAE: {resultados["metricas"]["MAE"]:.2f})',
                    linewidth=2, alpha=0.8, marker='o', markersize=4)
        
        plt.title(f'Comparación de Modelos LSTM - {nombre_serie.replace("_", " ")}', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # subplot 2: predicciones futuras con contexto histórico extendido
        plt.subplot(3, 1, 2)
        # Mostrar más datos históricos para mejor contexto (18 meses en lugar de 12)
        serie_reciente = serie.iloc[-18:]
        plt.plot(serie_reciente.index, serie_reciente.values, 'k-', label='Serie Histórica', linewidth=3, alpha=0.8)
        
        # Agregar los datos del conjunto de test para mostrar la continuidad
        if len(test) > 0:
            plt.plot(test.index, test.values, 'gray', label='Datos Test (Reales)', 
                    linewidth=2, alpha=0.6, linestyle=':', marker='s', markersize=3)
        
        # Generar predicciones futuras para cada modelo
        for i, (config_name, datos) in enumerate(resultados_serie.items()):
            modelo = datos['modelo']
            scaler = datos['resultados']['scaler']
            config = datos['config']
            pred_futuras = predecir_futuro(modelo, scaler, serie, config['n_steps'], 6, config['batch_size'])
        
            # Usamos el mismo rango de fechas (una sola vez)
            if i == 0:
                ultima_fecha = serie.index[-1]
                fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1), periods=6, freq='ME')
            
            plt.plot(fechas_futuras, pred_futuras,
                    label=f'{config_name} - Predicciones', 
                    color=colors[i % len(colors)], 
                    marker='o', linewidth=2, markersize=5)
        
        # Línea vertical para marcar el inicio de las predicciones
        plt.axvline(x=ultima_fecha, color='red', linestyle='--', alpha=0.7, 
                   label='Inicio Predicciones', linewidth=2)
        
        plt.title(f'Predicciones Futuras con Contexto - {nombre_serie.replace("_", " ")}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # subplot 3: curvas de entrenamiento
        plt.subplot(3, 1, 3)
        for i, (config_name, datos) in enumerate(resultados_serie.items()):
            history = datos['resultados']['history']
            plt.plot(history.history['loss'], color=colors[i], 
                    label=f'{config_name} - Training Loss', alpha=0.7)
            plt.plot(history.history['val_loss'], color=colors[i], 
                    linestyle='--', label=f'{config_name} - Validation Loss', alpha=0.7)
        
        plt.title('Curvas de Entrenamiento', fontsize=14, fontweight='bold')
        plt.xlabel('Épocas')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"imagenes/{nombre_serie}_LSTM_Completo.png", dpi=300, bbox_inches='tight')
        print(f"gráfico guardado: imagenes/{nombre_serie}_LSTM_Completo.png")
        plt.show()
        
        # tabla de comparación
        print(f"tabla de comparación - {nombre_serie.replace('_', ' ')}")
        metricas_comparacion = []
        for config_name, datos in resultados_serie.items():
            metricas = datos['resultados']['metricas']
            config = datos['config']
            metricas_comparacion.append({
                'Modelo': config_name,
                'MAE': f"{metricas['MAE']:.4f}",
                'RMSE': f"{metricas['RMSE']:.4f}",
                'MAPE': f"{metricas['MAPE']:.2f}%" if not np.isnan(metricas['MAPE']) else "N/A",
                'Ventana': config['n_steps'],
                'Unidades': config['lstm_units'],
                'Capas': config['layers']
            })
        
        df_metricas = pd.DataFrame(metricas_comparacion)
        print(df_metricas.to_string(index=False))
        
    resultados_lstm[nombre_serie] = {
        'train': train,
        'test': test,
        'modelos': resultados_serie,
        'predicciones_futuras': predicciones_6_meses,
        'fechas_futuras': fechas_futuras
    }

# resumen final
print("resumen final - mejores modelos lstm por serie")

for nombre_serie, mejor_info in mejores_modelos.items():
    modelo_data = mejor_info['modelo_data']
    config = modelo_data['config']
    metricas = modelo_data['resultados']['metricas']
    
    print(f"{nombre_serie.replace('_', ' ')}:")
    print(f"   mejor modelo: {mejor_info['nombre']}")
    print(f"   mae: {metricas['MAE']:.4f}")
    print(f"   rmse: {metricas['RMSE']:.4f}")
    print(f"   mape: {metricas['MAPE']:.2f}%" if not np.isnan(metricas['MAPE']) else "   mape: n/a")
    print(f"   configuración:")
    print(f"      ventana temporal: {config['n_steps']} meses")
    print(f"      unidades lstm: {config['lstm_units']}")
    print(f"      capas lstm: {config['layers']}")
    print(f"      dropout: {config['dropout_rate']}")

print(f"archivos generados:")
for nombre_serie in series_analizar.keys():
    print(f"   imagenes/{nombre_serie}_LSTM_Completo.png")

print("laboratorio completado exitosamente")
print("   2 modelos lstm creados para cada serie")
print("   tuneo de parámetros realizado")
print("   mejores modelos seleccionados")
print("   predicciones futuras generadas")
