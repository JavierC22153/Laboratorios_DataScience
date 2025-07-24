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

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

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
    
    print(f"\nMétricas para {nombre_modelo}:")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%" if not np.isnan(mape) else "   MAPE: N/A")
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def entrenar_y_evaluar_lstm(train_data, test_data, config, nombre_serie, config_name):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_data.values.reshape(-1, 1))
    
    X_train, y_train = create_sequences(train_scaled.flatten(), config['n_steps'])
    X_test, y_test = create_sequences(test_scaled.flatten(), config['n_steps'])
    
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"No hay suficientes datos para crear secuencias con n_steps={config['n_steps']}")
        return None, None
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
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
        X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    
    test_values_for_comparison = test_data.iloc[config['n_steps']:].values
    
    metricas = evaluar_modelo_lstm(test_values_for_comparison, y_pred.flatten(), f"{nombre_serie} - {config_name}")
    
    return model, {'predicciones': y_pred.flatten(), 
                   'fechas': test_data.index[config['n_steps']:],
                   'reales': test_values_for_comparison,
                   'metricas': metricas,
                   'history': history,
                   'scaler': scaler}

series_analizar = {
    'Importación Diesel': serie_importacion_diesel,
    'Consumo Gasolina Superior': serie_consumo_gasolina
}

configuraciones_lstm = {
    'Config 1': {
        'n_steps': 12,
        'lstm_units': 50,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'layers': 1,
        'epochs': 100,
        'batch_size': 32
    },
    'Config 2': {
        'n_steps': 24,
        'lstm_units': 100,
        'dropout_rate': 0.3,
        'learning_rate': 0.0005,
        'layers': 2,
        'epochs': 150,
        'batch_size': 16
    },
    'Config 3': {
        'n_steps': 18,
        'lstm_units': 75,
        'dropout_rate': 0.25,
        'learning_rate': 0.002,
        'layers': 3,
        'epochs': 120,
        'batch_size': 24
    },
    'Config 4': {
        'n_steps': 6,
        'lstm_units': 25,
        'dropout_rate': 0.1,
        'learning_rate': 0.003,
        'layers': 1,
        'epochs': 80,
        'batch_size': 64
    }
}

resultados_lstm = {}

for nombre_serie, serie in series_analizar.items():
    print(f"\n{'='*60}")
    print(f"ANÁLISIS LSTM: {nombre_serie}")
    print(f"{'='*60}")
    
    train, test, fecha_corte = crear_conjuntos_entrenamiento_prueba(serie)
    
    print(f"\nDivisión de datos:")
    print(f"   Entrenamiento: {train.index.min().strftime('%Y-%m')} a {train.index.max().strftime('%Y-%m')} ({len(train)} obs)")
    print(f"   Prueba: {test.index.min().strftime('%Y-%m')} a {test.index.max().strftime('%Y-%m')} ({len(test)} obs)")
    
    if len(test) == 0 or len(train) < 30:
        print("⚠️  No hay suficientes datos para LSTM")
        continue
    
    resultados_serie = {}
    
    for config_name, config in configuraciones_lstm.items():
        print(f"\n🔧 Probando configuración: {config_name}")
        print(f"   Parámetros: steps={config['n_steps']}, units={config['lstm_units']}, layers={config['layers']}")
        
        try:
            modelo, resultados = entrenar_y_evaluar_lstm(train, test, config, nombre_serie, config_name)
            if modelo is not None and resultados is not None:
                resultados_serie[config_name] = {
                    'modelo': modelo,
                    'resultados': resultados,
                    'config': config
                }
            else:
                print(f"   ❌ Falló la configuración {config_name}")
        except Exception as e:
            print(f"   ❌ Error en {config_name}: {str(e)}")
    
    resultados_lstm[nombre_serie] = {
        'train': train,
        'test': test,
        'modelos': resultados_serie
    }
    
    if resultados_serie:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(serie.index, serie.values, 'k-', label='Serie Original', linewidth=2)
        plt.axvline(x=fecha_corte, color='red', linestyle='--', alpha=0.7, label='División Train/Test')
        
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (config_name, datos) in enumerate(resultados_serie.items()):
            resultados = datos['resultados']
            plt.plot(resultados['fechas'], resultados['predicciones'], 
                    color=colors[i % len(colors)], 
                    label=f'{config_name} (MAE: {resultados["metricas"]["MAE"]:.1f})',
                    linewidth=2, alpha=0.8)
        
        plt.title(f'Predicciones LSTM: {nombre_serie}')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        for i, (config_name, datos) in enumerate(resultados_serie.items()):
            history = datos['resultados']['history']
            plt.plot(history.history['loss'], color=colors[i % len(colors)], 
                    label=f'{config_name} - Training Loss', alpha=0.7)
            plt.plot(history.history['val_loss'], color=colors[i % len(colors)], 
                    linestyle='--', label=f'{config_name} - Validation Loss', alpha=0.7)
        
        plt.title('Curvas de Entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"imagenes/lstm_resultados_{nombre_serie.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 RESUMEN DE RESULTADOS - {nombre_serie}")
        print("-" * 50)
        metricas_comparacion = []
        for config_name, datos in resultados_serie.items():
            metricas = datos['resultados']['metricas']
            metricas_comparacion.append({
                'Configuración': config_name,
                'MAE': metricas['MAE'],
                'RMSE': metricas['RMSE'],
                'MAPE': metricas['MAPE']
            })
        
        df_metricas = pd.DataFrame(metricas_comparacion)
        df_metricas = df_metricas.sort_values('MAE')
        print(df_metricas.to_string(index=False, float_format='%.3f'))
        
        mejor_config = df_metricas.iloc[0]['Configuración']
        print(f"\n🏆 Mejor configuración: {mejor_config}")

print(f"\n{'='*80}")
print("RESUMEN FINAL DE TODOS LOS MODELOS LSTM")
print(f"{'='*80}")

for nombre_serie, datos in resultados_lstm.items():
    if 'modelos' in datos and datos['modelos']:
        print(f"\n📈 {nombre_serie}:")
        print("-" * 40)
        
        mejor_mae = float('inf')
        mejor_modelo = None
        
        for config_name, modelo_datos in datos['modelos'].items():
            metricas = modelo_datos['resultados']['metricas']
            config = modelo_datos['config']
            
            print(f"   {config_name}:")
            print(f"      MAE: {metricas['MAE']:.3f}")
            print(f"      RMSE: {metricas['RMSE']:.3f}")
            print(f"      MAPE: {metricas['MAPE']:.2f}%")
            print(f"      Parámetros: steps={config['n_steps']}, units={config['lstm_units']}, layers={config['layers']}")
            
            if metricas['MAE'] < mejor_mae:
                mejor_mae = metricas['MAE']
                mejor_modelo = config_name
        
        print(f"\n   🥇 Mejor modelo: {mejor_modelo} (MAE: {mejor_mae:.3f})")
