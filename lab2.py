# -*- coding: utf-8 -*-
"""
Lab1_LSTM_Tuning.py

Script para generar y tunear modelos LSTM para dos series:
- Importación mensual de diesel
- Consumo mensual de gasolina superior

Incluye guardar gráficos de entrenamiento y predicciones.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Crear carpeta de imágenes si no existe
o_images = 'imagenes'
os.makedirs(o_images, exist_ok=True)

# Funciones de carga y preparación de datos
def detectar_header(df):
    for i in range(10):
        row = df.iloc[i].astype(str).str.lower()
        if any(w in row.to_string() for w in ['regular','super','súper','diesel','gas']):
            return i
    return 0

def cargar_con_encabezado(path):
    preview = pd.read_excel(path, header=None, nrows=15)
    hr = detectar_header(preview)
    df = pd.read_excel(path, header=hr)
    df.columns = df.columns.str.lower().str.strip()
    return df

def preparar_df(df, fuente):
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.dropna(subset=['fecha'])
    df['diesel'] = df.get('diesel bajo azufre', 0) + df.get('diesel ultra bajo azufre', 0) + df.get('diesel alto azufre', 0)
    df['gasolina_superior'] = df.get('gasolina superior', 0)
    return pd.DataFrame({
        'fecha': df['fecha'],
        'diesel': df['diesel'],
        'gasolina_superior': df['gasolina_superior'],
        'fuente': fuente
    })

# Carga de datos
files = {
    'import_2024': 'IMPORTACION-HIDROCARBUROS-VOLUMEN-2024-12.xlsx',
    'import_2025': 'IMPORTACION-HIDROCARBUROS-VOLUMEN-2025-05.xlsx',
    'ventas_2025': 'VENTAS-HIDROCARBUROS-2025-05.xlsx'
}
dfs = {k: cargar_con_encabezado(v) for k, v in files.items()}
import_total = pd.concat([
    preparar_df(dfs['import_2024'], 'import'),
    preparar_df(dfs['import_2025'], 'import')
], ignore_index=True)
ventas_total = preparar_df(dfs['ventas_2025'], 'ventas')
imp_diesel = import_total.set_index('fecha')['diesel'].resample('ME').sum().dropna()
con_super   = ventas_total.set_index('fecha')['gasolina_superior'].resample('ME').sum().dropna()

# Partición y supervisado
def crear_train_test(arr, test_years=1):
    m = test_years * 12
    if m > 0:
        return arr[:-m], arr[-m:]
    return arr.copy(), np.array([])

def series_a_supervisada(data, lb=3):
    X, y = [], []
    for i in range(len(data) - lb):
        X.append(data[i:i+lb])
        y.append(data[i+lb])
    return np.array(X), np.array(y)

# Preprocesamiento
def preprocesar(serie):
    sc = MinMaxScaler((0,1))
    scaled = sc.fit_transform(serie.values.reshape(-1,1)).flatten()
    train, test = crear_train_test(scaled, test_years=1)
    lb = 3
    Xtr, Ytr = series_a_supervisada(train, lb)
    start = train[-lb:] if len(train) >= lb else train
    Xte, Yte = series_a_supervisada(np.concatenate([start, test]), lb)
    Xtr = Xtr.reshape(-1, lb, 1)
    Xte = Xte.reshape(-1, lb, 1)
    return sc, Xtr, Ytr, Xte, Yte

results = {
    'ImportDiesel': preprocesar(imp_diesel),
    'ConsGasSup':  preprocesar(con_super)
}

# Función de modelo
from tensorflow.keras.optimizers import Adam

def build_lstm(units=50, drop=0.0):
    model = Sequential()
    model.add(Input(shape=(None,1)))
    model.add(LSTM(units))
    if drop > 0:
        model.add(Dropout(drop))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    return model

# Configuraciones
tuned = [
    {'units':32, 'drop':0.0, 'epochs':50,  'bs':8},
    {'units':64, 'drop':0.2, 'epochs':100, 'bs':16}
]
summary = {}

for name, vals in results.items():
    sc, Xtr, Ytr, Xte, Yte = vals
    best = {'mse': np.inf}
    for cfg in tuned:
        model = build_lstm(cfg['units'], cfg['drop'])
        hist = model.fit(Xtr, Ytr, epochs=cfg['epochs'], batch_size=cfg['bs'], verbose=0)
        # Gráfico de pérdida
        loss = hist.history.get('loss', [])
        plt.figure()
        plt.plot(loss, label='Train Loss')
        plt.title(f"{name} Loss ep={cfg['epochs']} bs={cfg['bs']}")
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(o_images, f"{name}_hist_{cfg['units']}_{cfg['drop']}.png"))
        plt.close()

        pred = model.predict(Xte, verbose=0).flatten()
        mse = mean_squared_error(Yte, pred)
        mae = mean_absolute_error(Yte, pred)
        if mse < best['mse']:
            best = {'cfg':cfg, 'mse':mse, 'mae':mae, 'pred':pred}
    # Plot mejor vs real\    
    pred = best['pred']
    real = Yte
    pred_real = sc.inverse_transform(pred.reshape(-1,1)).flatten()
    real_val = sc.inverse_transform(real.reshape(-1,1)).flatten()
    plt.figure()
    plt.plot(real_val, marker='o', label='Real')
    plt.plot(pred_real, marker='x', label='Predicción')
    plt.title(f"{name} best u{best['cfg']['units']} d{best['cfg']['drop']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(o_images, f"{name}_best.png"))
    plt.close()

    summary[name] = best

# Resumen
if __name__ == '__main__':
    print('Resumen LSTM:')
    for name, res in summary.items():
        cfg = res['cfg']
        print(f"{name}: units={cfg['units']}, drop={cfg['drop']}, mse={res['mse']:.4f}, mae={res['mae']:.4f}")

