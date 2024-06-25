import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt

# Cargar datos
dfa = pd.read_csv('Base de Datos.csv')
dfb = pd.read_csv('Relación_de_Precios_Modificado.csv')

# Unir datos
df = pd.merge(dfa, dfb, on="NOM_MUSEO")

# Seleccionar columnas relevantes
datos = df[['COD_DPTO', 'NOM_MUSEO', 'COD_MES', 'COD_TIPO', 'General', 'Preferencial', '<18 años', 'TOTAL_PAGANTES']]

# Codificar columnas categóricas
categorical_cols = ['COD_DPTO', 'NOM_MUSEO', 'COD_MES', 'COD_TIPO']
one_hot_encoder = OneHotEncoder()
X_categorical = one_hot_encoder.fit_transform(datos[categorical_cols]).toarray()

# Combinar columnas categóricas codificadas con las numéricas
X_numerical = datos[['General', 'Preferencial', '<18 años']].values
X = np.hstack((X_categorical, X_numerical))

# Variable objetivo
y = datos['TOTAL_PAGANTES'].values

# Dividir los datos en conjunto de entrenamiento y prueba
X_ent, X_prb, y_ent, y_prb = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

# Escalar los datos
scaler_X = StandardScaler().fit(X_ent)
scaler_y = StandardScaler().fit(y_ent.reshape(-1, 1))

X_ent = scaler_X.transform(X_ent)
X_prb = scaler_X.transform(X_prb)
y_ent = scaler_y.transform(y_ent.reshape(-1, 1)).ravel()
y_prb = scaler_y.transform(y_prb.reshape(-1, 1)).ravel()

# Crear la red neuronal
red = MLPRegressor(hidden_layer_sizes=(10,),
                   activation='relu',
                   solver='adam',
                   alpha=0.001,
                   batch_size='auto',
                   learning_rate_init=0.01,
                   max_iter=500,
                   random_state=42,
                   verbose=True)

# Ajustar el modelo
red.fit(X_ent, y_ent)

# Realizar predicciones
y_pred_e = red.predict(X_ent)
y_pred_p = red.predict(X_prb)

# Evaluar el desempeño
rmse_e = mean_squared_error(y_true=y_ent, y_pred=y_pred_e, squared=False)
rmse_p = mean_squared_error(y_true=y_prb, y_pred=y_pred_p, squared=False)

print('rmse_e:', rmse_e)
print('rmse_p:', rmse_p)

# Gráficos
plt.figure(figsize=(10,5))
plt.scatter(y_ent, y_pred_e, label='Entrenamiento', color='darkred')
plt.scatter(y_prb, y_pred_p, label='Prueba', color='darkblue')
#plt.plot([min(y_ent), max(y_ent)], [min(y_ent), max(y_ent)], 'k--', lw=3)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.grid()
plt.legend()
plt.show()
