import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as plt

# Cargar datos
dfa = pd.read_csv("C:/Users/Fabox/Downloads/Base de Datos.csv")
dfb = pd.read_csv('C:/Users/Fabox/Downloads/Relacion_de_Precios_Modificado.csv')


# Unir datos
df = pd.merge(dfa, dfb, on="NOM_MUSEO")

# Seleccionar columnas relevantes
datos = df[['COD_DPTO', 'NOM_MUSEO', 'COD_MES', 'General', 'Preferencial', '<18 años', 'TOTAL_PAGANTES']]

# Codificar columnas categóricas
categorical_cols = ['NOM_MUSEO']
one_hot_encoder = OneHotEncoder()
X_categorical = one_hot_encoder.fit_transform(datos[categorical_cols]).toarray()

# Combinar columnas categóricas codificadas con las numéricas
X_numerical = datos[['COD_DPTO','General', 'Preferencial', '<18 años', 'COD_MES']].values

X = np.hstack((X_categorical, X_numerical))



# Variable objetivo
y = datos['TOTAL_PAGANTES'].values

# Dividir los datos en conjunto de entrenamiento y prueba
X_ent, X_prb, y_ent, y_prb = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

#xventana

#X_ventana = X_prb[0]


# Escalar los datos
scaler_X = StandardScaler().fit(X_ent)
scaler_y = StandardScaler().fit(y_ent.reshape(-1, 1))

X_ent = scaler_X.transform(X_ent)
X_prb = scaler_X.transform(X_prb)
y_ent = scaler_y.transform(y_ent.reshape(-1, 1)).ravel()
y_prb = scaler_y.transform(y_prb.reshape(-1, 1)).ravel()

# Crear la red neuronal
red = MLPRegressor(hidden_layer_sizes=(2,),
                   activation='logistic',
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

coefs = red.coefs_
intercepts = red.intercepts_

#print(coefs)
# Función para realizar predicciones manualmente
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def manual_predict(X):
    layer_input = X
    for coef, intercept in zip(coefs, intercepts):
        #print(coef,'coef')
        #print(intercept,'inter')
        layer_output = np.dot(layer_input, coef) + intercept
        layer_input = sigmoid(layer_output) # Aplicar función de activación ReLU 
    return layer_output

print(X_prb)
# Realizar predicciones manualmente
y_manual_pred = manual_predict(X_prb).flatten()

y_manual_pred = scaler_y.inverse_transform(y_manual_pred.reshape(-1, 1)).flatten()
y_pred_p = scaler_y.inverse_transform(y_pred_p.reshape(-1, 1)).flatten()

y_pred_p = y_pred_p.T
print(y_pred_p)
print(y_manual_pred)


score = red.score(X_prb, y_prb)
print("El R2 es",str(round(score*100,2)),"%")

#LISTA DE MUSEOS QUE PODEMOS INGRESAR
"""
museos = datos["NOM_MUSEO"].unique()

#ingresas tu vector de datos como X_ventana
X_ventana = []
#conviertes el vector a array
X_ventana = np.array(X_ventana)
#cambias las dimensiones de tu array para poder transformarlos y multiplicarlos por los coeficientes
X_ventana = X_ventana.reshape(1,-1)
X_ventana= scaler_X.transform(X_ventana)


#esto normaliza el vector

#aplicas la funcion de calculo de pagantes en base a los coeficiente de la red neuronal

y_ventana = manual_predict(X_ventana).flatten()
#y ventana seria el total de pagantes predichos segun tu vector
y_ventana = scaler_y.inverse_transform(y_ventana.reshape(-1, 1)).flatten()
"""



print("beta 0: ",coefs[1].tolist()[0][0])
for i in range(len(coefs[0].tolist())):
       
   print("beta",i+1,":",coefs[0].tolist()[i][0])
        
print("beta 0: ",coefs[1].tolist()[1][0])
for i in range(len(coefs[0].tolist())):
   
    print("beta",i+1,":",coefs[0].tolist()[i][1])
    
print(coefs[1].tolist())

print(coefs)
