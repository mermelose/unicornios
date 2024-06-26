import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as plt

# Cargar datos
dfa = pd.read_csv("Base de Datos.csv")
dfb = pd.read_csv('Relacion_de_Precios_Modificado.csv')


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

#print('rmse_e:', rmse_e)
#print('rmse_p:', rmse_p)

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

#print(X_prb)
# Realizar predicciones manualmente
y_manual_pred = manual_predict(X_prb).flatten()

y_manual_pred = scaler_y.inverse_transform(y_manual_pred.reshape(-1, 1)).flatten()
y_pred_p = scaler_y.inverse_transform(y_pred_p.reshape(-1, 1)).flatten()

y_pred_p = y_pred_p.T
#print(y_pred_p)
#print(y_manual_pred)
#print(y)
score = red.score(X_prb, y_prb)

#print("El R2 es",score)

#LISTA DE MUSEOS QUE PODEMOS INGRESAR

from tkinter import *
import tkinter as tk

dptos ={
            "Amazonas": 1,
            "Ancash": 2,
            "Apurimac": 3,
            "Arequipa": 4,
            "Ayacucho": 5,
            "Cajamarca": 6,
            "Cusco": 8,
            "Huanuco": 10,
            "Ica": 11,
            "La Libertad": 13,
            "Lambayeque": 14,
            "Lima": 15,
            "Moquegua": 18,
            "Puno": 21,
            "Tumbes": 24
        }
def predecir_datos():
    # Obtener los valores ingresados en los campos de entrada
    museo= museo_entry.get()
    dpto = dptos.get(dpto_entry.get())
    general = general_entry.get()
    preferencial = preferencial_entry.get()
    menores = menores_entry.get()
    mes=mes_entry.get()
    museos = datos["NOM_MUSEO"].unique()
    lista=[0]*len(museos)
    
    for idx,i in enumerate(museos):
        if i==museo:
            lista[idx]=1
        else:
            lista[idx]=0
    lista.append(dpto)
    lista.append(int(general))
    lista.append(int(preferencial))
    lista.append(int(menores))
    lista.append(int(mes))
    #print(lista)
    #ingresas tu vector de datos como X_ventana
    X_ventana = lista
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
    #print(y_ventana)
    if y_ventana<0:
        y_ventana=0
    total_text.delete(0, tk.END)
    total_text.insert(tk.END, np.round(y_ventana,0).tolist())
    



# ----------------------------------Programa principal

ventana = Tk()


#Título de la ventana
ventana.title("Registro de Clientes")
#Tamaño de la ventana
ventana.geometry("600x400")

# Crear los campos de entrada
museo_label = Label(ventana, text='Nombre del museo:') # Etiqueta muestra texto
museo_label.pack()
museo_entry = Entry(ventana)  # ingreso de texto
museo_entry.pack()

dpto_label = Label(ventana, text='Nombre del Departamento:')
dpto_label.pack()
dpto_entry = Entry(ventana)
dpto_entry.pack()

general_label = Label(ventana, text='Precio General:')
general_label.pack()
general_entry = Entry(ventana)
general_entry.pack()

preferencial_label = Label(ventana, text='Precio Preferencial:')
preferencial_label.pack()
preferencial_entry = Entry(ventana)
preferencial_entry.pack()

menores_label = Label(ventana, text='Precio para menores:')
menores_label.pack()
menores_entry = Entry(ventana)
menores_entry.pack()

mes_label = Label(ventana, text='Número del mes:')
mes_label.pack()
mes_entry = Entry(ventana)
mes_entry.pack()

total_label=Label(ventana,text='Número de asistentes:')
total_label.pack()
total_text = Entry(ventana)
total_text.pack()
# Crear el botón de guardar
guardar_button = Button(ventana, text='Predecir', command=predecir_datos)
guardar_button.pack()

# Iniciar el ciclo de la ventana
ventana.mainloop()
