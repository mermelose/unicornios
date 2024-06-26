import pyodbc
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
server = '172.23.1.140'
bd = 'Unicornio'
user = 'tp_28'
clave = '281202'
try:
    conexion = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL server};SERVER='+server+';DATABASE='+bd+';UID='+user+';PWD='+clave) # Cadena de conexión a la base de datos
    print("Conexión exitosa")
    cursor=conexion.cursor()
except:
       print("Error al intentar conectarse")
   
def ejecutar_query(query,con):
  df_resultado=pd.read_sql(query, con)    
  return df_resultado
ventana = tk.Tk()
ventana.title("Búsqueda de museos y precio por departamento")
ventana.configure(background="#CCEEFF")
ventana.geometry("1000x500")

def buscar_dpto():
    dpto = dpto_entry.get()
    try:
      cursor = conexion.cursor()
      query="select distinct s.NOM_MUSEO as Nombre_de_Museo, s.General as Precio_General,s.menoredad as Precio_Menores,s.Preferencial as Precio_Preferencial from Sitiooo s where s.NOM_DPTO='"+dpto+"'"
      df=pd.read_sql(query, conexion)
    except Exception as e:
          messagebox.showerror("Error", f"Error al BUSCAR dpto: {str(e)}")
    # Limpiar el Treeview
    for item in listam.get_children():
        listam.delete(item)        # Configurar las columnas del Treeview
    listam["columns"] = list(df.columns)
    listam["show"] = "headings"
    for column in df.columns:
        listam.heading(column, text=column)
        listam.column(column, anchor="center")

    # Insertar los datos del DataFrame filtrado en el Treeview
    for index, row in df.iterrows():
        listam.insert("", "end", values=list(row))



#--------Elementos de la ventana
dpto_label = Label(ventana, text='Ingrese el departamento:',background="#CCEEFF") # Etiqueta muestra texto
dpto_label.pack()
dpto_entry = Entry(ventana)  # ingreso de texto
dpto_entry.pack()

listam = ttk.Treeview(ventana)
listam.pack(expand=True, fill='both')

buscar_button = Button(ventana, text='BUSCAR', command=buscar_dpto, background="#E0E6F8")
buscar_button.pack()

ventana.mainloop()
