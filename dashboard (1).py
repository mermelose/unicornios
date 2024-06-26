import pyodbc
import pandas as pd
from matplotlib import pyplot as plt
server = '172.23.1.140'
bd = 'Unicornio'
user = 'TP_28'
clave = '281202'
try:
    conexion = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL server};SERVER='+server+';DATABASE='+bd+';UID='+user+';PWD='+clave) # Cadena de conexión a la base de datos
    print("Conexión exitosa")
    cursor=conexion.cursor()
except:
       print("Error al intentar conectarse")

def QueryEjecutar(query, con):
    df_resultado=pd.read_sql(query, con)
    return df_resultado

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.tight_layout(pad=5.0)

query1="SELECT TOP 5 NOM_DPTO, sum(TOTAL) as TotalVisitantes FROM Sitiooo GROUP BY NOM_DPTO ORDER BY TotalVisitantes desc"
df1=QueryEjecutar(query1, conexion)
print(df1)

query2="SELECT cod_mes ,NOM_MES, sum(TOTAL) AS Visitantes FROM Sitiooo GROUP BY cod_mes,NOM_MES ORDER BY COD_MES ASC"
df2=QueryEjecutar(query2, conexion)
print(df2)

query3="SELECT  TOP 3 NOM_MES, SUM(EST_PAGANTES) AS VisitantesEST FROM Sitiooo GROUP BY NOM_MES ORDER BY VisitantesEST DESC"
query4="Select sum(ADU_PAGANTES) as ADULTOS,sum(EST_PAGANTES) as ESTUDIANTES,sum(NIN_PAGANTES) as NIÑOS from Sitiooo"
df3=QueryEjecutar(query3, conexion)
df4=QueryEjecutar(query4, conexion)
df4t=df4.transpose()
print(df3)
print(df4)


plt.subplot(2,2,1)
plt.barh(df1['NOM_DPTO'],df1['TotalVisitantes'],color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title("5 Ciudades con mayor asistencia")
plt.subplot(2,2,2)
plt.plot(df2['cod_mes'], df2['Visitantes'],color='#008000')
plt.xticks(range(1,13,1))
plt.title("Asistencia por meses")
plt.subplot(2,2,3)
plt.bar(df3['NOM_MES'],df3['VisitantesEST'],color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title("3 meses con mayor asistencia de estudiantes")
plt.subplot(2,2,4)
plt.pie(df4t[0], labels=df4.columns,autopct='%1.1f%%')
plt.title("Porcentajes de tipos de asistentes")
