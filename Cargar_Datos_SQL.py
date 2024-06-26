import pyodbc
import pandas as pd

bd = pd.read_csv("C:/Users\cz.zhoul\Downloads\Base de Datos.csv")
#print(bd)

precios = pd.read_csv("C:/Users\cz.zhoul\Downloads\Relación de Precios.csv")
#print(precios)

df = pd.merge(bd, precios, how= "inner", on="NOM_MUSEO")
#print(df)

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


for index, row in df.iterrows(): # insertar tabla Gender
   
     cursor.execute("INSERT INTO dbo.Sitiooo (ANIO,COD_DPTO,NOM_DPTO,NOM_MUSEO,COD_MES,NOM_MES,COD_TIPO,ADU_BOLESPPAGANTES,EST_BOLESPPAGANTES,NIN_BOLESPPAGANTES,MIL_BOLESPPAGANTES,ADM_BOLESPPAGANTES,ADU_PAGANTES,EST_PAGANTES,NIN_PAGANTES, TOTAL_PAGANTES,ADU_BOLESPNOPAGANTES,EST_BOLESPNOPAGANTES,NIN_BOLESPNOPAGANTES,MIL_BOLESPNOPAGANTES, ADM_BOLESPNOPAGANTES,ADU_NOPAGANTES,EST_NOPAGANTES,	NIN_NOPAGANTES,	TOTAL_NOPAGANTES,TOTAL,TIPO_cat_cod, General,Preferencial,menoredad) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", 
                    row.ANIO, row.COD_DPTO, row.NOM_DPTO, row.NOM_MUSEO, row.COD_MES, row.NOM_MES, row.COD_TIPO, row.ADU_BOLESPPAGANTES, row.EST_BOLESPPAGANTES, row.NIN_BOLESPPAGANTES, row.MIL_BOLESPPAGANTES, row.ADM_BOLESPPAGANTES,
row.ADU_PAGANTES,
row.EST_PAGANTES,
row.NIN_PAGANTES,
row.TOTAL_PAGANTES,
row.ADU_BOLESPNOPAGANTES,
row.EST_BOLESPNOPAGANTES,
row.NIN_BOLESPNOPAGANTES,
row.MIL_BOLESPNOPAGANTES,
row.ADM_BOLESPNOPAGANTES,
row.ADU_NOPAGANTES,
row.EST_NOPAGANTES,
row.NIN_NOPAGANTES,
row.TOTAL_NOPAGANTES,
row.TOTAL,
row.TIPO_cat_cod,
row.General,
row.Preferencial,
row.menoredad
)

     conexion.commit()
