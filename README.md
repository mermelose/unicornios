# Bìlǔ kǎogǔ yízhǐ cānguān rénshù yùcè móxíng (Modelo Predictivo Sobre Visitas a Lugares Arqueológicos en el Perú)

# Descripción del Proyecto
El turismo es uno de los sectores más relevantes para la economía peruana, donde cada rincón del país ofrece una historia única y valiosa.
Sin embargo, a pesar del significativo valor cultural de algunos sitios arqueológicos, no reciben la misma atención que otras zonas turísticas.
Además, la pandemia de la COVID-19 ha tenido un impacto considerable en los ingresos por turismo, destacando la necesidad crítica de analizar las variables que afectan las visitas a estos espacios culturales.

Este repositorio contiene un proyecto enfocado en desarrollar un modelo de predicción para estimar la afluencia de visitantes y optimizar los beneficios de los centros culturales en el Perú.
El objetivo principal es incrementar el número de visitas a estos sitios históricos y culturales, promoviendo así su preservación y valorización económica.


## Tabla de Contenidos
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Tabla de Contenidos](#tabla-de-contenidos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Datos](#datos)
- [Autores](#autores)


## Instalación
Para instalar y ejecutar este proyecto localmente, sigue estos pasos:

1. Clona este repositorio:
    ```sh
    git clone (https://github.com/mermelose/unicornios.git)
    ```

2. Navega al directorio del proyecto:
    ```sh
    cd unicornios
    ```

3. Instala las dependencias necesarias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

Para utilizar el modelo predictivo, sigue estos pasos:

1. Asegúrate de tener los archivos de datos necesarios en el directorio `data/`.
2. Ejecuta el script de predicción con el modelo

   ```sh
   redNeuronal_con_ventana.py
   ```

## Datos
El conjunto de datos incluye registros históricos de asistencia, horarios de eventos, datos meteorológicos y otros factores relevantes. Los datos se almacenan en el directorio `data/` e incluyen los siguientes archivos:

- `Datos_Visitas_Arqueologicas`: Registros históricos de asistencia.
- `Relación_de_Precios.csv`:  Precios de las entradas a los museos.

## Construido con

* Pandas: Biblioteca de análisis de datos utilizada para manipular conjuntos de datos
* NumPy: Biblioteca para el cálculo numérico y manejo de arreglos.
* sklearn.neural_network: Módulo de scikit-learn que proporciona herramientas para la creación de redes neuronales.
* sklearn.model_selection: Módulo de scikit-learn utilizado para dividir los datos en conjuntos de entrenamiento y prueba.
* sklearn.metrics: Módulo de scikit-learn que ofrece herramientas para evaluar el rendimiento del modelo.
* sklearn.preprocessing: Módulo de scikit-learn utilizado para la normalización y codificación de características.
* Matplotlib: Biblioteca para la creación de gráficos y visualización de datos.
* tkinter: Biblioteca para la creación de interfaces gráficas de usuario en Python.
## Autores

* **Montalvo, Fabrizio**
* **Salazar, Paolo**
* **Zhou, Cynthia**


## Agradecimientos

* Agradecemos a todos aquellos cuyo código fue utilizado como inspiración y aporte para este proyecto.
