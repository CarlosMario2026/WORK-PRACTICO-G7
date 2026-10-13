 PADCM - Precision Agriculture Drone Crop Monitoring

Descripción

PADCM es una aplicación de escritorio desarrollada en Python para el monitoreo y gestión optimizada de flotas de drones agrícolas autónomos. 

El sistema modela la navegación entre parcelas como una Cadena de Márkov en Tiempo Discreto (DTMC) e integra el algoritmo de PageRanK junto con índices de salud vegetal (NDVI). Esto permite identificar lotes agrícolas estructuralmente críticos que presentan estrés hídrico o fitosanitario y generar alertas de inspección prioritaria.

 Estructura del Proyecto

* "WORKSHOP-P3.py": Código fuente principal que despliega la interfaz gráfica (GUI) en Tkinter y ejecuta los modelos estocásticos.
* "datageneratorWS3.py": Script independiente encargado de construir el archivo de datos inicial.
* "dataset_drone_flights.json" Conjunto de datos en formato JSON con la telemetría de vuelos y valores NDVI por campo.
* "README.md": Documentación y guía del proyecto.

---

Requisitos e Instalación

Prerrequisitos
* Python 3.8 o superior.


---

INSTRUCCIONES DE USO:

1) Abre VS Code

2)Abre la carpeta del proyecto en VS Code donde se encuentran los archivos WORKSHOP-P3.py y DatageneratorWS3.py.py.

3)Abre la terminal integrada en VS Code (Ctrl + ~ o Terminal > New Terminal) y ejecuta el siguiente comando para instalar las librerías necesarias:
pip install matplotlib numpy pandas

4)Ejecuta el código del archivo generador.py para crear el archivo JSON de datos llamado dataset_drone_flights.json (solo es necesario si el archivo no existe en la carpeta).

5)Ejecuta el código del archivo WORKSHOP-P3.py. Se abrirá una ventana interactiva con las pestañas nombradas según cada tarea del proyecto.

6) Hacer clic en la pestaña que desees visualizar (Task 1, Task 2, Task 3, Task 4 o Panel D - Dashboard).

