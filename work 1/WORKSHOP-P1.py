import io
import json
import math
import random
import sys
import time
import tkinter as tk
from tkinter import ttk
import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from datetime import datetime, timedelta

# Data generation function to create a synthetic dataset for crop monitoring using drones#

def generate_crop_monitoring_dataset(num_records=10000, output_filename="dataset_drones.json"):
    fields = [f"FIELD-{i:02d}" for i in range(1, 11)]  # FIELD-01 a FIELD-10
    drones = [f"DRONE-{i:02d}" for i in range(1, 6)]    # DRONE-01 a DRONE-05

    start_date = datetime(2026, 1, 1, 8, 0, 0)
    dataset = []

    for i in range(1, num_records + 1):
        # 1. reading_id único
        reading_id = f"READ-{i:06d}"

        # 2. timestamp (simula lecturas tomadas a lo largo de varios días)
        random_minutes = random.randint(0, 180 * 24 * 60) # Últimos 6 meses
        timestamp = (start_date + timedelta(minutes=random_minutes)).isoformat()

        # 3. field_id y drone_id
        field_id = random.choice(fields)
        drone_id = random.choice(drones)

        # 4. ndvi_index (rango de -1.0 a 1.0, redondeado a 2 decimales)
        ndvi_index = round(random.uniform(-1.0, 1.0), 2)

        # 5. soil_moisture (porcentaje de 0.0 a 100.0, redondeado a 1 decimal)
        soil_moisture = round(random.uniform(5.0, 50.0), 1)

        # 6. alert_flag (boolean: True si el ndvi_index es menor a 0.2 o la humedad es muy baja)
        alert_flag = True if (ndvi_index < 0.2 or soil_moisture < 15.0) else False

        record = {
            "reading_id": reading_id,
            "timestamp": timestamp,
            "field_id": field_id,
            "ndvi_index": ndvi_index,
            "soil_moisture": soil_moisture,
            "drone_id": drone_id,
            "alert_flag": alert_flag
        }
        dataset.append(record)

    # Guardar en un archivo JSON
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"¡Éxito! Se generó el dataset con {num_records} registros en '{output_filename}'.")

# Ejecutar la función para generar 10,000 registros
if __name__ == "__main__":
    generate_crop_monitoring_dataset(num_records=10000)


#TASK 1: Big Data Context Calculations
#Estimate readings/sec for a fleet of drones covering many fields and storage at 10x farm-network growth

def calculate_big_data_metrics(farms, drones_per_farm, hz_rate, flight_hours_per_day, payload_bytes=200):
    total_drones = farms * drones_per_farm
    readings_per_sec = total_drones * hz_rate
    bytes_per_sec = readings_per_sec * payload_bytes

    # Cálculos de volumen
    daily_readings = readings_per_sec * (flight_hours_per_day * 3600)
    daily_gb = (daily_readings * payload_bytes) / (1024 ** 3)
    annual_tb = (daily_gb * 365) / 1024

    return {
        "total_drones": total_drones,
        "readings_per_sec": readings_per_sec,
        "kb_per_sec": bytes_per_sec / 1024,
        "daily_readings": daily_readings,
        "daily_gb": round(daily_gb, 2),
        "annual_tb": round(annual_tb, 2)
    }


def print_scenario(title, metrics):
    print(f"\n--- {title} ---")
    print(f"• Drones activos simultáneos: {metrics['total_drones']:,}")
    print(f"• Ingesta en tiempo real (Readings/sec): {metrics['readings_per_sec']:,} req/s")
    print(f"• Ancho de banda de entrada: {metrics['kb_per_sec']:.2f} KB/s")
    print(f"• Lecturas por día: {metrics['daily_readings']:,}")
    print(f"• Almacenamiento diario: {metrics['daily_gb']} GB/día")
    print(f"• Almacenamiento anual estimado: {metrics['annual_tb']} TB/año")

#TASK 2 Randomized Algorithm
#Reservoir-sample readings to estimate average ndvi_index per field, 
#or randomized QuickSort fields by health index for a priority-inspection list; compare vs exact averages.

def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

#TASK 3 Universal Hashing
#Hash field_id for a fast lookup index used by the drone routing system; measure collisions
json_filename = "dataset_drones.json"

if not os.path.exists(json_filename):
    print(f"⚠️ El archivo '{json_filename}' no se encuentra en el directorio actual.")
    print("Asegúrate de subir 'dataset_drones.json' al panel de archivos de Google Colab (icono de carpeta a la izquierda).")
else:
    with open(json_filename, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"✅ Archivo '{json_filename}' cargado exitosamente. Total registros: {len(records):,}")


# 2. UNIVERSAL HASHING  
class UniversalHashIndex:
    def __init__(self, table_size=100000, prime=1000003):
        self.m = table_size
        self.p = prime
        # Selección aleatoria de coeficientes 'a' y 'b' para la familia Universal Hash
        self.a = random.randint(1, self.p - 1)
        self.b = random.randint(0, self.p - 1)
        self.table = [[] for _ in range(self.m)]
        self.collisions = 0

    def _hash(self, key_str):
        """Calcula el índice hash a partir de una cadena (reading_id)."""
        k = sum(ord(c) * (31 ** i) for i, c in enumerate(key_str))
        return ((self.a * k + self.b) % self.p) % self.m

    def insert(self, record):
        """Inserta un registro usando reading_id como clave única."""
        key = record["reading_id"]
        idx = self._hash(key)
        
        # Si el bucket ya contiene algún registro, se detecta una colisión
        if len(self.table[idx]) > 0:
            self.collisions += 1
            
        self.table[idx].append(record)

    def search(self, reading_id):
        """Busca un registro por su ID único en tiempo O(1)."""
        idx = self._hash(reading_id)
        bucket = self.table[idx]
        for record in bucket:
            if record["reading_id"] == reading_id:
                return record
        return None
    
#TASK 4 Hash Table Operations
#Build a hash table keyed by drone_id bucketing readings per drone; compare chaining vs power-of-two-choices.

# 1. TABLA HASH CON CHAINING TRADICIONAL
class ChainingHashTable:
    def __init__(self, num_buckets=20, prime=1000003):
        self.m = num_buckets
        self.p = prime
        self.a = random.randint(1, self.p - 1)
        self.b = random.randint(0, self.p - 1)
        self.buckets = [[] for _ in range(self.m)]

    def _hash(self, key_str):
        k = sum(ord(c) * (31 ** i) for i, c in enumerate(key_str))
        return ((self.a * k + self.b) % self.p) % self.m

    def insert(self, drone_id, record):
        idx = self._hash(drone_id)
        self.buckets[idx].append(record)

    def get_bucket_sizes(self):
        return [len(b) for b in self.buckets]


# 2. TABLA HASH CON POWER-OF-TWO-CHOICES
class PowerOfTwoHashTable:
    def __init__(self, num_buckets=20, prime=1000003):
        self.m = num_buckets
        self.p = prime

        # Dos funciones hash independientes
        self.a1, self.b1 = random.randint(1, self.p - 1), random.randint(0, self.p - 1)
        self.a2, self.b2 = random.randint(1, self.p - 1), random.randint(0, self.p - 1)

        self.buckets = [[] for _ in range(self.m)]

    def _hash1(self, key_str):
        k = sum(ord(c) * (31 ** i) for i, c in enumerate(key_str))
        return ((self.a1 * k + self.b1) % self.p) % self.m

    def _hash2(self, key_str):
        k = sum(ord(c) * (37 ** i) for i, c in enumerate(key_str))
        return ((self.a2 * k + self.b2) % self.p) % self.m

    def insert(self, drone_id, record):
        idx1 = self._hash1(drone_id)
        idx2 = self._hash2(drone_id)

        # Se elige el bucket que tenga MENOS elementos actualmente
        if len(self.buckets[idx1]) <= len(self.buckets[idx2]):
            self.buckets[idx1].append(record)
        else:
            self.buckets[idx2].append(record)

    def get_bucket_sizes(self):
        return [len(b) for b in self.buckets]

    
# ==========================================
# 2. FUNCIONES DE GENERACIÓN DE CONTENIDO PARA CADA PESTAÑA
# ==========================================

def crear_area_texto(parent):
    """Crea un área de texto estilizada tipo consola para mostrar reportes."""
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    txt = tk.Text(frame, wrap="word", font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
    sy = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=sy.set)
    
    sy.pack(side="right", fill="y")
    txt.pack(side="left", fill="both", expand=True)
    return txt


# --- PESTAÑA 1: TABLA DATASET ---
def cargar_tab1_dataset(tab, filepath):
    frame = ttk.Frame(tab)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    sy = ttk.Scrollbar(frame, orient="vertical")
    sx = ttk.Scrollbar(frame, orient="horizontal")
    tree = ttk.Treeview(frame, yscrollcommand=sy.set, xscrollcommand=sx.set)

    sy.config(command=tree.yview)
    sx.config(command=tree.xview)
    sy.pack(side="right", fill="y")
    sx.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if data:
            cols = list(data[0].keys())
            tree["columns"] = cols
            tree["show"] = "headings"
            for c in cols:
                tree.heading(c, text=c.replace("_", " ").title())
                tree.column(c, width=110, anchor="center")
            for record in data[:1000]:  # Carga rápida de 1000 filas
                tree.insert("", "end", values=[record[k] for k in cols])
    except Exception as e:
        print(f"Error cargando dataset: {e}")


# --- PESTAÑA 2: TASK 1 ---
def cargar_tab2_task1(tab):
    txt = crear_area_texto(tab)
    
    # Redirigimos la salida de consola al widget de texto
    buffer = io.StringIO()
    sys.stdout = buffer

    # 1. PARÁMETROS BASE
    base_farms = 50
    drones_per_farm = 2
    hz_rate = 5         # 5 lecturas por segundo por dron
    flight_hours = 8    # 8 horas de vuelo al día

    # 2. CÁLCULO ESCALA ACTUAL Y ESCALA 10X
    current_metrics = calculate_big_data_metrics(base_farms, drones_per_farm, hz_rate, flight_hours)
    growth_10x_metrics = calculate_big_data_metrics(base_farms * 10, drones_per_farm, hz_rate, flight_hours)

    # Mostrar resultados de la estimación en consola integrada
    print("=== TASK 1: BIG DATA CONTEXT CALCULATIONS ===")
    print_scenario("Escenario Actual (50 Granjas)", current_metrics)
    print_scenario("Escenario 10x Growth (500 Granjas)", growth_10x_metrics)

    # Restaurar consola del sistema y escribir texto en Tkinter
    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())


# --- PESTAÑA 3: TASK 2 ---
def cargar_tab3_task2(tab, records):
    txt = crear_area_texto(tab)
    
    buffer = io.StringIO()
    sys.stdout = buffer
    
    print("=== TASK 2: RESERVOIR SAMPLING STREAMING ANALYSIS ===\n")
    k = 10
    print(f"Generando una muestra aleatoria imparcial de k={k} lecturas desde el flujo de 10,000 registros...\n")
    
    t0 = time.time()
    sample = reservoir_sample(records, k)
    t_exec = time.time() - t0
    
    print(f"Tiempo de Ejecución: {t_exec:.6f} segundos\n")
    print("--- Registros Seleccionados en la Muestra ---")
    sample_df = pd.DataFrame(sample)
    print(sample_df.to_string(index=False))
    
    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())


# --- PESTAÑA 4: TASK 3 ---
def cargar_tab4_task3(tab, records):
    txt = crear_area_texto(tab)
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=== TASK 3: UNIVERSAL HASHING & FAST LOOKUP INDEX ===\n")

    # Asegurar conversión si viniera en otro formato
    if hasattr(records, "to_dict"):
        records = records.to_dict(orient="records")

    table_size = 100000
    index = UniversalHashIndex(table_size=table_size)

    t0 = time.time()
    for r in records:
        if isinstance(r, dict):
            index.insert(r)
    t_build = time.time() - t0

    print(f"Total de registros insertados: {len(records):,}")
    print(f"Tamaño de la tabla Hash (buckets 'm'): {table_size:,}")
    print(f"Tiempo de construcción del índice: {t_build:.5f} segundos")
    print(f"Número total de colisiones detectadas: {index.collisions:,}")

    if records and isinstance(records[0], dict):
        target_id = records[22]["reading_id"]
        print(f"\n--- BÚSQUEDA EN TIEMPO REAL POR ID ÚNICO ---")
        print(f"ID buscado: '{target_id}'")

        t0 = time.time()
        result = index.search(target_id)
        t_search = (time.time() - t0) * 1000

        print(f"Tiempo de consulta con Hash Index: {t_search:.4f} ms\n")
        if result:
            print("Ejemplo de lectura recuperada del índice:")
            print(json.dumps(result, indent=4, default=str))

    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())

# --- PESTAÑA 5: TASK 4 ---
def cargar_tab5_task4(tab, records):
    txt = crear_area_texto(tab)
    
    buffer = io.StringIO()
    sys.stdout = buffer
    
    json_filename = "dataset_drones.json"
    df = pd.read_json(json_filename)
    records = df.to_dict(orient="records")

    # Fijamos 20 buckets para visualizar claramente el balanceo de carga
    num_buckets = 20

    chaining_table = ChainingHashTable(num_buckets=num_buckets)
    power2_table = PowerOfTwoHashTable(num_buckets=num_buckets)

    # Insertar registros en Chaining
    start_t = time.time()
    for r in records:
        chaining_table.insert(r["drone_id"], r)
    chaining_time = time.time() - start_t

    # Insertar registros en Power-of-Two-Choices
    start_t = time.time()
    for r in records:
        power2_table.insert(r["drone_id"], r)
    power2_time = time.time() - start_t

    # Obtener tamaños de buckets
    chaining_sizes = chaining_table.get_bucket_sizes()
    power2_sizes = power2_table.get_bucket_sizes()

    
    

    def calculate_std_dev(lst):
        mean = sum(lst) / len(lst)
        variance = sum((x - mean) ** 2 for x in lst) / len(lst)
        return math.sqrt(variance)

    summary_df = pd.DataFrame({
    "Estrategia": ["Standard Chaining", "Power-of-Two-Choices"],
    "Tiempo de Inserción (s)": [round(chaining_time, 5), round(power2_time, 5)],
    "Carga Máxima (Max Bucket Size)": [max(chaining_sizes), max(power2_sizes)],
    "Carga Mínima (Min Bucket Size)": [min(chaining_sizes), min(power2_sizes)],
    "Desviación Estándar (Balanceo)": [round(calculate_std_dev(chaining_sizes), 2),
                                      round(calculate_std_dev(power2_sizes), 2)]
})

    print("=== TASK 4: HASH TABLE OPERATIONS (DRONE_ID BUCKETING) ===")
    print(f"Total de lecturas procesadas: {len(records):,}")
    print(f"Número de buckets en la tabla Hash: {num_buckets}\n")
    print(summary_df.to_string(index=False))

    print("\n--- DISTRIBUCIÓN DE ELEMENTOS POR BUCKET (PRIMEROS 10 BUCKETS) ---")
    buckets_comp = pd.DataFrame({
        "Bucket Index": list(range(10)),
        "Chaining Size": chaining_sizes[:10],
        "Power-of-Two Size": power2_sizes[:10]
    })
    print(buckets_comp.to_string(index=False))
    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())

# --- PESTAÑA 6: TASK 6 (DASHBOARD) ---
def cargar_tab6_dashboard(tab, df, records):
    fig = plt.figure(figsize=(11, 5.5), dpi=90)

    # Panel A
    ax1 = plt.subplot(2, 2, 1)
    num_buckets = 10
    ct = ChainingHashTable(num_buckets=num_buckets)
    pt = PowerOfTwoHashTable(num_buckets=num_buckets)
    for r in records:
        ct.insert(r["drone_id"], r)
        pt.insert(r["drone_id"], r)

    b = np.arange(num_buckets)
    w = 0.35
    ax1.bar(b - w/2, ct.get_bucket_sizes(), w, label='Chaining', color='#e74c3c')
    ax1.bar(b + w/2, pt.get_bucket_sizes(), w, label='Power of 2', color='#2ecc71')
    ax1.set_title("Panel A: Bucket-Load Distribution", fontsize=9, fontweight='bold')
    ax1.legend(fontsize=7)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Panel B
    ax2 = plt.subplot(2, 2, 2)
    sizes = [1000, 2500, 5000, 7500, 10000]
    res_t, det_t = [], []
    for s in sizes:
        sub = records[:s]
        t0 = time.time()
        _ = reservoir_sample(sub, 100)
        res_t.append(time.time() - t0)

        t0 = time.time()
        _ = sorted(sub, key=lambda x: x["reading_id"])[:100]
        det_t.append(time.time() - t0)

    ax2.plot(sizes, res_t, marker='o', color='#3498db', label='Reservoir O(N)')
    ax2.plot(sizes, det_t, marker='s', linestyle='--', color='#e67e22', label='Deterministic O(N log N)')
    ax2.set_title("Panel B: Runtime Comparison", fontsize=9, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Panel C
    ax3 = plt.subplot(2, 1, 2)
    f_stats = df.groupby("field_id")["ndvi_index"].mean().reset_index().head(10)
    m_data = f_stats["ndvi_index"].values.reshape(2, 5)
    f_labels = f_stats["field_id"].values.reshape(2, 5)
    ann = np.array([[f"{fl}\nNDVI: {val:.3f}" for fl, val in zip(rf, rv)] for rf, rv in zip(f_labels, m_data)])

    sns.heatmap(m_data, annot=ann, fmt="", cmap="RdYlGn", cbar=True, ax=ax3, linewidths=1, cbar_kws={'label': 'NDVI'})
    ax3.set_title("Panel C: Field Health Heatmap (Red = Alert)", fontsize=9, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ==========================================
# 3. INTERFAZ PRINCIPAL CON 6 PESTAÑAS
# ==========================================

def main():
    json_path = "dataset_drones.json"
    try:
        df = pd.read_json(json_path)
        records = df.to_dict(orient="records")
    except Exception as e:
        print(f"Error cargando {json_path}: {e}")
        return

    root = tk.Tk()
    root.title("AgriData System - Precision Agriculture Tasks")
    root.geometry("1150x700")

    # Contenedor de las 6 pestañas
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # 1. Pestaña Dataset
    t1 = ttk.Frame(notebook)
    notebook.add(t1, text="📋 Dataset")
    cargar_tab1_dataset(t1, json_path)

    # 2. Pestaña Task 1
    t2 = ttk.Frame(notebook)
    notebook.add(t2, text="📊 Task 1")
    cargar_tab2_task1(t2)

    # 3. Pestaña Task 2
    t3 = ttk.Frame(notebook)
    notebook.add(t3, text="🎲 Task 2")
    cargar_tab3_task2(t3, records)

    # 4. Pestaña Task 3
    t4 = ttk.Frame(notebook)
    notebook.add(t4, text="🔍 Task 3")
    cargar_tab4_task3(t4, df)

    # 5. Pestaña Task 4
    t5 = ttk.Frame(notebook)
    notebook.add(t5, text="⚡ Task 4")
    cargar_tab5_task4(t5, records)

    # 6. Pestaña Dashboard (Task 6)
    t6 = ttk.Frame(notebook)
    notebook.add(t6, text="📈 Dashboard (Task 6)")
    cargar_tab6_dashboard(t6, df, records)

    root.mainloop()

if __name__ == "__main__":
    main()