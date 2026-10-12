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

# ==========================================
# GENERADOR DE DATASET (UNIDAD 2 - STREAMING)
# ==========================================

def generate_crop_monitoring_dataset_v2(num_records=10000, output_filename="dataset_drones_v2.json"):
    fields = [f"FIELD-{i:02d}" for i in range(1, 11)]  # FIELD-01 a FIELD-10
    drones = [f"DRONE-{i:02d}" for i in range(1, 6)]    # DRONE-01 a DRONE-05

    start_date = datetime(2026, 1, 1, 8, 0, 0)
    dataset = []

    for i in range(1, num_records + 1):
        reading_id = f"READ-{i:06d}"
        random_minutes = random.randint(0, 180 * 24 * 60)
        timestamp = (start_date + timedelta(minutes=random_minutes)).isoformat()

        field_id = random.choice(fields)
        drone_id = random.choice(drones)

        ndvi_index = round(random.uniform(-1.0, 1.0), 2)
        soil_moisture = round(random.uniform(5.0, 50.0), 1)

        alert_flag = True if (ndvi_index < 0.2 or soil_moisture < 15.0) else False

        # Campos de Unidad 2 (Streaming)
        stream_seq = i
        battery_pct = round(random.uniform(15.0, 100.0), 1)

        record = {
            "reading_id": reading_id,
            "timestamp": timestamp,
            "field_id": field_id,
            "ndvi_index": ndvi_index,
            "soil_moisture": soil_moisture,
            "drone_id": drone_id,
            "alert_flag": alert_flag,
            "stream_seq": stream_seq,
            "battery_pct": battery_pct
        }
        dataset.append(record)

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"¡Éxito! Dataset v2 generado con {num_records} registros en '{output_filename}'.")

# ==========================================
# IMPLEMENTACIÓN DE ALGORITMOS (TASKS 1 - 4)
# ==========================================

# --- TASK 1: STREAM MODEL & QUERY DESIGN ---
def process_task1_queries(records):
    df = pd.DataFrame(records)
    
    # 1. Standing Query: Promedio móvil NDVI por parcela (últimas 200 lecturas)
    # Simula el flujo usando las últimas 200 lecturas
    recent_200 = df.tail(200)
    rolling_ndvi = recent_200.groupby("field_id")["ndvi_index"].mean().round(3).to_dict()

    # 2. Ad-hoc Query: Conteo de parcelas únicas registradas en el día más reciente
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    max_date = df['date'].max()
    today_records = df[df['date'] == max_date]
    distinct_fields_today = today_records['field_id'].nunique()

    return rolling_ndvi, max_date, distinct_fields_today


# --- TASK 2: BLOOM FILTER ---
class BloomFilter:
    def __init__(self, expected_elements=100, false_positive_rate=0.05):
        self.n = expected_elements
        self.p = false_positive_rate
        self.m = int(- (self.n * math.log(self.p)) / (math.log(2) ** 2))
        self.k = int((self.m / self.n) * math.log(2))
        self.bit_array = [0] * self.m
        
        # Generar coeficientes hash aleatorios
        self.hash_seeds = [random.randint(1, 99999) for _ in range(self.k)]

    def _hashes(self, item_str):
        indices = []
        for seed in self.hash_seeds:
            h = sum((ord(char) * seed) for char in item_str) % self.m
            indices.append(h)
        return indices

    def add(self, item_str):
        for idx in self._hashes(item_str):
            self.bit_array[idx] = 1

    def contains(self, item_str):
        return all(self.bit_array[idx] == 1 for idx in self._hashes(item_str))

    def theoretical_fp_rate(self, inserted_count):
        return (1 - math.exp(-self.k * inserted_count / self.m)) ** self.k


# --- TASK 3: RESERVOIR SAMPLING (S3) ---
def reservoir_sample_stream(records, k):
    reservoir = []
    for i, record in enumerate(records):
        if i < k:
            reservoir.append(record)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = record
    return reservoir


# --- TASK 4: FLAJOLET-MARTIN (S6 - DISTINCT COUNT) ---
class FlajoletMartin:
    def __init__(self, num_hashes=10, num_buckets=1024):
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.a = [random.randint(1, 9999) for _ in range(num_hashes)]
        self.b = [random.randint(0, 9999) for _ in range(num_hashes)]
        self.max_tail = [0] * num_hashes

    def _hash(self, val_str, i):
        k = sum(ord(c) * (31 ** idx) for idx, c in enumerate(str(val_str)))
        return ((self.a[i] * k + self.b[i]) % 1000003) % self.num_buckets

    def _trailing_zeros(self, val):
        if val == 0:
            return 0
        s = bin(val)[2:]
        return len(s) - len(s.rstrip('0'))

    def add(self, item_str):
        for i in range(self.num_hashes):
            h_val = self._hash(item_str, i)
            t = self._trailing_zeros(h_val)
            if t > self.max_tail[i]:
                self.max_tail[i] = t

    def estimate(self):
        # 2^R / phi donde phi ≈ 0.77351
        phi = 0.77351
        estimates = [(2 ** R) / phi for R in self.max_tail]
        return int(np.median(estimates))


# ==========================================
# INTERFAZ GRÁFICA TKINTER CON PESTAÑAS
# ==========================================

def crear_area_texto(parent):
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    txt = tk.Text(frame, wrap="word", font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
    sy = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=sy.set)
    
    sy.pack(side="right", fill="y")
    txt.pack(side="left", fill="both", expand=True)
    return txt


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
                tree.column(c, width=100, anchor="center")
            for record in data[:1000]:
                tree.insert("", "end", values=[record[k] for k in cols])
    except Exception as e:
        print(f"Error cargando dataset: {e}")


def cargar_tab2_task1(tab, records):
    txt = crear_area_texto(tab)
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=== TASK 1: STREAM MODEL & QUERY DESIGN ===\n")
    rolling_ndvi, last_date, count_today = process_task1_queries(records)

    print("1. STANDING QUERY (Promedio Móvil NDVI - Últimas 200 Lecturas del Stream):")
    for field, val in sorted(rolling_ndvi.items()):
        print(f"   • {field}: NDVI Promedio = {val}")

    print(f"\n2. AD-HOC QUERY:")
    print(f"   • Fecha evaluada: {last_date}")
    print(f"   • Parcelas únicas monitoreadas hoy: {count_today} parcelas")

    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())


def cargar_tab3_task2(tab, records):
    txt = crear_area_texto(tab)
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=== TASK 2: BLOOM FILTER (REDUNDANT ALERT FILTERING) ===\n")
    
    bf = BloomFilter(expected_elements=10, false_positive_rate=0.05)
    
    # Identificar parcelas marcadas con alerta
    alert_fields = set(r["field_id"] for r in records if r["alert_flag"])
    all_fields = [f"FIELD-{i:02d}" for i in range(1, 11)]

    # Insertar parcelas con alerta en el Bloom Filter
    for f in alert_fields:
        bf.add(f)

    print(f"Tamaño del Bit Array (m): {bf.m} bits")
    print(f"Número de Funciones Hash (k): {bf.k}")
    print(f"Parcelas insertadas con alerta ({len(alert_fields)}): {sorted(list(alert_fields))}\n")

    # Evaluación de Falsos Positivos
    inserted_count = len(alert_fields)
    theoretical_fp = bf.theoretical_fp_rate(inserted_count)

    # Prueba empírica probando elementos no insertados (si los hay)
    non_alert_fields = [f for f in all_fields if f not in alert_fields]
    false_positives = 0
    if non_alert_fields:
        for f in non_alert_fields:
            if bf.contains(f):
                false_positives += 1
        empirical_fp = false_positives / len(non_alert_fields)
    else:
        empirical_fp = 0.0

    print(f"--- TASA DE FALSOS POSITIVOS ---")
    print(f"• Tasa Teórica de Falsos Positivos: {theoretical_fp * 100:.2f}%")
    print(f"• Tasa Empírica de Falsos Positivos: {empirical_fp * 100:.2f}% ({false_positives}/{len(non_alert_fields) if non_alert_fields else 1})")

    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())


def cargar_tab4_task3(tab, records):
    txt = crear_area_texto(tab)
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=== TASK 3: RESERVOIR SAMPLING (S3 - ESTIMACIÓN DE NDVI Y HUMEDAD) ===\n")

    # Valores reales
    df = pd.DataFrame(records)
    true_ndvi = df["ndvi_index"].mean()
    true_soil = df["soil_moisture"].mean()

    print(f"Valores Reales del Flujo Completo (10,000 registros):")
    print(f"• NDVI Promedio Real: {true_ndvi:.4f}")
    print(f"• Humedad Suelo Promedio Real: {true_soil:.4f}%\n")

    print("--- COMPARACIÓN EN 3 TAMAÑOS DE RESERVORIO (k) ---")
    reservoir_sizes = [100, 500, 1000]

    for k in reservoir_sizes:
        sample = reservoir_sample_stream(records, k)
        s_df = pd.DataFrame(sample)
        s_ndvi = s_df["ndvi_index"].mean()
        s_soil = s_df["soil_moisture"].mean()

        error_ndvi = abs(s_ndvi - true_ndvi)
        error_soil = abs(s_soil - true_soil)

        print(f"Tamaño k={k}:")
        print(f"  - Est. NDVI: {s_ndvi:.4f} (Error Absoluto: {error_ndvi:.4f})")
        print(f"  - Est. Humedad: {s_soil:.4f}% (Error Absoluto: {error_soil:.4f})")

    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())


def cargar_tab5_task4(tab, records):
    txt = crear_area_texto(tab)
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=== TASK 4: FLAJOLET-MARTIN (S6 - DISTINCT FIELD_ID COUNT) ===\n")

    fm = FlajoletMartin(num_hashes=10)
    for r in records:
        fm.add(r["field_id"])

    estimated_count = fm.estimate()
    actual_count = len(set(r["field_id"] for r in records))

    print(f"• Conteo Real de parcelas (field_id): {actual_count}")
    print(f"• Conteo Estimado con Flajolet-Martin (10 Hash Funcs): {estimated_count}")
    print(f"• Error Relativo: {abs(estimated_count - actual_count) / actual_count * 100:.2f}%")

    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())


def cargar_tab6_dashboard(tab, df, records):
    fig = plt.figure(figsize=(11, 5.5), dpi=90)

    # Panel A: Bloom Filter FP Rates
    ax1 = plt.subplot(2, 2, 1)
    bf = BloomFilter(expected_elements=10, false_positive_rate=0.05)
    alert_fields = set(r["field_id"] for r in records if r["alert_flag"])
    for f in alert_fields:
        bf.add(f)
    t_fp = bf.theoretical_fp_rate(len(alert_fields)) * 100
    all_fields = [f"FIELD-{i:02d}" for i in range(1, 11)]
    non_alert = [f for f in all_fields if f not in alert_fields]
    fps = sum(1 for f in non_alert if bf.contains(f))
    e_fp = (fps / len(non_alert) * 100) if non_alert else 0.0

    ax1.bar(["Teórica", "Empírica"], [t_fp, e_fp], color=['#3498db', '#e74c3c'], width=0.4)
    ax1.set_ylabel("Tasa (%)")
    ax1.set_title("Panel A: Bloom Filter False-Positive Rate", fontsize=9, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Panel B: Flajolet-Martin vs Real Count
    ax2 = plt.subplot(2, 2, 2)
    fm = FlajoletMartin(num_hashes=10)
    for r in records:
        fm.add(r["field_id"])
    est_count = fm.estimate()
    real_count = len(set(r["field_id"] for r in records))

    ax2.bar(["Real", "Flajolet-Martin"], [real_count, est_count], color=['#2ecc71', '#f39c12'], width=0.4)
    ax2.set_ylabel("Parcelas Únicas")
    ax2.set_title("Panel B: Distinct field_id Estimate vs True Count", fontsize=9, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Panel C: Live Field Health Table / Heatmap
    ax3 = plt.subplot(2, 1, 2)
    f_stats = df.groupby("field_id")["ndvi_index"].mean().reset_index().head(10)
    m_data = f_stats["ndvi_index"].values.reshape(2, 5)
    f_labels = f_stats["field_id"].values.reshape(2, 5)
    ann = np.array([[f"{fl}\nNDVI: {val:.3f}" for fl, val in zip(rf, rv)] for rf, rv in zip(f_labels, m_data)])

    sns.heatmap(m_data, annot=ann, fmt="", cmap="RdYlGn", cbar=True, ax=ax3, linewidths=1, cbar_kws={'label': 'NDVI'})
    ax3.set_title("Panel C: Live Field Health Heatmap (Red = Alert / Low NDVI)", fontsize=9, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ==========================================
# CÓDIGO PRINCIPAL
# ==========================================

def main():
    json_path = "dataset_drones_v2.json"
    
    # Generar dataset si no existe
    if not os.path.exists(json_path):
        generate_crop_monitoring_dataset_v2(num_records=10000, output_filename=json_path)

    try:
        df = pd.read_json(json_path)
        records = df.to_dict(orient="records")
    except Exception as e:
        print(f"Error cargando {json_path}: {e}")
        return

    root = tk.Tk()
    root.title("AgriData System - Workshop 2 (Data Streams & Precision Agriculture)")
    root.geometry("1150x700")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Pestañas
    t1 = ttk.Frame(notebook)
    notebook.add(t1, text="📋 Dataset v2")
    cargar_tab1_dataset(t1, json_path)

    t2 = ttk.Frame(notebook)
    notebook.add(t2, text="📊 Task 1")
    cargar_tab2_task1(t2, records)

    t3 = ttk.Frame(notebook)
    notebook.add(t3, text="🎯 Task 2 (Bloom Filter)")
    cargar_tab3_task2(t3, records)

    t4 = ttk.Frame(notebook)
    notebook.add(t4, text="🎲 Task 3 (Reservoir)")
    cargar_tab4_task3(t4, records)

    t5 = ttk.Frame(notebook)
    notebook.add(t5, text="🔢 Task 4 (Flajolet-Martin)")
    cargar_tab5_task4(t5, records)

    t6 = ttk.Frame(notebook)
    notebook.add(t6, text="📈 Dashboard (Task 6)")
    cargar_tab6_dashboard(t6, df, records)

    root.mainloop()

if __name__ == "__main__":
    main()