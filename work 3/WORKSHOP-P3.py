from dataclasses import fields
import io
import json
import os
import random
import sys
import time
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

JSON_FILENAME = "dataset_drone_flights.json"

#Task 1 - DTMC Construction & State Classification
class DroneDTMC:
    
    def __init__(self, edges): #Modela la red de vuelos como una Cadena de MÁRKOV en Tiempo Discreto (DTMC)
        self.edges = edges
        self.states = sorted(list(set([e["from_field"] for e in edges] + [e["to_field"] for e in edges])))
        self.n = len(self.states)
        
        #Mapeo bidireccional entre nombres de campos e índices matriciales
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        
        #Matriz de adyacencia ponderada por el número de vuelos
        self.adj_matrix = np.zeros((self.n, self.n))
        for e in edges:
            u_idx = self.state_to_idx[e["from_field"]]
            v_idx = self.state_to_idx[e["to_field"]]
            self.adj_matrix[u_idx, v_idx] += e["flight_count"]
            
        #Matriz de transición estocastica M (normalización por filas)
        self.M = np.zeros((self.n, self.n))
        for i in range(self.n):
            row_sum = np.sum(self.adj_matrix[i, :])
            if row_sum > 0:
                self.M[i, :] = self.adj_matrix[i, :] / row_sum
            else:
                self.M[i, :] = 1.0 / self.n  # Distribución uniforme si no hay salidas

    def classify_states(self):
        #Clasifica los campos según su nivel de tráfico respecto a la mediana del sistema.
        out_degrees = np.sum(self.adj_matrix, axis=1)
        in_degrees = np.sum(self.adj_matrix, axis=0)
        total_activity = out_degrees + in_degrees
        
        threshold = np.median(total_activity)
        classification = {}
        for i, s in enumerate(self.states):
            if total_activity[i] >= threshold:
                classification[s] = "Recurrent (Hub Field)"
            else:
                classification[s] = "Transient (Peripheral Field)"
        return classification

#Task 2 - Stationary Distribution

def compute_stationary_distribution(M, max_iter=1000, tol=1e-12):
    #Calculo de la probabilidad límite pi por iteración de potencias
    n = M.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_next = np.dot(pi, M)
        if np.linalg.norm(pi_next - pi, ord=1) < tol:
            break
        pi = pi_next
    return pi / np.sum(pi)

#Task 3 - S3 - Standard PageRank 

def compute_pagerank(M, d=0.85, max_iter=1000, tol=1e-12):
    #Calcula el ranking de importancia estructural de los campos
    n = M.shape[0]
    pr = np.ones(n) / n
    teleport = np.ones(n) / n
    
    for _ in range(max_iter):
        pr_next = d * np.dot(pr, M) + (1 - d) * teleport
        if np.linalg.norm(pr_next - pr, ord=1) < tol:
            break
        pr = pr_next
    return pr / np.sum(pr)



#Task 4 - S7 - Lazy Random Walk Simulation 

def lazy_random_walk(M, steps=50000, self_loop_prob=0.5):
    #Simulacion con 50% de probabilidad de permanencia mediante Monte Carlo.
    n = M.shape[0]
    M_lazy = self_loop_prob * np.eye(n) + (1.0 - self_loop_prob) * M
    
    current_state = random.randint(0, n - 1)
    visit_counts = np.zeros(n)
    
    for _ in range(steps):
        visit_counts[current_state] += 1
        probs = M_lazy[current_state, :]
        current_state = np.random.choice(n, p=probs)
        
    empirical_pi = visit_counts / steps
    theoretical_pi = compute_stationary_distribution(M_lazy)
    return empirical_pi, theoretical_pi


#Vistas Pestañas 
def crear_area_texto(parent):
    #Crea una consola interna con desplazamiento para mostrar los resultados en pantalla
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    txt = tk.Text(frame, wrap="word", font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
    sy = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=sy.set)
    sy.pack(side="right", fill="y")
    txt.pack(side="left", fill="both", expand=True)
    return txt

#Pestaña 1- Visualización de la Matriz M y clasificación de estados.
def cargar_tab1_dtmc(tab, dtmc):
    txt = crear_area_texto(tab)
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=== TASK 1: DTMC CONSTRUCTION & STATE CLASSIFICATION ===")
    print(f"Total de Estados (Campos): {dtmc.n}\n")
    print("Matriz de Transición Estocástica M (Primeros 5x5):")
    df_M = pd.DataFrame(dtmc.M, index=dtmc.states, columns=dtmc.states)
    print(df_M.iloc[:5, :5].round(4).to_string())

    print("\n--- Clasificación de Estados ---")
    classification = dtmc.classify_states()
    for state, cls in classification.items():
        print(f"• {state}: {cls}")

    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())

#Pestaña 2- Comparativa entre distribución estacionaria y volumen de tráfico
def cargar_tab2_stationary(tab, dtmc, pi):
    txt = crear_area_texto(tab)
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=== TASK 2: STATIONARY DISTRIBUTION (pi) ===")
    print("Porcentaje de tiempo/atención a largo plazo por campo:\n")
    
    raw_counts = np.sum(dtmc.adj_matrix, axis=1) + np.sum(dtmc.adj_matrix, axis=0)
    raw_shares = raw_counts / np.sum(raw_counts)

    df_comp = pd.DataFrame({
        "Campo": dtmc.states,
        "Estacionario pi": np.round(pi, 4),
        "Volumen Bruto Share": np.round(raw_shares, 4)
    }).sort_values(by="Estacionario pi", ascending=False)

    print(df_comp.to_string(index=False))

    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())

#Pestaña 3: Ranking PageRank y detección de lotes prioritarios con NDVI bajo
def cargar_tab3_pagerank(tab, dtmc, pr, edges_df):
    txt = crear_area_texto(tab)
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=== TASK 3: STANDARD PAGERANK (S3) ===")
    ndvi_per_field = edges_df.groupby("from_field")["avg_ndvi_index"].mean().to_dict()
    raw_counts = np.sum(dtmc.adj_matrix, axis=1) + np.sum(dtmc.adj_matrix, axis=0)
    
    df_pr = pd.DataFrame({
        "Campo": dtmc.states,
        "PageRank": pr,
        "Raw Flights": raw_counts,
        "Avg NDVI": [round(ndvi_per_field.get(s, 0.5), 3) for s in dtmc.states]
    })
    
    df_pr["Rank PageRank"] = df_pr["PageRank"].rank(ascending=False).astype(int)
    df_pr["Rank Raw"] = df_pr["Raw Flights"].rank(ascending=False).astype(int)
    df_pr = df_pr.sort_values("Rank PageRank")

    print(df_pr[["Campo", "Rank PageRank", "Rank Raw", "PageRank", "Raw Flights", "Avg NDVI"]].to_string(index=False))
    
    print("\n--- Alerta de Campos Sub-Monitoreados ---")
    for _, r in df_pr.iterrows():
        if r["Rank PageRank"] < r["Rank Raw"] and r["Avg NDVI"] < 0.55:
            print(f"¡ALERTA! El campo '{r['Campo']}' tiene alto PageRank (importancia estructural), "
                  f"pero bajo volumen bruto y NDVI bajo ({r['Avg NDVI']}). Requiere inspección prioritaria.")

    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())

#Pestaña 4- Resultados de la simulación lazy walk
def cargar_tab4_lazywalk(tab, dtmc):
    txt = crear_area_texto(tab)
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=== TASK 4: LAZY RANDOM WALK SIMULATION (S7) ===")
    steps = 50000
    emp_pi, theo_pi = lazy_random_walk(dtmc.M, steps=steps, self_loop_prob=0.5)

    print(f"Simulación ejecutada sobre {steps:,} pasos (Self-loop prob = 0.5):\n")
    df_walk = pd.DataFrame({
        "Campo": dtmc.states,
        "Empírico (Visitas)": np.round(emp_pi, 5),
        "Teórico (pi_lazy)": np.round(theo_pi, 5),
        "Error Absoluto": np.round(np.abs(emp_pi - theo_pi), 5)
    })
    print(df_walk.to_string(index=False))

    sys.stdout = sys.__stdout__
    txt.insert("1.0", buffer.getvalue())

#Pestaña 5 -Panel D Dashboard de PageRank vs Vol Bruto con NDVI Overlay
def cargar_tab6_dashboard(tab, dtmc, pr, edges_df):
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=90)

    ndvi_per_field = edges_df.groupby("from_field")["avg_ndvi_index"].mean().to_dict()
    fields = dtmc.states
    x = np.arange(len(fields))
    width = 0.35

    # Barras de PageRank
    ax1.bar(x - width/2, pr, width, label='PageRank (Importancia)', color='#2980b9')
    ax1.set_ylabel('Score de PageRank', color='#2980b9', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fields, rotation=45, ha='right')

    # Línea de NDVI superpuesta
    ax2 = ax1.twinx()
    ndvis = [ndvi_per_field.get(s, 0.5) for s in fields]
    ax2.plot(x, ndvis, color='#27ae60', marker='o', linewidth=2.5, label='Promedio NDVI')
    ax2.set_ylabel('Índice Salud Vegetal (NDVI)', color='#27ae60', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    ax2.axhline(0.5, color='#e74c3c', linestyle='--', alpha=0.7, label='Umbral Estrés (0.50)')

    plt.title("Panel D: PageRank vs Raw Volume per Field with NDVI Overlay", fontsize=11, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

#Main APP
def main():
    if not os.path.exists(JSON_FILENAME):
        print(f"Error: No se encontró '{JSON_FILENAME}'. Ejecuta primero tu script generador.")
        sys.exit(1)

    with open(JSON_FILENAME, "r", encoding="utf-8") as f:
        edges = json.load(f)
        
    edges_df = pd.DataFrame(edges)
    dtmc = DroneDTMC(edges)
    pi = compute_stationary_distribution(dtmc.M)
    pr = compute_pagerank(dtmc.M, d=0.85)

    root = tk.Tk()
    root.title("PADCM - Precision Agriculture Drone Crop Monitoring")
    root.geometry("1150x700")

    # Mantiene la ventana visible al inicio
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.focus_force()

    # Cierre correcto app
    def on_closing():
        plt.close('all')
        root.quit()
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Contenedor de pestañas
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    t1 = ttk.Frame(notebook)
    notebook.add(t1, text="Task 1 - DTMC")
    cargar_tab1_dtmc(t1, dtmc)

    t2 = ttk.Frame(notebook)
    notebook.add(t2, text="Task 2 - Stationary Dist")
    cargar_tab2_stationary(t2, dtmc, pi)

    t3 = ttk.Frame(notebook)
    notebook.add(t3, text="Task 3 - PageRank")
    cargar_tab3_pagerank(t3, dtmc, pr, edges_df)

    t4 = ttk.Frame(notebook)
    notebook.add(t4, text="Task 4 - Lazy Walk")
    cargar_tab4_lazywalk(t4, dtmc)

    t5 = ttk.Frame(notebook)
    notebook.add(t5, text="Panel D - Dashboard")
    cargar_tab6_dashboard(t5, dtmc, pr, edges_df)

    root.mainloop()


if __name__ == "__main__":
    main()