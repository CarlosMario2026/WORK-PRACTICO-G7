#GEN JSON

def generate_synthetic_dataset(filename=JSON_FILENAME):
    fields = [f"Field_{chr(65+i)}" for i in range(10)]  #10 campos Field_A a Field_J
    
    edges = []
    edge_id_counter = 1
    
    for u in fields:
        #Cada campo se conecta aleatoriamente con 2 a 4 destinos
        num_targets = random.randint(2, 4)
        targets = random.sample([f for f in fields if f != u], num_targets)
        
        for v in targets:
            #Se asigna mayor volumen de vuelos a los campos A y B para modelar 'Hubs'
            if u in ["Field_A", "Field_B"]:
                flight_count = random.randint(50, 120)
            elif v in ["Field_A", "Field_B"]:
                flight_count = random.randint(30, 80)
            else:
                flight_count = random.randint(5, 25)
                
            ndvi = round(random.uniform(0.25, 0.85), 3) #indice de salud vegetativa(NDVI)
            battery = round(random.uniform(35.0, 95.0), 1) #Batería promedio durante el vuelo
            
            edges.append({
                "edge_id": f"e{edge_id_counter:03d}",
                "from_field": u,
                "to_field": v,
                "flight_count": flight_count,
                "avg_ndvi_index": ndvi,
                "avg_battery_pct": battery
            })
            edge_id_counter += 1
            
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=4)


if not os.path.exists(JSON_FILENAME):
    generate_synthetic_dataset()