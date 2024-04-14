import pandas as pd

def log_data(run_id, archivo_excel, data, embedding_name, embedding_size, model, cv, best_params, report, acc_global, std_global, exec_time, timestamp, type_id, balance):

    # Si existe, lee el contenido actual
    df = pd.read_excel(archivo_excel, engine='openpyxl')
    

    # Crea nueva fila con los datos de la ejecucción
    nueva_fila = {
            'Run_ID': run_id,
            'Data': data,
            'Type': type_id,
            'Balance': balance,
            'Embedding_Name': embedding_name,
            'Embedding_Size': embedding_size,
            'Model': model,
            'Cross_Validation': cv,
            'Best_params': str(best_params),
            'Accuracy_Global': acc_global,
            'Std_Global': std_global,
            #'Report': [report],
            'Time (s)': exec_time,
            'Timestamp': timestamp
        }
    
    # Formatea classification_report para juntarlo con nueva_fila
    format_report = {}
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                format_report[f"{key}_{sub_key}"] = sub_value
        else:
            format_report[key] = value
    
    # Junta los dos diccionarios
    nueva_fila.update(format_report)

    # Agrega una nueva fila al DataFrame existente
    df = df._append(nueva_fila, ignore_index=True)

    print(nueva_fila)

    # Guarda el archivo de excel con la nueva fila
    df.to_excel(archivo_excel, index=False, engine='openpyxl')

    print(f"Datos guardados en {archivo_excel}")
