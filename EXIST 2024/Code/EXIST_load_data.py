import json
import pandas as pd

def load_data(data):
    # Carga los datos desde un archivo JSON
    with open(data, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Preparar una lista para almacenar los datos aplanados
    flat_data = []

    # Iterar sobre cada tweet anidado
    for tweet_id, tweet_info in data.items():
        # Aquí 'tweet_info' contiene todos los datos del tweet individual
        flat_data.append(tweet_info)

    # Convertir la lista de datos aplanados a un DataFrame
    df = pd.DataFrame(flat_data)

    return df