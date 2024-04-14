import pandas as pd

def load_data(data):
    # Leer el archivo de datos
    df = pd.read_csv(data)
    
    return df