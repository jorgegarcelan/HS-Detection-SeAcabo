import uuid
import pandas as pd
from sklearn.utils import resample
import torch
import numpy as np

# ======================= FUNCTIONS =======================

def init(embedding_name, embedding_size, model, cv):
    # Genera un UUID
    run_id = str(uuid.uuid4())  
    print(f"ID de ejecución: {run_id}")

    print(f"Embedding Name: {embedding_name}")
    print(f"Embedding Size: {embedding_size}")
    print(f"Model: {model}")
    print(f"Cross-Validation: {cv}")
    
    return run_id

def downsample(df):
    # Contar el número de instancias por clase
    class_counts = df['label'].value_counts()
    # Encontrar el número de instancias de la clase menos representada
    min_class_count = class_counts.min()

    # Crear un nuevo DataFrame vacío
    df_balanced = pd.DataFrame()

    # Iterar sobre cada clase y reducir al número de instancias de la clase menos representada
    for label in df['label'].unique():
        df_class = df[df['label'] == label]
        df_class_downsampled = df_class.sample(min_class_count)
        df_balanced = pd.concat([df_balanced, df_class_downsampled], axis=0)

    # Mezclar las filas para evitar cualquier sesgo
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
    return df_balanced

def upsample(df):
    # Contar el número de instancias por clase
    class_counts = df['label'].value_counts()
    # Encontrar el número de instancias de la clase más representada
    max_class_count = class_counts.max()
    print(f"{max_class_count=}")

    # Crear un nuevo DataFrame vacío
    df_balanced = pd.DataFrame()

    # Iterar sobre cada clase y aumentar al número de instancias de la clase más representada
    for label in df['label'].unique():
        df_class = df[df['label'] == label]
        df_class_upsampled = resample(df_class, 
                                      replace=True,     # Muestra con reemplazo
                                      n_samples=max_class_count,    # Para igualar la clase mayoritaria
                                      random_state=123) # Reproducible
        df_balanced = pd.concat([df_balanced, df_class_upsampled], axis=0)

    # Mezclar las filas para evitar cualquier sesgo
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
    return df_balanced


def get_roberta_embeddings(texts, tokenizer, model, device='cpu'):

    # Mover el modelo al dispositivo adecuado
    model.to(device)
    
    embeddings = []
    for text in texts:
        # Tokenizar el texto y agregar los tokens especiales
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        
        # Obtener los embeddings del modelo
        with torch.no_grad():
            outputs = model(**encoded_input)
        
        # Usar los embeddings del último estado oculto
        last_hidden_state = outputs.last_hidden_state
        
        # Promediar los embeddings del token a lo largo de la dimensión de la secuencia para obtener un único vector de embedding por texto
        mean_embedding = torch.mean(last_hidden_state, dim=1)
        embeddings.append(mean_embedding.cpu().numpy())
        
    return np.vstack(embeddings)