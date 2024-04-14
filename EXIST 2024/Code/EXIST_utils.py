import uuid
import pandas as pd
from sklearn.utils import resample
import torch
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

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


def resampling(X_train, y_train, balance):

    if balance == "None":
        return X_train, y_train

    # Balance classes by resampling
    if balance == "downsampling":
        # Convertir X_train y y_train a DataFrame para facilitar el manejo
        df_train_temp = pd.concat([X_train, y_train], axis=1)

        # Aplicar downsampling
        df_train_downsampled = downsample(df_train_temp)

        # Actualizar X_train y y_train después del downsampling
        X_train = df_train_downsampled.drop(['label'], axis=1)
        y_train = df_train_downsampled['label']

    # Ahora, dependiendo de la técnica de balance que quieras aplicar:
    if balance == "upsampling":
        # Convertir X_train y y_train a DataFrame para facilitar el manejo
        df_train_temp = pd.concat([X_train, y_train], axis=1)
        
        # Aplicar upsampling
        df_train_upsampled = upsample(df_train_temp)
        
        # Actualizar X_train y y_train después del upsampling
        X_train = df_train_upsampled.drop(['label'], axis=1)
        y_train = df_train_upsampled['label']

    elif balance == "smote":
        # Aplicar SMOTE
        sm = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
        X_train = X_train_resampled
        y_train = y_train_resampled

    elif balance == "adasyn":
        # Aplicar ADASYN
        ada = ADASYN(random_state=42)
        X_train_resampled, y_train_resampled = ada.fit_resample(X_train, y_train)
        X_train = X_train_resampled
        y_train = y_train_resampled

    # Imprimir el soporte de clases después del remuestreo
    unique, counts = np.unique(y_train, return_counts=True)

    # Crear un diccionario a partir de valores únicos y sus conteos
    value_counts = dict(zip(unique, counts))

    # Imprimir el conteo de valores
    print(value_counts)

    return X_train, y_train


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