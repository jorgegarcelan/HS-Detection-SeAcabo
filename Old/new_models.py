# ==============================================
# VERSION 1 POR JORGE GARCELÁN GÓMEZ - 12/12/23
# ==============================================

import uuid

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import re



def init(embedding_name, embedding_size):
    # Genera un UUID
    run_id = str(uuid.uuid4())  
    print(f"ID de ejecución: {run_id}")
    print(f"Embedding Name: {embedding_name}")
    print(f"Embedding Size: {embedding_size}")

    return run_id


def load_data():
    # Leer el archivo y realizar transformaciones en los datos
    df = pd.read_csv("data/seacabo_2023.csv")
    #df2 = pd.read_csv("data/seacabo_2023.csv")
    return df



def process_data(df):

    # filter df to only tweets from Aug
    cutoff = "2023-08-25" # asamblea

    # Ensure the 'date' column is in datetime format
    df = df[df["timestamp"] >= cutoff]

    # Initialize stemmer
    ##stemmer = SnowballStemmer('spanish')
    
    # Define function to remove stopwords, punctuation, and apply stemming
    def remove_spanish_stopwords(text):

        # Eliminar menciones a usuarios (palabras que comienzan con @)
        text = re.sub(r'@\w+', '', text)

        # Eliminar enlaces (todo lo que comienza con http o https)
        text = re.sub(r'http\S+|https\S+', '', text)

        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Remove stopwords
        spanish_stopwords = set(stopwords.words('spanish'))
        spanish_stopwords.remove("no")  # Retain "no" as it provides negative context
        words = word_tokenize(text)
        filtered_words = [word.lower() for word in words if word.lower() not in spanish_stopwords]
        
        # Apply stemming (MIRAR)
        ##stemmed_words = [stemmer.stem(word) for word in filtered_words]
        
        ##return ' '.join(stemmed_words)  
        return ' '.join(filtered_words)

    df['full_text_processed'] = df['text'].apply(remove_spanish_stopwords)

    return df


def embedding_data(df, embedding_name, embedding_size, run_id):
    if embedding_name == "fasttext":
        # Create a FastText model
        sentences = df['full_text_processed'].str.split().tolist()
        embedding_size = embedding_size  # you can adjust this value as needed
        model = FastText(sentences, vector_size=embedding_size, window=15, min_count=1, workers=8, sg=1, epochs=10)

        # Convert text data into FastText embeddings
        def text_to_vector(text):
            words = text.split()
            vector = np.mean([model.wv[word] for word in words if word in model.wv.index_to_key], axis=0)
            return vector

        df['full_text_filtered'] = df['full_text_processed'].apply(text_to_vector)


    elif embedding_name == "word2vec":
        # Create a FastText model
        sentences = df['full_text_processed'].str.split().tolist()
        embedding_size = embedding_size  # you can adjust this value as needed
        model = Word2Vec(sentences, vector_size=embedding_size, window=15, min_count=1, workers=8, sg=1, epochs=10)

        # Convert text data into FastText embeddings
        def text_to_vector(text):
            words = text.split()
            vector = np.mean([model.wv[word] for word in words if word in model.wv.index_to_key], axis=0)
            return vector

        df['full_text_filtered'] = df['full_text_processed'].apply(text_to_vector)


    elif embedding_name == "bow":
        # Creating the bag of Word Model
        model = CountVectorizer(max_features=embedding_size, ngram_range=(1, 5))
        model.fit()

    elif embedding_name == "tfidf":
        # Creating the TF-IDF model
        model = TfidfVectorizer(max_features=embedding_size, ngram_range=(1, 5))
        model.fit() # esto hay que hacerlo


    ## SAVE MODEL when running best model -> future (https://radimrehurek.com/gensim/models/fasttext.html)

    dump(model, f"new_embeddings/embedding_{run_id}.pkl")


    return


archivo_excel = "new_Training.xlsx"
def log_data(run_id, embedding_name, embedding_size, exec_time, timestamp):

    # Si existe, lee el contenido actual
    df = pd.read_excel(archivo_excel, engine='openpyxl')
    

    # Crea nueva fila con los datos de la ejecucción
    nueva_fila = {
            'Run_ID': run_id,
            'Embedding_Name': embedding_name,
            'Embedding_Size': embedding_size,
            'Time (s)': exec_time,
            'Timestamp': timestamp
        }
    
    # Agrega una nueva fila al DataFrame existente
    df = df._append(nueva_fila, ignore_index=True)

    print(nueva_fila)

    # Guarda el archivo de excel con la nueva fila
    df.to_excel(archivo_excel, index=False, engine='openpyxl')

    print(f"Datos guardados en {archivo_excel}")


# ============================ MAIN ============================

if __name__ == "__main__":
    embedding_name = ["tfidf"] #["fasttext", "word2vec", "bow", "tfidf"]
    embedding_size = [100, 500, 1000]

    for name in embedding_name:
        for size in embedding_size:
            start_time = time.time()
            timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

            run_id = init(name, size)

            # Cargamos datos
            df = load_data()

            # Preprocesado de datos
            df = process_data(df)
            
            # Embeddings
            embedding_data(df, name, size, run_id)

            end_time = time.time()
            exec_time = end_time - start_time

            # Create Log:
            log_data(run_id, name, size, exec_time, timestamp)