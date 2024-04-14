from utils import *

import re
import string
import pandas as pd
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import emoji


def filter_by_type(df, type_id, balance):
    #["analisis_general", "contenido_negativo", "insultos"]
    if type_id == "analisis_general":
        # Define the specific labels to keep
        #etiquetas = ["Comentario Positivo", "Comentario Negativo", "Comentario Neutro"]
        etiquetas = ["Comentario Positivo", "Comentario Negativo"]
        
        df['Análisis General'] = df['Análisis General'].where(df['Análisis General'].isin(etiquetas))


        # Remove NAs
        df = df.dropna(subset=['Análisis General'])
        

        # Factorize the 'Análisis General' column
        labels, labels_names = pd.factorize(df['Análisis General'])

        # 'labels' now contains the numeric representation of your original labels
        # 'label_names' contains the unique values from your original column in the order they were encoded

        # Replace the original column with the numeric labels
        df['label'] = labels

        # If you want to keep a record of the mapping from the original labels to the numeric labels
        label_mapping = dict(zip(labels_names, range(len(labels_names))))
        #print("Label Mapping:", label_mapping)

    if type_id == "contenido_negativo":

        # Filtrar el DataFrame para seleccionar solo los "Comentario Negativo"
        df = df.loc[df['Análisis General'] == 'Comentario Negativo']

        # Define the specific labels to keep
        etiquetas = ["Desprestigiar Víctima", "Desprestigiar Acto", "Insultos", "Desprestigiar Deportista Autora"]
        df['Contenido Negativo'] = df['Contenido Negativo'].where(df['Contenido Negativo'].isin(etiquetas))

        # Remove NAs
        df = df.dropna(subset=['Contenido Negativo'])
        

        # Factorize the 'Análisis General' column
        labels, labels_names = pd.factorize(df['Contenido Negativo'])

        # 'labels' now contains the numeric representation of your original labels
        # 'label_names' contains the unique values from your original column in the order they were encoded

        # Replace the original column with the numeric labels
        df['label'] = labels

        # If you want to keep a record of the mapping from the original labels to the numeric labels
        label_mapping = dict(zip(labels_names, range(len(labels_names))))
        #print("Label Mapping:", label_mapping)


    if type_id == "insultos":

        # Filtrar el DataFrame para seleccionar solo los "Comentario Negativo"
        df = df.loc[df['Análisis General'] == 'Comentario Negativo']

        # Define the specific labels to keep
        etiquetas = ["Deseo de Dañar", "Genéricos", "Sexistas/misóginos", ""]

        # Replace labels that are not in the list with "Genéricos"
        df['Insultos'] = df['Insultos'].where(df['Insultos'].isin(etiquetas), other="Genéricos")

        # Remove NAs
        df = df.dropna(subset=['Insultos'])
        

        # Factorize the 'Insultos' column
        labels, labels_names = pd.factorize(df['Insultos'])

        # 'labels' now contains the numeric representation of your original labels
        # 'label_names' contains the unique values from your original column in the order they were encoded

        # Replace the original column with the numeric labels
        df['label'] = labels

        # If you want to keep a record of the mapping from the original labels to the numeric labels
        label_mapping = dict(zip(labels_names, range(len(labels_names))))
        #print("Label Mapping:", label_mapping)


    # Balance classes by resampling
    if balance == "downsampling":
        df = downsample(df)



    # Contar el soporte de cada etiqueta
    soporte_etiquetas = df['label'].value_counts()

    # Imprimir el soporte para cada etiqueta
    print("\nSoporte de etiquetas con nombres originales:")
    for nombre_etiqueta, codigo in label_mapping.items():
        print(f"{nombre_etiqueta}: {soporte_etiquetas[codigo]}")


    return df, labels_names


def load_insults_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        insults_lexicon = {line.strip().lower() for line in file if line.strip()}
    return insults_lexicon

def load_misogyny_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        misogyny_lexicon = {line.strip().lower() for line in file if line.strip()}
    return misogyny_lexicon



def process_data(df):

    # Initialize stemmer
    ##stemmer = SnowballStemmer('spanish')
        

    # Normalize "view_count"
    scaler = StandardScaler()
    df['view_count_scaled'] = scaler.fit_transform(df[['view_count']])

    # User Mentions
    def count_user_mentions(mentions):
        if pd.isna(mentions) or mentions == "":
            return 0
        else:
            return len(mentions.split(';'))
        
    df['mention_count'] = df['user_mentions'].apply(count_user_mentions)

    # Length Tweet
    df['tweet_length'] = df['full_text'].str.len()

    # Num Adjetives
    def count_adjectives(text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        return sum(1 for word, tag in pos_tags if tag.startswith('JJ'))
    df['num_adjectives'] = df['full_text'].apply(count_adjectives)
    
    # Define function to remove stopwords, punctuation, and apply stemming
    def normalize_content(text):

        # Convertir a minúsculas
        text = text.lower()

        # Eliminar emojis
        #text = emoji.replace_emoji(text, replace='')
        
        # Replace emojis with the meaning
        text = emoji.demojize(text, language="es", delimiters=("_", "_")) 

        # Eliminar menciones a usuarios y URLs
        text = re.sub(r'@\w+|http\S+|https\S+', '', text)

        # Eliminar números y marcas de puntuación
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])

        # Tokenizer
        words = word_tokenize(text)
        
        # Remove stopwords
        spanish_stopwords = set(stopwords.words('spanish'))
        spanish_stopwords.remove("no")  # Retain "no" as it provides negative context
        
        filtered_words = [word.lower() for word in words if word.lower() not in spanish_stopwords]

        # Reducir palabras elongadas y caracteres repetidos
        filtered_words = [re.sub(r'(.)\1+', r'\1', palabra) for palabra in filtered_words]

    
        # Apply stemming (MIRAR)
        ##stemmed_words = [stemmer.stem(word) for word in filtered_words]
        
        ##return ' '.join(stemmed_words)  
        return ' '.join(filtered_words)
    df['full_text_processed'] = df['full_text'].apply(normalize_content)

    def add_tokens(text):
        # words list
        words = text.split()

        # lexicon
        misogyny_list = load_misogyny_lexicon("lexicons_train_misogyny_lexicon.txt")
        insults_list = load_insults_lexicon("lexicons_train_insults_lexicon.txt")

        # Reemplaza insultos por un token especial
        processed_words = ['[INSULT]' if word.lower() in insults_list or word.lower() in misogyny_list else word for word in words]
        return ' '.join(processed_words)
    
    df['full_text_processed'] = df['full_text_processed'].apply(add_tokens)
    
    # Eliminar filas donde 'full_text_processed' es una cadena vacía
    df = df[df['full_text_processed'] != ""]

    return df