import collections
from EXIST_utils import *

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


def process_target(df, type_id, type_dataset, label_mapping):
    # Función para elegir la etiqueta más frecuente dentro de cada lista
    def most_frequent_label(labels_list):
        if labels_list:
            return collections.Counter(labels_list).most_common(1)[0][0]
        else:
            return None  # O manejar listas vacías según sea necesario
    """
    # Función para elegir la etiqueta más frecuente dentro de cada lista, "tie" if 3 yes and 3 no
    def most_frequent_label(labels_list):
        if labels_list:
            # Obtener las etiquetas más comunes y sus conteos
            common_labels = collections.Counter(labels_list).most_common()
            if len(common_labels) > 1 and common_labels[0][1] == common_labels[1][1]:
                # Hay un empate entre al menos las dos etiquetas más frecuentes
                return "Tie"
            else:
                # No hay empate, devolver la etiqueta más frecuente
                return common_labels[0][0]
        else:
            # Manejar listas vacías según sea necesario
            return None
    """
    

    #["task_1", "task_2", "task_3"]
    if type_id == "task_1":
        # Remove NAs
        df = df.dropna(subset=['labels_task1'])


        if type_dataset == "train":

            # Aplicar la función a cada fila y crear una nueva columna con la etiqueta más frecuente
            df['label'] = df['labels_task1'].apply(most_frequent_label)

            # Factorizar la columna de etiquetas más frecuentes
            labels, labels_names = pd.factorize(df['label'])
            df['label'] = labels

            # If you want to keep a record of the mapping from the original labels to the numeric labels
            label_mapping = dict(zip(labels_names, range(len(labels_names))))

            # Contar el soporte de cada etiqueta
            soporte_etiquetas = df['label'].value_counts()

            # Imprimir el soporte para cada etiqueta
            print("\nSoporte de etiquetas con nombres originales:")
            for nombre_etiqueta, codigo in label_mapping.items():
                print(f"{nombre_etiqueta}: {soporte_etiquetas[codigo]}")



            return df, labels_names, label_mapping

        if type_dataset == "dev":
            
            # Aplicar la función a cada fila y crear una nueva columna con la etiqueta más frecuente
            df['label'] = df['labels_task1'].apply(most_frequent_label)
            df['label'] = df['label'].map(label_mapping)

            # Contar el soporte de cada etiqueta
            soporte_etiquetas = df['label'].value_counts()

            # Imprimir el soporte para cada etiqueta
            print("\nSoporte de etiquetas con nombres originales:")
            for nombre_etiqueta, codigo in label_mapping.items():
                print(f"{nombre_etiqueta}: {soporte_etiquetas[codigo]}")

            return df
        




def load_insults_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        insults_lexicon = {line.strip().lower() for line in file if line.strip()}
    return insults_lexicon

def load_misogyny_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        misogyny_lexicon = {line.strip().lower() for line in file if line.strip()}
    return misogyny_lexicon



def process_data(df, type_id):

    # Initialize stemmer
    ##stemmer = SnowballStemmer('spanish')
        

    # Tweets in 'es':
    df = df.loc[df['lang'] == 'es']

    # Length Tweet
    df['tweet_length'] = df['tweet'].str.len()

    # Num Adjetives
    def count_adjectives(text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        return sum(1 for word, tag in pos_tags if tag.startswith('JJ'))
    df['num_adjectives'] = df['tweet'].apply(count_adjectives)
    
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
    df['tweet_processed'] = df['tweet'].apply(normalize_content)

    def add_tokens(text):
        # words list
        words = text.split()

        # lexicon
        misogyny_list = load_misogyny_lexicon("../../lexicons_train_misogyny_lexicon.txt")
        insults_list = load_insults_lexicon("../../lexicons_train_insults_lexicon.txt")

        # Reemplaza insultos por un token especial
        processed_words = ['[INSULT]' if word.lower() in insults_list or word.lower() in misogyny_list else word for word in words]
        return ' '.join(processed_words)
    
    df['tweet_processed'] = df['tweet_processed'].apply(add_tokens)
    
    # Eliminar filas donde 'full_text_processed' es una cadena vacía
    df = df[df['tweet_processed'] != ""]

    return df