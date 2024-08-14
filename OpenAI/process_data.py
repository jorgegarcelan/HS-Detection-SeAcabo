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


def filter_by_lang(df):
    langs = ['es', 'cy', 'ht', 'in', 'lt', 'qam', 'tl', 'und']
    df = df[df['lang'].isin(langs)]
    
    return df


def load_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lexicon = {line.strip().lower() for line in file if line.strip()}
    return lexicon




def add_special_tokens(df):
    # lexicons
    misogyny_list = load_lexicon("../Lexicons/lexicons_train_misogyny_lexicon.txt")
    insults_list = load_lexicon("../Lexicons/lexicons_train_insults_lexicon.txt")
    victim_list = load_lexicon("../Lexicons/lexicons_victim_seacabo.txt")
    aggressor_list = load_lexicon("../Lexicons/lexicons_aggressor_seacabo.txt")
    insults_seacabo_list = load_lexicon("../Lexicons/lexicons_insults_seacabo.txt")

    def replace_words_with_tokens(text):
        words = text.split()
        processed_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in insults_list or word_lower in misogyny_list or word_lower in insults_seacabo_list:
                processed_words.append('[INSULT]')
            elif word_lower in victim_list:
                processed_words.append('[VICTIM]')
            elif word_lower in aggressor_list:
                processed_words.append('[AGGRESSOR]')
            else:
                processed_words.append(word)
        return ' '.join(processed_words)
    
    # Aplicar la función de procesamiento a la columna especificada
    df['full_text_processed'] = df['full_text_processed'].apply(replace_words_with_tokens)
    return df



def normalize_data(df):

    
    # Define function to remove stopwords, punctuation, and apply stemming
    def normalize_text(text):

        # Convertir a minúsculas
        text = text.lower()

        # Eliminar emojis
        #text = emoji.replace_emoji(text, replace='')
        
        # Replace emojis with the meaning
        text = emoji.demojize(text, language="es")

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

        # remove characters that are repeated
        filtered_words = [re.sub(r'(.)\1+', r'\1', palabra) for palabra in filtered_words]

    
        # Apply stemming (MIRAR)
        ##stemmed_words = [stemmer.stem(word) for word in filtered_words]
        
        ##return ' '.join(stemmed_words)  
        return ' '.join(filtered_words)
    df['full_text_processed'] = df['full_text'].apply(normalize_text)

    # Eliminar filas donde 'full_text_processed' es una cadena vacía
    df = df[df['full_text_processed'] != ""]

    return df


def filter_by_type(df, type_col):

    if type_col == "analisis_general":
        # Define the specific labels to keep
        etiquetas = ["Comentario Positivo", "Comentario Negativo"]
        
        df['Análisis General'] = df['Análisis General'].where(df['Análisis General'].isin(etiquetas))

        # Remove NAs
        df = df.dropna(subset=['Análisis General'])

    if type_col == "contenido_negativo":
        # Filtrar el DataFrame para seleccionar solo los "Comentario Negativo"
        df = df.loc[df['Análisis General'] == 'Comentario Negativo']

        # Define the specific labels to keep
        etiquetas = ["Desprestigiar Víctima", "Desprestigiar Acto", "Insultos", "Desprestigiar Deportista Autora"]
        df['Contenido Negativo'] = df['Contenido Negativo'].where(df['Contenido Negativo'].isin(etiquetas))

        # Remove NAs
        df = df.dropna(subset=['Contenido Negativo'])

    if type_col == "insultos":
        # Filtrar el DataFrame para seleccionar solo los "Comentario Negativo"
        df = df.loc[df['Análisis General'] == 'Comentario Negativo']

        # Filtrar el DataFrame para seleccionar solo los "Insultos" no vacíos
        df = df.loc[df['Insultos'].notna() & (df['Insultos'].str.strip() != '')]

        # Define the specific labels to keep
        etiquetas = ["Deseo de Dañar", "Genéricos", "Sexistas/misóginos"]

        # Replace labels that are not in the list with "Genéricos"
        df['Insultos'] = df['Insultos'].where(df['Insultos'].isin(etiquetas), other="Genéricos")

        # Remove NAs
        df = df.dropna(subset=['Insultos'])

    return df



def process_data(df, type_col):

    # Filter by lang
    df = filter_by_lang(df)

    # Normalize data
    df = normalize_data(df)

    # Add special tokens
    df = add_special_tokens(df)

    # Filter by type
    df = filter_by_type(df, type_col)

    # Reset the index after filtering
    df = df.reset_index(drop=True)

    # Create new columns for the predictions
    df['Predicted AG'] = pd.NA

    df['Predicted CNeg'] = pd.NA

    df['Predicted I'] = pd.NA

    #print(df)

    return df