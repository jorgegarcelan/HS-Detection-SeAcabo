# ==============================================
# VERSION 1 POR JORGE GARCELÁN GÓMEZ - 19/03/24
# ==============================================

# Importación de librerías

import uuid

from sklearn.model_selection import train_test_split
import json
import collections
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
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from joblib import dump
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import re
import torch
from transformers import BertTokenizer, DistilBertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


# ======================= FUNCTIONS =======================
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

# ======================= MAIN =======================


def init(embedding_name, embedding_size, model, cv):
    # Genera un UUID
    run_id = str(uuid.uuid4())  
    print(f"ID de ejecución: {run_id}")

    print(f"Embedding Name: {embedding_name}")
    print(f"Embedding Size: {embedding_size}")
    print(f"Model: {model}")
    print(f"Cross-Validation: {cv}")
    
    return run_id


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


    
def process_data(df, type_id):


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



    

    # Initialize stemmer
    ##stemmer = SnowballStemmer('spanish')
    
    # Define function to remove stopwords, punctuation, and apply stemming
    def remove_spanish_stopwords(text):

        # Eliminar menciones a usuarios (palabras que comienzan con @)
        text = re.sub(r'@\w+', '', text)

        # Eliminar enlaces (todo lo que comienza con http o https)
        #text = re.sub(r'http\S+|https\S+', '', text)

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

    df['tweet_processed'] = df['tweet'].apply(remove_spanish_stopwords)
    # Eliminar filas donde 'tweet_processed' es una cadena vacía
    df = df[df['tweet_processed'] != ""]


    return df





def embedding_data(df, embedding_name, embedding_size, run_id):

    if embedding_name == "fasttext":
        # Create a FastText model
        sentences = df['tweet_processed'].str.split().tolist()
        #print(f"{sentences=}")
        embedding_size = embedding_size  # you can adjust this value as needed
        model = FastText(sentences, vector_size=embedding_size, window=15, min_count=1, workers=8, sg=1, epochs=10)

        # Convert text data into FastText embeddings
        def text_to_vector(text):
            #print(f"{text=}")
            words = text.split()
            #print(f"{words=}")
            vector = np.mean([model.wv[word] for word in words if word in model.wv.index_to_key], axis=0)
            if len(vector) == 0:
                return np.zeros(embedding_size)
            else:
                return vector

        df['tweet_filtered'] = df['tweet_processed'].apply(text_to_vector)

        X = np.stack(df['tweet_filtered'].values)
        y = df['label'].values

    elif embedding_name == "word2vec":
        # Create a FastText model
        sentences = df['tweet_processed'].str.split().tolist()
        embedding_size = embedding_size  # you can adjust this value as needed
        model = Word2Vec(sentences, vector_size=embedding_size, window=15, min_count=1, workers=8, sg=1, epochs=10)

        # Convert text data into FastText embeddings
        def text_to_vector(text):
            words = text.split()
            vector = np.mean([model.wv[word] for word in words if word in model.wv.index_to_key], axis=0)
            if len(vector) == 0:
                return np.zeros(embedding_size)
            else:
                return vector


        df['tweet_filtered'] = df['tweet_processed'].apply(text_to_vector)

        X = np.stack(df['tweet_filtered'].values)
        y = df['label'].values


    elif embedding_name == "bow":
        # Creating the bag of Word Model
        model = CountVectorizer(max_features = 5000, ngram_range=(1, 5))
        X = model.fit_transform(df['tweet_processed']).toarray()
        y = df['label'].values

    elif embedding_name == "tfidf":
        # Creating the TF-IDF model
        model = TfidfVectorizer(max_features=5000, ngram_range=(1, 5))
        X = model.fit_transform(df['tweet_processed']).toarray()
        y = df['label'].values
        
    elif embedding_name == "custom":
        wordvectors_file = 'embeddings-l-model.bin'
        model = FastText.load_fasttext_format(wordvectors_file)
        # Function to convert text to vector using the pre-trained model
        def text_to_vector(text):
            words = text.split()
            vector = np.mean([model.wv[word] for word in words if word in model.wv], axis=0)
            if len(vector) == 0:
                return np.zeros(model.vector_size)
            else:
                return vector
            
        df['tweet_filtered'] = df['tweet_processed'].apply(text_to_vector)

        X = np.stack(df['tweet_filtered'].values)
        y = df['label'].values

    elif embedding_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne') 
        model = RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-large-bne") 

        # Asumiendo que 'get_roberta_embeddings' está definida
        embeddings = get_roberta_embeddings(df['tweet_processed'].tolist(), tokenizer, model)
        
        X = embeddings
        y = df['label'].values

    # Add the additional features to your embeddings
    additional_features = df[['tweet_length', 'num_adjectives']].values
    X = np.hstack((X, additional_features))

    ## SAVE MODEL when running best model -> future (https://radimrehurek.com/gensim/models/fasttext.html)
    #dump(model, f"embeddings/embedding_{run_id}.pkl")
    return df, X, y


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


def model_data(X_train, y_train, model, cv):

    if model == "random_forest":
        # Define Random Forest Classifier
        model = RandomForestClassifier(random_state = 0)

        # Define the hyperparameters and their possible values
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
            }

    elif model == "logistic_regression":
        # Define Logistic Regression Model
        model = LogisticRegression()

        # Define the hyperparameters and their possible values
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [50, 100, 200]
        }

    elif model == "svc":
        # Define SVC Model
        model = SVC()

        # Define the hyperparameters and their possible values
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],  # only used when kernel is 'poly'
            'gamma': ['scale', 'auto']
        }

    elif model == "xgboost":
        # Define XGBoosting Model
        model = XGBClassifier(use_label_encoder=False, eval_metrix='logloss')

        # Define the hyperparameters and their possible values
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.5, 0.7, 1.0]
        }

    elif model == "mlp":
        # MLPClassifier does not support class_weight; consider using sample_weight in fit() method
        model = MLPClassifier(max_iter=1000)  # Increased max_iter for convergence
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }

    elif model == "naive_bayes":
        # GaussianNB does not support class_weight; consider using sample_weight in fit() method
        model = GaussianNB()
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }

    # Set up the grid search with k-fold cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=cv, verbose=10, n_jobs=-1)

    # Fit the model with the data
    grid_search.fit(X_train, y_train)

    # Save the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_index = grid_search.best_index_
    std_dev = grid_search.cv_results_['std_test_score'][best_index]

    return best_params, best_model, std_dev

"""
def eval_data(best_model, X_test, y_test, labels_names):
    y_pred = best_model.predict(X_test)
    acc_global = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True, target_names=labels_names)

    return acc_global, report
"""

def eval_data(run_id, best_model, X_test, y_test, labels_names):
    y_pred = best_model.predict(X_test)
    acc_global = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True, target_names=labels_names)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels_names))
    plt.xticks(tick_marks, labels_names, rotation=45)
    plt.yticks(tick_marks, labels_names)

    # Annotate the matrix with text
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the figure
    plt.savefig(f"Conf_matrix/{run_id}.png")
    plt.close()

    return acc_global, report



def log_data(run_id, data_train, data_dev, embedding_name, embedding_size, model, cv, best_params, report, acc_global, std_global, exec_time, timestamp, type_id, balance):

    # Si existe, lee el contenido actual
    df = pd.read_excel(archivo_excel, engine='openpyxl')
    

    # Crea nueva fila con los datos de la ejecucción
    nueva_fila = {
            'Run_ID': run_id,
            'Data Train': data_train,
            'Data Dev': data_dev,
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


data_train = "EXIST2024_Tweets_Dataset/training/EXIST2024_training.json" 
data_dev = "EXIST2024_Tweets_Dataset/dev/EXIST2024_dev.json"
archivo_excel = "EXIST2024_Training.xlsx"
type_id = "task_1" # ["task_1", "task_2", "task_3"]
balance = "None" # ["downsampling", "upsampling", "smote", "adasyn", "None"]
embedding_name = ["roberta"] #["fasttext", "word2vec", "bow", "tfidf", "custom"]
embedding_size = [1000] #[100, 500] 
models = ["random_forest", "logistic_regression", "svc", "xgboost","mlp", "naive_bayes"] #["random_forest", "logistic_regression", "svc", "xgboost", "mlp", "naive_bayes"]
cv = [3] #[3, 5]

# for resample in balance:
for name in embedding_name:
    for size in embedding_size:
        for model in models:
            for c in cv:
                start_time = time.time()
                timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

                run_id = init(name, size, model, c)

                # Cargamos datos
                df_train = load_data(data_train)
                df_dev = load_data(data_dev)

                # Process target
                df_train, labels_names, label_mapping = process_target(df_train, type_id, "train", None)
                df_dev = process_target(df_dev, type_id, "dev", label_mapping)

                # Preprocesado de datos
                df_train = process_data(df_train, type_id)
                df_dev = process_data(df_dev, type_id)

                print(f"{labels_names}")

                # Embeddings
                df_train, X_train, y_train = embedding_data(df_train, name, size, run_id)
                df_dev, X_dev, y_dev = embedding_data(df_dev, name, size, run_id)

                # Resampling
                X_train, y_train = resampling(X_train, y_train, balance)
                X_dev, y_dev = resampling(X_dev, y_dev, balance)

                # Models
                best_params, best_model, std_global = model_data(X_train, y_train, model, c)

                # Evaluation
                acc_global, report = eval_data(run_id, best_model, X_dev, y_dev, labels_names)

                # Save Model
                #dump(best_model, f"models/model_{run_id}.pkl") ////// por tema espacio no lo guardo ahora -- consultar docs

                end_time = time.time()
                exec_time = end_time - start_time

                # Create Log:
                log_data(run_id, data_train, data_dev, name, size, model, c, best_params, report, acc_global, std_global, exec_time, timestamp, type_id, balance)
