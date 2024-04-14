
# ========================== LIBRARIES ==========================
from flask import jsonify
import joblib
import configparser
import torch
from transformers import RobertaTokenizer
from transformers import RobertaModel
import ast
from markupsafe import escape
import re
#from langid.langid import LanguageIdentifier, model
import langdetect as ld
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string
import numpy as np
from datetime import datetime
import nltk


# ========================== CONFIG ==========================
config = configparser.ConfigParser()
config.read('config.ini')

# params
LABELS = ast.literal_eval(config['PARAMS']['LABELS'])
N_LABELS = len(LABELS)
LABELS_TO_RISK = ast.literal_eval(config['PARAMS']['LABELS_TO_RISK'])
LOC_MODELS = ast.literal_eval(config['PARAMS']['LOC_MODELS'])
MAX_LENGTH_PHRASE = int(config['PARAMS']['MAX_LENGTH_PHRASE'])
LANG_CONF = float(config['PARAMS']['LANG_CONF'])
LANGUAGES = ast.literal_eval(config['PARAMS']['LANGUAGES'])
MODELS_NAMES = ast.literal_eval(config['PARAMS']['MODELS_NAMES'])
MODELS_SIMPLE = [model for model in MODELS_NAMES if model != "ROBERTA"]
LOC_EMBEDDINGS = ast.literal_eval(config['PARAMS']['LOC_EMBEDDINGS'])
PATH_EMBEDDINGS_GLOBAL = config['PARAMS']['PATH_EMBEDDINGS_GLOBAL']
PATH_EMBEDDINGS_LOCAL = config['PARAMS']['PATH_EMBEDDINGS_LOCAL']


# ======================= MODELS =======================

#### ONLY for Download and save locally (DO NOT EXECUTE MORE THAN ONCE)
#tokenizer = RobertaTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne", cache_dir=local_directory)
#model_nlp = RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne", cache_dir=local_directory)

#### ONLY for loading models from HuggingFace web
tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model_nlp = RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-large-bne")

# tokenizer and model (LOCAL)
# tokenizer = RobertaTokenizer.from_pretrained(DOWNLOADS_MODEL_PATH)
# model_nlp = RobertaModel.from_pretrained(DOWNLOADS_MODEL_PATH)

class BERTClass(torch.nn.Module):
    
    def __init__(self):
        super(BERTClass, self).__init__()
        
        self.l1 = model_nlp
        self.pre_classifier = torch.nn.Linear(1024, 264)   #### If you're using a large model the input dimension might not be 768 but higher
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(264, N_LABELS)
    def forward(self, input_ids, attention_mask, labels):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        #output = torch.softmax(output)
        return output


# ========================== FUNCTIONS ==========================

def return_500_if_errors(f):
    """
    Decorador para manejar excepciones en las funciones de la app.

    Este decorador intenta ejecutar la función proporcionada. Si ocurre una excepción 
    durante la ejecución, se atrapa y se devuelve una respuesta JSON con un código de 
    estado HTTP 500, que indica un error interno del servidor.

    Args:
        f (function): La función a la que se le aplicará el decorador. Esta función es
                    la que se ejecutará dentro del bloque try-except.
    Returns:
        wrapper (function): La función envoltura que se devuelve y que reemplaza a f.
                            Esta función envoltura es la que se llamará finalmente, y manejará
                            la excepción si ocurre alguna al llamar a f.
    """
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            print(f(*args, **kwargs))
            response = {
                'status_code': 500,
                'status': 'Internal Server Error: models not loaded'
            }
            return jsonify(response), 500
    # Renaming the function name so we avoid repeated endpoints error
    wrapper.__name__ = f.__name__
    return wrapper


def log(msg:str):
    """
    Permite mostrar en pantalla la info indicada en msg junto con la fecha.

    Args:
        msg (str): Mensaje a mostrar.

    Returns:
    """
    print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} - {msg}')


def load_models(ob:str) -> dict:
    """
    Carga la información de los modelos de predicción definidos en el archivo de configuración.

    Args:
        ob (str): Nombre de la OB seleccionada.

    Returns:
        models (dict): Diccionario de modelos con su correspondiente información.
    """
    # Diccionario para almacenar la configuración de los modelos
    models = {}

    for model_name in MODELS_NAMES:
        model_config = config[model_name]
        models[model_name] = {}

        for variant in ['local', 'global']:
            model_path = model_config.get(f'{variant.upper()}_PATH', '')  # '' Valor predeterminado si PATH no está definido
            if variant == "global":
                models[model_name][variant] = {
                    'model_path': model_path,
                }
            else:
                models[model_name][variant] = {
                    'model_path': f'{model_path}{LOC_MODELS[ob]}',
                }
    
    print(models)
    return models


def load_embeddings(ob:str):
    """
    Carga el embedding local y global del modelo seleccionado para la OB indicada. Si no se
    encuentra el embedding seleccionado, se devuelve None.

    Args:
        ob (str): Nombre de la OB seleccionada.

    Returns:
        embedding_global (object): Embedding correspondiente al modelo global.
        embedding_loc (object): Embedding correspondiente al modelo local.
    """
    try:
        # Cargar los embeddings desde archivo .pkl
        embedding_global = joblib.load(f'{PATH_EMBEDDINGS_GLOBAL}.pkl')
        embedding_loc = joblib.load(f'{PATH_EMBEDDINGS_LOCAL}{LOC_MODELS[ob]}.pkl')
        return embedding_global, embedding_loc
    
    except FileNotFoundError:
        # si no se encuentra embedding, se devuelve None
        print(f"No se encontró embedding para {ob}")
        return None


def load_local_model(model_name:str, model_path:str):
    """
    Carga el modelo de predicción local.

    Args:
        model_name (str): Nombre del modelo de predicción local.
        model_path (str): Ruta del modelo de predicción local.

    Returns:
        model_loc (object): Modelo de predicción local.
    """
    if model_name == "ROBERTA":
        # carga del modelo de roberta
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_loc = BERTClass()
        model_loc.to(device)
        model_loc.load_state_dict(torch.load(f'{model_path}.model', map_location=torch.device('cpu')))
    if model_name in MODELS_SIMPLE:
        # carga de modelos simples
        model_loc = joblib.load(f'{model_path}.pkl')
    
    return model_loc


def load_global_model(model_name:str, model_path:str):
    """
    Carga el modelo de predicción global.

    Args:
    model_name (str): Nombre del modelo de predicción local.
    model_path (str): Ruta del modelo de predicción local.

    Returns:
    model_global (object): Modelo de predicción de global.
    """
    if model_name == "ROBERTA":
        # carga del modelo de roberta
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_global = BERTClass()
        model_global.to(device)
        model_global.load_state_dict(torch.load(f'{model_path}.model', map_location=torch.device('cpu')))
    if model_name in MODELS_SIMPLE:
        # carga de modelos simples
        model_global = joblib.load(f'{model_path}.pkl')
    
    return model_global


def validate_input(ob:str, phrase:str, model_name:str) -> tuple[str, int] | tuple[dict, int]:
    """
    Crea un JSON con la respuesta a la request de la web y la predicción del modelo.

    Args:
    ob (str): Nombre de la OB seleccionada.
    phrase (str): Frase que ha introducido el usuario en la web.
    model_name (str): Nombre del modelo de predicción.

    Returns:
    phrase (str): Frase validada que ha introducido el usuario en la web.
    """
    # validation
    if ob is None:
        return jsonify({"error": "Missing 'ob' parameter"}), 400
    
    if phrase is None:
        return jsonify({"error": "Missing 'phrase' parameter"}), 400
    
    if not re.search("[a-zA-Z]", phrase): # check if 'phrase' contains at least one alphabetic character
        return jsonify({"error": "'phrase' must contain at least one alphabetic character"}), 400
    
    if len(phrase) > MAX_LENGTH_PHRASE: # Check if input lengths are within the allowed range
        return jsonify({"error": "'phrase' exceeds maximum length"}), 400
    

    # detect language
    item = ld.detect_langs(phrase)[0]
    lang = item.lang # The first one returned is usually the one that has the highest probability
    conf = item.prob

    #langid
    #identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    #lang2, conf2 = identifier.classify(phrase)
    
    
    print(f"{phrase=}")
    print(f"{lang=}")
    print(f"{conf=}")
    #print(f"{lang2=}")
    #print(f"{conf2=}")
    print(f"{LANGUAGES=}")
    print(f"{model_name=}")
    print(f"{MODELS_NAMES=}")

    ###que pasa cuando escribes español y error en ingles: "me pasa que la maquina me dice error in the system"
    if lang not in LANGUAGES:
        print("1")
        return jsonify({"error": "No se ha podido analizar el texto recibido al contener expresiones desconocidas"}), 400
    
    if conf < LANG_CONF:
        print("2")
        return jsonify({"error": "No se ha podido analizar el texto recibido al contener expresiones desconocidas"}), 400

    if model_name not in MODELS_NAMES:
        print("3")
        return jsonify({"error": "No se ha podido analizar el texto recibido al contener expresiones desconocidas"}), 400

    print("4")
    return phrase, 200


def process_text(text:str):
    """
    Procesa el texto eliminando stopwords, signos de puntuación y aplicando stemming.

    Args:
    text (str): Frase que ha introducido el usuario en la web.
    embedding_name (str): Nombre del embedding.
    embedding (object): Embedding.

    Returns:
    phrase (list): Vector del embedding local con la frase procesada.
    """
    
    # Initialize stemmer
    stemmer = SnowballStemmer('spanish')

    # Lower-case text
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    spanish_stopwords = set(stopwords.words('spanish'))
    spanish_stopwords.remove("no")  # Retain "no" as it provides negative context
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.lower() not in spanish_stopwords]
    
    # Apply stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    out_text = ' '.join(stemmed_words)

    return out_text


def phrase_to_vector(text:str, embedding_name:str, embedding):
    """
    Convierte una cadena de texto en vectores en función del embedding.

    Args:
    text (str): Frase que ha introducido el usuario en la web.
    embedding_name (str): Nombre del embedding.
    embedding (object): Embedding.

    Returns:
    phrase (list): Vector del embedding local con la frase procesada.
    """

    if embedding_name == "bow" or embedding_name == "tfidf":
        phrase = embedding.transform([text]).toarray()[0]
        return phrase
    
    if embedding_name == "fasttext" or "word2vec":
        words = text.split()
        vect = [embedding.wv.get_vector(word) for word in words if word in embedding.wv.key_to_index]
        phrase = np.mean(vect, axis=0)
        return phrase


def process_input(phrase:str, model_name:str, ob:str) -> tuple[list, list]:
    """
    Crea un JSON con la respuesta a la request de la web y la predición del modelo.

    Args:
    phrase (str): Frase que ha introducido el usuario en la web.
    model_name (str): Nombre del modelo de NLP.
    ob (str): Nombre de la OB seleccionada.

    Returns:
    phrase_loc (list): Vector del embedding local con la frase procesada.
    phrase_global (list): Vector del embedding global con la frase procesada.
    """
    # escape special characters to prevent XSS
    phrase = escape(phrase) ### TO FINISH 

    # Si el modelo es roberta, phrase es la misma para global y local al tener BERT el propio embedding
    phrase_loc, phrase_global = phrase, phrase
    
    # Check models
    if model_name != "ROBERTA":
        # Procesa el texto
        phrase = process_text(phrase)
        # comprobamos que no es una cadena vacía al limpiar el texto
        if phrase == "":
            return jsonify({"error": "No se ha podido analizar el texto recibido al contener expresiones desconocidas"}), 400

        # cargamos embeddings
        embedding_global, embedding_loc = load_embeddings(ob)
        embedding_name_loc = LOC_EMBEDDINGS[LOC_MODELS[ob]]
        embedding_name_global = LOC_EMBEDDINGS["GLOBAL"]
        
        # convertimos phrase (texto) en vectores
        phrase_loc = phrase_to_vector(phrase, embedding_name_loc, embedding_loc)
        phrase_global = phrase_to_vector(phrase, embedding_name_global, embedding_global)
    
    return phrase_loc, phrase_global


def evaluate_input(phrase_loc:list, phrase_global:list, model_name:str, model_global, model_loc) -> tuple[list, list, list, list]:
    """
    Crea un JSON con la respuesta a la request de la web y la predición del modelo.

    Args:
    phrase_loc (list): Vector del embedding local con la frase procesada.
    phrase_global (list): Vector del embedding global con la frase procesada.
    model_name (str): Nombre del modelo de NLP.
    model_global (object): Modelo de predicción global.
    model_loc (object): Modelo de predicción local.

    Returns:
    probabilities_list_global (list): Lista con las probs de cada clase para el modelo global.
    predicted_labels_list_global (list): Lista con las clases ordenadas por probabilidad para el modelo global.
    probabilities_list_loc (list): Lista con las probs de cada clase para el modelo local.
    predicted_labels_list_loc (list): Lista con las clases ordenadas por probabilidad para el modelo local.
    """
    if model_name == "ROBERTA":
        phrase = phrase_loc # phrase_loc es igual que phrase_global pq el embedding es el mismo en roberta
        # tokenize input
        inputs = tokenizer(phrase, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_masks = inputs['attention_mask']

        # evaluate models
        model_global.eval()
        model_loc.eval()

        # get outputs
        outputs_global = model_global(input_ids = input_ids, attention_mask = attention_masks, labels = LABELS)
        outputs_loc = model_loc(input_ids = input_ids, attention_mask = attention_masks, labels = LABELS)

        # get probabilities
        probabilities_global = torch.softmax(outputs_global, dim=1)
        probabilities_list_global = probabilities_global.tolist()[0]

        probabilities_loc = torch.softmax(outputs_loc, dim=1)
        probabilities_list_loc = probabilities_loc.tolist()[0]

        # get labels
        predicted_labels_global = torch.argmax(probabilities_global, dim=1)
        predicted_labels_list_global = predicted_labels_global.tolist()

        predicted_labels_loc = torch.argmax(probabilities_loc, dim=1)
        predicted_labels_list_loc = predicted_labels_loc.tolist()
    
    if model_name in MODELS_SIMPLE:
        outputs_global = model_global.predict(phrase_global.reshape(1, -1))[0]
        outputs_loc = model_loc.predict(phrase_loc.reshape(1, -1))[0]
        
        probabilities_global = model_global.predict_proba(phrase_global.reshape(1, -1))[0]
        probabilities_list_global = probabilities_global.tolist()

        probabilities_loc = model_loc.predict_proba(phrase_loc.reshape(1, -1))[0]
        probabilities_list_loc = probabilities_loc.tolist()

        # Convertir los arrays de NumPy a tensores de PyTorch
        probabilities_global = torch.tensor(probabilities_global)
        probabilities_loc = torch.tensor(probabilities_loc)

        # Obtener las etiquetas predichas para los modelos global y local
        predicted_labels_global = torch.argmax(probabilities_global).item()
        predicted_labels_list_global = [predicted_labels_global]

        predicted_labels_loc = torch.argmax(probabilities_loc).item()
        predicted_labels_list_loc = [predicted_labels_loc]

    return probabilities_list_global, probabilities_list_loc, predicted_labels_list_global, predicted_labels_list_loc


def response_model(model_name:str, probabilities_list_global:list, predicted_labels_list_global:list, probabilities_list_loc:list, predicted_labels_list_loc:list) -> dict:
    """
    Crea un JSON con las predicción de un modelo para el endpoint 'predict_multiple'.

    Args:
    model_name (str): Nombre del modelo de NLP.
    probabilities_list_global (list): Lista con las probs de cada clase para el modelo global.
    predicted_labels_list_global (list): Lista con las clases ordenadas por probabilidad para el modelo global.
    probabilities_list_loc (list): Lista con las probs de cada clase para el modelo local.
    predicted_labels_list_loc (list): Lista con las clases ordenadas por probabilidad para el modelo local.

    Returns:
    res (dict): Diccionario con los datos especificados.
    """
    res = {
            "model_name": model_name,
            "global": {
                "probabilities": {
                            "high": probabilities_list_global[2],
                            "medium": probabilities_list_global[1],
                            "low": probabilities_list_global[0]
                        },
                "probabilities_risk_global": "{:.2f}%".format(probabilities_list_global[predicted_labels_list_global[0]]* 100),
                "label": predicted_labels_list_global[0],
                "risk": LABELS_TO_RISK[predicted_labels_list_global[0]]
            },
            "local": {
                "probabilities": {
                            "high": probabilities_list_loc[2],
                            "medium": probabilities_list_loc[1],
                            "low": probabilities_list_loc[0]
                        },
                "probabilities_risk_local": "{:.2f}%".format(probabilities_list_loc[predicted_labels_list_loc[0]]* 100),
                "label": predicted_labels_list_loc[0],
                "risk": LABELS_TO_RISK[predicted_labels_list_loc[0]]
            }
        }

    return res


def generate_response(ob:str, phrase:str, models_results:dict) -> dict:
    """
    Crea un JSON con la respuesta formateada a la request de la web y las predicciones de los modelos
    seleccionados para el endpoint 'predict_multiple'.

    Args:
    ob (str): Nombre de la OB seleccionada.
    phrase (str): Frase que ha introducido el usuario en la web.
    models_results (dict): Diccionario con las predicciones de todos los modelos seleccionados
    
    Returns:
    res (dict): Diccionario con los datos especificados.
    """
    res = {
            "ob": ob,
            "phrase": phrase,
            "models": models_results
        }

    return res