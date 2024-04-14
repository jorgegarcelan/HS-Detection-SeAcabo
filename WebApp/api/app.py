### VERSION 1.0 - 16/02/24 (inicio)

# ========================== LIBRARIES ==========================
from flask import Flask, jsonify, request
import configparser
import logging
import ast
from pymongo import MongoClient
from utils import *

# ========================== CONFIG ==========================
config = configparser.ConfigParser()
config.read('config.ini')


LOCAL = ast.literal_eval(config['DEFAULT']['LOCAL']) # change this parameter in config.ini (for dockers set to False)
print(f"{LOCAL=}")
if LOCAL == True:
    # local
    WEB_HOST = config['DEFAULT']['WEB_HOST']
    WEB_PORT = config['DEFAULT']['WEB_PORT']
    WEB_URL = f'http://{WEB_HOST}:{WEB_PORT}'
    API_HOST = config['DEFAULT']['API_HOST']
    API_PORT = config['DEFAULT']['API_PORT']
    API_URL = f'http://{API_HOST}:{API_PORT}'
    BBDD_HOST = config['DEFAULT']['BBDD_HOST']
    BBDD_PORT = config['DEFAULT']['BBDD_PORT']
    BBDD_URL = f'mongodb://{BBDD_HOST}:{BBDD_PORT}/'
    
else:
    # dockers
    WEB_HOST = config['DOCKERS']['WEB_HOST']
    WEB_PORT = config['DOCKERS']['WEB_PORT']
    WEB_URL = f'http://{WEB_HOST}:{WEB_PORT}'
    API_HOST = config['DOCKERS']['API_HOST']
    API_PORT = config['DOCKERS']['API_PORT']
    API_URL = f'http://{API_HOST}:{API_PORT}'
    BBDD_HOST = config['DOCKERS']['BBDD_HOST']
    BBDD_PORT = config['DOCKERS']['BBDD_PORT']
    BBDD_URL = f'mongodb://{BBDD_HOST}:{BBDD_PORT}/'

# database
BBDD_NAME = config['PARAMS']['BBDD_NAME']
BBDD_COLLECTION_FEEDBACK = config['PARAMS']['BBDD_COLLECTION_FEEDBACK']

# ========================== LOGGING ==========================
logging.basicConfig(filename='api.log', 
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

# ========================== DATABASE ==========================
client = MongoClient(BBDD_URL)
db = client[BBDD_NAME]
logging.info("BBDD active")

# ========================== APP ==========================
app = Flask(__name__)


# ========================== FUNCTIONS ==========================

@app.route("/healthcheck", methods=['GET'])
def health_check():
    """
    Esta función actúa como health check para verificar que el servicio api está operativo
    y responde correctamente a las solicitudes.

    Args:

    Returns:
        json: mensaje en formato json que indica que la app funciona.
    """
    app.logger.info('health check')
    logging.info('health check')
    return  jsonify({"message" : "Here I am!"})

@app.route('/get_data_from_mongo', methods=['GET'])
def get_data_from_mongo():
    """
    Devuelve todos los datos de la colección "collection_name" de la base de datos.

    Args:

    Returns:

    """
    collection_name = request.args.get('collection_name')

    if collection_name:
        collection = db[collection_name]
        
        data = list(collection.find({}, {'_id': 0})) # Excluye el campo '_id' para evitar problemas de serialización
        return jsonify(data)
    else:
        return "Por favor, especifica el nombre de la colección", 400


@app.route('/add_feedback', methods=['POST'])
@return_500_if_errors
def add_feedback():
    """
    El siguiente endpoint está dedicado a recoger el feedback de una ejecución de la web
    y añadirlo a una base de datos en mongoDB.

    Args:
    
    Returns:
        None
    """
    logging.info('New request to add_feedback endpoint ....')

    # Obtener los datos enviados a esta ruta
    data = request.json
    collection = db[BBDD_COLLECTION_FEEDBACK]
    result = collection.insert_one(data)
    documento_insertado = collection.find_one({'_id': result.inserted_id})
    if documento_insertado:
        print("El documento fue insertado con éxito:", documento_insertado)
        return "Data inserted"
    else:
        print("No se encontró el documento.")
        return "Error: data was not inserted"



@app.route("/predict", methods=['GET', 'POST'])
@return_500_if_errors
def predict():
    """
    El siguiente endpoint está dedicado para cuando se selecciona un sólo modelo:
    Recoge la petición de la web con los datos que ha introducido el usuario, 
    carga el modelo local y global, y responde con la predicción en formato json.

    Args:
    
    Returns:
        res (json): Respuesta en formato json con la predicción de los modelos y las probabilidades.
    """
    logging.info('New request to predict endpoint ....')
    
    # PREPROCESSING -------------------------------------
    # Obtener input de la web:
    logging.info('Extracting input data ....')
    json = request.get_json()
    phrase = json.get('phrase')
    ob = json.get('ob')
    model_name = json.get('model')[0] # el input es una lista con un solo elemento

    # Cargar el diccionario de modelos
    MODELS = load_models(ob)
    

    # VALIDATION ------------------------------------------
    # Validar input:
    logging.info('Validating input data ....')
    validate_output, validate_status = validate_input(ob, phrase, model_name)
    if validate_status != 200:
        # Manejar el error, validate_output contiene el mensaje de error
        return jsonify(validate_output), validate_status
    else:
        # Continuar con la lógica normal, validate_output contiene la frase válida
        phrase = validate_output
        print(f"{phrase=}")


    # PROCESSING ------------------------------------------
    # Procesar input en función del model_name:
    logging.info(f'Processing input data for model {model_name} ....')
    phrase_loc, phrase_global = process_input(phrase, model_name, ob)
    logging.info(f'Successful input data for model {model_name} ....')


    # MODELS ------------------------------------------
    # Cargar modelos global y local en función del model_name:
    logging.info(f'Loading {model_name} global model ....')
    print(MODELS[model_name]['global'])
    global_path = MODELS[model_name]["global"]["model_path"]
    model_global = load_global_model(model_name, global_path)

    logging.info(f'Loading {model_name} local model ....')
    print(MODELS[model_name]['local'])
    loc_path = MODELS[model_name]["local"]["model_path"]
    model_loc = load_local_model(model_name, loc_path)
    
    # EVALUATION ------------------------------------------
    logging.info('Performing evaluation ....')
    probabilities_list_global, probabilities_list_loc, predicted_labels_list_global, predicted_labels_list_loc = evaluate_input(phrase_loc, phrase_global, model_name, model_global, model_loc)
    

    # RESPONSE ---------------------------------------------
    logging.info('Responding ...')
    models_results = response_model(model_name, probabilities_list_global, predicted_labels_list_global, probabilities_list_loc, predicted_labels_list_loc)

    # Respuesta final con todos los resultados de los modelos
    res = generate_response(ob, phrase, models_results)

    print(f"{res=}")
    logging.info(res)
    return jsonify(res)


@app.route("/predict_multiple", methods=['GET', 'POST'])
@return_500_if_errors
def predict_multiple():
    """
    El siguiente endpoint está dedicado para cuando se seleccionan multiples modelos:
    Recoge la petición de la web con los datos que ha introducido el usuario, 
    carga los modelos local y global, y responde con las predicciones en formato json.

    Args:
    
    Returns:
        res (json): Respuesta en formato json con la predicción de los modelos y las probabilidades.
    """
    logging.info('New request to predict endpoint ....')
    
    # PREPROCESSING -------------------------------------
    # Obtener input de la web
    logging.info('Extracting input data ....')
    json = request.get_json()
    phrase = json.get('phrase')
    ob = json.get('ob')
    model_names = json.get('model')

    print(f"{ob=}")

    # Cargar el diccionario de modelos
    MODELS = load_models(ob)

    # Inicializa un diccionario para los resultados de todos los modelos
    models_results = []

    for model_name in model_names:
        # VALIDATION ------------------------------------------
        # Validar input:
        logging.info('Validating input data ....')
        validate_output, validate_status = validate_input(ob, phrase, model_name)
        print(f"{validate_status=}")
        if validate_status != 200:
            # Manejar el error, validate_output contiene el mensaje de error
            return validate_output, validate_status
        else:
            # Continuar con la lógica normal, validate_output contiene la frase válida
            phrase = validate_output
            print(f"{phrase=}")


        # PROCESSING ------------------------------------------
        # Procesar input en función del model_name:
        logging.info('Processing input data ....')
        phrase_loc, phrase_global = process_input(phrase, model_name, ob)
        logging.info('Successful input data ....')


        # MODELS ------------------------------------------
        # Cargar modelos global y local en función del model_name:
        logging.info(f'Loading {model_name} global model ....')
        #print(MODELS[model_name]['global'])
        global_path = MODELS[model_name]["global"]["model_path"]
        model_global = load_global_model(model_name, global_path)

        logging.info(f'Loading {model_name} local model ....')
        #print(MODELS[model_name]['local'])
        loc_path = MODELS[model_name]["local"]["model_path"]
        model_loc = load_local_model(model_name, loc_path)
        

        # EVALUATION ------------------------------------------
        logging.info('Performing evaluation ....')
        probabilities_list_global, probabilities_list_loc, predicted_labels_list_global, predicted_labels_list_loc = evaluate_input(phrase_loc, phrase_global, model_name, model_global, model_loc)
        

        # RESPONSE ---------------------------------------------
        logging.info('Responding ...')
        
        model_result = response_model(model_name, probabilities_list_global, predicted_labels_list_global, probabilities_list_loc, predicted_labels_list_loc)
        
        # Añade los resultados de este modelo al diccionario de resultados
        models_results.append(model_result)

    # Respuesta final con todos los resultados de los modelos
    res = generate_response(ob, phrase, models_results)

    print(f"{res=}")
    logging.info(res)
    return jsonify(res)


# ========================== RUN APP ==========================
if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=LOCAL)