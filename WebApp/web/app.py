### VERSION 1.0 - 16/02/24 (inicio)

# ========================== LIBRARIES ==========================
from functools import wraps
import time
from flask import Flask, jsonify, request, render_template, redirect, url_for, session
from flask_caching import Cache
from datetime import datetime
import json
import configparser
import requests
import logging
import os
import ast
#import flask_monitordashboard as dashboard

# ========================== CONFIG ==========================
config = configparser.ConfigParser()
config.read('config.ini')

LOCAL = ast.literal_eval(config['DEFAULT']['LOCAL']) # change this parameter in config.ini (for dockers set to False)
print(f"{LOCAL=}")
if LOCAL == True:
    # local
    WEB_HOST = config['DEFAULT']['WEB_HOST']
    API_HOST = config['DEFAULT']['API_HOST']
    API_PORT = config['DEFAULT']['API_PORT']
    API_URL = f'http://{API_HOST}:{API_PORT}'
    WEB_PORT = config['DEFAULT']['WEB_PORT']
    WEB_URL = f'http://{WEB_HOST}:{WEB_PORT}'
else:
    # dockers
    WEB_HOST = config['DOCKERS']['WEB_HOST']
    API_HOST = config['DOCKERS']['API_HOST']
    API_PORT = config['DOCKERS']['API_PORT']
    API_URL = f'http://{API_HOST}:{API_PORT}'
    WEB_PORT = config['DOCKERS']['WEB_PORT']
    WEB_URL = f'http://{WEB_HOST}:{WEB_PORT}'

# cache
CACHE_TYPE = config['CACHE']['CACHE_TYPE']
CACHE_TIMEOUT = int(config['CACHE']['CACHE_TIMEOUT']) # que no sea muy grande pq puede hacer que vaya más lento

# endpoints
WEB_HEALTHCHECK_ENDPOINT = config['ENDPOINTS']['WEB_HEALTHCHECK_ENDPOINT']
WEB_HEALTHCHECK_URL = f'{WEB_URL}/{WEB_HEALTHCHECK_ENDPOINT}'

API_HEALTHCHECK_ENDPOINT = config['ENDPOINTS']['API_HEALTHCHECK_ENDPOINT']
API_HEALTHCHECK_URL = f'{API_URL}/{API_HEALTHCHECK_ENDPOINT}'

API_PREDICT_ENDPOINT = config['ENDPOINTS']['API_PREDICT_ENDPOINT']
API_PREDICT_URL = f'{API_URL}/{API_PREDICT_ENDPOINT}'

API_PREDICT_MULTIPLE_ENDPOINT = config['ENDPOINTS']['API_PREDICT_MULTIPLE_ENDPOINT']
API_PREDICT_MULTIPLE_URL = f'{API_URL}/{API_PREDICT_MULTIPLE_ENDPOINT}'

API_ADD_FEEDBACK_ENDPOINT = config['ENDPOINTS']['API_ADD_FEEDBACK_ENDPOINT']
API_ADD_FEEDBACK_URL = f'{API_URL}/{API_ADD_FEEDBACK_ENDPOINT}'

API_GET_DATA_FROM_MONGO_ENDPOINT = config['ENDPOINTS']['API_GET_DATA_FROM_MONGO_ENDPOINT']
API_GET_DATA_FROM_MONGO_URL = f'{API_URL}/{API_GET_DATA_FROM_MONGO_ENDPOINT}'

# ========================== LOGGING ==========================
logging.basicConfig(filename='web.log', 
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

# ========================== APP ==========================
app = Flask(__name__)

app.secret_key = os.urandom(24) #app.secret_key = os.environ.get('SECRET_KEY')
#dashboard.bind(app) # in /dashboard

app.config['CACHE_TYPE'] = CACHE_TYPE
cache = Cache(app)
cache_keys = set()

# ========================== FUNCTIONS ==========================

def add_to_cache(key, value, timeout=CACHE_TIMEOUT):
    """
    Permite añadir las predicciones en cache.

    Args:
        key (str): Identificador en formato string.
        value (obj): Objeto a almacenar en caché.
        timeout (int): Tiempo en segundos por el que el objeto será almacenado en caché. 1 día por defecto

    Returns:
    """
    cache.set(key, value, timeout=timeout)
    expire_at = time.time() + timeout  # Tiempo actual + segundos de timeout
    cache_keys.add((key, expire_at))


def get_from_cache(key):
    """
    Recupera un objeto de la caché y muestra el tiempo restante de expiración.

    Args:
        key (str): La clave del objeto a recuperar.

    Returns:
        JSON con los valores almacenados en caché para la clave dada, o None si no se encuentra en caché.
    """
    cache_item = cache.get(key)
    if cache_item:
        expire_info = next((item for item in cache_keys if item[0] == key), None)
        remaining_time = expire_info[1] - time.time() if expire_info else None
        return {'value': cache_item, 'remaining_time': remaining_time}
    else:
        return {'value': None, 'remaining_time': None}


def log(msg):
    """
    Permite mostrar en pantalla la info indicada en msg junto con la fecha.

    Args:
        msg (str): Mensaje a mostrar.

    Returns:
    """
    print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} - {msg}')


def require_api_token(func):
    @wraps(func)
    def check_token(*args, **kwargs):
        # Check to see if it's in their session
        if 'api_session_token' not in session:
            # If it isn't return our access denied message (you can also return a redirect or render_template)
            #return requests.Response("Access denied")
            return redirect(url_for('auth'))

        # Otherwise just send them where they wanted to go
        return func(*args, **kwargs)

    return check_token


def return_500_if_errors(f):
    """
    Decorador para manejar excepciones en las funciones de la app.

    Este decorador intenta ejecutar la función proporcionada. Si ocurre una excepción 
    durante la ejecución, se atrapa y se devuelve una respuesta JSON con un código de 
    estado HTTP 500, que indica un error interno del servidor.

    Input:
        f (function): La función a la que se le aplicará el decorador. Esta función es
                    la que se ejecutará dentro del bloque try-except.
    Output:
        wrapper (function): La función envoltura que se devuelve y que reemplaza a 'f'.
                            Esta función envoltura es la que se llamará finalmente, y manejará
                            la excepción si ocurre alguna al llamar a 'f'.
    """

    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            print(f(*args, **kwargs))
            response = {
                'status_code': 500,
                'status': 'Internal Server Error: There was an error with the application'
            }
            return jsonify(response), 500
    # Renaming the function name so we avoid repeated endpoints error
    wrapper.__name__ = f.__name__
    return wrapper


# ========================== ENDPOINTS ==========================

@app.route('/')
def index():
    """
    Renderiza el html de la página principal.

    Args:

    Returns:
        render_template('index.html'): página principal de la app.
    """

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Acceder al texto del formulario
    text = request.form['chatInput']  
    print(f"Received text: {text}")

    

    return jsonify({'response': text})



@app.route('/home')
def home():
    """
    Renderiza el html de la página principal.

    Args:

    Returns:
        render_template('index.html'): página principal de la app.
    """

    return render_template('home.html')


@app.route('/home/predict')
def home_predict():
    """
    Renderiza el html de la página principal.

    Args:

    Returns:
        render_template('index.html'): página principal de la app.
    """

    return render_template('home_predict.html')

@app.route('/home/info')
def home_info():
    """
    Renderiza el html de la página principal.

    Args:

    Returns:
        render_template('index.html'): página principal de la app.
    """

    return render_template('home_info.html')






# ========================== RUN APP ==========================
if __name__ == "__main__":
    app.run(host=WEB_HOST, port=WEB_PORT, debug=LOCAL)




