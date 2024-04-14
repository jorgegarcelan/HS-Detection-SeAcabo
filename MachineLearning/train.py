# ==============================================
# VERSION 1 POR JORGE GARCELÁN GÓMEZ - 12/12/23
# VERSION 2 POR JORGE GARCELÁN GÓMEZ - 07/04/24
# ==============================================

# Importación de librerías
from utils import *
from load_data import *
from process_data import *
from create_embeddings import *
from split_data import *
from create_model import *
from evaluate_model import *
from run_to_excel import *

from joblib import dump
import time
from datetime import datetime
from pysentimiento.preprocessing import preprocess_tweet

# ======================= PARAMS =======================
archivo_excel = "Training.xlsx"

data = "data/BBDD_SeAcabo.csv" # "data/BBDD_SeAcabo.csv" "AMI_IBEREVAL2018/es_AMI_TrainingSet_NEW.csv"
type_id = "insultos" # ["analisis_general", "contenido_negativo", "insultos"]
balance = "smote" # ["downsampling", "upsampling", "smote", "adasyn", "None"] # falta "smote", "adasyn" para analisis_general y contenido_negativo
embedding_name = ["robertuito"] #["fasttext", "word2vec", "bow", "tfidf", "custom", "roberta", "beto", "bert-multi", "xlm-roberta-base", "robertuito"] // falta custom para ["analisis_general", "contenido_negativo", "insultos"]
embedding_size = [500] #[100, 500] 
models = ["random_forest", "logistic_regression", "xgboost", "mlp", "naive_bayes"] #["random_forest", "logistic_regression", "svc", "xgboost", "mlp", "naive_bayes"]  // falta svc para ["analisis_general", "contenido_negativo", "insultos"] y para ["downsampling", "upsampling", "smote", "adasyn"]
cv = [5] #[3, 5]


# ======================= MAIN =======================
for name in embedding_name:
    for size in embedding_size:
        for model in models:
            for c in cv:
                start_time = time.time()
                timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

                run_id = init(name, size, model, c)

                # Cargamos datos
                df = load_data(data)

                # Filtrar por type_id
                df, labels_names = filter_by_type(df, type_id, balance)

                # Preprocesado de datos
                if embedding_name == "robertuito":
                    df['full_text'] = df['full_text'].apply(lambda x: preprocess_tweet(x, lang="es", preprocess_handles=False))
                else:
                    df = process_data(df)

                # Embeddings
                df, X, y = embedding_data(df, name, size)

                # Split
                X_train, X_test, y_train, y_test = split_data(X, y, balance)

                # Models
                best_params, best_model, std_global = model_data(X_train, y_train, model, c)

                # Evaluation
                acc_global, report = eval_data(run_id, best_model, X_test, y_test, labels_names)

                # Save Model
                #dump(best_model, f"models/model_{run_id}.pkl") ////// por tema espacio no lo guardo ahora -- consultar docs

                end_time = time.time()
                exec_time = end_time - start_time

                # Create Log:
                log_data(run_id, archivo_excel, data, name, size, model, c, best_params, report, acc_global, std_global, exec_time, timestamp, type_id, balance)
