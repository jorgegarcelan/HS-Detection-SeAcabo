# ==============================================
# VERSION 1 POR JORGE GARCELÁN GÓMEZ - 12/12/23
# VERSION 2 POR JORGE GARCELÁN GÓMEZ - 07/04/24
# ==============================================

# Importación de librerías
from EXIST_utils import *
from EXIST_load_data import *
from EXIST_process_data import *
from EXIST_create_embeddings import *
from EXIST_split_data import *
from EXIST_create_model import *
from EXIST_evaluate_model import *
from EXIST_run_to_excel import *

from joblib import dump
import time
from datetime import datetime
from pysentimiento.preprocessing import preprocess_tweet

# ======================= PARAMS =======================
archivo_excel = "'EXIST 2024'/EXIST2024_Training.xlsx"

data_train = "../EXIST2024_Tweets_Dataset/training/EXIST2024_training.json" 
data_dev = "../EXIST2024_Tweets_Dataset/dev/EXIST2024_dev.json"

type_id = "task_1" # ["task_1", "task_2", "task_3"]
balance = "None" # ["downsampling", "upsampling", "smote", "adasyn", "None"]
embedding_name = ["fasttext", "word2vec", "bow", "tfidf", "roberta", "beto", "bert-multi", "xlm-roberta-base"] #["fasttext", "word2vec", "bow", "tfidf", "custom", "roberta", "beto", "bert-multi", "xlm-roberta-base", "robertuito"]
embedding_size = [500] #[100, 500] 
models = ["random_forest", "logistic_regression", "xgboost","mlp", "naive_bayes"] #["random_forest", "logistic_regression", "svc", "xgboost", "mlp", "naive_bayes"]
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
                df_train, X_train, y_train = embedding_data(df_train, name, size)
                df_dev, X_dev, y_dev = embedding_data(df_dev, name, size)

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
