import openai
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import time
import json
import configparser
import uuid
from datetime import datetime
import os

from process_data import *
from model import *
from eval_model import *
from run_to_excel import *

# ============================= FUNCTIONS =============================
def init(type_class, context, prompt, model):
    # Genera un UUID
    run_id = str(uuid.uuid4())  
    print(f"ID de ejecución: {run_id}")
    print(f"Type: {type_class}")
    print(f"Context: {context}")
    print(f"Prompt: {prompt}")
    print(f"Model: {model}")
    return run_id


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# ============================= MAIN =============================

config = configparser.ConfigParser()
config.read('config.ini')

openai.api_key = config['DEFAULT']['OPENAI_API_KEY']


DATASET = pd.read_csv("C:/Users/jorge/Desktop/UNI/4-CUARTO/4-2-TFG/CODE/Gender-Bias/data/BBDD_SeAcabo.csv") #seacabo_2023.csv
MODELS = ["lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF", "lmstudio-ai/gemma-2b-it-GGUF", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "lmstudio-community/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"] #["lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF", "lmstudio-ai/gemma-2b-it-GGUF", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "lmstudio-community/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"] # ["gpt-4o-mini", "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF", "lmstudio-ai/gemma-2b-it-GGUF", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"] #"gpt-4-0613" #"gpt-4-turbo" # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo || "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"||"lmstudio-ai/gemma-2b-it-GGUF"||"TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
TYPES = ["insultos"] # ["analisis_general", "contenido_negativo", "insultos"]
ARCHIVO_EXCEL = "Training_GenAI.xlsx"
TEMPERATURES = [0.1, 0.5]
PREPROCCESED_DATA = False

for TYPE in TYPES:

    # load context and prompt files
    context_files = [f"context/{file}" for file in os.listdir("context")]
    prompt_files = [f"prompts/{TYPE}/{file}" for file in os.listdir(f"prompts/{TYPE}")]

    print(f"{context_files=}")
    print(f"{prompt_files=}")

    for context_path in context_files:
        CONTEXT = read_file(context_path)

        for prompt_path in prompt_files:
            PROMPT = read_file(prompt_path)

            for MODEL in MODELS:
                for temp in TEMPERATURES:
                    print("\n")
                    print("-"*80)
                    # Run ID
                    run_id = init(type_class=TYPE, context=context_path, prompt=prompt_path, model=MODEL)

                    # Start timer
                    start_time = time.time() 
                    timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

                    # load data
                    #dataset = pd.read_csv(f"C:/Users/jorge/Desktop/UNI/4-CUARTO/4-2-TFG/CODE/Gender-Bias/data/preprocessed_data/preprocessed_data_{TYPE}.csv")

                    # Process Data (only if preprocessed data is not loaded)
                    dataset = process_data(df=DATASET, type_col=TYPE)

                    # Classify 
                    if TYPE=="analisis_general":
                        pred_column = "Predicted AG"

                        #dataset = predict_AG(dataset, PROMPT, MODEL)

                        # Prediction via OpenAI
                        if "gpt" in MODEL:
                            res_AG, discarded_indices = predict_gpt(dataset, PROMPT, MODEL, pred_column)
                        # Prediction via LM Studio
                        else:
                            res_AG, discarded_indices = predict_lmstudio(dataset, CONTEXT, PROMPT, MODEL, pred_column, temp, PREPROCCESED_DATA)

                        # Save the dataset to a CSV file
                        output_csv_path = f"processed_dataset/processed_dataset_{run_id}.csv"
                        dataset.to_csv(output_csv_path, index=False)

                        # Filter out discarded indices
                        if discarded_indices:
                            filtered_dataset = dataset.drop(discarded_indices).reset_index(drop=True)
                        else:
                            filtered_dataset = dataset

                        y_pred = filtered_dataset[pred_column]
                        y_test = filtered_dataset['Análisis General']


                        # Convert y_test to 0 for "Comentario Positivo" and 1 for "Comentario Negativo"
                        y_test = y_test.map({'Comentario Positivo': 0, 'Comentario Negativo': 1})
                        labels_names = [0, 1]
                        

                        # Get unique classes and number of total classes in y_pred and y_test
                        unique_classes_test = y_test.unique()
                        unique_classes_pred = y_pred.unique()
                        num_classes_test = len(unique_classes_test)
                        num_classes_pred = len(unique_classes_pred)

                        # Print class names and number of classes
                        print(f"Classes in y_test: {unique_classes_test}, Total: {num_classes_test}")
                        print(f"Classes in y_pred: {unique_classes_pred}, Total: {num_classes_pred}")

                        
                        # Eval
                        acc_global, std_global, report = eval_data(run_id, y_pred, y_test, labels_names)
                        
                        
                    if TYPE=="contenido_negativo":
                        #classify_CNeg(dataset=DATASET, prompt=PROMPT, model=MODEL)

                        pred_column = "Predicted CNeg"

                        # Prediction via OpenAI
                        if "gpt" in MODEL:
                            res_CNeg, discarded_indices = predict_gpt(dataset, PROMPT, MODEL, pred_column)
                        # Prediction via LM Studio
                        else:
                            res_CNeg, discarded_indices = predict_lmstudio(dataset, CONTEXT, PROMPT, MODEL, pred_column, temp, PREPROCCESED_DATA)

                        # Save the dataset to a CSV file
                        output_csv_path = f"processed_dataset/processed_dataset_{run_id}.csv"
                        dataset.to_csv(output_csv_path, index=False)

                        # Filter out discarded indices
                        if discarded_indices:
                            filtered_dataset = dataset.drop(discarded_indices).reset_index(drop=True)
                        else:
                            filtered_dataset = dataset

                        y_pred = filtered_dataset[pred_column]
                        y_test = filtered_dataset['Contenido Negativo']

                        # Convert y_test to 0 for "Comentario Positivo" and 1 for "Comentario Negativo"
                        y_test = y_test.map({"Desprestigiar Víctima": 0, "Desprestigiar Acto": 1, "Insultos": 2, "Desprestigiar Deportista Autora": 3})

                        labels_names = [0, 1, 2, 3]

                        # Get unique classes and number of total classes in y_pred and y_test
                        unique_classes_test = y_test.unique()
                        unique_classes_pred = y_pred.unique()
                        num_classes_test = len(unique_classes_test)
                        num_classes_pred = len(unique_classes_pred)

                        # Print class names and number of classes
                        print(f"Classes in y_test: {unique_classes_test}, Total: {num_classes_test}")
                        print(f"Classes in y_pred: {unique_classes_pred}, Total: {num_classes_pred}")

                        
                        # Eval
                        acc_global, std_global, report = eval_data(run_id, y_pred, y_test, labels_names)

                    if TYPE=="insultos":
                        #classify_Ins(dataset=DATASET, prompt=PROMPT, model=MODEL)

                        pred_column = "Predicted I"

                        # Prediction via OpenAI
                        if "gpt" in MODEL:
                            res_I, discarded_indices = predict_gpt(dataset, PROMPT, MODEL, pred_column)
                        # Prediction via LM Studio
                        else:
                            res_I, discarded_indices = predict_lmstudio(dataset, CONTEXT, PROMPT, MODEL, pred_column, temp, PREPROCCESED_DATA)

                        # Save the dataset to a CSV file
                        output_csv_path = f"processed_dataset/processed_dataset_{run_id}.csv"
                        dataset.to_csv(output_csv_path, index=False)

                        # Filter out discarded indices
                        if discarded_indices:
                            filtered_dataset = dataset.drop(discarded_indices).reset_index(drop=True)
                        else:
                            filtered_dataset = dataset

                        y_pred = filtered_dataset[pred_column]
                        y_test = filtered_dataset['Insultos']

                        # Convert y_test to 0 for "Comentario Positivo" and 1 for "Comentario Negativo"
                        y_test = y_test.map({"Sexistas/misóginos": 0, "Genéricos": 1, "Deseo de Dañar": 2})

                        labels_names = [0, 1, 2]

                        # Get unique classes and number of total classes in y_pred and y_test
                        unique_classes_test = y_test.unique()
                        unique_classes_pred = y_pred.unique()
                        num_classes_test = len(unique_classes_test)
                        num_classes_pred = len(unique_classes_pred)

                        # Print class names and number of classes
                        print(f"Classes in y_test: {unique_classes_test}, Total: {num_classes_test}")
                        print(f"Classes in y_pred: {unique_classes_pred}, Total: {num_classes_pred}")

                        
                        # Eval
                        acc_global, std_global, report = eval_data(run_id, y_pred, y_test, labels_names)


                    # Record time taken
                    end_time = time.time() 

                    # Compute total time taken
                    exec_time = end_time - start_time

                    # Log run to excel
                    log_data(run_id=run_id, 
                            archivo_excel=ARCHIVO_EXCEL, 
                            data=DATASET,
                            context=context_path,
                            prompt=prompt_path,
                            temperature=temp,
                            model=MODEL, 
                            type_id=TYPE,
                            acc_global=acc_global, 
                            std_global=std_global,
                            report=report, 
                            exec_time=exec_time, 
                            timestamp=timestamp,
                            discarded_indices=discarded_indices)
                    
                    print("-"*80)




