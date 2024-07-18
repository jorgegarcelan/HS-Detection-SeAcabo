import pandas as pd

def load_data(dataset_name, embedding_name):
    # OpenAI Embedding
    if embedding_name == "text-embedding-3-large":
        df = pd.read_csv('C:/Users/jorge/Desktop/UNI/4-CUARTO/4-2-TFG/CODE/Gender-Bias/OpenAI/seacabo_embeddings.csv')
    else:
        df = pd.read_csv(dataset_name)
    
    return df