from utils import *
from process_data import *

import numpy as np
from transformers import BertTokenizer, DistilBertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel, XLMRobertaModel, AutoTokenizer
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def embedding_data(df, type_id, embedding_name, embedding_size):

    if embedding_name == "fasttext":
        # Create a FastText model
        sentences = df['full_text_processed'].str.split().tolist()
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

        df['full_text_filtered'] = df['full_text_processed'].apply(text_to_vector)

        X = np.stack(df['full_text_filtered'].values)
        y = df['label'].values

    elif embedding_name == "word2vec":
        # Create a FastText model
        sentences = df['full_text_processed'].str.split().tolist()
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


        df['full_text_filtered'] = df['full_text_processed'].apply(text_to_vector)

        X = np.stack(df['full_text_filtered'].values)
        y = df['label'].values


    elif embedding_name == "bow":
        # Creating the bag of Word Model
        model = CountVectorizer(max_features = 5000, ngram_range=(1, 5))
        X = model.fit_transform(df['full_text_processed']).toarray()
        y = df['label'].values

    elif embedding_name == "tfidf":
        # Creating the TF-IDF model
        model = TfidfVectorizer(max_features=5000, ngram_range=(1, 5))
        X = model.fit_transform(df['full_text_processed']).toarray()
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
            
        df['full_text_filtered'] = df['full_text_processed'].apply(text_to_vector)

        X = np.stack(df['full_text_filtered'].values)
        y = df['label'].values

    elif embedding_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne') 
        model = RobertaModel.from_pretrained("PlanTL-GOB-ES/roberta-large-bne") 

        embeddings = get_roberta_embeddings(df['full_text_processed'].tolist(), tokenizer, model)
        
        X = embeddings
        y = df['label'].values

    elif embedding_name == "beto-cased":
        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased') #beto
        model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased") #beto

        embeddings = get_roberta_embeddings(df['full_text_processed'].tolist(), tokenizer, model)
        
        X = embeddings
        y = df['label'].values
    
    elif embedding_name == "beto-uncased":
        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased') #beto
        model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased") #beto

        embeddings = get_roberta_embeddings(df['full_text_processed'].tolist(), tokenizer, model)
        
        X = embeddings
        y = df['label'].values

    elif embedding_name == "bert-multi":
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained('bert-base-multilingual-cased')

        embeddings = get_roberta_embeddings(df['full_text_processed'].tolist(), tokenizer, model)
        
        X = embeddings
        y = df['label'].values

    elif embedding_name == "xlm-roberta-base":
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")

        embeddings = get_roberta_embeddings(df['full_text_processed'].tolist(), tokenizer, model)
        
        X = embeddings
        y = df['label'].values

    elif embedding_name == "robertuito":
        tokenizer = AutoTokenizer.from_pretrained('pysentimiento/robertuito-base-cased')
        embeddings =  tokenizer(df['full_text'].tolist(), truncation=True, padding=True, max_length=512)
        X = embeddings['input_ids']
        y = df['label'].values

    elif embedding_name == "text-embedding-3-large":
        #embeddings = pd.read_csv('C:/Users/jorge/Desktop/UNI/4-CUARTO/4-2-TFG/CODE/Gender-Bias/OpenAI/seacabo_embeddings.csv')
        # Filter by type
        #embeddings, _ = process_data(embeddings, type_id, balance="None")
        df['embeddings'] = df['embeddings'].apply(lambda x : convert_embedding_to_list(x)) 

        X = np.stack(df['embeddings'].values).astype(np.float32)
        # Ensure that X is a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y = df['label'].values.astype(np.float32)

    # Print dimensions of X
    print(f"Dimensions of X: {X.shape}")

    # Add the additional features to your embeddings
    additional_features = df[['mention_count', 'view_count_scaled', 'tweet_length', 'num_adjectives']].values.astype(np.float32)

    # Print dimensions of additional_features
    print(f"Dimensions of additional_features: {additional_features.shape}")

    # Assuming X is your text embeddings
    X = np.hstack((X, additional_features))

    # Print dimensions of the final concatenated array
    print(f"Dimensions of X after concatenation: {X.shape}")

    ## SAVE MODEL when running best model -> future (https://radimrehurek.com/gensim/models/fasttext.html)
    #dump(model, f"embeddings/embedding_{run_id}.pkl")


    return df, X, y
