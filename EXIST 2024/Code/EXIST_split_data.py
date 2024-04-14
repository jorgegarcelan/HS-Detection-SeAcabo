from EXIST_utils import *

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

def split_data(X, y, balance):

    # Splitting the dataset into the Training set and Test set with stratify=y so training and test sets have a similar distribution of classes as original dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, stratify=y)

    # Combinar X_train y y_train para upsampling
    # upsample after split so train examples are not repeated in test

    if balance == "upsampling":
        print(f"{balance=}")
        # Convertir X_train y y_train a DataFrame para facilitar el manejo
        df_train = pd.DataFrame(X_train)
        df_train['label'] = y_train

        # Aplicar upsampling
        df_train_upsampled = upsample(df_train)

        # Separar las características y las etiquetas después del upsampling
        y_train_upsampled = df_train_upsampled['label'].values
        X_train_upsampled = df_train_upsampled.drop('label', axis=1).values

        # Imprimir el soporte de las clases después del upsampling
        print(df_train_upsampled['label'].value_counts())

        # Asegurar que X_train y y_train estén actualizados
        X_train, y_train = X_train_upsampled, y_train_upsampled

    # Aplicar SMOTE solo si se especifica
    if balance == "smote":
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("Después de aplicar SMOTE:")
        print(pd.Series(y_train).value_counts())

    # Aplicar ADASYN solo si se especifica
    if balance == "adasyn":
        ada = ADASYN(random_state=42)
        X_train, y_train = ada.fit_resample(X_train, y_train)
        print("Después de aplicar ADASYN:")
        print(pd.Series(y_train).value_counts())

    return X_train, X_test, y_train, y_test