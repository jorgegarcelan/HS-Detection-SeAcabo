from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def model_data(X_train, y_train, model, cv):

    if model == "random_forest":
        # Define Random Forest Classifier
        model = RandomForestClassifier(random_state = 0, class_weight='balanced')

        # Define the hyperparameters and their possible values
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
            }

    elif model == "logistic_regression":
        # Define Logistic Regression Model
        model = LogisticRegression(class_weight='balanced')

        # Define the hyperparameters and their possible values
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [50, 100, 200]
        }

    elif model == "svc":
        # Define SVC Model
        model = SVC(class_weight='balanced')

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