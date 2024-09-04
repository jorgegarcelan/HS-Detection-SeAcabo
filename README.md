# Detection of Gender-Based Hate Speech against Women in Social Media via Natural Language Processing: An Analysis of the *#SeAcabó* Movement

[![License](https://img.shields.io/badge/license-Creative%20Commons%20Attribution%20%E2%80%93%20Non%20Commercial%20%E2%80%93%20Non%20Derivatives-blue)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)


## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Author](#author)
4. [License](#license)


## Project Overview

This repository contains the code and resources for the thesis titled **"Detection of Gender-Based Hate Speech against Women in Social Media via Natural Language Processing: An Analysis of the *#SeAcabó* Movement"**. The project focuses on analyzing hate speech directed at women on social media using various machine learning, deep learning, and generative AI techniques.

## Repository Structure

The repository is organized into three main directories, each containing scripts and notebooks specific to the type of models used:

### MachineLearning

- **`ML Experiments/`**: Contains Excel files with results from the Machine Learning experiments.
- **`EDA.ipynb`**: Jupyter notebook for Exploratory Data Analysis (EDA) to understand the dataset before model development.
- **`Plots.ipynb`**: Jupyter notebook for visualizing various metrics and results obtained from Machine Learning models.
- **`visualize_embeddings.ipynb`**: Jupyter notebook for visualizing embeddings generated from the dataset to understand the feature space.
- **`load_data.py`**: Script to load the dataset into the environment.
- **`process_data.py`**: Handles data preprocessing before model training.
- **`create_embeddings.py`**: Generates embeddings from the dataset for use in models.
- **`split_data.py`**: Splits the dataset into training and testing sets.
- **`create_model.py`**: Defines and configures the Machine Learning models.
- **`evaluate_model.py`**: Evaluates the performance of the trained models and provides metrics.
- **`run_to_excel.py`**: Exports model results to an Excel file for further analysis.
- **`train_ML.py`**: Integrates all functions and is used to train the Machine Learning models.
- **`utils.py`**: Contains utility functions that support various tasks in the Machine Learning workflow.

### DeepLearning

- **`DL Experiments/`**: Contains Excel files with results from the Deep Learning experiments.
- **`train_DL.ipynb`**: Jupyter notebook for training Deep Learning models.
- **`Plots.ipynb`**: Jupyter notebook for visualizing the results and performance metrics of Deep Learning models.

### GenerativeAI

- **`GenAI Experiments/`**: Contains Excel files with results from the Generative AI experiments.
- **`Plots.ipynb`**: Jupyter notebook for visualizing the results and performance metrics of Generative AI models.
- **`embeddings.ipynb`**: Jupyter notebook for generating and processing embeddings using the OpenAI API.
- **`process_data.py`**: Script for preprocessing data before feeding it into the Generative AI models.
- **`model.py`**: Defines and configures the Generative AI models.
- **`eval_model.py`**: Evaluates the performance of the trained Generative AI models.
- **`run_to_excel.py`**: Exports the results of the Generative AI models to an Excel file for further analysis.
- **`train_GenAI.py`**: Principal Python script for training Generative AI models.

## Author

- **Jorge Garcelán Gómez**
- Email: [100442062@alumnos.uc3m.es](mailto:100442062@alumnos.uc3m.es)
- Alternative Email: [jorgegarcelan@gmail.com](mailto:jorgegarcelan@gmail.com)

## License

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Cc_by-nc-nd_icon.svg/2560px-Cc_by-nc-nd_icon.svg.png" alt="Creative Commons Logo" width="150"/>

This work is licensed under Creative Commons **Attribution – Non Commercial – Non Derivatives**.
