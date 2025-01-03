# ML Ops (Spam Detection)

This project implements an end-to-end machine learning pipeline with MLOps best practices, including data versioning (DVC), model training, model tracking, and deployment.

## Prerequisites

- Conda (24.11.1 or higher)
- Python (3.12.3 or higher)

## Installation

1. First, install Anaconda from [Anaconda's official website](https://www.anaconda.com/download)

2. conda install cookiecutter

3. Clone this repository using cookiecutter:
    ```
    cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
    ```
   - It creates the structure for ML-model project.


## Extension for VS-Code

- Git 
- DVC (Data Version Control) 

## Create the Git repo (to track files and changes)
1. Initialize git repo (Using terminal)
    ```
    git init
    ```
2. Add a remote repository to local Git
    ```
    git remote add origin <URL>
    ```
3. Add changes
    ```
    git add .
    ```
4. Commit changes
    ```
    git commit -m "Initial commit"
    ```
5. Push changes
    ```
    git push origin master
    ```
Alternative:
- Install git extension on VS-code and perform same action using Git UI.

## How to connect DVC
1. First install dvc extension from vscode Extensions.
2. Initialize DVC
    ```
    dvc init
    ```
3. Add data directory (it creates a data.dvc to track data)
    ```
    dvc add data
    ```
4. Add Models directory
    ```
    dvc add models
    ```
5. Managing Data Versions: Add Cloud directory (to upload data to remote storage)
    ```
    dvc remote add -d myremote cloud
    ```
6. Commit the changes
    ``` 
    dvc commit
    ```
7. Push the changes
    ```
    dvc push
    ```
8. Also view DVC experiments on VS code by clicking on DVC > Show Experiments


## Running the pipeline
First create stages on dvc.yaml then run the command:
```
dvc exp run
```

## How to connect ML-Flow
1. Install ML-flow(2.19.0 or higher) using pip:
    ```
    pip install mlflow
    ```
2. Then run mlflow_train.py using command:
    ```
    mlflow server --host 0.0.0.0 --port 5000
    ```
3. Then also run mlflow_train.py in new terminal:
    ```
    python mlflow_train.py
    ```
4. Check all records running on:
   - http://localhost:5000


## Add .gitignore Configuration
```
/data
/models
/cloud
/dvclive
/mlruns
*.joblib
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    |── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    |
    |── train.py           <- Script to check dvc experiments
    |
    |── mlflow_train.py    <- Script to check mlflow experiments
    |
    |── nltk_install.py    <- Script to install nltk required modules for text preprocessing
    |
    |── params.yaml        <- to store model configuration
    |
    |── data.dvc           <- to track and manage data versions.
    |
    └── dvc.yaml           <- to create and store model process pipeline 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
   