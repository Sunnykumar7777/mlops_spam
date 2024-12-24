# ML Ops (Spam Detection)

This project implements an end-to-end machine learning pipeline with MLOps best practices, including data versioning (DVC), model training, model tracking, and deployment.

## Prerequisites

- Conda (24.11.1 or higher)
- Python (3.12.3 or higher)

## Installation

1. First, install Anaconda from [Anaconda's official website](https://www.anaconda.com/download)

2. conda install cookiecutter

3. Clone this repository using cookiecutter:
   - cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science


## Extension for VS-Code

- DVC (Data Version Control) 
- Git 

## Initialize dvc 
```
dvc init
```

## Running the pipeline
```
dvc exp run
```

## DVC commands
```
dvc init # to initialize dvc repo
dvc add models # add models folder
dvc status # to check status
dvc remote add -d myremote cloud # include cloud experiments
dvc commit # commit the changes
dvc push # push the changes
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
    └── dvc.yaml           <- to create and store model process pipeline 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
   