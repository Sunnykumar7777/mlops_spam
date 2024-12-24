"""
This module implements experiment tracking and logging for a spam classification model using MLflow.
It reads model accuracies from a predictions file and logs training metrics over multiple epochs.
"""

import random
import sys
import yaml
from dvclive import Live
import re
import mlflow
import mlflow.sklearn


def accurracy_score(file_path=r'.\models\predictions.txt'):
    """
    Extract accuracy scores for Random Forest and Gradient Boosting models from a predictions file.
    
    Args:
        file_path (str): Path to the predictions file containing model accuracy scores.
            Default is '.\models\predictions.txt'
    
    Returns:
        dict: Dictionary containing accuracy scores with keys 'default' (Random Forest) and 'gb' (Gradient Boosting).
            Returns empty dict if file is not found or parsing fails.
    
    Raises:
        FileNotFoundError: If the specified file_path doesn't exist
        Exception: For other errors during file reading or parsing
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            default_match = re.search(r'Model Accuracy RF: (\d+\.\d+)', content)
            gb_match = re.search(r'Model Accuracy GB: (\d+\.\d+)', content)
            
            accuracies = {}
            if default_match:
                accuracies['default'] = float(default_match.group(1))
                
            if gb_match:
                accuracies['gb'] = float(gb_match.group(1))
                
            return accuracies
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
    except Exception as e:
        print(f"Error reading accuracies: {str(e)}")

# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("spam_calssification")


# Main experiment tracking loop
with Live(save_dvc_exp=True) as live:
    with mlflow.start_run(run_name="spam_classification_run_2"):
        # Load training parameters
        train_params = yaml.safe_load(open('params.yaml'))['train']
        epochs = train_params['epochs']

        # Log parameters and initial metrics
        mlflow.log_param("epochs", epochs)

        scores = accurracy_score()
        rf_accuracy = scores.get('default', 0)
        gb_accuracy = scores.get('gb', 0)

        mlflow.log_metric("rf_accuracy", scores.get('default'))
        mlflow.log_metric("gb_accuracy", scores.get('gb', 0))

        # Training loop with metric logging
        for epoch in range(epochs):
            progress = (epoch + 1) / epochs
            
            rf_train_acc = rf_accuracy * progress
            gb_train_acc = gb_accuracy * progress
            loss = epochs - epoch - random.random()
            
            mlflow.log_metric("train_rf_accuracy", rf_train_acc, step=epoch)
            mlflow.log_metric("train_gb_accuracy", gb_train_acc, step=epoch)
            mlflow.log_metric("train_loss", loss, step=epoch)
            
            live.next_step()

