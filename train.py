"""
This module implements experiment tracking and metric logging using DVC (Data Version Control).
It reads model accuracies from a predictions file and logs training metrics over multiple epochs.
"""

import random
import sys
import yaml
from dvclive import Live
import re

def accurracy_score(file_path=r'.\models\predictions.txt'):
    """
    Extract accuracy scores for Random Forest and Gradient Boosting models from a predictions file.
    
    Args:
        file_path (str): Path to the predictions file containing model accuracy scores.
            Default is '.\models\predictions.txt'
    
    Returns:
        dict: Dictionary containing accuracy scores with keys 'default' (RF) and 'gb' (Gradient Boosting).
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


# Main DVC experiment tracking loop
with Live(save_dvc_exp=True) as live:
    # Load training parameters from config
    trains_params = yaml.safe_load(open('params.yaml'))['train']
    epochs = trains_params['epochs']

    # Log training parameters
    live.log_param("epochs", epochs)

    # Get model accuracy scores
    scores = accurracy_score()

    # Training loop with metric logging
    for epoch in range(epochs):
        live.log_metric("train/default_accuracy", scores.get('default'))
        live.log_metric("train/gb_accuracy", scores.get('gb', 0))
        live.log_metric("train/loss", epochs - epoch - random.random())
        live.next_step()