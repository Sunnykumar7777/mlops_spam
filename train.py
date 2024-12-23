import random
import sys
import yaml
from dvclive import Live
import re


def accurracy_score(file_path=r'.\models\predictions.txt'):
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


with Live(save_dvc_exp=True) as live:
    trains_params = yaml.safe_load(open('params.yaml'))['train']
    epochs = trains_params['epochs']
    live.log_param("epochs", epochs)
    scores = accurracy_score()
    for epoch in range(epochs):
        live.log_metric("train/default_accuracy", scores.get('default'))
        live.log_metric("train/gb_accuracy", scores.get('gb', 0))
        live.log_metric("train/loss", epochs - epoch - random.random())
        live.next_step()