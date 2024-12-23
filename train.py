import random
import sys
import yaml
from dvclive import Live
import re

def read_accuracy(file_path=r'.\models\predictions.txt'):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            match = re.search(r'Model Accuracy RF: (\d+\.\d+)', content)
            if match:
                return float(match.group(1))
            else:
                raise ValueError("Not Found")
    except Exception as e:
        print(f"Error reading accuracy: {str(e)}")
        sys.exit(1)

with Live(save_dvc_exp=True) as live:
    trains_params = yaml.safe_load(open('params.yaml'))['train']
    epochs = trains_params['epochs']
    live.log_param("epochs", epochs)
    accuracy = read_accuracy()
    for epoch in range(epochs):
        live.log_metric("train/accuracy", accuracy)
        live.log_metric("train/loss", epochs - epoch - random.random())
        live.next_step()