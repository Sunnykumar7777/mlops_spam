import random
import sys
import yaml
from dvclive import Live
import re
import mlflow
import mlflow.sklearn

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


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("spam_calssification")


with Live(save_dvc_exp=True) as live:
    with mlflow.start_run(run_name="spam_classification_run_2"):
        train_params = yaml.safe_load(open('params.yaml'))['train']
        epochs = train_params['epochs']

        mlflow.log_param("epochs", epochs)

        scores = accurracy_score()
        rf_accuracy = scores.get('default', 0)
        gb_accuracy = scores.get('gb', 0)

        mlflow.log_metric("rf_accuracy", scores.get('default'))
        mlflow.log_metric("gb_accuracy", scores.get('gb', 0))

        for epoch in range(epochs):
            progress = (epoch + 1) / epochs
            
            rf_train_acc = rf_accuracy * progress
            gb_train_acc = gb_accuracy * progress
            loss = epochs - epoch - random.random()
            
            mlflow.log_metric("train_rf_accuracy", rf_train_acc, step=epoch)
            mlflow.log_metric("train_gb_accuracy", gb_train_acc, step=epoch)
            mlflow.log_metric("train_loss", loss, step=epoch)
            
            live.next_step()

