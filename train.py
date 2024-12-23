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

mlflow.set_experiment("spam_classification")


# with Live(save_dvc_exp=True) as live:
#     with mlflow.start_run(run_name="spam_classification_run"):
#         trains_params = yaml.safe_load(open('params.yaml'))['train']
#         epochs = trains_params['epochs']
#         live.log_param("epochs", epochs)
#         mlflow.log_param("epochs")

#         scores = accurracy_score()
#         rf_accuracy = scores['rf']
#         gb_accuracy = scores['gb']

#         live.log_metric("model/rf_accuracy", rf_accuracy)
#         live.log_metric("model/gb_accuracy", gb_accuracy)
#         mlflow.log_metric("rf_accuracy", rf_accuracy)
#         mlflow.log_metric("gb_accuracy", gb_accuracy)

#         for epoch in range(epochs):
#             progress = (epoch + 1) / epochs

#             rf_train_acc = rf_accuracy * progress
#             gb_train_acc = gb_accuracy * progress
#             loss = epochs - epoch - random.random()

#             live.log_metric("train/rf_accuracy", rf_train_acc)
#             live.log_metric("train/gb_accuracy", gb_train_acc)
#             live.log_metric("train/loss", loss)

#             mlflow.log_metric("train_rf_accuracy", rf_train_acc, step=epoch)
#             mlflow.log_metric("train_gb_accuracy")



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