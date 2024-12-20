import pandas as pd
import logging
from pathlib import Path
import joblib
import click
from dotenv import find_dotenv, load_dotenv


def load_model(model_filepath):
    logger = logging.getLogger(__name__)
    logger.info(f'Loading model from {model_filepath}')
    return joblib.load(model_filepath)

def load_vectorizer(vectorizer_filepath):
    logger = logging.getLogger(__name__)
    logger.info(f'Loading vectorizer from {vectorizer_filepath}')
    return joblib.load(vectorizer_filepath)

def predict_spam(text, model, vectorizer):
    text_features = vectorizer.transform([text])
    
    prediction = model.predict(text_features)
    probability = model.predict_proba(text_features)
    
    return prediction[0], probability[0]

def main(model_filepath, vectorizer_filepath):
    logger = logging.getLogger(__name__)
    
    model = load_model(model_filepath)
    vectorizer = load_vectorizer(vectorizer_filepath)

    example_texts = [
        "URGENT! You have won a prize of £1000",
        "Hi, when are you coming home for dinner?",
    ]
    
    for text in example_texts:
        prediction, probability = predict_spam(text, model, vectorizer)
        logger.info(f'\nText: {text}')
        logger.info(f'Prediction: {"Spam" if prediction == 1 else "Not Spam"}')
        logger.info(f'Probability: Spam: {probability[1]:.2f}, Not Spam: {probability[0]:.2f}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    model_filepath = 'models/random_forest_spam.joblib'
    vectorizer_filepath = 'data/interim/tfidf_vectorizer.joblib'
    
    main(model_filepath, vectorizer_filepath)