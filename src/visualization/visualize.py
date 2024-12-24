"""
This module creates visualizations for spam classification data analysis:
- Data balance between spam and non-spam classes
- Top frequent words in spam and non-spam messages
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import yaml
import click
import re
from collections import Counter
from dotenv import find_dotenv, load_dotenv
import dvclive



def load_params(params_path):
   """Load parameters from YAML config file."""
   with open(params_path, 'r') as f:
       params = yaml.safe_load(f)
   return params

def plot_data_balance(df, output_path, target_column):
   """Create bar plot showing distribution of spam vs non-spam messages."""
   value_counts = df[target_column].value_counts()
   
   plt.figure(figsize=(6, 4))
   value_counts.plot(kind='bar')
   plt.title('Spam and Not-Spam')
   plt.xlabel('Value')
   plt.ylabel('Count')
   plt.xticks(rotation=0)
   
   for i, count in enumerate(value_counts):
       plt.text(i, count, str(count), ha='center', va='bottom')
   
   plt.tight_layout()
   plt.savefig(str(output_path / 'data_balance.png'))
   plt.close()

# Common stop words to filter out from analysis
stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                'for', 'of', 'with', 'by','is', 'i', 'how', 'it'])

def top_words(df, column, top_n=10, label=None):
   """Find most frequent words in text data, excluding stop words."""
   if label is not None:
       df = df[df['v1'] == label]
   
   all_text = ' '.join(df[column].fillna('').astype(str))
   all_text = all_text.lower()
   words = re.findall(r'\w+', all_text)
   filtered_words = [word for word in words if word not in stop_words]
   word_counts = Counter(filtered_words)
   top_words = word_counts.most_common(top_n)
   return top_words

def plot_top_words(df, output_path, label, title):
   """
   Create bar plot of most frequent words.
   
   Args:
       df (pd.DataFrame): Input dataframe with text data
       output_path (Path): Directory to save plot
       label (int): Label value to analyze (0 for non-spam, 1 for spam)
       title (str): Plot title
   """
   top_words_data = top_words(df, 'v2', top_n=20, label=label)
   
   plt.figure(figsize=(8, 6))
   colors = cm.rainbow(np.linspace(0, 1, len(top_words_data)))
   plt.bar(
       [word[0] for word in top_words_data],
       [word[1] for word in top_words_data],
       color=colors
   )
   plt.title(title, fontsize=15)
   plt.xlabel('Words', fontsize=12)
   plt.ylabel('Frequency', fontsize=12)
   plt.xticks(rotation=45, ha='right')
   plt.tight_layout()
   
   filename = f'top_words_{"spam" if label==1 else "non_spam"}.png'
   plt.savefig(str(output_path / filename))
   plt.close()

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
   """
   Main function to generate visualizations.
   
   Args:
       input_filepath (str): Path to input data CSV
       output_filepath (str): Directory to save visualization plots
       
   The function:
   1. Loads data and parameters
   2. Creates plot of class distribution
   3. Creates plots of top words for spam and non-spam messages
   """
   logger = logging.getLogger(__name__)
   
   params_path = Path(__file__).resolve().parents[2] / 'params.yaml'
   params = load_params(params_path)
   
   logger.info(f'Loading data from {input_filepath}')
   df = pd.read_csv(input_filepath)
   
   output_path = Path(output_filepath)
   output_path.mkdir(parents=True, exist_ok=True)
   
   logger.info('Generating data balance plot')
   plot_data_balance(df, output_path, params['data']['target_column'])
   
   logger.info('Generating top words plots')
   plot_top_words(df, output_path, label=0, title='Top 20 Repeated Words (Non-Spam)')
   plot_top_words(df, output_path, label=1, title='Top 20 Repeated Words (Spam)')

if __name__ == '__main__':
   log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   logging.basicConfig(level=logging.INFO, format=log_fmt)

   project_dir = Path(__file__).resolve().parents[2]
   load_dotenv(find_dotenv())

   main()
