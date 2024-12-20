import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import logging
from pathlib import Path

def top_words(df, column, top_n=20, label=None):
    if label is not None:
        text_data = ' '.join(df[df['v1'] == label][column]).lower().split()
    else:
        text_data = ' '.join(df[column]).lower().split()
    
    word_freq = pd.Series(text_data).value_counts()
    return list(zip(word_freq.index[:top_n], word_freq.values[:top_n]))

def plot_spam_distribution(df, save_path=None):
    logger = logging.getLogger(__name__)
    logger.info('Plotting spam distribution')
    
    value_counts = df['v1'].value_counts()
    
    plt.figure(figsize=(6, 4))
    value_counts.plot(kind='bar')
    plt.title('Spam and Not-Spam')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    for i, count in enumerate(value_counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f'Saved spam distribution plot to {save_path}')
    else:
        plt.show()
    plt.close()

def plot_top_words(df, label, save_path=None):
    logger = logging.getLogger(__name__)
    label_type = "Spam" if label == 1 else "Non-Spam"
    logger.info(f'Plotting top words for {label_type}')
    
    top_words_data = top_words(df, 'v2', top_n=20, label=label)
    
    plt.figure(figsize=(8, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(top_words_data)))
    
    plt.bar(
        [word[0] for word in top_words_data],
        [word[1] for word in top_words_data],
        color=colors
    )
    
    plt.title(f'Top 20 Repeated Words ({label_type})', fontsize=15)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f'Saved top words plot to {save_path}')
    else:
        plt.show()
    plt.close()

def main(data_filepath, figures_dir):
    logger = logging.getLogger(__name__)
    
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f'Loading data from {data_filepath}')
    df = pd.read_csv(data_filepath)
    
    plot_spam_distribution(
        df, 
        save_path=figures_path / 'spam_distribution.png'
    )
    
    plot_top_words(
        df, 
        label=0, 
        save_path=figures_path / 'top_words_non_spam.png'
    )
    
    plot_top_words(
        df, 
        label=1, 
        save_path=figures_path / 'top_words_spam.png'
    )

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    data_filepath = r'D:\spam_detection\spam_detection\data\raw\spam.csv'
    figures_dir = r'D:\spam_detection\spam_detection\reports\figures'
    
    main(data_filepath, figures_dir)