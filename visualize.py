import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_word_cloud(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Concatenate all text data in the specified column
    text = " ".join(df[column_name].dropna().astype(str))

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()