# generate_data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_dummy_data():
    np.random.seed(0)
    data_size = 1000  # number of data points

    # Generate random RGB values
    rgb_data = np.random.randint(0, 256, size=(data_size, 3))

    # Define emotions
    emotions = [
        'happy', 'joyful', 'elated', 'content', 'proud', 
        'excited', 'amused', 'hopeful', 'inspired', 'grateful',
        'sad', 'angry', 'frustrated', 'anxious', 'depressed', 
        'melancholy', 'gloomy', 'disappointed', 'bitter', 'fearful'
    ]

    # Randomly assign emotion labels
    emotion_data = np.random.choice(emotions, size=(data_size,))

    # Create a DataFrame
    df = pd.DataFrame(rgb_data, columns=['rgb1', 'rgb2', 'rgb3'])
    df['emotion'] = emotion_data

    # Normalize RGB data
    scaler = MinMaxScaler()
    df[['rgb1', 'rgb2', 'rgb3']] = scaler.fit_transform(df[['rgb1', 'rgb2', 'rgb3']])

    # Save to CSV
    df.to_csv('dummy_data.csv', index=False)

if __name__ == "__main__":
    create_dummy_data()
