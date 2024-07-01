import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def main():
    input_data_path = os.path.join('/opt/ml/processing/input', 'data.csv')
    df = pd.read_csv(input_data_path)

    # Preprocessing steps
    df['feature'] = StandardScaler().fit_transform(df[['feature']])  # Example feature scaling
    train, test = train_test_split(df, test_size=0.2)
    
    # Save the processed datasets
    train.to_csv('/opt/ml/processing/train/train.csv', index=False)
    test.to_csv('/opt/ml/processing/test/test.csv', index=False)

if __name__ == "__main__":
    main()
