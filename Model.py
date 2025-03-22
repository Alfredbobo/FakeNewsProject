import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------------------------------------
# *) Split Data (default: 80% / 10% / 10%)
# ----------------------------------------------------------------------------------------------------------
def split_data(csv):
    df = pd.read_csv(csv)  

    df = df.dropna(subset=["content", "type"])     # Resolves NaN issues

    X = df["content"]        # INPUT (tokenized text in ['content'])
    y = df["type"]           # TARGET (label: fake, unreliable, etc.)

    # FIRST SPLIT: (80% train, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SECOND SPLIT: (temp → 10% val, 10% test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test