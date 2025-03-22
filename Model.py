import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split                         
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ----------------------------------------------------------------------------------------------------------
# *) Grouping Types (['types'] --> reliable or fake)
# ----------------------------------------------------------------------------------------------------------
"""
input: CSV with multiple types under ['type'] & name of the new updated CSV
returns: CSV with only 2 types of ['type'] = ['reliable'] or ['fake']
"""
def group_types(csv, name):
    df = pd.read_csv(csv)
    fake_types = ['conspiracy', 'unreliable', 'junksci', 'clickbait']
    rows_to_drop = []

    # Iterate over rows and update the ['type'] column:
    for index, row in df.iterrows():
        current = str(row['type']).strip()
        if current == 'fake' or current == 'reliable':
            continue
        elif current in fake_types:
            df.loc[index, 'type'] = 'fake'
        else:
            rows_to_drop.append(index) 
    # Remove all other rows that isn't [reliable] or in 'fake_types'
    df.drop(index = rows_to_drop, inplace=True)

    # Save & print head
    print("First 20 rows of ['type']: ")
    print(df["type"].head(20))
    df.to_csv(f"{name}.csv", index=False)
    return df


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


def train_model(X_train, X_val, X_test, y_train, y_val, y_test):
    # VECTORIZE 10000 most frequent words
    vectorizer = CountVectorizer(max_features=10000)    # CountVectorizer turns text into vectors of word counts
    X_train_vec = vectorizer.fit_transform(X_train)     # Vectorize the X_train, X_val & X_test
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # TRAIN MODEL (Logistic-Regression)
    model = LogisticRegression(max_iter=1000, C=1.0)   # C, can be tweaked
    model.fit(X_train_vec, y_train)                    # (small C = less overfitting, big C = risk of overfitting)

    # EVALUATE ON TEST-SET
    y_pred = model.predict(X_test_vec)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))