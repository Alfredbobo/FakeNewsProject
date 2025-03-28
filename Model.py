import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split                         
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack
import numpy as np
import os


# ----------------------------------------------------------------------------------------------------------
# *) Grouping Types (['types'] --> reliable or fake)
# ----------------------------------------------------------------------------------------------------------
"""
input: CSV with multiple types under ['type'] & name of the new updated CSV
returns: CSV with only 2 types of ['type'] = ['reliable'] or ['fake']
"""
def group_types(csv, name):
    df = pd.read_csv(csv)
    fake_types = ['conspiracy', 'unreliable', 'junksci', 'clickbait', 'hate']
    reliable_types = ['political']
    rows_to_drop = []

    # Iterate over rows and update the ['type'] column:
    for index, row in df.iterrows():
        current = str(row['type']).strip()
        if current == 'fake' or current == 'reliable':
            continue
        elif current in fake_types:
            df.loc[index, 'type'] = 'fake'
        elif current in reliable_types:
            df.loc[index, 'type'] = 'reliable'
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
    
    # Remove duplicates before splitting
    df = df.drop_duplicates(subset=["content"])

    X = df["content"]        # INPUT (tokenized text in ['content'])
    y = df["type"]           # TARGET (label: fake, unreliable, etc.)

    # FIRST SPLIT: (80% train, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SECOND SPLIT: (temp → 10% val, 10% test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# ----------------------------------------------------------------------------------------------------------
# *) TRAIN MODEL
# ----------------------------------------------------------------------------------------------------------
def train_model(X_train, X_val, X_test, y_train, y_val, y_test):
    # VECTORIZE 10000 most frequent words
    vectorizer = CountVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # TRAIN MODEL (Logistic Regression)
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    # EVALUATE ON TEST-SET
    y_pred = model.predict(X_test_vec)

    # Print report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["fake", "reliable"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["fake", "reliable"])
    disp.plot()
    plt.savefig("confusion_matrix_simple_model.png", dpi=300, bbox_inches="tight")

    return model, vectorizer

    
# ----------------------------------------------------------------------------------------------------------
# *) SPLIT & TRAIN (including MetaData ['domain'])
# ----------------------------------------------------------------------------------------------------------
def split_with_meta(csv):
    df = pd.read_csv(csv)
    df = df.dropna(subset=["content", "type", "domain"])

    # Remove duplicates before splitting
    df = df.drop_duplicates(subset=["content"])

    # Features and labels
    X = df[["content", "domain"]]
    y = df["type"]

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_with_meta(X_train, X_val, X_test, y_train, y_val, y_test):
    # Extract columns
    X_train_text = X_train["content"]
    X_train_domain = X_train["domain"]

    X_val_text = X_val["content"]
    X_val_domain = X_val["domain"]

    X_test_text = X_test["content"]
    X_test_domain = X_test["domain"]

    # VECTORIZE
    vectorizer = CountVectorizer(max_features=10000)
    X_train_text_vec = vectorizer.fit_transform(X_train_text)
    X_val_text_vec = vectorizer.transform(X_val_text)
    X_test_text_vec = vectorizer.transform(X_test_text)

    # Encode Domain using OneHotEncoding
    domain_encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_domain_vec = domain_encoder.fit_transform(X_train_domain.values.reshape(-1, 1))
    X_val_domain_vec = domain_encoder.transform(X_val_domain.values.reshape(-1, 1))
    X_test_domain_vec = domain_encoder.transform(X_test_domain.values.reshape(-1, 1))

    # Combine [content] and [domain] using hstack (creates a matrix)
    # hstack = Joins word-vectors and domain vectors side-by-side
    X_train_combined = hstack([X_train_text_vec, X_train_domain_vec])
    X_val_combined = hstack([X_val_text_vec, X_val_domain_vec])
    X_test_combined = hstack([X_test_text_vec, X_test_domain_vec])

    # Train model
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_combined, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test_combined)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))


def train_with_meta_and_extra_data(X_train, X_val, X_test, y_train, y_val, y_test):
    # Extract columns
    X_train_text = X_train["content"]
    X_train_domain = X_train["domain"]

    X_val_text = X_val["content"]
    X_val_domain = X_val["domain"]

    X_test_text = X_test["content"]
    X_test_domain = X_test["domain"]

    # VECTORIZE
    vectorizer = CountVectorizer(max_features=10000)
    X_train_text_vec = vectorizer.fit_transform(X_train_text)
    X_val_text_vec = vectorizer.transform(X_val_text)
    X_test_text_vec = vectorizer.transform(X_test_text)

    # Encode Domain using OneHotEncoding
    domain_encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_domain_vec = domain_encoder.fit_transform(X_train_domain.values.reshape(-1, 1))
    X_val_domain_vec = domain_encoder.transform(X_val_domain.values.reshape(-1, 1))
    X_test_domain_vec = domain_encoder.transform(X_test_domain.values.reshape(-1, 1))

    # Combine [content] and [domain] using hstack (creates a matrix)
    X_train_combined = hstack([X_train_text_vec, X_train_domain_vec])
    X_val_combined = hstack([X_val_text_vec, X_val_domain_vec])
    X_test_combined = hstack([X_test_text_vec, X_test_domain_vec])

    # Train model
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_combined, y_train)
    
    # Check overlap clearly:
    overlap = set(X_train["content"]).intersection(set(X_test["content"]))
    print(f"Overlap between train and test set: {len(overlap)} articles")

    
    # Evaluate
    y_pred = model.predict(X_test_combined)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))


def train_final_svm_tfidf(X_train, X_val, X_test, y_train, y_val, y_test):
    # Build pipeline with best parameters
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
        ("clf", LinearSVC(C=1, loss="squared_hinge", dual=True, max_iter=10000))
    ])

    # Fit on the full training data
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=["fake", "reliable"])

    # Display it
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["fake", "reliable"])
    disp.plot()
    plt.savefig("confusion_matrix_SVM.png", dpi=300, bbox_inches="tight")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))

    return pipeline


def evaluate_model_on_liar(pipeline, liar_df):
    X_liar = liar_df["content"]
    y_liar = liar_df["type"]

    y_pred = pipeline.predict(X_liar)

    print("\nPerformance on LIAR Dataset:")
    print(classification_report(y_liar, y_pred))

    f1 = f1_score(y_liar, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_liar, y_pred)
    print("Accuracy:", round(acc, 3))

    # ➕ Add confusion matrix
    cm = confusion_matrix(y_liar, y_pred, labels=["fake", "reliable"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["fake", "reliable"])
    disp.plot()
    plt.savefig("confusion_matrix_LAIR.png", dpi=300, bbox_inches="tight")

def evaluate_model_on_liar_simple(model, vectorizer, liar_df):
    # Prepare LIAR data
    X_liar = liar_df["content"]
    y_liar = liar_df["type"]

    # Transform using the trained vectorizer
    X_liar_vec = vectorizer.transform(X_liar)

    # Predict
    y_pred = model.predict(X_liar_vec)

    # Classification metrics
    print("\nPerformance on LIAR Dataset (Simple Model):")
    print(classification_report(y_liar, y_pred))

    f1 = f1_score(y_liar, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_liar, y_pred)
    print("Accuracy:", round(acc, 3))

    # Confusion matrix
    cm = confusion_matrix(y_liar, y_pred, labels=["fake", "reliable"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["fake", "reliable"])
    disp.plot()
    plt.title("Confusion Matrix - Simple Model on LIAR")
    plt.savefig("confusion_matrix_simple_LIAR.png", dpi=300, bbox_inches="tight")
    plt.show()
