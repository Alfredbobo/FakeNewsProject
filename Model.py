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
from scipy.sparse import hstack
from embedding import *
import numpy as np

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

def train_advanced_model_tfidf_naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    # Extract content only (no domain used)
    X_train_text = X_train["content"]
    X_val_text = X_val["content"]
    X_test_text = X_test["content"]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_val_vec = vectorizer.transform(X_val_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # Train Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_vec)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))


def train_svm_with_tfidf(X_train, X_val, X_test, y_train, y_val, y_test):
    # Use TF-IDF instead of CountVectorizer
    vectorizer = TfidfVectorizer(max_features=20000)
    X_train_vec = vectorizer.fit_transform(X_train["content"])
    X_val_vec = vectorizer.transform(X_val["content"])
    X_test_vec = vectorizer.transform(X_test["content"])

    # Train an SVM model
    model = LinearSVC(max_iter=10000)
    model.fit(X_train_vec, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_vec)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))
   
    return model, vectorizer


def train_svm_with_gridsearch(X_train, X_val, X_test, y_train, y_val, y_test):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC(max_iter=10000))
    ])

    param_grid = {
        'tfidf__max_features': [10000, 20000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__stop_words': [None, 'english'],
        'clf__C': [0.1, 1, 10]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)
    X_small = X_train["content"][:10000]  # or [:5000] for even faster
    y_small = y_train[:10000]

    grid.fit(X_small, y_small)  

    print("\nBest Parameters:", grid.best_params_)
    print("Best CV Score (f1_macro):", round(grid.best_score_, 4))

    # Evaluate on test set
    y_pred = grid.predict(X_test["content"])

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))

    return grid.best_estimator_  


def train_final_svm_tfidf(X_train, X_val, X_test, y_train, y_val, y_test):
    # Build pipeline with best parameters
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words=None
        )),
        ("clf", LinearSVC(C=1, max_iter=10000))
    ])

    # Fit on the full training data
    pipeline.fit(X_train["content"], y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test["content"])

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

def train_final_mlp_tfidf(X_train, X_val, X_test, y_train, y_val, y_test):
    # Build pipeline with best parameters
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words=None
        )),
        ("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42))
    ])

    # Fit on full training data
    pipeline.fit(X_train["content"], y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test["content"])

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))

    return pipeline


def train_final_mlp_glove(X_train, X_val, X_test, y_train, y_val, y_test):
    glove_path = "glove.6B.100d.txt"
    embeddings_index = load_glove_embeddings(glove_path)

    def embed_df(df):
        return np.vstack([sentence_to_embedding(row["content"], embeddings_index) for _, row in df.iterrows()])

    X_train_vec = embed_df(X_train)
    X_val_vec = embed_df(X_val)
    X_test_vec = embed_df(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 3))

    return clf, embeddings_index

def evaluate_mlpmodel_on_liar(model, liar_df, embeddings_index):
    from embedding import sentence_to_embedding
    X_liar = np.vstack([sentence_to_embedding(row["content"], embeddings_index) for _, row in liar_df.iterrows()])
    y_liar = liar_df["type"]

    y_pred = model.predict(X_liar)

    print("\nPerformance on LIAR Dataset:")
    print(classification_report(y_liar, y_pred))

    f1 = f1_score(y_liar, y_pred, pos_label="fake")
    print("F1 Score (fake):", round(f1, 3))

    acc = accuracy_score(y_liar, y_pred)
    print("Accuracy:", round(acc, 3))
