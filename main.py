from CleanData import *
from Model import *
from embedding import *

#-----------------------------------------------------------------------------------------------------------------
# RENAME COLUMNS OF BBC_ARTICLES
#-----------------------------------------------------------------------------------------------------------------
#rename_columns("bbc_articles_scraped.csv", "scraped_articles.csv")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'FULL_CLEANING' ON A CSV FILE
#-----------------------------------------------------------------------------------------------------------------
#full_cleaning("scraped_articles.csv", "scraped_articles.csv")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'GROUP_TYPES' for binary classification
#-----------------------------------------------------------------------------------------------------------------
#group_types("merged_cleaned.csv", "updated")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model
#-----------------------------------------------------------------------------------------------------------------
#X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated.csv")
#train_model(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model WITH METADATA (url)
#-----------------------------------------------------------------------------------------------------------------
#X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated.csv")
#train_with_meta(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model WITH METADATA AND BBC ARTICLES 
#-----------------------------------------------------------------------------------------------------------------

#combined_df = prepare_combined_dataset("updated.csv", "scraped_articles.csv", "updated_with_bbc.csv")

#print(combined_df[combined_df["type"] == "reliable"]["domain"].value_counts().head(10))
#print(combined_df[combined_df["type"] == "fake"]["domain"].value_counts().head(10))
#count_bbc = combined_df[combined_df["type"] == "reliable"]["domain"].value_counts().get("bbc.com", 0)
#print(f"Number of reliable articles from bbc.com: {count_bbc}")
#X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated.csv")
#train_with_meta_and_extra_data(X_train, X_val, X_test, y_train, y_val, y_test)
#overlap = set(X_train["content"]) & set(X_test["content"])
#print("Overlap:", len(overlap))
#train_svm_with_tfidf(X_train, X_val, X_test, y_train, y_val, y_test)

#model, vectorizer = train_final_svm_tfidf(X_train["content"], X_val["content"], X_test["content"], y_train, y_val, y_test)


#bbc_df = pd.read_csv("scraped_articles.csv")
#bbc_df = bbc_df.dropna(subset=["content"])

# Transform with the same TF-IDF vectorizer
#bbc_vec = vectorizer.transform(bbc_df["content"])

# Predict
#bbc_preds = model.predict(bbc_vec)
#bbc_df["model_prediction"] = bbc_preds

# Count predictions
#print(bbc_df["model_prediction"].value_counts())

# Save if needed
#bbc_df.to_csv("bbc_articles_with_predictions.csv", index=False)


#TEST FOR BEST PARAMETERS

#X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated_with_bbc.csv")

#best_model = train_svm_with_gridsearch(X_train, X_val, X_test, y_train, y_val, y_test)


#X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated.csv")
# Train the final model
#pipeline = train_final_svm_tfidf(X_train, X_val, X_test, y_train, y_val, y_test)

# Load and clean the BBC articles
#bbc_df = pd.read_csv("scraped_articles.csv")
#bbc_df = bbc_df.dropna(subset=["content"])

# Use pipeline directly to predict
#bbc_preds = pipeline.predict(bbc_df["content"])
#bbc_df["model_prediction"] = bbc_preds

# Show and/or save results
#print(bbc_df["model_prediction"].value_counts())
#bbc_df.to_csv("bbc_articles_with_predictions.csv", index=False)

############################################################################################################################
# Step 1: Train model on FakeNewsCorpus
X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated_with_bbc.csv")
ipeline = train_final_mlp_tfidf(X_train, X_val, X_test, y_train, y_val, y_test)

# Step 2: Load LIAR test set
liar_df = load_liar_test("test.tsv")

# Step 3: Evaluate on LIAR
evaluate_model_on_liar(pipeline, liar_df)


#X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated.csv")
#model, embeddings_index = train_final_mlp_glove(X_train, X_val, X_test, y_train, y_val, y_test)

# Step 2: Load LIAR test set
#liar_df = load_liar_test("test.tsv")

# Step 3: Evaluate on LIAR
#evaluate_model_on_liar(model, liar_df, embeddings_index)
