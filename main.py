from CleanData import *
from Model import *
import os

#-----------------------------------------------------------------------------------------------------------------
# RENAME COLUMNS OF BBC_ARTICLES
#-----------------------------------------------------------------------------------------------------------------
#rename_columns("bbc_articles_scraped.csv", "scraped_articles.csv")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'FULL_CLEANING' ON A CSV FILE
#-----------------------------------------------------------------------------------------------------------------
#full_cleaning("updated_with_CB_P.csv", "updated_with_CB_P_clean.csv")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'GROUP_TYPES' for binary classification
#-----------------------------------------------------------------------------------------------------------------
##group_types("updated_with_CB_P_clean.csv", "updated_with_CB_P")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model
#-----------------------------------------------------------------------------------------------------------------
#X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated_with_CB_P.csv")
#train_model(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model WITH METADATA (url)
#-----------------------------------------------------------------------------------------------------------------
#X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated_with_CB_P.csv")
#train_with_meta(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model WITH METADATA AND BBC ARTICLES 
#-----------------------------------------------------------------------------------------------------------------

#combined_df = prepare_combined_dataset("updated.csv", "scraped_articles.csv", "updated_with_bbc.csv")
#X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated.csv")
#train_with_meta_and_extra_data(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model SIMPLE MODEL & EVALUATE ON LIAR DATASET
#-----------------------------------------------------------------------------------------------------------------

#X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated_with_CB_P_clean.csv")
#model, vectorizer = train_model(X_train, X_val, X_test, y_train, y_val, y_test)
#liar_df = load_liar_file("train.tsv")
#evaluate_model_on_liar_simple(model, vectorizer, liar_df)


#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model ADVANCED MODEL & EVALUATE ON LIAR DATASET
#-----------------------------------------------------------------------------------------------------------------

X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated_with_CB_P_clean.csv")
pipeline = train_final_svm_tfidf(X_train, X_val, X_test, y_train, y_val, y_test)
liar_df = load_liar_file("train.tsv")
evaluate_model_on_liar(pipeline, liar_df)


