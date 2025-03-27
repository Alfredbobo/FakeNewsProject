from CleanData import *
from Model import *
import os
import gc
T=True
F=False


RUN_SIMPLE_MODEL = F
RUN_SIMPLE_MODEL_AND_EVALUATE_ON_LIAR = F
RUN_ADVANCED_MODEL = F
RUN_ADVANCED_MODEL_AND_EVALUATE_ON_LIAR = F

COMBINE_FAKENEWSCORPUS_WITH_BBC = F #MAKE SURE TO MAKE A COPY OF UPDATED IF YOU WANT TO RUN ON THE DATA WITHOUT THE BBC FILES (THIS OVERWRITES THE CURRENT "updated.CSV")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'GROUP_TYPES' for binary classification
#-----------------------------------------------------------------------------------------------------------------
#group_types("merged_cleaned.csv", "updated.csv")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model
#-----------------------------------------------------------------------------------------------------------------
if RUN_SIMPLE_MODEL:
    print("Running Simple Model...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated.csv")
    train_model(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model WITH METADATA (url)
#-----------------------------------------------------------------------------------------------------------------
#X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated.csv")
#train_with_meta(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# COMBINE FAKENEWSCORPUS WITH BBC:
#-----------------------------------------------------------------------------------------------------------------
if COMBINE_FAKENEWSCORPUS_WITH_BBC:
    #FIRST CLEAN THE BBC ARTICLES:
    rename_columns("bbc_articles_scraped.csv", "scraped_articles.csv")
    full_cleaning("scraped_articles.csv", "scraped_articles.csv")
    combined_df = prepare_combined_dataset("updated.csv", "scraped_articles.csv", "updated.csv")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model WITH METADATA AND BBC ARTICLES 
#-----------------------------------------------------------------------------------------------------------------
X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated.csv")
train_with_meta_and_extra_data(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model SIMPLE MODEL & EVALUATE ON LIAR DATASET
#-----------------------------------------------------------------------------------------------------------------
if RUN_SIMPLE_MODEL_AND_EVALUATE_ON_LIAR:
    X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated.csv")
    model, vectorizer = train_model(X_train, X_val, X_test, y_train, y_val, y_test)
    liar_df = load_liar_file("train.tsv")
    evaluate_model_on_liar_simple(model, vectorizer, liar_df)

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model ADVANCED MODEL & EVALUATE ON LIAR DATASET
#-----------------------------------------------------------------------------------------------------------------
if RUN_ADVANCED_MODEL:
    print("Running Advanced Model...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated.csv")
    pipeline = train_final_svm_tfidf(X_train, X_val, X_test, y_train, y_val, y_test)


#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model ADVANCED MODEL & EVALUATE ON LIAR DATASET
#-----------------------------------------------------------------------------------------------------------------
if RUN_ADVANCED_MODEL_AND_EVALUATE_ON_LIAR:
    print("Running Advanced Model and evaluate on LIAR...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated.csv")
    pipeline = train_final_svm_tfidf(X_train, X_val, X_test, y_train, y_val, y_test)
    liar_df = load_liar_file("train.tsv")
    evaluate_model_on_liar(pipeline, liar_df)


liar_df = load_liar_file("test.tsv")
