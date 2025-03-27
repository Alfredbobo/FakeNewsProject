from CleanData import *
from Model import *
import os
import gc

# =========================================================================================
# MODEL SELECTION CONFIGURATION
# Toggle which models to run by setting the flags below to True or False
# Note that ONLY ONE MODEL CAN RUN AT 1 TIME. This is due to the splitting of the dataset
# =========================================================================================

# Simple logistic regression model
RUN_SIMPLE_MODEL = False

# Simple model + metadata (URL.)
RUN_SIMPLE_MODEL_WITH_METADATA = False

# Combine FakeNewsCorpus with BBC articles
# !!! This will overwrite 'updated.csv' — back up if needed or grab new updated.csv file from releases!!!
COMBINE_FAKENEWSCORPUS_WITH_BBC = False

# Simple model + metadata + BBC articles
RUN_SIMPLE_MODEL_WITH_METADATA_AND_BBC = False

# Simple model evaluated on LIAR dataset (No metadata)
RUN_SIMPLE_MODEL_AND_EVALUATE_ON_LIAR = False

# Advanced model (SVM + TF-IDF) (No metadata)
RUN_ADVANCED_MODEL = False

# Advanced model evaluated on LIAR dataset (No metadata)
RUN_ADVANCED_MODEL_AND_EVALUATE_ON_LIAR = False





#INPUTFILES CAN BE CHANGED IN THE CODE BELOW"

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
if RUN_SIMPLE_MODEL_WITH_METADATA:
    print("Running Simple Model with METADATA...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated.csv")
    train_with_meta(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# COMBINE FAKENEWSCORPUS WITH BBC:
#-----------------------------------------------------------------------------------------------------------------
if COMBINE_FAKENEWSCORPUS_WITH_BBC:
    #FIRST CLEAN THE BBC ARTICLES:
    rename_columns("scraped_articles.csv", "scraped_articles.csv")
    full_cleaning("scraped_articles.csv", "scraped_articles.csv")
    combined_df = prepare_combined_dataset("updated.csv", "scraped_articles.csv", "updated.csv")

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model WITH METADATA AND BBC ARTICLES 
#-----------------------------------------------------------------------------------------------------------------
if RUN_SIMPLE_MODEL_WITH_METADATA_AND_BBC:
    print("Running Simple Model with METADAT & BBC...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_with_meta("updated.csv")
    train_with_meta_and_extra_data(X_train, X_val, X_test, y_train, y_val, y_test)

#-----------------------------------------------------------------------------------------------------------------
# RUN 'SPLIT_DATA' & Train Model SIMPLE MODEL & EVALUATE ON LIAR DATASET
#-----------------------------------------------------------------------------------------------------------------
if RUN_SIMPLE_MODEL_AND_EVALUATE_ON_LIAR:
    print("Running Simple Model and evaluateing on LIAR...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated.csv")
    model, vectorizer = train_model(X_train, X_val, X_test, y_train, y_val, y_test)
    liar_df = load_liar_file("test.tsv")
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
    print("Running Advanced Model and evaluateing on LIAR...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated.csv")
    pipeline = train_final_svm_tfidf(X_train, X_val, X_test, y_train, y_val, y_test)
    liar_df = load_liar_file("test.tsv")
    evaluate_model_on_liar(pipeline, liar_df)

