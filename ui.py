import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import os
from Model import *
from main import *
from CleanData import *

def run_models(run_simple, run_svm, evaluate_on_liar):
    results = ""

    if run_simple:
        results += "✅ Training Simple Model (Logistic Regression)...\n"
        X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated.csv")
        model, vectorizer = train_model(X_train, X_val, X_test, y_train, y_val, y_test)

        if evaluate_on_liar:
            results += "\n📊 Evaluating Simple Model on LIAR dataset...\n"
            liar_df = load_liar_file("train.tsv")
            evaluate_model_on_liar_simple(model, vectorizer, liar_df)
            results += "Confusion matrix saved: confusion_matrix_simple_LIAR.png\n"

    if run_svm:
        results += "\n✅ Training Advanced Model (SVM + TF-IDF)...\n"
        X_train, X_val, X_test, y_train, y_val, y_test = split_data("updated.csv")
        pipeline = train_final_svm_tfidf(X_train, X_val, X_test, y_train, y_val, y_test)

        if evaluate_on_liar:
            results += "\n📊 Evaluating Advanced Model on LIAR dataset...\n"
            liar_df = load_liar_file("train.tsv")
            evaluate_model_on_liar(pipeline, liar_df)
            results += "Confusion matrix saved: confusion_matrix_LAIR.png\n"

    return results

# Gradio Interface
iface = gr.Interface(
    fn=run_models,
    inputs=[
        gr.Checkbox(label="Run Simple Model (Logistic Regression)"),
        gr.Checkbox(label="Run Advanced Model (SVM + TF-IDF)"),
        gr.Checkbox(label="Evaluate on LIAR Dataset"),
    ],
    outputs=gr.Textbox(label="Console Output"),
    title="Fake News Detection UI",
    description="Choose which models to train and whether to evaluate them on the LIAR dataset.",
)

if __name__ == "__main__":
    iface.launch(share=True)
