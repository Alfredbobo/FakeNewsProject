# 📰 FakeNewsProject  
**Group: ML BOLD** *(Alfred, Jon, August, Nikolai)*

---

## 🚀 HOW TO RUN:

1.  **Download the project** and open a terminal in the project folder.
2.  **Download the dataset files**:  
   - Add both `updated.csv` and `test.tsv` to the root of the project.  
   - *Want to preprocess it yourself?* Download the original  
     👉 [`995,000_rows.csv`](https://absalon.ku.dk/courses/80486/files/9275000/download?download_frd=1)
3.   Activate your Python environment (if needed).
4.  Open `main.py` and set the model you want to run by changing its flag to `True`  
   _(e.g. `RUN_SIMPLE_MODEL = True`)_ ⚠️ **Only one flag should be `True` at a time** due to dataset splitting!
6. ▶️ Run the script:

---

### AVAILABLE CONFIGURATIONS:

Set one of the following flags to `True` in `main.py` to run a specific model or operation:

- `RUN_SIMPLE_MODEL`  
  → Trains a logistic regression model on `updated.csv`.

- `RUN_SIMPLE_MODEL_WITH_METADATA`  
  → Same as above, but includes URL metadata.

- `COMBINE_FAKENEWSCORPUS_WITH_BBC`  
  → Cleans and merges scraped BBC articles with `updated.csv`.  
  ⚠️ This will overwrite `updated.csv`!

- `RUN_SIMPLE_MODEL_WITH_METADATA_AND_BBC`  
  → Trains model with metadata and additional BBC data.

- `RUN_SIMPLE_MODEL_AND_EVALUATE_ON_LIAR`  
  → Trains on `updated.csv` and evaluates on the LIAR dataset.

- `RUN_ADVANCED_MODEL`  
  → Trains an SVM + TF-IDF model (no metadata) on `updated.csv`.

- `RUN_ADVANCED_MODEL_AND_EVALUATE_ON_LIAR`  
  → Same as above, with additional evaluation on LIAR dataset.

---

### HOW TO GET SAME RESULTS *(as in our report)*:

1. Place the original dataset file `995,000_rows.csv` inside the `FakeNewsProject` folder.
2. Open `FakeNewsCorpusData.ipynb` and run:
   - "CLEAN BIG DATA FILES"
   - Then "MERGE ALL CLEANED_CHUNKS.csv"
3. This will generate `updated.csv` — the standard input for all models.
4. Open `main.py`, set the desired flag to `True`, and run:

---

## 🧪 Results

Results are displayed and stored in multiple ways:

- 📄 **Console Output**:
  - Classification Report (precision, recall, f1-score)
  - Accuracy and macro/weighted averages
  - F1-score for the fake class

- 📊 **Visuals**:
  - Confusion matrices are saved automatically as `.png` files.
  - A bar chart for cross-domain accuracy and F1-scores can be created via LaTeX or Python.

---

## 🧠 Project Team – ML-BOLD

| Name                           | Alias   |
|--------------------------------|---------|
| Jon Broby Tinghuus Petersen    | `jbm823` |
| Alfred Tolstrup                | `zmv455` |
| August Bromann                | `cdz558` |
| Nikolai Lysholdt Petersen      | `mqh859` |

---

```bash
python main.py
