# FakeNewsProject
Group: ML BOLD *(Alfred, Jon, August, Nikolai)*

---

### HOW TO RUN:

1. **Download the project** and open a terminal in the folder.
2. **Download "Dataset"** from the feature and add the file **"updated.csv"** and **"test.tsv"** to the project folder. If you want to try to clean the data yourself Download the original [995,000_rows.csv](https://absalon.ku.dk/courses/80486/files/9275000/download?download_frd=1)
3. **Activate your Python environment** if needed.
4. To run operations or evaluations, open `main.py` and set the corresponding model flags (e.g. `RUN_SIMPLE_MODEL = True`).
5. Only **one flag should be set to `True` at a time** due to dataset splitting.
6. Run the script with:

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

```bash
python main.py
