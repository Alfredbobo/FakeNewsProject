import re
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import ast
from urllib.parse import urlparse
from Model import group_types
import os

# ----------------------------------------------------------------------------------------------------------
# *) Helper function to change column names in "bbc_articles_scraped"
# ----------------------------------------------------------------------------------------------------------
def rename_columns(input_csv, output_csv):
    """
    Renames 'text' → 'content' and 'url' → 'domain' in given CSV,
    output: new CSV file
    """
    df = pd.read_csv(input_csv)

    # Rename columns if they exist
    df.rename(columns={"text": "content", "url": "domain"}, inplace=True)

    # Save updated DataFrame
    df.to_csv(output_csv, index=False)

    print(f"Renamed columns and saved to '{output_csv}'")

# ----------------------------------------------------------------------------------------------------------
# *) Helper function to save a CSV
# ----------------------------------------------------------------------------------------------------------
def save_csv(df, name):
    df.to_csv(f"{name}", index=False)
    print(f"Saved: {name}")

# ----------------------------------------------------------------------------------------------------------
# *) Helper function to find the domain and change it from bbc.com/.../... to bbc.com.
# ----------------------------------------------------------------------------------------------------------

def extract_domain(url):
    try:
        return urlparse(str(url)).netloc.lower().replace("www.", "")
    except:
        return str(url).lower().replace("www.", "")
    
# ----------------------------------------------------------------------------------------------------------
# *) Helper function combine the FakeNewsCorpus with the BBC articles
# ----------------------------------------------------------------------------------------------------------
def prepare_combined_dataset(original_csv, bbc_csv, cleaned_csv_name):

    # Step 1: Clean original dataset
    group_types(original_csv, "updated_cleaned")
    original_df = pd.read_csv("updated_cleaned.csv")

    # Step 2: Load and fix BBC dataset
    bbc_df = pd.read_csv(bbc_csv)
    bbc_df["type"] = "reliable"
    bbc_df["domain"] = bbc_df["domain"].apply(extract_domain)

    # Step 3: Combine & save
    combined_df = pd.concat([original_df, bbc_df], ignore_index=True)
    combined_df.to_csv(cleaned_csv_name, index=False)

    print(f"Combined dataset saved as: {cleaned_csv_name}")
    return combined_df


# ----------------------------------------------------------------------------------------------------------
# *) Part 1 Cleaning
# ----------------------------------------------------------------------------------------------------------
def part1_cleaning(df):
    """
      - Everything to string
      - Lowercase
      - Replace emails
      - Replace dates
      - Remove punctuation / symbols
      - Remove extra whitespace
    """

    df["content"] = df["content"].astype(str)
    df["content"] = df["content"].str.lower()

    # EMAIL
    df["content"] = df["content"].str.replace(
        r'([\w\-\.]+@([\w-]+\.)+[\w-]{2,4})', '<EMAIL>', regex=True
    )
    # DATES
    df["content"] = df["content"].str.replace(
        r'(?i)\b(?:\d{1,4}-\d{1,2}-\d{1,4}|[a-z]{3,9}\.?\s+\d{4}|[a-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[a-z]{3,9}\.?\s+\d{4}|[a-z]{2,3}\s+\d{1,2}\s+[a-z]{2,3}\s+\d{4})\b',
        '<DATE>',
        regex=True
    )
    # Remove punctuation
    df["content"] = df["content"].str.replace(r'[^\w\s\<\>]', '', regex=True)
    # Remove whitespace
    df["content"] = df["content"].str.replace(r'\s+', ' ', regex=True)

    return df


# ----------------------------------------------------------------------------------------------------------
# *) Part 2 Cleaning
# ----------------------------------------------------------------------------------------------------------
def part2_cleaning(df):
    """
    Perform the second-level cleaning:
      - Replace URLs
      - Replace numeric values (<NUM>)
    """
    def clean_text(text):
        text = str(text)
        # URL
        text = re.sub(r'\b(?:https?|www)(www)*([A-Za-z0-9])+', '<URL> ', text, flags=re.IGNORECASE)
        # Whole numbers
        text = re.sub(r'(?<![A-Za-z])\b\d+\b', '<NUM>', text)
        # Decimal numbers (1,234 or 123.456)
        text = re.sub(r'(?<![A-Za-z])\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', '<NUM>', text)
        # Currency
        text = re.sub(r'(?<![A-Za-z])(?:\$|€|¥|£)\s?\d+(?:[.,]\d+)?', '<NUM>', text)
        # Percentage
        text = re.sub(r'(?<![A-Za-z])\d+(?:\.\d+)?%', '<NUM>', text)
        # 1th place, etc. (any trailing letters after a digit)
        text = re.sub(r'(?<![A-Za-z])\d+\w*', '<NUM>', text)
        return text
    
    df["content"] = df["content"].apply(clean_text)
    return df


# ----------------------------------------------------------------------------------------------------------
# *) Tokenization
# ----------------------------------------------------------------------------------------------------------
def tokenize(df):
    """
    Tokenize the 'content' of the dataframe using a RegexpTokenizer
    that captures words and special tokens like <DATE>, <URL>, etc.
    """
    tokenizer = RegexpTokenizer(r'<[^>]+>|\w+')
    
    def tokenize(text):
        return tokenizer.tokenize(text)
    
    df["content"] = df["content"].apply(tokenize)
    return df


# ----------------------------------------------------------------------------------------------------------
# *) Stop-word Removal
# ----------------------------------------------------------------------------------------------------------
def remove_stopwords(df):
    """
    Remove English stopwords from the tokenized 'content' in df.
    Return the vocabulary sizes (before/after) and the updated df.
    """
    stop_words = set(stopwords.words("english"))

    # if ['content'] is str(lst) then convert to → list  ////  e.g: "['hej', 'ole']"  →  ['hej', 'ole']
    if type(df["content"][0]) is str:
        df["content"] = df["content"].apply(ast.literal_eval)
    else:
        pass

    # COMPUTATION (before stop-word removal)
    before_stopword_removal = set()            
    for i in range(len(df)):                         # Loop through each row in the CSV
        for word in df.loc[i, "content"]:            # Loop through each word in the row (only in "content")
            before_stopword_removal.add(word)  


    # Remove stop-words
    def remove_stopwords(all_tokens):
        filtered_tokens = []
        for token in all_tokens:                 # Iterate over each token in df['content']
            if token not in stop_words:          # If token is NOT in the stop-words set: - append token to list
                filtered_tokens.append(token)
        return filtered_tokens

    # apply stop-word removal
    df["content"] = df["content"].apply(remove_stopwords)  


    # COMPUTATION (after stop-word removal)
    stopword_removal = set()         
    for i in range(len(df)):         
        for word in df.loc[i, "content"]:   
            stopword_removal.add(word) 

    reduction_rate = ((len(before_stopword_removal) - len(stopword_removal)) / len(before_stopword_removal)) * 100


    # PRINT STATEMENTS
    print(f"Total vocabulary BEFORE stop-word removal: {len(before_stopword_removal)}")
    print(f"Total vocabulary AFTER stop-word removal: {len(stopword_removal)}")
    print(f"Reduction Rate AFTER stop-word removal: {reduction_rate:.2f}%")

    return df, stopword_removal


# ----------------------------------------------------------------------------------------------------------
# *) Stemming
# ----------------------------------------------------------------------------------------------------------
def stemming(df, stopword_removal):
    stemmer = PorterStemmer()
    skip_tokens = {"<NUM>", "<DATE>", "<URL>", "<EMAIL>"}

    for i in range(len(df)):                              # Iterate through each row in CSV
        stemmed_tokens = []
        for token in df.loc[i, "content"]:
            if token in skip_tokens:                      # SKIP token if: <NUM>, <DATE>, <URL> or <EMAIL>
                stemmed_tokens.append(token)
            else:
                stemmed_word = stemmer.stem(token)        # Apply stemming to each token
                stemmed_tokens.append(stemmed_word)       # append stemmed tokens to list
        df.loc[i, "content"] = str(stemmed_tokens)


    if type(df["content"][0]) is str:
        df["content"] = df["content"].apply(ast.literal_eval)
    else:
        pass


    # COMPUTATION (after stemming)
    unique_stem_words = set()
    for token in df["content"]:
        unique_stem_words.update(token)

    reduction_rate = ((len(stopword_removal) - len(unique_stem_words)) / len(stopword_removal)) * 100

    # PRINT STATEMENTS
    print(f"Total vocabulary AFTER stemming: {len(unique_stem_words)}")
    print(f"Reduction Rate AFTER stemming): {reduction_rate:.2f}%")

    return df

# ----------------------------------------------------------------------------------------------------------
# *) RUN CLEANING (main pipeline function)
# ----------------------------------------------------------------------------------------------------------
def full_cleaning(csv_file, name):
    """
    Loads the CSV, applies all cleaning steps in the following sequence.
    
    Steps:
      1) Part1 cleaning
      2) Part2 cleaning
      3) Tokenization
      4) Stop-word removal & stats
      5) Stemming & final stats
    
    Returns:
      Cleaned DataFrame
    """
    #1) Load CSV
    df = pd.read_csv(csv_file)

    #2) Part 1 clean
    df = part1_cleaning(df)

    #3) Part 2 clean
    df = part2_cleaning(df)

    #4) Tokenization
    df = tokenize(df)

    #5) Stop-word removal
    df, stopword_removal = remove_stopwords(df)

    #6) Stemming
    df = stemming(df, stopword_removal)

    #7) Save cleaned csv
    folder = "FakeNewsCorpus_chunks/"     # cleaned_csv to this folder - (change it if you want to save somewhere else)
    df.to_csv(folder + name, index=False)
    print(f"Saved: {name}")

    return df



# Test run
# full_cleaning("news_sample.csv", "TEST.csv")


# ----------------------------------------------------------------------------------------------------------
# *) RUN CLEANING FOR THE LIARDATASET
# ----------------------------------------------------------------------------------------------------------
def full_cleaning_liar(csv_file, name):
    """
    Loads the CSV, applies all cleaning steps in the following sequence.
    
    Steps:
      1) Part1 cleaning
      2) Part2 cleaning
      3) Tokenization
      4) Stop-word removal & stats
      5) Stemming & final stats
    
    Returns:
      Cleaned DataFrame
    """
    #1) Load CSV
    df = pd.read_csv(csv_file)

    #2) Part 1 clean
    df = part1_cleaning(df)

    #3) Part 2 clean
    df = part2_cleaning(df)

    #4) Tokenization
    df = tokenize(df)

    #5) Stop-word removal
    df, stopword_removal = remove_stopwords(df)

    #6) Stemming
    df = stemming(df, stopword_removal)

    #7) Save cleaned csv
    folder = "operation_csv_files/"     # cleaned_csv to this folder - (change it if you want to save somewhere else)
    df.to_csv(folder + name, index=False)
    print(f"Saved: {name}")

    return df

# ----------------------------------------------------------------------------------------------------------
# *) LOAD, CHANGE AND CLEAN THE LIAR DATA-SET
# ----------------------------------------------------------------------------------------------------------

def load_liar_file(file_path="test.tsv"):
    # Load the LIAR dataset
    df = pd.read_csv(file_path, sep="\t", header=None)
    df.columns = [
        "id", "label", "statement", "subject", "speaker", "job_title", "state", "party",
        "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context"
    ]

    label_map = {
        "true": "reliable",
        "mostly-true": "reliable",
        "false": "fake",
        "pants-fire": "fake",
        "barely-true": "fake"
    }

    df = df[df["label"].isin(label_map)]
    df["type"] = df["label"].map(label_map)
    df_clean = df[["statement", "type"]].rename(columns={"statement": "content"})

    # Add the correct directory
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    raw_csv = f"liar_{base_name}.csv"
    cleaned_csv = f"liar_{base_name}_cleaned.csv"

    df_clean.to_csv(raw_csv, index=False)
    full_cleaning_liar(raw_csv, cleaned_csv)

    #  Read from correct folder
    return pd.read_csv("operation_csv_files/" + cleaned_csv)

