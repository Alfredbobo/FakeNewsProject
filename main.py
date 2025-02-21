import re
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import *

import ast

# Load CSV:
csv = pd.read_csv("https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv")
csv.to_csv(f"news_sample.csv", index=False)    # The beginning CSV-file is saved as 'news_sample.csv'
df = pd.read_csv("news_sample.csv")


########################################################################################################
# Save CSV          (helper-function to save a new CSV-file)
########################################################################################################
def save_csv(df, name):
    df.to_csv(f"news_sample_{name}.csv", index=False)
    print(f"Saved: news_sample_{name}.csv")


#########################################################################################################
# CLEANING DATA
#########################################################################################################
"""
PART 1)
""" 
df["content"] = df["content"].astype(str)           # Everything as str()
df["content"] = df["content"].str.lower()           # Everything lowercase

# EMAIL--------------------------------------------------------------------------------------------------
df["content"] = df["content"].str.replace(r'([\w\-\.]+@([\w-]+\.)+[\w-]{2,4})', '<EMAIL>', regex=True)

# DATES--------------------------------------------------------------------------------------------------
df["content"] = df["content"].str.replace(r'(?i)\b(?:\d{1,4}-\d{1,2}-\d{1,4}|[a-z]{3,9}\.?\s+\d{4}|[a-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[a-z]{3,9}\.?\s+\d{4}|[a-z]{2,3}\s+\d{1,2}\s+[a-z]{2,3}\s+\d{4})\b', '<DATE>', regex=True)

df["content"] = df["content"].str.replace(r'[^\w\s\<\>]', '', regex=True)       # Remove: .,!?;:/-(){}[]
df["content"] = df["content"].str.replace(r'\s+', ' ', regex=True)              # Remove: whitespace

# saving part1-csv using save_csv()
save_csv(df, "first_cleaning")

"""
PART 2) 
"""
# Second part of the cleaning. Cleaning the new CSV file "news_sample_first_cleaning.csv" and naming it df2.
df2 = pd.read_csv("news_sample_first_cleaning.csv")
def clean_text(text):    
    text = str(text) 
    
    # URL----------------------------------------------------------------------------------------------------
    text = re.sub(r'\b(?:https?|www)(www)*([A-Za-z0-9])+', '<URL> ', text, flags=re.IGNORECASE)
    
    # NUMBERS------------------------------------------------------------------------------------------------
    # whole numbers
    text = re.sub(r'(?<![A-Za-z])\b\d+\b', '<NUM>', text)
    # decimal numbers
    text = re.sub(r'(?<![A-Za-z])\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', '<NUM>', text)
    # currency
    text = re.sub(r'(?<![A-Za-z])(?:\$|€|¥|£)\s?\d+(?:[.,]\d+)?', '<NUM>', text)
    # percentage
    text = re.sub(r'(?<![A-Za-z])\d+(?:\.\d+)?%', '<NUM>', text)
    # 1th place... / spelling mistakes / others
    text = re.sub(r'(?<![A-Za-z])\d+\w*', '<NUM>', text)
    
    return text 

# Applying clean_text() and saving new csv using save_csv()
df2["content"] = df2["content"].apply(clean_text)  
save_csv(df2, "CLEANED")  # Cleaned data is stored in: 'news_sample_CLEANED.csv', and ready for Tokenization


#########################################################################################################
# TOKENIZATION & STEMMING
#########################################################################################################
"""
* Tokenize the text - split text into words (tokens)
* Compute vocabulary BEFORE stop-word removal
* Remove stopwords - (filter out words like: 'the', 'is', 'and' etc...)
* Compute vocabulary AFTER stop-word removal

Apply stemming - Reduce words to root form (e.g: 'running' --> 'run')
Compute vocabulary size and reduction rate before and after stemming

returns:
vocabulary(BEFORE) = int
vocabulary(AFTER) = int
stop-word removal CSV = news_sample_Stopwords.csv

"""
# TOKENS--------------------------------------------------------------------------------------------------
df3 = pd.read_csv("news_sample_CLEANED.csv")
tokenizer = RegexpTokenizer(r'<[^>]+>|\w+')       # Defining regex to treat words like <DATE> as a single word
                                                           
def tokenize(text):
    return tokenizer.tokenize(text)

# Applying function and saving tokenized data
df3["content"] = df3["content"].apply(tokenize)
save_csv(df3, "Tokenization")


# STOP-WORD REMOVAL & COMPUTE-----------------------------------------------------------------------------
df4 = pd.read_csv("news_sample_Tokenization.csv")
stop_words = set(stopwords.words("english"))              # Set of english stop-words

df4["content"] = df4["content"].apply(ast.literal_eval)   # Converts str(lst) → lst /// "['hej', 'ole']"  →  ['hej', 'ole']

# compute unique words BEFORE stop-word removal (16522)
before_stopword_removal = set()            # 
for i in range(len(df4)):                  # Loop through each row in the CSV
    for word in df4.loc[i, "content"]:     # Loop through each word in the row (only in "content")
        before_stopword_removal.add(word)  

# remove stop-words
def remove_stopwords(all_tokens):
    filtered_tokens = []
    for token in all_tokens:                       # Iterate over each token in df['content']
        if token not in stop_words:                # If token is NOT in the stop-words set:
            filtered_tokens.append(token)          #        - append to filtered_tokens
    return filtered_tokens 
# apply stop-word removal and save
df4["content"] = df4["content"].apply(remove_stopwords)    # (!!!) .apply() iterates through all the rows in the CSV
save_csv(df4, "Stopwords")

# compute unique words AFTER stop-word removal (16390) & reduction rate
stopword_removal = set()         
for i in range(len(df4)):                  # Loop through each row in the CSV
    for word in df4.loc[i, "content"]:     # Loop through each word in the row (only in "content")
        stopword_removal.add(word) 

print(f"Total vocabulary BEFORE stop-word removal: {len(before_stopword_removal)}")
print(f"Total vocabulary AFTER stop-word removal: {len(stopword_removal)}")
reduction_rate = ((len(before_stopword_removal) - len(stopword_removal)) / len(before_stopword_removal)) * 100
print(f"Reduction Rate of the vocabulary: {reduction_rate:.2f}%")


# STEMMING------------------------------------------------------------------------------------------------
stemmer = PorterStemmer()
skip_tokens = {"<NUM>", "<DATE>", "<URL>", "<EMAIL>"}
                  
for i in range(len(df4)):                           # Iterate through each row in CSV
    stemmed_tokens = []                             # Empty list to store stemmed tokens
    for token in df4.loc[i, "content"]:             # Iterate through each token in row ["content"]
        if token in skip_tokens:                    # SKIP token if <NUM>, <DATE>, etc....
            stemmed_tokens.append(token)
        else:
            stemmed_word = stemmer.stem(token)      # Apply stemming
            stemmed_tokens.append(stemmed_word)     # Store the stemmed word
    df4.loc[i, "content"] = str(stemmed_tokens)     # Update with stemmed tokens

# Save csv
save_csv(df4, "Stemmed")
 