import re
import pandas as pd

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

# applying clean_text() function
df2["content"] = df2["content"].apply(clean_text)  

# saving new csv using save_csv()
save_csv(df2, "CLEANED")

"""
Cleaned data is now stored in news_sample_CLEANED.csv, and ready for the Tokenization part
"""

#########################################################################################################
# TOKENIZATION
#########################################################################################################


