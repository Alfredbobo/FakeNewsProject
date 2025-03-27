# FakeNewsProject
Group: ML BOLD *(Alfred, Jon, August, Nikolai)*


### HOW TO RUN:
Download *FakeNewsProject* and open a terminal in the folder. 
Every operation or test-evaluation should <br> be runned through **main.py** with >> python main.py 

#### In main.py:
Uncomment (*remove (#))* the operation of evaluation you want to run. For example: <br>
To run *full_cleaning* - uncomment that line and run >> *(#full_cleaning("scraped_articles.csv", "scraped_articles.csv"))* <br>
Operations you don't want to run should be uncommented when doing >> python main.py

### HOW TO GET SAME RESULTS *(as us)*:
1) Make sure to have the ***995,000_rows.csv*** file inside the *FakeNewsProject* folder
2) Open *FakeNewsCorpusData.ipynb* and **run** >> "CLEAN BIG DATA FILES", and thereafter >> "MERGE ALL CLEANED_CHUNKS.csv"
3) Open *main.py* and **run** >> *group_types(input_csv, name_of_new_csv)*
   ##### Simple Model
4) Then, (still in *main.py*), uncomment the two lines under the **title**: "RUN 'SPLIT_DATA' & Train Model" and **run**
   ##### Simple Model with MetaData
4.2) Then, (still in *main.py*), uncomment the two lines under the **title**: "RUN 'SPLIT_DATA' & Train Model WITH METADATA (url)" and **run**

