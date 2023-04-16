# t-BGER


## Introduction

This is the repo of "Temporal-weighted bipartite graph model for sparse expert recommendation in community question answering" 31st ACM International Conference on User Modeling, Adaptation and Personalization (ACM-UMAP), 2023

## Preparing Data

* The archive of the dataset is available [here](https://archive.org/download/stackexchange). Download the dataset and unzip the 7z files into ./Raw_Data/. 

For example, download for the SuperUser SE community, store the xml files in Raw_Data/SuperUser folder into Data_Extracted folder

* Then run
$ python preprocessing.py

to process the xml files to dataframes and save them into ./Data_Extracted/.

## Run

Execute the Main_Code.py to get the performance on the dataset.
Execute the cold_StartU.py to get the ranking of cold-start users.




