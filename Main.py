import numpy as np
import pandas as pd

import tBGER

#


#### MAIN 


## defining the datasets (saved in Data_Extracted folder in the data-preprocessing step)
data_file = ['philosophy','ebooks','history','3dprinting','bioinformatics','ai']

end_dt = '2019-06-01'   #### Define the end date of the dataset
min_post= 4             #### Define the minimum answers by users for them to be included in the training data
aa = 1                  #### Define aa = 1 to train on questions with only accepted answers; aa = 0 to train on questions with all answers

User_Rank=[]

for CQA in (range(6)):
    
    
    i=data_file[CQA]
    
    post_dat = pd.read_feather('Data_Extracted/{}_Posts.ft'.format(i))
    post_dat['CreationDate'] =  pd.to_datetime(post_dat['CreationDate'], format='%Y-%m-%dT%H:%M:%S.%f')
    post_dat['Post_Year'] = post_dat['CreationDate'].dt.year

    post_dat = post_dat[post_dat.OwnerUserId != '']
    post_dat = post_dat[-post_dat.OwnerUserId.isna()].reset_index(drop=True)
    post_dat['A_Scr'] = post_dat.Score.astype(str).astype(int)
    post_dat = post_dat[post_dat['CreationDate'] < end_dt]


    q_dat = post_dat[-post_dat.AcceptedAnswerId.isna()].reset_index(drop=True)
    q_dat = q_dat[['Id','CreationDate','OwnerUserId','Tags','AcceptedAnswerId']]
    q_dat.columns= ['Q_Id','Q_Date','Q_UserId','Tags','AcceptedAnswerId']

    ### Define train - test data based on date split (80:20)
    date_split = q_dat.Q_Date.quantile(0.8)
    
    
    train_q = q_dat[q_dat['Q_Date'] < date_split]

    ## getting the training dataset
    train_df,model_df = tBGER.UMAP_traindf(post_dat,date_split, min_post,aa)

    
    ### getting the test dataset
    test_q = q_dat[q_dat['Q_Date'] > date_split]
    test_q = pd.merge(test_q,post_dat[['Id','OwnerUserId','ParentId']],how='left',left_on='AcceptedAnswerId',right_on = 'Id')
    test_q = test_q[test_q.OwnerUserId.isin(train_df.OwnerUserId)]

    
    # getting TAG-level users train data
    model_Udf = tBGER.TAG_Udf(model_df,date_split)


    # getting TAG-level users test data 
    test_df = tBGER.UMAP_testDf(test_q,model_Udf,post_dat)

    
    #### Estimating the rating of users on each TAG
    pred_rating_dat = tBGER.BiRecSys_train_df(model_Udf)


    #Predicting users rank for questions in test data
    Test_U_rank, MRR, P_1, P_3 = tBGER.BiRecSys_eval_df(test_df,model_Udf, pred_rating_dat)

    User_Rank.append({
        'CQA': i, 
        'MRR' : MRR,
        'HIT_1' : P_1,
        'HIT_3' : P_3

    })

print(pd.DataFrame(User_Rank))

