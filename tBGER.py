

import numpy as np
import pandas as pd


### defing the train data - with minimum number of answer users
def UMAP_traindf(train_dat,date, minPost,AA_Ind):
    
    TAG_Qdf = train_dat[(train_dat['PostTypeId'] == '1')]#.reset_index(drop=True)
    TAG_Qdf = TAG_Qdf[TAG_Qdf['CreationDate'] < date]
    
    if AA_Ind == 1:
        TAG_Qdf = TAG_Qdf[-TAG_Qdf.AcceptedAnswerId.isna()].reset_index(drop=True)


    TAG_Qdf = TAG_Qdf.filter(items=['Id', 'Tags','AcceptedAnswerId'])
    TAG_Qdf.columns = ['Q_Id','Tags','AcceptedAnswerId']

    
    TAG_Adf = train_dat[(train_dat['PostTypeId'] == '2') & (train_dat['A_Scr'] > 0)].reset_index(drop=True)
    TAG_Adf = TAG_Adf.filter(items=['Id', 'CreationDate','A_Scr','OwnerUserId','ParentId'])
    TAG_Adf = TAG_Adf[TAG_Adf['CreationDate'] < date]
    
    TAG_QnAdf = pd.merge(TAG_Adf,TAG_Qdf,how = 'left',left_on='ParentId',right_on='Q_Id')
    TAG_QnAdf = TAG_QnAdf[-TAG_QnAdf.Q_Id.isnull()]

    arr_slice = TAG_QnAdf[['OwnerUserId']].values
    unq,unqtags,counts = np.unique(arr_slice.astype(str),return_inverse=True,return_counts=True)
    TAG_QnAdf['Ttl_Ans'] = counts[unqtags]

    TAG_QnAdf = TAG_QnAdf[TAG_QnAdf.Ttl_Ans > minPost].reset_index(drop=True)
    
    train_QnA_df = TAG_QnAdf.copy()


    # defining TAG level data
    TAG_QnAdf['N_Tags'] = TAG_QnAdf.Tags.str.count("<")
    TAG_QnAdf.Tags = TAG_QnAdf.Tags.apply(lambda x: x.replace('<',''))
    TAG_QnAdf = TAG_QnAdf.drop('Tags', axis=1).join(TAG_QnAdf['Tags'].str.split('>', expand=True).stack().reset_index(level=1, drop=True).rename('Tags'))
    TAG_QnAdf = TAG_QnAdf.loc[TAG_QnAdf['Tags'] != '']
    
    TAG_QnAdf['wt_Cnt'] = 1/TAG_QnAdf['N_Tags']
    
    return train_QnA_df,TAG_QnAdf


## Defing test data
def UMAP_testDf(test_Q_df,train_df,post_df):
    
    test_Q_df = test_Q_df[['Q_Id','AcceptedAnswerId','Tags']]
    test_Q_df.columns = ['Id','AcceptedAnswerId','Tags']
    
    testAdf = post_df[post_df['PostTypeId'] == '2'].reset_index(drop=True)
    testAAdf = testAdf[testAdf['Id'].isin(test_Q_df.AcceptedAnswerId)]
    testAAdf = testAAdf[testAAdf['OwnerUserId'] != '']
    testAAdf = testAAdf[testAAdf['OwnerUserId'].isin(train_df.OwnerUserId)]
    testAAdf = testAAdf.filter(items=['Id','OwnerUserId','ParentId'])
    testAAdf.columns = ['A_Id','OwnerUserId','ParentId']
    
    testAAdf_U = pd.merge(test_Q_df,testAAdf,how='left',left_on='AcceptedAnswerId',right_on='A_Id')   
    testAAdf_U = testAAdf_U[testAAdf_U.A_Id.notnull()]
    testAAdf_U['N_Tags'] = testAAdf_U.Tags.str.count("<")
    testAAdf_U.Tags = testAAdf_U.Tags.apply(lambda x: x.replace('<',''))
    testAAdf_U = testAAdf_U.drop('Tags', axis=1).join(testAAdf_U['Tags'].str.split('>', expand=True).stack().reset_index(level=1, drop=True).rename('Tags'))
    testAAdf_U = testAAdf_U.loc[testAAdf_U['Tags'] != '']
    testAAdf_U = testAAdf_U[testAAdf_U.Tags.isin(train_df.Tags)]
    testAAdf_U['Tag_wt'] = 1/testAAdf_U['N_Tags']
    
    return testAAdf_U



# Defining temporal discounting function

def temporal_dis(df):
 
    TempDis_df = df.copy()
    TempDis_df['Ans_Cnt_Wt'] = TempDis_df['wt_Cnt']/(1+TempDis_df['Months_Post'])
    
    return TempDis_df

def month_diff(a, b):
    return 12 * (a.dt.year - b.dt.year) + (a.dt.month - b.dt.month)



# Getting temporal discounted user's activity score at TAG level


def TAG_Udf(TAG_QnA_df,present_dt):
    
    TAG_Udf = TAG_QnA_df[TAG_QnA_df.OwnerUserId != ''].reset_index(drop=True)
    TAG_Udf['U_Tag'] = TAG_Udf['OwnerUserId'] + TAG_Udf['Tags']
    
    arr_slice = TAG_Udf[['U_Tag']].values
    unq,unqtags,counts = np.unique(arr_slice.astype(str),return_inverse=True,return_counts=True)
    TAG_Udf['Ttl_Ans'] = counts[unqtags]    

    TAG_Udf['Ttl_wt_Ans'] = TAG_Udf['wt_Cnt'].groupby(TAG_Udf['U_Tag']).transform('sum')

    
    TAG_Udf['ref_date'] = pd.Timestamp(present_dt)
    TAG_Udf['Months_Post'] = month_diff(TAG_Udf['ref_date'], TAG_Udf['CreationDate'])
    TAG_Udf['Days_Ans'] = (TAG_Udf['ref_date'] -  TAG_Udf['CreationDate']).dt.days#.astype('timedelta64[h]')


    TAG_U_TempD_df = temporal_dis(TAG_Udf)
    

    TAG_U_TempD_df['AC_Wt'] = TAG_U_TempD_df['Ans_Cnt_Wt'].groupby(TAG_U_TempD_df['U_Tag']).transform('sum')
    TAG_df = TAG_U_TempD_df.filter(items=['OwnerUserId','Tags','Ttl_Ans','AC_Wt']).drop_duplicates()

    return TAG_df



# Estimating predicted scores on all TAG based on temporal-MassDiffusion on bipartite network

def BiRecSys_train_df(train_df):
    
    ## Mass diffusion with temporal discounting
    df_matrix = train_df.pivot_table(values='AC_Wt', index='Tags', columns='OwnerUserId')

    df_User_Degree = df_matrix.count().reset_index()
    df_User_Degree.columns = ['UserId','Degree']
    
    df_Tag_Degree = df_matrix.count(axis=1).reset_index()
    df_Tag_Degree.columns = ['Tags','Degree']
    
    
    rating_matrix_iu = df_matrix.copy()
    rating_matrix_iu.fillna(0, inplace=True)
    
    rating_matrix_ui = df_matrix.T    
    rating_matrix_ui.fillna(0, inplace=True)
    
    #### Normalise with User Degree
    U_deg = (np.array(df_User_Degree.Degree))
    O_deg = (np.array(df_Tag_Degree.Degree))

    ## Mass diffusion
    item_normU_deg = rating_matrix_ui.divide(U_deg, axis=0)
    W_Item_diff = (rating_matrix_iu @ item_normU_deg).divide(O_deg, axis=1)
    

    pred_rating_Mdiff = W_Item_diff @ rating_matrix_iu
    
    return pred_rating_Mdiff.T
    
    
# Predicting users rank on questions in Test data    
def BiRecSys_eval_df(test_df,train_df,pred_rating):

    ### USERS
    eval_Udf = test_df[['Id','OwnerUserId']]

    all_Udf = train_df[['OwnerUserId','Ttl_Ans']]
    all_Udf = all_Udf.assign(Id=-9)
    all_Udf = all_Udf[['Id','OwnerUserId']].drop_duplicates().reset_index(drop = True)

    test_Udf = all_Udf.append(eval_Udf)
    test_Udf = test_Udf.assign(Ind=1)

    df_Utest = test_Udf.pivot_table(values='Ind', index='Id', columns='OwnerUserId')
    df_Utest = df_Utest.iloc[1:]
    df_Utest.fillna(0, inplace=True)
    
    
    ### TAGS
    tag_df = test_df.copy()
    tag_df = tag_df[tag_df.Tags.isin(train_df.Tags)]
    eval_Tagdf = tag_df[['Id','Tags','Tag_wt']]

    all_tags = train_df[['Tags','AC_Wt']]
    all_tags = all_tags.assign(Id=-9)
    all_tags = all_tags.assign(Tag_wt=1)

    all_tags = all_tags[['Id','Tags','Tag_wt']].drop_duplicates().reset_index(drop = True)

    test_tag_df = all_tags.append(eval_Tagdf)
    df_Tagtest = test_tag_df.pivot_table(values='Tag_wt', index='Id', columns='Tags')
    df_Tagtest = df_Tagtest.iloc[1:]
    df_Tagtest.fillna(0, inplace=True)
    
    
    test_user = df_Tagtest @ pred_rating.T
    user_rank = test_user.rank(axis=1, ascending=False, method='first').astype(int)

    Quser_ranks = user_rank * df_Utest
    Quser_ranks['User_Rank'] = Quser_ranks.sum(axis=1)

    Quser_ranks['RR'] = 1/Quser_ranks['User_Rank']
    Quser_ranks['Prec_at_1'] = np.where(Quser_ranks['User_Rank'] < 2, 1, 0)
    Quser_ranks['Prec_at_3'] = np.where(Quser_ranks['User_Rank'] < 4, 1, 0)
    

    MRR = round(Quser_ranks['RR'].mean(),3)
    Prec = round(Quser_ranks['Prec_at_1'].mean(),3)
    HIT = round(Quser_ranks['Prec_at_3'].mean(),3)
    
    Quser_ranks.reset_index(inplace=True)
        
    return Quser_ranks,MRR,Prec,HIT


