

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import xml.etree.ElementTree as ET
import pandas as pd

from tqdm import tqdm



t = input("Enter the CQA folder downloaded from SE archieve (example::askubuntu.com)")

print("StackExchange Community:",t)

file = open('Raw_Data/{}/Posts.xml'.format(t), 'r')
posts = {"posts":[]}

## Parsing XML file
for event, elem in tqdm(ET.iterparse(file)):    

    if event == 'end':
        tag = {}
        if elem.get("Id") is not None:


            post = {}
            post["Id"] = elem.attrib['Id']
            post["PostTypeId"] = elem.attrib['PostTypeId']
            post["CreationDate"] = elem.attrib['CreationDate']
            post["Score"] = elem.attrib['Score']
            post["CommentCount"] = elem.attrib['CommentCount']

            if elem.get("OwnerUserId") is None:
                post["OwnerUserId"] = ""
            else:    
                post["OwnerUserId"] = elem.attrib['OwnerUserId']


            if elem.attrib['PostTypeId'] == '1':

                post["Tags"] = elem.attrib['Tags']
                post["AnswerCount"] = elem.attrib['AnswerCount']
                post["ViewCount"] = elem.attrib['ViewCount']
                if elem.get("AcceptedAnswerId") is not None:
                    post["AcceptedAnswerId"] = elem.attrib['AcceptedAnswerId']

            elif elem.attrib['PostTypeId'] == '2': 

                post["ParentId"] = elem.attrib['ParentId']

            else:
                post["Body"] = elem.attrib['Body']


            posts["posts"].append(post)

            elem.clear()

df_posts = pd.DataFrame(posts["posts"])

df_posts.to_feather('PreprocessedData/{}_Posts.ft'.format(t))  # dataset to save

