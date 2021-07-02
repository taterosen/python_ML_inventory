#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:02:15 2021

@author: taterosen

twitter sentiment analysis
"""

##import os 
##os.system("python twitter_search.py brexit -c 180")

import pandas as pd

#load data downloaded from twitter
df = pd.read_csv('result2.csv', index_col=0)

from textblob import TextBlob

for index, row in df.iterrows():
    tweet = row["text"]
    text = TextBlob(tweet)
    
    #sentiment analysis
    subjectivity = text.sentiment.subjectivity  #runs from 0 to 1
    polarity = text.sentiment.polarity          #runs from -1 to 1
    print(tweet, subjectivity, polarity)
