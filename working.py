# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:32:40 2019

@author: Jeff
"""
import pandas as pd

LETTERS="ABCDE"

test_words = [
        "DAA",
        "DAB",
        "DBC",
        "DAD",
        "DAE",
        "DBF",
        "BBB",
        "BBC",
        "BED",
        "CBB"]

class StatisticalProb:
    def __init__(self, words): 
        df=pd.DataFrame({
                'first' : [t[0] for t in words],
                'second' : [t[1] for t in words]
                    })
        df['one']=1
        
        # First letter probabilities
        cnts1=df[["first","one"]].groupby("first").count()
        p1=cnts1/cnts1.sum()
        for l in LETTERS:
            if l not in p1.index:
                p1.loc[l,"one"]=0
        
        p1=p1.loc[[k for k in LETTERS],:]        
        self._first_prob=p1.copy()

        
        # Second letter probabilities conditoned on first
        cnts2=df.pivot_table(index='first', columns='second', aggfunc='count')
        cnts2.columns=[t[1] for t in cnts2.columns]
         
        for l in LETTERS:
            if l not in cnts2.index:
                cnts2.loc[l,:]=0
            if l not in cnts2.columns:
                cnts2.loc[:,l]=0

        cnts2=cnts2.fillna(0)
        p2=cnts2/cnts2.sum().sum()
        p2=p2.loc[[k for k in LETTERS],[k for k in LETTERS]]   
        self._second_prob=p2.copy()
        

    def get_first_prob(self, letter):
        return self._first_prob.loc[letter].values                        
       
        
    def get_second_prob(self, letter):
        return self._second_prob.loc[letter,:].values                
        
sp=StatisticalProb(test_words)
this_p1=sp.get_first_prob("D")
this_p2=sp.get_second_prob("D")
