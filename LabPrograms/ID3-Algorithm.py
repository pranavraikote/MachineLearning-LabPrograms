#Import modules
import pandas as pd
from collections import Counter
import math

#Read dataset file
tennis = pd.read_csv('ID3.csv')
print("\n Given Play Tennis Data Set:\n\n", tennis)

#Calculation of Entropy
def entropy(alist):    
    c = Counter(x for x in alist)   
    instances = len(alist)  
    prob = [x / instances for x in c.values()]  
    return sum( [-p*math.log(p, 2) for p in prob] ) 

#Calculation of Information Gain
def information_gain(d, split, target):    
    splitting = d.groupby(split)
    n = len(d.index)
    agent = splitting.agg({target : [entropy, lambda x: len(x)/n] })[target] #aggregating
    agent.columns = ['Entropy', 'observations']
    newentropy = sum( agent['Entropy'] * agent['observations'] )
    oldentropy = entropy(d[target])
    return oldentropy - newentropy

def id3(sub, target, a):

    #Class of YES/NO
    count = Counter(x for x in sub[target])
    
    if len(count) == 1:
        #Next input data set, or raises StopIteration when EOF is hit
        return next(iter(count))  
    
    else:              
        gain = [information_gain(sub, attr, target) for attr in a] 
        print("Gain=",gain)
        maximum = gain.index(max(gain)) 
        best = a[maximum]
        print("Best Attribute:",best)
        tree = {best:{}} 
        remaining = [i for i in a if i != best]
                
        for val, subset in sub.groupby(best):
            subtree = id3(subset,target,remaining)
            tree[best][val] = subtree
        return tree

#Driver Code for generation of Decision Tree
names = list(tennis.columns)
print("List of Attributes:", names) 
names.remove('PlayTennis')  
print("Predicting Attributes:", names)

tree = id3(tennis,'PlayTennis',names)
print("\n\nThe Resultant Decision Tree is :\n")
print(tree)
