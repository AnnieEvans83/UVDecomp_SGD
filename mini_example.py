import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sgd_AE import sgdescent

'''Please see SGD_Mini_Example.jpynb for documenation and explanations.'''

def load_mini():
    mini_headings= ['Users','Food','Ratings']
    mini_table= [['cat','chicken',5],['cat','veg',1],['dog','fries',5],['dog','pizza',5],\
    ['dog','chicken',5],['dog','veg',5],['fish','fries',2],['fish','veg',3],['turtle','fries',1],\
    ['turtle','pizza',1],['turtle','veg',5],['wolf','pizza',3],['wolf','chicken',5],\
    ['cow','veg',4],['chicken','fries',3],['chicken','pizza',3],['chicken','chicken',3],['chicken','veg',3]]
    mini_df=pd.DataFrame(mini_table,columns=mini_headings)
    return mini_df

df = load_mini()

mini_model = sgdescent(df, latent_k = 4, lambda_UV =0.01, lambda_bias=0.01, n_inter=220,alpha_rate= 0.01)
df_Test=pd.DataFrame([['cat','pizza',3],['cow','fries',2]],columns=['Users','Food','Ratings'])
mini_model.Test(df_Test)
mini_model.fit()

# #New user squirrel rates chicken a 4
df_row = pd.DataFrame([['squirrel','chicken',5],['squirrel','pizza',2]],columns=['Users','Food','Ratings'])
mini_me.predict(df_row)
