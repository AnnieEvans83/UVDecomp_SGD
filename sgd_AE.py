import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from random import randint
import random



class sgdescent(object):
    def __init__(self, df,latent_k = 10, lambda_UV =0.01, lambda_bias=0.01, n_inter=100, alpha_rate= .01):
        '''
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        UVD + SGD High Level
        Re(mxn) ~= U(mxk) V(kxn)
        1) Randomize U, V
        2) Randomly choose a (i,j)
        3) Calculate re
        4) Update U[i,:], V[:,j], b', b* using partial derivatives
        5) Repeat
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        INPUT:
        df= Pandas DataFrame with the columns [Users, Items, int:Ratings]
        latent_k = Parameter that is an integer of latent features
        lambda_UV = Regularization parameter on U and V
        lambda_bias = Regularization parameter on bias terms b' and b*
        n_iters = Integer of the number of iterations of SGD to execute
        alpha_rate= Parameter controling the learning rate
        OUTPUT: NONE
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        ATTRIBUTES:
        user_map and user_rev= Maps Row index to Users
        item_map and item_rev= Maps Column index to Items
        U= Numpy Matrix dimensions m(users) x latent_k
        V= Numpy Matrix dimensions latent_k x n(items)
        R= Numpy Matrix dimensions with the original ratings
        Re=Numpy Matrix dimensions with the model predicted ratings
        '''
        self.df=df
        self.mean_ = df.iloc[:,2].mean()
        self.latent_k=latent_k
        self.lambda_UV=lambda_UV
        self.lambda_bias=lambda_bias
        self.n_inter=n_inter
        self.alpha=alpha_rate
        self.user_map,self.user_rev,self.item_map,self.item_rev = self.map_it(self.df)
        self.U = np.random.rand(len(self.user_map),self.latent_k)
        self.V = np.random.rand(self.latent_k,len(self.item_map))
        self.item_bias=np.random.rand(len(self.item_map))
        self.user_bias=np.random.rand(len(self.user_map),1)
        self.MSE=[]
        self.MSE_Test =[]

    def Test(self,df):
        '''INPUT: Pandas DataFrame with the columns [Users, Items, int:Ratings]
        OUTPUT: NONE
        Stores Test database in self.df_Test, used in scoring model'''
        self.df_Test= df

    def map_it(self,df):
        '''INPUT:Pandas DataFrame with the columns [Users, Items, int:Ratings]
        OUTPUT:
        user_map and user_rev= Maps Row index to Users
        item_map and item_rev= Maps Column index to Items'''
        headers=self.df.columns.values
        user_map={}
        user_rev={}
        i=0
        for user in self.df[headers[0]].unique():
            user_map[user]=i
            user_rev[i]=user
            i+=1
        item_map={}
        item_rev={}
        j=0
        for item in self.df[headers[1]].unique():
            item_map[item]=j
            item_rev[j]=item
            j+=1
        return user_map,user_rev,item_map,item_rev

    def print_matrix(self,M_):
        '''INPUT: Matrix
        OUTPUT: NONE
        NOTE: This will Crash your computer if the Matrix is too large.
        print a  readable df'''
        M_= np.around(M_,decimals=1)
        item_labels = [self.item_rev[j] for j in range(len(self.item_rev))]
        user_labels = [self.user_rev[i] for i in range(len(self.user_rev))]
        self.df_M=pd.DataFrame(M_,columns=item_labels)
        self.df_M.insert(0,'Users',user_labels)
        print self.df_M

    def plot_MSE(self):
        x=np.linspace(0,self.n_inter-self.mse_inter,len(self.MSE))
        Training=self.MSE
        Test=self.MSE_Test
        plt.scatter(x,Training,c='b')
        plt.scatter(x,Test,c='r')
        plt.xlabel('SGD Iterations')
        print 'K: ',self.latent_k
        print 'Lamda UV: ' ,self.lambda_UV
        print 'Lamda bias: ' , self.lambda_bias
        print 'Iterations: ', self.n_inter
        print 'Learning rate alpha: ', self.alpha
        print 'Training = blue & Test = red'
        plt.show()


    def fit(self,mse_inter):
        '''
        INPUT:
        R: Matrix (numpy array of arrays)
        _iter: Integer for number of times to runs
        OUTPUT: NONE
        '''
        self.mse_inter=mse_inter
        for n in range(self.n_inter):
            (i,j,Rating_ij) = self.notnull()
            eRror= (Rating_ij-self.mean_-self.user_bias[i][0]-self.item_bias[j]-np.dot(self.U[i],self.V[:,j]))
            self.O_OUVij(i,j,eRror)
            self.O_Obibj(i,j,eRror)
            if (n%self.mse_inter) == 0:
                self.meanSquare(n)
                self.meanSquare_Test(n)
        print
        return self.MSE, self.MSE_Test

    def O_OUVij(self,i,j,eRror):
        '''INPUT:
        OUTPUT:
        Loop over and Update the ith row in U and the jth rowin V
        with the partial derivative of the loss function'''
        for f in range(self.latent_k):
            Uold = self.U[i][f]
            Vold = self.V[f][j]
            self.U[i][f]=Uold-self.alpha*(-2*eRror*Vold+self.lambda_UV*2*Uold)
            self.V[f][j]=Vold-self.alpha*(-2*eRror*Uold+self.lambda_UV*2*Vold)


    def O_Obibj(self,i,j,eRror):
        '''INPUT:
        OUTPUT:
        Update the ith element in user_bias and the jth element in
        item_bias with the partial derivative of the loss function'''
        b_iold=self.user_bias[i]
        b_jold=self.item_bias[j]
        self.user_bias[i][0]=b_iold-self.alpha*(-2*eRror+self.lambda_bias*2*b_iold)
        self.item_bias[j]=b_jold-self.alpha*(-2*eRror+self.lambda_bias*2*b_jold)

    def notnull(self):
        '''
        INPUT: None
        OUTPUT: (i,j,Rating_ij)
        '''
        (row,column) =self.df.shape
        r = randint(0,row-1)
        i= self.user_map[self.df.iloc[r,0]]
        j= self.item_map[self.df.iloc[r,1]]
        Rating_ij = self.df.iloc[r,2]
        return (i,j,Rating_ij)

    def notnull_row(self,df_row):
        '''
        INPUT: df
        OUTPUT: (j)
        '''
        (row,column) =df_row.shape
        r = randint(0,row-1)
        j= self.item_map[self.df.iloc[r,1]]
        Rating_ij = df_row.iloc[r,2]
        return (j,Rating_ij)

    def predict(self,df_row, top_num =1):
        '''INPUT: df of new ratings
        OUTPUT: Top (top_num) best choices'''
        #Assuming new user
        U_row = np.random.rand(self.latent_k)
        user_bias_row = np.random.rand(1)
        (row,column) =self.df.shape
        alpha = .01
        for n in range(self.n_inter/row):
            (j, Rating_ij) = self.notnull_row(df_row)
            eRror= (Rating_ij-self.mean_-user_bias_row[0]-self.item_bias[j]-np.dot(U_row,self.V[:,j]))
            for f in range(self.latent_k):
                Uold = U_row[f]
                Vold = self.V[f][j]
                U_row[f]=Uold-alpha*(-2*eRror*Vold+self.lambda_UV*2*Uold)
            b_iold=user_bias_row[0]
            b_jold=self.item_bias[j]
            user_bias_row[0]=b_iold-alpha*(-2*eRror+self.lambda_bias*2*b_iold)
        predictions = np.add(np.dot(U_row,self.V),user_bias_row)
        predictions = np.add(predictions,self.item_bias)
        predictions = np.add(predictions, self.mean_)
        #Add the predictions to a dictionary and sort by vales
        Rated = df_row.iloc[:,1].values
        Pre_list= []
        for c in range(len(predictions)):
            if self.item_rev[c] not in Rated:
                Pre_list.append([self.item_rev[c],predictions[c]])
        pre_df =pd.DataFrame(Pre_list,columns=['Top_Items', 'Relevance_Score'])
        print df_row
        print 'Predictions'
        pre_df =pre_df.sort_values(['Relevance_Score'],ascending=False).head(top_num)
        print pre_df
        return pre_df.reset_index(drop=True)

    def meanSquare(self,n_it):
        '''
        INPUT: Pandas DataFrame with the columns [Users, Items, int:Ratings]
        OUTPUT: Mean Square Error as an integer
        '''
        (row,column) =self.df.shape
        sum_error= 0
        for r in range(row):
            i= self.user_map[self.df.iloc[r,0]]
            j= self.item_map[self.df.iloc[r,1]]
            Rating_ij = self.df.iloc[r,2]
            sum_error += (Rating_ij-self.mean_-self.user_bias[i]-self.item_bias[j]-np.dot(self.U[i],self.V[:,j]))**2
        mse = sum_error/row
        self.MSE.append(mse)
        print 'TRAIN MSE: {}, Iteration {} '.format(mse,n_it)
        return


    def meanSquare_Test(self,n_it):
        '''
        INPUT: Pandas DataFrame with the columns [Users, Items, int:Ratings]
        OUTPUT: Mean Square Error as an integer
        '''
        self.missing=[]
        (row,column) =self.df_Test.shape
        sum_error_T= 0
        for r in range(row):
            if self.df_Test.iloc[r,0] in self.user_map and self.df_Test.iloc[r,1] in self.item_map:
                i= self.user_map[self.df_Test.iloc[r,0]]
                j= self.item_map[self.df_Test.iloc[r,1]]
                Rating_ij = self.df_Test.iloc[r,2]
                sum_error_T += (Rating_ij-self.mean_-self.user_bias[i]-self.item_bias[j]-np.dot(self.U[i],self.V[:,j]))**2
            else:
                self.missing.append([self.df_Test.iloc[r,0],self.df_Test.iloc[r,1],self.df_Test.iloc[r,2]])
        mse_T = sum_error_T/row
        self.MSE_Test.append(mse_T)
        print 'TEST MSE: {}, Iteration {} , Missing {}'.format(mse_T,n_it,len(self.missing))
        return
