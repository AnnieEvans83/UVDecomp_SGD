###File Explanation

README.md is a high level overview of my Capstone project.

sgd_AE.py is the UVDecomposition using Stochastic Gradient Descent class that I wrote using Python, NumPY, and Pandas.

SGD_Mini_Example.ipynb is a working example on how to use the UVDecomposition using Stochastic Gradient Descent class using the same animal example from the Capstone presentation.

Handwritten calcs.pdf is a scan of my hand calcuations deriving the partial derivaites for the Stochastic Gradient Descent

#Outline

*Introduction

*Mini Example explaining how the model works

*Book Recommendation overview

*Next Steps for the Book Recommender


#Introduction:

Everyone in my family is a big reader. We often recommend and share books with each other despite each us having different taste in genres. We also occasionally get in a rut where no one has read anything awesome enough to share with the rest of the group. Therefore for my Capstone project and beyond I wanted to build a book recommendation system that would let you rate books and then have the model recommend books for you based on your past preferences. There was a barrier though, currently there is not an effective open sourced Matrix Factorization for Recommendations Systems available for Data Scientists.

###Solution:
I wrote a Matrix Factorization library using Python, numPy, and Pandas.

#Mini Example explaining how the UVDecomposition and Stochastic Gradient Descent works:

##Mini Example: Recommending Food (items) to Animals (Users)
We are going to over a mini dataset that I created to test my code.
I did not actually poll any animals to determine there food preferences.
![User Cat](/user_cat_chicken.png?raw=true "User: Cat")

##How do we fill in the unknowns?
![Rating Matrix](/cat_rating_matrix.png?raw=true "Rating Matrix")

#SOLVE IT WITH MATH!!

##UVDecompization with Stochastic Gradient Descent
![UVDecomp + SGD](/predict_formula.png?raw=true "UVDecomp + SGD")
###Steps:
1) Initiate Matrix U, Matrix V, user & item bias vectors.
(Random values between 0 and 1)

2)Choose a random user and random item with a rating.

3)Predict the rating using the Matrix U, Matrix V and the user & bias vectors.

4)Update the terms in the Matrix U, Matrix V and the user & bias vectors using
Gradient Descent.

a) Partial derivatives:

![Partial derivatives](/partial_derivatives.png?raw=true "Partial derivatives")

b) Update terms:

![Update Terms](/update_terms.png?raw=true "Update Terms")

5) Repeat until adequate convergence.

##Model Mean Square Error Training and Test set Results:
![Model Results](/mini_model_results.png?raw=true "Model Results")

##SGD Model Rating Prediction Table:
![Prediction Table](/mini_prediction_table.png?raw=true "Prediction Table")

##SGD Model Recommendations on a new User:
![New User](/squirrel.png?raw=true "New User")

#Real World Application:
![Book Recommender](/book_intro.png?raw=true "Book Recommender")

##Model Mean Square Error Training and Test set Results:
![Model Results](/book_Model_Results.png?raw=true "Model Results")

##Examples of Prediction Results:
![Prediction Results](/book_predict_1.png?raw=true "Book Recommendations")

#Next Steps

##Web Application with FLASK on a EC2 on AWS

##SQL database to store users ratings

##Rest API

