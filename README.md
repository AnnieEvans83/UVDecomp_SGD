#Problem Statement:
Currently there is not an effective open sourced Matrix Factorization for Recommendations Systems available for Data Scientists.

#Solution:
I wrote a Matrix Factorization library using Python, numPy, and Pandas.

#Mini Example: Recommending Food (items) to Animals (Users)
We are going to over a mini dataset that I created to test my code.
I did not actually poll any animals to determine there food preferences.
![User Cat](/user_cat_chicken.png?raw=true "User: Cat")

#How do we fill in the unknowns?
![Rating Matrix](/cat_rating_matrix.png?raw=true "Rating Matrix")

#UVDecompization with Stochastic Gradient Descent
![UVDecomp + SGD](/predict_formula.png?raw=true "UVDecomp + SGD")
#Steps:
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
#Model Mean Square Error Training and Test set Results:
![Model Results](/mini_model_results.png?raw=true "Model Results")

#SGD Model Rating Prediction Table:
![Prediction Table](/mini_prediction_table.png?raw=true "Prediction Table")

#SGD Model Recommendations on a new User:
![New User](/squirrel.png?raw=true "New User")

##Real World Application:
#Book Recommender System
![Book Recommender](/book_intro.png?raw=true "Book Recommender")

#Model Mean Square Error Training and Test set Results:
![Model Results](/book_Model_Results.png?raw=true "Model Results")

#Examples of Prediction Results:
