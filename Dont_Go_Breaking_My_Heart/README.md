# Don't Go Breaking My Heart
A Duet in Machine Learning

This code is in reponse to the "Heart Failure" challenge on Kaggle the description for which can be found at https://www.kaggle.com/andrewmvd/heart-failure-clinical-data


"Right from the start..."
My approach is to test the researchers' hypothesis that not only can heart failure be accurately predicted using just serum creatinine and ejection fraction, but models using the complete set of variables in their dataset actually perform worse than just the two variable model.

I aim to test this by not only comparing accuracy and F1 scores across a handful of classical and ensemble machine learning models, but also importantly by studying the variation of these two scores for each respective model during cross-validation. My central question is does the two-variable model generalize just as well, if not better, and produce more consistent accuracy and F1 scores than the model with all variables. As a follow-up, if the two-variable model does have more variation, is it acceptable because the scores far exceed the all variable model anyway?

I then move on to understanding if there is a distinct difference in how accurate the probability scores are for decision predictions for the two-factor model versus the model containing all variables. Probability scores can be useful measures to guide physicians trust in a model decision but obviously only if they are accurate.

I conclude by creating a Monte Carlo function that tests the significance of each model's results without assuming a distribution type.

Two notes:

My code and descriptions will use the word "focus" to describe datasets and models where only ejection fraction and serum creatinine are used.
The model that has all variables actually does not include the variable, "time," which the Kaggle page describes as "follow up period (days)" but the research report does not explain very well. I excluded this varariable completely from all of my analysis because it seems very likely that this isn't information that a physician would know during the screening where all the other data points are recorded. Additionally, the "time" variable is extremely predictive and the correlation seems to suggest that a shorter time correlates with higher mortality, which doesn't intuitively make a lot of sense.
