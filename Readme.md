# Parallelized predictive analysis of Video Game Sales

The code include predictive analysis of video game sales using Support Vector Regressor, Random Forest, Gradient Booster, MLP Regressor (Neural Network) and KNN, analyzing the performance of these models, and finally evaluating the parallelization capabilities of the model with highest weighted score.

## Instructions:

- Run ml_model_for_game_mode.ipynb for generating game mode attribute for the missing data. (final dataset already has game mode)
- Open final_model.ipynb
- Run data preprocessing section to: 
  - Read the dataset
  - Remove outliers
  - Visualize dataset
- Run correlation of variables to see the correlation matrix
- Run the feature engineering section which includes:
  - Handling missing values (either by dropping or filling with median or mode)
  - Keeping only relevant platforms
  - Handling categorical variable by creating dummies
- Run the next section to divide data into variables and target.
- Split the data into training and testing data.
- Run the scaling data section to scale the target variable
- Run grid preparation section to define hyper parameter grids for various models.
- Run the weighted score section to define weights for each metric.
- For each model, run the first section to train the model, make predictions and calculation regression metrics. Run the second section to calculate the overall weighted score.
- Run the model comparison section to visualize the comparison of models.
- Run the feature importance section to get the feature importance according to gradient booster.
- Run the parallelization section to parallelize gradient boosting.
- Run the comparison section to compare parallel and serial execution of gradient boosting.
