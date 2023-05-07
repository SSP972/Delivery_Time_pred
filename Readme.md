
# Delivery Time Prediction


This project aims to build a predictive model to estimate the "Time to deliver food order". Raw data is provided in the form of a CSV file, which undergoes data transformation and preprocessing using feature engineering techniques. The final model is trained on this processed data and stored in a pickle file.
## Flowchart

![Project flowchart](https://github.com/SSP972/Delivery_Time_pred/blob/main/Live_img/Flowchart_DTP.png)


## Model_details
 
* Linear Regression: A linear model that assumes a linear relationship between the input features and the output variable. It tries to fit a straight line that best fits the data by minimizing the sum of the squared differences between the predicted and actual values.

* Lasso: A linear model that performs L1 regularization, which adds a penalty term to the loss function that encourages sparse solutions (i.e., sets some coefficients to zero). This helps to prevent overfitting and improve model interpretability.

* Ridge: A linear model that performs L2 regularization, which adds a penalty term to the loss function that shrinks the coefficients towards zero. This helps to prevent overfitting and improve model generalization.

* Elasticnet: A linear model that combines both L1 and L2 regularization, which adds a penalty term that is a combination of the L1 and L2 norms. This helps to balance the benefits of L1 and L2 regularization and can be particularly useful when there are many features.

* XGBoost: A tree-based ensemble model that uses a gradient boosting algorithm to iteratively add decision trees to the model, each one correcting the errors of the previous trees. It incorporates several advanced features such as regularization, missing value imputation, and parallel processing, which help to improve model performance.

* SVR: A non-linear model that uses a kernel function to transform the input features into a higher-dimensional space, where a linear regression model is fit to the transformed data. It tries to find the hyperplane that maximizes the margin between the support vectors and the decision boundary. It is particularly useful when there are non-linear relationships between the input features and the output variable.

## Performance matrics
* Root Mean Squared Error (RMSE): Measures the average deviation between the predicted and actual values, with larger errors being penalized more heavily than smaller errors. A lower RMSE indicates better model performance.

* Mean Absolute Error (MAE): Measures the average absolute difference between the predicted and actual values, with each error being weighted equally. A lower MAE indicates better model performance.

* R-squared score: Measures the proportion of variance in the target variable that can be explained by the input features, with a score of 1 indicating a perfect fit and a score of 0 indicating no relationship. A higher R-squared score indicates better model performance.

    | Model | RMSE | MAE | R-squared |
    |-------|------|-----|-----------|
    | Linear Regression | 6.692 | 5.282 | 0.482 |
    | Lasso | 7.167 | 5.708 | 0.406 |
    | Ridge | 6.692 | 5.282 | 0.482 |
    | Elasticnet | 7.243 | 5.798 | 0.393 |
    | XGBoost | 4.001 | 3.177 | 0.815 |
    | SVR | 6.556 | 5.123 | 0.503 |


## Screenshoots

Front page:

![Front page](https://github.com/SSP972/Delivery_Time_pred/blob/main/Live_img/front.jpg)


Prediction data:

![Prediction data](https://github.com/SSP972/Delivery_Time_pred/blob/main/Live_img/input.jpg)


Result page:

![Result page](https://github.com/SSP972/Delivery_Time_pred/blob/main/Live_img/result.jpg)



## Deployment

AWS Beanstack

    python 3.8

## Work in Progress

* Optimization