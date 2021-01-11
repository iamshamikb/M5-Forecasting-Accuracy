# M5-Forecasting-Accuracy
Solved as a Project.
Predicting Walmart sales - A Solution to Kaggle M5 Forecasting Accuracy Competition
Find the Blog - https://iamshamikb.wordpress.com/2020/11/22/a-solution-to-kaggle-m5-forecasting-accuracy-competition-2/

It is a Time Series Forecasting problem. Predicted Sales of next 28 days from past 5 years sales data for 3000+ products across 10 Walmart Stores.
Used XGBoost, LGBM, LSTM, RMSE Metric.

The submission file for checking the results achieved is given here as - submissible_ML_7.5.T2.csv.

The Final Model uses LGBMRegressor and a Dept Level Aggregation.
The other approaches tried involved several Feature Engineering, Dept Level and Store Level Aggregation, XGBoost, Treating the problem as Timeseries Problem and Treating it as a Regression problem. 
The Metric used was RMSE.

An extensive Exploratory Data Analysis was performed in EDA.ipynb and the insights are given alongwith the plots.
All Approaches Tried.ipynb contains the ML models. Various aggregation and Feature Engineering were tried.
Deep Learning Models are given in separate ipynbs.

The blog link above has a walkthrough video and detailed explanation of everything tried.
