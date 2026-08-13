# Customer Purchase Prediction

## Overview

This is my first end-to-end machine learning project. The goal is to predict whether a customer will make another purchase based on customer behavior and spending data.

## Technologies

- Python
- Pandas
- Scikit-learn
- Logistic Regression

## Machine Learning Process

1. Created and loaded the customer dataset
2. Explored the data using Pandas
3. Identified data-quality issues
4. Engineered new features
5. Selected features (`X`) and the target label (`y`)
6. Split the data into training and testing sets
7. Trained a Logistic Regression model
8. Made predictions on unseen test data
9. Compared predictions with the actual labels

## Features

The model uses information such as:

- Age
- Annual Income
- Purchases
- Total Spent
- Cart Value
- Items in Cart
- Time on Site
- Average Spend Per Purchase
- High Value Customer
- Big Shopper
- Customers Over 30

## Feature Engineering

I created additional features from the original data, including:

- `Average_Spend_Per_Purchase`
- `High_Value_Customer`
- `Big_Shopper`
- `Customers_over_30`
- `Older30&Spent_over$1000`

## Model

I used Logistic Regression from Scikit-learn to predict whether a customer would buy again.

## Evaluation

The model was tested using a held-out test set and its predictions were compared with the actual labels.

Because this project uses a very small dataset for learning purposes, the evaluation result should not be interpreted as evidence of real-world model performance.

## What I Learned

Through this project, I learned:

- How to inspect a dataset
- How to identify suspicious data
- How to clean data
- How to filter Pandas DataFrames
- How to engineer features
- The difference between features and labels
- How train/test splitting works
- Why reproducibility matters
- How Logistic Regression is trained
- How to make predictions with a trained model
- Why model evaluation and data quality matter

## Limitations

This project uses a small, manually created dataset. A real-world version would require a much larger dataset, more extensive data cleaning, stronger evaluation methods, and comparison between multiple machine learning models.

## Project Status

Completed — Project 1