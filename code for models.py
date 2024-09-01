#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 06:44:20 2024

@author: mac
"""


###
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter       
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras.regularizers import l2
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
import openpyxl

# Define the path to your folder containing the CSV files
folder_path = '/Users/mac/Documents/Manchester/Dissertation/dataset'

# List all CSV files in the folder
files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Function to load a CSV file and add a company column
def load_data_with_company(filepath, company_name):
    df = pd.read_csv(filepath)
    df['Company'] = company_name
    return df

# Load and combine all files into a single DataFrame
combined_data = pd.concat([load_data_with_company(os.path.join(folder_path, file), os.path.splitext(file)[0]) for file in files], ignore_index=True)

# Display the combined dataframe
print(combined_data.head())
combined_data.to_csv('Combined_dataset.csv', index=True)

# Load the combined dataset
combined_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/dataset/Combined_dataset.csv')

# Display the first few rows of the dataset
print(combined_data.head())

# Display summary statistics
print(combined_data.describe())

# Generate descriptive statistics
desc_stats = combined_data.describe()

# Transpose the statistics
desc_stats_transposed = desc_stats.transpose()

# Export the transposed statistics to an Excel file
desc_stats_transposed.to_excel('/Users/mac/Documents/Manchester/Dissertation/dataset/Descriptive_Statistics_Transposed.xlsx')

print("Descriptive statistics exported successfully.")

# Display information about the dataset (e.g., column names, data types, non-null counts)
print(combined_data.info())

mean_values = combined_data.mean()
median_values = combined_data.median()
std_dev_values = combined_data.std()

# Display the calculated metrics
print("Mean Values:\n", mean_values)
print("\nMedian Values:\n", median_values)
print("\nStandard Deviation Values:\n", std_dev_values)

# Example: Calculate correlation matrix
correlation_matrix = combined_data.corr()

# Display the correlation matrix
print("\nCorrelation Matrix:\n", correlation_matrix)

# Check for missing values
missing_values = combined_data.isnull().sum()

# Display basic information about the dataset
print(combined_data.info())
print(combined_data.describe())
print("Missing values in each column:\n", missing_values)

# Drop unnecessary columns "Unnamed" 
combined_data = combined_data.drop(columns=['Unnamed: 0'])

# Drop unused columns Open and Close price of stock  
combined_data = combined_data.drop(columns=['Open', 'Close'])

# Check remaining columns
remaining_columns = combined_data.columns
print("Remaining columns:")
print(remaining_columns)

# Histograms of numerical columns
combined_data.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

# Box plots to detect outliers
numerical_columns = combined_data.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns) // 3 + 1, 3, i)
    sns.boxplot(y=combined_data[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = combined_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# KDE plots for numerical columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns) // 3 + 1, 3, i)
    sns.kdeplot(combined_data[col], shade=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Key statistics grouped by 'Company'
company_stats = combined_data.groupby('Company').describe().transpose()
print(company_stats)


##### XGBOOST 

# Load the combined dataset
combined_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/dataset/Combined_dataset.csv')

# Check for missing values
missing_values = combined_data.isnull().sum()

# Display basic information about the dataset
print(combined_data.info())
print(combined_data.describe())
print("Missing values in each column:\n", missing_values)

# Drop unnecessary columns "Unnamed" 
combined_data = combined_data.drop(columns=['Unnamed: 0'])

# Drop unused columns Open and Close price of stock  
combined_data = combined_data.drop(columns=['Open', 'Close'])

# Check remaining columns
remaining_columns = combined_data.columns
print("Remaining columns:")
print(remaining_columns)

# Set random seeds for reproducibility
np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

def train_and_evaluate_per_company(original_data, features):
    companies = original_data['Company'].unique()
    results = []

    for company in companies:
        print(f"\nProcessing company: {company}")

        # Filter data for the current company
        company_data = original_data[original_data['Company'] == company]

        # Split the data into train, validation, and test sets
        train_data = company_data[company_data['Date'] < '2022-01-01']
        val_data = company_data[(company_data['Date'] >= '2022-01-01') & (company_data['Date'] < '2023-01-01')]
        test_data = company_data[(company_data['Date'] >= '2023-01-01') & (company_data['Date'] <= '2023-12-31')]

        # Prepare the features and target
        X_train = train_data[features]
        y_train = train_data['Adj Close']
        X_val = val_data[features]
        y_val = val_data['Adj Close']
        X_test = test_data[features]
        y_test = test_data['Adj Close']

        # Normalize the data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [50, 100, 200],
            'alpha': [0, 0.1, 1],  # L1 regularization
            'lambda': [1, 0.1, 0.01],  # L2 regularization
            'random_state': [1]
        }

        # Initialize and perform GridSearchCV with time series split
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(estimator=xgb.XGBRegressor(),
                                   param_grid=param_grid,
                                   scoring='r2',
                                   cv=tscv,
                                   n_jobs=-1,
                                   verbose=1)

        # Fit the GridSearchCV
        grid_result = grid_search.fit(X_train_scaled, y_train)

        # Extract the best model
        best_model = grid_result.best_estimator_

        # Evaluate on the validation set
        y_pred_val = best_model.predict(X_val_scaled)
        mse_val, rmse_val, r2_val = calculate_metrics(y_val, y_pred_val)

        # Evaluate on the test set
        y_pred_test = best_model.predict(X_test_scaled)
        mse_test, rmse_test, r2_test = calculate_metrics(y_test, y_pred_test)

        # Save the results including predictions
        results.append({
            'Company': company,
            'Validation': {
                'MSE': mse_val,
                'RMSE': rmse_val,
                'R2': r2_val
            },
            'Test': {
                'MSE': mse_test,
                'RMSE': rmse_test,
                'R2': r2_test
            },
            'Predictions': {
                'Date': test_data['Date'].tolist(),
                'Actual': y_test.tolist(),
                'Predicted': y_pred_test.tolist()
            }
        })

    return results
# Convert results to DataFrames
def results_to_dataframe(results):
    metrics_data = []
    predictions_data = []

    for result in results:
        company = result['Company']
        
        # Metrics data
        metrics_data.append({
            'Company': company,
            'Set': 'Validation',
            'MSE': result['Validation']['MSE'],
            'RMSE': result['Validation']['RMSE'],
            'R2': result['Validation']['R2']
        })
        metrics_data.append({
            'Company': company,
            'Set': 'Test',
            'MSE': result['Test']['MSE'],
            'RMSE': result['Test']['RMSE'],
            'R2': result['Test']['R2']
        })

        # Predictions data
        for date, actual, predicted in zip(result['Predictions']['Date'], result['Predictions']['Actual'], result['Predictions']['Predicted']):
            predictions_data.append({
                'Company': company,
                'Date': date,
                'Actual': actual,
                'Predicted': predicted
            })

    metrics_df = pd.DataFrame(metrics_data)
    predictions_df = pd.DataFrame(predictions_data)

    return metrics_df, predictions_df

# Define features with and without sentiment
features_with_sentiment = ['sentiment_score', 'Annual_Report_Sentiment_Score',
                           'SMA', 'EMA', 'RSI', 'Upper_BB', 'Lower_BB', 
                           'Stochastic_Oscillator', 'ATR', 'MACD', 'MACD_Signal', 
                           'MACD_Hist', 'OBV', 'Lag_1', 'Lag_2', 'Lag_3', 
                           'Lag_7', 'Lag_14', 'Lag_30', 'Day', 'Month', 'Year']

features_without_sentiment = ['SMA', 'EMA', 'RSI', 'Upper_BB', 'Lower_BB', 
                              'Stochastic_Oscillator', 'ATR', 'MACD', 'MACD_Signal', 
                              'MACD_Hist', 'OBV', 'Lag_1', 'Lag_2', 'Lag_3', 
                              'Lag_7', 'Lag_14', 'Lag_30', 'Day', 'Month', 'Year']

# Assuming you have combined_data and features_with_sentiment defined
results_with_sentiment = train_and_evaluate_per_company(combined_data, features_with_sentiment)
metrics_df_with_sentiment, predictions_df_with_sentiment = results_to_dataframe(results_with_sentiment)

# Export to Excel
with pd.ExcelWriter('XGB_company_results_with_sentiment.xlsx') as writer:
    metrics_df_with_sentiment.to_excel(writer, sheet_name='Metrics', index=False)
    predictions_df_with_sentiment.to_excel(writer, sheet_name='Predictions', index=False)
print("Results with sentiment have been exported to 'XGB_company_results_with_sentiment.xlsx'.")  
  
results_without_sentiment = train_and_evaluate_per_company(combined_data, features_without_sentiment)
metrics_df_without_sentiment, predictions_df_without_sentiment = results_to_dataframe(results_without_sentiment)
with pd.ExcelWriter('XGB_company_results_without_sentiment.xlsx') as writer:
    metrics_df_without_sentiment.to_excel(writer, sheet_name='Metrics', index=False)
    predictions_df_without_sentiment.to_excel(writer, sheet_name='Predictions', index=False)
print("Results without sentiment have been exported to 'XGB_company_results_without_sentiment.xlsx'.")

#### VISUALIZATION

# Define file paths
file_path_with_sentiment = '/Users/mac/Documents/Manchester/Dissertation/comparison model/XGB_company_results_with_sentiment.xlsx'
file_path_without_sentiment = '/Users/mac/Documents/Manchester/Dissertation/comparison model/XGB_company_results_without_sentiment.xlsx'

# Load data from Excel files
metrics_with_sentiment = pd.read_excel(file_path_with_sentiment, sheet_name='Metrics')
predictions_with_sentiment = pd.read_excel(file_path_with_sentiment, sheet_name='Predictions')

metrics_without_sentiment = pd.read_excel(file_path_without_sentiment, sheet_name='Metrics')
predictions_without_sentiment = pd.read_excel(file_path_without_sentiment, sheet_name='Predictions')

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

# Filter for test set
test_with_sentiment = metrics_with_sentiment[metrics_with_sentiment['Set'] == 'Test']
test_without_sentiment = metrics_without_sentiment[metrics_without_sentiment['Set'] == 'Test']

# Extract company names and R2 values of the test set
companies = test_with_sentiment['Company'].values
r2_with_sentiment = test_with_sentiment['R2'].values
r2_without_sentiment = test_without_sentiment['R2'].values

# Create bar plot
x = np.arange(len(companies))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - width/2, r2_with_sentiment, width, label='With Sentiment')
rects2 = ax.bar(x + width/2, r2_without_sentiment, width, label='Without Sentiment')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Company')
ax.set_ylabel('R² (Test Set)')
ax.set_title('Comparison of R² (Test Set) with and without Sentiment')
ax.set_xticks(x)
ax.set_xticklabels(companies, rotation=90)
ax.legend()

# Add R² value annotations on the bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)

fig.tight_layout()

plt.show()


# Extract company names and R2 values of the test set
companies = test_with_sentiment['Company'].values
r2_with_sentiment = test_with_sentiment['R2'].values
r2_without_sentiment = test_without_sentiment['R2'].values

# Sort by R2 values of with sentiment in descending order
sorted_indices = np.argsort(r2_with_sentiment)
companies = companies[sorted_indices]
r2_with_sentiment = r2_with_sentiment[sorted_indices]
r2_without_sentiment = r2_without_sentiment[sorted_indices]

# Create horizontal bar plot
fig, ax = plt.subplots(figsize=(12, 8))
y = np.arange(len(companies))  # the label locations

rects1 = ax.barh(y + 0.2, r2_with_sentiment, 0.4, label='With Sentiment')
rects2 = ax.barh(y - 0.2, r2_without_sentiment, 0.4, label='Without Sentiment')

# Add some text for labels, title and custom y-axis tick labels, etc.
ax.set_xlabel('R² (Test Set)')
ax.set_title('Comparison of R² (Test Set) with and without Sentiment')
ax.set_yticks(y)
ax.set_yticklabels(companies)
ax.legend()

# Add R² value annotations on the bars
def add_labels(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.2f}',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center')

add_labels(rects1)
add_labels(rects2)

fig.tight_layout()

plt.show()

# Function to plot predictions for each company
def plot_predictions_for_each_company(predictions_with_sentiment, predictions_without_sentiment):
    # Ensure 'Date' is in datetime format
    predictions_with_sentiment['Date'] = pd.to_datetime(predictions_with_sentiment['Date'])
    predictions_without_sentiment['Date'] = pd.to_datetime(predictions_without_sentiment['Date'])
    
    companies = predictions_with_sentiment['Company'].unique()

    for company in companies:
        plt.figure(figsize=(14, 7))

        # Filter data for the current company
        company_preds_with_sentiment = predictions_with_sentiment[predictions_with_sentiment['Company'] == company]
        company_preds_without_sentiment = predictions_without_sentiment[predictions_without_sentiment['Company'] == company]

        # Plot actual values
        plt.plot(company_preds_with_sentiment['Date'], company_preds_with_sentiment['Actual'], label='Actual', color='blue', linestyle='--')

        # Plot predictions with sentiment
        plt.plot(company_preds_with_sentiment['Date'], company_preds_with_sentiment['Predicted'], label='Predicted with Sentiment', color='red')

        # Plot predictions without sentiment
        plt.plot(company_preds_without_sentiment['Date'], company_preds_without_sentiment['Predicted'], label='Predicted without Sentiment', color='green')

        plt.title(f'Predictions vs Actuals for {company} - XGBoost Model')
        plt.xlabel('Date')
        plt.ylabel('Adj Close')
        plt.legend(loc='upper left')
        
        # Set monthly grid
        ax = plt.gca()
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
        plt.grid(True)
        plt.gcf().autofmt_xdate()  # Auto format date labels

        plt.tight_layout()
        plt.show()
        
# Plot predictions for each company
plot_predictions_for_each_company(predictions_with_sentiment, predictions_without_sentiment)

# Function to calculate average predictions and plot
def plot_average_predictions(predictions_with_sentiment, predictions_without_sentiment):
    # Ensure 'Date' is in datetime format
    predictions_with_sentiment['Date'] = pd.to_datetime(predictions_with_sentiment['Date'])
    predictions_without_sentiment['Date'] = pd.to_datetime(predictions_without_sentiment['Date'])

    # Aggregate predictions across all companies
    avg_preds_with_sentiment = predictions_with_sentiment.groupby('Date').agg({
        'Actual': 'mean',
        'Predicted': 'mean'
    }).reset_index()

    avg_preds_without_sentiment = predictions_without_sentiment.groupby('Date').agg({
        'Predicted': 'mean'
    }).reset_index()

    plt.figure(figsize=(14, 7))

    # Plot actual values
    plt.plot(avg_preds_with_sentiment['Date'], avg_preds_with_sentiment['Actual'], label='Actual', color='blue', linestyle='--')

    # Plot average predictions with sentiment
    plt.plot(avg_preds_with_sentiment['Date'], avg_preds_with_sentiment['Predicted'], label='Average Predicted with Sentiment', color='red')

    # Plot average predictions without sentiment
    plt.plot(avg_preds_without_sentiment['Date'], avg_preds_without_sentiment['Predicted'], label='Average Predicted without Sentiment', color='green')

    plt.title('Average Predictions vs Actuals Across All Companies - XGBoost model')
    plt.xlabel('Date')
    plt.ylabel('Adj Close')
    plt.legend(loc='upper left')
    
    # Set monthly grid
    ax = plt.gca()
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
    plt.grid(True)
    plt.gcf().autofmt_xdate()  # Auto format date labels

    plt.tight_layout()
    plt.show()

# Plot average predictions
plot_average_predictions(predictions_with_sentiment, predictions_without_sentiment)

def plot_all_companies(predictions_with_sentiment, predictions_without_sentiment, overall_title):
    # Ensure 'Date' is in datetime format
    predictions_with_sentiment['Date'] = pd.to_datetime(predictions_with_sentiment['Date'])
    predictions_without_sentiment['Date'] = pd.to_datetime(predictions_without_sentiment['Date'])
    
    companies = predictions_with_sentiment['Company'].unique()
    
    num_companies = len(companies)
    ncols = 4  # Number of columns for subplots
    nrows = (num_companies + ncols - 1) // ncols  # Calculate rows needed
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), constrained_layout=True)
    
    # Set the super title for the entire figure
    fig.suptitle(overall_title, fontsize=16, fontweight='bold')

    # Flatten axes array if there's only one row or column
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, company in enumerate(companies):
        ax = axes[i // ncols, i % ncols]
        
        # Filter data for the current company
        company_preds_with_sentiment = predictions_with_sentiment[predictions_with_sentiment['Company'] == company]
        company_preds_without_sentiment = predictions_without_sentiment[predictions_without_sentiment['Company'] == company]

        # Plot actual values
        ax.plot(company_preds_with_sentiment['Date'], company_preds_with_sentiment['Actual'], label='Actual', color='blue', linestyle='--')

        # Plot predictions with sentiment
        ax.plot(company_preds_with_sentiment['Date'], company_preds_with_sentiment['Predicted'], label='Predicted with Sentiment', color='red')

        # Plot predictions without sentiment
        ax.plot(company_preds_without_sentiment['Date'], company_preds_without_sentiment['Predicted'], label='Predicted without Sentiment', color='green')

        ax.set_title(f'Company: {company}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adj Close')
        ax.legend(loc='upper left')
        
        # Set monthly grid
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)  # Rotate date labels for better readability
    
    # Turn off unused axes
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])
    
    plt.show()

overall_title = "Comparison of Actual Prices and Predictions Across Companies - XGBoost Model"
plot_all_companies(predictions_with_sentiment, predictions_without_sentiment, overall_title)

# Function to calculate average metrics for the entire dataset
def calculate_average_metrics(results):
    total_val_mse, total_val_rmse, total_val_r2 = 0, 0, 0
    total_test_mse, total_test_rmse, total_test_r2 = 0, 0, 0
    n = len(results)
    
    for result in results:
        total_val_mse += result['Validation']['MSE']
        total_val_rmse += result['Validation']['RMSE']
        total_val_r2 += result['Validation']['R2']
        total_test_mse += result['Test']['MSE']
        total_test_rmse += result['Test']['RMSE']
        total_test_r2 += result['Test']['R2']
    
    avg_val_mse = total_val_mse / n
    avg_val_rmse = total_val_rmse / n
    avg_val_r2 = total_val_r2 / n
    avg_test_mse = total_test_mse / n
    avg_test_rmse = total_test_rmse / n
    avg_test_r2 = total_test_r2 / n
    
    return {
        'avg_val_mse': avg_val_mse,
        'avg_val_rmse': avg_val_rmse,
        'avg_val_r2': avg_val_r2,
        'avg_test_mse': avg_test_mse,
        'avg_test_rmse': avg_test_rmse,
        'avg_test_r2': avg_test_r2
    }

# Calculate average metrics for results with sentiment
avg_metrics_with_sentiment = calculate_average_metrics(results_with_sentiment)

# Calculate average metrics for results without sentiment
avg_metrics_without_sentiment = calculate_average_metrics(results_without_sentiment)

def plot_average_metrics(avg_metrics_with_sentiment, avg_metrics_without_sentiment):
    # Define the metrics and their labels
    metrics = ['mse', 'rmse', 'r2']
    metric_labels = ['MSE', 'RMSE', 'R²']
    
    # Extract values for each metric
    with_sentiment = [avg_metrics_with_sentiment[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_with_sentiment[f'avg_test_{metric}'] for metric in metrics]
    without_sentiment = [avg_metrics_without_sentiment[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_without_sentiment[f'avg_test_{metric}'] for metric in metrics]
    
    # Define positions and width for the bars
    x = np.arange(len(metrics))  # Positions for MSE, RMSE, R²
    width = 0.3  # Reduced width of the bars

    # Create a figure and axis objects
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=False)  # Adjusted figure size

    # Define subplot titles
    for i, metric in enumerate(metrics):
        # Filter the metrics for the current subplot
        val_metric = [avg_metrics_with_sentiment[f'avg_val_{metric}'], avg_metrics_without_sentiment[f'avg_val_{metric}']]
        test_metric = [avg_metrics_with_sentiment[f'avg_test_{metric}'], avg_metrics_without_sentiment[f'avg_test_{metric}']]
        
        # Define the x positions for each bar group
        x_pos = np.arange(2)  # Two bars for with and without sentiment

        # Plot bars for validation and test metrics
        axs[i].bar(x_pos - width/2, val_metric, width, label='Validation', color='b', alpha=0.7)
        axs[i].bar(x_pos + width/2, test_metric, width, label='Test', color='r', alpha=0.7)

        # Set subplot titles and labels
        axs[i].set_xlabel('Sentiment', fontsize=12)
        axs[i].set_title(metric_labels[i], fontsize=14)
        axs[i].set_xticks(x_pos)
        axs[i].set_xticklabels(['With Sentiment', 'Without Sentiment'])
        axs[i].legend()
        
        # Adjust y-axis limits based on the metric
        if metric == 'r2':
            axs[i].set_ylim(0, 1)  # R² ranges from 0 to 1
        elif metric == 'rmse' or metric == 'mse':
            # Apply a logarithmic scale for large values
            axs[i].set_yscale('log')
            max_val = max(with_sentiment[:len(metrics)] + without_sentiment[:len(metrics)])
            axs[i].set_ylim(bottom=1e-2, top=max_val * 2)  # Set upper limit to 200% of max value
            
        # Increase font sizes for better readability
        axs[i].tick_params(axis='both', which='major', labelsize=10)
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set the common y-label and the overall title
    fig.suptitle('Average Metrics Comparison XGBoost Model', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate suptitle
    plt.show()

plot_average_metrics(avg_metrics_with_sentiment, avg_metrics_without_sentiment)

#### SVM 
            
from sklearn.svm import SVR 
# Load the combined dataset
combined_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/dataset/Combined_dataset.csv')

# Check for missing values
missing_values = combined_data.isnull().sum()

# Display basic information about the dataset
print(combined_data.info())
print(combined_data.describe())
print("Missing values in each column:\n", missing_values)

# Drop unnecessary columns "Unnamed" 
combined_data = combined_data.drop(columns=['Unnamed: 0'])

# Drop unused columns Open and Close price of stock  
combined_data = combined_data.drop(columns=['Open', 'Close'])

# Check remaining columns
remaining_columns = combined_data.columns
print("Remaining columns:")
print(remaining_columns)
   
# Define features
features_with_sentiment = [
    'sentiment_score', 'Annual_Report_Sentiment_Score', 'SMA', 'EMA', 'RSI', 
    'Upper_BB', 'Lower_BB', 'Stochastic_Oscillator', 'ATR', 'MACD', 'MACD_Signal', 
    'MACD_Hist', 'OBV', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_14', 'Lag_30', 
    'Day', 'Month', 'Year'
]
features_without_sentiment = [
    'SMA', 'EMA', 'RSI', 'Upper_BB', 'Lower_BB', 'Stochastic_Oscillator', 
    'ATR', 'MACD', 'MACD_Signal', 'MACD_Hist', 'OBV', 'Lag_1', 'Lag_2', 
    'Lag_3', 'Lag_7', 'Lag_14', 'Lag_30', 'Day', 'Month', 'Year'
]

# Set random seeds
np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

# Define a function to train and evaluate SVM models
def train_and_evaluate_svm(data, features):
    results = {}
    
    companies = data['Company'].unique()
    
    for company in companies:
        company_data = data[data['Company'] == company]
        
        train_data = company_data[company_data['Date'] < '2022-01-01']
        val_data = company_data[(company_data['Date'] >= '2022-01-01') & (company_data['Date'] < '2023-01-01')]
        test_data = company_data[company_data['Date'] >= '2023-01-01']
        
        X_train = train_data[features]
        y_train = train_data['Adj Close']
        X_val = val_data[features]
        y_val = val_data['Adj Close']
        X_test = test_data[features]
        y_test = test_data['Adj Close']
        
        # Normalize data
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'C': [0.1, 1, 10, 50],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3],
            'gamma': ['scale', 'auto']
        }
        
        # Initialize SVM model
        svm = SVR()
        
        # KFold cross-validation with a fixed random state
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        
        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=kf, scoring='r2', verbose=2, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_scaled)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predict on validation set with the best model
        y_val_pred_scaled = best_model.predict(X_val_scaled)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
        
        # Predict on test set with the best model
        y_test_pred_scaled = best_model.predict(X_test_scaled)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        
        # Evaluate performance on validation set
        mse_val = mean_squared_error(y_val, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val, y_val_pred)
        
        # Evaluate performance on test set
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_test_pred)
        
        # Save results
        results[company] = {
            'model': best_model,
            'val_metrics': {'mse': mse_val, 'rmse': rmse_val, 'r2': r2_val},
            'test_metrics': {'mse': mse_test, 'rmse': rmse_test, 'r2': r2_test},
            'Predictions': {
                'Date': test_data['Date'].values,
                'Adj Close': test_data['Adj Close'].values,
                'Predicted': y_test_pred
            }
        }
    
    return results

# Function to convert results to DataFrames
def results_to_dataframe(results):
    metrics_list = []
    predictions_list = []
    
    for company, result in results.items():
        val_metrics = result['val_metrics']
        test_metrics = result['test_metrics']
        
        metrics_list.append({
            'Company': company,
            'Validation MSE': val_metrics['mse'],
            'Validation RMSE': val_metrics['rmse'],
            'Validation R²': val_metrics['r2'],
            'Test MSE': test_metrics['mse'],
            'Test RMSE': test_metrics['rmse'],
            'Test R²': test_metrics['r2']
        })
        
        predictions_df = pd.DataFrame({
            'Company': company,
            'Date': result['Predictions']['Date'],
            'Actual Adj Close': result['Predictions']['Adj Close'],
            'Predicted Adj Close': result['Predictions']['Predicted']
        })
        
        predictions_list.append(predictions_df)
    
    metrics_df = pd.DataFrame(metrics_list)
    predictions_df = pd.concat(predictions_list, ignore_index=True)
    
    return metrics_df, predictions_df

# Define and call the function for results with sentiment
results_with_sentiment_svm = train_and_evaluate_svm(combined_data, features_with_sentiment)
metrics_df_with_sentiment, predictions_df_with_sentiment = results_to_dataframe(results_with_sentiment_svm)

# Export to Excel
with pd.ExcelWriter('SVM_Results_with_sentiment.xlsx') as writer:
    metrics_df_with_sentiment.to_excel(writer, sheet_name='Metrics', index=False)
    predictions_df_with_sentiment.to_excel(writer, sheet_name='Predictions', index=False)

print("Results with sentiment have been exported to 'SVM_Results_with_sentiment.xlsx'.")

# Define and call the function for results without sentiment
results_without_sentiment_svm = train_and_evaluate_svm(combined_data, features_without_sentiment)
metrics_df_without_sentiment, predictions_df_without_sentiment = results_to_dataframe(results_without_sentiment_svm)

# Export to Excel
with pd.ExcelWriter('SVM_Results_without_sentiment.xlsx') as writer:
    metrics_df_without_sentiment.to_excel(writer, sheet_name='Metrics', index=False)
    predictions_df_without_sentiment.to_excel(writer, sheet_name='Predictions', index=False)

print("Results without sentiment have been exported to 'SVM_Results_without_sentiment.xlsx'.")

def plot_company_predictions(results_with_sentiment, results_without_sentiment, company):
    plt.figure(figsize=(14, 7))
    
    # Convert dates to datetime
    dates_with_sentiment = pd.to_datetime(results_with_sentiment[company]['Predictions']['Date'])
    dates_without_sentiment = pd.to_datetime(results_without_sentiment[company]['Predictions']['Date'])
    
    # Plot actual prices
    plt.plot(dates_with_sentiment, results_with_sentiment[company]['Predictions']['Adj Close'], label='Actual', color='blue')
    
    # Plot predicted prices with sentiment
    plt.plot(dates_with_sentiment, results_with_sentiment[company]['Predictions']['Predicted'], label='Predicted With Sentiment', color='red')
    
    # Plot predicted prices without sentiment
    plt.plot(dates_without_sentiment, results_without_sentiment[company]['Predictions']['Predicted'], label='Predicted Without Sentiment', color='green')
    
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.title(f'Actual vs Predicted Prices for {company}')
    plt.legend()
    
    # Set the major locator to show ticks at the start of each month
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    
    # Set the date range from 1/1/2023 to end of 2023
    ax.set_xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'))
    
    # Enable grid lines
    plt.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.show()
    
for company in combined_data['Company'].unique():
    plot_company_predictions(results_with_sentiment_svm, results_without_sentiment_svm, company)

def calculate_average_predictions(results):
    all_predictions = pd.DataFrame()
    
    for company, metrics in results.items():
        company_predictions = pd.DataFrame(metrics['Predictions'])
        all_predictions = all_predictions.append(company_predictions, ignore_index=True)
    
    all_predictions.sort_values(by='Date', inplace=True)
    avg_predictions = all_predictions.groupby('Date').mean().reset_index()
    return avg_predictions

avg_with_sentiment_svm = calculate_average_predictions(results_with_sentiment_svm)
avg_without_sentiment_svm = calculate_average_predictions(results_without_sentiment_svm)

def plot_average_predictions(avg_with_sentiment, avg_without_sentiment):
    plt.figure(figsize=(14, 7))
    
    # Convert dates to datetime
    dates_with_sentiment = pd.to_datetime(avg_with_sentiment['Date'])
    dates_without_sentiment = pd.to_datetime(avg_without_sentiment['Date'])
    
    # Plot actual prices
    plt.plot(dates_with_sentiment, avg_with_sentiment['Adj Close'], label='Actual', color='blue')
    
    # Plot predicted prices with sentiment
    plt.plot(dates_with_sentiment, avg_with_sentiment['Predicted'], label='Predicted With Sentiment', color='red')
    
    # Plot predicted prices without sentiment
    plt.plot(dates_without_sentiment, avg_without_sentiment['Predicted'], label='Predicted Without Sentiment', color='green')
    
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.title('Average Actual vs Predicted Prices - SVM Model')
    plt.legend()
    
    # Set the major locator to show ticks at the start of each month
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    
    # Set the date range from 1/1/2023 to end of 2023
    ax.set_xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'))
    
    # Enable grid lines
    plt.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.show()

# Assuming you have the average predictions as avg_with_sentiment_svm and avg_without_sentiment_svm
plot_average_predictions(avg_with_sentiment_svm, avg_without_sentiment_svm)

def plot_all_companies(predictions_df_with_sentiment, predictions_df_without_sentiment, overall_title):
    # Ensure 'Date' is in datetime format
    predictions_df_with_sentiment['Date'] = pd.to_datetime(predictions_df_with_sentiment['Date'])
    predictions_df_without_sentiment['Date'] = pd.to_datetime(predictions_df_without_sentiment['Date'])
    
    companies = predictions_df_with_sentiment['Company'].unique()
    
    num_companies = len(companies)
    ncols = 4  # Number of columns for subplots
    nrows = (num_companies + ncols - 1) // ncols  # Calculate rows needed
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), constrained_layout=True)
    
    # Set the super title for the entire figure
    fig.suptitle(overall_title, fontsize=16, fontweight='bold')

    # Flatten axes array if there's only one row or column
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, company in enumerate(companies):
        ax = axes[i // ncols, i % ncols]
        
        # Filter data for the current company
        company_preds_with_sentiment = predictions_df_with_sentiment[predictions_df_with_sentiment['Company'] == company]
        company_preds_without_sentiment = predictions_df_without_sentiment[predictions_df_without_sentiment['Company'] == company]

        # Plot actual values
        ax.plot(company_preds_with_sentiment['Date'], company_preds_with_sentiment['Actual Adj Close'], label='Actual', color='blue', linestyle='--')

        # Plot predictions with sentiment
        ax.plot(company_preds_with_sentiment['Date'], company_preds_with_sentiment['Predicted Adj Close'], label='Predicted with Sentiment', color='red')

        # Plot predictions without sentiment
        ax.plot(company_preds_without_sentiment['Date'], company_preds_without_sentiment['Predicted Adj Close'], label='Predicted without Sentiment', color='green')

        ax.set_title(f'Company: {company}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adj Close')
        ax.legend(loc='upper left')
        
        # Set monthly grid
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)  # Rotate date labels for better readability
    
    # Turn off unused axes
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])
    
    plt.show()

overall_title = "Comparison of Actual Prices and Predictions Across Companies - SVM Model"
plot_all_companies(predictions_df_with_sentiment, predictions_df_without_sentiment, overall_title)

# Function to calculate average metrics for the entire dataset
def calculate_average_metrics(results):
    total_val_mse, total_val_rmse, total_val_r2 = 0, 0, 0
    total_test_mse, total_test_rmse, total_test_r2 = 0, 0, 0
    n = len(results)
    
    for result in results.values():
        total_val_mse += result['val_metrics']['mse']
        total_val_rmse += result['val_metrics']['rmse']
        total_val_r2 += result['val_metrics']['r2']
        total_test_mse += result['test_metrics']['mse']
        total_test_rmse += result['test_metrics']['rmse']
        total_test_r2 += result['test_metrics']['r2']
    
    avg_val_mse = total_val_mse / n
    avg_val_rmse = total_val_rmse / n
    avg_val_r2 = total_val_r2 / n
    avg_test_mse = total_test_mse / n
    avg_test_rmse = total_test_rmse / n
    avg_test_r2 = total_test_r2 / n
    
    return {
        'avg_val_mse': avg_val_mse,
        'avg_val_rmse': avg_val_rmse,
        'avg_val_r2': avg_val_r2,
        'avg_test_mse': avg_test_mse,
        'avg_test_rmse': avg_test_rmse,
        'avg_test_r2': avg_test_r2
    }
# Calculate average metrics
avg_metrics_with_sentiment_svm = calculate_average_metrics(results_with_sentiment_svm)
avg_metrics_without_sentiment_svm = calculate_average_metrics(results_without_sentiment_svm)

# Print out the average metrics for inspection
print("Average Metrics with Sentiment:", avg_metrics_with_sentiment)
print("Average Metrics without Sentiment:", avg_metrics_without_sentiment)

def plot_average_metrics(avg_metrics_with_sentiment_svm, avg_metrics_without_sentiment_svm):
    # Define the metrics and their labels
    metrics = ['mse', 'rmse', 'r2']
    metric_labels = ['MSE', 'RMSE', 'R²']
    
    # Extract values for each metric
    with_sentiment = [avg_metrics_with_sentiment_svm[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_with_sentiment_svm[f'avg_test_{metric}'] for metric in metrics]
    without_sentiment = [avg_metrics_without_sentiment_svm[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_without_sentiment_svm[f'avg_test_{metric}'] for metric in metrics]
    
    # Define positions and width for the bars
    x = np.arange(len(metrics))  # Positions for MSE, RMSE, R²
    width = 0.3  # Reduced width of the bars

    # Create a figure and axis objects
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=False)  # Adjusted figure size

    # Define subplot titles
    for i, metric in enumerate(metrics):
        # Filter the metrics for the current subplot
        val_metric = [avg_metrics_with_sentiment_svm[f'avg_val_{metric}'], avg_metrics_without_sentiment_svm[f'avg_val_{metric}']]
        test_metric = [avg_metrics_with_sentiment_svm[f'avg_test_{metric}'], avg_metrics_without_sentiment_svm[f'avg_test_{metric}']]
        
        # Define the x positions for each bar group
        x_pos = np.arange(2)  # Two bars for with and without sentiment

        # Plot bars for validation and test metrics
        axs[i].bar(x_pos - width/2, val_metric, width, label='Validation', color='b', alpha=0.7)
        axs[i].bar(x_pos + width/2, test_metric, width, label='Test', color='r', alpha=0.7)

        # Set subplot titles and labels
        axs[i].set_xlabel('Sentiment', fontsize=12)
        axs[i].set_title(metric_labels[i], fontsize=14)
        axs[i].set_xticks(x_pos)
        axs[i].set_xticklabels(['With Sentiment', 'Without Sentiment'])
        axs[i].legend()
        
        # Adjust y-axis limits based on the metric
        if metric == 'r2':
            axs[i].set_ylim(0, 1)  # R² ranges from 0 to 1
        elif metric == 'rmse' or metric == 'mse':
            # Apply a logarithmic scale for large values
            axs[i].set_yscale('log')
            max_val = max(with_sentiment[:len(metrics)] + without_sentiment[:len(metrics)])
            axs[i].set_ylim(bottom=1e-2, top=max_val * 2)  # Set upper limit to 200% of max value
            
        # Increase font sizes for better readability
        axs[i].tick_params(axis='both', which='major', labelsize=10)
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set the common y-label and the overall title
    fig.suptitle('Average Metrics Comparison SVM Model', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate suptitle
    plt.show()

plot_average_metrics(avg_metrics_with_sentiment_svm, avg_metrics_without_sentiment_svm)

svm_with_sentiment = pd.read_excel('/Users/mac/Documents/Manchester/Dissertation/comparison model/SVM_Results_with_sentiment.xlsx')
svm_without_sentiment = pd.read_excel('/Users/mac/Documents/Manchester/Dissertation/comparison model/SVM_Results_without_sentiment.xlsx')

def calculate_average_metrics(data):
    avg_metrics = {
        'avg_val_mse': data['Validation MSE'].mean(),
        'avg_val_rmse': data['Validation RMSE'].mean(),
        'avg_val_r2': data['Validation R²'].mean(),
        'avg_test_mse': data['Test MSE'].mean(),
        'avg_test_rmse': data['Test RMSE'].mean(),
        'avg_test_r2': data['Test R²'].mean()
    }
    return avg_metrics

# Calculate average metrics
avg_metrics_with_sentiment = calculate_average_metrics(svm_with_sentiment)
avg_metrics_without_sentiment = calculate_average_metrics(svm_without_sentiment)

# Function to plot the average metrics
def plot_average_metrics(avg_metrics_with_sentiment, avg_metrics_without_sentiment):
    # Define the metrics and their labels
    metrics = ['mse', 'rmse', 'r2']
    metric_labels = ['MSE', 'RMSE', 'R²']
    
    # Extract values for each metric
    with_sentiment = [avg_metrics_with_sentiment[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_with_sentiment[f'avg_test_{metric}'] for metric in metrics]
    without_sentiment = [avg_metrics_without_sentiment[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_without_sentiment[f'avg_test_{metric}'] for metric in metrics]
    
    # Define positions and width for the bars
    x = np.arange(len(metrics))  # Positions for MSE, RMSE, R²
    width = 0.3  # Reduced width of the bars

    # Create a figure and axis objects
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=False)  # Adjusted figure size

    # Define subplot titles
    for i, metric in enumerate(metrics):
        # Filter the metrics for the current subplot
        val_metric = [avg_metrics_with_sentiment[f'avg_val_{metric}'], avg_metrics_without_sentiment[f'avg_val_{metric}']]
        test_metric = [avg_metrics_with_sentiment[f'avg_test_{metric}'], avg_metrics_without_sentiment[f'avg_test_{metric}']]
        
        # Define the x positions for each bar group
        x_pos = np.arange(2)  # Two bars for with and without sentiment

        # Plot bars for validation and test metrics
        bars_val = axs[i].bar(x_pos - width/2, val_metric, width, label='Validation', color='b', alpha=0.7)
        bars_test = axs[i].bar(x_pos + width/2, test_metric, width, label='Test', color='r', alpha=0.7)

        # Set subplot titles and labels
        axs[i].set_xlabel('Sentiment', fontsize=12)
        axs[i].set_title(metric_labels[i], fontsize=14)
        axs[i].set_xticks(x_pos)
        axs[i].set_xticklabels(['With Sentiment', 'Without Sentiment'])
        axs[i].legend()
        
        # Adjust y-axis limits based on the metric
        if metric == 'r2':
            axs[i].set_ylim(0, 1)  # R² ranges from 0 to 1
        elif metric == 'rmse' or metric == 'mse':
            # Apply a logarithmic scale for large values
            axs[i].set_yscale('log')
            max_val = max(with_sentiment[:len(metrics)] + without_sentiment[:len(metrics)])
            axs[i].set_ylim(bottom=1e-2, top=max_val * 2)  # Set upper limit to 200% of max value
            
        # Increase font sizes for better readability
        axs[i].tick_params(axis='both', which='major', labelsize=10)
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add value labels on top of bars
        for bar in bars_val + bars_test:
            yval = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom' if yval < 0 else 'top', ha='center', fontsize=10)

    # Set the common y-label and the overall title
    fig.suptitle('Average Metrics Comparison SVM Model', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate suptitle
    plt.show()

# Plot the average metrics for SVM
plot_average_metrics(avg_metrics_with_sentiment, avg_metrics_without_sentiment)

def plot_average_metrics(avg_metrics_with_sentiment, avg_metrics_without_sentiment):
    # Define the metrics and their labels
    metrics = ['mse', 'rmse', 'r2']
    metric_labels = ['MSE', 'RMSE', 'R²']
    
    # Extract values for each metric
    with_sentiment = [avg_metrics_with_sentiment[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_with_sentiment[f'avg_test_{metric}'] for metric in metrics]
    without_sentiment = [avg_metrics_without_sentiment[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_without_sentiment[f'avg_test_{metric}'] for metric in metrics]
    
    # Define positions and width for the bars
    x = np.arange(len(metrics))  # Positions for MSE, RMSE, R²
    width = 0.3  # Reduced width of the bars

    # Create a figure and axis objects
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=False)  # Adjusted figure size

    # Define subplot titles
    for i, metric in enumerate(metrics):
        # Filter the metrics for the current subplot
        val_metric = [avg_metrics_with_sentiment[f'avg_val_{metric}'], avg_metrics_without_sentiment[f'avg_val_{metric}']]
        test_metric = [avg_metrics_with_sentiment[f'avg_test_{metric}'], avg_metrics_without_sentiment[f'avg_test_{metric}']]
        
        # Define the x positions for each bar group
        x_pos = np.arange(2)  # Two bars for with and without sentiment

        # Plot bars for validation and test metrics
        bars_val = axs[i].bar(x_pos - width/2, val_metric, width, label='Validation', color='b', alpha=0.7)
        bars_test = axs[i].bar(x_pos + width/2, test_metric, width, label='Test', color='r', alpha=0.7)

        # Set subplot titles and labels
        axs[i].set_xlabel('Sentiment', fontsize=12)
        axs[i].set_title(metric_labels[i], fontsize=14)
        axs[i].set_xticks(x_pos)
        axs[i].set_xticklabels(['With Sentiment', 'Without Sentiment'])
        
        # Move the legend to the bottom
        axs[i].legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=1)
        
        # Adjust y-axis limits based on the metric
        if metric == 'r2':
            axs[i].set_ylim(0, 1)  # R² ranges from 0 to 1
        elif metric == 'rmse' or metric == 'mse':
            # Apply a logarithmic scale for large values
            axs[i].set_yscale('log')
            max_val = max(with_sentiment[:len(metrics)] + without_sentiment[:len(metrics)])
            axs[i].set_ylim(bottom=1e-2, top=max_val * 2)  # Set upper limit to 200% of max value
            
        # Increase font sizes for better readability
        axs[i].tick_params(axis='both', which='major', labelsize=10)
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add value labels on top of bars
        for bar in bars_val + bars_test:
            yval = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom' if yval < 0 else 'top', ha='center', fontsize=10)

    # Set the common y-label and the overall title
    fig.suptitle('Average Metrics Comparison SVM Model', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate suptitle
    plt.show()

# Plot the average metrics for SVM
plot_average_metrics(avg_metrics_with_sentiment, avg_metrics_without_sentiment)

### Plot comparision R2 test set 

# Extract Test R² for each company
test_r2_with_sentiment = svm_with_sentiment[['Company', 'Test R²']]
test_r2_without_sentiment = svm_without_sentiment[['Company', 'Test R²']]

# Rename columns for clarity
test_r2_with_sentiment.rename(columns={'Test R²': 'Test_R2_with'}, inplace=True)
test_r2_without_sentiment.rename(columns={'Test R²': 'Test_R2_without'}, inplace=True)

# Merge the two DataFrames on the Company column
test_r2_comparison = pd.merge(test_r2_with_sentiment, test_r2_without_sentiment, on='Company')

# Sort by Test R² with sentiment
test_r2_comparison.sort_values(by='Test_R2_with', ascending=True, inplace=True)

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define positions for the bars
y = np.arange(len(test_r2_comparison))
width = 0.35  # Width of the bars

# Plot bars
bars_with = ax.barh(y - width/2, test_r2_comparison['Test_R2_with'], width, label='With Sentiment', alpha=0.7)
bars_without = ax.barh(y + width/2, test_r2_comparison['Test_R2_without'], width, label='Without Sentiment', alpha=0.7)

# Add value labels on top of bars
for bar in bars_with:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)
for bar in bars_without:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)

# Set labels and title
ax.set_ylabel('Company', fontsize=12)
ax.set_xlabel('Test R²', fontsize=12)
ax.set_title('Comparison of Test R² with and without sentiment score for Each Company (SVM Model)', fontsize=14)
ax.set_yticks(y)
ax.set_yticklabels(test_r2_comparison['Company'])
ax.legend(loc='lower center')

# Increase font sizes for better readability
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.tight_layout()
plt.show()


### LSTM RNN 

import numpy as np
import pandas as pd
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras.regularizers import l2
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl

# Set random seed for reproducibility
def set_random_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_random_seed(1)

def train_and_evaluate_lstm(data, features):
    results = {}
    
    companies = data['Company'].unique()
    
    for company in companies:
        company_data = data[data['Company'] == company]
        
        train_data = company_data[company_data['Date'] < '2022-01-01']
        val_data = company_data[(company_data['Date'] >= '2022-01-01') & (company_data['Date'] < '2023-01-01')]
        test_data = company_data[company_data['Date'] >= '2023-01-01']
        
        X_train = train_data[features]
        y_train = train_data['Adj Close']
        X_val = val_data[features]
        y_val = val_data['Adj Close']
        X_test = test_data[features]
        y_test = test_data['Adj Close']
        
        # Normalize data
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Reshape data for LSTM [samples, time steps, features]
        X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_val_lstm = np.reshape(X_val_scaled, (X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
        X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(Dropout(0.05))
        model.add(LSTM(units=50, kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
        model.add(Dropout(0.05))
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train the model
        history = model.fit(X_train_lstm, y_train_scaled,
                            epochs=100, batch_size=64,
                            validation_data=(X_val_lstm, y_val_scaled),
                            callbacks=[early_stopping],
                            verbose=1)
        
        # Predict on validation set
        y_val_pred_scaled = model.predict(X_val_lstm)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled).flatten()
        
        # Predict on test set
        y_test_pred_scaled = model.predict(X_test_lstm)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
        
        # Evaluate performance on validation set
        mse_val = mean_squared_error(y_val, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val, y_val_pred)
        
        # Evaluate performance on test set
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_test_pred)
        
        # Save results
        results[company] = {
            'model': model,
            'val_metrics': {'mse': mse_val, 'rmse': rmse_val, 'r2': r2_val},
            'test_metrics': {'mse': mse_test, 'rmse': rmse_test, 'r2': r2_test},
            'Predictions': {
                'Date': test_data['Date'].values,
                'Adj Close': test_data['Adj Close'].values,
                'Predicted': y_test_pred
            }
        }
    
    return results

lstm_results_with_sentiment = train_and_evaluate_lstm(combined_data, features_with_sentiment)
lstm_results_without_sentiment = train_and_evaluate_lstm(combined_data, features_without_sentiment)

def export_results_to_excel(results, filename):
    metrics_list = []
    all_predictions = []

    for company, result in results.items():
        val_metrics = result['val_metrics']
        test_metrics = result['test_metrics']
        predictions = result['Predictions']

        metrics_list.append([
            company,
            val_metrics['mse'], val_metrics['rmse'], val_metrics['r2'],
            test_metrics['mse'], test_metrics['rmse'], test_metrics['r2']
        ])

        pred_df = pd.DataFrame(predictions)
        pred_df['Company'] = company
        all_predictions.append(pred_df)

    metrics_df = pd.DataFrame(metrics_list, columns=[
        'Company', 'Validation MSE', 'Validation RMSE', 'Validation R²',
        'Test MSE', 'Test RMSE', 'Test R²'
    ])
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)

    with pd.ExcelWriter(filename) as writer:
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        all_predictions_df.to_excel(writer, sheet_name='Predictions', index=False)

# Export LSTM results
export_results_to_excel(lstm_results_with_sentiment, 'lstm_results_with_sentiment.xlsx')
export_results_to_excel(lstm_results_without_sentiment, 'lstm_results_without_sentiment.xlsx')

def calculate_average_metrics(results):
    total_val_mse, total_val_rmse, total_val_r2 = 0, 0, 0
    total_test_mse, total_test_rmse, total_test_r2 = 0, 0, 0
    n = len(results)
    
    for result in results.values():
        total_val_mse += result['val_metrics']['mse']
        total_val_rmse += result['val_metrics']['rmse']
        total_val_r2 += result['val_metrics']['r2']
        total_test_mse += result['test_metrics']['mse']
        total_test_rmse += result['test_metrics']['rmse']
        total_test_r2 += result['test_metrics']['r2']
    
    avg_val_mse = total_val_mse / n
    avg_val_rmse = total_val_rmse / n
    avg_val_r2 = total_val_r2 / n
    avg_test_mse = total_test_mse / n
    avg_test_rmse = total_test_rmse / n
    avg_test_r2 = total_test_r2 / n
    
    return {
        'avg_val_mse': avg_val_mse,
        'avg_val_rmse': avg_val_rmse,
        'avg_val_r2': avg_val_r2,
        'avg_test_mse': avg_test_mse,
        'avg_test_rmse': avg_test_rmse,
        'avg_test_r2': avg_test_r2
    }

def calculate_average_predictions(results):
    all_dates = []
    all_actual = []
    all_predicted = []
    
    for result in results.values():
        predictions = result['Predictions']
        all_dates.extend(predictions['Date'])
        all_actual.extend(predictions['Adj Close'])
        all_predicted.extend(predictions['Predicted'])
    
    avg_df = pd.DataFrame({
        'Date': all_dates,
        'Actual': all_actual,
        'Predicted': all_predicted
    })
    avg_df = avg_df.groupby('Date').mean().reset_index()
    
    return avg_df

avg_metrics_with_sentiment = calculate_average_metrics(lstm_results_with_sentiment)
avg_metrics_without_sentiment = calculate_average_metrics(lstm_results_without_sentiment)

avg_predictions_with_sentiment = calculate_average_predictions(lstm_results_with_sentiment)
avg_predictions_without_sentiment = calculate_average_predictions(lstm_results_without_sentiment)

def plot_average_metrics(avg_metrics_with_sentiment, avg_metrics_without_sentiment):
    # Define the metrics and their labels
    metrics = ['mse', 'rmse', 'r2']
    metric_labels = ['MSE', 'RMSE', 'R²']
    
    # Extract values for each metric
    with_sentiment = [avg_metrics_with_sentiment[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_with_sentiment[f'avg_test_{metric}'] for metric in metrics]
    without_sentiment = [avg_metrics_without_sentiment[f'avg_val_{metric}'] for metric in metrics] + [avg_metrics_without_sentiment[f'avg_test_{metric}'] for metric in metrics]
    
    # Define positions and width for the bars
    x = np.arange(len(metrics))  # Positions for MSE, RMSE, R²
    width = 0.35  # Width of the bars

    # Create a figure and axis objects
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Define subplot titles
    for i, metric in enumerate(metrics):
        # Filter the metrics for the current subplot
        val_metric = [avg_metrics_with_sentiment[f'avg_val_{metric}'], avg_metrics_without_sentiment[f'avg_val_{metric}']]
        test_metric = [avg_metrics_with_sentiment[f'avg_test_{metric}'], avg_metrics_without_sentiment[f'avg_test_{metric}']]
        
        # Define the x positions for each bar group
        x_pos = np.arange(2)  # Two bars for with and without sentiment

        # Plot bars for validation and test metrics
        bars_val = axs[i].bar(x_pos - width/2, val_metric, width, label='Validation', color='b', alpha=0.7)
        bars_test = axs[i].bar(x_pos + width/2, test_metric, width, label='Test', color='r', alpha=0.7)

        # Set subplot titles and labels
        axs[i].set_xlabel('Sentiment')
        axs[i].set_title(metric_labels[i])
        axs[i].set_xticks(x_pos)
        axs[i].set_xticklabels(['With Sentiment', 'Without Sentiment'])
        axs[i].legend()

        # Annotate bars with values
        for bars in [bars_val, bars_test]:
            for bar in bars:
                yval = bar.get_height()
                axs[i].text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
        
        # Set y-axis limits for RMSE and R² to zoom in
        if metric in ['rmse', 'r2']:
            axs[i].set_ylim(0, max(max(val_metric), max(test_metric)) * 1.1)
        
    # Set the common y-label and the overall title
    fig.suptitle('Average Metrics Comparison - LSTM-RNN Model')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate suptitle
    plt.show()
    
plot_average_metrics(avg_metrics_with_sentiment, avg_metrics_without_sentiment)

def plot_predictions(results_with_sentiment, results_without_sentiment):
    for company in results_with_sentiment.keys():
        plt.figure(figsize=(12, 6))
        
        # Extracting data
        dates = results_with_sentiment[company]['Predictions']['Date']
        actual = results_with_sentiment[company]['Predictions']['Adj Close']
        predicted_with_sentiment = results_with_sentiment[company]['Predictions']['Predicted']
        predicted_without_sentiment = results_without_sentiment[company]['Predictions']['Predicted']
        
        # Ensure dates are in datetime format
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates)
        
        plt.plot(dates, actual, label='Actual')
        plt.plot(dates, predicted_with_sentiment, label='Predicted with Sentiment')
        plt.plot(dates, predicted_without_sentiment, label='Predicted without Sentiment')
        
        plt.title(f'Predictions for {company}')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        
        # Formatting the x-axis
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.xticks(rotation=45)  # Rotate dates for better readability
        
        # Adding grid
        plt.grid(True)
        
        plt.legend()
        plt.tight_layout()  # Adjust layout to fit labels
        plt.show()
        
plot_predictions(lstm_results_with_sentiment, lstm_results_without_sentiment)

def plot_average_predictions(avg_predictions_with_sentiment, avg_predictions_without_sentiment):
    plt.figure(figsize=(12, 6))
    
    # Extracting data
    dates = avg_predictions_with_sentiment['Date']
    actual = avg_predictions_with_sentiment['Actual']
    predicted_with_sentiment = avg_predictions_with_sentiment['Predicted']
    predicted_without_sentiment = avg_predictions_without_sentiment['Predicted']
    
    # Ensure dates are in datetime format
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)
    
    # Plotting data
    plt.plot(dates, actual, label='Average Actual', color='blue')
    plt.plot(dates, predicted_with_sentiment, label='Average Predicted with Sentiment', color='red')
    plt.plot(dates, predicted_without_sentiment, label='Average Predicted without Sentiment', color='green')
    
    # Title and labels
    plt.title('Average Predictions vs Actuals Across All Companies - LSTM-RNN Model')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    
    # Formatting the x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.xticks(rotation=45)  # Rotate dates for better readability
    
    # Adding grid
    plt.grid(True)
    
    plt.legend()
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()

plot_average_predictions(avg_predictions_with_sentiment, avg_predictions_without_sentiment)

def list_to_dict(results_list):
    results_dict = {}
    for result in results_list:
        # Assume 'Company' key exists in each dictionary in the list
        company_name = result['Company']
        results_dict[company_name] = result
    return results_dict

# Convert the lists to dictionaries
results_with_sentiment_dict = list_to_dict(results_with_sentiment)
results_without_sentiment_dict = list_to_dict(results_without_sentiment)

# Now you can call the plotting function with dictionaries
plot_predictions(results_with_sentiment_dict, results_without_sentiment_dict, overall_title)

def plot_predictions(results_with_sentiment, results_without_sentiment, overall_title):
    if not isinstance(results_with_sentiment, dict) or not isinstance(results_without_sentiment, dict):
        raise TypeError("Both results_with_sentiment and results_without_sentiment should be dictionaries.")

    num_companies = len(results_with_sentiment)
    num_cols = 5
    num_rows = (num_companies // num_cols) + (num_companies % num_cols > 0)
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15), constrained_layout=True)
    axs = axs.flatten()
    
    for i, company in enumerate(results_with_sentiment.keys()):
        ax = axs[i]
        
        # Extracting data
        result_with_sentiment = results_with_sentiment.get(company, {})
        result_without_sentiment = results_without_sentiment.get(company, {})
        
        # Ensure 'Predictions' key exists and contains expected keys
        if 'Predictions' not in result_with_sentiment:
            print(f"Missing 'Predictions' in result_with_sentiment for {company}. Skipping.")
            continue
        
        pred_data_with_sentiment = result_with_sentiment['Predictions']
        dates = pred_data_with_sentiment.get('Date', [])
        actual = pred_data_with_sentiment.get('Actual', [])
        predicted_with_sentiment = pred_data_with_sentiment.get('Predicted', [])
        
        pred_data_without_sentiment = result_without_sentiment.get('Predictions', {})
        predicted_without_sentiment = pred_data_without_sentiment.get('Predicted', [])
        
        # Check the lengths of arrays
        print(f"Company: {company}")
        print(f"Length of dates: {len(dates)}")
        print(f"Length of actual: {len(actual)}")
        print(f"Length of predicted_with_sentiment: {len(predicted_with_sentiment)}")
        print(f"Length of predicted_without_sentiment: {len(predicted_without_sentiment)}")
        
        # Convert dates to datetime if necessary
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates)
        
        # Check if lengths match before plotting
        if len(dates) == len(actual) == len(predicted_with_sentiment):
            ax.plot(dates, actual, label='Actual')
            ax.plot(dates, predicted_with_sentiment, label='Predicted with Sentiment')
            if predicted_without_sentiment:
                ax.plot(dates, predicted_without_sentiment, label='Predicted without Sentiment')
        else:
            print(f"Data length mismatch for {company}. Skipping plot.")
            continue
        
        ax.set_title(company)
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted Close Price')
        
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
    
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    fig.suptitle(overall_title, fontsize=16)
    
    plt.show()

# Example usage
overall_title = "Comparison of Actual Prices and Predictions Across Companies - LSTM-RNN Model"
plot_predictions(results_with_sentiment_dict, results_without_sentiment_dict, overall_title)

### Plot 
# Read the Excel files for LSTM model
lstm_with_sentiment = pd.read_excel('/Users/mac/Documents/Manchester/Dissertation/comparison model/lstm_results_with_sentiment.xlsx')
lstm_without_sentiment = pd.read_excel('/Users/mac/Documents/Manchester/Dissertation/comparison model/lstm_results_without_sentiment.xlsx')

# Extract Test R² for each company
test_r2_with_sentiment = lstm_with_sentiment[['Company', 'Test R²']]
test_r2_without_sentiment = lstm_without_sentiment[['Company', 'Test R²']]

# Rename columns for clarity
test_r2_with_sentiment.rename(columns={'Test R²': 'Test_R2_with'}, inplace=True)
test_r2_without_sentiment.rename(columns={'Test R²': 'Test_R2_without'}, inplace=True)

# Convert Test R² columns to numeric, forcing errors to NaN
test_r2_with_sentiment['Test_R2_with'] = pd.to_numeric(test_r2_with_sentiment['Test_R2_with'], errors='coerce')
test_r2_without_sentiment['Test_R2_without'] = pd.to_numeric(test_r2_without_sentiment['Test_R2_without'], errors='coerce')

# Merge the two DataFrames on the Company column
test_r2_comparison = pd.merge(test_r2_with_sentiment, test_r2_without_sentiment, on='Company')

# Check for and handle non-numeric values in the DataFrame
print(test_r2_comparison.info())  # Verify column types
print(test_r2_comparison.head())  # Inspect the first few rows

# Convert columns to numeric again to ensure proper sorting
test_r2_comparison['Test_R2_with'] = pd.to_numeric(test_r2_comparison['Test_R2_with'], errors='coerce')
test_r2_comparison['Test_R2_without'] = pd.to_numeric(test_r2_comparison['Test_R2_without'], errors='coerce')

# Drop rows where Test_R2_with is NaN
test_r2_comparison.dropna(subset=['Test_R2_with'], inplace=True)

# Sort by Test R² with sentiment in descending order
test_r2_comparison.sort_values(by='Test_R2_with', ascending=True, inplace=True)

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define positions for the bars
y = np.arange(len(test_r2_comparison))
width = 0.35  # Width of the bars

# Plot bars
bars_with = ax.barh(y - width/2, test_r2_comparison['Test_R2_with'], width, label='With Sentiment', alpha=0.7)
bars_without = ax.barh(y + width/2, test_r2_comparison['Test_R2_without'], width, label='Without Sentiment', alpha=0.7)

# Add value labels on the bars with three decimal places
for bar in bars_with:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)
for bar in bars_without:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)

# Set labels and title
ax.set_ylabel('Company', fontsize=12)
ax.set_xlabel('Test R²', fontsize=12)
ax.set_title('Comparison of Test R² for Each Company (LSTM Model)', fontsize=14)
ax.set_yticks(y)
ax.set_yticklabels(test_r2_comparison['Company'])
ax.legend(loc='lower center')

# Increase font sizes for better readability
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.tight_layout()
plt.show()

