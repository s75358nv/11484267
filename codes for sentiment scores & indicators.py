#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 06:51:48 2024

@author: mac
"""

import requests
from newspaper import Article
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
from dateutil import parser
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import ta
import fitz  # PyMuPDF

### ABRDN

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['ABRDN']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('ABRDN_news_sentiments.csv', index=False)
    print("Results exported to ABRDN_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /ABRDN'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/ABRDN_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/ABDN.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.9732447862625122, 0.00026686713681556284, 3.51582930306904e-05,
                               0.00018046327750198543, 0.0016918806359171867, 0.001888458849862218,
                               1.9013550911495258e-07, 0.00026976066874340177, 3.5398225008975714e-05,
                               0.003808643901720643]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('ABRDN_merged_scores.csv', index=False)
print("ABRDN merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/ABRDN_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
    
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/ABRDN_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('ABRDN_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

###HSBC

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = 'HSBC'
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('HSBC_news_sentiments.csv', index=False)
    print("Results exported to HSBC_news_sentiments.csv")
    
### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /HSBC'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/hsbc_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/HSBA.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [6.992425369389821e-06, 8.412522583967075e-05, 0.00019627070287242532,
                               0.0004966686246916652, 0.24200308322906494, 0.9112055897712708,
                               0.00011462833208497614, 2.522461181797553e-05, 0.00012210884597152472,
                               5.339875497156754e-05]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('HSBC_merged_scores.csv', index=False)
print("HSBC merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/HSBC_merged_scores.csv')
print(final_data.columns)
# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 

############# CALCUALTION INDICATORS

import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/HSBC_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('HSBC_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))


##Barclays

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = 'Barclays'
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Barclays_news_sentiments.csv', index=False)
    print("Results exported to Barclays_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Barclays'  

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Barclays_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/BARC.L (1).csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('barclays_stock_prices_with_sentiments.csv', index=False)

print("Sentiment scores merged with historical stock prices and saved to 'barclays_stock_prices_with_sentiments.csv'")


annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.00014728325186297297, 0.00010887560347327963, 5.91094431001693e-05,
                               0.0001084283139789477, 6.752726039849222e-05, 0.9990355968475342,
                               0.9999977350234985, 0.03855664283037186, 0.12397757172584534,
                               0.9975101947784424]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Barclays_merged_scores.csv', index=False)
print("Barclays merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Barclays_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
  
#### Fill missing value

df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Barclays_merged_scores.csv')

# Print column names to verify the 'Date' column
print("Column names in the dataset:", df.columns)

# If the 'Date' column exists, proceed with conversion and imputation
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Define the columns to be imputed
    columns_to_impute = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Impute missing values
    for column in columns_to_impute:
        if column == 'Volume':
            df[column] = df[column].fillna(0)  # Fill missing volume with 0
        else:
            df[column] = df[column].ffill()  # Forward fill to use the last available previous day's value

    # Check for any remaining missing values
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("Imputation successful. No missing values remaining.")
    else:
        print(f"Warning: {missing_values} missing values still exist after imputation.")
print(missing_values)
df.to_csv('revised_Barclays_merged_scores.csv', index=True)
print("revised_Barclays merged scores")
print("Column names in the dataset:", df.columns)

############# CALCUALTION INDICATORS

import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Barclays_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Barclays_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

####LLoyds
API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = 'Lloyds'
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Lloyds_news_sentiments.csv', index=False)
    print("Results exported to Lloyds_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Lloyds'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Lloyds_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/LLOY.L (1).csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.23426295816898346, 5.6509801652282476e-05, 0.9998787641525269,
                               0.5285735130310059, 0.0047827064990997314, 8.739393524592742e-05,
                               0.00016071328718680888, 0.9999974966049194, 0.9999839067459106,
                               0.9940150380134583]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Lloyds_merged_scores.csv', index=False)
print("Lloyds merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Lloyds_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
  
#### Fill missing value

df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Lloyds_merged_scores.csv')

# Print column names to verify the 'Date' column
print("Column names in the dataset:", df.columns)

# If the 'Date' column exists, proceed with conversion and imputation
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Define the columns to be imputed
    columns_to_impute = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Impute missing values
    for column in columns_to_impute:
        if column == 'Volume':
            df[column] = df[column].fillna(0)  # Fill missing volume with 0
        else:
            df[column] = df[column].ffill()  # Forward fill to use the last available previous day's value

    # Check for any remaining missing values
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("Imputation successful. No missing values remaining.")
    else:
        print(f"Warning: {missing_values} missing values still exist after imputation.")
print(df)
print(missing_values)
df.to_csv('revised_Lloyds_merged_scores.csv', index=True)
print("revised_Lloyds merged scores")
print("Column names in the dataset:", df.columns)

############# CALCUALTION INDICATORS

import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Lloyds_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Lloydss_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

#### Natwest

API_KEY = '45f1b3a41edcbade3af84c195ca394981e83f4ad15deb55b557bd51220736379'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = 'Natwest'
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Natwest_news_sentiments.csv', index=False)
    print("Results exported to Natwest_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Natwest'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Natwest_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/NWG.L (1).csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [2.556854633439798e-05, 0.9908903241157532, 0.00015013240044936538,
                               0.009084170684218407, 0.959880530834198, 5.551562935579568e-05,
                               0.0003552218549884856, 2.9609060220536776e-05, 0.23875866830348969,
                               1.701107066764962e-05]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Natwest_merged_scores.csv', index=False)
print("Natwest merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Natwest_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 

############# CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Natwest_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Natwest_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### LSEG
API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = 'London_Stock_Exchange_Group_LSEG'
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('LSEG_news_sentiments.csv', index=False)
    print("Results exported to LSEG_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /LSEG'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/LSEG_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/LSEG.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [4.4496946429717354e-06, 1.907655814648024e-06, 2.2503829768538708e-06,
                               8.370582690986339e-06, 6.5272170104435645e-06, 5.425981726148166e-05,
                               0.012856491841375828, 0.2976546883583069, 0.00013359056902118027,
                               0.005383729003369808]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('LSEG_merged_scores.csv', index=False)
print("LSEG merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/LSEG_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
                
############# CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/LSEG_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('LSEG_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### Prudential 

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = 'Prudential Plc'
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Prudential_news_sentiments.csv', index=False)
    print("Results exported to Prudential_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Prudential'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Prudential_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/PRU.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.17397519946098328, 0.9935975074768066, 1.546532985230442e-05,
                               2.0518098608590662e-05, 5.562438673223369e-06, 3.87560467061121e-05,
                               6.929576193215325e-05, 0.9906172156333923, 0.9999829530715942,
                               0.00012382016575429589]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Prudential_merged_scores.csv', index=False)
print("Prudential merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Prudential_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
                
############# CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Prudential_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Prudential_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

#### AVIVA

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = 'Aviva Plc'
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Aviva_news_sentiments.csv', index=False)
    print("Results exported to Aviva_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /AViva'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Aviva_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/AV.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.003049994818866253, 0.9983251690864563, 0.7531392574310303,
                               4.0454502595821396e-05, 3.525543797877617e-05, 1.555821108922828e-05,
                               7.928854029159993e-05, 8.91994423000142e-05, 2.0180348656140268e-05,
                               0.0005217308644205332]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Prudential_merged_scores.csv', index=False)
print("Prudential merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Aviva_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 

############# CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Aviva_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Aviva_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### F&C

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['F&C_Investment', 'Columbia_Threadneedle']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('F&C_news_sentiments.csv', index=False)
    print("Results exported to F&C_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /F&C'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/F&C_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/FCIT.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [6.7146979745302815e-06, 2.2628706574323587e-05, 2.8119240596424788e-05,
                               0.00011784218077082187, 0.0006054860423319042, 4.7506346163572744e-05,
                               4.2184085032204166e-05, 5.183105531614274e-05, 0.00023549240722786635,
                               9.839601261774078e-05]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('F&C_merged_scores.csv', index=False)
print("F&C merged scores")
    
final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/F&C_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
                
############# CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/F&C_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('F&C_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### Legal & Genetal

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['Legal & General']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Legal & General_news_sentiments.csv', index=False)
    print("Results exported to Legal & General_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /L&G'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Legal & General_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/LGEN.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.9998965263366699, 0.9360056519508362, 0.9986580610275269,
                               0.06711582094430923, 0.9998564720153809, 0.46595126390457153,
                               0.10098287463188171, 6.987973029026762e-05, 0.002596807200461626,
                               2.6166140742134303e-05]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('L&G_merged_scores.csv', index=False)
print("L&G merged scores")
    
final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/L&G_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
             
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/L&G_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('L&G_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### Admiral

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['Admiral']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Admiral_news_sentiments.csv', index=False)
    print("Results exported to Admiral_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Admiral '  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Admiral_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/ADM.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.008205738849937916, 0.13673053681850433, 0.015060077421367168,
                               1.298908955504885e-05, 6.757472147000954e-05, 4.5492932258639485e-05,
                               0.00020529353059828281, 0.00742283696308732, 0.0004826437507290393,
                               0.00034390040673315525]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Admiral_merged_scores.csv', index=False)
print("Admiral merged scores")
    
final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Admiral_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
             
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Admiral_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Admiral_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### Rathbones

API_KEY = '45f1b3a41edcbade3af84c195ca394981e83f4ad15deb55b557bd51220736379'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['Rathbones']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Rathbones_news_sentiments.csv', index=False)
    print("Results exported to Rathbones_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Rathbones'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Rathbones_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/RAT.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [4.4908183554071e-06, 4.248269760864787e-05, 5.391234935814282e-06,
                               2.3035074718791293e-06, 1.7717107766657136e-05, 6.570330151589587e-05,
                               1.1633388567133807e-05, 1.379525383526925e-05, 4.815411102754297e-06,
                               9.342803969047964e-05]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Rathbones_merged_scores.csv', index=False)
print("Rathbones merged scores")
    
final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Rathbones_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
                     
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Rathbones_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Rathbones_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

## Lancashire

API_KEY = '45f1b3a41edcbade3af84c195ca394981e83f4ad15deb55b557bd51220736379'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['Lancashire_Holdings']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Lancashire_news_sentiments.csv', index=False)
    print("Results exported to Lancashire_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Lancashire'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Lancashire_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/LRE.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.9424868822097778, 0.00022921463823877275, 0.008724680170416832,
                               0.020958399400115013, 0.0006483359611593187, 0.9999798536300659,
                               0.9999734163284302, 0.12711183726787567, 0.9998868703842163,
                               0.9999731779098511]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Lancashire_merged_scores.csv', index=False)
print("Lancashire merged scores")
    
final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Lancashire_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
                     
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Lancashire_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Lancashire_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### Schroders

API_KEY = '45f1b3a41edcbade3af84c195ca394981e83f4ad15deb55b557bd51220736379'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['Schroders PLC']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Schroders_news_sentiments.csv', index=False)
    print("Results exported to Schroders_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Schroders'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Schroders_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/SDR.L (1).csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [2.87337570625823e-05, 0.0001499900099588558, 0.025540873408317566,
                               0.05932480841875076, 0.9998621940612793, 0.6782357096672058,
                               0.9998987913131714, 0.9818521738052368, 6.490753003163263e-05,
                               0.9999977350234985]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Schroders_merged_scores.csv', index=False)
print("Schroders merged scores")
    
final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Schroders_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 

############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Schroders_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Schroders_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### Alliance

API_KEY = '45f1b3a41edcbade3af84c195ca394981e83f4ad15deb55b557bd51220736379'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['Alliance_Trust']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Alliance_news_sentiments.csv', index=False)
    print("Results exported to Alliance_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Alliance'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Alliance_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/ATST.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.7526746392250061, 0.220457524061203, 2.0274022972444072e-05,
                               0.5662381052970886, 0.020126093178987503, 0.0015547346556559205,
                               0.9997546076774597,  0.5857217907905579, 0.6458922028541565,
                               0.5291786789894104]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Alliance_merged_scores.csv', index=False)
print("Alliance merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Alliance_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
    
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Alliance_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Alliance_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

#### Blackrock

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['BlackRock_World_Mining']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('BlackRock_World_Mining_news_sentiments.csv', index=False)
    print("Results exported to BlackRock_World_Mining_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Blackrock'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/BlackRock_World_Mining_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/BRWM.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.030687568709254265, 1.0710211881814757e-06, 3.326004389236914e-06,
                               1.5632782606189721e-06, 0.024003885686397552, 0.000647016684524715,
                               0.004639412742108107, 3.830478453892283e-05, 1.3270142517285421e-05,
                               1.2044226423313376e-05]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('BlackRock_World_Mining_merged_scores.csv', index=False)
print("BlackRock_World_Mining merged scores")
    
final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/BlackRock_World_Mining_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 

############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/BlackRock_World_Mining_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Blackrock_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### JPMorgan

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['JPMorgan_American']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('JPMorgan_American_news_sentiments.csv', index=False)
    print("Results exported to JPMorgan_American_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /JP Morgan'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/JPMorgan_American_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/JAM.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [3.12669794766407e-06, 3.765612200368196e-05, 0.030687568709254265,
                               2.1865838789381087e-05, 1.0838192793016788e-05, 1.986454299185425e-05,
                               7.601941433676984e-06, 1.2388656614348292e-05, 2.8501939596026205e-05,
                               1.0819278941198718e-05]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('JPMorgan_American_merged_scores.csv', index=False)
print("JPMorgan_American merged scores")
    
final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/JPMorgan_American_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
    
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/JPMorgan_American_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('JPMorgan_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### ABRDN

API_KEY = 'c33705665b7c2f5c6548372097bb42a77cc2e5d4021ff9d3d9c180961dbd46a0'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['ABRDN']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('ABRDN_news_sentiments.csv', index=False)
    print("Results exported to ABRDN_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /ABRDN'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/ABRDN_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/ABDN.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.9732447862625122, 0.00026686713681556284, 3.51582930306904e-05,
                               0.00018046327750198543, 0.0016918806359171867, 0.001888458849862218,
                               1.9013550911495258e-07, 0.00026976066874340177, 3.5398225008975714e-05,
                               0.003808643901720643]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('ABRDN_merged_scores.csv', index=False)
print("ABRDN merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/ABRDN_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
    
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/ABRDN_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Function to adjust numeric format
def adjust_numeric_format(value):
    try:
        # Convert string to float, handle millions format
        return float(value.replace('.', '').replace(',', '.')) / 1000000.0
    except Exception as e:
        print(f"Error converting value {value}: {e}")
        return None

# Apply adjustment to numeric columns
for column in numeric_columns:
    final_data[column] = final_data[column].apply(adjust_numeric_format)

# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('ABRDN_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### Centrica

API_KEY = '45f1b3a41edcbade3af84c195ca394981e83f4ad15deb55b557bd51220736379'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['Centrica']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Centrica_news_sentiments.csv', index=False)
    print("Results exported to Centrica_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Centrica'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Centrica_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/CNA.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [0.6098423004150391, 0.9999333620071411, 0.2541638910770416,
                              4.5954540837556124e-05, 0.00014143428415991366, 0.9952130317687988,
                               0.00046074730926193297, 0.0073048993945121765, 0.9999880790710449,
                               2.4951852537924424e-05]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Centrica_merged_scores.csv', index=False)
print("Centrica merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Centrica_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
    
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Centrica_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Centrica_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### Witan

API_KEY = '45f1b3a41edcbade3af84c195ca394981e83f4ad15deb55b557bd51220736379'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['Witan_Investment_Trust']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Witan_news_sentiments.csv', index=False)
    print("Results exported to Witan_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Witan'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Witan_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/WTAN.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [8.86928137333598e-06, 4.176525180810131e-06, 1.8370775478615542e-06,
                              1.7690562117422814e-06, 0.00015034826355986297, 0.0009562232298776507,
                              3.3851472835522145e-05, 2.705510996747762e-05, 3.119527900707908e-05,
                               2.8004606065223925e-05]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Witan_merged_scores.csv', index=False)
print("Witan merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Witan_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
    
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Witan_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Witan_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))

### Tyman

API_KEY = '45f1b3a41edcbade3af84c195ca394981e83f4ad15deb55b557bd51220736379'
SERPAPI_ENDPOINT = 'https://serpapi.com/search'
finbert_model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def analyze_sentiment(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    sentiment_score = probabilities[1]
    return sentiment_score

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch article content from {url}: {e}")
        return None

def get_all_serpapi_results(query):
    all_results = []
    page = 0
    max_date = '31/12/2023'  # Adjust the date format as required by SerpApi
    min_date = '01/01/2014'

    while True:
        params = {
            'api_key': API_KEY,
            'engine': 'google',
            'q': query,
            'tbm': 'nws',
            'num': 100,
            'start': page * 100,
            'tbs': f'cdr:1,cd_min:{min_date},cd_max:{max_date}',
            'output': 'json',
        }

        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params)
            response.raise_for_status()
            results = response.json().get('news_results', [])
            
            if not results:
                print(f"No more articles found. Ending search.")
                break
            else:
                print(f"Total articles found on page {page + 1}: {len(results)}")
                all_results.extend(results)
            
            page += 1
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results for page {page + 1}: {e}")
            break

    return all_results

def filter_results_by_date(results, min_date, max_date):
    filtered_results = []
    min_date = datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.strptime(max_date, '%Y-%m-%d')

    for result in results:
        pub_date_str = result.get('date')
        
        try:
            pub_date = parser.parse(pub_date_str)
            
            # Check if the article date is within the desired range
            if min_date <= pub_date <= max_date:
                filtered_results.append(result)
            else:
                # Skip articles outside of the specified date range
                print(f"Skipping article with date {pub_date_str} outside of range")
        
        except ValueError:
            # Handle non-parseable dates or skip them
            print(f"Skipping article with non-parseable date: {pub_date_str}")
            continue
    
    return filtered_results

# Example usage:
query = ['Tyman']
all_results = get_all_serpapi_results(query)

if not all_results:
    print("No news articles found.")
else:
    print(f"Total articles found: {len(all_results)}\n")
    
    # Filter articles by date
    filtered_results = filter_results_by_date(all_results, '2014-01-01', '2023-12-31')
    print(f"Articles filtered by date range: {len(filtered_results)}\n")
    
    results_list = []
    
    # Analyze sentiment for filtered results
    for result in filtered_results:
        title = result.get('title')
        link = result.get('link')
        date = result.get('date')
        
        # Fetch article content
        article_content = fetch_article_content(link)
        
        if article_content:
            sentiment_score = analyze_sentiment(article_content)
            
            results_list.append({
                'title': title,
                'link': link,
                'date': date,
                'sentiment_score': sentiment_score
            })
            
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Published Date: {date}")
            print(f"Sentiment Score: {sentiment_score}")
            print("-" * 20)
        else:
            print(f"Failed to fetch content for article: {title} ({link})")
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv('Tyman_news_sentiments.csv', index=False)
    print("Results exported to Tyman_news_sentiments.csv")

### Sentiment score for annual reports 

import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to perform sentiment analysis using FinBERT
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]  # Get probabilities for each sentiment class
    sentiment_score = probabilities[1]  # Assuming index 1 is for positive sentiment
    return sentiment_score

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Directory containing your PDF reports
pdf_directory = '/Users/mac/Documents/Manchester/Dissertation/FS /Tyman'  # Change this to your actual directory path

# Iterate through all PDFs in the directory and perform sentiment analysis
sentiment_scores = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing {pdf_file}...")
    text = extract_text_from_pdf(pdf_path)
    sentiment_score = get_sentiment_score(text, tokenizer, model)
    sentiment_scores.append({'pdf': pdf_file, 'sentiment_score': sentiment_score})

# Display or save the results
for result in sentiment_scores:
    print(f"PDF: {result['pdf']}, Sentiment Score: {result['sentiment_score']}")
    
### Add sentiment score of financial news into historical stock price dataset
sentiment_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/sentiment_news/Tyman_news_sentiments.csv')
stock_prices_df = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/daily stock price/TYMN.L.csv', delimiter=';')

print("Sentiment DataFrame Columns:", sentiment_df.columns)
print("Stock Prices DataFrame Columns:", stock_prices_df.columns)

# Convert the date columns to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
stock_prices_df['Date'] = pd.to_datetime(stock_prices_df['Date'], dayfirst=True)

# Calculate average sentiment score for each day
average_sentiment_df = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
average_sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

# Merge sentiment scores with historical stock prices
merged_df = stock_prices_df.merge(average_sentiment_df, how='left', on='Date')

# Fill NaN sentiment scores with 0.5 (neutral sentiment)
merged_df['sentiment_score'].fillna(0.5, inplace=True)

annual_sentiments = pd.DataFrame({
    'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Annual_Report_Sentiment_Score': [6.704373390675755e-06, 0.9998175501823425, 0.999998927116394,
                              0.9999977350234985, 0.9858369827270508, 5.7512461353326216e-05,
                              0.9999992847442627, 4.8210589739028364e-05, 0.20243458449840546,
                               0.5955125093460083]  
})

merged_df['Year'] = merged_df['Date'].dt.year

# Merge the annual sentiment scores with the merged dataset
final_data = pd.merge(merged_df, annual_sentiments, on='Year', how='left')

# Display the first few rows of the final dataset
print(final_data.head())
final_data.to_csv('Tyman_merged_scores.csv', index=False)
print("Tyman merged scores")

final_data = pd.read_csv('/Users/mac/Documents/Manchester/Dissertation/Company/Tyman_merged_scores.csv')

# Check for NaN values
nan_counts = final_data.isnull().sum()

# Print NaN counts for each column
print("NaN counts for each column:")
print(nan_counts)

# Check if any NaN values exist in the DataFrame
if final_data.isnull().values.any():
    print("\nThere are missing values (NaN) in the DataFrame.")
else:
    print("\nNo missing values (NaN) found in the DataFrame.") 
    
############ CALCUALTION INDICATORS
import pandas as pd
import os
import ta

file_path = '/Users/mac/Documents/Manchester/Dissertation/Company/Tyman_merged_scores.csv'
final_data = pd.read_csv(file_path)

# Columns to adjust
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# Convert Date to datetime and set as index
if 'Date' in final_data.columns:
    final_data['Date'] = pd.to_datetime(final_data['Date'])
    final_data.set_index('Date', inplace=True)


# Function to compute features
def compute_features(df):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = ta.trend.ema_indicator(df['Adj Close'], window=20)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    
    # Calculate Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['Upper_BB'] = bb.bollinger_hband()
    df['Lower_BB'] = bb.bollinger_lband()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'], window=14)
    df['Stochastic_Oscillator'] = stoch.stoch()
    
    # Calculate Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Calculate MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['Adj Close'], df['Volume'])
    
    # Add lag features
    df['Lag_1'] = df['Adj Close'].shift(1)
    df['Lag_2'] = df['Adj Close'].shift(2)
    df['Lag_3'] = df['Adj Close'].shift(3)
    df['Lag_7'] = df['Adj Close'].shift(7)
    df['Lag_14'] = df['Adj Close'].shift(14)
    df['Lag_30'] = df['Adj Close'].shift(30)
    
    # Add day, month, year features
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    
    return df

# Compute features
final_data = compute_features(final_data)

# Check for NaN values and handle them if necessary
nan_counts = final_data.isnull().sum()
print("NaN counts for each column:")
print(nan_counts)
print(final_data.head())
# Impute missing values
for column in final_data.columns:
    if column == 'Volume':
        final_data[column] = final_data[column].fillna(0)  # Fill missing volume with 0
    else:
        final_data[column] = final_data[column].ffill().bfill()  # Forward and backward fill

# Verify no NaN values remain
if final_data.isnull().values.any():
    print("\nWarning: There are still missing values (NaN) in the DataFrame.")
else:
    print("\nImputation successful. No missing values remaining.")
    
# Display the first few rows of the DataFrame to verify
print(final_data.head())

# Save the revised DataFrame
final_data.to_csv('Tyman_final.csv', index=True)
final_data.to_excel('output.xlsx', float_format='%.6f')
final_data.to_csv('output.csv', index=False, float_format='%.6f')
print(final_data.head(20))







