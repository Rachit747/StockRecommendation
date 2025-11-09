import yfinance as yf
import pandas as pd
import numpy as np
import requests
import nltk
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from pandas_datareader import data as pdr
from datetime import datetime, timedelta,date
import pytz
from dateutil.relativedelta import relativedelta
import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import boto3
import warnings
warnings.filterwarnings('ignore')

def load_from_s3(bucket_name, file_name):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    return pd.read_csv(obj['Body'])

risk_data = load_from_s3('your-bucket-name', 'extracted_data_stocks.csv')
best_xgb = joblib.load('/tmp/xgboost_risk_model_tuned.pkl')

# Load NLP Model
nlp = spacy.load("en_core_web_trf")
# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def save_to_s3(dataframe, bucket_name, file_name):
    """Save a DataFrame to S3 as a CSV file."""
    s3 = boto3.client('s3')
    csv_buffer = dataframe.to_csv(index=False)
    s3.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer)

def lambda_handler(event, context):
    # Your existing logic to fetch and process stock data
    risk_data = get_stocks()
    
    # Save the processed data to S3
    save_to_s3(risk_data, 'riskdatametricsbucket', 'extracted_data_stocks.csv')
    return {"statusCode": 200, "body": "Data extraction completed and uploaded to S3"}


def get_ticker(company_name):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
    res = requests.get(url=url, params=params, headers=headers)
    data = res.json()
    try:
        company_code = data['quotes'][0]['symbol']
        return company_code
    except Exception as e:
        print(f"Error: {e}")
        return None

# Get S&P 500 tickers
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]

# Function to Clean Text
def clean_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())  
    words = [word for word in words if word.isalnum()]  
    words = [word for word in words if word not in stop_words]  
    return " ".join(words)

# Function to fetch Google News sentiment
def fetch_google_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "xml")
        headlines = [item.title.text for item in soup.find_all("item")]

        if not headlines:
            return None, None

        cleaned_headlines = [clean_text(headline) for headline in headlines]
        sentiment_scores = [sia.polarity_scores(headline)["compound"] for headline in cleaned_headlines]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

        return len(cleaned_headlines), avg_sentiment

    except Exception as e:
        print(f"Error fetching news: {e}")
        return None, None

market = yf.Ticker("^GSPC").history(period="2y")["Close"]

# Function to fetch stock data with lag features
def fetch_stock_data(ticker, period="2y"):
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        print(f"No data found for {ticker}")
        return None

    hist["50_MA"] = hist["Close"].rolling(window=50).mean()
    if pd.isna(hist["50_MA"]).all():
        hist["50_MA"] = hist["Close"].rolling(window=len(hist)).mean()
       
    hist["200_MA"] = hist["Close"].rolling(window=200).mean()
    if pd.isna(hist["200_MA"]).all():
        hist["200_MA"] = hist["Close"].rolling(window=len(hist)).mean()
        
    

    hist["Daily_Return"] = hist["Close"].pct_change()
    volatility = np.log(hist["Close"] / hist["Close"].shift(1)).dropna().std() * np.sqrt(252)

    hist["Cumulative_Max"] = hist["Close"].cummax()
    hist["Drawdown"] = (hist["Close"] - hist["Cumulative_Max"]) / hist["Cumulative_Max"]
    max_drawdown = hist["Drawdown"].min()

    info = stock.info
    pe_ratio = info.get("trailingPE", np.nan)
    eps = info.get("trailingEps", np.nan)
    beta = info.get("beta", np.nan)
    
    if np.isnan(beta):
        stock_data = hist["Close"]
        market_data = market
        data = pd.DataFrame({"Stock": stock_data, "Market": market_data}).dropna()
        # Calculate daily returns
        data["Stock_Returns"] = data["Stock"].pct_change()
        data["Market_Returns"] = data["Market"].pct_change()
        # Drop NaN values after pct_change
        data = data.dropna()
        # Compute Beta
        covariance = np.cov(data["Stock_Returns"], data["Market_Returns"])[0, 1]
        variance = np.var(data["Market_Returns"])

        beta = covariance / variance
    
    if np.isnan(pe_ratio):
        # Get latest closing price
        latest_close = hist["Close"].iloc[-1]
        # Compute PE Ratio
        pe_ratio = latest_close / eps if eps != 0 else np.nan

    mean_return = hist["Daily_Return"].mean() * 252
    sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility != 0 else np.nan

    number_of_news, avg_sentiment = fetch_google_news(ticker)
    
    lag_values = {}
    current_date = pd.Timestamp(date.today(), tz="America/New_York")
    # lag values for beta
    for months in [1, 2, 3, 6]:
        target_date=current_date-relativedelta(months=months)
        stock_data = hist["Close"]
        market_data = market
        data = pd.DataFrame({"Stock": stock_data, "Market": market_data}).dropna()
        data["Stock_Returns"] = data["Stock"].pct_change()
        data["Market_Returns"] = data["Market"].pct_change()
        data = data.dropna()
        filtered_data = data[data.index <= target_date]
        covariance = np.cov(filtered_data["Stock_Returns"], filtered_data["Market_Returns"])[0, 1]
        variance = np.var(filtered_data["Market_Returns"])
        
        lag_beta = covariance / variance
        
        lag_values[f"Beta_{months}"] = lag_beta
        
    for months in [1, 2, 3, 6]:
        window_days=months*21
        lag_values[f"PE_Ratio_{months}"]= hist["Close"].iloc[-window_days] / eps if len(hist) > window_days else np.nan
        lag_values[f"Volatility_{months}"] = np.log(hist["Close"] / hist["Close"].shift(window_days)).dropna().std() * np.sqrt(252)

    stock_metrics = {
        "Stock": ticker,
        "Volatility": volatility,
        "Beta": beta,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "PE_Ratio": pe_ratio,
        "EPS": eps,
        "50_MA_Last": hist["50_MA"].iloc[-1],
        "200_MA_Last": hist["200_MA"].iloc[-1],
        "Number_of_news_headlines": number_of_news,
        "Average_sentiment_score": avg_sentiment,
    }
    
    stock_metrics.update(lag_values)
    
    return stock_metrics

# Function to fetch stock data for all S&P 500 stocks
def get_stocks():
    tickers = get_sp500_tickers()
    portfolio_risk_data = []
    
    for ticker in tickers['Symbol']:
        stock_data = fetch_stock_data(ticker)
        if stock_data:
            portfolio_risk_data.append(stock_data)
    
    # for ticker in tickers:
    #     stock_data = fetch_stock_data(ticker)
    #     if stock_data:
    #         portfolio_risk_data.append(stock_data)

    return pd.DataFrame(portfolio_risk_data)

# Fetch risk-free rate
risk_free_rate = pdr.get_data_fred('DGS3MO').iloc[-1, 0] / 100  

# Store final extracted data
risk_data = get_stocks()
# Feature Selection
features = ["Volatility", "Beta", "Max_Drawdown", "Sharpe_Ratio", "Average_sentiment_score"]

# Heuristic risk score
def heuristic_risk_score(row):
    return (
        0.3 * row["Volatility"] +
        0.3 * abs(row["Max_Drawdown"]) +
        0.2 * row["Beta"] +
        -0.1 * row["Sharpe_Ratio"] +  # Negative weight (higher Sharpe = lower risk)
        -0.1 * row["Average_sentiment_score"]  # Negative weight (higher sentiment = lower risk)
    )

risk_data["Heuristic_Risk_Score"] = risk_data.apply(heuristic_risk_score, axis=1)

# Normalize heuristic scores
scaler = MinMaxScaler(feature_range=(0, 1))
risk_data["Heuristic_Risk_Score"] = scaler.fit_transform(risk_data[["Heuristic_Risk_Score"]])

# Normalize features
risk_data_scaled = scaler.fit_transform(risk_data[features])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
risk_data["Cluster_Label"] = kmeans.fit_predict(risk_data_scaled)

# Dynamically map cluster labels to risk scores based on cluster centroids
cluster_centers = kmeans.cluster_centers_.mean(axis=1)
sorted_clusters = np.argsort(cluster_centers)
cluster_risk_mapping = {sorted_clusters[i]: (i + 1) / 6 for i in range(5)}  # 0.166, 0.333, 0.5, 0.667, 0.833 range
risk_data["Cluster_Risk_Score"] = risk_data["Cluster_Label"].map(cluster_risk_mapping)

# Weighted Average
risk_data["Final_Risk_Score"] = (0.6 * risk_data["Heuristic_Risk_Score"] + 0.4 * risk_data["Cluster_Risk_Score"])

#XGBoost modeling
features = ['Volatility', 'Beta', 'Sharpe_Ratio', 'Max_Drawdown', 'PE_Ratio', 'EPS', '50_MA_Last', '200_MA_Last', 'Number_of_news_headlines', 'Average_sentiment_score', 'Beta_1', 'Beta_2', 'Beta_3', 'Beta_6', 'PE_Ratio_1', 'Volatility_1', 'PE_Ratio_2', 'Volatility_2', 'PE_Ratio_3', 'Volatility_3', 'PE_Ratio_6', 'Volatility_6', 'Heuristic_Risk_Score', 'Cluster_Label', 'Cluster_Risk_Score', 'Final_Risk_Score', 'Volatility_Beta', 'Max_Drawdown_Sharpe']
# Feature Engineering
risk_data["Volatility_Beta"] = risk_data["Volatility"] * risk_data["Beta"]
risk_data["Max_Drawdown_Sharpe"] = risk_data["Max_Drawdown"] * risk_data["Sharpe_Ratio"]
features.extend(["Volatility_Beta", "Max_Drawdown_Sharpe"])
