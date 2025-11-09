import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import requests
from bs4 import BeautifulSoup
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax
import asyncio
import aiohttp


# Set Streamlit page configuration with custom theme and favicon
st.set_page_config(
    page_title="Portfolio Assessment",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the app
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #222222;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 5px;
    }
    .stDataFrame {
        background-color: #222222;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load FinBERT model for sentiment analysis
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Function to fetch company name from ticker using Yahoo Finance API
@st.cache_data(show_spinner=False)
def get_company_name(ticker):
    url = f"https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": ticker, "quotes_count": 1, "country": "United States"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            return data["quotes"][0]["longname"]
        return None
    except Exception as e:
        print(f"Error fetching company name for ticker {ticker}: {e}")
        return None

# Function to fetch news asynchronously with increased header limits
async def fetch_news_async(url, headers):
    connector = aiohttp.TCPConnector(limit_per_host=10)
    async with aiohttp.ClientSession(
        connector=connector,
        read_bufsize=2**16,
        max_line_size=65536,
        max_field_size=65536
    ) as session:
        async with session.get(url, headers=headers) as response:
            return await response.text()

# Asynchronous function to fetch Yahoo Finance news
async def get_news_yahoo_async(stock_ticker):
    url = f"https://finance.yahoo.com/quote/{stock_ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = await fetch_news_async(url, headers)
    soup = BeautifulSoup(html, "html.parser")
    return [h.text for h in soup.find_all("h3")]

# Asynchronous function to fetch Google News RSS feed
async def get_news_google_async(company_name):
    url = f"https://news.google.com/rss/search?q={company_name}+stock&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    xml = await fetch_news_async(url, headers)
    soup = BeautifulSoup(xml, "xml")
    return [item.title.text for item in soup.find_all("item")]

# Wrapper function to fetch all news asynchronously
@st.cache_data(show_spinner=False)
def fetch_all_news(stock_ticker, company_name):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yahoo_news, google_news = loop.run_until_complete(
        asyncio.gather(
            get_news_yahoo_async(stock_ticker),
            get_news_google_async(company_name)
        )
    )
    return yahoo_news + google_news

# Batch processing for sentiment analysis
@st.cache_data(show_spinner=False)
def analyze_sentiment_batch(news_headlines):
    batch_size = 8  # Process headlines in batches
    sentiment_results = []
    
    for i in range(0, len(news_headlines), batch_size):
        batch = news_headlines[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs_batch = softmax(outputs.logits.numpy(), axis=1)
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        
        for text, probs in zip(batch, probs_batch):
            sentiment_scores = {sentiment_labels[i]: probs[i] for i in range(len(sentiment_labels))}
            sentiment_scores["headline"] = text
            sentiment_results.append(sentiment_scores)
    
    df_sentiment = pd.DataFrame(sentiment_results)
    df_sentiment["Final Sentiment"] = df_sentiment[["Positive", "Neutral", "Negative"]].idxmax(axis=1)
    
    sentiment_score = (
        (df_sentiment["Final Sentiment"].value_counts().get("Positive", 0) * 1) +
        (df_sentiment["Final Sentiment"].value_counts().get("Neutral", 0) * 0) +
        (df_sentiment["Final Sentiment"].value_counts().get("Negative", 0) * -1)
    ) / len(df_sentiment) if len(df_sentiment) > 0 else 0
    
    return df_sentiment, sentiment_score

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


# Load the saved model and data for risk assessment
best_xgb = XGBRegressor()
best_xgb.load_model("xgboost_risk_model.json")
risk_data = pd.read_csv("extracted_data_stocks.csv")
stock_dataset=risk_data['Stock_Name']

# Sidebar Navigation with emojis and tooltips
st.sidebar.title("📊 Navigation")
page_selection_tooltip = "Select the page you'd like to explore."
page = st.sidebar.radio("Go to:", ["Home", "Risk and Sentiment Analysis", "Tableau Dashboard"], help=page_selection_tooltip)

features_for_modeling=['Volatility', 'Beta', 'Sharpe_Ratio', 'Max_Drawdown', 'PE_Ratio', 'EPS', '50_MA_Last', '200_MA_Last', 'Number_of_news_headlines', 'Average_sentiment_score', 'Beta_1', 'Beta_2', 'Beta_3', 'Beta_6', 'PE_Ratio_1', 'Volatility_1', 'PE_Ratio_2', 'Volatility_2', 'PE_Ratio_3', 'Volatility_3', 'PE_Ratio_6', 'Volatility_6', 'Heuristic_Risk_Score', 'Cluster_Label', 'Cluster_Risk_Score', 'Final_Risk_Score', 'Volatility_Beta', 'Max_Drawdown_Sharpe',"Volatility_Beta", "Max_Drawdown_Sharpe"]
features_to_display=['Stock_Name', 'Volatility', 'Beta', 'Sharpe_Ratio', 'Max_Drawdown']

# Initialize session state for percentages
if "percentages" not in st.session_state:
    st.session_state.percentages = []
if "remaining" not in st.session_state:
    st.session_state.remaining = 1.0  # Start with 100% available

# Callback function to update remaining percentage
def update_remaining():
    try:
        # Parse percentages from user input
        percentages = [float(p.strip()) for p in st.session_state.percentages_input.split(",") if p.strip()]
        total_percentage = sum(percentages)
        
        # Update remaining percentage
        st.session_state.remaining = max(0, 1 - total_percentage)
    except ValueError:
        st.error("Please enter valid numeric percentages.")

# Home Page
if page == "Home":
    st.title("Welcome to the Portfolio Risk Assessment Tool")
    st.subheader("How It Works:")
    st.write(
        """
        This tool helps you analyze the risk and sentiment of your investment portfolio.
        
        **Features:**
        - **Risk and Sentiment Analysis**: Enter your portfolio details (stock names and percentages) to calculate risk scores, sentiment analysis, and other performance metrics.
        - **Tableau Dashboard**: Visualize your portfolio's performance using interactive dashboards.
        
        **How to Use:**
        1. Navigate to the desired section using the buttons on the navigation panel.
        2. For risk assessment, provide stock names and their respective allocation percentages.
        3. View detailed visualizations in the Tableau Dashboard section.
        """
    )


# Risk Assessment Page with Sentiment Analysis Integration and Risk Scores Calculation
elif page == "Risk and Sentiment Analysis":
    st.header("Enter Portfolio Information")
    
    #user stocks
    user_stocks = st.multiselect(
        label="Enter stock names:",
        options=stock_dataset,
        help="Start typing a stock name or ticker symbol to see suggestions."
    )
    # percentage_stocks = st.text_input("Enter their respective percentages (comma-separated):").split(",")
    
    # Input for percentages with dynamic updates
    st.text_input(
        "Enter their respective percentages (comma-separated):",
        key="percentages_input",
        on_change=update_remaining,
        help="Enter percentages as comma-separated values (e.g., 0.4, 0.3)."
    )
    
    # Display remaining percentage dynamically
    if st.session_state.remaining >= 0:
        st.info(f"Remaining Percentage: {st.session_state.remaining:.2f}")
    else:
        st.error("The total exceeds 1. Please adjust your percentages.")
    
    if st.button("Calculate Risk Scores and Sentiments"):
        percentage_stocks = [float(p.strip()) for p in st.session_state.percentages_input.split(",") if p.strip()]
        if len(user_stocks) != len(percentage_stocks):
            st.error("The number of stocks and percentages must match.")
        elif len(user_stocks) == 0 or len(percentage_stocks) == 0:
            st.error("Please enter valid stock names and percentages.")
        else:
            # Initialize variables to store results
            stock_sentiments = {}
            total_weighted_sentiment_score = 0
            
            ticker_list=[]
            for stock in user_stocks:
                ticker_list.append(get_ticker(stock))
            
            # Filter data for selected stocks and calculate risk scores
            risk_data_filtered = risk_data[risk_data['Stock'].isin(ticker_list)]
            risk_data_filtered['Stock_Name']=user_stocks
            if risk_data_filtered.empty:
                st.error("No data found for the entered stocks.")
            else:
                risk_data_filtered.dropna(inplace=True)
                risk_data_filtered.reset_index(inplace=True, drop=True)
                # Normalize features for risk assessment
                scaler = MinMaxScaler()
                risk_data_scaled = scaler.fit_transform(risk_data_filtered[features_for_modeling])

                # Predict individual risk scores using the model
                individual_risk_scores = best_xgb.predict(risk_data_scaled)

                # Calculate portfolio risk score (weighted average)
                portfolio_risk_score = sum(
                    [float(x) * float(y) for x, y in zip(individual_risk_scores, percentage_stocks)]
                ) / sum(percentage_stocks)

                # Analyze each stock's sentiment and calculate weighted average sentiment score
                for ticker,stock, weight in zip(ticker_list,user_stocks,percentage_stocks):
                    all_news = fetch_all_news(ticker, ticker)
                    
                    if not all_news:
                        st.warning(f"No news articles found for {stock}.")
                        stock_sentiments[stock] = {"Sentiment Score": 0}
                    else:
                        _, sentiment_score = analyze_sentiment_batch(all_news)
                        stock_sentiments[stock] = {"Sentiment Score": sentiment_score}
                        total_weighted_sentiment_score += sentiment_score * weight
                
                # Normalize weighted average sentiment score by total weights
                weighted_avg_sentiment_score = total_weighted_sentiment_score / sum(percentage_stocks)

                # Display results
                st.subheader("Risk Assessment Results")
                st.write(risk_data_filtered[features_to_display])
                
                for stock, score in zip(user_stocks, individual_risk_scores):
                    st.write(f"{stock}: Risk Score: {score:.2f}")
                
                st.subheader(f"Risk Score for Portfolio: {portfolio_risk_score:.2f}")

                st.subheader("Sentiment Analysis Results")
                for stock, data in stock_sentiments.items():
                    st.write(f"{stock}: Sentiment Score: {data['Sentiment Score']:.2f}")
                
                st.subheader(f"Sentiment Score for Portfolio: {weighted_avg_sentiment_score:.2f}")

# Tableau Dashboard Page
elif page == "Tableau Dashboard":
    st.header("Portfolio Visualization Dashboard")
    
    tableau_embed_code = """
    <div class='tableauPlaceholder' id='viz1743610208386' style='position: relative; width: 100%; height: 90vh;'>
        <noscript>
            <a href='#'>
                <img alt='Dashboard 1' src='https://public.tableau.com/static/images/Ca/Capstone_17436101380210/Dashboard1/1_rss.png' style='border: none;' />
            </a>
        </noscript>
        <object class='tableauViz'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' />
            <param name='site_root' value='' />
            <param name='name' value='Capstone_17436101380210/Dashboard1' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https://public.tableau.com/static/images/Ca/Capstone_17436101380210/Dashboard1/1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-US' />
        </object>
    </div>
    <script type="text/javascript">
        var divElement = document.getElementById('viz1743610208386');
        var vizElement = divElement.getElementsByTagName('object')[0];
        vizElement.style.width = "100%"; // Full width of parent container
        vizElement.style.height = "90vh"; // Adjust height for better visibility
        var scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """
    
    # Embed the Tableau dashboard in Streamlit
    st.components.v1.html(tableau_embed_code, width=20000, height=8000, scrolling=False)

