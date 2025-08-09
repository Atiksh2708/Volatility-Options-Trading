import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def fetch_stock_data(symbol, period="5y", start_date=None, end_date=None):
    try:
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            data = ticker.history(start=start_date, end=end_date,auto_adjust=False)
        else:
            data = ticker.history(period=period,auto_adjust=False)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        data['Symbol'] = symbol
        
        data.reset_index(inplace=True)
        
        columns_order = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
        data = data[columns_order]
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def save_to_parquet(file_name,data):

    if data is None or data.empty:
        print("No data to save")
        return
    try:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        
        data.to_parquet(file_name, index=False)
        print(f"Data saved to {file_name}")
    except Exception as e:
        print(f"Error saving data to {file_name}: {str(e)}")

def get_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        relevant_info = {
            'Symbol': symbol,
            'Company Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Current Price': info.get('currentPrice', 'N/A')
        }
        
        return relevant_info
        
    except Exception as e:
        print(f"Error getting stock info: {str(e)}")
        return None
    
symbol = "AAPL"  # Apple Inc. - S&P 500 company
    
print(f"Fetching data for {symbol}...")
    
stock_info = get_stock_info(symbol)
if stock_info:
    print("\nStock Information:")
    for key, value in stock_info.items():
        print(f"{key}: {value}")
data = fetch_stock_data(symbol, period="5y")

if data is not None:
    print(f"\nFetched {len(data)} days of data")
    print(f"Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
    
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nBasic Statistics:")
    print(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].describe())
    
    save_to_parquet(f"data/{symbol}_5y.parquet", data)

else:
    print("Failed to fetch data")


apple_data=pd.read_parquet("data/AAPL_5y.parquet")
print("\nData loaded from parquet:")
print(apple_data.head())