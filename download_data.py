"""
Stock Data Download Script
Downloads historical stock data from Yahoo Finance and saves to JSON

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --start 2018-01-01 --end 2024-12-31
    python scripts/download_data.py --tickers AAPL,MSFT,GOOGL
"""

import yfinance as yf
import json
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path


def clean_data(df, start_date, end_date, threshold=0.75):
    """
    Clean and fill missing data
    
    Args:
        df: DataFrame with stock data
        start_date: Start date string
        end_date: End date string
        threshold: Threshold for extreme changes (75%)
    
    Returns:
        Cleaned DataFrame
    """
    # Convert index to date only (remove time)
    df.index = pd.to_datetime(df.index.date)
    
    # Calculate means for filling
    means = df[['Open', 'High', 'Low', 'Close', 'Volume']].mean()
    
    # Create full date range
    all_dates = pd.date_range(start_date, end_date, freq='D')
    cleaned = df.reindex(all_dates)
    
    # Fill missing values
    cleaned[['Open', 'High', 'Low', 'Close', 'Volume']] = \
        cleaned[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(means)
    
    # Handle extreme fluctuations
    for col in ['Open', 'High', 'Low', 'Close']:
        pct_change = cleaned[col].pct_change().abs()
        extreme = (pct_change >= threshold) & (~cleaned[col].isna())
        cleaned.loc[extreme, col] = means[col]
    
    cleaned.index.name = "Date"
    return cleaned.reset_index()


def download_stock(ticker, start_date, end_date):
    """
    Download historical data for a single stock
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
    
    Returns:
        List of dictionaries with stock data
    """
    print(f"Downloading {ticker}... ", end='')
    
    try:
        # Download from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"❌ No data found")
            return None
        
        # Clean data
        df = clean_data(df, start_date, end_date)
        
        # Convert to JSON-serializable format
        data = df.to_dict(orient='records')
        
        print(f"✓ Downloaded {len(data)} days")
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def save_to_json(data, output_file):
    """
    Save stock data to JSON file
    
    Args:
        data: Dictionary of {ticker: data}
        output_file: Output JSON file path
    """
    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
        except:
            pass
    
    # Merge with new data
    existing_data.update(data)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=2, default=str)
    
    print(f"\n✓ Saved to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Download historical stock data from Yahoo Finance'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        default='AAPL,AMZN,GOOGL,MSFT,TSLA,JNJ,JPM,V,PG,UNH',
        help='Comma-separated list of stock tickers'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default='2018-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/historical.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    
    print("=" * 60)
    print("STOCK DATA DOWNLOAD")
    print("=" * 60)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()
    
    # Download all stocks
    all_data = {}
    success_count = 0
    
    for ticker in tickers:
        data = download_stock(ticker, args.start, args.end)
        if data:
            all_data[ticker] = data
            success_count += 1
    
    # Save to file
    if all_data:
        save_to_json(all_data, args.output)
        
        print("\n" + "=" * 60)
        print(f"✓ Successfully downloaded {success_count}/{len(tickers)} stocks")
        print("=" * 60)
    else:
        print("\n❌ No data downloaded")
        return 1
    
    return 0


# Additional utility functions

def update_existing_data(json_file, end_date=None):
    """
    Update existing JSON file with latest data
    
    Args:
        json_file: Path to existing JSON file
        end_date: Optional end date (default: today)
    """
    if not os.path.exists(json_file):
        print(f"File {json_file} not found")
        return
    
    # Load existing data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Updating {len(data)} stocks to {end_date}...")
    
    # Update each stock
    updated_data = {}
    for ticker, entries in data.items():
        # Find latest date in existing data
        latest = max(e['Date'] for e in entries)
        start_date = (datetime.fromisoformat(latest) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        if start_date >= end_date:
            print(f"{ticker}: Already up to date")
            updated_data[ticker] = entries
            continue
        
        # Download new data
        new_data = download_stock(ticker, start_date, end_date)
        
        if new_data:
            # Merge with existing
            updated_data[ticker] = entries + new_data
        else:
            updated_data[ticker] = entries
    
    # Save updated data
    save_to_json(updated_data, json_file)


def get_stock_info(ticker):
    """
    Get detailed information about a stock
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Dictionary with stock info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD')
        }
    except:
        return None


def validate_data(json_file):
    """
    Validate downloaded data for issues
    
    Args:
        json_file: Path to JSON file
    """
    print(f"Validating {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    issues = []
    
    for ticker, entries in data.items():
        # Check if data exists
        if not entries:
            issues.append(f"{ticker}: No data")
            continue
        
        # Check for gaps
        dates = [datetime.fromisoformat(e['Date'][:10]) for e in entries]
        dates.sort()
        
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap > 7:  # More than a week gap
                issues.append(f"{ticker}: {gap}-day gap at {dates[i-1].date()}")
        
        # Check for zero/negative prices
        for entry in entries:
            if entry.get('Close', 0) <= 0:
                issues.append(f"{ticker}: Invalid price on {entry['Date']}")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✓ Data validation passed")


if __name__ == "__main__":
    # Run main download
    exit_code = main()
    
    # Optionally validate
    if exit_code == 0:
        import sys
        if '--validate' in sys.argv:
            validate_data('data/historical.json')
    
    exit(exit_code)