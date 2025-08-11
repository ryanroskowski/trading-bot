from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import random

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class PriceData:
    close: pd.DataFrame
    open: pd.DataFrame


def fetch_yf_ohlcv_direct_api(tickers: List[str], start: str = "2005-01-01", end: Optional[str] = None) -> PriceData:
    """
    Fetch OHLCV data directly from Yahoo Finance API, bypassing yfinance library.
    This avoids the rate limiting issues that yfinance sometimes encounters.
    """
    import requests
    import json
    from datetime import datetime
    
    def date_to_timestamp(date_str: str) -> int:
        """Convert date string to Unix timestamp."""
        dt = pd.to_datetime(date_str)
        return int(dt.timestamp())
    
    start_ts = date_to_timestamp(start)
    end_ts = date_to_timestamp(end) if end else int(datetime.now().timestamp())
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    all_close_data = {}
    all_open_data = {}
    
    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")
            
            # Use Yahoo Finance Chart API directly
            url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': '1d',
                'includePrePost': 'false',
                'events': 'div,splits'
            }
            
            response = session.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"Failed to fetch {ticker}: HTTP {response.status_code}")
                continue
                
            data = response.json()
            
            if 'chart' not in data or not data['chart']['result']:
                print(f"No chart data for {ticker}")
                continue
                
            result = data['chart']['result'][0]
            
            if 'timestamp' not in result or not result['timestamp']:
                print(f"No timestamp data for {ticker}")
                continue
                
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Check if we have required OHLC data
            if not all(key in quotes for key in ['open', 'high', 'low', 'close']):
                print(f"Missing OHLC data for {ticker}")
                continue
            
            # Convert timestamps to datetime index
            dates = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('America/New_York').tz_localize(None)
            
            # Extract OHLC data
            opens = quotes['open']
            closes = quotes['close']
            
            # Handle adjusted close if available
            if 'adjclose' in result['indicators']:
                adj_closes = result['indicators']['adjclose'][0]['adjclose']
            else:
                adj_closes = closes
            
            # Create raw series
            open_raw = pd.Series(opens, index=dates).astype(float)
            close_raw = pd.Series(closes, index=dates).astype(float)
            adj_close = pd.Series(adj_closes, index=dates).astype(float)

            # Adjust open to be consistent with adjusted close to avoid spurious open-to-close returns
            # factor = adj_close / close_raw; adj_open = open_raw * factor
            with pd.option_context('mode.use_inf_as_na', True):
                factor = (adj_close / close_raw).replace([np.inf, -np.inf], np.nan)
            open_adj = (open_raw * factor).dropna()
            close_series = adj_close.dropna()
            open_series = open_adj.align(close_series, join='inner')[0]
            
            if len(open_series) > 0 and len(close_series) > 0:
                all_open_data[ticker] = open_series
                all_close_data[ticker] = close_series
                print(f"Successfully fetched {len(close_series)} days for {ticker}")
            else:
                print(f"No valid data after cleaning for {ticker}")
                
            # Add small delay between requests
            time.sleep(0.5 + random.uniform(0, 0.5))
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue
    
    if not all_close_data:
        raise RuntimeError(f"Failed to fetch data for any of the requested tickers: {tickers}")
    
    # Combine into DataFrames
    close_df = pd.DataFrame(all_close_data).sort_index()
    open_df = pd.DataFrame(all_open_data).sort_index()
    
    print(f"Successfully fetched data for {len(all_close_data)} tickers")
    print(f"Date range: {close_df.index[0]} to {close_df.index[-1]}")
    print(f"Shape: {close_df.shape}")
    
    return PriceData(close=close_df, open=open_df)


def fetch_yf_ohlcv(tickers: List[str], start: str = "2005-01-01", end: Optional[str] = None) -> PriceData:
    """
    Fetch OHLCV data from Yahoo Finance. First tries direct API, falls back to yfinance library.
    """
    try:
        print("Attempting direct Yahoo Finance API...")
        return fetch_yf_ohlcv_direct_api(tickers, start, end)
    except Exception as e:
        print(f"Direct API failed: {e}")
        print("Falling back to yfinance library...")
        
        # Fallback to original yfinance approach with conservative settings
        try:
            print(f"Fetching data for {len(tickers)} tickers using yfinance...")
            data = yf.download(
                tickers=tickers, 
                start=start, 
                end=end, 
                auto_adjust=False, 
                progress=False, 
                group_by="ticker" if len(tickers) > 1 else None
            )
            
            if data.empty:
                raise RuntimeError("yfinance returned empty data")
            
            # Process the data
            frames_close = {}
            frames_open = {}
            
            for t in tickers:
                try:
                    if len(tickers) == 1:
                        df = data
                    else:
                        if isinstance(data.columns, pd.MultiIndex) and t in data.columns.get_level_values(0):
                            df = data[t]
                        else:
                            continue
                    
                    if 'Adj Close' in df.columns and 'Open' in df.columns and 'Close' in df.columns:
                        # Adjust open by same factor as close adjustment to ensure consistency
                        close_series = df["Adj Close"].astype(float)
                        factor = (df["Adj Close"].astype(float) / df["Close"].astype(float)).replace([np.inf, -np.inf], np.nan)
                        open_series = (df["Open"].astype(float) * factor)
                        # Drop rows where either side is NaN
                        aligned = pd.concat([open_series, close_series], axis=1, join='inner').dropna()
                        open_series = aligned.iloc[:, 0]
                        close_series = aligned.iloc[:, 1]
                        
                        if len(close_series) > 0 and len(open_series) > 0:
                            frames_close[t] = close_series.rename(t)
                            frames_open[t] = open_series.rename(t)
                            
                except Exception as e:
                    print(f"Error processing {t}: {e}")
                    continue
            
            if not frames_close:
                raise RuntimeError("No valid data extracted from yfinance")
            
            close = pd.concat(frames_close.values(), axis=1).sort_index()
            open_ = pd.concat(frames_open.values(), axis=1).sort_index()
            
            # Clean timezone
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)
            open_.index = pd.DatetimeIndex(open_.index).tz_localize(None)
            
            print(f"yfinance fallback successful: {close.shape}")
            return PriceData(close=close, open=open_)
            
        except Exception as yf_error:
            raise RuntimeError(f"Both direct API and yfinance failed. "
                             f"Direct API error: {e}. yfinance error: {yf_error}")



def align_next_bar_execution(signals: pd.DataFrame, open_prices: pd.DataFrame) -> pd.DataFrame:
    # Shift signals one bar forward to ensure next-open execution without lookahead
    return signals.shift(1).reindex(open_prices.index)


