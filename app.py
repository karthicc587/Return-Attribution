import streamlit as st
import pandas as pd
import yfinance as yf
import re
from datetime import datetime

# --- Robust Data Cleaning ---
def clean_currency(value):
    if pd.isna(value) or str(value).strip() in ['', '--']: 
        return 0.0
    val_str = str(value).strip()
    is_negative = False
    if val_str.startswith('(') and val_str.endswith(')'):
        is_negative = True
        val_str = val_str[1:-1]
    cleaned = re.sub(r'[^\d.-]', '', val_str)
    try:
        res = float(cleaned)
        return -res if is_negative else res
    except ValueError:
        return 0.0

def extract_date_from_filename(filename):
    match = re.search(r'([A-Z][a-z]{2}-\d{2}-\d{4})', filename)
    if match:
        return datetime.strptime(match.group(1), '%b-%d-%Y')
    return datetime.now()

# --- Streamlit UI ---
st.set_page_config(page_title="Portfolio Reconstructor", layout="wide")
st.title("📈 Portfolio vs. Russell 3000")

col1, col2 = st.columns(2)
with col1:
    pos_file = st.file_uploader("Upload Positions CSV", type=["csv"])
with col2:
    hist_file = st.file_uploader("Upload History CSV", type=["csv"])

if pos_file and hist_file:
    # 1. Load Data
    end_date = extract_date_from_filename(pos_file.name)
    pos_df = pd.read_csv(pos_file, index_col=False)
    hist_df = pd.read_csv(hist_file, skiprows=2, index_col=False)
    
    # 2. Extract Current State
    pos_df = pos_df[pos_df['Symbol'].notna() | pos_df['Account Name'].str.contains('Pending', na=False)].copy()
    
    spaxx_row = pos_df[pos_df['Symbol'] == 'SPAXX**']
    pending_row = pos_df[pos_df['Symbol'] == 'Pending activity']
    if pending_row.empty:
        pending_row = pos_df[pos_df['Account Name'] == 'Pending activity']
    
    current_cash = clean_currency(spaxx_row['Current Value'].iloc[0]) if not spaxx_row.empty else 0
    current_cash += clean_currency(pending_row['Current Value'].iloc[0]) if not pending_row.empty else 0
    
    stocks_df = pos_df[~pos_df['Symbol'].isin(['SPAXX**', 'Pending activity', None]) & 
                       ~pos_df['Account Name'].isin(['Pending activity'])].copy()
    
    current_positions = {str(row['Symbol']).strip(): clean_currency(row['Quantity']) 
                         for _, row in stocks_df.iterrows() if str(row['Symbol']).strip() != 'nan'}

    # 3. Process History
    hist_df['Run Date'] = pd.to_datetime(hist_df['Run Date'], errors='coerce')
    hist_df = hist_df[hist_df['Run Date'].notna()].copy()
    hist_df = hist_df.sort_values('Run Date', ascending=False)
    start_date = hist_df['Run Date'].min()
    
    all_tickers = sorted(list(set(list(current_positions.keys()) + hist_df['Symbol'].dropna().unique().tolist())))
    all_tickers = [t for t in all_tickers if t not in ['SPAXX**', 'SPAXX', 'Cash', 'Pending activity']]

    # 4. Fetch Historical Prices (Portfolio + Benchmark)
    with st.spinner("Fetching Market Data & Russell 3000 Index..."):
        # Fetch Portfolio Tickers
        price_data = yf.download(all_tickers, start=start_date, end=end_date + pd.Timedelta(days=1))['Close']
        
        # Fetch Russell 3000 (^RUA)
        benchmark_data = yf.download("^RUA", start=start_date, end=end_date + pd.Timedelta(days=1))['Close']
        
        # Drop Holidays and align indices
        price_data = price_data.dropna(how='all')
        price_data = price_data.ffill().bfill()
        benchmark_data = benchmark_data.reindex(price_data.index).ffill().bfill()

    # 5. Reconstruct Timeline
    history_records = []
    trading_days = price_data.index.sort_values(ascending=False)
    temp_positions, temp_cash = current_positions.copy(), current_cash
    
    for current_day in trading_days:
        day_ts = pd.Timestamp(current_day)
        
        # Calculate market value
        market_value = sum(temp_positions[t] * price_data.loc[day_ts, t] 
                           for t in temp_positions if t in price_data.columns)
        
        history_records.append({
            "Date": current_day.date(),
            "Market Value": round(market_value, 2),
            "Cash": round(temp_cash, 2),
            "Total Portfolio Value": round(market_value + temp_cash, 2),
            "Russell 3000": round(benchmark_data.loc[day_ts], 2) if day_ts in benchmark_data.index else None
        })
        
        # Backtrack transactions
        day_tx = hist_df[hist_df['Run Date'].dt.date == current_day.date()]
        for _, tx in day_tx.iterrows():
            ticker, qty = str(tx['Symbol']).strip(), float(tx['Quantity']) if not pd.isna(tx['Quantity']) else 0
            temp_positions[ticker] = temp_positions.get(ticker, 0) - qty
            temp_cash -= clean_currency(tx['Amount ($)'])

    # 6. Display Results
    reconstructed_df = pd.DataFrame(history_records).sort_values("Date")
    
    st.subheader("Performance Comparison")
    # Multi-axis chart or normalized chart might be better here, but showing total value for now
    st.line_chart(reconstructed_df.set_index("Date")[["Total Portfolio Value", "Russell 3000"]])
    
    st.subheader("Daily Data")
    st.dataframe(reconstructed_df, use_container_width=True)
    
    csv = reconstructed_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data (CSV)", csv, "Portfolio_vs_Russell3000.csv", "text/csv")
