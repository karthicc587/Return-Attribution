import streamlit as st
import pandas as pd
import yfinance as yf
import re
from datetime import datetime

# --- Helper Functions ---
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
st.title("📈 Portfolio Performance vs. Russell 3000")

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
    
    # 2. Extract Current State (End Date)
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

    # 4. Fetch Historical Prices
    with st.spinner("Fetching market data..."):
        # Fetch Portfolio Tickers
        prices = yf.download(all_tickers, start=start_date, end=end_date + pd.Timedelta(days=1))['Close']
        
        # Fetch Russell 3000 (^RUA) and ensure it's a Series
        bench_raw = yf.download("^RUA", start=start_date, end=end_date + pd.Timedelta(days=1))['Close']
        if isinstance(bench_raw, pd.DataFrame):
            benchmark_data = bench_raw.iloc[:, 0]
        else:
            benchmark_data = bench_raw
            
        prices = prices.dropna(how='all').ffill().bfill()
        benchmark_data = benchmark_data.reindex(prices.index).ffill().bfill()

    # 5. Reconstruct Timeline
    history_records = []
    trading_days = prices.index.sort_values(ascending=False)
    temp_positions, temp_cash = current_positions.copy(), current_cash
    
    for current_day in trading_days:
        day_ts = pd.Timestamp(current_day)
        
        market_value = 0
        for t, q in temp_positions.items():
            if q != 0 and t in prices.columns:
                val = prices.loc[day_ts, t]
                # Ensure we handle cases where loc might return a series (duplicates)
                price = val.iloc[0] if isinstance(val, pd.Series) else val
                market_value += (q * price)
        
        # Get benchmark scalar
        b_val = benchmark_data.loc[day_ts]
        benchmark_scalar = b_val.iloc[0] if isinstance(b_val, pd.Series) else b_val
        
        history_records.append({
            "Date": current_day.date(),
            "Market Value": float(market_value),
            "Cash": float(temp_cash),
            "Total Portfolio Value": float(market_value + temp_cash),
            "Russell 3000": float(benchmark_scalar)
        })
        
        # Backtrack
        day_tx = hist_df[hist_df['Run Date'].dt.date == current_day.date()]
        for _, tx in day_tx.iterrows():
            ticker, qty = str(tx['Symbol']).strip(), float(tx['Quantity']) if not pd.isna(tx['Quantity']) else 0
            temp_positions[ticker] = temp_positions.get(ticker, 0) - qty
            temp_cash -= clean_currency(tx['Amount ($)'])

    # 6. Display Results
    df = pd.DataFrame(history_records).sort_values("Date")
    # Ensure numeric types for Streamlit charting
    df["Total Portfolio Value"] = pd.to_numeric(df["Total Portfolio Value"])
    df["Russell 3000"] = pd.to_numeric(df["Russell 3000"])
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Create Normalized Data for better comparison
    df['Portfolio (Indexed 100)'] = (df['Total Portfolio Value'] / df['Total Portfolio Value'].iloc[0]) * 100
    df['Russell 3000 (Indexed 100)'] = (df['Russell 3000'] / df['Russell 3000'].iloc[0]) * 100

    st.subheader("Relative Performance (Indexed to 100)")
    st.line_chart(df.set_index("Date")[['Portfolio (Indexed 100)', 'Russell 3000 (Indexed 100)']])
    
    with st.expander("View Absolute Values"):
        st.line_chart(df.set_index("Date")[["Total Portfolio Value", "Russell 3000"]])
        st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "Portfolio_Reconstruction.csv", "text/csv")
