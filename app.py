import streamlit as st
import pandas as pd
import yfinance as yf
import re
from datetime import datetime

# --- Helper Functions ---
def clean_currency(value):
    """Cleans currency and quantity strings: handles $, commas, and negatives in parentheses."""
    if pd.isna(value) or str(value).strip() in ['', '--']: 
        return 0.0
    val_str = str(value).strip()
    is_negative = False
    # Handle Fidelity's (10.00) notation for negative numbers
    if val_str.startswith('(') and val_str.endswith(')'):
        is_negative = True
        val_str = val_str[1:-1]
    # Remove everything except digits, dots, and minus signs
    cleaned = re.sub(r'[^\d.-]', '', val_str)
    try:
        res = float(cleaned)
        return -res if is_negative else res
    except ValueError:
        return 0.0

def extract_date_from_filename(filename):
    """Extracts date from 'Portfolio_Positions_Apr-15-2026' format."""
    match = re.search(r'([A-Z][a-z]{2}-\d{2}-\d{4})', filename)
    if match:
        return datetime.strptime(match.group(1), '%b-%d-%Y')
    return datetime.now()

# --- Streamlit UI ---
st.set_page_config(page_title="Portfolio Reconstructor", layout="wide")
st.title("📈 Portfolio vs. Russell 3000 Benchmark")
st.markdown("""
This tool reconstructs your daily portfolio value by backtracking transactions from your current positions.
It also pulls **^RUA** (Russell 3000) data for comparison.
""")

col1, col2 = st.columns(2)
with col1:
    pos_file = st.file_uploader("Upload Positions CSV", type=["csv"])
with col2:
    hist_file = st.file_uploader("Upload History CSV", type=["csv"])

if pos_file and hist_file:
    # 1. Load Data
    # index_col=False prevents columns from shifting if there are trailing commas
    end_date = extract_date_from_filename(pos_file.name)
    pos_df = pd.read_csv(pos_file, index_col=False)
    hist_df = pd.read_csv(hist_file, skiprows=2, index_col=False)
    
    # 2. Extract End-Date State
    # Keep only rows with symbols or 'Pending activity'
    pos_df = pos_df[pos_df['Symbol'].notna() | pos_df['Account Name'].str.contains('Pending', na=False)].copy()
    
    # Identify Cash: SPAXX** + Pending Activity
    spaxx_row = pos_df[pos_df['Symbol'] == 'SPAXX**']
    pending_row = pos_df[pos_df['Symbol'] == 'Pending activity']
    if pending_row.empty:
        pending_row = pos_df[pos_df['Account Name'] == 'Pending activity']
    
    current_cash = clean_currency(spaxx_row['Current Value'].iloc[0]) if not spaxx_row.empty else 0
    current_cash += clean_currency(pending_row['Current Value'].iloc[0]) if not pending_row.empty else 0
    
    # Identify stock positions
    stocks_df = pos_df[~pos_df['Symbol'].isin(['SPAXX**', 'Pending activity', None]) & 
                       ~pos_df['Account Name'].isin(['Pending activity'])].copy()
    
    current_positions = {
        str(row['Symbol']).strip(): clean_currency(row['Quantity']) 
        for _, row in stocks_df.iterrows() if str(row['Symbol']).strip() != 'nan'
    }

    # 3. Process Transaction History
    hist_df['Run Date'] = pd.to_datetime(hist_df['Run Date'], errors='coerce')
    hist_df = hist_df[hist_df['Run Date'].notna()].copy()
    hist_df = hist_df.sort_values('Run Date', ascending=False)
    start_date = hist_df['Run Date'].min()
    
    # Collect all unique tickers encountered in history or current positions
    all_tickers = sorted(list(set(list(current_positions.keys()) + hist_df['Symbol'].dropna().unique().tolist())))
    # Filter out non-equity symbols
    all_tickers = [t for t in all_tickers if t not in ['SPAXX**', 'SPAXX', 'Cash', 'Pending activity']]

    # 4. Fetch Market Data (Portfolio + Russell 3000)
    with st.spinner("Fetching market data..."):
        # Download Portfolio Prices
        prices = yf.download(all_tickers, start=start_date, end=end_date + pd.Timedelta(days=1))['Close']
        # Download Russell 3000 Index
        benchmark = yf.download("^RUA", start=start_date, end=end_date + pd.Timedelta(days=1))['Close']
        
        # HOLIDAY FIX: Drop days where the market was closed (no price data)
        prices = prices.dropna(how='all')
        
        # Ensure individual ticker gaps are filled
        prices = prices.ffill().bfill()
        
        # Align benchmark dates with portfolio trading days
        benchmark = benchmark.reindex(prices.index).ffill().bfill()

    # 5. Backtracking Algorithm
    history_records = []
    trading_days = prices.index.sort_values(ascending=False)
    
    temp_positions = current_positions.copy()
    temp_cash = current_cash
    
    for current_day in trading_days:
        day_ts = pd.Timestamp(current_day)
        
        # Calculate market value for the end of THIS day
        market_value = 0
        for t, q in temp_positions.items():
            if q != 0 and t in prices.columns:
                try:
                    price = prices.loc[day_ts, t]
                    market_value += (q * price)
                except KeyError:
                    pass
        
        # Log values
        history_records.append({
            "Date": current_day.date(),
            "Market Value": round(market_value, 2),
            "Cash": round(temp_cash, 2),
            "Total Portfolio Value": round(market_value + temp_cash, 2),
            "Russell 3000": round(benchmark.loc[day_ts], 2) if day_ts in benchmark.index else None
        })
        
        # REVERSE transactions from this day to get state for the day BEFORE
        day_tx = hist_df[hist_df['Run Date'].dt.date == current_day.date()]
        for _, tx in day_tx.iterrows():
            ticker = str(tx['Symbol']).strip()
            qty = float(tx['Quantity']) if not pd.isna(tx['Quantity']) else 0
            amt = clean_currency(tx['Amount ($)'])
            
            # Reverse Position: If we bought (+), subtract from current balance
            temp_positions[ticker] = temp_positions.get(ticker, 0) - qty
            # Reverse Cash: If we spent money today (amt was -), add it back
            temp_cash -= amt

    # 6. Results Display
    reconstructed_df = pd.DataFrame(history_records).sort_values("Date")
    
    st.subheader("Performance Tracking")
    # Plotting Total Portfolio Value and Russell 3000
    st.line_chart(reconstructed_df.set_index("Date")[["Total Portfolio Value", "Russell 3000"]])
    
    st.subheader("Daily Data Table")
    st.dataframe(reconstructed_df, use_container_width=True)
    
    # Download Options
    csv = reconstructed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Reconstruction as CSV",
        data=csv,
        file_name=f"Portfolio_Analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )
