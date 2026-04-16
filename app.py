import streamlit as st
import pandas as pd
import yfinance as yf
import re
from datetime import datetime
import io

# --- Robust Data Cleaning ---
def clean_currency(value):
    """Cleans currency strings: handles $, commas, and negatives in parentheses."""
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
st.title("📈 Portfolio Historical Reconstruction")
st.markdown("""
This tool reconstructs your portfolio value day-by-day by backtracking from your current positions 
using your transaction history and Yahoo Finance market data.
""")

col1, col2 = st.columns(2)
with col1:
    pos_file = st.file_uploader("Upload Positions CSV", type=["csv"])
with col2:
    hist_file = st.file_uploader("Upload History CSV", type=["csv"])

if pos_file and hist_file:
    # 1. Load Data
    # index_col=False is critical to prevent column shifting
    end_date = extract_date_from_filename(pos_file.name)
    pos_df = pd.read_csv(pos_file, index_col=False)
    hist_df = pd.read_csv(hist_file, skiprows=2, index_col=False)
    
    # 2. Extract Current State (End Date)
    # Filter rows to keep only valid positions or pending activity
    pos_df = pos_df[pos_df['Symbol'].notna() | pos_df['Account Name'].str.contains('Pending', na=False)].copy()
    
    # Identify Cash components (SPAXX + Pending)
    spaxx_row = pos_df[pos_df['Symbol'] == 'SPAXX**']
    pending_row = pos_df[pos_df['Symbol'] == 'Pending activity']
    if pending_row.empty:
        pending_row = pos_df[pos_df['Account Name'] == 'Pending activity']
    
    spaxx_val = clean_currency(spaxx_row['Current Value'].iloc[0]) if not spaxx_row.empty else 0
    pending_val = clean_currency(pending_row['Current Value'].iloc[0]) if not pending_row.empty else 0
    current_cash = spaxx_val + pending_val
    
    # Identify stock positions
    stocks_df = pos_df[~pos_df['Symbol'].isin(['SPAXX**', 'Pending activity', None]) & 
                       ~pos_df['Account Name'].isin(['Pending activity'])].copy()
    
    current_positions = {}
    for _, row in stocks_df.iterrows():
        ticker = str(row['Symbol']).strip()
        if ticker and ticker != 'nan':
            current_positions[ticker] = clean_currency(row['Quantity'])

    # 3. Process History
    hist_df['Run Date'] = pd.to_datetime(hist_df['Run Date'], errors='coerce')
    hist_df = hist_df[hist_df['Run Date'].notna()].copy()
    hist_df = hist_df.sort_values('Run Date', ascending=False)
    start_date = hist_df['Run Date'].min()
    
    # Collect all tickers that ever existed in the portfolio
    all_tickers = sorted(list(set(list(current_positions.keys()) + hist_df['Symbol'].dropna().unique().tolist())))
    all_tickers = [t for t in all_tickers if t not in ['SPAXX**', 'SPAXX', 'Cash', 'Pending activity']]

    # 4. Fetch Historical Prices
    with st.spinner(f"Fetching market data for {len(all_tickers)} tickers..."):
        price_data = yf.download(all_tickers, start=start_date, end=end_date + pd.Timedelta(days=1))['Close']
        
        # DROPPING HOLIDAYS: Remove rows where all tickers are NaN (Market was closed)
        price_data = price_data.dropna(how='all')
        
        # Fill missing data for individual tickers (stale prices)
        price_data = price_data.ffill().bfill()

    # 5. Reconstruct Timeline (Backwards)
    history_records = []
    
    # Use the price_data index to ensure we only iterate on valid trading days
    trading_days = price_data.index.sort_values(ascending=False)
    
    temp_positions = current_positions.copy()
    temp_cash = current_cash
    
    for current_day in trading_days:
        day_ts = pd.Timestamp(current_day)
        
        # A. Calculate total value for the end of THIS trading day
        market_value = 0
        for t, q in temp_positions.items():
            if q != 0 and t in price_data.columns:
                try:
                    price = price_data.loc[day_ts, t]
                    market_value += (q * price)
                except KeyError:
                    pass
        
        history_records.append({
            "Date": current_day.date(),
            "Market Value": round(market_value, 2),
            "Cash": round(temp_cash, 2),
            "Total Portfolio Value": round(market_value + temp_cash, 2)
        })
        
        # B. BACKTRACK: Reverse transactions that occurred ON this day 
        # to find the state for the day BEFORE.
        day_tx = hist_df[hist_df['Run Date'].dt.date == current_day.date()]
        for _, tx in day_tx.iterrows():
            tx_ticker = str(tx['Symbol']).strip()
            tx_qty = float(tx['Quantity']) if not pd.isna(tx['Quantity']) else 0
            tx_amt = clean_currency(tx['Amount ($)'])
            
            # If we bought today, we had fewer shares yesterday
            if tx_ticker in temp_positions:
                temp_positions[tx_ticker] -= tx_qty
            else:
                temp_positions[tx_ticker] = -tx_qty
            
            # Reverse cash impact
            temp_cash -= tx_amt

    # 6. Display Results
    reconstructed_df = pd.DataFrame(history_records).sort_values("Date")
    
    st.subheader("Performance Summary")
    st.line_chart(reconstructed_df.set_index("Date")["Total Portfolio Value"])
    
    st.dataframe(reconstructed_df, use_container_width=True)
    
    # Excel/CSV Export
    csv = reconstructed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Reconstruction (CSV)",
        data=csv,
        file_name=f"Portfolio_History_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )
    
    st.success("Reconstruction complete. All holiday 'zeroes' have been filtered out.")
