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
    match = re.search(r'([A-Z][a-z]{2}-\d{2}-\d{4})', filename)
    if match:
        return datetime.strptime(match.group(1), '%b-%d-%Y')
    return datetime.now()

# --- Streamlit UI ---
st.set_page_config(page_title="Portfolio Reconstructor", layout="wide")
st.title("📈 Portfolio Historical Reconstruction")
st.markdown("Reconstruct your daily portfolio value by backtracking from current positions.")

col1, col2 = st.columns(2)
with col1:
    pos_file = st.file_uploader("Upload Positions CSV (Portfolio_Positions_...)", type=["csv"])
with col2:
    hist_file = st.file_uploader("Upload History CSV (History_for_...)", type=["csv"])

if pos_file and hist_file:
    # 1. Load Data with index_col=False to prevent column shifting
    end_date = extract_date_from_filename(pos_file.name)
    pos_df = pd.read_csv(pos_file, index_col=False)
    hist_df = pd.read_csv(hist_file, skiprows=2, index_col=False)
    
    # 2. Extract Current State (End Date)
    # Filter out footer rows
    pos_df = pos_df[pos_df['Symbol'].notna() | pos_df['Account Name'].str.contains('Pending', na=False)].copy()
    
    # Identify Cash components (SPAXX + Pending)
    spaxx_row = pos_df[pos_df['Symbol'] == 'SPAXX**']
    # Check both Symbol and Account Name for 'Pending activity' due to potential shifts
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
            # Use clean_currency on quantity too, as it might have commas
            current_positions[ticker] = clean_currency(row['Quantity'])

    # 3. Process History
    hist_df['Run Date'] = pd.to_datetime(hist_df['Run Date'], errors='coerce')
    hist_df = hist_df[hist_df['Run Date'].notna()].copy()
    hist_df = hist_df.sort_values('Run Date', ascending=False)
    start_date = hist_df['Run Date'].min()
    
    all_tickers = sorted(list(set(list(current_positions.keys()) + hist_df['Symbol'].dropna().unique().tolist())))
    # Remove cash-like symbols from yfinance list
    all_tickers = [t for t in all_tickers if t not in ['SPAXX**', 'SPAXX', 'Cash']]

    # 4. Fetch Historical Prices
    with st.spinner(f"Fetching historical prices for {len(all_tickers)} tickers..."):
        # We fetch from start_date to end_date
        price_data = yf.download(all_tickers, start=start_date, end=end_date + pd.Timedelta(days=1))['Close']
        price_data = price_data.ffill().bfill() # Handle weekends/holidays

    # 5. Reconstruct Timeline (Backwards)
    history_records = []
    # Get all business days in the range, reversed (Newest to Oldest)
    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')[::-1]
    
    temp_positions = current_positions.copy()
    temp_cash = current_cash
    
    for current_day in trading_days:
        day_ts = pd.Timestamp(current_day)
        
        # A. Calculate total value for the end of THIS day
        market_value = 0
        for t, q in temp_positions.items():
            if q != 0 and t in price_data.columns:
                # Get closest available price for that day
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
            
            # If we bought today (qty > 0), we had FEWER shares yesterday
            if tx_ticker in temp_positions:
                temp_positions[tx_ticker] -= tx_qty
            else:
                temp_positions[tx_ticker] = -tx_qty
            
            # If cash left today (Amount is negative for buy), we had MORE cash yesterday
            temp_cash -= tx_amt

    # 6. Display Results
    reconstructed_df = pd.DataFrame(history_records).sort_values("Date")
    
    st.subheader("Total Portfolio Value Over Time")
    st.line_chart(reconstructed_df.set_index("Date")["Total Portfolio Value"])
    
    st.subheader("Daily Breakdown")
    st.dataframe(reconstructed_df, use_container_width=True)
    
    # Download Button
    csv = reconstructed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Reconstruction (CSV)",
        data=csv,
        file_name=f"Portfolio_Reconstruction_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )
