import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import re
import io

st.set_page_config(page_title="Portfolio Reconstructor", layout="wide")

st.title("📊 Portfolio Value Reconstructor")
st.markdown("""
Upload your Fidelity **Positions** and **History** files to see your daily portfolio value over time.
""")

# --- Helper Functions ---

def clean_currency(column):
    """Removes symbols like $, %, and commas, converting to float."""
    if column.dtype == 'object':
        return pd.to_numeric(column.str.replace(r'[$\s,%\+]', '', regex=True), errors='coerce')
    return column

def get_end_date_from_filename(filename):
    """Extracts date from 'Portfolio_Positions_MMM-DD-YYYY.csv'"""
    match = re.search(r'([A-Z][a-z]{2}-\d{1,2}-\d{4})', filename)
    if match:
        return pd.to_datetime(match.group(1), format='%b-%d-%Y')
    return datetime.now()

# --- File Uploads ---
col1, col2 = st.columns(2)
with col1:
    pos_file = st.file_uploader("Upload Portfolio Positions CSV", type=["csv"])
with col2:
    hist_file = st.file_uploader("Upload History CSV", type=["csv"])

if pos_file and hist_file:
    # 1. Parse Dates
    end_date = get_end_date_from_filename(pos_file.name)
    
    # 2. Load and Clean Positions
    # Fidelity positions usually have footer rows; we filter for rows with Account Numbers
    pos_df = pd.read_csv(pos_file)
    pos_df = pos_df[pos_df['Account Number'].notna()].copy()
    
    # Clean numeric columns
    for col in ['Quantity', 'Current Value', 'Average Cost Basis']:
        if col in pos_df.columns:
            pos_df[col] = clean_currency(pos_df[col])

    # Extract Cash and Positions
    # Cash is SPAXX** plus any "Pending activity"
    cash_row = pos_df[pos_df['Symbol'].str.contains('SPAXX', na=False)]
    pending_row = pos_df[pos_df['Symbol'].str.contains('Pending activity', na=False)]
    
    initial_cash = cash_row['Current Value'].sum() + pending_row['Current Value'].sum()
    
    # Exclude non-stock symbols for ticker processing
    stock_positions = pos_df[~pos_df['Symbol'].str.contains('SPAXX|Pending activity', na=False, case=False)]
    stock_positions = stock_positions[stock_positions['Quantity'] > 0]
    
    final_state = stock_positions.set_index('Symbol')['Quantity'].to_dict()

    # 3. Load and Clean History
    # History starts at row 3 (skip 2)
    hist_df = pd.read_csv(hist_file, skiprows=2)
    hist_df = hist_df[hist_df['Run Date'].notna()].copy()
    hist_df['Run Date'] = pd.to_datetime(hist_df['Run Date'])
    hist_df['Amount ($)'] = clean_currency(hist_df['Amount ($)'])
    hist_df['Quantity'] = clean_currency(hist_df['Quantity'])
    
    start_date = hist_df['Run Date'].min()
    
    st.info(f"Reconstructing from **{start_date.date()}** to **{end_date.date()}**")

    # 4. Fetch Historical Prices
    unique_tickers = list(set(list(final_state.keys()) + hist_df['Symbol'].dropna().unique().tolist()))
    unique_tickers = [t for t in unique_tickers if t and 'SPAXX' not in t and 'Pending' not in t]
    
    @st.cache_data
    def fetch_prices(tickers, start, end):
        # Buffer the end date slightly to ensure we get the latest price
        data = yf.download(tickers, start=start, end=end + timedelta(days=2))['Close']
        return data

    with st.spinner("Fetching historical market data..."):
        price_data = fetch_prices(unique_tickers, start_date, end_date)

    # 5. Reconstruct Daily State
    # We work backwards from the final state (the positions file)
    current_pos = final_state.copy()
    current_cash = initial_cash
    
    daily_records = []
    
    # All trading days in range (reversed)
    trading_days = price_data.index[price_data.index <= end_date].sort_values(ascending=False)
    
    for dt in trading_days:
        # 1. Record value at END of this day
        daily_val = 0
        for ticker, qty in current_pos.items():
            if ticker in price_data.columns:
                price = price_data.loc[dt, ticker]
                if pd.isna(price): # Handle weekend/holiday gaps if any
                    price = price_data.loc[:dt, ticker].iloc[-1]
                daily_val += (qty * price)
        
        daily_records.append({
            "Date": dt.date(),
            "Equity Value": round(daily_val, 2),
            "Cash": round(current_cash, 2),
            "Total Portfolio Value": round(daily_val + current_cash, 2)
        })
        
        # 2. Reverse transactions that happened ON this day to find state at BEGINNING of day
        todays_trans = hist_df[hist_df['Run Date'].dt.date == dt.date()]
        
        for _, row in todays_trans.iterrows():
            symbol = row['Symbol']
            qty = row['Quantity']
            amt = row['Amount ($)'] # Amount is negative for buys, positive for sells
            
            # Reverse Cash impact: If we spent money (negative amt), we add it back to find previous state
            current_cash -= amt
            
            # Reverse Share impact: If we bought shares (positive qty), we subtract them
            if pd.notna(symbol) and symbol in unique_tickers:
                current_pos[symbol] = current_pos.get(symbol, 0) - qty

    # 6. Display and Download
    history_df = pd.DataFrame(daily_records).sort_values('Date')
    
    st.dataframe(history_df, use_container_width=True)

    # Download Buttons
    col_dl1, col_dl2 = st.columns(2)
    
    csv = history_df.to_csv(index=False).encode('utf-8')
    col_dl1.download_button("Download as CSV", data=csv, file_name="portfolio_history.csv", mime="text/csv")
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        history_df.to_excel(writer, index=False, sheet_name='History')
    col_dl2.download_button("Download as Excel", data=buffer.getvalue(), file_name="portfolio_history.xlsx", mime="application/vnd.ms-excel")

else:
    st.warning("Please upload both files to begin.")
