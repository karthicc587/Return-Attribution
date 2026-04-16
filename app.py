import streamlit as st
import pandas as pd
import yfinance as yf
import re
from datetime import datetime
import io

# --- Helper Functions ---
def clean_currency(value):
    if pd.isna(value) or str(value).strip() in ['', '--']: 
        return 0.0
    return float(re.sub(r'[^\d.-]', '', str(value)))

def extract_date_from_filename(filename):
    match = re.search(r'([A-Z][a-z]{2}-\d{2}-\d{4})', filename)
    if match:
        return datetime.strptime(match.group(1), '%b-%d-%Y')
    return datetime.now()

# --- Streamlit UI ---
st.set_page_config(page_title="Portfolio Reconstructor", layout="wide")
st.title("📈 Portfolio Historical Reconstruction")
st.markdown("Upload your Fidelity Positions and History files to see your portfolio's value over time.")

col1, col2 = st.columns(2)
with col1:
    pos_file = st.file_uploader("Upload Positions CSV", type=["csv"])
with col2:
    hist_file = st.file_uploader("Upload History CSV", type=["csv"])

if pos_file and hist_file:
    # 1. Load Data
    end_date = extract_date_from_filename(pos_file.name)
    pos_df = pd.read_csv(pos_file)
    # The history file typically has two empty rows at the top
    hist_df = pd.read_csv(hist_file, skiprows=2)
    
    # 2. Extract Current State (End Date)
    # Clean symbols and quantities
    pos_df = pos_df[pos_df['Symbol'].notna()].copy()
    
    # Identify Cash components
    spaxx_val = clean_currency(pos_df[pos_df['Symbol'] == 'SPAXX**']['Current Value'].iloc[0]) if 'SPAXX**' in pos_df['Symbol'].values else 0
    pending_val = clean_currency(pos_df[pos_df['Symbol'] == 'Pending activity']['Current Value'].iloc[0]) if 'Pending activity' in pos_df['Symbol'].values else 0
    current_cash = spaxx_val + pending_val
    
    # Identify stock positions
    stocks_df = pos_df[~pos_df['Symbol'].isin(['SPAXX**', 'Pending activity'])].copy()
    current_positions = {}
    for _, row in stocks_df.iterrows():
        ticker = str(row['Symbol']).strip()
        qty = float(str(row['Quantity']).replace(',', ''))
        current_positions[ticker] = qty

    # 3. Process History
    hist_df['Run Date'] = pd.to_datetime(hist_df['Run Date'])
    hist_df = hist_df.sort_values('Run Date', ascending=False)
    start_date = hist_df['Run Date'].min()
    
    all_tickers = list(current_positions.keys()) + hist_df['Symbol'].dropna().unique().tolist()
    all_tickers = sorted(list(set([t for t in all_tickers if t and t != 'nan'])))

    # 4. Fetch Historical Prices
    with st.spinner(f"Fetching historical prices for {len(all_tickers)} tickers..."):
        # Download prices from start to end
        price_data = yf.download(all_tickers, start=start_date, end=end_date + pd.Timedelta(days=1))['Close']
        # Fill missing data (holidays/weekends)
        price_data = price_data.fillna(method='ffill').fillna(method='bfill')

    # 5. Reconstruct Timeline (Backwards)
    history_records = []
    
    # Get all trading days in range
    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')[::-1]
    
    temp_positions = current_positions.copy()
    temp_cash = current_cash
    
    for current_day in trading_days:
        # Calculate market value for the end of THIS day
        market_value = 0
        for t, q in temp_positions.items():
            if t in price_data.columns and q != 0:
                price = price_data.loc[current_day, t] if current_day in price_data.index else 0
                market_value += (q * price)
        
        total_portfolio_value = market_value + temp_cash
        history_records.append({
            "Date": current_day.date(),
            "Market Value": round(market_value, 2),
            "Cash": round(temp_cash, 2),
            "Total Value": round(total_portfolio_value, 2)
        })
        
        # Now, adjust positions/cash for transactions that happened ON this day 
        # to get the state for the day BEFORE.
        day_tx = hist_df[hist_df['Run Date'].dt.date == current_day.date()]
        for _, tx in day_tx.iterrows():
            tx_ticker = str(tx['Symbol']).strip()
            tx_qty = float(tx['Quantity']) if not pd.isna(tx['Quantity']) else 0
            tx_amt = clean_currency(tx['Amount ($)'])
            
            # If we BOUGHT today, we had FEWER shares yesterday
            if tx_ticker in temp_positions:
                temp_positions[tx_ticker] -= tx_qty
            else:
                temp_positions[tx_ticker] = -tx_qty
            
            # If cash left today (Amount is negative for buy), we had MORE cash yesterday
            temp_cash -= tx_amt

    # 6. Display Results
    reconstructed_df = pd.DataFrame(history_records).sort_values("Date")
    
    st.subheader("Historical Portfolio Value")
    st.line_chart(reconstructed_df.set_index("Date")["Total Value"])
    
    st.dataframe(reconstructed_df, use_container_width=True)
    
    # Download Button
    csv = reconstructed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Reconstruction as CSV",
        data=csv,
        file_name=f"Portfolio_Reconstruction_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )
