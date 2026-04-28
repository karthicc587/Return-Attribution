import streamlit as st
import pandas as pd
import yfinance as yf
import re
from datetime import datetime
import time

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
st.set_page_config(page_title="Portfolio Analysis Tool", layout="wide")
st.title("📈 Portfolio Performance & Turnover Analysis")

col1, col2 = st.columns(2)
with col1:
    pos_file = st.file_uploader("Upload Positions CSV", type=["csv"])
with col2:
    hist_file = st.file_uploader("Upload History CSV", type=["csv"])

# Factor ETF Proxies for Excel Regression
FACTOR_PROXIES = ["IWM", "VLUE", "MTUM"]

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
    
    # Sanitize Tickers
    raw_tickers = list(set(list(current_positions.keys()) + hist_df['Symbol'].dropna().unique().tolist()))
    clean_tickers = [t.strip() for t in raw_tickers if t and str(t).strip() not in ['', 'nan', 'SPAXX**', 'Cash']]
    all_req_tickers = sorted(list(set(clean_tickers + ["^RUA"] + FACTOR_PROXIES)))

    # 4. Fetch Historical Prices
    with st.spinner("Fetching market data..."):
        try:
            raw_prices = yf.download(all_req_tickers, start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)['Close']
        except Exception:
            time.sleep(2)
            raw_prices = yf.download(all_req_tickers, start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)['Close']
        
        prices = raw_prices.dropna(how='all').ffill().bfill()

    # 5. Reconstruct Timeline
    history_records = []
    trading_days = prices.index.sort_values(ascending=False)
    temp_pos = current_positions.copy()
    temp_cash = current_cash
    last_valid_market_val = 0
    
    for current_day in trading_days:
        day_ts = pd.Timestamp(current_day)
        market_val = 0
        for t, q in temp_pos.items():
            if q != 0 and t in prices.columns:
                val = prices.loc[day_ts, t]
                price = val.iloc[0] if isinstance(val, pd.Series) else val
                if not pd.isna(price):
                    market_val += (q * price)
        
        # Stability Guard
        if market_val == 0 and last_valid_market_val > 0:
            market_val = last_valid_market_val
        else:
            last_valid_market_val = market_val
        
        record = {
            "Date": current_day.date(),
            "Market Value": float(market_val),
            "Cash": float(temp_cash),
            "Total Portfolio Value": float(market_val + temp_cash),
            "Russell 3000": float(prices.loc[day_ts, "^RUA"]) if "^RUA" in prices.columns else 0.0
        }
        # Add Factor Proxies for Excel usage
        for ticker in FACTOR_PROXIES:
            if ticker in prices.columns:
                record[ticker] = float(prices.loc[day_ts, ticker])
        
        history_records.append(record)
        
        # Backtrack
        day_tx = hist_df[hist_df['Run Date'].dt.date == current_day.date()]
        for _, tx in day_tx.iterrows():
            ticker, qty = str(tx['Symbol']).strip(), float(tx['Quantity']) if not pd.isna(tx['Quantity']) else 0
            temp_pos[ticker] = temp_pos.get(ticker, 0) - qty
            temp_cash -= clean_currency(tx['Amount ($)'])

    # 6. Metrics & Display
    df = pd.DataFrame(history_records).sort_values("Date")
    df["Total Portfolio Value"] = pd.to_numeric(df["Total Portfolio Value"])
    df["Date"] = pd.to_datetime(df["Date"])

    # Turnover Logic: Use MIN of buys/sells
    buys = hist_df[hist_df['Action'].str.contains('YOU BOUGHT', na=False)]['Amount ($)'].apply(clean_currency).abs().sum()
    sells = hist_df[hist_df['Action'].str.contains('YOU SOLD', na=False)]['Amount ($)'].apply(clean_currency).abs().sum()
    avg_value = df['Total Portfolio Value'].mean()
    
    turnover_ratio = (min(buys, sells) / avg_value) if avg_value > 0 else 0
    days_in_period = (end_date - start_date).days
    annualized_turnover = turnover_ratio * (365 / max(days_in_period, 1))

    st.subheader("Key Portfolio Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average AUM", f"${avg_value:,.2f}")
    m2.metric("Total Buys", f"${buys:,.2f}")
    m3.metric("Total Sells", f"${sells:,.2f}")
    m4.metric("Turnover Ratio", f"{turnover_ratio:.2%}")

    # Charts
    df['Portfolio (Indexed 100)'] = (df['Total Portfolio Value'] / df['Total Portfolio Value'].iloc[0]) * 100
    df['Russell 3000 (Indexed 100)'] = (df['Russell 3000'] / df['Russell 3000'].iloc[0]) * 100

    st.subheader("Relative Performance (Indexed to 100)")
    st.line_chart(df.set_index("Date")[['Portfolio (Indexed 100)', 'Russell 3000 (Indexed 100)']])
    
    with st.expander("Detailed History & Factor Data (Excel Export)"):
        st.write("Factor Proxies: IWM (Size), VLUE (Value), MTUM (Momentum)")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data (CSV)", csv, "Portfolio_Analysis_Export.csv", "text/csv")
