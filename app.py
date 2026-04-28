import streamlit as st
import pandas as pd
import yfinance as yf
import re
from datetime import datetime, timedelta
import statsmodels.api as sm
import pandas_datareader.data as web
import time

# --- 1. Helper Functions ---
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

# --- 2. Streamlit UI Setup ---
st.set_page_config(page_title="Portfolio & Factor Analytics", layout="wide")
st.title("📈 Advanced Portfolio Analytics")

col1, col2 = st.columns(2)
with col1:
    pos_file = st.file_uploader("Upload Positions CSV", type=["csv"])
with col2:
    hist_file = st.file_uploader("Upload History CSV", type=["csv"])

if pos_file and hist_file:
    # --- 3. Data Processing ---
    end_date = extract_date_from_filename(pos_file.name)
    pos_df = pd.read_csv(pos_file, index_col=False)
    hist_df = pd.read_csv(hist_file, skiprows=2, index_col=False)
    
    pos_df = pos_df[pos_df['Symbol'].notna() | pos_df['Account Name'].str.contains('Pending', na=False)].copy()
    
    spaxx_row = pos_df[pos_df['Symbol'] == 'SPAXX**']
    pending_row = pos_df[pos_df['Symbol'] == 'Pending activity']
    if pending_row.empty:
        pending_row = pos_df[pos_df['Account Name'] == 'Pending activity']
    
    current_cash = clean_currency(spaxx_row['Current Value'].iloc[0]) if not spaxx_row.empty else 0
    current_cash += clean_currency(pending_row['Current Value'].iloc[0]) if not pending_row.empty else 0
    
    stocks_df = pos_df[~pos_df['Symbol'].isin(['SPAXX**', 'Pending activity', None]) & 
                       ~pos_df['Account Name'].isin(['Pending activity'])].copy()
    
    current_positions = {}
    total_equity_value = 0
    for _, row in stocks_df.iterrows():
        ticker = str(row['Symbol']).strip()
        if ticker and ticker != 'nan':
            val = clean_currency(row['Current Value'])
            current_positions[ticker] = {'qty': clean_currency(row['Quantity']), 'value': val}
            total_equity_value += val
            
    equity_weights = {t: info['value'] / total_equity_value for t, info in current_positions.items() if total_equity_value > 0}

    # --- 4. Factor Exposure Analysis (6M Daily Regression) ---
    st.subheader("Factor Exposures (Current Equity Weights)")
    with st.spinner("Calculating factor betas..."):
        try:
            analysis_end = min(end_date, datetime.now())
            analysis_start = analysis_end - timedelta(days=180)
            factors = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=analysis_start, end=analysis_end)[0]
            mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start=analysis_start, end=analysis_end)[0]
            factor_df = factors.join(mom).dropna() / 100.0
            
            tickers = list(equity_weights.keys())
            ticker_data = yf.download(tickers, start=analysis_start, end=analysis_end, progress=False)['Close']
            ticker_returns = ticker_data.pct_change().dropna()
            
            port_returns = ticker_returns.mul(pd.Series(equity_weights)).sum(axis=1)
            reg_data = pd.DataFrame({'Portfolio': port_returns}).join(factor_df).dropna()
            reg_data['Excess'] = reg_data['Portfolio'] - reg_data['RF']
            
            X = sm.add_constant(reg_data[['Mkt-RF', 'SMB', 'HML', 'Mom']])
            model = sm.OLS(reg_data['Excess'], X).fit()
            
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Market Beta (β)", f"{model.params['Mkt-RF']:.2f}")
            f2.metric("Size (SMB)", f"{model.params['SMB']:.2f}")
            f3.metric("Value (HML)", f"{model.params['HML']:.2f}")
            f4.metric("Momentum (WML)", f"{model.params['Mom']:.2f}")
        except Exception as e:
            st.info("Factor analysis unavailable for this ticker set or timeframe.")

    # --- 5. Portfolio Reconstruction ---
    st.divider()
    hist_df['Run Date'] = pd.to_datetime(hist_df['Run Date'], errors='coerce')
    hist_df = hist_df[hist_df['Run Date'].notna()].copy()
    hist_df = hist_df.sort_values('Run Date', ascending=False)
    start_date = hist_df['Run Date'].min()
    
    raw_tickers = list(set(list(current_positions.keys()) + hist_df['Symbol'].dropna().unique().tolist()))
    clean_tickers = [t.strip() for t in raw_tickers if t and str(t).strip() not in ['', 'nan', 'SPAXX**', 'Cash']]
    all_req_tickers = sorted(list(set(clean_tickers + ["^RUA", "IWM", "VLUE", "MTUM"])))

    with st.spinner("Reconstructing portfolio history..."):
        try:
            raw_prices = yf.download(all_req_tickers, start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)['Close']
        except Exception:
            time.sleep(2)
            raw_prices = yf.download(all_req_tickers, start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)['Close']
        
        prices = raw_prices.dropna(how='all').ffill().bfill()
        benchmark_data = (prices["^RUA"] if isinstance(prices, pd.DataFrame) else prices).ffill().bfill()

        history_records = []
        trading_days = prices.index.sort_values(ascending=False)
        temp_pos = {t: info['qty'] for t, info in current_positions.items()}
        temp_cash = current_cash
        last_val = 0
        
        for current_day in trading_days:
            day_ts = pd.Timestamp(current_day)
            market_val = 0
            for t, q in temp_pos.items():
                if q != 0 and t in prices.columns:
                    val = prices.loc[day_ts, t]
                    price = val.iloc[0] if isinstance(val, pd.Series) else val
                    if not pd.isna(price): market_val += (q * price)
            
            if market_val == 0 and last_val > 0: market_val = last_val
            else: last_val = market_val
            
            history_records.append({
                "Date": current_day.date(),
                "Market Value": float(market_val),
                "Cash": float(temp_cash),
                "Total Value": float(market_val + temp_cash),
                "Russell 3000": float(prices.loc[day_ts, "^RUA"]) if "^RUA" in prices.columns else 0.0
            })
            
            day_tx = hist_df[hist_df['Run Date'].dt.date == current_day.date()]
            for _, tx in day_tx.iterrows():
                t_sym, t_qty = str(tx['Symbol']).strip(), float(tx['Quantity']) if not pd.isna(tx['Quantity']) else 0
                temp_pos[t_sym] = temp_pos.get(t_sym, 0) - t_qty
                temp_cash -= clean_currency(tx['Amount ($)'])

    # --- 6. Final Metrics & Visuals ---
    df = pd.DataFrame(history_records).sort_values("Date")
    df["Total Value"] = pd.to_numeric(df["Total Value"])
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Turnover Metrics
    buys = hist_df[hist_df['Action'].str.contains('YOU BOUGHT', na=False)]['Amount ($)'].apply(clean_currency).abs().sum()
    sells = hist_df[hist_df['Action'].str.contains('YOU SOLD', na=False)]['Amount ($)'].apply(clean_currency).abs().sum()
    avg_aum = df['Total Value'].mean()
    turnover = (min(buys, sells) / avg_aum) if avg_aum > 0 else 0
    annualized_turnover = turnover * (365 / max((end_date - start_date).days, 1))

    st.subheader("Performance & Turnover Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Average AUM", f"${avg_aum:,.2f}")
    m2.metric("Total Buys/Sells", f"${(buys + sells):,.2f}")
    m3.metric("Annualized Turnover", f"{annualized_turnover:.2%}")

    df['Portfolio (Indexed 100)'] = (df['Total Value'] / df['Total Value'].iloc[0]) * 100
    df['Russell 3000 (Indexed 100)'] = (df['Russell 3000'] / df['Russell 3000'].iloc[0]) * 100
    st.line_chart(df.set_index("Date")[['Portfolio (Indexed 100)', 'Russell 3000 (Indexed 100)']])
    
    with st.expander("Raw Data & Download"):
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "Analysis.csv", "text/csv")
