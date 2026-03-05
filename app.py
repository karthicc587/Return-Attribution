"""
Portfolio Analyzer — Streamlit App
------------------------------------
Upload your two Fidelity CSV exports to analyze portfolio performance.

To run locally:
    pip install streamlit yfinance matplotlib pandas
    streamlit run app.py
"""

import csv
import io
import re
from copy import deepcopy
from datetime import date, datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io as _io
import zipfile
import requests
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────

BG    = "#f9fafb"
GRAY  = "#374151"
LIGHT = "#d1d5db"
DARK  = "#111827"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.titleweight": "bold",
})

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_money(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(re.sub(r"[$,+]", "", value.strip()))
    except ValueError:
        return 0.0


def classify_action(action_text: str):
    t = action_text.upper()
    if "REINVESTMENT" in t: return "REINVESTMENT"
    if "DIVIDEND"     in t: return "DIVIDEND"
    if "YOU BOUGHT"   in t: return "BUY"
    if "YOU SOLD"     in t: return "SELL"
    return None


def detect_file_type(content: str) -> str:
    for line in content.lstrip("\ufeff").splitlines():
        if not line.strip():
            continue
        if "Run Date" in line or "Action" in line:
            return "orders"
        if "Symbol" in line and "Quantity" in line:
            return "positions"
        break
    return "positions"


def prev_friday(d: date) -> date:
    days_back = (d.weekday() - 4) % 7
    return d - timedelta(days=days_back)


# ─────────────────────────────────────────────────────────────────────────────
# File parsers
# ─────────────────────────────────────────────────────────────────────────────

def read_positions(content: str) -> dict:
    positions = {}
    reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")))
    for row in reader:
        symbol = (row.get("Symbol") or "").strip()
        if not symbol:
            continue
        low = symbol.lower()
        if low.startswith("the data") or low.startswith("brokerage") or low.startswith("date"):
            continue
        qty_raw = (row.get("Quantity") or "").strip()
        if symbol == "SPAXX**":
            positions["CASH"] = clean_money(row.get("Current Value") or "")
        elif low == "pending activity":
            positions["PENDING"] = clean_money(row.get("Current Value") or "")
        else:
            try:
                positions[symbol] = float(qty_raw) if qty_raw else 0.0
            except ValueError:
                positions[symbol] = 0.0
    return positions


def read_orders(content: str):
    orders = []
    clean_lines = [ln for ln in content.splitlines(keepends=True) if ln.strip()]
    reader = csv.DictReader(io.StringIO("".join(clean_lines)))
    for row in reader:
        run_date_str = (row.get("Run Date")   or "").strip()
        action_text  = (row.get("Action")     or "").strip()
        symbol       = (row.get("Symbol")     or "").strip()
        qty_raw      = (row.get("Quantity")   or "").strip()
        amount_raw   = (row.get("Amount ($)") or "").strip()
        if not re.match(r"\d{2}/\d{2}/\d{4}", run_date_str):
            continue
        order_type = classify_action(action_text)
        if order_type is None:
            continue
        try:
            order_date = datetime.strptime(run_date_str, "%m/%d/%Y").date()
        except ValueError:
            continue
        try:
            quantity = float(qty_raw) if qty_raw else 0.0
        except ValueError:
            quantity = 0.0
        orders.append({
            "date":     order_date,
            "type":     order_type,
            "ticker":   symbol,
            "quantity": quantity,
            "amount":   clean_money(amount_raw),
        })
    orders.sort(key=lambda o: o["date"])
    earliest = orders[0]["date"] if orders else None
    return earliest, orders


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def build_starting_portfolio(current_positions: dict, orders: list) -> dict:
    portfolio = {}
    for sym, qty in current_positions.items():
        if sym in ("CASH", "PENDING"):
            continue
        portfolio[sym] = qty
    portfolio["CASH"] = current_positions.get("CASH", 0.0)
    for o in orders:
        sym = o["ticker"]
        if sym == "SPAXX":
            continue
        portfolio[sym] = portfolio.get(sym, 0.0) - o["quantity"]
        portfolio["CASH"] -= o["amount"]
    return {k: v for k, v in portfolio.items() if abs(v) > 1e-6}


def apply_orders_up_to(base_portfolio: dict, orders: list, cutoff: date) -> dict:
    portfolio = deepcopy(base_portfolio)
    for o in orders:
        if o["date"] > cutoff:
            break
        sym = o["ticker"]
        if sym == "SPAXX":
            continue
        portfolio[sym] = portfolio.get(sym, 0.0) + o["quantity"]
        portfolio["CASH"] = portfolio.get("CASH", 0.0) + o["amount"]
    return {k: v for k, v in portfolio.items() if abs(v) > 1e-6}


# ─────────────────────────────────────────────────────────────────────────────
# Price fetching (cached so re-runs don't re-download)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: tuple, start_str: str, end_str: str) -> dict:
    """Returns { ticker: { date: close_price } }"""
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end   = datetime.strptime(end_str,   "%Y-%m-%d").date()

    ticker_list = list(tickers)
    if not ticker_list:
        return {}

    raw = yf.download(
        ticker_list,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    prices = {t: {} for t in ticker_list}
    if raw.empty:
        return prices

    close = raw["Close"] if len(ticker_list) > 1 else raw[["Close"]]
    if len(ticker_list) == 1:
        close.columns = ticker_list

    for idx_dt, row in close.iterrows():
        d = idx_dt.date() if hasattr(idx_dt, "date") else idx_dt
        for ticker in ticker_list:
            val = row.get(ticker)
            if val is not None and val == val:   # NaN check
                prices[ticker][d] = float(val)

    return prices


def get_price_on_or_before(prices_for_ticker: dict, target: date):
    for delta in range(6):
        d = target - timedelta(days=delta)
        if d in prices_for_ticker:
            return prices_for_ticker[d]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def make_growth_chart(fridays, growth_portfolio, growth_bench):
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    date_labels = list(fridays)
    series = [
        ("Portfolio",    growth_portfolio,             "#1d4ed8", 2.8, "solid"),
        ("S&P 500",      growth_bench["S&P 500"],      "#b91c1c", 1.8, "dashed"),
        ("Russell 3000", growth_bench["Russell 3000"], "#15803d", 1.8, "dashed"),
    ]

    final_vals = {}
    for label, vals, color, lw, ls in series:
        valid_dates = [d for d, v in zip(date_labels, vals) if v is not None]
        valid_vals  = [v for v in vals if v is not None]
        ax.plot(valid_dates, valid_vals, linewidth=lw, color=color,
                label=label, linestyle=ls, zorder=3)
        if valid_vals:
            final_vals[label] = (valid_vals[-1], valid_dates[-1], color)

    ax.axhline(y=1.0, color="#9ca3af", linewidth=1, linestyle=":", zorder=1,
               label="Baseline ($1.00)")

    # Staggered end-of-line labels
    sorted_finals = sorted(
        [(lbl, val, dt, col) for lbl, (val, dt, col) in final_vals.items()],
        key=lambda x: x[1],
    )
    MIN_GAP = 0.012
    adjusted_y = []
    for i, (_, val, _, _) in enumerate(sorted_finals):
        if i == 0:
            adjusted_y.append(val)
        else:
            adjusted_y.append(max(val, adjusted_y[-1] + MIN_GAP))

    for (lbl, val, dt, col), adj_y in zip(sorted_finals, adjusted_y):
        ax.annotate(
            f"{lbl}  {val:.3f}x",
            xy=(dt, val),
            xytext=(10, (adj_y - val) * 120),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold" if lbl == "Portfolio" else "normal",
            color=col,
            va="center",
            annotation_clip=False,
        )

    start_str = fridays[0].strftime("%B %d, %Y")
    end_str   = fridays[-1].strftime("%B %d, %Y")
    ax.set_title(
        "Portfolio Performance vs. Benchmarks" + "\n"
        + f"Growth of $1  \u00b7  {start_str} \u2013 {end_str}",
        fontsize=14, fontweight="bold", pad=14, loc="left", color=DARK,
    )
    ax.set_ylabel("Growth of $1.00", fontsize=11, color=GRAY)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:.2f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.FR, interval=2))
    plt.xticks(rotation=40, ha="right", color=GRAY)
    plt.yticks(color=GRAY)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color=LIGHT)
    ax.grid(axis="x", linestyle=":",  alpha=0.25, color=LIGHT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)
    ax.legend(fontsize=10, framealpha=0.85, edgecolor="#e5e7eb",
              loc="upper left", handlelength=2.2, labelspacing=0.5)
    plt.subplots_adjust(right=0.82)
    plt.tight_layout()
    return fig


def make_bar_fig(title, subtitle, labels, values, colors, ylabel, figsize=(8, 5.5)):
    fig2, ax2 = plt.subplots(figsize=figsize)
    fig2.patch.set_facecolor(BG)
    ax2.set_facecolor(BG)

    bar_w = 0.38 if len(labels) == 1 else 0.5
    x     = range(len(labels))
    bars  = ax2.bar(x, [v * 100 for v in values], color=colors,
                    width=bar_w, edgecolor=BG, linewidth=0, zorder=3)

    pct_vals = [v * 100 for v in values]
    y_min = min(pct_vals)
    y_max = max(pct_vals)
    pad   = max(abs(y_max), abs(y_min)) * 0.22 + 0.4
    ax2.set_ylim(
        y_min - pad if y_min < 0 else -pad * 0.3,
        y_max + pad if y_max > 0 else  pad * 0.3,
    )

    for bar, val in zip(bars, values):
        pct    = val * 100
        is_pos = pct >= 0
        y_pos  = pct + (pad * 0.18 if is_pos else -pad * 0.18)
        ax2.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{pct:+.2f}%",
            ha="center", va="bottom" if is_pos else "top",
            fontsize=13, fontweight="bold", color=bar.get_facecolor(),
        )

    ax2.axhline(0, color=LIGHT, linewidth=1.2, zorder=2)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=11, color=GRAY, fontweight="semibold")
    ax2.tick_params(axis="x", bottom=False, pad=6)
    ax2.set_ylabel(ylabel, fontsize=10.5, color=GRAY, labelpad=8)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:+.1f}%"))
    ax2.tick_params(axis="y", colors=GRAY, labelsize=9.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(LIGHT)
    ax2.spines["bottom"].set_color(LIGHT)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, color=LIGHT, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title(
        title + "\n" + subtitle,
        fontsize=13, fontweight="bold", pad=14, loc="left",
        color=DARK, linespacing=1.6,
    )
    plt.tight_layout(pad=1.8)
    return fig2


def _bar_color(val, base):
    return base if val >= 0 else "#dc2626"


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 Portfolio Analyzer")
st.markdown("Upload your two Fidelity CSV exports below to analyze portfolio performance.")

col_up1, col_up2 = st.columns(2)
with col_up1:
    positions_file = st.file_uploader("Portfolio Positions CSV", type="csv", key="pos")
with col_up2:
    orders_file = st.file_uploader("Order History CSV", type="csv", key="ord")

if not positions_file or not orders_file:
    st.info("Upload both files above to get started.")
    st.stop()

# ── Decode and detect ─────────────────────────────────────────────────────────
contents = {}
for f in [positions_file, orders_file]:
    content = f.read().decode("utf-8-sig")
    ftype   = detect_file_type(content)
    contents[ftype] = (f.name, content)

if "positions" not in contents or "orders" not in contents:
    st.error("Could not identify both files. Make sure you upload one positions file and one order history file.")
    st.stop()

pos_filename,  positions_content = contents["positions"]
ord_filename,  orders_content    = contents["orders"]

# ── Extract end date from positions filename ───────────────────────────────────
# Expected format: Portfolio_Positions_Mar-04-2026.csv
_date_match = re.search(r"(\w{3}-\d{2}-\d{4})", pos_filename)
if _date_match:
    try:
        end_date = datetime.strptime(_date_match.group(1), "%b-%d-%Y").date()
    except ValueError:
        end_date = date.today()
else:
    end_date = date.today()

st.success(f"Positions: **{pos_filename}**   |   Orders: **{ord_filename}**   |   End date: **{end_date.strftime('%B %d, %Y')}**")

# ── Parse ─────────────────────────────────────────────────────────────────────
with st.spinner("Parsing files..."):
    current_positions     = read_positions(positions_content)
    earliest_date, orders = read_orders(orders_content)
    starting_portfolio    = build_starting_portfolio(current_positions, orders)

# ── Fridays ───────────────────────────────────────────────────────────────────
first_friday = prev_friday(earliest_date)
fridays = []
f = first_friday
while f <= end_date:
    fridays.append(f)
    f += timedelta(weeks=1)
if prev_friday(end_date) not in fridays:
    fridays.append(prev_friday(end_date))
fridays = sorted(set(fridays))

# ── Fetch prices ──────────────────────────────────────────────────────────────
all_equity_tickers = set()
for sym in current_positions:
    if sym not in ("CASH", "PENDING"):
        all_equity_tickers.add(sym)
for o in orders:
    if o["ticker"] != "SPAXX":
        all_equity_tickers.add(o["ticker"])

BENCHMARKS = {"S&P 500": "^GSPC", "Russell 3000": "^RUA"}
all_yf_tickers = tuple(sorted(all_equity_tickers | set(BENCHMARKS.values())))

with st.spinner("Fetching price data from Yahoo Finance..."):
    price_data = fetch_prices(
        all_yf_tickers,
        start_str=(first_friday - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_str=end_date.strftime("%Y-%m-%d"),
    )

# ── Weekly portfolio values ───────────────────────────────────────────────────
weekly_values  = []
missing_by_week = []

for friday in fridays:
    port = apply_orders_up_to(starting_portfolio, orders, friday)
    total = port.get("CASH", 0.0) + current_positions.get("PENDING", 0.0)
    missing = []
    for sym, shares in port.items():
        if sym == "CASH":
            continue
        price = get_price_on_or_before(price_data.get(sym, {}), friday)
        if price is None:
            missing.append(sym)
        else:
            total += shares * price
    weekly_values.append(total)
    missing_by_week.append(missing)

# ── Growth series ─────────────────────────────────────────────────────────────
base_port       = weekly_values[0]
growth_portfolio = [v / base_port for v in weekly_values]

growth_bench = {}
for name, sym in BENCHMARKS.items():
    prices = price_data.get(sym, {})
    series = [get_price_on_or_before(prices, f) for f in fridays]
    base   = next((p for p in series if p is not None), None)
    growth_bench[name] = [(p / base if p is not None else None) for p in series] if base else [None] * len(fridays)

# ── Return calculations ───────────────────────────────────────────────────────
abs_return    = growth_portfolio[-1] - 1.0
bench_returns = {}
for name in BENCHMARKS:
    vals  = growth_bench[name]
    first = next((v for v in vals if v is not None), None)
    last  = next((v for v in reversed(vals) if v is not None), None)
    bench_returns[name] = (last / first - 1.0) if (first and last) else None

excess_returns = {
    name: (abs_return - br) if br is not None else None
    for name, br in bench_returns.items()
}

date_range_str = (fridays[0].strftime("%b %d, %Y")
                  + " \u2013 "
                  + fridays[-1].strftime("%b %d, %Y"))

# ═════════════════════════════════════════════════════════════════════════════
# Section 1 — Time-frame & key metrics
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Overview")

delta_days = (end_date - earliest_date).days
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Start Date",   earliest_date.strftime("%b %d, %Y"))
m2.metric("End Date",     end_date.strftime("%b %d, %Y"))
m3.metric("Portfolio Return", f"{abs_return*100:+.2f}%")

er_sp = excess_returns.get("S&P 500")
er_ru = excess_returns.get("Russell 3000")
m4.metric("vs. S&P 500",      f"{er_sp*100:+.2f}%" if er_sp is not None else "N/A")
m5.metric("vs. Russell 3000", f"{er_ru*100:+.2f}%" if er_ru is not None else "N/A")

# ═════════════════════════════════════════════════════════════════════════════
# Section 2 — Weekly portfolio values table
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
with st.expander("Weekly Portfolio Values", expanded=False):
    import pandas as pd
    rows = []
    for friday, val, missing in zip(fridays, weekly_values, missing_by_week):
        rows.append({
            "Date":            friday.strftime("%b %d, %Y"),
            "Portfolio Value": f"${val:,.2f}",
            "Missing Prices":  ", ".join(missing) if missing else "—",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# Section 3 — Current positions
# ═════════════════════════════════════════════════════════════════════════════

with st.expander("Current Positions", expanded=False):
    import pandas as pd
    pos_rows = []
    for sym, qty in current_positions.items():
        if sym in ("CASH", "PENDING"):
            pos_rows.append({"Symbol": sym, "Quantity / Value": f"${qty:,.2f}"})
        else:
            pos_rows.append({"Symbol": sym, "Quantity / Value": f"{qty:,.4f} shares"})
    st.dataframe(pd.DataFrame(pos_rows), width='stretch', hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# Section 4 — Order history
# ═════════════════════════════════════════════════════════════════════════════

with st.expander("Order History", expanded=False):
    import pandas as pd
    ord_rows = []
    for o in reversed(orders):
        ord_rows.append({
            "Date":     o["date"].strftime("%m/%d/%Y"),
            "Type":     o["type"],
            "Ticker":   o["ticker"],
            "Quantity": f"{abs(o['quantity']):,.4f}",
            "Amount":   f"${abs(o['amount']):,.2f}" if o["amount"] != 0 else "—",
        })
    st.dataframe(pd.DataFrame(ord_rows), width='stretch', hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# Section 5 — Growth of $1 chart
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Growth of $1")
fig_growth = make_growth_chart(fridays, growth_portfolio, growth_bench)
st.pyplot(fig_growth, width='stretch')
plt.close(fig_growth)

# ═════════════════════════════════════════════════════════════════════════════
# Section 6 — Return summary charts
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Return Summary")

# Absolute return — full width
all_labels = ["Portfolio", "S&P 500", "Russell 3000"]
all_rets   = [abs_return] + [bench_returns.get(n, 0) or 0 for n in ["S&P 500", "Russell 3000"]]
all_colors = [
    _bar_color(abs_return,                              "#1d4ed8"),
    _bar_color(bench_returns.get("S&P 500")    or 0,   "#b91c1c"),
    _bar_color(bench_returns.get("Russell 3000") or 0,  "#15803d"),
]
fig_abs = make_bar_fig(
    title    = "Absolute Return",
    subtitle = date_range_str,
    labels   = all_labels,
    values   = all_rets,
    colors   = all_colors,
    ylabel   = "Total Return (%)",
    figsize  = (9, 5.5),
)
st.pyplot(fig_abs, width='stretch')
plt.close(fig_abs)

# Excess return charts side by side
col1, col2 = st.columns(2)
if er_sp is not None:
    with col1:
        fig_sp = make_bar_fig(
            title    = "Excess Return vs. S&P 500",
            subtitle = date_range_str,
            labels   = ["Portfolio \u2212 S&P 500"],
            values   = [er_sp],
            colors   = [_bar_color(er_sp, "#1d4ed8")],
            ylabel   = "Excess Return (%)",
            figsize  = (6, 5.5),
        )
        st.pyplot(fig_sp, width='stretch')
        plt.close(fig_sp)

if er_ru is not None:
    with col2:
        fig_ru = make_bar_fig(
            title    = "Excess Return vs. Russell 3000",
            subtitle = date_range_str,
            labels   = ["Portfolio \u2212 Russell 3000"],
            values   = [er_ru],
            colors   = [_bar_color(er_ru, "#1d4ed8")],
            ylabel   = "Excess Return (%)",
            figsize  = (6, 5.5),
        )
        st.pyplot(fig_ru, width='stretch')
        plt.close(fig_ru)


# ═════════════════════════════════════════════════════════════════════════════
# Section 7 — Factor Betas (Fama-French 4-Factor)
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Factor Betas")
st.markdown(
    "4-factor regression (MKT, SMB, HML, WML) for each current equity position "
    "and the portfolio as a whole. Uses 3 years of monthly returns."
)

LOOKBACK_YEARS = 3
FACTORS        = ["MKT-RF", "SMB", "HML", "WML"]
FACTOR_LABELS  = {"MKT-RF": "Market (β)", "SMB": "SMB (Size)", "HML": "HML (Value)", "WML": "WML (Momentum)"}
FACTOR_COLORS  = {"MKT-RF": "#1d4ed8", "SMB": "#b91c1c", "HML": "#15803d", "WML": "#7c3aed"}

@st.cache_data(show_spinner=False)
def load_ff_factors(lookback_years: int) -> pd.DataFrame:
    """
    Download Fama-French 3-factor + momentum monthly data directly from
    Kenneth French's data library (no pandas-datareader needed).
    Returns a DataFrame indexed by month-end date with columns:
        MKT-RF, SMB, HML, WML, RF  (all as decimals, e.g. 0.012)
    """
    def _fetch_ff_zip(url: str) -> pd.DataFrame:
        """Download a FF zip, extract the named CSV, parse the monthly table."""
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(_io.BytesIO(resp.content)) as zf:
            # Don't hardcode the filename — just grab the first .csv/.CSV entry
            csv_name = next(
                n for n in zf.namelist()
                if n.lower().endswith(".csv")
            )
            with zf.open(csv_name) as f:
                raw = f.read().decode("utf-8", errors="replace")

        # FF files have a header block of text, then a CSV block, then more text.
        # The monthly data block starts after a blank line following the header
        # and ends at the next blank line.
        lines = raw.splitlines()
        blocks = []
        current = []
        for line in lines:
            if line.strip() == "":
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(line.strip())
        if current:
            blocks.append(current)

        # Find the first block whose first line looks like a CSV header
        # (contains letters) and whose data rows are 6-digit YYYYMM integers
        data_block = None
        for block in blocks:
            if len(block) < 2:
                continue
            # Check if second row starts with a 6-digit date
            parts = block[1].split(",")
            if parts and re.match(r"^\d{6}$", parts[0].strip()):
                data_block = block
                break

        if data_block is None:
            raise ValueError(f"Could not parse factor data from {url}")

        header = [h.strip() for h in data_block[0].split(",")]
        rows = []
        for line in data_block[1:]:
            parts = [p.strip() for p in line.split(",")]
            if not parts or not re.match(r"^\d{6}$", parts[0]):
                break
            rows.append(parts)

        df = pd.DataFrame(rows, columns=header)
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
        # Convert to month-end
        df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
        df = df.set_index("Date")
        return df.apply(pd.to_numeric, errors="coerce") / 100.0

    # Fama-French 3-factor monthly
    ff3 = _fetch_ff_zip(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
    )

    # Normalise FF3 column names — French sometimes uses "Mkt-RF" or "MKT-RF"
    # Standardise to uppercase and strip whitespace
    ff3.columns = [c.strip().upper().replace("MKT-RF", "MKT-RF") for c in ff3.columns]
    # Map any variant spellings to canonical names
    ff3_rename = {}
    for col in ff3.columns:
        cu = col.upper().replace(" ", "")
        if "MKT" in cu and "RF" in cu:
            ff3_rename[col] = "MKT-RF"
        elif cu == "SMB":
            ff3_rename[col] = "SMB"
        elif cu == "HML":
            ff3_rename[col] = "HML"
        elif cu == "RF":
            ff3_rename[col] = "RF"
    ff3 = ff3.rename(columns=ff3_rename)

    # Momentum factor monthly
    mom = _fetch_ff_zip(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
    )
    # Momentum column may be "Mom", "WML", "PR1YR", etc. — just take the first data column
    mom.columns = ["WML"]

    factors = ff3.join(mom, how="inner")

    # Verify required columns are present before returning
    required = {"MKT-RF", "SMB", "HML", "RF", "WML"}
    missing_cols = required - set(factors.columns)
    if missing_cols:
        raise ValueError(
            f"FF factor data missing expected columns: {missing_cols}. "
            f"Got: {list(factors.columns)}"
        )

    # Trim to lookback window
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=lookback_years)
    factors = factors[factors.index >= cutoff]

    return factors


@st.cache_data(show_spinner=False)
def get_monthly_returns(tickers: tuple, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Download monthly closing prices via yfinance and compute simple returns.
    Reindexes to month-end dates to align with Fama-French factor data.
    """
    raw = yf.download(
        list(tickers),
        start=start_str,
        end=end_str,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return pd.DataFrame()

    close = raw["Close"] if len(tickers) > 1 else raw[["Close"]]
    if len(tickers) == 1:
        close.columns = list(tickers)

    # yfinance monthly bars are timestamped at month-start; snap to month-end
    # so they align with the FF factor index (which is month-end)
    close.index = close.index.to_period("M").to_timestamp("M")

    returns = close.pct_change().dropna(how="all")
    return returns


def run_ff_regression(excess_ret: pd.Series, factors: pd.DataFrame):
    """
    OLS regression of excess stock returns on the 4 Fama-French factors.
    Both series must share a month-end DatetimeIndex.
    Returns a result dict, or None if fewer than 12 overlapping observations.
    """
    # Snap both indexes to period then back to month-end to guarantee alignment
    excess_ms = excess_ret.copy()
    excess_ms.index = excess_ms.index.to_period("M").to_timestamp("M")

    factors_ms = factors.copy()
    factors_ms.index = factors_ms.index.to_period("M").to_timestamp("M")

    combined = factors_ms.join(excess_ms.rename("r"), how="inner").dropna()
    if len(combined) < 12:
        return None

    X = sm.add_constant(combined[FACTORS])
    y = combined["r"]
    model = sm.OLS(y, X).fit()

    result = {"alpha": model.params["const"], "r_squared": model.rsquared, "n_obs": len(combined)}
    for f in FACTORS:
        result[f]           = model.params[f]
        result[f + "_tstat"] = model.tvalues[f]
        result[f + "_sig"]   = abs(model.tvalues[f]) > 1.96
    return result


# ── Load factor data ──────────────────────────────────────────────────────────
with st.spinner("Loading Fama-French factor data..."):
    try:
        ff_factors = load_ff_factors(LOOKBACK_YEARS)
        ff_ok = True
    except Exception as e:
        st.warning(f"Could not load Fama-French data: {e}")
        ff_ok = False

if ff_ok:
    # Current equity positions only (exclude CASH and PENDING)
    equity_positions = {
        sym: qty for sym, qty in current_positions.items()
        if sym not in ("CASH", "PENDING") and qty > 0
    }

    # Fetch 3 years of monthly data; add a 1-month buffer on each end
    factor_start = (ff_factors.index[0]  - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    factor_end   = (ff_factors.index[-1] + pd.DateOffset(months=2)).strftime("%Y-%m-%d")

    with st.spinner("Fetching monthly price history for factor regressions..."):
        monthly_rets = get_monthly_returns(
            tickers   = tuple(sorted(equity_positions.keys())),
            start_str = factor_start,
            end_str   = factor_end,
        )

    # Diagnostics expander — helpful if something still goes wrong
    with st.expander("Factor regression diagnostics", expanded=False):
        st.write(f"**FF factors:** {len(ff_factors)} months, "
                 f"{ff_factors.index[0].date()} → {ff_factors.index[-1].date()}")
        st.write(f"**FF columns:** {list(ff_factors.columns)}")
        if not monthly_rets.empty:
            st.write(f"**Monthly returns:** {len(monthly_rets)} rows, "
                     f"{monthly_rets.index[0].date()} → {monthly_rets.index[-1].date()}")
            st.write(f"**Tickers fetched:** {list(monthly_rets.columns)}")
            # Show sample of index to confirm month-end alignment
            st.write(f"**Return index sample:** {[str(d.date()) for d in monthly_rets.index[:3]]}")
            st.write(f"**FF index sample:** {[str(d.date()) for d in ff_factors.index[:3]]}")
        else:
            st.write("**Monthly returns:** empty — yfinance returned no data")

    # Run regression for each ticker
    betas_rows = []
    per_ticker = {}

    for sym in sorted(equity_positions.keys()):
        if monthly_rets.empty or sym not in monthly_rets.columns:
            continue
        stock_ret = monthly_rets[sym].dropna()
        rf        = ff_factors["RF"].copy()
        rf.index  = rf.index.to_period("M").to_timestamp("M")
        rf        = rf.reindex(stock_ret.index)
        excess    = (stock_ret - rf).dropna()
        result    = run_ff_regression(excess, ff_factors)
        if result is None:
            continue
        per_ticker[sym] = result
        betas_rows.append({
            "Ticker":   sym,
            "Market β": round(result["MKT-RF"], 3),
            "SMB":      round(result["SMB"],    3),
            "HML":      round(result["HML"],    3),
            "WML":      round(result["WML"],    3),
            "α (mo.)":  f"{result['alpha']*100:+.2f}%",
            "R²":       f"{result['r_squared']:.2f}",
            "Obs.":     result["n_obs"],
        })

    # ── Portfolio-level weighted betas ───────────────────────────────────────
    # Weight by current market value (shares × latest price)
    latest_prices = {
        sym: get_price_on_or_before(price_data.get(sym, {}), end_date)
        for sym in equity_positions
    }
    market_vals = {
        sym: equity_positions[sym] * latest_prices[sym]
        for sym in equity_positions
        if latest_prices.get(sym) and sym in per_ticker
    }
    total_equity_val = sum(market_vals.values())

    port_betas = {f: 0.0 for f in FACTORS}
    if total_equity_val > 0:
        for sym, mv in market_vals.items():
            w = mv / total_equity_val
            for f in FACTORS:
                port_betas[f] += w * per_ticker[sym][f]

    # ── Table ─────────────────────────────────────────────────────────────────
    if betas_rows:
        df_betas = pd.DataFrame(betas_rows)

        # Colour-code numeric beta columns via pandas Styler
        def _colour_beta(val):
            try:
                v = float(val)
                if v > 0.15:   return "color: #15803d; font-weight:600"
                if v < -0.15:  return "color: #b91c1c; font-weight:600"
            except (TypeError, ValueError):
                pass
            return "color: #374151"

        styled = (
            df_betas.style
            .applymap(_colour_beta, subset=["Market β", "SMB", "HML", "WML"])
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
        )
        st.dataframe(styled, width='stretch', hide_index=True)

        # ── Portfolio summary metrics ─────────────────────────────────────────
        st.markdown("**Portfolio-level factor exposures** (market-value weighted)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        factor_meta = [
            ("MKT-RF", "Market β",     mc1),
            ("SMB",    "SMB",          mc2),
            ("HML",    "HML",          mc3),
            ("WML",    "WML (Mom.)",   mc4),
        ]
        for fkey, flabel, col in factor_meta:
            val = port_betas[fkey]
            col.metric(flabel, f"{val:+.3f}")

        # ── Visualisation ─────────────────────────────────────────────────────
        st.markdown("---")

        # 1. Portfolio beta bar chart
        fig_pb, ax_pb = plt.subplots(figsize=(8, 4))
        fig_pb.patch.set_facecolor(BG)
        ax_pb.set_facecolor(BG)

        pb_labels = [FACTOR_LABELS[f] for f in FACTORS]
        pb_values = [port_betas[f] for f in FACTORS]
        pb_colors = [FACTOR_COLORS[f] for f in FACTORS]
        pb_bars   = ax_pb.bar(pb_labels, pb_values, color=pb_colors,
                              width=0.45, edgecolor=BG, zorder=3)

        y_ext = max(abs(v) for v in pb_values) if pb_values else 1
        pad   = y_ext * 0.25 + 0.05
        ax_pb.set_ylim(-y_ext - pad, y_ext + pad)

        for bar, val in zip(pb_bars, pb_values):
            is_pos = val >= 0
            ax_pb.text(
                bar.get_x() + bar.get_width() / 2,
                val + (pad * 0.35 if is_pos else -pad * 0.35),
                f"{val:+.3f}",
                ha="center", va="bottom" if is_pos else "top",
                fontsize=11, fontweight="bold", color=bar.get_facecolor(),
            )

        ax_pb.axhline(0, color=LIGHT, linewidth=1.2, zorder=2)
        ax_pb.set_title(
            "Portfolio Factor Exposures" + "\n" + "Market-value weighted betas",
            fontsize=13, fontweight="bold", pad=12, loc="left", color=DARK,
        )
        ax_pb.set_ylabel("Beta", fontsize=10.5, color=GRAY, labelpad=8)
        ax_pb.tick_params(axis="x", bottom=False, labelsize=10.5, colors=GRAY)
        ax_pb.tick_params(axis="y", colors=GRAY, labelsize=9)
        ax_pb.spines["top"].set_visible(False)
        ax_pb.spines["right"].set_visible(False)
        ax_pb.spines["left"].set_color(LIGHT)
        ax_pb.spines["bottom"].set_color(LIGHT)
        ax_pb.grid(axis="y", linestyle="--", alpha=0.4, color=LIGHT, zorder=0)
        ax_pb.set_axisbelow(True)
        plt.tight_layout(pad=1.8)
        st.pyplot(fig_pb, width='stretch')
        plt.close(fig_pb)

        # 2. Per-ticker heatmap
        st.markdown("**Per-position factor betas**")
        hm_data = df_betas.set_index("Ticker")[["Market β", "SMB", "HML", "WML"]].astype(float)

        fig_hm, ax_hm = plt.subplots(
            figsize=(max(7, len(hm_data.columns) * 1.6),
                     max(4, len(hm_data) * 0.38 + 1.2))
        )
        fig_hm.patch.set_facecolor(BG)
        ax_hm.set_facecolor(BG)

        mat    = hm_data.values
        vmax   = np.percentile(np.abs(mat[~np.isnan(mat)]), 95) if mat.size else 1
        im     = ax_hm.imshow(mat, cmap="RdYlGn", aspect="auto",
                               vmin=-vmax, vmax=vmax)

        ax_hm.set_xticks(range(len(hm_data.columns)))
        col_label_map = {"Market β": "Market (β)", "SMB": "SMB (Size)",
                             "HML": "HML (Value)", "WML": "WML (Momentum)"}
        ax_hm.set_xticklabels(
            [col_label_map.get(c, c) for c in hm_data.columns],
            fontsize=10.5, color=DARK, fontweight="semibold"
        )
        ax_hm.set_yticks(range(len(hm_data.index)))
        ax_hm.set_yticklabels(hm_data.index, fontsize=9, color=GRAY)
        ax_hm.tick_params(length=0)

        for i in range(len(hm_data.index)):
            for j in range(len(hm_data.columns)):
                val = mat[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > vmax * 0.6 else DARK
                    ax_hm.text(j, i, f"{val:+.2f}", ha="center", va="center",
                               fontsize=8.5, color=text_color, fontweight="600")

        cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=8, colors=GRAY)
        cbar.outline.set_edgecolor(LIGHT)

        ax_hm.set_title(
            "Factor Beta Heatmap  ·  Per Position" + "\n"
            + f"3-year monthly regression  ·  Green = positive exposure, Red = negative",
            fontsize=12, fontweight="bold", pad=12, loc="left", color=DARK,
        )
        ax_hm.spines[:].set_visible(False)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig_hm, width='stretch')
        plt.close(fig_hm)

    else:
        st.warning("No factor regressions could be completed — insufficient price history for current positions.")
