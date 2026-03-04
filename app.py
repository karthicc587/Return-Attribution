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

st.success(f"Positions: **{pos_filename}**   |   Orders: **{ord_filename}**")

# ── Parse ─────────────────────────────────────────────────────────────────────
with st.spinner("Parsing files..."):
    current_positions     = read_positions(positions_content)
    earliest_date, orders = read_orders(orders_content)
    today                 = date.today()
    starting_portfolio    = build_starting_portfolio(current_positions, orders)

# ── Fridays ───────────────────────────────────────────────────────────────────
first_friday = prev_friday(earliest_date)
fridays = []
f = first_friday
while f <= today:
    fridays.append(f)
    f += timedelta(weeks=1)
if prev_friday(today) not in fridays:
    fridays.append(prev_friday(today))
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
        end_str=today.strftime("%Y-%m-%d"),
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

delta_days = (today - earliest_date).days
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Start Date",   earliest_date.strftime("%b %d, %Y"))
m2.metric("End Date",     today.strftime("%b %d, %Y"))
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

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
    st.dataframe(pd.DataFrame(ord_rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# Section 5 — Growth of $1 chart
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Growth of $1")
fig_growth = make_growth_chart(fridays, growth_portfolio, growth_bench)
st.pyplot(fig_growth, use_container_width=True)
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
st.pyplot(fig_abs, use_container_width=True)
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
        st.pyplot(fig_sp, use_container_width=True)
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
        st.pyplot(fig_ru, use_container_width=True)
        plt.close(fig_ru)
