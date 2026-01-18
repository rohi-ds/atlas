import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yfinance as yf
from fpdf import FPDF

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="PortEngine by Rohi Aluede", layout="wide", initial_sidebar_state="collapsed")

# Initialize Session States
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'stocks_df' not in st.session_state:
    st.session_state.stocks_df = pd.DataFrame(columns=['Ticker', 'Sector', 'Expected Return (%)', 'Volatility (%)'])
if 'lookback' not in st.session_state:
    st.session_state.lookback = "5y"
if 'max_vol_limit' not in st.session_state:
    st.session_state.max_vol_limit = 35.0
if 'opt_results' not in st.session_state:
    st.session_state.opt_results = None

# --- UI & STYLING ENGINE ---
def apply_custom_styles():
    # Force visibility for Light Mode: Ensure text is NEVER white on white
    if st.session_state.theme == 'light':
        bg_color, card_bg, text_color = "#F8FAFC", "#FFFFFF", "#1E293B"
        border_color, secondary_text = "#E2E8F0", "#475569"
    else:
        bg_color, card_bg, text_color = "#0F172A", "#1E293B", "#F8FAFC"
        border_color, secondary_text = "#334155", "#94A3B8"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {{
        font-family: 'Inter', sans-serif;
        color: {text_color} !important;
    }}
    
    .stApp {{ background-color: {bg_color}; }}
    
    /* Button Padding & Visibility */
    div.stButton > button {{
        padding: 14px 30px !important;
        border-radius: 8px !important;
        border: 1px solid {border_color};
        background-color: {card_bg};
        color: {text_color} !important;
        font-weight: 600;
        margin: 8px 0;
    }}
    
    /* Ensure Dataframe text is visible */
    [data-testid="stDataFrame"] {{ background-color: {card_bg}; border: 1px solid {border_color}; }}
    
    /* Metric Card */
    .metric-card {{
        background-color: {card_bg};
        padding: 24px;
        border-radius: 12px;
        border: 1px solid {border_color};
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }}
    
    /* Layout Spacing */
    .section-spacer {{ margin-top: 60px; padding-top: 30px; border-top: 1px solid {border_color}; }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# --- DATA ENGINE ---
def fetch_asset_info(symbol, period):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty: return None
        
        # Calculate daily returns for metrics
        rets = hist['Close'].pct_change().dropna()
        if rets.empty: return None
        
        ann_ret = rets.mean() * 252 * 100
        ann_vol = rets.std() * np.sqrt(252) * 100
        
        # Sector detection
        try:
            sector = ticker.info.get('sector', 'Unknown')
        except:
            sector = 'Unknown'
            
        return {
            'Ticker': symbol.upper(),
            'Sector': sector,
            'Expected Return (%)': round(ann_ret, 2),
            'Volatility (%)': round(ann_vol, 2)
        }
    except:
        return None

def calculate_portfolio_performance(weights, returns, vols, corr_matrix, rf=4.0):
    w = np.array(weights) / 100
    r = np.array(returns) / 100
    v = np.array(vols) / 100
    cov = np.outer(v, v) * corr_matrix
    
    p_ret = np.sum(r * w)
    p_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
    sharpe = (p_ret - (rf/100)) / p_vol if p_vol > 0 else 0
    
    return {'Return': p_ret * 100, 'Volatility': p_vol * 100, 'Sharpe': sharpe}

# --- PDF ENGINE (FIXED) ---
class PortEngineReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'PortEngine Optimization Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Architect: Rohi Aluede', 0, 1, 'C')
        self.ln(10)

def generate_pdf_report(df, metrics, lookback):
    pdf = PortEngineReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(0, 10, f"Configuration: {lookback} Lookback Period", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Portfolio Metrics", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Expected Annual Return: {metrics['Return']:.2f}%", ln=True)
    pdf.cell(0, 10, f"Annualized Volatility: {metrics['Volatility']:.2f}%", ln=True)
    pdf.cell(0, 10, f"Sharpe Ratio: {metrics['Sharpe']:.2f}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Optimized Weights", ln=True)
    pdf.set_font("Arial", size=10)
    for _, row in df.iterrows():
        pdf.cell(0, 8, f"{row['Ticker']} ({row['Sector']}): {row['Weight']:.2f}%", ln=True)
    
    # CRITICAL FIX: Convert bytearray to bytes for Streamlit
    return bytes(pdf.output())

# --- HEADER ---
h_col1, h_col2 = st.columns([5, 1])
with h_col1:
    st.title("PortEngine")
    st.write("**System by Rohi Aluede** | Modern Portfolio Theory Optimization Engine")
with h_col2:
    if st.button("🌓 Toggle Theme", use_container_width=True):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

# --- TABS ---
tabs = st.tabs(["📂 Portfolio Setup", "🎯 Optimization", "📊 Risk Analysis", "🎲 Monte Carlo", "⚙️ Engine Settings"])

with tabs[0]:
    st.header("Asset Universe Management")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.lookback = st.selectbox("Historical Lookback (Sets time frame for returns/vol)", ["1y", "2y", "5y", "10y"], index=2)
    with c2:
        st.session_state.max_vol_limit = st.number_input("Max Volatility Warning (%)", value=35.0)

    st.markdown("---")
    
with st.form("add_ticker_form"):
    in_col1, in_col2 = st.columns([3, 1])
    ticker_input = in_col1.text_input("Add Ticker (e.g., TSLA, BTC-USD, VOO)", placeholder="Enter symbol and press Enter or click button...")
    submit = in_col2.form_submit_button("Search & Add", use_container_width=True)

if submit and ticker_input:
    with st.spinner(f"Pulling {ticker_input} historical data..."):
        info = fetch_asset_info(ticker_input, st.session_state.lookback)
        if info:
            # Remove if exists to update with current lookback
            st.session_state.stocks_df = st.session_state.stocks_df[st.session_state.stocks_df['Ticker'] != info['Ticker']]
            st.session_state.stocks_df = pd.concat([st.session_state.stocks_df, pd.DataFrame([info])], ignore_index=True)
            st.success(f"Successfully integrated {info['Ticker']}")
        else:
            st.error(f"Ticker '{ticker_input}' not found. Please remove and try a valid Yahoo Finance symbol.")

if not st.session_state.stocks_df.empty:
    st.subheader("Current Candidates")
    
    # FIX 1: Proper styling for high volatility rows
    def style_dataframe(df):
        def highlight_vol(row):
            if row['Volatility (%)'] > st.session_state.max_vol_limit:
                return ['background-color: #991b1b; color: white'] * len(row)
            else:
                return [''] * len(row)
        return df.style.apply(highlight_vol, axis=1)

    # Display styled dataframe (not editable to preserve styling)
    st.dataframe(
        style_dataframe(st.session_state.stocks_df),
        use_container_width=True,
        hide_index=True
    )
    
    # Add separate option to remove rows
    if len(st.session_state.stocks_df) > 0:
        ticker_to_remove = st.selectbox("Remove a ticker:", ["Select to remove..."] + st.session_state.stocks_df['Ticker'].tolist())
        if ticker_to_remove != "Select to remove..." and st.button("Remove Selected Ticker"):
            st.session_state.stocks_df = st.session_state.stocks_df[st.session_state.stocks_df['Ticker'] != ticker_to_remove]
            st.rerun()
    
    if st.button("Clear Engine Data"):
        st.session_state.stocks_df = pd.DataFrame(columns=['Ticker', 'Sector', 'Expected Return (%)', 'Volatility (%)'])
        st.session_state.opt_results = None
        st.rerun()
        
with tabs[1]:
    st.header("Portfolio Optimization")
    if len(st.session_state.stocks_df) < 2:
        st.info("Add at least 2 assets to run the optimization engine.")
    else:
        op1, op2 = st.columns(2)
        max_const = op1.slider("Max Concentration per Asset (%)", 10, 100, 40)
        rf_rate = op2.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.5)
        
        if st.button("Run PortEngine Optimizer", type="primary", use_container_width=True):
            tickers = st.session_state.stocks_df['Ticker'].tolist()
            rets = st.session_state.stocks_df['Expected Return (%)'].values
            vols = st.session_state.stocks_df['Volatility (%)'].values
            
            # Optimization Math: Ensure correlation matrix is stable
            with st.spinner("Analyzing cross-asset correlations..."):
                prices = yf.download(tickers, period=st.session_state.lookback)['Close']
                # Drop rows where any stock has missing data to ensure aligned correlation
                corr_matrix = prices.pct_change().dropna().corr().fillna(0.35).values
            
            n = len(tickers)
            cov = np.outer(vols/100, vols/100) * corr_matrix
            
            def objective(w):
                p_ret = np.sum((rets/100) * w)
                p_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
                if p_vol == 0: return 0
                return -(p_ret - rf_rate/100) / p_vol
            
            res = minimize(objective, [1/n]*n, method='SLSQP', 
                           bounds=tuple((0, max_const/100) for _ in range(n)), 
                           constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            if res.success:
                st.session_state.opt_results = {'weights': res.x * 100, 'corr': corr_matrix, 'rf': rf_rate}
            else:
                st.error("Optimization failed. Try reducing Max Concentration or checking data.")

        if st.session_state.opt_results:
            ores = st.session_state.opt_results
            p_metrics = calculate_portfolio_performance(ores['weights'], st.session_state.stocks_df['Expected Return (%)'], st.session_state.stocks_df['Volatility (%)'], ores['corr'], ores['rf'])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Exp. Annual Return", f"{p_metrics['Return']:.2f}%")
            m2.metric("Portfolio Volatility", f"{p_metrics['Volatility']:.2f}%")
            m3.metric("Sharpe Ratio", f"{p_metrics['Sharpe']:.2f}")
            
            st.plotly_chart(px.bar(x=st.session_state.stocks_df['Ticker'], y=ores['weights'], labels={'x':'Ticker', 'y':'Weight (%)'}, title="Optimized Asset Allocation"), use_container_width=True)
            
            # PDF Download
            report_df = pd.DataFrame({'Ticker': st.session_state.stocks_df['Ticker'], 'Sector': st.session_state.stocks_df['Sector'], 'Weight': ores['weights']})
            pdf_data = generate_pdf_report(report_df, p_metrics, st.session_state.lookback)
            st.download_button("📥 Download PDF Portfolio Report", data=pdf_data, file_name=f"PortEngine_Report_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")

with tabs[2]:
    st.header("Risk Decomposition")
    # FIX 2: Changed condition from 'opt_data' to 'opt_results'
    if st.session_state.opt_results is not None:
        res = st.session_state.opt_results
        
        col_risk1, col_risk2 = st.columns(2)
        with col_risk1:
            st.subheader("Allocation by Sector")
            temp_df = st.session_state.stocks_df.copy()
            temp_df['Weight'] = res['weights']
            sect_fig = px.pie(temp_df.groupby('Sector')['Weight'].sum().reset_index(), names='Sector', values='Weight', hole=0.4)
            st.plotly_chart(sect_fig, use_container_width=True)
            
        with col_risk2:
            st.subheader("Risk Contribution")
            # Proportional Risk
            risk_fig = px.bar(x=st.session_state.stocks_df['Ticker'], y=res['weights'], title="Proportional Exposure")
            st.plotly_chart(risk_fig, use_container_width=True)
    else:
        st.info("Run the optimizer first to view risk analysis.")

with tabs[3]:
    st.header("Monte Carlo Engine")
    st.info("""
    **Understanding the Simulation:**
    This tool projects **1,000 potential future paths** for your optimized portfolio over the next 252 trading days (1 year). 
    It uses your portfolio's specific return and volatility to simulate 'market shocks' based on Geometric Brownian Motion.
    
    **The Y-Axis:** Represents the **Portfolio Wealth Index**, starting at a base of 100. If the line ends at 115, it means that specific path resulted in a 15% profit.
    """)
    
    if st.session_state.opt_results:
        ores = st.session_state.opt_results
        p_metrics = calculate_portfolio_performance(ores['weights'], st.session_state.stocks_df['Expected Return (%)'], st.session_state.stocks_df['Volatility (%)'], ores['corr'], ores['rf'])
        
        if st.button("Execute Portfolio Simulation"):
            # Math for simulation
            mu = (p_metrics['Return'] / 100) / 252
            sigma = (p_metrics['Volatility'] / 100) / np.sqrt(252)
            n_sims, t_days = 1000, 252
            
            paths = np.zeros((t_days, n_sims))
            paths[0] = 100
            for t in range(1, t_days):
                paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal(0, 1, n_sims))
            
            fig_mc = go.Figure()
            for i in range(min(n_sims, 60)): # Plot subset for performance
                fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=1), opacity=0.25, showlegend=False))
            
            st.plotly_chart(fig_mc, use_container_width=True)
            
            p5, p50, p95 = np.percentile(paths[-1], 5), np.percentile(paths[-1], 50), np.percentile(paths[-1], 95)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Worst Case (5%)", f"{p5:.2f}")
            mc2.metric("Median Scenario", f"{p50:.2f}")
            mc3.metric("Best Case (95%)", f"{p95:.2f}")
    else:
        st.warning("Run the optimizer first to simulate the portfolio.")

with tabs[4]:
    st.header("Engine Settings & System Info")
    
    # Correlation Map
    st.subheader("Diversification Heatmap")
    if len(st.session_state.stocks_df) >= 2:
        with st.spinner("Calculating relationships..."):
            raw_prices = yf.download(st.session_state.stocks_df['Ticker'].tolist(), period=st.session_state.lookback)['Close']
            corr_data = raw_prices.pct_change().dropna().corr()
            
            fig_heat = px.imshow(
                corr_data, 
                text_auto=".2f", 
                color_continuous_scale='RdYlGn_r', # Red high correlation, Green low
                title=f"Asset Correlation Matrix ({st.session_state.lookback} Window)"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("**Red (1.0)**: Assets move together (Low diversification) | **Green (<0.4)**: Assets move independently (High diversification)")
    else:
        st.write("Add assets to generate heatmap.")

    # Spacing between segments to prevent overlap
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    
    st.subheader("About PortEngine")
    st.info(f"""
    **System Version:** 3.0 (Production)  
    **Architect:** Rohi Aluede  
    **Engine Logic:** Mean-Variance Optimization (Sharpe Maximization)  
    **Real-Time API:** Yahoo Finance Integrated
    """)